# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for MiniMax M3 CUTLASS sparse decode."""

import math
from types import SimpleNamespace

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.models.minimax_m3.common import sparse_attention as sparse_attention_module
from vllm.models.minimax_m3.common.ops.sparse_attn import (
    minimax_m3_sparse_attn_decode,
)
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseMetadataBuilder,
)
from vllm.models.minimax_m3.nvidia import (
    msa_cutlass_sparse_decode as msa_cutlass_module,
)
from vllm.models.minimax_m3.nvidia.msa_cutlass_sparse_decode import (
    MSACutlassDecodeMetadata,
    MSACutlassDecodePlanCache,
    MSACutlassSparseDecodeRunner,
    _static_fallback_reason,
    prepare_decode_metadata,
)
from vllm.platforms import current_platform

if not current_platform.is_device_capability_family(100):
    pytest.skip(
        "fmha_sm100 sparse decode requires SM100 (Blackwell).",
        allow_module_level=True,
    )


NUM_Q_HEADS = 64
NUM_KV_HEADS = 4
HEAD_DIM = 128
BLOCK_SIZE = 128
TOPK = 16
SM_SCALE = HEAD_DIM**-0.5


def test_msa_cutlass_decode_falls_back_for_small_batches(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_MINIMAX_M3_MSA_DECODE_BACKEND", "cutlass")
    query = torch.empty(1, device="cuda")
    seq_lens = torch.empty(8, dtype=torch.int32, device="cuda")

    reason = _static_fallback_reason(
        query,
        query,
        query,
        seq_lens,
        query,
        MSACutlassDecodeMetadata(plan=None, page_table=query),
        num_kv_heads=NUM_KV_HEADS,
        block_size=BLOCK_SIZE,
        topk_blocks=TOPK,
        decode_query_len=4,
        q_scale=None,
    )

    assert reason == "batch size is below 16"


@pytest.mark.parametrize(
    ("query_len", "expected_reason"),
    [
        (0, "decode query length is outside [1, 32]"),
        (1, None),
        (32, None),
        (33, "decode query length is outside [1, 32]"),
    ],
)
def test_msa_cutlass_decode_query_len_bounds(
    monkeypatch: pytest.MonkeyPatch,
    query_len: int,
    expected_reason: str | None,
):
    monkeypatch.setenv("VLLM_MINIMAX_M3_MSA_DECODE_BACKEND", "cutlass")
    batch = 16
    total_q = batch * query_len
    query = torch.empty(
        total_q,
        NUM_Q_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    output = torch.empty_like(query)
    kv_cache = torch.empty(
        1,
        NUM_KV_HEADS,
        BLOCK_SIZE,
        2 * HEAD_DIM,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    topk = torch.empty(
        total_q,
        NUM_KV_HEADS,
        TOPK,
        dtype=torch.int32,
        device="cuda",
    )
    seq_lens = torch.full((batch,), 257, dtype=torch.int32, device="cuda")
    q_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    metadata = MSACutlassDecodeMetadata(
        plan=None,
        page_table=torch.empty(1, dtype=torch.int32, device="cuda"),
    )

    reason = _static_fallback_reason(
        query,
        kv_cache,
        topk,
        seq_lens,
        output,
        metadata,
        num_kv_heads=NUM_KV_HEADS,
        block_size=BLOCK_SIZE,
        topk_blocks=TOPK,
        decode_query_len=query_len,
        q_scale=q_scale,
    )

    assert reason == expected_reason


def test_msa_cutlass_plan_cache_keys_query_len(
    monkeypatch: pytest.MonkeyPatch,
):
    batch = 16
    block_table = torch.zeros(batch, 3, dtype=torch.int32, device="cuda")
    seq_lens = torch.full((batch,), 257, dtype=torch.int32, device="cuda")
    plan_cache = MSACutlassDecodePlanCache()
    built_query_lens = []

    def fake_build_plan(**kwargs):
        query_len = kwargs["decode_query_len"]
        built_query_lens.append(query_len)
        num_rows = batch * query_len
        return (
            None,
            None,
            None,
            {
                "kv_segment_lens": torch.empty(
                    num_rows, dtype=torch.int32, device="cuda"
                ),
                "qo_offset": torch.empty(num_rows, dtype=torch.int32, device="cuda"),
            },
        )

    monkeypatch.setattr(plan_cache, "_build_plan", fake_build_plan)
    first = prepare_decode_metadata(
        block_table,
        seq_lens,
        1,
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        page_size=BLOCK_SIZE,
        topk_blocks=TOPK,
        plan_cache=plan_cache,
    )
    repeated = prepare_decode_metadata(
        block_table,
        seq_lens,
        1,
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        page_size=BLOCK_SIZE,
        topk_blocks=TOPK,
        plan_cache=plan_cache,
    )
    different = prepare_decode_metadata(
        block_table,
        seq_lens,
        2,
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        page_size=BLOCK_SIZE,
        topk_blocks=TOPK,
        plan_cache=plan_cache,
    )
    current_platform.synchronize()

    assert first.plan is repeated.plan
    assert different.plan is not first.plan
    assert built_query_lens == [1, 2]


def test_sparse_metadata_builder_prepares_cutlass_for_regular_decode(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_MINIMAX_M3_MSA_DECODE_BACKEND", "cutlass")
    batch = 16
    query_start_loc_cpu = torch.arange(batch + 1, dtype=torch.int32)
    seq_lens = torch.full((batch,), 257, dtype=torch.int32, device="cuda")
    block_table = torch.zeros(batch, 3, dtype=torch.int32, device="cuda")
    common_metadata = SimpleNamespace(
        num_reqs=batch,
        num_actual_tokens=batch,
        query_start_loc=query_start_loc_cpu.cuda(),
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        block_table_tensor=block_table,
        slot_mapping=torch.zeros(batch, dtype=torch.int64, device="cuda"),
        max_seq_len=257,
        compute_num_computed_tokens=lambda: torch.zeros(
            batch, dtype=torch.int32, device="cuda"
        ),
    )
    builder = object.__new__(MiniMaxM3SparseMetadataBuilder)
    builder.reorder_batch_threshold = 1
    builder.context_len_buffer = torch.empty(batch, dtype=torch.int32, device="cuda")
    builder.num_q_heads = NUM_Q_HEADS
    builder.topk_blocks = TOPK
    builder.kv_cache_spec = SimpleNamespace(num_kv_heads=NUM_KV_HEADS)
    builder.msa_cutlass_plan_cache = object()
    expected_metadata = object()

    monkeypatch.setattr(
        sparse_attention_module,
        "split_decodes_and_prefills",
        lambda *args, **kwargs: (batch, 0, batch, 0),
    )
    monkeypatch.setattr(
        msa_cutlass_module,
        "prepare_decode_metadata",
        lambda *args, **kwargs: expected_metadata,
    )

    metadata = builder.build(0, common_metadata)

    assert metadata.decode is not None
    assert metadata.decode.decode_query_len == 1
    assert metadata.decode.msa_cutlass is expected_metadata


def _make_topk(seq_lens: list[int], query_len: int) -> torch.Tensor:
    topk = torch.full(
        (len(seq_lens) * query_len, NUM_KV_HEADS, TOPK),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    for request, seq_len in enumerate(seq_lens):
        for local_query in range(query_len):
            token = request * query_len + local_query
            visible_tokens = seq_len - query_len + local_query + 1
            visible_pages = math.ceil(visible_tokens / BLOCK_SIZE)
            topk[token, :, :visible_pages] = torch.arange(
                visible_pages, dtype=torch.int32, device="cuda"
            )
    return topk


@pytest.mark.parametrize(
    ("query_len", "capture_graph"),
    [(1, True), (2, False), (3, False), (4, True), (8, False)],
)
def test_msa_cutlass_decode_matches_triton_with_interleaved_cache(
    monkeypatch: pytest.MonkeyPatch,
    query_len: int,
    capture_graph: bool,
):
    monkeypatch.setenv("VLLM_MINIMAX_M3_MSA_DECODE_BACKEND", "cutlass")
    torch.manual_seed(0)
    seq_lens_list = [257, 513] * 8
    seq_lens_cpu = torch.tensor(seq_lens_list, dtype=torch.int32)
    seq_lens = seq_lens_cpu.cuda()
    pages_per_request = [math.ceil(seq_len / BLOCK_SIZE) for seq_len in seq_lens_list]
    num_pages = sum(pages_per_request)
    max_pages = max(pages_per_request)

    block_table = torch.zeros(
        len(seq_lens_list), max_pages, dtype=torch.int32, device="cuda"
    )
    physical_pages = torch.randperm(num_pages, dtype=torch.int32, device="cuda")
    offset = 0
    for request, request_pages in enumerate(pages_per_request):
        block_table[request, :request_pages] = physical_pages[
            offset : offset + request_pages
        ]
        offset += request_pages

    key = (
        torch.randn(
            num_pages,
            NUM_KV_HEADS,
            BLOCK_SIZE,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.25
    ).to(torch.float8_e4m3fn)
    value = (torch.randn_like(key, dtype=torch.bfloat16) * 0.25).to(torch.float8_e4m3fn)
    kv_cache = torch.cat((key, value), dim=-1)
    assert kv_cache.stride() == (
        NUM_KV_HEADS * BLOCK_SIZE * 2 * HEAD_DIM,
        BLOCK_SIZE * 2 * HEAD_DIM,
        2 * HEAD_DIM,
        1,
    )

    num_query_tokens = len(seq_lens_list) * query_len
    query = torch.randn(
        num_query_tokens,
        NUM_Q_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    q_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    query_fp8 = torch.empty_like(query, dtype=torch.float8_e4m3fn)
    ops.scaled_fp8_quant(
        query.view(num_query_tokens, -1),
        scale=q_scale,
        output=query_fp8.view(num_query_tokens, -1),
    )
    query_dequantized = query_fp8.to(torch.bfloat16) * q_scale

    topk_token_major = _make_topk(seq_lens_list, query_len)
    expected = torch.empty_like(query)
    minimax_m3_sparse_attn_decode(
        query_dequantized,
        kv_cache,
        topk_token_major.transpose(0, 1),
        block_table,
        seq_lens,
        NUM_KV_HEADS,
        SM_SCALE,
        expected,
        query_len,
        k_scale=None,
        v_scale=None,
    )

    plan_cache = MSACutlassDecodePlanCache()
    metadata = prepare_decode_metadata(
        block_table,
        seq_lens,
        query_len,
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        page_size=BLOCK_SIZE,
        topk_blocks=TOPK,
        plan_cache=plan_cache,
    )
    assert metadata.page_table.data_ptr() == block_table.data_ptr()
    actual = torch.empty_like(query)
    runner = MSACutlassSparseDecodeRunner()
    used = runner.try_decode(
        query,
        kv_cache,
        topk_token_major,
        seq_lens,
        actual,
        metadata,
        num_kv_heads=NUM_KV_HEADS,
        scale=SM_SCALE,
        block_size=BLOCK_SIZE,
        topk_blocks=TOPK,
        decode_query_len=query_len,
        q_scale=q_scale,
        q_scale_float=1.0,
        k_scale_float=1.0,
        v_scale_float=1.0,
    )

    assert used
    torch.testing.assert_close(runner._get_query_buffer(query), query_fp8)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)

    # The same captured plan must remain correct as ragged lengths change.
    updated_seq_lens_list = [129, 385] * 8
    seq_lens.copy_(
        torch.tensor(updated_seq_lens_list, dtype=torch.int32, device="cuda")
    )
    topk_token_major.copy_(_make_topk(updated_seq_lens_list, query_len))
    updated_metadata = prepare_decode_metadata(
        block_table,
        seq_lens,
        query_len,
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        page_size=BLOCK_SIZE,
        topk_blocks=TOPK,
        plan_cache=plan_cache,
    )
    assert updated_metadata.plan is metadata.plan
    assert updated_metadata.page_table.data_ptr() == metadata.page_table.data_ptr()

    minimax_m3_sparse_attn_decode(
        query_dequantized,
        kv_cache,
        topk_token_major.transpose(0, 1),
        block_table,
        seq_lens,
        NUM_KV_HEADS,
        SM_SCALE,
        expected,
        query_len,
        k_scale=None,
        v_scale=None,
    )
    assert runner.try_decode(
        query,
        kv_cache,
        topk_token_major,
        seq_lens,
        actual,
        updated_metadata,
        num_kv_heads=NUM_KV_HEADS,
        scale=SM_SCALE,
        block_size=BLOCK_SIZE,
        topk_blocks=TOPK,
        decode_query_len=query_len,
        q_scale=q_scale,
        q_scale_float=1.0,
        k_scale_float=1.0,
        v_scale_float=1.0,
    )
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)

    if capture_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            assert runner.try_decode(
                query,
                kv_cache,
                topk_token_major,
                seq_lens,
                actual,
                updated_metadata,
                num_kv_heads=NUM_KV_HEADS,
                scale=SM_SCALE,
                block_size=BLOCK_SIZE,
                topk_blocks=TOPK,
                decode_query_len=query_len,
                q_scale=q_scale,
                q_scale_float=1.0,
                k_scale_float=1.0,
                v_scale_float=1.0,
            )

        replay_seq_lens_list = [257, 513] * 8
        seq_lens.copy_(
            torch.tensor(replay_seq_lens_list, dtype=torch.int32, device="cuda")
        )
        topk_token_major.copy_(_make_topk(replay_seq_lens_list, query_len))
        prepare_decode_metadata(
            block_table,
            seq_lens,
            query_len,
            num_q_heads=NUM_Q_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            page_size=BLOCK_SIZE,
            topk_blocks=TOPK,
            plan_cache=plan_cache,
        )
        minimax_m3_sparse_attn_decode(
            query_dequantized,
            kv_cache,
            topk_token_major.transpose(0, 1),
            block_table,
            seq_lens,
            NUM_KV_HEADS,
            SM_SCALE,
            expected,
            query_len,
            k_scale=None,
            v_scale=None,
        )
        graph.replay()
        current_platform.synchronize()
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
