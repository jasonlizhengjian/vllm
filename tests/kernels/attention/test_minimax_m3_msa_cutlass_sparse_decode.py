# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for MiniMax M3 CUTLASS sparse speculative decode."""

import math

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.models.minimax_m3.common.ops.sparse_attn import (
    minimax_m3_sparse_attn_decode,
)
from vllm.models.minimax_m3.nvidia.msa_cutlass_sparse_decode import (
    MSACutlassDecodePlanCache,
    MSACutlassDecodeMetadata,
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
QUERY_LEN = 4
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
        decode_query_len=QUERY_LEN,
        q_scale=None,
    )

    assert reason == "batch size is below 16"


def _make_topk(seq_lens: list[int]) -> torch.Tensor:
    topk = torch.full(
        (sum(QUERY_LEN for _ in seq_lens), NUM_KV_HEADS, TOPK),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    for request, seq_len in enumerate(seq_lens):
        for local_query in range(QUERY_LEN):
            token = request * QUERY_LEN + local_query
            visible_tokens = seq_len - QUERY_LEN + local_query + 1
            visible_pages = math.ceil(visible_tokens / BLOCK_SIZE)
            topk[token, :, :visible_pages] = torch.arange(
                visible_pages, dtype=torch.int32, device="cuda"
            )
    return topk


def test_msa_cutlass_decode_matches_triton_with_interleaved_cache(
    monkeypatch: pytest.MonkeyPatch,
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

    num_query_tokens = len(seq_lens_list) * QUERY_LEN
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

    topk_token_major = _make_topk(seq_lens_list)
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
        QUERY_LEN,
        k_scale=None,
        v_scale=None,
    )

    plan_cache = MSACutlassDecodePlanCache()
    metadata = prepare_decode_metadata(
        block_table,
        seq_lens,
        QUERY_LEN,
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
        decode_query_len=QUERY_LEN,
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
    topk_token_major.copy_(_make_topk(updated_seq_lens_list))
    updated_metadata = prepare_decode_metadata(
        block_table,
        seq_lens,
        QUERY_LEN,
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
        QUERY_LEN,
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
        decode_query_len=QUERY_LEN,
        q_scale=q_scale,
        q_scale_float=1.0,
        k_scale_float=1.0,
        v_scale_float=1.0,
    )
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)

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
            decode_query_len=QUERY_LEN,
            q_scale=q_scale,
            q_scale_float=1.0,
            k_scale_float=1.0,
            v_scale_float=1.0,
        )

    replay_seq_lens_list = [257, 513] * 8
    seq_lens.copy_(torch.tensor(replay_seq_lens_list, dtype=torch.int32, device="cuda"))
    topk_token_major.copy_(_make_topk(replay_seq_lens_list))
    prepare_decode_metadata(
        block_table,
        seq_lens,
        QUERY_LEN,
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
        QUERY_LEN,
        k_scale=None,
        v_scale=None,
    )
    graph.replay()
    current_platform.synchronize()
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
