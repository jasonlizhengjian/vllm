# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Layer-level test for FusedMoE with DP+EP and all-to-all backends.

Exercises the full FusedMoE nn.Module end-to-end including:
- Router (fused_topk)
- All-to-all dispatch/combine (DeepEP HT, DeepEP LL, allgather-reducescatter)
- Weight handling and expert sharding
- The DefaultMoERunner orchestration

Run: pytest -v -s tests/kernels/moe/test_fused_moe_ep_dp.py
"""

import pytest
import torch
import torch.multiprocessing as mp

import vllm._custom_ops as ops
from tests.kernels.moe.utils import make_test_weights
from tests.kernels.quantization.nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    dequantize_nvfp4_to_dtype,
)
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    fused_topk,
)
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_deep_ep
from vllm.utils.system_utils import update_environment_variables
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.workspace import init_workspace_manager

from ...utils import multi_gpu_test

mp.set_start_method("spawn", force=True)


def torch_moe_ref(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Single-GPU PyTorch MoE reference (all experts, no sharding)."""
    m, _ = a.shape
    topk = topk_ids.size(1)
    out = torch.zeros_like(a)
    for i in range(m):
        for j in range(topk):
            e = topk_ids[i][j]
            e_w = topk_weights[i][j]
            out[i] += (
                SiluAndMul()(a[i] @ w1[e].transpose(0, 1)) @ w2[e].transpose(0, 1)
            ) * e_w
    return out


def _distributed_run(fn, world_size, *args, extra_env=None):
    """Launch fn across world_size processes, following eplb_utils pattern."""
    processes: list[mp.Process] = []
    for i in range(world_size):
        env: dict[str, str] = {
            "RANK": str(i),
            "LOCAL_RANK": str(i),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12345",
        }
        if extra_env:
            env.update(extra_env)
        p = mp.Process(target=fn, args=(env, world_size, *args))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    for p in processes:
        assert p.exitcode == 0


def _worker(
    env: dict[str, str],
    world_size: int,
    backend: str,
    num_experts: int,
    topk: int,
    hidden_size: int,
    intermediate_size: int,
    M: int,
    dtype: torch.dtype,
):
    rank = int(env["RANK"])

    # 1. Set env vars and device
    update_environment_variables(env)
    torch.accelerator.set_device_index(rank)

    # 2. Init vLLM world group with bare config
    bare_config = VllmConfig()
    with set_current_vllm_config(bare_config):
        init_distributed_environment()

    # 3. Configure DP+EP
    vllm_config = VllmConfig()
    vllm_config.parallel_config.data_parallel_size = world_size
    vllm_config.parallel_config.data_parallel_rank = rank
    vllm_config.parallel_config.enable_expert_parallel = True
    vllm_config.parallel_config.all2all_backend = backend
    vllm_config.parallel_config.is_moe_model = True
    vllm_config.compilation_config.fast_moe_cold_start = False

    with set_current_vllm_config(vllm_config):
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )
        init_workspace_manager(torch.accelerator.current_device_index())

        # 4. Generate full weights (same seed on all ranks)
        set_random_seed(42)
        w13_full = (
            torch.randn(
                num_experts,
                2 * intermediate_size,
                hidden_size,
                device="cuda",
                dtype=dtype,
            )
            / 10
        )
        w2_full = (
            torch.randn(
                num_experts,
                hidden_size,
                intermediate_size,
                device="cuda",
                dtype=dtype,
            )
            / 10
        )

        # 5. Generate per-rank input (different per rank)
        set_random_seed(100 + rank)
        hidden_states = (
            torch.randn(
                M,
                hidden_size,
                device="cuda",
                dtype=dtype,
            )
            / 10
        )
        router_logits = (
            torch.randn(
                M,
                num_experts,
                device="cuda",
                dtype=dtype,
            )
            / 10
        )

        # 6. Compute reference output using all experts locally
        topk_weights, topk_ids, _ = fused_topk(
            hidden_states,
            router_logits.float(),
            topk,
            renormalize=True,
        )
        ref_output = torch_moe_ref(
            hidden_states,
            w13_full,
            w2_full,
            topk_ids,
            topk_weights,
        )

        # 7. Create FusedMoE layer
        fml = FusedMoE(
            num_experts=num_experts,
            top_k=topk,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            prefix=f"test_layer_{rank}",
            activation="silu",
            is_act_and_mul=True,
            params_dtype=dtype,
            reduce_results=False,
        )

        # Move layer to GPU (ensures expert_map buffer is on device)
        fml = fml.to("cuda")

        # 8. Fill local expert weights from the full set
        ep_rank = fml.moe_parallel_config.ep_rank
        n_local = fml.local_num_experts
        fml.w13_weight.data.copy_(w13_full[ep_rank * n_local : (ep_rank + 1) * n_local])
        fml.w2_weight.data.copy_(w2_full[ep_rank * n_local : (ep_rank + 1) * n_local])

        # 9. Process weights (sets up kernel) and init modular kernel
        #    (creates all2all prepare_finalize for DeepEP backends)
        fml.quant_method.process_weights_after_loading(fml)
        fml.maybe_init_modular_kernel()

        # 10. Forward with DPMetadata
        num_tokens_across_dp = torch.tensor(
            [M] * world_size,
            dtype=torch.int,
            device="cpu",
        )
        with set_forward_context(
            None,
            vllm_config,
            num_tokens=M,
            num_tokens_across_dp=num_tokens_across_dp,
        ):
            output = fml(hidden_states, router_logits)

        # 11. Compare against reference
        torch.testing.assert_close(
            ref_output,
            output,
            atol=5e-2,
            rtol=5e-2,
        )


BACKENDS = [
    "allgather_reducescatter",
    "deepep_high_throughput",
    "deepep_low_latency",
]

SHAPES = [
    # (M, hidden_size, intermediate_size)
    (16, 256, 512),
    (64, 512, 1024),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("m,hidden_size,intermediate_size", SHAPES)
@pytest.mark.parametrize("num_experts,topk", [(8, 2)])
@multi_gpu_test(num_gpus=2)
def test_fused_moe_ep_dp(
    backend: str,
    m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
):
    world_size = 2

    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Need {world_size} GPUs, have {torch.accelerator.device_count()}")

    if (
        backend in ("deepep_high_throughput", "deepep_low_latency")
        and not has_deep_ep()
    ):
        pytest.skip("DeepEP not available")

    dtype = torch.bfloat16
    _distributed_run(
        _worker,
        world_size,
        backend,
        num_experts,
        topk,
        hidden_size,
        intermediate_size,
        m,
        dtype,
    )


# ---------------------------------------------------------------------------
# NVFP4 + DP+EP test (CuTeDSL experts + DeepEP LL)
# ---------------------------------------------------------------------------


def _worker_nvfp4(
    env: dict[str, str],
    world_size: int,
    backend: str,
    num_experts: int,
    topk: int,
    hidden_size: int,
    intermediate_size: int,
    M: int,
    dtype: torch.dtype,
):
    rank = int(env["RANK"])

    # 1. Set env vars and device
    update_environment_variables(env)
    torch.accelerator.set_device_index(rank)

    # 2. Init vLLM world group with bare config
    bare_config = VllmConfig()
    with set_current_vllm_config(bare_config):
        init_distributed_environment()

    # 3. Configure DP+EP
    vllm_config = VllmConfig()
    vllm_config.parallel_config.data_parallel_size = world_size
    vllm_config.parallel_config.data_parallel_rank = rank
    vllm_config.parallel_config.enable_expert_parallel = True
    vllm_config.parallel_config.all2all_backend = backend
    vllm_config.parallel_config.is_moe_model = True
    vllm_config.compilation_config.fast_moe_cold_start = False

    with set_current_vllm_config(vllm_config):
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )
        init_workspace_manager(torch.accelerator.current_device_index())

        # 4. Generate full quantized weights (same seed on all ranks)
        set_random_seed(42)
        (_, w13_q, w13_bs, w13_gs), (_, w2_q, w2_bs, w2_gs) = make_test_weights(
            num_experts,
            intermediate_size,
            hidden_size,
            in_dtype=dtype,
            quant_dtype="nvfp4",
        )

        # 5. Dequantize weights for reference (before process_weights_after_loading)
        w13_deq = torch.empty(
            num_experts, 2 * intermediate_size, hidden_size, device="cuda", dtype=dtype
        )
        w2_deq = torch.empty(
            num_experts, hidden_size, intermediate_size, device="cuda", dtype=dtype
        )
        for idx in range(num_experts):
            w13_deq[idx] = dequantize_nvfp4_to_dtype(
                w13_q[idx], w13_bs[idx], w13_gs[idx], dtype, w13_q.device
            )
            w2_deq[idx] = dequantize_nvfp4_to_dtype(
                w2_q[idx], w2_bs[idx], w2_gs[idx], dtype, w2_q.device
            )

        # 6. Generate per-rank input (different per rank)
        set_random_seed(100 + rank)
        hidden_states = torch.randn(M, hidden_size, device="cuda", dtype=dtype) / 10
        router_logits = torch.randn(M, num_experts, device="cuda", dtype=dtype) / 10

        # 7. Compute reference: dequantize activations + torch_moe_ref
        a_global_scale = (
            (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX)
            / torch.amax(hidden_states.abs().flatten(), dim=-1)
        ).to(torch.float32)
        a_fp4, a_scale = ops.scaled_fp4_quant(hidden_states, a_global_scale)
        a_deq = dequantize_nvfp4_to_dtype(
            a_fp4, a_scale, a_global_scale, dtype, hidden_states.device
        )

        topk_weights, topk_ids, _ = fused_topk(
            hidden_states, router_logits.float(), topk, renormalize=True
        )
        ref_output = torch_moe_ref(a_deq, w13_deq, w2_deq, topk_ids, topk_weights)

        # 8. Create FusedMoE layer with NVFP4 quant config
        quant_config = ModelOptNvFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=None,
            exclude_modules=[],
        )
        fml = FusedMoE(
            num_experts=num_experts,
            top_k=topk,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            prefix=f"test_nvfp4_layer_{rank}",
            activation="silu",
            is_act_and_mul=True,
            params_dtype=dtype,
            reduce_results=False,
            quant_config=quant_config,
        )
        fml = fml.to("cuda")

        # 9. Fill local expert weights and scales
        ep_rank = fml.moe_parallel_config.ep_rank
        n_local = fml.local_num_experts
        start = ep_rank * n_local
        end = (ep_rank + 1) * n_local

        fml.w13_weight.data.copy_(w13_q[start:end])
        fml.w2_weight.data.copy_(w2_q[start:end])
        fml.w13_weight_scale.data.copy_(w13_bs[start:end])
        fml.w2_weight_scale.data.copy_(w2_bs[start:end])

        # Per-expert global weight scales
        fml.w13_weight_scale_2.data[:, 0] = w13_gs[start:end]
        fml.w13_weight_scale_2.data[:, 1] = w13_gs[start:end]
        fml.w2_weight_scale_2.data.copy_(w2_gs[start:end])

        # Input scales: 1.0 (simplifies reference; activations are small)
        fml.w13_input_scale.data.fill_(1.0)
        fml.w2_input_scale.data.fill_(1.0)

        # 10. Process weights (creates kernel + DeepEP LL handle)
        fml.quant_method.process_weights_after_loading(fml)
        fml.maybe_init_modular_kernel()

        # 11. Forward with DPMetadata
        num_tokens_across_dp = torch.tensor(
            [M] * world_size, dtype=torch.int, device="cpu"
        )
        with set_forward_context(
            None,
            vllm_config,
            num_tokens=M,
            num_tokens_across_dp=num_tokens_across_dp,
        ):
            output = fml(hidden_states, router_logits)

        # 12. Compare against reference (loose tolerance for quantization error)
        torch.testing.assert_close(ref_output, output, atol=1e-1, rtol=1e-1)


NVFP4_BACKENDS = ["deepep_low_latency"]

NVFP4_SHAPES = [
    # (M, hidden_size, intermediate_size)
    # hidden_size must be in DeepEP LL SUPPORTED_HIDDEN_SIZES
    (16, 2048, 256),
    (64, 4096, 512),
]


@pytest.mark.parametrize("backend", NVFP4_BACKENDS)
@pytest.mark.parametrize("m,hidden_size,intermediate_size", NVFP4_SHAPES)
@pytest.mark.parametrize("num_experts,topk", [(8, 2)])
@multi_gpu_test(num_gpus=2)
def test_fused_moe_ep_dp_nvfp4(
    backend: str,
    m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
):
    world_size = 2

    if not current_platform.has_device_capability(100):
        pytest.skip("NVFP4 CuTeDSL requires SM100+ (Blackwell)")

    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Need {world_size} GPUs, have {torch.accelerator.device_count()}")

    if not has_deep_ep():
        pytest.skip("DeepEP not available")

    dtype = torch.bfloat16
    _distributed_run(
        _worker_nvfp4,
        world_size,
        backend,
        num_experts,
        topk,
        hidden_size,
        intermediate_size,
        m,
        dtype,
        extra_env={
            "VLLM_USE_FLASHINFER_MOE_FP4": "1",
            "VLLM_FLASHINFER_MOE_BACKEND": "cutedsl",
        },
    )
