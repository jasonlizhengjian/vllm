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


def _distributed_run(fn, world_size, *args):
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
    torch.cuda.set_device(rank)

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
        init_workspace_manager(torch.cuda.current_device())

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

    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

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
