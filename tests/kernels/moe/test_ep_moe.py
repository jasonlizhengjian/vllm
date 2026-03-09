# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test expert-parallel MoE with distributed communication.

Validates that sharding experts across multiple GPUs and combining results
via all-reduce produces the same output as a single-GPU PyTorch reference.

Run `pytest tests/kernels/moe/test_ep_moe.py`.
"""

import dataclasses

import pytest
import torch
import torch.distributed

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.workspace import init_workspace_manager

from ...utils import multi_gpu_test
from .parallel_utils import ProcessGroupInfo, parallel_launch


@dataclasses.dataclass
class TestConfig:
    m: int  # number of tokens
    n: int  # intermediate size
    k: int  # hidden size
    num_experts: int
    topk: int
    dtype: torch.dtype


def make_weights(
    e: int, n: int, k: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create random expert weights on CPU."""
    w1 = torch.randn((e, 2 * n, k), device="cpu", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cpu", dtype=dtype) / 10
    return w1, w2


def make_test_tensors(
    config: TestConfig, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create random input tokens, topk ids, and topk weights."""
    tokens = torch.randn((config.m, config.k), device=device, dtype=config.dtype) / 10
    topk_ids = torch.randint(
        low=0,
        high=config.num_experts,
        size=(config.m, config.topk),
        device=device,
        dtype=torch.int64,
    )
    topk_weights = torch.randn(
        (config.m, config.topk), dtype=torch.float32, device=device
    )
    return tokens, topk_ids, topk_weights


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


def _worker_ep_moe(
    pgi: ProcessGroupInfo,
    config: TestConfig,
    w1: torch.Tensor,
    w2: torch.Tensor,
):
    """Worker function executed on each GPU.

    Each rank:
    1. Computes local experts using fused_experts with expert_map
    2. All-reduces partial results across EP ranks
    3. Compares against single-GPU reference
    """
    device = pgi.device
    init_workspace_manager(device)

    with set_current_vllm_config(VllmConfig()):
        # Move full weights to device
        w1_dev = w1.to(device=device)
        w2_dev = w2.to(device=device)

        # Create test tensors (same seed on all ranks → same input)
        set_random_seed(42)
        tokens, topk_ids, topk_weights = make_test_tensors(config, device)

        # Single-GPU reference (full expert set)
        ref_output = torch_moe_ref(tokens, w1_dev, w2_dev, topk_ids, topk_weights)

        # Shard experts for this rank
        num_local_experts = config.num_experts // pgi.world_size
        e_start = pgi.rank * num_local_experts
        e_end = e_start + num_local_experts
        w1_local = w1_dev[e_start:e_end]
        w2_local = w2_dev[e_start:e_end]

        # Build global→local expert map (-1 = not on this rank)
        expert_map = torch.full(
            (config.num_experts,),
            fill_value=-1,
            dtype=torch.int32,
            device=device,
        )
        expert_map[e_start:e_end] = torch.arange(
            num_local_experts, dtype=torch.int32, device=device
        )

        # Compute local experts only (tokens routed to remote experts get 0)
        local_output = fused_experts(
            hidden_states=tokens,
            w1=w1_local,
            w2=w2_local,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=config.num_experts,
            expert_map=expert_map,
        )

        # All-reduce across EP ranks to combine partial expert results
        torch.distributed.all_reduce(local_output)

        torch.testing.assert_close(
            ref_output,
            local_output,
            atol=5e-2,
            rtol=5e-2,
        )


MNKs = [
    (2, 128, 128),
    (32, 128, 512),
    (64, 512, 1024),
    (128, 1024, 2048),
]


@pytest.mark.parametrize("m,n,k", MNKs)
@pytest.mark.parametrize("num_experts", [8, 32])
@pytest.mark.parametrize("topk", [2, 6])
@pytest.mark.parametrize("world_size", [2])
@multi_gpu_test(num_gpus=2)
def test_ep_moe(
    m: int,
    n: int,
    k: int,
    num_experts: int,
    topk: int,
    world_size: int,
):
    """Test expert-parallel MoE: shard experts across GPUs, all-reduce, compare."""
    set_random_seed(7)
    dtype = torch.bfloat16
    config = TestConfig(m=m, n=n, k=k, num_experts=num_experts, topk=topk, dtype=dtype)
    w1, w2 = make_weights(num_experts, n, k, dtype)

    parallel_launch(world_size, _worker_ep_moe, config, w1, w2)
