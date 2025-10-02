#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quick experiment to see if FP4 fusion pattern works"""
import torch

from vllm.compilation.collective_fusion import (
    CutlassScaledFP4MMReduceScatterPattern)
from vllm.config import CompilationConfig, PassConfig, VllmConfig

# Create a minimal config
config = VllmConfig()
config.compilation_config = CompilationConfig(pass_config=PassConfig(
    enable_async_tp=True))

# Create the pattern
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

try:
    pattern = CutlassScaledFP4MMReduceScatterPattern(dtype, device)
    inputs = pattern.get_inputs()

    print("✓ Pattern created successfully")
    print(f"  Input shapes: {[x.shape for x in inputs]}")
    print(f"  Input dtypes: {[x.dtype for x in inputs]}")

    # The real test would be calling the fused op
    # But let's just see if we can construct the inputs
    print("\n✓ Experiment setup complete!")
    print("  Next step: Try running in actual async TP test")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
