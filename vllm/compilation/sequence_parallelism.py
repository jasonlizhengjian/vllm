# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm.config import VllmConfig
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size)
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticTensorSym)
from vllm.platforms import current_platform

from .inductor_pass import enable_fake_mode
from .matcher_utils import MatcherFusedAddRMSNorm, MatcherQuant, MatcherRMSNorm
from .vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)


class _RMSNormAndQuantOpHelper:
    """Base helper for RMSNorm functionalization."""

    def __init__(self, epsilon: float, dtype: torch.dtype, device: str):
        self.epsilon = epsilon
        self.dtype = dtype
        self.device = device

    def _functional_rmsnorm(self, result_buffer, input_tensor, weight_tensor):
        return torch.ops.higher_order.auto_functionalized(
            torch.ops._C.rms_norm.default,
            result=result_buffer,
            input=input_tensor,
            weight=weight_tensor,
            epsilon=self.epsilon)

    def _functional_fused_add_rmsnorm(self, input_tensor, residual_tensor,
                                      weight_tensor):
        return torch.ops.higher_order.auto_functionalized(
            torch.ops._C.fused_add_rms_norm.default,
            input=input_tensor,
            residual=residual_tensor,
            weight=weight_tensor,
            epsilon=self.epsilon)


class _SequenceParallelPatternHelper(_RMSNormAndQuantOpHelper):
    """Helper for sequence parallelism patterns."""

    def __init__(self, epsilon: float, dtype: torch.dtype, device: str):
        super().__init__(epsilon, dtype, device)
        self.tp_group = get_tp_group()
        self.tp_size = get_tensor_model_parallel_world_size()

    def _all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return tensor_model_parallel_all_reduce(x)

    def _reduce_scatter(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.reduce_scatter.default(
            x,
            dim=0,
            world_size=self.tp_size,
            group_name=self.tp_group.unique_name)

    def _all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.all_gather.default(
            x,
            dim=0,
            world_size=self.tp_size,
            group_name=self.tp_group.unique_name)


class FirstAllReduceRMSNormPattern(_SequenceParallelPatternHelper):

    def get_inputs(self):
        input = torch.empty([1, 8, 4], device=self.device, dtype=self.dtype)
        permute = torch.empty([1, 8, 4], device=self.device, dtype=self.dtype)
        arg3_1 = torch.empty([4], device=self.device, dtype=self.dtype)

        return [input, permute, arg3_1]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(
            input: torch.Tensor,
            permute: torch.Tensor,
            arg3_1: torch.Tensor,
        ):
            all_reduce = self._all_reduce(input)
            rmsnorm = self._functional_rmsnorm(permute, all_reduce, arg3_1)

            return rmsnorm[1], all_reduce

        def replacement(
            input: torch.Tensor,
            permute: torch.Tensor,
            arg3_1: torch.Tensor,
        ):
            reduce_scatter = self._reduce_scatter(input)

            rmsnorm_result = torch.empty_like(reduce_scatter)
            rmsnorm = self._functional_rmsnorm(rmsnorm_result, reduce_scatter,
                                               arg3_1)

            all_gather = self._all_gather(rmsnorm[1])

            return all_gather, reduce_scatter

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class MiddleAllReduceRMSNormPattern(_SequenceParallelPatternHelper):

    def get_inputs(self):
        mm_1 = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        residual = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        rms_norm_weights = torch.empty([4, 4],
                                       device=self.device,
                                       dtype=self.dtype)

        return [
            residual,
            mm_1,
            rms_norm_weights,
        ]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_reduce = self._all_reduce(mm_1)
            rmsnorm = self._functional_fused_add_rmsnorm(
                all_reduce, residual, rms_norm_weights)
            return rmsnorm[1], rmsnorm[2]

        def replacement(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            reduce_scatter = self._reduce_scatter(mm_1)
            rmsnorm = self._functional_fused_add_rmsnorm(
                reduce_scatter, residual, rms_norm_weights)
            all_gather = self._all_gather(rmsnorm[1])
            return all_gather, rmsnorm[2]

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class LastAllReduceRMSNormPattern(_SequenceParallelPatternHelper):

    def get_inputs(self):
        mm_1 = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        residual = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        rms_norm_weights = torch.empty([4, 4],
                                       device=self.device,
                                       dtype=self.dtype)

        return [
            residual,
            mm_1,
            rms_norm_weights,
        ]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_reduce = self._all_reduce(mm_1)
            rmsnorm = self._functional_fused_add_rmsnorm(
                all_reduce, residual, rms_norm_weights)
            return rmsnorm[1]

        def replacement(
            residual: torch.Tensor,
            mm_1: torch.Tensor,
            rms_norm_weights: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            reduce_scatter = self._reduce_scatter(mm_1)
            rmsnorm = self._functional_fused_add_rmsnorm(
                reduce_scatter, residual, rms_norm_weights)
            normalized = self._all_gather(rmsnorm[1])
            return normalized

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


FP8_DTYPE = current_platform.fp8_dtype()


class FirstAllReduceRMSNormStaticFP8Pattern(_SequenceParallelPatternHelper):

    def __init__(self, epsilon: float, dtype: torch.dtype, device: str):
        super().__init__(epsilon, dtype, device)
        self.rmsnorm_matcher = MatcherRMSNorm(epsilon)
        self.quant_matcher = MatcherQuant(kFp8StaticTensorSym)

    def get_inputs(self):
        return [
            *self.rmsnorm_matcher.inputs(),  # input, weight
            torch.tensor(1.0, device=self.device,
                         dtype=torch.float32),  # scale
        ]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            all_reduce = self._all_reduce(input)
            rmsnorm_output = self.rmsnorm_matcher(all_reduce, weight)
            quant_output, _ = self.quant_matcher(rmsnorm_output, scale)
            return quant_output, all_reduce

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            reduce_scatter = self._reduce_scatter(input)
            rmsnorm_output = self.rmsnorm_matcher(reduce_scatter, weight)
            quant_output, _ = self.quant_matcher(rmsnorm_output, scale)
            all_gather = self._all_gather(quant_output)
            return all_gather, reduce_scatter

        inputs = self.get_inputs()
        pattern(*inputs)  # Trace the pattern

        pm.register_replacement(pattern, replacement, inputs, pm.fwd_only,
                                pm_pass)


class MiddleAllReduceRMSNormStaticFP8Pattern(_SequenceParallelPatternHelper):

    def __init__(self, epsilon: float, dtype: torch.dtype, device: str):
        super().__init__(epsilon, dtype, device)
        self.rmsnorm_matcher = MatcherFusedAddRMSNorm(epsilon)
        self.quant_matcher = MatcherQuant(kFp8StaticTensorSym)

    def get_inputs(self):
        mm_1 = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        scale = torch.empty([1, 1], device=self.device, dtype=torch.float32)
        return [
            mm_1,
            *self.rmsnorm_matcher.inputs(),  # input, weight, residual
            scale,
        ]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(
            mm_1: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_reduce = self._all_reduce(mm_1)
            rmsnorm_output, residual_output = self.rmsnorm_matcher(
                all_reduce, weight, residual)
            quant_output, _ = self.quant_matcher(rmsnorm_output, scale)
            return quant_output, residual_output

        def replacement(
            mm_1: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            reduce_scatter = self._reduce_scatter(mm_1)
            rmsnorm_output, residual_output = self.rmsnorm_matcher(
                reduce_scatter, weight, residual)
            quant_output, _ = self.quant_matcher(rmsnorm_output, scale)
            all_gather = self._all_gather(quant_output)
            return all_gather, residual_output

        inputs = self.get_inputs()
        pattern(*inputs)  # Trace the pattern

        pm.register_replacement(pattern, replacement, inputs, pm.fwd_only,
                                pm_pass)


class LastAllReduceRMSNormStaticFP8Pattern(_SequenceParallelPatternHelper):

    def __init__(self, epsilon: float, dtype: torch.dtype, device: str):
        super().__init__(epsilon, dtype, device)
        self.rmsnorm_matcher = MatcherFusedAddRMSNorm(epsilon)
        self.quant_matcher = MatcherQuant(kFp8StaticTensorSym)

    def get_inputs(self):
        mm_1 = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        scale = torch.empty([1, 1], device=self.device, dtype=torch.float32)
        return [
            mm_1,
            *self.rmsnorm_matcher.inputs(),  # input, weight, residual
            scale,
        ]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(
            mm_1: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            scale: torch.Tensor,
        ) -> torch.Tensor:
            all_reduce = self._all_reduce(mm_1)
            rmsnorm_output, _ = self.rmsnorm_matcher(all_reduce, weight,
                                                     residual)
            quant_output, _ = self.quant_matcher(rmsnorm_output, scale)
            return quant_output

        def replacement(
            mm_1: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            scale: torch.Tensor,
        ) -> torch.Tensor:
            reduce_scatter = self._reduce_scatter(mm_1)
            rmsnorm_output, _ = self.rmsnorm_matcher(reduce_scatter, weight,
                                                     residual)
            quant_output, _ = self.quant_matcher(rmsnorm_output, scale)
            all_gather = self._all_gather(quant_output)
            return all_gather

        inputs = self.get_inputs()
        pattern(*inputs)  # Trace the pattern

        pm.register_replacement(pattern, replacement, inputs, pm.fwd_only,
                                pm_pass)


class SequenceParallelismPass(VllmPatternMatcherPass):
    """
    This pass enables sequence parallelism for models.
    It identifies patterns where an AllReduce operation is followed by
    an RMSNorm (or RMSNorm and then Quantization) operation.
    These patterns are replaced with a ReduceScatter operation, followed by
    a local RMSNorm/Quantization, and then an AllGather operation.

    The general transformation is:
    Input -> AllReduce -> RMSNorm -> Output
    becomes
    Input -> ReduceScatter -> RMSNorm -> AllGather -> Output

    While this pass itself does not directly yield performance improvements,
    it lays the groundwork for subsequent fusion passes, such as
    GEMM + ReduceScatter and AllGather + GEMM fusions. These fusions can
    significantly reduce communication overhead and improve overall model
    performance.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig):
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="sequence_parallelism_pass")

        for epsilon in [1e-5, 1e-6]:
            # RMSNorm + Static FP8 quantization patterns
            # These now use matcher utilities to match both custom ops
            # and native decomposed primitives
            FirstAllReduceRMSNormStaticFP8Pattern(
                epsilon, self.model_dtype, self.device).register(self.patterns)
            MiddleAllReduceRMSNormStaticFP8Pattern(
                epsilon, self.model_dtype, self.device).register(self.patterns)
            LastAllReduceRMSNormStaticFP8Pattern(
                epsilon, self.model_dtype, self.device).register(self.patterns)

            # Normal RMSNorm patterns
            FirstAllReduceRMSNormPattern(epsilon, self.model_dtype,
                                         self.device).register(self.patterns)

            MiddleAllReduceRMSNormPattern(epsilon, self.model_dtype,
                                          self.device).register(self.patterns)

            LastAllReduceRMSNormPattern(epsilon, self.model_dtype,
                                        self.device).register(self.patterns)
        self.dump_patterns(config, self.patterns)

    def is_applicable_for_shape(self, shape: Optional[int]) -> bool:
        tp_size = get_tensor_model_parallel_world_size()
        return shape is not None and shape % tp_size == 0

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph):
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)
