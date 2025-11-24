# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Code inside this file can safely assume cuda platform, e.g. importing
pynvml. However, it should not initialize cuda context.
"""

import os
from collections.abc import Callable
from ctypes import CDLL, POINTER, Structure, c_int, c_uint, c_ulong, get_errno
from functools import cache, wraps
from typing import TYPE_CHECKING, TypeVar

import torch
from typing_extensions import ParamSpec

# import custom ops, trigger op registration
import vllm._C  # noqa
import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.import_utils import import_pynvml
from vllm.utils.torch_utils import cuda_device_count_stateless

from .interface import DeviceCapability, Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.attention.backends.registry import AttentionBackendEnum
    from vllm.config import VllmConfig
    from vllm.config.cache import CacheDType
else:
    AttentionBackendEnum = None
    VllmConfig = None
    CacheDType = None

logger = init_logger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")

pynvml = import_pynvml()

# pytorch 2.5 uses cudnn sdpa by default, which will cause crash on some models
# see https://github.com/huggingface/diffusers/issues/9704 for details
torch.backends.cuda.enable_cudnn_sdp(False)


def _migrate_pages_via_libnuma(
    pid: int, source_nodes: list[int], target_node: int
) -> bool:
    """
    Migrate memory pages from source NUMA nodes to target NUMA node using libnuma.
    
    This uses ctypes to directly call numa_migrate_pages() from libnuma,
    which is more efficient than subprocess calls to migratepages.
    
    The approach is based on the CPU backend implementation in csrc/cpu/utils.cpp:
    - Get current memory binding (membind) to identify source nodes
    - Create a bitmask with source nodes XOR target node
    - Call numa_migrate_pages(pid, from_mask, to_mask)
    
    Args:
        pid: Process ID whose memory pages should be migrated
        source_nodes: List of source NUMA node IDs
        target_node: Target NUMA node ID
        
    Returns:
        True if migration was successful, False if libnuma is not available
        or migration failed.
    """
    try:
        # Try to load libnuma
        libnuma = CDLL("libnuma.so.1", use_errno=True)
        
        # Check if NUMA is available
        numa_available = libnuma.numa_available
        numa_available.restype = c_int
        if numa_available() == -1:
            logger.debug("NUMA is not available on this system")
            return False
        
        # Get the maximum number of nodes to determine bitmask size
        numa_max_node = libnuma.numa_max_node
        numa_max_node.restype = c_int
        max_node = numa_max_node()
        
        if max_node < 0:
            logger.debug("Failed to get max NUMA node")
            return False
        
        # Define struct bitmask structure matching libnuma
        # struct bitmask {
        #     unsigned long size;  /* number of bits in the map */
        #     unsigned long *maskp;
        # }
        from ctypes import POINTER, Structure, c_void_p
        
        class Bitmask(Structure):
            _fields_ = [("size", c_ulong), ("maskp", POINTER(c_ulong))]
        
        # Get function signatures
        numa_allocate_nodemask = libnuma.numa_allocate_nodemask
        numa_allocate_nodemask.restype = POINTER(Bitmask)
        
        numa_bitmask_setbit = libnuma.numa_bitmask_setbit
        numa_bitmask_setbit.argtypes = [POINTER(Bitmask), c_uint]
        numa_bitmask_setbit.restype = POINTER(Bitmask)
        
        numa_bitmask_clearbit = libnuma.numa_bitmask_clearbit
        numa_bitmask_clearbit.argtypes = [POINTER(Bitmask), c_uint]
        numa_bitmask_clearbit.restype = POINTER(Bitmask)
        
        numa_migrate_pages = libnuma.numa_migrate_pages
        numa_migrate_pages.argtypes = [c_int, POINTER(Bitmask), POINTER(Bitmask)]
        numa_migrate_pages.restype = c_int
        
        # Try to get the free function - different names on different versions
        # numa_bitmask_free is the modern API, numa_free_nodemask is older
        try:
            numa_free_fn = libnuma.numa_bitmask_free
        except AttributeError:
            try:
                numa_free_fn = libnuma.numa_free_nodemask
            except AttributeError:
                logger.debug("Neither numa_bitmask_free nor numa_free_nodemask found")
                return False
        
        numa_free_fn.argtypes = [POINTER(Bitmask)]
        numa_free_fn.restype = None
        
        # Create bitmasks for source and target
        from_mask = numa_allocate_nodemask()
        to_mask = numa_allocate_nodemask()
        
        if not from_mask or not to_mask:
            logger.debug("Failed to allocate NUMA node masks")
            return False
        
        try:
            # Set bits for source nodes
            for source_node in source_nodes:
                numa_bitmask_setbit(from_mask, source_node)
            
            # Set bit for target node
            numa_bitmask_setbit(to_mask, target_node)
            
            # Call numa_migrate_pages
            # Return value: number of pages that could NOT be moved (0 = all pages moved)
            # -1 on error
            result = numa_migrate_pages(pid, from_mask, to_mask)
            
            if result == -1:
                errno = get_errno()
                logger.warning(
                    "numa_migrate_pages failed for pid %d: errno=%d", pid, errno
                )
                return False
            
            # result is the number of pages that could NOT be moved
            # 0 means all pages were successfully migrated
            if result == 0:
                logger.info(
                    "Successfully migrated all pages from nodes %s to node %d for pid %d using libnuma system call",
                    source_nodes,
                    target_node,
                    pid,
                )
            else:
                logger.warning(
                    "Migrated pages from nodes %s to node %d for pid %d using libnuma system call, "
                    "but %d pages could not be moved",
                    source_nodes,
                    target_node,
                    pid,
                    result,
                )
            return True
            
        finally:
            # Clean up allocated masks
            numa_free_fn(from_mask)
            numa_free_fn(to_mask)
        
    except (OSError, AttributeError) as e:
        # libnuma not available or function not found
        logger.debug("libnuma not available for direct memory migration: %s", str(e))
        return False
    except Exception as e:
        # Catch any other errors during migration
        logger.warning("Error during libnuma memory migration: %s", str(e))
        return False


@cache
def _get_backend_priorities(
    use_mla: bool,
    device_capability: DeviceCapability,
) -> list[AttentionBackendEnum]:
    """Get backend priorities with lazy import to avoid circular dependency."""
    from vllm.attention.backends.registry import AttentionBackendEnum

    if use_mla:
        if device_capability.major == 10:
            return [
                AttentionBackendEnum.CUTLASS_MLA,
                AttentionBackendEnum.FLASHINFER_MLA,
                AttentionBackendEnum.FLASH_ATTN_MLA,
                AttentionBackendEnum.FLASHMLA,
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASHMLA_SPARSE,
            ]
        else:
            return [
                AttentionBackendEnum.FLASH_ATTN_MLA,
                AttentionBackendEnum.FLASHMLA,
                AttentionBackendEnum.FLASHINFER_MLA,
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASHMLA_SPARSE,
            ]
    else:
        if device_capability.major == 10:
            return [
                AttentionBackendEnum.FLASHINFER,
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.TRITON_ATTN,
                AttentionBackendEnum.FLEX_ATTENTION,
            ]
        else:
            return [
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.FLASHINFER,
                AttentionBackendEnum.TRITON_ATTN,
                AttentionBackendEnum.FLEX_ATTENTION,
            ]


def with_nvml_context(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    @wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        pynvml.nvmlInit()
        try:
            return fn(*args, **kwargs)
        finally:
            pynvml.nvmlShutdown()

    return wrapper


class CudaPlatformBase(Platform):
    _enum = PlatformEnum.CUDA
    device_name: str = "cuda"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"
    ray_device_key: str = "GPU"
    dist_backend: str = "nccl"
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"

    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        if self.has_device_capability(80):
            # Ampere and Hopper or later NVIDIA GPUs.
            return [torch.bfloat16, torch.float16, torch.float32]
        if self.has_device_capability(60):
            # Pascal, Volta and Turing NVIDIA GPUs, BF16 is not supported
            return [torch.float16, torch.float32]
        # Kepler and Maxwell NVIDIA GPUs, only FP32 is supported,
        # though vLLM doesn't support these GPUs.
        return [torch.float32]

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """
        Set the device for the current platform.
        """
        torch.cuda.set_device(device)
        # With this trick we can force the device to be set eagerly
        # see https://github.com/pytorch/pytorch/issues/155668
        # for why and when it is needed
        _ = torch.zeros(1, device=device)

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        raise NotImplementedError

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        raise NotImplementedError

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        raise NotImplementedError

    @classmethod
    def is_fully_connected(cls, device_ids: list[int]) -> bool:
        raise NotImplementedError

    @classmethod
    def log_warnings(cls):
        pass

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        parallel_config = vllm_config.parallel_config
        model_config = vllm_config.model_config

        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"

        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 16

        # TODO(lucas): handle this more gracefully
        # Note: model_config may be None during testing
        # Note: block_size is initialized in
        # HybridAttentionMambaModelConfig.verify_and_update_config
        # for models with both attention and mamba,
        # and doesn't need to be reinitialized here
        if (
            model_config is not None
            and model_config.use_mla
            and cache_config.block_size is not None
        ):
            use_sparse = hasattr(vllm_config.model_config.hf_config, "index_topk")
            # If `VLLM_ATTENTION_BACKEND` is not set and we are using MLA,
            # then we default to FlashMLA backend for non-blackwell GPUs,
            # else we default to CutlassMLA. For each case, we force the
            # required block_size.
            use_flashmla = False
            use_cutlass_mla = False
            use_flashinfer_mla = False

            if envs.VLLM_ATTENTION_BACKEND is None:
                # Default case
                if cls.is_device_capability(100):
                    # Blackwell => Force CutlassMLA.
                    use_cutlass_mla = True
                    # TODO: This does not work, because the
                    # global_force_attn_backend_context_manager is not set.
                    # See vllm/attention/selector.py:_cached_get_attn_backend
                    envs.VLLM_ATTENTION_BACKEND = "CUTLASS_MLA"
                else:
                    # Not Blackwell
                    use_flashmla = True
            else:
                # Forced case
                use_flashmla = envs.VLLM_ATTENTION_BACKEND == "FLASHMLA"
                use_cutlass_mla = envs.VLLM_ATTENTION_BACKEND == "CUTLASS_MLA"
                use_flashinfer_mla = envs.VLLM_ATTENTION_BACKEND == "FLASHINFER_MLA"

            from vllm.attention.ops.flashmla import is_flashmla_dense_supported

            if (
                use_flashmla
                and is_flashmla_dense_supported()[0]
                and cache_config.block_size % 64 != 0
            ):
                cache_config.block_size = 64
                logger.info("Forcing kv cache block size to 64 for FlashMLA backend.")

            if use_cutlass_mla and cache_config.block_size % 128 != 0:
                cache_config.block_size = 128
                logger.info(
                    "Forcing kv cache block size to 128 for CUTLASS_MLA backend."
                )

            if (
                use_flashinfer_mla
                and cache_config.block_size != 32
                and cache_config.block_size % 64 != 0
            ):
                cache_config.block_size = 64
                logger.info(
                    "Forcing kv cache block size to 64 for FlashInferMLA backend."
                )

            # TODO(Chen): remove this hacky code
            if use_sparse and cache_config.block_size != 64:
                cache_config.block_size = 64
                logger.info(
                    "Forcing kv cache block size to 64 for FlashMLASparse backend."
                )
        # lazy import to avoid circular import
        from vllm.config import CUDAGraphMode

        compilation_config = vllm_config.compilation_config
        if (
            parallel_config.all2all_backend == "deepep_high_throughput"
            and parallel_config.data_parallel_size > 1
            and compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            # TODO: Piecewise Cuda graph might be enabled
            # if torch compile cache key issue fixed
            # See https://github.com/vllm-project/vllm/pull/25093
            logger.info(
                "WideEP: Disabling CUDA Graphs since DeepEP high-throughput "
                "kernels are optimized for prefill and are incompatible with "
                "CUDA Graphs. "
                "In order to use CUDA Graphs for decode-optimized workloads, "
                "use --all2all-backend with another option, such as "
                "deepep_low_latency, pplx, or allgather_reducescatter."
            )
            compilation_config.cudagraph_mode = CUDAGraphMode.NONE

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        return torch.cuda.max_memory_allocated(device)

    @classmethod
    def get_vit_attn_backend(
        cls, head_size: int, dtype: torch.dtype
    ) -> "AttentionBackendEnum":
        from vllm.attention.backends.registry import AttentionBackendEnum

        # Try FlashAttention first
        try:
            backend_class = AttentionBackendEnum.FLASH_ATTN.get_class()
            if backend_class.supports_head_size(
                head_size
            ) and backend_class.supports_dtype(dtype):
                return AttentionBackendEnum.FLASH_ATTN
        except ImportError:
            pass

        return AttentionBackendEnum.TORCH_SDPA

    @classmethod
    def get_valid_backends(
        cls,
        head_size,
        dtype,
        kv_cache_dtype,
        block_size,
        use_mla,
        has_sink,
        use_sparse,
        device_capability,
        attn_type,
    ) -> tuple[
        list[tuple["AttentionBackendEnum", int]],
        dict["AttentionBackendEnum", list[str]],
    ]:
        valid_backends_priorities = []
        invalid_reasons = {}

        backend_priorities = _get_backend_priorities(use_mla, device_capability)
        for priority, backend in enumerate(backend_priorities):
            try:
                backend_class = backend.get_class()
                invalid_reasons_i = backend_class.validate_configuration(
                    head_size,
                    dtype,
                    kv_cache_dtype,
                    block_size,
                    use_mla,
                    has_sink,
                    use_sparse,
                    device_capability,
                    attn_type,
                )
            except ImportError:
                invalid_reasons_i = ["ImportError"]
            if invalid_reasons_i:
                invalid_reasons[backend] = invalid_reasons_i
            else:
                valid_backends_priorities.append((backend, priority))

        return valid_backends_priorities, invalid_reasons

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: "CacheDType | None",
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        attn_type: str | None = None,
    ) -> str:
        from vllm.attention import AttentionType

        if attn_type is None:
            attn_type = AttentionType.DECODER

        device_capability = cls.get_device_capability()
        assert device_capability is not None

        # First try checking just the selected backend, if there is one.
        if selected_backend is not None:
            try:
                backend_class = selected_backend.get_class()
                invalid_reasons = backend_class.validate_configuration(
                    head_size,
                    dtype,
                    kv_cache_dtype,
                    None,
                    use_mla,
                    has_sink,
                    use_sparse,
                    device_capability,
                    attn_type,
                )
            except ImportError:
                invalid_reasons = ["ImportError"]
            if invalid_reasons:
                raise ValueError(
                    f"Selected backend {selected_backend} is not valid for "
                    f"this configuration. Reason: {invalid_reasons}"
                )
            else:
                logger.info("Using %s backend.", selected_backend)
                return selected_backend.get_path()

        # No selected backend or the selected backend is invalid,
        # so we try finding a valid backend.
        valid_backends_priorities, invalid_reasons = cls.get_valid_backends(
            head_size,
            dtype,
            kv_cache_dtype,
            None,
            use_mla,
            has_sink,
            use_sparse,
            device_capability,
            attn_type,
        )
        reasons_str = (
            "{"
            + ", ".join(
                f"{backend.name}: [{', '.join(reasons)}]"
                for backend, reasons in invalid_reasons.items()
            )
            + "}"
        )
        config_str = (
            f"head_size: {head_size}, dtype: {dtype}, "
            f"kv_cache_dtype: {kv_cache_dtype}, block_size: {block_size}, "
            f"use_mla: {use_mla}, has_sink: {has_sink}, use_sparse: {use_sparse}"
        )
        logger.debug_once(
            f"Some attention backends are not valid for {cls.device_name} with "
            f"{config_str}. Reasons: {reasons_str}."
        )
        if len(valid_backends_priorities) == 0:
            raise ValueError(
                f"No valid attention backend found for {cls.device_name} "
                f"with {config_str}. Reasons: {reasons_str}."
            )

        # We have found some valid backends. Select the one with the
        # highest priority.
        logger.info(
            "Valid backends: %s", [b[0].name for b in valid_backends_priorities]
        )
        sorted_indices = sorted(
            range(len(valid_backends_priorities)),
            key=lambda i: valid_backends_priorities[i][1],
        )
        selected_index = sorted_indices[0]
        selected_backend = valid_backends_priorities[selected_index][0]
        logger.info(
            "Using %s backend.",
            selected_backend.name,
        )

        return selected_backend.get_path()

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return (
            "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator"  # noqa
        )

    @classmethod
    def supports_fp8(cls) -> bool:
        return cls.has_device_capability(89)

    @classmethod
    def use_custom_allreduce(cls) -> bool:
        return True

    @classmethod
    def opaque_attention_op(cls) -> bool:
        return True

    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        return "vllm.compilation.cuda_graph.CUDAGraphWrapper"

    @classmethod
    def device_count(cls) -> int:
        return cuda_device_count_stateless()

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype):
        if dtype == torch.bfloat16:  # noqa: SIM102
            if not cls.has_device_capability(80):
                capability = cls.get_device_capability()
                gpu_name = cls.get_device_name()

                if capability is None:
                    compute_str = "does not have a compute capability"
                else:
                    version_str = capability.as_version_str()
                    compute_str = f"has compute capability {version_str}"

                raise ValueError(
                    "Bfloat16 is only supported on GPUs "
                    "with compute capability of at least 8.0. "
                    f"Your {gpu_name} GPU {compute_str}. "
                    "You can use float16 instead by explicitly setting the "
                    "`dtype` flag in CLI, for example: --dtype=half."
                )

    @classmethod
    def insert_blocks_to_device(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        """Copy blocks from src_cache to dst_cache on GPU."""
        _src_cache = src_cache[:, src_block_indices]
        dst_cache[:, dst_block_indices] = _src_cache.to(dst_cache.device)

    @classmethod
    def swap_out_blocks_to_host(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        """Copy blocks from GPU to host (CPU)."""
        _src_cache = src_cache[:, src_block_indices]
        dst_cache[:, dst_block_indices] = _src_cache.cpu()

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        return True


# NVML utils
# Note that NVML is not affected by `CUDA_VISIBLE_DEVICES`,
# all the related functions work on real physical device ids.
# the major benefit of using NVML is that it will not initialize CUDA
class NvmlCudaPlatform(CudaPlatformBase):
    @classmethod
    @cache
    @with_nvml_context
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        try:
            physical_device_id = cls.device_id_to_physical_device_id(device_id)
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            return DeviceCapability(major=major, minor=minor)
        except RuntimeError:
            return None

    @classmethod
    @with_nvml_context
    def has_device_capability(
        cls,
        capability: tuple[int, int] | int,
        device_id: int = 0,
    ) -> bool:
        try:
            return super().has_device_capability(capability, device_id)
        except RuntimeError:
            return False

    @classmethod
    @with_nvml_context
    def get_device_name(cls, device_id: int = 0) -> str:
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        return cls._get_physical_device_name(physical_device_id)

    @classmethod
    @with_nvml_context
    def get_device_uuid(cls, device_id: int = 0) -> str:
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
        return pynvml.nvmlDeviceGetUUID(handle)

    @classmethod
    @with_nvml_context
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
        return int(pynvml.nvmlDeviceGetMemoryInfo(handle).total)

    @classmethod
    @with_nvml_context
    def is_fully_connected(cls, physical_device_ids: list[int]) -> bool:
        """
        query if the set of gpus are fully connected by nvlink (1 hop)
        """
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_ids]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i < j:
                    try:
                        p2p_status = pynvml.nvmlDeviceGetP2PStatus(
                            handle,
                            peer_handle,
                            pynvml.NVML_P2P_CAPS_INDEX_NVLINK,
                        )
                        if p2p_status != pynvml.NVML_P2P_STATUS_OK:
                            return False
                    except pynvml.NVMLError:
                        logger.exception(
                            "NVLink detection failed. This is normal if"
                            " your machine has no NVLink equipped."
                        )
                        return False
        return True

    @classmethod
    def _get_physical_device_name(cls, device_id: int = 0) -> str:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        return pynvml.nvmlDeviceGetName(handle)

    @classmethod
    @with_nvml_context
    def log_warnings(cls):
        device_ids: int = pynvml.nvmlDeviceGetCount()
        if device_ids > 1:
            device_names = [cls._get_physical_device_name(i) for i in range(device_ids)]
            if (
                len(set(device_names)) > 1
                and os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
            ):
                logger.warning(
                    "Detected different devices in the system: %s. Please"
                    " make sure to set `CUDA_DEVICE_ORDER=PCI_BUS_ID` to "
                    "avoid unexpected behavior.",
                    ", ".join(device_names),
                )

    @classmethod
    @with_nvml_context
    def set_cpu_affinity(cls, device_id: int) -> None:
        """
        Set CPU affinity for the current process based on GPU device ID.
        
        This binds the process to CPUs in the same NUMA node as the GPU.
        Can be disabled by setting VLLM_DISABLE_CPU_AFFINITY=1.
        
        Args:
            device_id: Logical CUDA device index (0-based relative to visible devices).
                      This will be mapped to the physical device ID internally.
        """
        # Allow disabling CPU affinity via environment variable
        if os.environ.get("VLLM_DISABLE_CPU_AFFINITY", "0") == "1":
            logger.info("CPU affinity setting is disabled via VLLM_DISABLE_CPU_AFFINITY")
            return
            
        try:
            import psutil
        except ImportError:
            logger.warning(
                "psutil is not available. Cannot set CPU affinity. "
                "Install psutil to enable NUMA affinity optimization."
            )
            return

        try:
            # Get current affinity before setting
            current_process = psutil.Process()
            original_affinity = current_process.cpu_affinity()
            
            # Read original affinity from /proc filesystem for verification
            original_affinity_from_proc = None
            try:
                with open(f"/proc/{current_process.pid}/status", "r") as f:
                    for line in f:
                        if line.startswith("Cpus_allowed_list:"):
                            original_affinity_from_proc = line.split(":", 1)[1].strip()
                            break
            except Exception:
                pass  # If we can't read it, just continue
            
            cpu_count = os.cpu_count()
            if cpu_count is None:
                logger.warning(
                    "Cannot determine CPU count. Skipping CPU affinity setting."
                )
                return

            cpu_set_size = (cpu_count + 63) // 64

            # device_id is a logical CUDA device index (0-based relative to CUDA_VISIBLE_DEVICES)
            # We need to map it to the physical device ID for NVML
            # NVML uses physical device IDs (PCI bus order), not affected by CUDA_VISIBLE_DEVICES
            physical_device_id = cls.device_id_to_physical_device_id(device_id)
            
            # Get CUDA device UUID to verify we're querying the right device
            cuda_uuid = torch.cuda.get_device_properties(device_id).uuid
            
            # Get NVML handle and verify it matches
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
            nvml_uuid = pynvml.nvmlDeviceGetUUID(handle)
            
            # Normalize UUIDs for comparison (NVML includes "GPU-" prefix, PyTorch doesn't)
            cuda_uuid_normalized = str(cuda_uuid).replace("GPU-", "")
            nvml_uuid_normalized = str(nvml_uuid).replace("GPU-", "")
            
            # Verify the mapping is correct
            if cuda_uuid_normalized != nvml_uuid_normalized:
                logger.warning(
                    "Device ID mapping mismatch: logical GPU %d has UUID %s, "
                    "but physical GPU %d has UUID %s. Skipping CPU affinity setting.",
                    device_id,
                    cuda_uuid,
                    physical_device_id,
                    nvml_uuid,
                )
                return

            # Get CPU affinity from NVML for this physical GPU
            cpu_affinity_mask = pynvml.nvmlDeviceGetCpuAffinity(handle, cpu_set_size)

            # Convert the bitmask to a list of CPU IDs
            cpu_ids = []
            for i, mask in enumerate(cpu_affinity_mask):
                for bit in range(64):
                    cpu_id = i * 64 + bit
                    if cpu_id >= cpu_count:
                        break
                    if mask & (1 << bit):
                        cpu_ids.append(cpu_id)

            if cpu_ids:
                # Set CPU affinity using psutil
                current_process.cpu_affinity(cpu_ids)
                
                # Verify the affinity was actually set by reading from /proc filesystem
                try:
                    with open(f"/proc/{current_process.pid}/status", "r") as f:
                        for line in f:
                            if line.startswith("Cpus_allowed_list:"):
                                actual_affinity = line.split(":", 1)[1].strip()
                                logger.info(
                                    "Set CPU affinity for process %d to CPUs %s for logical GPU %d (physical GPU %d). "
                                    "Restricted from %d cores to %d cores. "
                                    "/proc verification: before=%s, after=%s. "
                                    "Set VLLM_DISABLE_CPU_AFFINITY=1 to disable.",
                                    current_process.pid,
                                    cpu_ids,
                                    device_id,
                                    physical_device_id,
                                    len(original_affinity),
                                    len(cpu_ids),
                                    original_affinity_from_proc or "unknown",
                                    actual_affinity,
                                )
                                break
                except Exception as e:
                    # Fallback if we can't read /proc
                    logger.info(
                        "Set CPU affinity for process %d to CPUs %s for logical GPU %d (physical GPU %d). "
                        "Restricted from %d cores to %d cores. "
                        "/proc before=%s. Set VLLM_DISABLE_CPU_AFFINITY=1 to disable.",
                        current_process.pid,
                        cpu_ids,
                        device_id,
                        physical_device_id,
                        len(original_affinity),
                        len(cpu_ids),
                        original_affinity_from_proc or "unknown",
                    )
                
                # Migrate existing memory to the target NUMA node
                # Only migrate if we successfully set CPU affinity
                if os.environ.get("VLLM_MIGRATE_NUMA_MEMORY", "1") == "1":
                    import time
                    
                    try:
                        # Determine target NUMA node using NVML's memory affinity API
                        # This is the correct way to get the NUMA node for a GPU
                        memory_affinity_mask = pynvml.nvmlDeviceGetMemoryAffinity(
                            handle, cpu_set_size, pynvml.NVML_AFFINITY_SCOPE_NODE
                        )
                        
                        # Find the NUMA node from the memory affinity bitmask
                        target_numa_node = -1
                        for i, mask in enumerate(memory_affinity_mask):
                            if mask != 0:
                                # Find the first set bit in this group
                                for bit in range(64):
                                    if mask & (1 << bit):
                                        target_numa_node = i * 64 + bit
                                        break
                                if target_numa_node != -1:
                                    break
                        
                        if target_numa_node == -1:
                            logger.warning(
                                "Could not determine target NUMA node from nvmlDeviceGetMemoryAffinity for GPU %d",
                                physical_device_id,
                            )
                            # Fallback to heuristic
                            target_numa_node = min(cpu_ids) // (cpu_count // 2)
                            logger.info(
                                "Using fallback heuristic: target NUMA node %d for GPU %d",
                                target_numa_node,
                                physical_device_id,
                            )
                        else:
                            logger.debug(
                                "Determined target NUMA node %d from nvmlDeviceGetMemoryAffinity for GPU %d",
                                target_numa_node,
                                physical_device_id,
                            )
                        
                        # Determine source nodes (all nodes except target)
                        num_numa_nodes = (cpu_count + (cpu_count // 2) - 1) // (cpu_count // 2)
                        source_nodes = [n for n in range(num_numa_nodes) if n != target_numa_node]
                        
                        if source_nodes:
                            logger.info(
                                "Migrating memory for process %d from NUMA nodes %s to node %d",
                                current_process.pid,
                                source_nodes,
                                target_numa_node,
                            )
                            start_time = time.time()
                            
                            # Try libnuma-based migration first
                            success = _migrate_pages_via_libnuma(
                                current_process.pid, source_nodes, target_numa_node
                            )
                            
                            if not success:
                                # Fall back to subprocess-based migration
                                import subprocess
                                
                                logger.info(
                                    "Using subprocess-based memory migration for process %d (libnuma not available or failed)",
                                    current_process.pid,
                                )
                                
                                # Run migratepages for each source node
                                for source_node in source_nodes:
                                    try:
                                        result = subprocess.run(
                                            ["migratepages", str(current_process.pid), 
                                             str(source_node), str(target_numa_node)],
                                            capture_output=True,
                                            text=True,
                                            timeout=30,
                                        )
                                        if result.returncode != 0 and result.returncode != 1:
                                            # returncode 1 can mean "no pages to migrate", which is fine
                                            logger.warning(
                                                "migratepages from node %d to %d returned code %d: %s",
                                                source_node,
                                                target_numa_node,
                                                result.returncode,
                                                result.stderr.strip() if result.stderr else "",
                                            )
                                    except FileNotFoundError:
                                        logger.warning(
                                            "migratepages command not found. Install numactl package to enable memory migration."
                                        )
                                        break
                                    except subprocess.TimeoutExpired:
                                        logger.warning(
                                            "Memory migration timed out after 30 seconds for process %d from node %d",
                                            current_process.pid,
                                            source_node,
                                        )
                            
                            elapsed = time.time() - start_time
                            logger.info(
                                "Memory migration completed for process %d in %.2f seconds",
                                current_process.pid,
                                elapsed,
                            )
                            
                            # Set memory binding for future allocations to the target NUMA node
                            # This matches the behavior of the CPU backend (csrc/cpu/utils.cpp)
                            # which uses numa_set_membind() and numa_set_strict()
                            try:
                                # Try to use libnuma for memory binding (preferred method)
                                libnuma = CDLL("libnuma.so.1", use_errno=True)
                                
                                # Define Bitmask structure
                                class Bitmask(Structure):
                                    _fields_ = [("size", c_ulong), ("maskp", POINTER(c_ulong))]
                                
                                # Get function signatures
                                numa_allocate_nodemask = libnuma.numa_allocate_nodemask
                                numa_allocate_nodemask.restype = POINTER(Bitmask)
                                
                                numa_bitmask_setbit = libnuma.numa_bitmask_setbit
                                numa_bitmask_setbit.argtypes = [POINTER(Bitmask), c_uint]
                                numa_bitmask_setbit.restype = POINTER(Bitmask)
                                
                                numa_set_membind = libnuma.numa_set_membind
                                numa_set_membind.argtypes = [POINTER(Bitmask)]
                                numa_set_membind.restype = None
                                
                                numa_set_strict = libnuma.numa_set_strict
                                numa_set_strict.argtypes = [c_int]
                                numa_set_strict.restype = None
                                
                                # Get the free function (handle different API versions)
                                try:
                                    numa_free_fn = libnuma.numa_bitmask_free
                                except AttributeError:
                                    numa_free_fn = libnuma.numa_free_nodemask
                                
                                numa_free_fn.argtypes = [POINTER(Bitmask)]
                                numa_free_fn.restype = None
                                
                                # Create bitmask for the target node
                                membind_mask = numa_allocate_nodemask()
                                if not membind_mask:
                                    logger.warning(
                                        "Failed to allocate nodemask for membind setting"
                                    )
                                else:
                                    try:
                                        # Set bit for target node
                                        numa_bitmask_setbit(membind_mask, target_numa_node)
                                        
                                        # Set memory binding to restrict allocations to target node
                                        numa_set_membind(membind_mask)
                                        
                                        # Set strict mode (fail if target node is full)
                                        numa_set_strict(1)
                                        
                                        logger.info(
                                            "Set memory binding to NUMA node %d for process %d (strict mode)",
                                            target_numa_node,
                                            current_process.pid,
                                        )
                                    finally:
                                        numa_free_fn(membind_mask)
                                        
                            except Exception as e:
                                logger.debug(
                                    "Failed to set memory binding via libnuma for process %d: %s. "
                                    "Attempting set_mempolicy syscall fallback.",
                                    current_process.pid,
                                    str(e),
                                )
                                
                                # Fallback to set_mempolicy syscall if libnuma fails
                                try:
                                    from ctypes import c_long
                                    import platform
                                    
                                    # MPOL_BIND = 2 (strict binding to specific nodes)
                                    MPOL_BIND = 2
                                    
                                    # syscall numbers differ by architecture
                                    machine = platform.machine()
                                    if machine == 'aarch64':
                                        SYS_set_mempolicy = 237
                                    elif machine == 'x86_64':
                                        SYS_set_mempolicy = 238
                                    else:
                                        logger.debug(
                                            "Unknown architecture %s for set_mempolicy, skipping",
                                            machine,
                                        )
                                        raise Exception(f"Unsupported architecture: {machine}")
                                    
                                    # Load libc to access syscall
                                    libc = CDLL("libc.so.6", use_errno=True)
                                    syscall = libc.syscall
                                    syscall.restype = c_long
                                    
                                    # Create nodemask for the target node
                                    max_nodes = 64
                                    nodemask = (c_ulong * ((max_nodes + 63) // 64))()
                                    nodemask[target_numa_node // 64] = 1 << (target_numa_node % 64)
                                    
                                    # Call set_mempolicy with MPOL_BIND for strict binding
                                    result = syscall(
                                        SYS_set_mempolicy,
                                        MPOL_BIND,
                                        nodemask,
                                        max_nodes,
                                    )
                                    
                                    if result == 0:
                                        logger.info(
                                            "Set memory policy (MPOL_BIND) to NUMA node %d for process %d via syscall",
                                            target_numa_node,
                                            current_process.pid,
                                        )
                                    else:
                                        errno = get_errno()
                                        logger.warning(
                                            "set_mempolicy syscall failed for process %d: errno=%d",
                                            current_process.pid,
                                            errno,
                                        )
                                        
                                except Exception as e2:
                                    logger.debug(
                                        "Failed to set memory policy for process %d: %s",
                                        current_process.pid,
                                        str(e2),
                                    )
                    except Exception as e:
                        logger.warning(
                            "Failed to migrate memory for process %d: %s",
                            current_process.pid,
                            str(e),
                        )
            else:
                logger.warning(
                    "No CPU affinity information available for logical GPU %d (physical GPU %d)",
                    device_id,
                    physical_device_id,
                )

        except Exception as e:
            logger.warning(
                "Failed to set CPU affinity for logical GPU %d: %s", device_id, str(e)
            )


class NonNvmlCudaPlatform(CudaPlatformBase):
    @classmethod
    @cache
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.cuda.get_device_name(device_id)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.cuda.get_device_properties(device_id)
        return device_props.total_memory

    @classmethod
    def is_fully_connected(cls, physical_device_ids: list[int]) -> bool:
        logger.exception(
            "NVLink detection not possible, as context support was"
            " not found. Assuming no NVLink available."
        )
        return False

    @classmethod
    def set_cpu_affinity(cls, device_id: int) -> None:
        """
        Set CPU affinity for the current process based on GPU device ID.
        This is a no-op for NonNvmlCudaPlatform as NVML is not available.
        """
        pass


# Autodetect either NVML-enabled or non-NVML platform
# based on whether NVML is available.
nvml_available = False
try:
    try:
        pynvml.nvmlInit()
        nvml_available = True
    except Exception:
        # On Jetson, NVML is not supported.
        nvml_available = False
finally:
    if nvml_available:
        pynvml.nvmlShutdown()

CudaPlatform = NvmlCudaPlatform if nvml_available else NonNvmlCudaPlatform

CudaPlatform.log_warnings()
