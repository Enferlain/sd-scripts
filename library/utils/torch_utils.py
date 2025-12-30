import random
import torch
import logging
from typing import Optional, Tuple

from accelerate.utils import set_seed
from torch import nn as nn

from ..config.dataclasses.training import TrainingConfig
from ..config.dataclasses.performance import PrecisionConfig
from ..config.dataclasses.output import SavingConfig

from library.utils.common_utils import setup_logging
from library.utils.device_utils import init_ipex   # TODO: is it needed?

init_ipex()  # TODO: is it needed?

setup_logging()  # TODO: is it needed?
logger = logging.getLogger(__name__)


def prepare_dtype(precision_config: PrecisionConfig, saving_config: Optional[SavingConfig] = None) -> Tuple[torch.dtype, Optional[torch.dtype]]:  # TODO: why does this handle both saving and training related concerns?
    """
    Prepare weight and save dtypes based on configuration.
    
    Args:
        precision_config: Config containing mixed_precision setting
        saving_config: Optional config containing save_precision setting
        
    Returns:
        Tuple of (weight_dtype, save_dtype)
    """
    weight_dtype = torch.float32
    if precision_config.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif precision_config.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    save_dtype = None
    if saving_config is not None:
        if saving_config.save_precision == "fp16":
            save_dtype = torch.float16
        elif saving_config.save_precision == "bf16":
            save_dtype = torch.bfloat16
        elif saving_config.save_precision == "float":
            save_dtype = torch.float32

    return weight_dtype, save_dtype


def set_torch_cuda_reduced_precision(precision_config: PrecisionConfig):  # FIXME cfg performance
    """Set CUDA reduced precision operations based on performance config."""
    if precision_config.disable_cuda_reduced_precision_operations:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    elif precision_config.enable_cuda_reduced_precision_operations:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)


def set_seed_from_config(training_config: TrainingConfig):
    if training_config.seed is None or training_config.seed == -1:
        training_config.seed = random.randint(0, 2 ** 32)
        logger.info(f"As seed provided is -1, randomly selected {training_config.seed} as the seed for this training run.")
    set_seed(int(training_config.seed))


def match_mixed_precision(precision_config: PrecisionConfig, weight_dtype):
    """Match mixed precision settings, returning weight_dtype if full precision is enabled."""
    if precision_config.full_fp16:
        assert (
                weight_dtype == torch.float16
        ), "full_fp16 requires mixed precision='fp16'"
        return weight_dtype
    elif precision_config.full_bf16:
        assert (
                weight_dtype == torch.bfloat16
        ), "full_bf16 requires mixed precision='bf16'"
        return weight_dtype
    else:
        return None


def swap_weight_devices(layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    weight_swap_jobs = []
    for module_to_cpu, module_to_cuda in zip(layer_to_cpu.modules(), layer_to_cuda.modules()):
        if hasattr(module_to_cpu, "weight") and module_to_cpu.weight is not None:
            weight_swap_jobs.append(
                (module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))

    torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # cuda to cpu
        for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
            cuda_data_view.record_stream(stream)
            module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True)

        stream.synchronize()

        # cpu to cuda
        for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
            cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
            module_to_cuda.weight.data = cuda_data_view

    stream.synchronize()
    torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value


def weighs_to_device(layer: nn.Module, device: torch.device):
    for module in layer.modules():
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.data = module.weight.data.to(device, non_blocking=True)


def str_to_dtype(s: Optional[str], default_dtype: Optional[torch.dtype] = None) -> torch.dtype:
    """
    Convert a string to a torch.dtype

    Args:
        s: string representation of the dtype
        default_dtype: default dtype to return if s is None

    Returns:
        torch.dtype: the corresponding torch.dtype

    Raises:
        ValueError: if the dtype is not supported

    Examples:
        >>> str_to_dtype("float32")
        torch.float32
        >>> str_to_dtype("fp32")
        torch.float32
        >>> str_to_dtype("float16")
        torch.float16
        >>> str_to_dtype("fp16")
        torch.float16
        >>> str_to_dtype("bfloat16")
        torch.bfloat16
        >>> str_to_dtype("bf16")
        torch.bfloat16
        >>> str_to_dtype("fp8")
        torch.float8_e4m3fn
        >>> str_to_dtype("fp8_e4m3fn")
        torch.float8_e4m3fn
        >>> str_to_dtype("fp8_e4m3fnuz")
        torch.float8_e4m3fnuz
        >>> str_to_dtype("fp8_e5m2")
        torch.float8_e5m2
        >>> str_to_dtype("fp8_e5m2fnuz")
        torch.float8_e5m2fnuz
    """
    if s is None:
        return default_dtype
    if s in ["bf16", "bfloat16"]:
        return torch.bfloat16
    elif s in ["fp16", "float16"]:
        return torch.float16
    elif s in ["fp32", "float32", "float"]:
        return torch.float32
    elif s in ["fp8_e4m3fn", "e4m3fn", "float8_e4m3fn"]:
        return torch.float8_e4m3fn
    elif s in ["fp8_e4m3fnuz", "e4m3fnuz", "float8_e4m3fnuz"]:
        return torch.float8_e4m3fnuz
    elif s in ["fp8_e5m2", "e5m2", "float8_e5m2"]:
        return torch.float8_e5m2
    elif s in ["fp8_e5m2fnuz", "e5m2fnuz", "float8_e5m2fnuz"]:
        return torch.float8_e5m2fnuz
    elif s in ["fp8", "float8"]:
        return torch.float8_e4m3fn  # default fp8
    else:
        raise ValueError(f"Unsupported dtype: {s}")
