import random
import torch
import logging

from accelerate.utils import set_seed
from torch import nn as nn

from library.config.dataclasses.training import TrainingConfig
from library.config.dataclasses.performance import PrecisionConfig
from library.config.dataclasses.output import SavingConfig

# Initialize logging before module-level logger
logger = logging.getLogger(__name__)


def prepare_dtype(
    precision_config: PrecisionConfig, saving_config: SavingConfig | None = None
) -> tuple[torch.dtype, torch.dtype | None]:  # Returns (weight_dtype, save_dtype) for convenience
    """
    Prepare weight and save dtypes based on configuration.

    Args:
        precision_config (PrecisionConfig): Config containing mixed_precision setting.
        saving_config (Optional[SavingConfig], optional): Optional config containing save_precision setting. Defaults to None.

    Returns:
        Tuple[torch.dtype, Optional[torch.dtype]]: Tuple of (weight_dtype, save_dtype).
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


def set_torch_cuda_reduced_precision(precision_config: PrecisionConfig):
    """
    Set CUDA reduced precision operations based on performance config.

    Args:
        precision_config (PrecisionConfig): The configuration for precision settings.
    """
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
    """
    Set the random seed from the training configuration.

    If the seed is -1 or None, a random seed is generated and logged.

    Args:
        training_config (TrainingConfig): The training configuration object containing the seed.
    """
    if training_config.seed is None or training_config.seed == -1:
        training_config.seed = random.randint(0, 2**32)
        logger.info(f"As seed provided is -1, randomly selected {training_config.seed} as the seed for this training run.")
    set_seed(int(training_config.seed))


def match_mixed_precision(precision_config: PrecisionConfig, weight_dtype):
    """
    Match mixed precision settings, returning weight_dtype if full fp16/bf16 mode is enabled.

    Args:
        precision_config (PrecisionConfig): The configuration for precision settings.
        weight_dtype (torch.dtype): The weight data type.

    Returns:
        Optional[torch.dtype]: The weight data type if full_fp16 or full_bf16 is enabled, otherwise None.
    """
    if precision_config.full_fp16:
        assert weight_dtype == torch.float16, "full_fp16 requires mixed precision='fp16'"
        return weight_dtype
    elif precision_config.full_bf16:
        assert weight_dtype == torch.bfloat16, "full_bf16 requires mixed precision='bf16'"
        return weight_dtype
    else:
        return None


def weights_to_device(layer: nn.Module, device: torch.device):
    """
    Move the weights of a layer to the specified device.

    Args:
        layer (nn.Module): The layer to move weights for.
        device (torch.device): The target device.
    """
    for module in layer.modules():
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.data = module.weight.data.to(device, non_blocking=True)  # type: ignore[union-attr]


def weighs_to_device(layer: nn.Module, device: torch.device):
    """
    Deprecated alias for weights_to_device.
    """
    logger.warning("weighs_to_device is deprecated and will be removed in a future version. Use weights_to_device instead.")
    weights_to_device(layer, device)


def str_to_dtype(s: str | None, default_dtype: torch.dtype | None = None) -> torch.dtype | None:
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
