from dataclasses import dataclass
import random
import torch
import logging

from accelerate.utils import set_seed
from ..config.dataclasses.training import TrainingConfig

from library.utils.common_utils import setup_logging
from library.utils.device_utils import init_ipex   # todo is it needed?

init_ipex()  # todo is it needed?

setup_logging()  # todo is it needed?
logger = logging.getLogger(__name__)


def prepare_dtype(cfg: TrainingConfig):
    weight_dtype = torch.float32
    if cfg.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif cfg.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    save_dtype = None
    if cfg.save_precision == "fp16":
        save_dtype = torch.float16
    elif cfg.save_precision == "bf16":
        save_dtype = torch.bfloat16
    elif cfg.save_precision == "float":
        save_dtype = torch.float32

    return weight_dtype, save_dtype


def set_torch_cuda_reduced_precision(cfg: TrainingConfig):
    if cfg.disable_cuda_reduced_precision_operations:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    elif cfg.enable_cuda_reduced_precision_operations:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)


def args_set_seed(cfg: TrainingConfig):
    if cfg.seed is None or cfg.seed == -1:
        cfg.seed = random.randint(0, 2 ** 32)
        logger.info(f"As seed provided is -1, randomly selected {cfg.seed} as the seed for this training run.")
    set_seed(int(cfg.seed))


def match_mixed_precision(cfg: TrainingConfig, weight_dtype):
    if cfg.full_fp16:
        assert (
                weight_dtype == torch.float16
        ), "full_fp16 requires mixed precision='fp16'"
        return weight_dtype
    elif cfg.full_bf16:
        assert (
                weight_dtype == torch.bfloat16
        ), "full_bf16 requires mixed precision='bf16'"
        return weight_dtype
    else:
        return None
