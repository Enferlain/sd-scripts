import argparse
import random
import torch
import logging

from accelerate.utils import set_seed

from library.utils.common_utils import setup_logging
from library.utils.device_utils import init_ipex   # todo is it needed?

init_ipex()  # todo is it needed?

setup_logging()  # todo is it needed?
logger = logging.getLogger(__name__)


def prepare_dtype(args: argparse.Namespace):
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    save_dtype = None
    if args.save_precision == "fp16":
        save_dtype = torch.float16
    elif args.save_precision == "bf16":
        save_dtype = torch.bfloat16
    elif args.save_precision == "float":
        save_dtype = torch.float32

    return weight_dtype, save_dtype


def set_torch_cuda_reduced_precision(args):
    if args.disable_cuda_reduced_precision_operations:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    elif args.enable_cuda_reduced_precision_operations:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)


def args_set_seed(args):
    if args.seed is None or args.seed == -1:
        args.seed = random.randint(0, 2 ** 32)
        logger.info(f"As seed provided is -1, randomly selected {args.seed} as the seed for this training run.")
    set_seed(int(args.seed))


def match_mixed_precision(args, weight_dtype):
    if args.full_fp16:
        assert (
                weight_dtype == torch.float16
        ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
        return weight_dtype
    elif args.full_bf16:
        assert (
                weight_dtype == torch.bfloat16
        ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
        return weight_dtype
    else:
        return None
