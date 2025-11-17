# common functions for training

import argparse
import time
import os
import random
import toml
import torch
import logging

from typing import Optional, Tuple
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed, TorchDynamoPlugin

import library.optimizations.deepspeed_utils as deepspeed_utils
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from library.train import custom_train_functions
from library.train.arguments import get_sanitized_config_or_none
from library.utils.common_utils import setup_logging
from library.utils.device_utils import init_ipex   # todo is it needed?

init_ipex()  # todo is it needed?

setup_logging()  # todo is it needed?
logger = logging.getLogger(__name__)


def prepare_accelerator(args: argparse.Namespace):
    """
    this function also prepares deepspeed plugin
    """

    if args.logging_dir is None:
        logging_dir = None
    else:
        log_prefix = "" if args.log_prefix is None else args.log_prefix
        logging_dir = args.logging_dir + "/" + log_prefix + time.strftime("%Y%m%d%H%M%S", time.localtime())

    if args.log_with is None:
        if logging_dir is not None:
            log_with = "tensorboard"
        else:
            log_with = None
    else:
        log_with = args.log_with
        if log_with in ["tensorboard", "all"]:
            if logging_dir is None:
                raise ValueError(
                    "logging_dir is required when log_with is tensorboard / Tensorboardを使う場合、logging_dirを指定してください"
                )
        if log_with in ["wandb", "all"]:
            try:
                import wandb
            except ImportError:
                raise ImportError("No wandb / wandb がインストールされていないようです")
            if logging_dir is not None:
                os.makedirs(logging_dir, exist_ok=True)
                os.environ["WANDB_DIR"] = logging_dir
            if args.wandb_api_key is not None:
                wandb.login(key=args.wandb_api_key)

    # torch.compile のオプション。 NO の場合は torch.compile は使わない
    # torch.compile のオプション。 NO の場合は torch.compile は使わない
    if args.torch_compile:
        # Configure the compilation backend
        dynamo_plugin = TorchDynamoPlugin(
            backend="inductor",  # Options: "inductor", "aot_eager", "aot_nvfuser", etc.
            mode="default",  # Options: "default", "reduce-overhead", "max-autotune"
            fullgraph=False,
            dynamic=True,
            use_regional_compilation=True,
        )
    else:
        dynamo_plugin = None

    #    (
    #        InitProcessGroupKwargs(
    #            backend="gloo" if os.name == "nt" or not torch.cuda.is_available() else "nccl",
    #            init_method=(
    #                "env://?use_libuv=False" if os.name == "nt" and Version(torch.__version__) >= Version("2.4.0") else None
    #            ),
    #            timeout=datetime.timedelta(minutes=args.ddp_timeout) if args.ddp_timeout else None,
    #        )
    #        if torch.cuda.device_count() > 1
    #        else None
    #    ),

    kwargs_handlers = [
        (
            DistributedDataParallelKwargs(
                gradient_as_bucket_view=args.ddp_gradient_as_bucket_view, static_graph=args.ddp_static_graph
            )
            if args.ddp_gradient_as_bucket_view or args.ddp_static_graph
            else None
        ),
    ]
    kwargs_handlers = [i for i in kwargs_handlers if i is not None]
    deepspeed_plugin = deepspeed_utils.prepare_deepspeed_plugin(args)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_with,
        project_dir=logging_dir,
        kwargs_handlers=kwargs_handlers,
        dynamo_plugin=dynamo_plugin,
        deepspeed_plugin=deepspeed_plugin,
    )
    return accelerator


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


def get_hidden_states(args: argparse.Namespace, input_ids, tokenizer, text_encoder, weight_dtype=None):
    # with no_token_padding, the length is not max length, return result immediately
    if input_ids.size()[-1] != tokenizer.model_max_length:
        return text_encoder(input_ids)[0]

    # input_ids: b,n,77
    b_size = input_ids.size()[0]
    input_ids = input_ids.reshape((-1, tokenizer.model_max_length))  # batch_size*3, 77

    if args.clip_skip is None:
        encoder_hidden_states = text_encoder(input_ids)[0]
    else:
        enc_out = text_encoder(input_ids, output_hidden_states=True, return_dict=True)
        encoder_hidden_states = enc_out["hidden_states"][-args.clip_skip]
        encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states)

    # bs*3, 77, 768 or 1024
    encoder_hidden_states = encoder_hidden_states.reshape((b_size, -1, encoder_hidden_states.shape[-1]))

    if args.max_token_length is not None:
        if args.v2:
            # v2: <BOS>...<EOS> <PAD> ... の三連を <BOS>...<EOS> <PAD> ... へ戻す　正直この実装でいいのかわからん
            states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]  # <BOS>
            for i in range(1, args.max_token_length, tokenizer.model_max_length):
                chunk = encoder_hidden_states[:, i: i + tokenizer.model_max_length - 2]  # <BOS> の後から 最後の前まで
                if i > 0:
                    for j in range(len(chunk)):
                        if input_ids[j, 1] == tokenizer.eos_token:  # 空、つまり <BOS> <EOS> <PAD> ...のパターン
                            chunk[j, 0] = chunk[j, 1]  # 次の <PAD> の値をコピーする
                states_list.append(chunk)  # <BOS> の後から <EOS> の前まで
            states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))  # <EOS> か <PAD> のどちらか
            encoder_hidden_states = torch.cat(states_list, dim=1)
        else:
            # v1: <BOS>...<EOS> の三連を <BOS>...<EOS> へ戻す
            states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]  # <BOS>
            for i in range(1, args.max_token_length, tokenizer.model_max_length):
                states_list.append(
                    encoder_hidden_states[:, i: i + tokenizer.model_max_length - 2]
                )  # <BOS> の後から <EOS> の前まで
            states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))  # <EOS>
            encoder_hidden_states = torch.cat(states_list, dim=1)

    if weight_dtype is not None:
        # this is required for additional network training
        encoder_hidden_states = encoder_hidden_states.to(weight_dtype)

    return encoder_hidden_states


def pool_workaround(
        text_encoder: CLIPTextModelWithProjection, last_hidden_state: torch.Tensor, input_ids: torch.Tensor,
        eos_token_id: int
):
    r"""
    workaround for CLIP's pooling bug: it returns the hidden states for the max token id as the pooled output
    instead of the hidden states for the EOS token
    If we use Textual Inversion, we need to use the hidden states for the EOS token as the pooled output

    Original code from CLIP's pooling function:

    \# text_embeds.shape = [batch_size, sequence_length, transformer.width]
    \# take features from the eot embedding (eot_token is the highest number in each sequence)
    \# casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
    pooled_output = last_hidden_state[
        torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
        input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
    ]
    """

    # input_ids: b*n,77
    # find index for EOS token

    # Following code is not working if one of the input_ids has multiple EOS tokens (very odd case)
    # eos_token_index = torch.where(input_ids == eos_token_id)[1]
    # eos_token_index = eos_token_index.to(device=last_hidden_state.device)

    # Create a mask where the EOS tokens are
    eos_token_mask = (input_ids == eos_token_id).int()

    # Use argmax to find the last index of the EOS token for each element in the batch
    eos_token_index = torch.argmax(eos_token_mask, dim=1)  # this will be 0 if there is no EOS token, it's fine
    eos_token_index = eos_token_index.to(device=last_hidden_state.device)

    # get hidden states for EOS token
    pooled_output = last_hidden_state[
        torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), eos_token_index]

    # apply projection: projection may be of different dtype than last_hidden_state
    pooled_output = text_encoder.text_projection(pooled_output.to(text_encoder.text_projection.weight.dtype))
    pooled_output = pooled_output.to(last_hidden_state.dtype)

    return pooled_output


def get_hidden_states_sdxl(
        max_token_length: int,
        input_ids1: torch.Tensor,
        input_ids2: torch.Tensor,
        tokenizer1: CLIPTokenizer,
        tokenizer2: CLIPTokenizer,
        text_encoder1: CLIPTextModel,
        text_encoder2: CLIPTextModelWithProjection,
        weight_dtype: Optional[str] = None,
        accelerator: Optional[Accelerator] = None,
):
    # input_ids: b,n,77 -> b*n, 77
    b_size = input_ids1.size()[0]
    input_ids1 = input_ids1.reshape((-1, tokenizer1.model_max_length))  # batch_size*n, 77
    input_ids2 = input_ids2.reshape((-1, tokenizer2.model_max_length))  # batch_size*n, 77

    # text_encoder1
    enc_out = text_encoder1(input_ids1, output_hidden_states=True, return_dict=True)
    hidden_states1 = enc_out["hidden_states"][11]

    # text_encoder2
    enc_out = text_encoder2(input_ids2, output_hidden_states=True, return_dict=True)
    hidden_states2 = enc_out["hidden_states"][-2]  # penuultimate layer

    # pool2 = enc_out["text_embeds"]
    unwrapped_text_encoder2 = text_encoder2 if accelerator is None else accelerator.unwrap_model(text_encoder2)
    pool2 = pool_workaround(unwrapped_text_encoder2, enc_out["last_hidden_state"], input_ids2, tokenizer2.eos_token_id)

    # b*n, 77, 768 or 1280 -> b, n*77, 768 or 1280
    n_size = 1 if max_token_length is None else max_token_length // 75
    hidden_states1 = hidden_states1.reshape((b_size, -1, hidden_states1.shape[-1]))
    hidden_states2 = hidden_states2.reshape((b_size, -1, hidden_states2.shape[-1]))

    if max_token_length is not None:
        # bs*3, 77, 768 or 1024
        # encoder1: <BOS>...<EOS> の三連を <BOS>...<EOS> へ戻す
        states_list = [hidden_states1[:, 0].unsqueeze(1)]  # <BOS>
        for i in range(1, max_token_length, tokenizer1.model_max_length):
            states_list.append(hidden_states1[:, i: i + tokenizer1.model_max_length - 2])  # <BOS> の後から <EOS> の前まで
        states_list.append(hidden_states1[:, -1].unsqueeze(1))  # <EOS>
        hidden_states1 = torch.cat(states_list, dim=1)

        # v2: <BOS>...<EOS> <PAD> ... の三連を <BOS>...<EOS> <PAD> ... へ戻す　正直この実装でいいのかわからん
        states_list = [hidden_states2[:, 0].unsqueeze(1)]  # <BOS>
        for i in range(1, max_token_length, tokenizer2.model_max_length):
            chunk = hidden_states2[:, i: i + tokenizer2.model_max_length - 2]  # <BOS> の後から 最後の前まで
            # this causes an error:
            # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
            # if i > 1:
            #     for j in range(len(chunk)):  # batch_size
            #         if input_ids2[n_index + j * n_size, 1] == tokenizer2.eos_token_id:  # 空、つまり <BOS> <EOS> <PAD> ...のパターン
            #             chunk[j, 0] = chunk[j, 1]  # 次の <PAD> の値をコピーする
            states_list.append(chunk)  # <BOS> の後から <EOS> の前まで
        states_list.append(hidden_states2[:, -1].unsqueeze(1))  # <EOS> か <PAD> のどちらか
        hidden_states2 = torch.cat(states_list, dim=1)

        # pool はnの最初のものを使う
        pool2 = pool2[::n_size]

    if weight_dtype is not None:
        # this is required for additional network training
        hidden_states1 = hidden_states1.to(weight_dtype)
        hidden_states2 = hidden_states2.to(weight_dtype)

    return hidden_states1, hidden_states2, pool2


def get_timesteps(min_timestep: int, max_timestep: int, b_size: int, device: torch.device) -> torch.Tensor:
    if min_timestep < max_timestep:
        timesteps = torch.randint(min_timestep, max_timestep, (b_size,), device="cpu")
    else:
        timesteps = torch.full((b_size,), max_timestep, device="cpu")
    timesteps = timesteps.long().to(device)
    return timesteps


def get_noise_noisy_latents_and_timesteps(
    args, noise_scheduler, latents: torch.FloatTensor, fixed_timesteps=None, is_train=True,
    min_timestep_override=None, max_timestep_override=None
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.IntTensor]:
    """
    todo
    """
    # --- 1. Determine Timestep Range ---
    # This part handles the dynamic timestep schedule!
    if min_timestep_override is not None:
        min_timestep = min_timestep_override
    else:
        min_timestep = 0 if args.min_timestep is None else args.min_timestep

    if max_timestep_override is not None:
        max_timestep = max_timestep_override
    else:
        max_timestep = noise_scheduler.config.num_train_timesteps if args.max_timestep is None else args.max_timestep

    # --- 2. Generate Base Noise ---
    noise = torch.randn_like(latents, device=latents.device)
    if args.noise_offset and is_train:
        noise_offset = torch.rand(1, device=latents.device) * args.noise_offset if args.noise_offset_random_strength else args.noise_offset
        noise = custom_train_functions.apply_noise_offset(latents, noise, noise_offset, args.adaptive_noise_scale)

    b_size = latents.shape[0]

    # --- 3. TIMESTEP SAMPLING ---
    if fixed_timesteps is not None:
        timesteps = fixed_timesteps
    elif is_train and hasattr(noise_scheduler, "edm2_laplace_weights"):
        timesteps = torch.multinomial(
            noise_scheduler.edm2_laplace_weights,
            num_samples=b_size,
            replacement=True
        ).to(dtype=torch.long, device=latents.device)
    elif is_train and hasattr(noise_scheduler, "laplace_weights"):
        timesteps = torch.multinomial(
            noise_scheduler.laplace_weights,
            num_samples=b_size,
            replacement=True
        ).to(dtype=torch.long, device=latents.device)
    elif is_train and args.timestep_sampling == "mix_adaptive":
        # The main script is now responsible for creating the sampler.
        # We just check that it exists and use it.
        if not hasattr(args, "la_sampler") or args.la_sampler is None:
            raise ValueError(
                "timestep_sampling is 'mix_adaptive' but args.la_sampler is not initialized. "
                "Please ensure the sampler is created in your main training script."
            )

        # Sample discrete indices in [0, T_range)
        # Note: The sampler should be initialized with the full range of timesteps (e.g., 1000)
        t_local = args.la_sampler.sample(
            b_size,
            latents.device,
            getattr(args, "global_step", 0),
            getattr(args, "max_train_steps", 1000),
            sigmoid_scale=getattr(args, "sigmoid_scale", 1.0),
            discrete_flow_shift=getattr(args, "discrete_flow_shift", 1.0),
        )

        # Map local [0, T_total) to the absolute training range [min_timestep, max_timestep)
        timesteps = t_local.clamp(min_timestep, max_timestep - 1).to(dtype=torch.long, device=latents.device)
    elif is_train and args.timestep_sampling != "uniform":
        shift = args.discrete_flow_shift
        logits_norm = torch.randn(b_size, device="cpu")
        logits_norm = logits_norm * args.sigmoid_scale
        timesteps = logits_norm.sigmoid()
        timesteps = (timesteps * shift) / (1 + (shift - 1) * timesteps)
        timesteps = min_timestep + (timesteps * (max_timestep - min_timestep)).to(dtype=torch.long,
                                                                                  device=latents.device)
    else:
        # Fallback to default (random) sampling
        timesteps = get_timesteps(min_timestep, max_timestep, b_size, latents.device)
    
    # --- 4. Advanced Noise Application (multires, ip_noise_gamma) ---
    if args.multires_noise_iterations and is_train:
        noise = custom_train_functions.pyramid_noise_like(
            noise, latents.device, args.multires_noise_iterations, args.multires_noise_discount
        )

    if args.ip_noise_gamma and is_train:
        strength = torch.rand(1, device=latents.device) * args.ip_noise_gamma if args.ip_noise_gamma_random_strength else args.ip_noise_gamma
        noisy_latents = noise_scheduler.add_noise(latents, noise + strength * torch.randn_like(latents), timesteps)
    else:
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    
    # Important! The old script had a .cpu() call here. It was a workaround.
    # Modern diffusers handles device placement better, so we can often omit this.
    # If you see device errors, we can add it back!
    # noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.cpu()

    return noise, noisy_latents, timesteps


def calculate_val_loss_check(args, global_step, epoch_step, val_dataloader, train_dataloader) -> bool:
    if val_dataloader is None:
        return False

    if global_step != 0 and global_step < args.max_train_steps:
        if args.validation_every_n_step is not None:
            if global_step % int(args.validation_every_n_step) != 0:
                return False
        else:
            if epoch_step != len(train_dataloader) - 1:
                return False
    return True


def append_lr_to_logs(logs, lr_scheduler, optimizer_type, including_unet=True):
    names = []
    if including_unet:
        names.append("unet")
    names.append("text_encoder1")
    names.append("text_encoder2")

    append_lr_to_logs_with_names(logs, lr_scheduler, optimizer_type, names)


def append_lr_to_logs_with_names(logs, lr_scheduler, optimizer_type, names):
    lrs = lr_scheduler.get_last_lr()

    for lr_index in range(len(lrs)):
        name = names[lr_index]
        logs["lr/" + name] = float(lrs[lr_index])

        if optimizer_type.lower().startswith("DAdapt".lower()) or optimizer_type.lower() == "Prodigy".lower():
            logs["lr/d*lr/" + name] = (
                    lr_scheduler.optimizers[-1].param_groups[lr_index]["d"] *
                    lr_scheduler.optimizers[-1].param_groups[lr_index]["lr"]
            )


def init_trackers(accelerator: Accelerator, args: argparse.Namespace, default_tracker_name: str):
    """
    Initialize experiment trackers with tracker specific behaviors
    """
    if accelerator.is_main_process:
        init_kwargs = {}
        if args.wandb_run_name:
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers(
            default_tracker_name if args.log_tracker_name is None else args.log_tracker_name,
            config=get_sanitized_config_or_none(args),
            init_kwargs=init_kwargs,
        )


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


def determine_grad_sync_context(args, accelerator, sync_gradients, training_model, edm2_model=None):
    # TODO
    # if args.full_bf16:
    #    if not sync_gradients and accelerator.num_processes > 1:
    #        if edm2_model is not None:
    #            return accelerator.no_sync(training_model, edm2_model)
    #        else:
    #            return accelerator.no_sync(training_model)
    #    else:
    #        return contextlib.nullcontext()
    # else:
    if edm2_model is not None:
        return accelerator.accumulate(training_model, edm2_model)
    else:
        return accelerator.accumulate(training_model)


def args_set_seed(args):
    if args.seed is None or args.seed == -1:
        args.seed = random.randint(0, 2 ** 32)
        logger.info(f"As seed provided is -1, randomly selected {args.seed} as the seed for this training run.")
    set_seed(int(args.seed))
