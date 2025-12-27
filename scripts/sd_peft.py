"""
SD1.5/2 PEFT (LoRA/LyCORIS) Training Script

This script contains the complete training loop for SD1.5/2 PEFT training.
The training logic was previously in library/training/peft_trainer.py.

Structure:
- train() function: The main training loop
- Hydra entry point: Loads config and calls train()

Model-specific operations are delegated to:
- library/strategies/peft_strategy_sd.py (strategy pattern)
- library/training/peft_common.py (shared utilities)
"""

import gc
import importlib
import math
import os
import typing
import subprocess
import sys
import random
import time
import json
import numpy as np
import ast
import itertools
import atexit
import torch
import torch.nn as nn
import logging
import hydra

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from typing import Any, List, Union, Optional
from multiprocessing import Value
from tqdm import tqdm
from accelerate import Accelerator
from diffusers import DDPMScheduler
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from ramtorch.helpers import replace_linear_with_ramtorch

import library.config.config_util as config_util
import library.utils.huggingface_util as huggingface_util

from library.config.validation import prepare_config, validate_config, validate_sd_peft
from library.constants import SS_METADATA_MINIMUM_KEYS
from library.strategies import strategy_sd, strategy_base
from library.performance import deepspeed_utils
from library.models import model_util
from library.utils import sai_model_spec
from library.utils.common_utils import setup_logging
from library.utils.device_utils import init_ipex, clean_memory_on_device
from library.utils.torch_utils import set_torch_cuda_reduced_precision, set_seed_from_config, prepare_dtype
from library.training.diffusion import get_noise_noisy_latents_and_timesteps
from library.training.model_prep import replace_unet_modules, patch_accelerator_for_fp16_training
from library.training.sd_model_prep import load_target_model
from library.training.optimizer import prepare_optimizer, get_scheduler_fix
from library.training.sample_generation import sample_images_check
from library.training.sd_sample_generation import sample_images
from library.losses.loss import get_huber_threshold_if_needed, conditional_loss, EMARecorder
from library.config.dataclasses.sd_peft import SDPeftConfig
from library.strategies.peft_strategy_sd import SdPeftStrategy
from library.training.peft_common import (
    generate_step_logs,
    step_logging,
    save_timestep_distribution_plot,
    init_timestep_sampler,
    create_training_metadata,
    setup_live_plotter,
    prepare_datasets,
    calculate_initial_step,
    parse_dynamic_timestep_schedule,
    register_adapter_state_hooks,
    resolve_adapter_kwargs,
)

from library.training.checkpointing import (
    resume_from_local_or_hf_if_specified,
    get_git_revision_hash,
    model_hash, calculate_sha256,
    get_step_ckpt_name,
    save_and_remove_state_stepwise, get_remove_step_no,
    get_epoch_ckpt_name,
    get_remove_epoch_no,
    save_and_remove_state_on_epoch_end,
    get_last_ckpt_name,
    save_state_on_train_end
)

from library.data.dataset import (
    DatasetGroup,
    MinimalDataset,
    load_arbitrary_dataset,
    collator_class,
    debug_dataset,
    DreamBoothDataset
)

from library.training.trainer_utils import (
    calculate_val_loss_check,
    prepare_accelerator,
    init_trackers,
    determine_grad_sync_context
)

from library.training.noise_utils import (
    prepare_scheduler_for_custom_training,
    fix_noise_scheduler_betas_for_zero_terminal_snr
)

from library.losses.edm2_loss_utils import (
    prepare_edm2_loss_weighting,
    plot_edm2_loss_weighting_check,
    plot_edm2_loss_weighting
)

from library.losses.loss_weighting import (
    apply_masked_loss, apply_snr_weight,
    scale_v_prediction_loss_like_noise_prediction,
    add_v_prediction_like_loss,
    apply_debiased_estimation
)

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

init_ipex()

setup_logging()
logger = logging.getLogger(__name__)


def train(cfg: SDPeftConfig, strategies: "SdPeftStrategy"):
    strategies.la_sampler = None

    session_id = random.randint(0, 2 ** 32)
    training_started_at = time.time()

    set_torch_cuda_reduced_precision(cfg.performance)
    deepspeed_utils.prepare_deepspeed_config(cfg.performance, cfg.training)
    setup_logging(cfg.output.logging, reset=True)

    cache_latents = cfg.dataset.cache_latents
    use_dreambooth_method = cfg.dataset.in_json is None
    use_user_config = cfg.dataset.dataset_config is not None

    set_seed_from_config(cfg.training)

    tokenize_strategy = strategies.get_tokenize_strategy(cfg)
    strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)
    tokenizers = strategies.get_tokenizers(tokenize_strategy)  # will be removed after sample_image is refactored

    # prepare caching strategy: this must be set before preparing dataset. because dataset may use this strategy for initialization.
    latents_caching_strategy = strategies.get_latents_caching_strategy(cfg)
    strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

    # Prepare datasets
    dataset_result = prepare_datasets(cfg, strategies)
    if dataset_result is None:
        return  # debug_dataset mode or no data found
    train_dataset_group, val_dataset_group, collator, current_epoch, current_step = dataset_result

    # acceleratorを準備する
    logger.info("preparing accelerator")
    accelerator = prepare_accelerator(cfg.performance, cfg.output.logging, cfg.training)
    is_main_process = accelerator.is_main_process

    # mixed precisionに対応した型を用意しておき適宜castする
    weight_dtype, save_dtype = prepare_dtype(cfg.performance, cfg.output.saving)
    vae_dtype = (torch.float32 if cfg.performance.precision.no_half_vae else weight_dtype) if strategies.cast_vae(cfg) else None

    # load target models: unet may be None for lazy loading
    model_version, text_encoder, vae, unet = strategies.load_target_model(cfg, weight_dtype, accelerator)

    if vae_dtype is None:
        vae_dtype = vae.dtype
        logger.info(f"vae_dtype is set to {vae_dtype} by the model since cast_vae() is false")

    # text_encoder is List[CLIPTextModel] or CLIPTextModel
    text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]

    # prepare dataset for latents caching if needed
    if cache_latents:
        vae.to(accelerator.device, dtype=vae_dtype)
        vae.requires_grad_(False)
        vae.eval()

        train_dataset_group.new_cache_latents(vae, accelerator)
        if val_dataset_group is not None:
            val_dataset_group.new_cache_latents(vae, accelerator)

        vae.to("cpu")
        clean_memory_on_device(accelerator.device)

        accelerator.wait_for_everyone()

    # 必要ならテキストエンコーダーの出力をキャッシュする: Text Encoderはcpuまたはgpuへ移される
    # cache text encoder outputs if needed: Text Encoder is moved to cpu or gpu
    text_encoding_strategy = strategies.get_text_encoding_strategy(cfg)
    strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

    text_encoder_outputs_caching_strategy = strategies.get_text_encoder_outputs_caching_strategy(cfg)
    if text_encoder_outputs_caching_strategy is not None:
        strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(text_encoder_outputs_caching_strategy)
    strategies.cache_text_encoder_outputs_if_needed(cfg, accelerator, unet, vae, text_encoders, train_dataset_group,
                                              weight_dtype)
    if val_dataset_group is not None:
        strategies.cache_text_encoder_outputs_if_needed(cfg, accelerator, unet, vae, text_encoders, val_dataset_group,
                                                  weight_dtype)

    if unet is None:
        # lazy load unet if needed. text encoders may be freed or replaced with dummy models for saving memory
        unet, text_encoders = strategies.load_unet_lazily(cfg, weight_dtype, accelerator, text_encoders)

    # 差分追加学習のためにモデルを読み込む
    sys.path.append(os.path.dirname(__file__))
    accelerator.print("import peft module:", cfg.peft.module)
    adapter_module = importlib.import_module(cfg.peft.module)

    if cfg.peft.base_weights is not None:
        # base_weights が指定されている場合は、指定された重みを読み込みマージする
        for i, weight_path in enumerate(cfg.peft.base_weights):
            if cfg.peft.base_weights_multiplier is None or len(cfg.peft.base_weights_multiplier) <= i:
                multiplier = 1.0
            else:
                multiplier = cfg.peft.base_weights_multiplier[i]

            accelerator.print(f"merging module: {weight_path} with multiplier {multiplier}")

            module, weights_sd = adapter_module.create_adapter_from_weights(
                multiplier, weight_path, vae, text_encoder, unet, for_inference=True
            )
            module.merge_to(text_encoder, unet, weights_sd, weight_dtype,
                            accelerator.device if cfg.performance.memory.lowram else "cpu")

        accelerator.print(f"all weights merged: {', '.join(cfg.peft.base_weights)}")

    # prepare peft
    net_kwargs = {}
    if cfg.peft.args is not None:
        for net_arg in cfg.peft.args:
            key, value = net_arg.split("=", 1)
            net_kwargs[key] = value

    # Schema 1: Resolve explicit LoRA fields from config to kwargs
    resolve_adapter_kwargs(cfg.peft, net_kwargs)

    # if a new adapter is added in the future, add if ~ then blocks for each adapter (;'∀')
    if cfg.peft.adapter_rank_from_weights:
        adapter, _ = adapter_module.create_adapter_from_weights(1, cfg.peft.weights, vae, text_encoder, unet,
                                                                **net_kwargs)
    else:
        if "dropout" not in net_kwargs:
            # workaround for LyCORIS (;^ω^)
            net_kwargs["dropout"] = cfg.peft.neuron_dropout

        adapter = adapter_module.create_adapter(
            1.0,
            cfg.peft.adapter_rank,
            cfg.peft.adapter_alpha,
            vae,
            text_encoder,
            unet,
            neuron_dropout=cfg.peft.neuron_dropout,
            **net_kwargs,
        )
    if adapter is None:
        return
    adapter_has_multiplier = hasattr(adapter, "set_multiplier")

    # TODO remove `hasattr` by setting up methods if not defined in the peft like below  (hacky but will work):
    # if not hasattr(peft, "prepare_adapter"):
    #    peft.prepare_adapter = lambda args: None

    if hasattr(adapter, "prepare_adapter"):
        adapter.prepare_adapter(cfg)
    if cfg.peft.scale_weight_norms and not hasattr(adapter, "apply_max_norm_regularization"):
        logger.warning(
            "warning: scale_weight_norms is specified but the peft does not support it / scale_weight_normsが指定されていますが、ネットワークが対応していません"
        )
        cfg.peft.scale_weight_norms = False

    strategies.post_process_adapter(cfg, accelerator, adapter, text_encoders, unet)

    # apply peft to unet and text_encoder
    train_unet = strategies.is_train_unet(cfg)
    train_text_encoder = strategies.is_train_text_encoder(cfg)
    adapter.apply_to(text_encoder, unet, train_text_encoder, train_unet)

    if cfg.peft.weights is not None:
        # FIXME consider alpha of weights: this assumes that the alpha is not changed
        info = adapter.load_weights(cfg.peft.weights)
        accelerator.print(f"load peft weights from {cfg.peft.weights}: {info}")

    # if args.use_ramtorch:
    #     logger.info("Applying RamTorch to peft/lora.")
    #     if isinstance(peft, torch.nn.Module):
    #         peft = replace_linear_with_ramtorch(peft, accelerator.device)
    #         logger.info("RamTorch applied to peft/lora.")

    if cfg.performance.memory.gradient_checkpointing:
        if cfg.performance.memory.cpu_offload_checkpointing:
            unet.enable_gradient_checkpointing(cpu_offload=True)
        else:
            unet.enable_gradient_checkpointing()

        for t_enc, flag in zip(text_encoders, strategies.get_text_encoders_train_flags(cfg, text_encoders)):
            if flag:
                if t_enc.supports_gradient_checkpointing:
                    t_enc.gradient_checkpointing_enable()
        del t_enc
        adapter.enable_gradient_checkpointing()  # may be overwritten by "adapter_multipliers" in the next step

    # 学習に必要なクラスを準備する
    accelerator.print("prepare optimizer, data loader etc.")

    (
        optimizer_name,
        optimizer_args,
        optimizer,
        optimizer_train_fn,
        optimizer_eval_fn,
        lr_descriptions,
        text_encoder_lr  # TODO: why only text_encoder_lr here?
    ) = prepare_optimizer(cfg.optimizer, cfg.peft, cfg.dataset, adapter)

    # prepare dataloader
    # strategies are set here because they cannot be referenced in another process. Copy them with the dataset
    # some strategies can be None
    train_dataset_group.set_current_strategies()
    if val_dataset_group is not None:
        val_dataset_group.set_current_strategies()

    # DataLoaderのプロセス数：0 は persistent_workers が使えないので注意
    n_workers = min(cfg.training.max_data_loader_n_workers, os.cpu_count())  # cpu_count or max_data_loader_n_workers

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=cfg.training.persistent_data_loader_workers,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset_group if val_dataset_group is not None else [],
        shuffle=False,
        batch_size=1,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=cfg.training.persistent_data_loader_workers,
    )

    if val_dataset_group is not None:
        val_dataloader = accelerator.prepare(val_dataloader)
        cyclic_val_dataloader = itertools.cycle(val_dataloader)
    else:
        val_dataloader, cyclic_val_dataloader = None, None

    # 学習ステップ数を計算する
    if cfg.training.max_train_epochs is not None:
        cfg.training.max_train_steps = cfg.training.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / cfg.training.gradient_accumulation_steps
        )
        accelerator.print(
            f"override steps. steps for {cfg.training.max_train_epochs} epochs is / 指定エポックまでのステップ数: {cfg.training.max_train_steps}"
        )

    # データセット側にも学習ステップを送信
    train_dataset_group.set_max_train_steps(cfg.training.max_train_steps)

    # lr schedulerを用意する
    lr_scheduler = get_scheduler_fix(cfg.optimizer, cfg.dataset, cfg.training, optimizer, accelerator.num_processes)

    # 実験的機能：勾配も含めたfp16/bf16学習を行う　モデル全体をfp16/bf16にする
    if cfg.performance.precision.full_fp16:
        accelerator.print("enable full fp16 training.")
        adapter.to(weight_dtype)
    elif cfg.performance.precision.full_bf16:
        accelerator.print("enable full bf16 training.")
        adapter.to(weight_dtype)

    unet_weight_dtype = te_weight_dtype = weight_dtype
    # Experimental Feature: Put base model into fp8 to save vram
    if cfg.performance.precision.fp8_base or cfg.performance.precision.fp8_base_unet:
        assert torch.__version__ >= "2.1.0", "fp8_base requires torch>=2.1.0 / fp8を使う場合はtorch>=2.1.0が必要です。"
        accelerator.print("enable fp8 training for U-Net.")
        unet_weight_dtype = torch.float8_e4m3fn

        if not cfg.performance.precision.fp8_base_unet:
            accelerator.print("enable fp8 training for Text Encoder.")
        te_weight_dtype = weight_dtype if cfg.performance.precision.fp8_base_unet else torch.float8_e4m3fn

        # unet.to(accelerator.device)  # this makes faster `to(dtype)` below, but consumes 23 GB VRAM
        # unet.to(dtype=unet_weight_dtype)  # without moving to gpu, this takes a lot of time and main memory

        # logger.info(f"set U-Net weight dtype to {unet_weight_dtype}, device to {accelerator.device}")
        # unet.to(accelerator.device, dtype=unet_weight_dtype)  # this seems to be safer than above
        logger.info(f"set U-Net weight dtype to {unet_weight_dtype}")
        unet.to(dtype=unet_weight_dtype)  # do not move to device because unet is not prepared by accelerator

    unet.requires_grad_(False)
    if strategies.cast_unet(cfg):
        unet.to(dtype=unet_weight_dtype)
    for i, t_enc in enumerate(text_encoders):
        t_enc.requires_grad_(False)

        # in case of cpu, dtype is already set to fp32 because cpu does not support fp8/fp16/bf16
        if t_enc.device.type != "cpu" and strategies.cast_text_encoder(cfg):
            t_enc.to(dtype=te_weight_dtype)

            # nn.Embedding not support FP8
            if te_weight_dtype != weight_dtype:
                strategies.prepare_text_encoder_fp8(i, t_enc, te_weight_dtype, weight_dtype)

    # acceleratorがなんかよろしくやってくれるらしい / accelerator will do something good
    if cfg.performance.deepspeed:
        flags = strategies.get_text_encoders_train_flags(cfg, text_encoders)
        ds_model = deepspeed_utils.prepare_deepspeed_model(
            cfg.training,
            unet=unet if train_unet else None,
            text_encoder1=text_encoders[0] if flags[0] else None,
            text_encoder2=(text_encoders[1] if flags[1] else None) if len(text_encoders) > 1 else None,
            adapter=adapter,
        )
        ds_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            ds_model, optimizer, train_dataloader, lr_scheduler
        )
        training_model = ds_model
    else:
        if train_unet:
            # default implementation is:  unet = accelerator.prepare(unet)
            unet = strategies.prepare_unet_with_accelerator(cfg, accelerator, unet)  # accelerator does some magic here
        else:
            # move to device because unet is not prepared by accelerator
            unet.to(accelerator.device, dtype=unet_weight_dtype if strategies.cast_unet(cfg) else None)
        if train_text_encoder:
            text_encoders = [
                (accelerator.prepare(t_enc) if flag else t_enc)
                for t_enc, flag in zip(text_encoders, strategies.get_text_encoders_train_flags(cfg, text_encoders))
            ]
            if len(text_encoders) > 1:
                text_encoder = text_encoders
            else:
                text_encoder = text_encoders[0]
        else:
            pass  # if text_encoder is not trained, no need to prepare. and device and dtype are already set

        adapter, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            adapter, optimizer, train_dataloader, lr_scheduler
        )
        training_model = adapter

    if val_dataset_group is not None:
        val_dataloader = accelerator.prepare(val_dataloader)
        cyclic_val_dataloader = itertools.cycle(val_dataloader)
    else:
        val_dataloader, cyclic_val_dataloader = None, None

    if cfg.performance.memory.gradient_checkpointing:
        # according to TI example in Diffusers, train is required
        unet.train()
        for i, (t_enc, frag) in enumerate(zip(text_encoders, strategies.get_text_encoders_train_flags(cfg, text_encoders))):
            t_enc.train()

            # set top parameter requires_grad = True for gradient checkpointing works
            if frag:
                strategies.prepare_text_encoder_grad_ckpt_workaround(i, t_enc)

    else:
        unet.eval()
        for t_enc in text_encoders:
            t_enc.eval()

    del t_enc

    accelerator.unwrap_model(adapter).prepare_grad_etc(text_encoder, unet)

    if not cache_latents:  # キャッシュしない場合はVAEを使うのでVAEを準備する
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=vae_dtype)

    # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
    if cfg.performance.precision.full_fp16:
        patch_accelerator_for_fp16_training(accelerator)

    # before resuming make hook for saving/loading to save/load the peft weights only
    get_steps_from_state = register_adapter_state_hooks(
        accelerator, adapter, cfg, current_epoch, current_step
    )

    # resumeする
    resume_from_local_or_hf_if_specified(accelerator, cfg.output.saving)
    steps_from_state = get_steps_from_state()

    # epoch数を計算する
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.training.gradient_accumulation_steps)
    num_train_epochs = math.ceil(cfg.training.max_train_steps / num_update_steps_per_epoch)
    if (cfg.output.saving.save_n_epoch_ratio is not None) and (cfg.output.saving.save_n_epoch_ratio > 0):
        cfg.output.saving.save_every_n_epochs = math.floor(num_train_epochs / cfg.output.saving.save_n_epoch_ratio) or 1

    # 学習する
    # TODO: find a way to handle total batch size when there are multiple datasets
    total_batch_size = cfg.training.train_batch_size * accelerator.num_processes * cfg.training.gradient_accumulation_steps

    accelerator.print("running training")
    accelerator.print(f"  num train images * repeats: {train_dataset_group.num_train_images}")
    accelerator.print(f"  num validation images * repeats: {val_dataset_group.num_train_images if val_dataset_group is not None else 0}")
    accelerator.print(f"  num reg images: {train_dataset_group.num_reg_images}")
    accelerator.print(f"  num batches per epoch: {len(train_dataloader)}")
    accelerator.print(f"  num epochs: {num_train_epochs}")
    accelerator.print(f"  batch size per device: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}")
    # accelerator.print(f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}")
    accelerator.print(f"  gradient accumulation steps: {cfg.training.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps: {cfg.training.max_train_steps}")

    # Create training metadata
    metadata, minimum_metadata = create_training_metadata(
        cfg=cfg,
        session_id=session_id,
        training_started_at=training_started_at,
        model_version=model_version,
        train_dataset_group=train_dataset_group,
        val_dataset_group=val_dataset_group,
        num_train_epochs=num_train_epochs,
        optimizer_name=optimizer_name,
        optimizer_args=optimizer_args,
        text_encoder_lr=text_encoder_lr,  # TODO: why only text_encoder_lr here?
        net_kwargs=net_kwargs,
        train_dataloader=train_dataloader,
        total_batch_size=total_batch_size,
        use_user_config=use_user_config,
        use_dreambooth_method=use_dreambooth_method,
    )
    strategies.update_metadata(metadata, cfg)  # architecture specific metadata

    # calculate steps to skip when resuming or starting from a specific step
    initial_step, epoch_to_start = calculate_initial_step(cfg, train_dataloader, accelerator, steps_from_state)

    global_step = 0

    noise_scheduler = strategies.get_noise_scheduler(cfg, accelerator.device)

    # --- Custom Timestep Sampler Initialization ---
    strategies.la_sampler = init_timestep_sampler(cfg, noise_scheduler, accelerator)

    # --- LIVE PLOTTER & STATIC PLOT SETUP ---
    timestep_counts = None
    plotter_settings = None
    if is_main_process:
        timestep_counts, plotter_settings = setup_live_plotter(
            cfg, noise_scheduler, strategies.la_sampler, strategies
        )

    edm2_model, edm2_optimizer, edm2_lr_scheduler = prepare_edm2_loss_weighting(cfg.loss, cfg.training, noise_scheduler,
                                                                                accelerator)

    init_trackers(accelerator, cfg, "adapter_train")

    loss_recorder = EMARecorder()
    val_loss_recorder = EMARecorder()

    if cfg.loss.edm2.edm2_loss_weighting:
        loss_scaled_recorder = EMARecorder()

    del train_dataset_group
    if val_dataset_group is not None:
        del val_dataset_group

    # callback for step start
    if hasattr(accelerator.unwrap_model(adapter), "on_step_start"):
        on_step_start_for_adapter = accelerator.unwrap_model(adapter).on_step_start
    else:
        on_step_start_for_adapter = lambda *args, **kwargs: None

    # function for saving/removing
    def save_model(ckpt_name, unwrapped_nw, steps, epoch_no, force_sync_upload=False, dtype_override=None):
        os.makedirs(cfg.output.saving.output_dir, exist_ok=True)
        ckpt_file = os.path.join(cfg.output.saving.output_dir, ckpt_name)

        accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
        metadata["ss_training_finished_at"] = str(time.time())
        metadata["ss_steps"] = str(steps)
        metadata["ss_epoch"] = str(epoch_no)

        metadata_to_save = minimum_metadata if cfg.output.saving.no_metadata else metadata
        sai_metadata = strategies.get_sai_model_spec(cfg)
        metadata_to_save.update(sai_metadata)

        unwrapped_nw.save_weights(ckpt_file, dtype_override or save_dtype, metadata_to_save)
        if cfg.output.huggingface.huggingface_repo_id is not None:
            huggingface_util.upload(cfg.output.huggingface, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)

    def remove_model(old_ckpt_name):
        old_ckpt_file = os.path.join(cfg.output.saving.output_dir, old_ckpt_name)
        if os.path.exists(old_ckpt_file):
            accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
            os.remove(old_ckpt_file)

    # if text_encoder is not needed for training, delete it to save memory.
    # TODO this can be automated after SDXL sample prompt cache is implemented
    if strategies.is_text_encoder_not_needed_for_training(cfg):
        logger.info("text_encoder is not needed for training. deleting to save memory.")
        for t_enc in text_encoders:
            del t_enc
        text_encoders = []
        text_encoder = None
        gc.collect()
        clean_memory_on_device(accelerator.device)

    current_val_loss, average_val_loss, val_logs = None, None, {}
    keys_scaled, mean_norm, maximum_norm = None, None, None
    mean_grad_norm, mean_combined_norm = None, None
    max_mean_logs = {}
    current_global_step_loss = 0.0
    current_global_step_loss_scaled = 0.0 if cfg.loss.edm2.edm2_loss_weighting else None
    average_loss_scaled = 0.0 if cfg.loss.edm2.edm2_loss_weighting else None
    avr_loss = 0.0
    accumulation_counter = 0

    # For --sample_at_first
    if sample_images_check(cfg.output.sampling, 0, global_step) or calculate_val_loss_check(cfg.validation, cfg.training, global_step, 0,
                                                                                     val_dataloader, train_dataloader):
        # Switch peft to eval mode
        accelerator.unwrap_model(adapter).eval()
        optimizer_eval_fn()
        strategies.sample_images(accelerator, cfg, 0, global_step, accelerator.device, vae, tokenizers, text_encoder, unet)
        if calculate_val_loss_check(cfg.validation, cfg.training, global_step, 0, val_dataloader, train_dataloader):
            current_val_loss, average_val_loss, val_logs = strategies.calculate_val_loss(
                global_step, 0, train_dataloader, val_loss_recorder, val_dataloader,
                cyclic_val_dataloader, adapter, tokenize_strategy,
                text_encoders, text_encoding_strategy, unet, vae, noise_scheduler,
                vae_dtype, weight_dtype, accelerator, cfg, 0, None, train_text_encoder)
        # Switch peft to train mode
        optimizer_train_fn()
        accelerator.unwrap_model(adapter).train()

    if plot_edm2_loss_weighting_check(cfg.loss, cfg.training, global_step):
        plot_edm2_loss_weighting(cfg.loss, cfg.output.saving.output_name, global_step, edm2_model, 1000, accelerator.device)

    is_tracking = len(accelerator.trackers) > 0
    if is_tracking:
        logs = generate_step_logs(
            cfg,
            current_global_step_loss,
            avr_loss,
            lr_scheduler,
            lr_descriptions,
            la_sampler=strategies.la_sampler,
            optimizer=optimizer,
            keys_scaled=keys_scaled,
            mean_norm=mean_norm,
            maximum_norm=maximum_norm,
            mean_grad_norm=mean_grad_norm,
            mean_combined_norm=mean_combined_norm,
            edm2_lr_scheduler=edm2_lr_scheduler,
            current_loss_scaled=current_global_step_loss_scaled,
            average_loss_scaled=average_loss_scaled,
            current_val_loss=current_val_loss,
            average_val_loss=average_val_loss
        )
        # log empty object to commit the sample images to wandb
        accelerator.log(logs, step=0)

        # training loop
    if initial_step > 0:  # only if skip_until_initial_step is specified
        for skip_epoch in range(epoch_to_start):  # skip epochs
            logger.info(f"skipping epoch {skip_epoch + 1} because initial_step (multiplied) is {initial_step}")
            initial_step -= len(train_dataloader)
        global_step = initial_step

    # log device and dtype for each model
    logger.info(f"unet dtype: {unet_weight_dtype}, device: {unet.device}")
    for i, t_enc in enumerate(text_encoders):
        params_itr = t_enc.parameters()
        params_itr.__next__()  # skip the first parameter
        params_itr.__next__()  # skip the second parameter. because CLIP first two parameters are embeddings
        param_3rd = params_itr.__next__()
        logger.info(f"text_encoder [{i}] dtype: {param_3rd.dtype}, device: {t_enc.device}")

    # --- Dynamic Timestep Schedule ---
    dynamic_timestep_schedule, current_min_timestep, current_max_timestep = parse_dynamic_timestep_schedule(
        cfg, noise_scheduler, accelerator
    )

    clean_memory_on_device(accelerator.device)

    progress_bar = tqdm(
        range(cfg.training.max_train_steps - initial_step), smoothing=0, disable=not accelerator.is_local_main_process,
        desc="steps"
    )

    for epoch in range(epoch_to_start, num_train_epochs):
        current_epoch.value = epoch + 1
        accelerator.print(f"\nepoch {current_epoch.value}/{num_train_epochs}\n")

        metadata["ss_epoch"] = str(current_epoch.value)

        accelerator.unwrap_model(adapter).on_epoch_start(text_encoder, unet)  # peft.train() is called here

        # TRAINING
        skipped_dataloader = None
        if initial_step > 0:
            skipped_dataloader = accelerator.skip_first_batches(train_dataloader, initial_step - 1)
            initial_step = 1

        for step, batch in enumerate(skipped_dataloader or train_dataloader):
            current_step.value = global_step

            # --- Add this block to update the timestep range ---
            if dynamic_timestep_schedule and len(dynamic_timestep_schedule) > 0 and global_step >= \
                    dynamic_timestep_schedule[0][0]:
                # Get the next schedule stage and remove it from the list
                _, new_min, new_max = dynamic_timestep_schedule.pop(0)
                current_min_timestep = new_min
                current_max_timestep = new_max
                accelerator.print(
                    f"\nStep {global_step}: Timestep range dynamically changed to [{current_min_timestep}, {current_max_timestep})"
                )
            # ---------------------------------------------------

            if initial_step > 0:
                initial_step -= 1
                continue

            with determine_grad_sync_context(cfg, accelerator, None, training_model, edm2_model):
                on_step_start_for_adapter(text_encoder, unet)

                accumulation_counter += 1

                # preprocess batch for each model
                strategies.on_step_start(cfg, accelerator, adapter, text_encoders, unet, batch, weight_dtype, is_train=True)

                loss, pre_scaling_loss, loss_scaled, timesteps = strategies.process_batch(
                    batch,
                    text_encoders,
                    unet,
                    adapter,
                    vae,
                    noise_scheduler,
                    vae_dtype,
                    weight_dtype,
                    accelerator,
                    cfg,
                    text_encoding_strategy,
                    tokenize_strategy,
                    is_train=True,
                    train_text_encoder=train_text_encoder,
                    train_unet=train_unet,
                    edm2_model=edm2_model,
                    min_timestep_override=current_min_timestep,
                    max_timestep_override=current_max_timestep,
                    global_step=global_step,
                )

                accelerator.backward(loss)

                loss = pre_scaling_loss

                if accelerator.sync_gradients:
                    strategies.all_reduce_adapter(accelerator, adapter)  # sync DDP grad manually
                    if cfg.optimizer.max_grad_norm != 0.0:
                        params_to_clip = accelerator.unwrap_model(adapter).get_trainable_params()
                        accelerator.clip_grad_norm_(params_to_clip, cfg.optimizer.max_grad_norm)

                    # if hasattr(peft, "update_grad_norms"):
                    #    peft.update_grad_norms()
                    # if hasattr(peft, "update_norms"):
                    #    peft.update_norms()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if cfg.loss.edm2.edm2_loss_weighting:
                    edm2_optimizer.step()
                    edm2_lr_scheduler.step()
                    # swap to pre_scaling_loss for logging
                    edm2_optimizer.zero_grad(set_to_none=True)

            if cfg.peft.scale_weight_norms and accelerator.sync_gradients:
                keys_scaled, mean_norm, maximum_norm = accelerator.unwrap_model(adapter).apply_max_norm_regularization(
                    cfg.peft.scale_weight_norms, accelerator.device
                )
                mean_grad_norm = None
                mean_combined_norm = None
                max_mean_logs = {"Keys Scaled": keys_scaled, "Average key norm": mean_norm}
            else:
                keys_scaled, mean_norm, maximum_norm = None, None, None
                mean_grad_norm = None
                mean_combined_norm = None
                max_mean_logs = {}

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if (sample_images_check(cfg.output.sampling, None, global_step) or
                        calculate_val_loss_check(cfg.validation, cfg.training, global_step, step, val_dataloader, train_dataloader) or
                        cfg.output.saving.save_every_n_steps is not None and global_step % cfg.output.saving.save_every_n_steps == 0):

                    accelerator.unwrap_model(adapter).eval()
                    optimizer_eval_fn()
                    strategies.sample_images(
                        accelerator, cfg, None, global_step, accelerator.device, vae, tokenizers, text_encoder, unet
                    )

                    if calculate_val_loss_check(cfg.validation, cfg.training, global_step, step, val_dataloader, train_dataloader):
                        current_val_loss, average_val_loss, val_logs = strategies.calculate_val_loss(global_step, step,
                                                                                               skipped_dataloader or train_dataloader,
                                                                                               val_loss_recorder,
                                                                                               val_dataloader,
                                                                                               cyclic_val_dataloader,
                                                                                               adapter,
                                                                                               tokenize_strategy,
                                                                                               text_encoders,
                                                                                               text_encoding_strategy,
                                                                                               unet,
                                                                                               vae,
                                                                                               noise_scheduler,
                                                                                               vae_dtype,
                                                                                               weight_dtype,
                                                                                               accelerator,
                                                                                               cfg,
                                                                                               batch,
                                                                                               current_epoch.value,
                                                                                               train_text_encoder)
                    else:
                        current_val_loss, average_val_loss, val_logs = None, None, None

                    # 指定ステップごとにモデルを保存
                    if cfg.output.saving.save_every_n_steps is not None and global_step % cfg.output.saving.save_every_n_steps == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            ckpt_name = get_step_ckpt_name(cfg.output.saving, "." + cfg.output.saving.save_model_as, global_step)
                            save_model(ckpt_name, accelerator.unwrap_model(adapter), global_step, epoch)

                            if cfg.loss.edm2.edm2_loss_weighting:
                                loss_weights_ckpt_name = get_step_ckpt_name(cfg.output.saving, "." + cfg.output.saving.save_model_as,
                                                                            global_step, "_edm2_loss_weights")
                                save_model(loss_weights_ckpt_name, accelerator.unwrap_model(edm2_model), global_step,
                                           epoch, dtype_override=torch.float32)

                            if cfg.output.saving.save_state:
                                save_and_remove_state_stepwise(cfg.output.saving, accelerator, global_step)

                            remove_step_no = get_remove_step_no(cfg.output.saving, global_step)
                            if remove_step_no is not None:
                                remove_ckpt_name = get_step_ckpt_name(cfg.output.saving, "." + cfg.output.saving.save_model_as,
                                                                      remove_step_no)
                                remove_model(remove_ckpt_name)

                                if cfg.loss.edm2.edm2_loss_weighting:
                                    remove_loss_weights_ckpt_name = get_step_ckpt_name(cfg.output.saving,
                                                                                       "." + cfg.output.saving.save_model_as,
                                                                                       remove_step_no,
                                                                                       "_edm2_loss_weights")
                                    remove_model(remove_loss_weights_ckpt_name)

                    if plot_edm2_loss_weighting_check(cfg.loss, cfg.training, global_step):
                        plot_edm2_loss_weighting(cfg.loss, cfg.output.saving.output_name, global_step, edm2_model, 1000,
                                                 accelerator.device)
                    optimizer_train_fn()
                    accelerator.unwrap_model(adapter).train()

            current_global_step_loss += loss.detach().item()
            if cfg.loss.edm2.edm2_loss_weighting:
                current_global_step_loss_scaled += loss_scaled.detach().item()
            else:
                current_global_step_loss_scaled = None

            if accelerator.sync_gradients:
                loss_recorder.add(current_global_step_loss / accumulation_counter)
                if cfg.loss.edm2.edm2_loss_weighting:
                    loss_scaled_recorder.add(current_global_step_loss_scaled / accumulation_counter)
                avr_loss: float = loss_recorder.average
                logs = {"avr_loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**{**max_mean_logs, **logs})

                if is_tracking:
                    current_global_step_loss = (current_global_step_loss / accumulation_counter)
                    if cfg.loss.edm2.edm2_loss_weighting:
                        current_global_step_loss_scaled = (current_global_step_loss_scaled / accumulation_counter)
                        average_loss_scaled: float = loss_scaled_recorder.average
                    else:
                        current_global_step_loss_scaled = None
                        average_loss_scaled = None

                    logs = generate_step_logs(
                        cfg,
                        current_global_step_loss,
                        avr_loss,
                        lr_scheduler,
                        lr_descriptions,
                        la_sampler=strategies.la_sampler,
                        optimizer=optimizer,
                        keys_scaled=keys_scaled,
                        mean_norm=mean_norm,
                        maximum_norm=maximum_norm,
                        mean_grad_norm=mean_grad_norm,
                        mean_combined_norm=mean_combined_norm,
                        edm2_lr_scheduler=edm2_lr_scheduler,
                        current_loss_scaled=current_global_step_loss_scaled,
                        average_loss_scaled=average_loss_scaled,
                        current_val_loss=current_val_loss,
                        average_val_loss=average_val_loss,
                        timesteps=timesteps
                    )
                    step_logging(accelerator, logs, global_step, epoch + 1)

                current_global_step_loss = 0.0

                if cfg.loss.edm2.edm2_loss_weighting:
                    current_global_step_loss_scaled = 0.0

                accumulation_counter = 0

                # --- LIVE PLOTTER & STATIC PLOT UPDATE ---
                if is_main_process:
                    timesteps_np = timesteps.cpu().numpy()

                    # Send data to the live plotter
                    if strategies.live_plotter_process and strategies.live_plotter_process.poll() is None:
                        try:
                            strategies.live_plotter_process.stdin.write(
                                f"{','.join(map(str, timesteps_np))}\n".encode('utf-8'))
                            strategies.live_plotter_process.stdin.flush()
                        except (BrokenPipeError, OSError):
                            logger.error("Live plotter connection lost.")
                            strategies.live_plotter_process = None

                    # Update counts and save static plot if needed
                    if timestep_counts is not None:
                        unique, counts = np.unique(timesteps_np, return_counts=True)
                        timestep_counts[unique] += counts

                        if global_step % cfg.output.logging.log_timestep_distribution_every_n_steps == 0:
                            save_timestep_distribution_plot(cfg, global_step, timestep_counts, plotter_settings)

            if global_step >= cfg.training.max_train_steps:
                break

        # END OF EPOCH
        if is_tracking:
            logs = {"loss/epoch_average": loss_recorder.average}
            accelerator.log(logs, step=global_step)

        accelerator.wait_for_everyone()

        if (sample_images_check(cfg.output.sampling, current_epoch.value, global_step) or
                cfg.output.saving.save_every_n_epochs is not None):

            # 指定エポックごとにモデルを保存
            optimizer_eval_fn()
            accelerator.unwrap_model(adapter).eval()
            if cfg.output.saving.save_every_n_epochs is not None:
                saving = current_epoch.value % cfg.output.saving.save_every_n_epochs == 0 and current_epoch.value < num_train_epochs
                if is_main_process and saving:
                    ckpt_name = get_epoch_ckpt_name(cfg.output.saving, "." + cfg.output.saving.save_model_as, current_epoch.value)
                    save_model(ckpt_name, accelerator.unwrap_model(adapter), global_step, current_epoch.value)

                    if cfg.loss.edm2.edm2_loss_weighting:
                        loss_weights_ckpt_name = get_epoch_ckpt_name(cfg.output.saving, "." + cfg.output.saving.save_model_as,
                                                                     current_epoch.value, "_edm2_loss_weights")
                        save_model(loss_weights_ckpt_name, accelerator.unwrap_model(edm2_model), global_step,
                                   current_epoch.value, dtype_override=torch.float32)

                    remove_epoch_no = get_remove_epoch_no(cfg.output.saving, current_epoch.value)
                    if remove_epoch_no is not None:
                        remove_ckpt_name = get_epoch_ckpt_name(cfg.output.saving, "." + cfg.output.saving.save_model_as,
                                                               remove_epoch_no)
                        remove_model(remove_ckpt_name)

                        if cfg.loss.edm2.edm2_loss_weighting:
                            remove_loss_weights_ckpt_name = get_epoch_ckpt_name(cfg.output.saving,
                                                                                "." + cfg.output.saving.save_model_as,
                                                                                remove_epoch_no, "_edm2_loss_weights")
                            remove_model(remove_loss_weights_ckpt_name)

                    if cfg.output.saving.save_state:
                        save_and_remove_state_on_epoch_end(cfg.output.saving, accelerator, current_epoch.value)

            strategies.sample_images(accelerator, cfg, current_epoch.value, global_step, accelerator.device, vae, tokenizers,
                               text_encoder, unet)
            progress_bar.unpause()
            optimizer_train_fn()
            accelerator.unwrap_model(adapter).train()

        # end of epoch

    # metadata["ss_epoch"] = str(num_train_epochs)
    metadata["ss_training_finished_at"] = str(time.time())

    if is_main_process:
        adapter = accelerator.unwrap_model(adapter)

    accelerator.end_training()
    optimizer_eval_fn()

    if is_main_process and (cfg.output.saving.save_state or cfg.output.saving.save_state_on_train_end):
        save_state_on_train_end(cfg.output.saving, accelerator)

    if is_main_process:
        ckpt_name = get_last_ckpt_name(cfg.output.saving, "." + cfg.output.saving.save_model_as)
        save_model(ckpt_name, adapter, global_step, num_train_epochs, force_sync_upload=True)

        if cfg.loss.edm2.edm2_loss_weighting:
            loss_weights_ckpt_name = get_last_ckpt_name(cfg.output.saving, "." + cfg.output.saving.save_model_as,
                                                        "_edm2_loss_weights")
            save_model(loss_weights_ckpt_name, accelerator.unwrap_model(edm2_model), global_step, num_train_epochs,
                       force_sync_upload=True, dtype_override=torch.float32)

    logger.info("model saved.")


# Register the structure config with Hydra
cs = ConfigStore.instance()
cs.store(name="sd_peft", node=SDPeftConfig)


@hydra.main(version_base=None, config_path="../configs", config_name="sd_peft")
def main(cfg: SDPeftConfig):
    """Main entry point for SD PEFT training."""
    prepare_config(cfg)
    validate_config(cfg)

    strategies = SdPeftStrategy()
    train(cfg, strategies)


if __name__ == "__main__":
    main()
