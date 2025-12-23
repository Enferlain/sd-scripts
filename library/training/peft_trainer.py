# cut and pasted from sd_peft.py
# used to be in Class SDPeftTrainer, used to be indented 1 tab in it
# imports are copied from sd_peft.py for now

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
from library.optimizations import deepspeed_utils
from library.models import model_util
from library.utils import sai_model_spec
from library.utils.common_utils import setup_logging
from library.utils.device_utils import init_ipex, clean_memory_on_device
from library.utils.torch_utils import set_torch_cuda_reduced_precision, set_seed_from_config, prepare_dtype
from library.training.diffusion import get_noise_noisy_latents_and_timesteps
from library.training.model_prep import load_target_model, replace_unet_modules, patch_accelerator_for_fp16_training
from library.training.optimizer import prepare_optimizer, get_scheduler_fix
from library.training.sample_generation import sample_images, sample_images_check
from library.losses.loss import get_huber_threshold_if_needed, conditional_loss, EMARecorder
from library.config.dataclasses.sd_peft import SDPeftConfig

from library.timestep_samplers.loss_aware_sampler import LossAwareTimestepSampler
from library.timestep_samplers.log_snr_sampler import LogSNRUniformSampler
from library.timestep_samplers.tempered_adaptive_sampler import TemperedAdaptiveSampler
from library.timestep_samplers.gaussian_mid_snr_sampler import GaussianMidSNRAdaptiveSampler
from library.timestep_samplers.snr_windowed_loss_aware_sampler import SNRWindowedLossAwareSampler

from library.config.config_util import (
    BlueprintGenerator,
)

from library.training.checkpointing import (
    get_sai_model_spec,
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


def train(self, cfg: SDPeftConfig):
    self.la_sampler = None

    session_id = random.randint(0, 2 ** 32)
    training_started_at = time.time()

    # verify_training_args(args)  # TODO VALIDATION FOR CONFIGS WHEREVER

    set_torch_cuda_reduced_precision(cfg.performance)
    deepspeed_utils.prepare_deepspeed_config(cfg.performance, cfg.training)
    setup_logging(cfg.logging, reset=True)

    cache_latents = cfg.dataset.cache_latents
    use_dreambooth_method = cfg.dataset.in_json is None
    use_user_config = cfg.dataset.dataset_config is not None

    set_seed_from_config(cfg.training)

    tokenize_strategy = self.get_tokenize_strategy(cfg)
    strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)
    tokenizers = self.get_tokenizers(tokenize_strategy)  # will be removed after sample_image is refactored

    # prepare caching strategy: this must be set before preparing dataset. because dataset may use this strategy for initialization.
    latents_caching_strategy = self.get_latents_caching_strategy(cfg)
    strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

    # データセットを準備する
    if cfg.dataset.dataset_class is None:
        # Check if we have manually provided subsets via train_data_dir/reg_data_dir
        if (cfg.dataset.train_data_dir is not None or cfg.dataset.reg_data_dir is not None) and len(
                cfg.dataset.subsets) == 0:
            # Generate subsets config from dirs
            user_config = config_util.generate_user_config_from_dataset(cfg.dataset)
            # We need to inject this into cfg.dataset.subsets
            # cfg.dataset.subsets is a List[dict] (or ListConfig)
            # user_config['datasets'][0]['subsets'] is the list we want
            if user_config['datasets']:
                cfg.dataset.subsets = user_config['datasets'][0]['subsets']

        blueprint_generator = BlueprintGenerator()
        blueprint = blueprint_generator.generate(cfg)
        train_dataset_group, val_dataset_group = config_util.generate_dataset_group_by_blueprint(
            blueprint.dataset_group)
    else:
        # use arbitrary dataset class
        # load_arbitrary_dataset expects args
        train_dataset_group = load_arbitrary_dataset(cfg.dataset)
        val_dataset_group = None  # placeholder until validation dataset supported for arbitrary

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if cfg.training.max_data_loader_n_workers == 0 else None
    collator = collator_class(current_epoch, current_step, ds_for_collator)

    if cfg.dataset.debug_dataset:
        train_dataset_group.set_current_strategies()  # dataset needs to know the strategies explicitly
        debug_dataset(train_dataset_group)

        if val_dataset_group is not None:
            val_dataset_group.set_current_strategies()  # dataset needs to know the strategies explicitly
            debug_dataset(val_dataset_group)
        return
    if len(train_dataset_group) == 0:
        logger.error(
            "No data found. Please verify arguments (train_data_dir must be the parent of folders with images) / 画像がありません。引数指定を確認してください（train_data_dirには画像があるフォルダではなく、画像があるフォルダの親フォルダを指定する必要があります）"
        )
        return

    if cache_latents:
        assert (
            train_dataset_group.is_latent_cacheable()
        ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"
        if val_dataset_group is not None:
            assert (
                val_dataset_group.is_latent_cacheable()
            ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"

    self.validate_extra_config(cfg, train_dataset_group, val_dataset_group)

    # acceleratorを準備する
    logger.info("preparing accelerator")
    accelerator = prepare_accelerator(cfg.performance, cfg.logging, cfg.training)
    is_main_process = accelerator.is_main_process

    # mixed precisionに対応した型を用意しておき適宜castする
    weight_dtype, save_dtype = prepare_dtype(cfg.performance, cfg.saving)
    vae_dtype = (torch.float32 if cfg.performance.no_half_vae else weight_dtype) if self.cast_vae(cfg) else None

    # load target models: unet may be None for lazy loading
    model_version, text_encoder, vae, unet = self.load_target_model(cfg, weight_dtype, accelerator)

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
    text_encoding_strategy = self.get_text_encoding_strategy(cfg)
    strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

    text_encoder_outputs_caching_strategy = self.get_text_encoder_outputs_caching_strategy(cfg)
    if text_encoder_outputs_caching_strategy is not None:
        strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(text_encoder_outputs_caching_strategy)
    self.cache_text_encoder_outputs_if_needed(cfg, accelerator, unet, vae, text_encoders, train_dataset_group,
                                              weight_dtype)
    if val_dataset_group is not None:
        self.cache_text_encoder_outputs_if_needed(cfg, accelerator, unet, vae, text_encoders, val_dataset_group,
                                                  weight_dtype)

    if unet is None:
        # lazy load unet if needed. text encoders may be freed or replaced with dummy models for saving memory
        unet, text_encoders = self.load_unet_lazily(cfg, weight_dtype, accelerator, text_encoders)

    # 差分追加学習のためにモデルを読み込む
    sys.path.append(os.path.dirname(__file__))
    accelerator.print("import network module:", cfg.network.network_module)
    network_module = importlib.import_module(cfg.network.network_module)

    if cfg.network.base_weights is not None:
        # base_weights が指定されている場合は、指定された重みを読み込みマージする
        for i, weight_path in enumerate(cfg.network.base_weights):
            if cfg.network.base_weights_multiplier is None or len(cfg.network.base_weights_multiplier) <= i:
                multiplier = 1.0
            else:
                multiplier = cfg.network.base_weights_multiplier[i]

            accelerator.print(f"merging module: {weight_path} with multiplier {multiplier}")

            module, weights_sd = network_module.create_network_from_weights(
                multiplier, weight_path, vae, text_encoder, unet, for_inference=True
            )
            module.merge_to(text_encoder, unet, weights_sd, weight_dtype,
                            accelerator.device if cfg.performance.lowram else "cpu")

        accelerator.print(f"all weights merged: {', '.join(cfg.network.base_weights)}")

    # prepare network
    net_kwargs = {}
    if cfg.network.network_args is not None:
        for net_arg in cfg.network.network_args:
            key, value = net_arg.split("=", 1)
            net_kwargs[key] = value

    # if a new network is added in future, add if ~ then blocks for each network (;'∀')
    if cfg.network.dim_from_weights:
        network, _ = network_module.create_network_from_weights(1, cfg.network.network_weights, vae, text_encoder, unet,
                                                                **net_kwargs)
    else:
        if "dropout" not in net_kwargs:
            # workaround for LyCORIS (;^ω^)
            net_kwargs["dropout"] = cfg.network.network_dropout

        network = network_module.create_network(
            1.0,
            cfg.network.network_dim,
            cfg.network.network_alpha,
            vae,
            text_encoder,
            unet,
            neuron_dropout=cfg.network.network_dropout,
            **net_kwargs,
        )
    if network is None:
        return
    network_has_multiplier = hasattr(network, "set_multiplier")

    # TODO remove `hasattr` by setting up methods if not defined in the network like below  (hacky but will work):
    # if not hasattr(network, "prepare_network"):
    #    network.prepare_network = lambda args: None

    if hasattr(network, "prepare_network"):
        network.prepare_network(cfg)
    if cfg.network.scale_weight_norms and not hasattr(network, "apply_max_norm_regularization"):
        logger.warning(
            "warning: scale_weight_norms is specified but the network does not support it / scale_weight_normsが指定されていますが、ネットワークが対応していません"
        )
        cfg.network.scale_weight_norms = False

    self.post_process_network(cfg, accelerator, network, text_encoders, unet)

    # apply network to unet and text_encoder
    train_unet = not cfg.network.network_train_text_encoder_only
    train_text_encoder = self.is_train_text_encoder(cfg)
    network.apply_to(text_encoder, unet, train_text_encoder, train_unet)

    if cfg.network.network_weights is not None:
        # FIXME consider alpha of weights: this assumes that the alpha is not changed
        info = network.load_weights(cfg.network.network_weights)
        accelerator.print(f"load network weights from {cfg.network.network_weights}: {info}")

    # if args.use_ramtorch:
    #     logger.info("Applying RamTorch to network/lora.")
    #     if isinstance(network, torch.nn.Module):
    #         network = replace_linear_with_ramtorch(network, accelerator.device)
    #         logger.info("RamTorch applied to network/lora.")

    if cfg.performance.gradient_checkpointing:
        if cfg.performance.cpu_offload_checkpointing:
            unet.enable_gradient_checkpointing(cpu_offload=True)
        else:
            unet.enable_gradient_checkpointing()

        for t_enc, flag in zip(text_encoders, self.get_text_encoders_train_flags(cfg, text_encoders)):
            if flag:
                if t_enc.supports_gradient_checkpointing:
                    t_enc.gradient_checkpointing_enable()
        del t_enc
        network.enable_gradient_checkpointing()  # may be overwritten by "network_multipliers" in the next step

    # 学習に必要なクラスを準備する
    accelerator.print("prepare optimizer, data loader etc.")

    (
        optimizer_name,
        optimizer_args,
        optimizer,
        optimizer_train_fn,
        optimizer_eval_fn,
        lr_descriptions,
        text_encoder_lr
    ) = prepare_optimizer(cfg.optimizer, cfg.network, cfg.dataset, network)

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
    if cfg.performance.full_fp16:
        accelerator.print("enable full fp16 training.")
        network.to(weight_dtype)
    elif cfg.performance.full_bf16:
        accelerator.print("enable full bf16 training.")
        network.to(weight_dtype)

    unet_weight_dtype = te_weight_dtype = weight_dtype
    # Experimental Feature: Put base model into fp8 to save vram
    if cfg.performance.fp8_base or cfg.performance.fp8_base_unet:
        assert torch.__version__ >= "2.1.0", "fp8_base requires torch>=2.1.0 / fp8を使う場合はtorch>=2.1.0が必要です。"
        accelerator.print("enable fp8 training for U-Net.")
        unet_weight_dtype = torch.float8_e4m3fn

        if not cfg.performance.fp8_base_unet:
            accelerator.print("enable fp8 training for Text Encoder.")
        te_weight_dtype = weight_dtype if cfg.performance.fp8_base_unet else torch.float8_e4m3fn

        # unet.to(accelerator.device)  # this makes faster `to(dtype)` below, but consumes 23 GB VRAM
        # unet.to(dtype=unet_weight_dtype)  # without moving to gpu, this takes a lot of time and main memory

        # logger.info(f"set U-Net weight dtype to {unet_weight_dtype}, device to {accelerator.device}")
        # unet.to(accelerator.device, dtype=unet_weight_dtype)  # this seems to be safer than above
        logger.info(f"set U-Net weight dtype to {unet_weight_dtype}")
        unet.to(dtype=unet_weight_dtype)  # do not move to device because unet is not prepared by accelerator

    unet.requires_grad_(False)
    if self.cast_unet(cfg):
        unet.to(dtype=unet_weight_dtype)
    for i, t_enc in enumerate(text_encoders):
        t_enc.requires_grad_(False)

        # in case of cpu, dtype is already set to fp32 because cpu does not support fp8/fp16/bf16
        if t_enc.device.type != "cpu" and self.cast_text_encoder(cfg):
            t_enc.to(dtype=te_weight_dtype)

            # nn.Embedding not support FP8
            if te_weight_dtype != weight_dtype:
                self.prepare_text_encoder_fp8(i, t_enc, te_weight_dtype, weight_dtype)

    # acceleratorがなんかよろしくやってくれるらしい / accelerator will do something good
    if cfg.performance.deepspeed:
        flags = self.get_text_encoders_train_flags(cfg, text_encoders)
        ds_model = deepspeed_utils.prepare_deepspeed_model(
            cfg.training,
            unet=unet if train_unet else None,
            text_encoder1=text_encoders[0] if flags[0] else None,
            text_encoder2=(text_encoders[1] if flags[1] else None) if len(text_encoders) > 1 else None,
            network=network,
        )
        ds_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            ds_model, optimizer, train_dataloader, lr_scheduler
        )
        training_model = ds_model
    else:
        if train_unet:
            # default implementation is:  unet = accelerator.prepare(unet)
            unet = self.prepare_unet_with_accelerator(cfg, accelerator, unet)  # accelerator does some magic here
        else:
            # move to device because unet is not prepared by accelerator
            unet.to(accelerator.device, dtype=unet_weight_dtype if self.cast_unet(cfg) else None)
        if train_text_encoder:
            text_encoders = [
                (accelerator.prepare(t_enc) if flag else t_enc)
                for t_enc, flag in zip(text_encoders, self.get_text_encoders_train_flags(cfg, text_encoders))
            ]
            if len(text_encoders) > 1:
                text_encoder = text_encoders
            else:
                text_encoder = text_encoders[0]
        else:
            pass  # if text_encoder is not trained, no need to prepare. and device and dtype are already set

        network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            network, optimizer, train_dataloader, lr_scheduler
        )
        training_model = network

    if val_dataset_group is not None:
        val_dataloader = accelerator.prepare(val_dataloader)
        cyclic_val_dataloader = itertools.cycle(val_dataloader)
    else:
        val_dataloader, cyclic_val_dataloader = None, None

    if cfg.performance.gradient_checkpointing:
        # according to TI example in Diffusers, train is required
        unet.train()
        for i, (t_enc, frag) in enumerate(zip(text_encoders, self.get_text_encoders_train_flags(cfg, text_encoders))):
            t_enc.train()

            # set top parameter requires_grad = True for gradient checkpointing works
            if frag:
                self.prepare_text_encoder_grad_ckpt_workaround(i, t_enc)

    else:
        unet.eval()
        for t_enc in text_encoders:
            t_enc.eval()

    del t_enc

    accelerator.unwrap_model(network).prepare_grad_etc(text_encoder, unet)

    if not cache_latents:  # キャッシュしない場合はVAEを使うのでVAEを準備する
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=vae_dtype)

    # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
    if cfg.performance.full_fp16:
        patch_accelerator_for_fp16_training(accelerator)

    # before resuming make hook for saving/loading to save/load the network weights only
    def save_model_hook(models, weights, output_dir):
        # pop weights of other models than network to save only network weights
        # only main process or deepspeed https://github.com/huggingface/diffusers/issues/2606
        if accelerator.is_main_process or cfg.performance.deepspeed:
            remove_indices = []
            for i, model in enumerate(models):
                if not isinstance(model, type(accelerator.unwrap_model(network))):
                    remove_indices.append(i)
            for i in reversed(remove_indices):
                if len(weights) > i:
                    weights.pop(i)
            # print(f"save model hook: {len(weights)} weights will be saved")

        # save current ecpoch and step
        train_state_file = os.path.join(output_dir, "train_state.json")
        # +1 is needed because the state is saved before current_step is set from global_step
        logger.info(
            f"save train state to {train_state_file} at epoch {current_epoch.value} step {current_step.value + 1}")
        with open(train_state_file, "w", encoding="utf-8") as f:
            json.dump({"current_epoch": current_epoch.value, "current_step": current_step.value + 1}, f)

    steps_from_state = None

    def load_model_hook(models, input_dir):
        # remove models except network
        remove_indices = []
        for i, model in enumerate(models):
            if not isinstance(model, type(accelerator.unwrap_model(network))):
                remove_indices.append(i)
        for i in reversed(remove_indices):
            models.pop(i)
        # print(f"load model hook: {len(models)} models will be loaded")

        # load current epoch and step to
        nonlocal steps_from_state
        train_state_file = os.path.join(input_dir, "train_state.json")
        if os.path.exists(train_state_file):
            with open(train_state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            steps_from_state = data["current_step"]
            logger.info(f"load train state from {train_state_file}: {data}")

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # resumeする
    resume_from_local_or_hf_if_specified(accelerator, cfg.saving)

    # epoch数を計算する
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.training.gradient_accumulation_steps)
    num_train_epochs = math.ceil(cfg.training.max_train_steps / num_update_steps_per_epoch)
    if (cfg.saving.save_n_epoch_ratio is not None) and (cfg.saving.save_n_epoch_ratio > 0):
        cfg.saving.save_every_n_epochs = math.floor(num_train_epochs / cfg.saving.save_n_epoch_ratio) or 1

    # 学習する
    # TODO: find a way to handle total batch size when there are multiple datasets
    total_batch_size = cfg.training.train_batch_size * accelerator.num_processes * cfg.training.gradient_accumulation_steps

    accelerator.print("running training / 学習開始")
    accelerator.print(
        f"  num train images * repeats / 学習画像の数×繰り返し回数: {train_dataset_group.num_train_images}")
    accelerator.print(
        f"  num validation images * repeats / 学習画像の数×繰り返し回数: {val_dataset_group.num_train_images if val_dataset_group is not None else 0}"
    )
    accelerator.print(f"  num reg images / 正則化画像の数: {train_dataset_group.num_reg_images}")
    accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
    accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
    accelerator.print(
        f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
    )
    # accelerator.print(f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}")
    accelerator.print(
        f"  gradient accumulation steps / 勾配を合計するステップ数 = {cfg.training.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps / 学習ステップ数: {cfg.training.max_train_steps}")

    # TODO refactor metadata creation and move to util
    metadata = {
        "ss_session_id": session_id,  # random integer indicating which group of epochs the model came from
        "ss_training_started_at": training_started_at,  # unix timestamp
        "ss_output_name": cfg.saving.output_name,
        "ss_learning_rate": cfg.optimizer.learning_rate,
        "ss_text_encoder_lr": text_encoder_lr,
        "ss_unet_lr": cfg.network.unet_lr,
        "ss_num_train_images": train_dataset_group.num_train_images,
        "ss_num_validation_images": val_dataset_group.num_train_images if val_dataset_group is not None else 0,
        "ss_num_reg_images": train_dataset_group.num_reg_images,
        "ss_num_batches_per_epoch": len(train_dataloader),
        "ss_num_epochs": num_train_epochs,
        "ss_gradient_checkpointing": cfg.performance.gradient_checkpointing,
        "ss_gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "ss_max_train_steps": cfg.training.max_train_steps,
        "ss_lr_warmup_steps": cfg.optimizer.lr_warmup_steps,
        "ss_lr_scheduler": cfg.optimizer.lr_scheduler,
        "ss_network_module": cfg.network.network_module,
        "ss_network_dim": cfg.network.network_dim,
        # None means default because another network than LoRA may have another default dim
        "ss_network_alpha": cfg.network.network_alpha,  # some networks may not have alpha
        "ss_network_dropout": cfg.network.network_dropout,  # some networks may not have dropout
        "ss_mixed_precision": cfg.performance.mixed_precision,
        "ss_full_fp16": bool(cfg.performance.full_fp16),
        "ss_v2": bool(cfg.model.v2),
        "ss_base_model_version": model_version,
        "ss_clip_skip": cfg.training.clip_skip,
        "ss_max_token_length": cfg.training.max_token_length,
        "ss_cache_latents": bool(cfg.dataset.cache_latents),
        "ss_seed": cfg.training.seed,
        "ss_lowram": cfg.performance.lowram,
        "ss_noise_offset": cfg.regularization.noise_offset,
        "ss_multires_noise_iterations": cfg.regularization.multires_noise_iterations,
        "ss_multires_noise_discount": cfg.regularization.multires_noise_discount,
        "ss_adaptive_noise_scale": cfg.regularization.adaptive_noise_scale,
        "ss_zero_terminal_snr": cfg.regularization.zero_terminal_snr,
        "ss_training_comment": cfg.network.training_comment,  # will not be updated after training
        "ss_sd_scripts_commit_hash": get_git_revision_hash(),
        "ss_optimizer": optimizer_name + (f"({optimizer_args})" if len(optimizer_args) > 0 else ""),
        "ss_max_grad_norm": cfg.optimizer.max_grad_norm,
        "ss_caption_dropout_rate": cfg.dataset.caption_dropout_rate,
        "ss_caption_dropout_every_n_epochs": cfg.dataset.caption_dropout_every_n_epochs,
        "ss_caption_tag_dropout_rate": cfg.dataset.caption_tag_dropout_rate,
        "ss_face_crop_aug_range": cfg.dataset.face_crop_aug_range,
        "ss_prior_loss_weight": cfg.loss.prior_loss_weight,
        "ss_min_snr_gamma": cfg.loss.min_snr_gamma,
        "ss_scale_weight_norms": cfg.network.scale_weight_norms,
        "ss_ip_noise_gamma": cfg.regularization.ip_noise_gamma,
        "ss_debiased_estimation": bool(cfg.loss.debiased_estimation_loss),
        "ss_noise_offset_random_strength": cfg.regularization.noise_offset_random_strength,
        "ss_ip_noise_gamma_random_strength": cfg.regularization.ip_noise_gamma_random_strength,
        "ss_loss_type": cfg.loss.loss_type,
        "ss_huber_schedule": cfg.loss.huber_schedule,
        "ss_huber_scale": cfg.loss.huber_scale,
        "ss_huber_c": cfg.loss.huber_c,
        "ss_fp8_base": bool(cfg.performance.fp8_base),
        "ss_fp8_base_unet": bool(cfg.performance.fp8_base_unet),
        "ss_validation_seed": cfg.dataset.validation_seed,
        "ss_validation_split": float(cfg.dataset.validation_split),
        "ss_max_validation_steps": cfg.training.max_validation_steps,
        "ss_validate_every_n_epochs": cfg.training.validate_every_n_epochs,
        "ss_validate_every_n_steps": cfg.training.validate_every_n_steps,
        "ss_resize_interpolation": cfg.dataset.resize_interpolation,
    }

    self.update_metadata(metadata, cfg)  # architecture specific metadata

    if use_user_config:
        # save metadata of multiple datasets
        # NOTE: pack "ss_datasets" value as json one time
        #   or should also pack nested collections as json?
        datasets_metadata = []
        tag_frequency = {}  # merge tag frequency for metadata editor
        dataset_dirs_info = {}  # merge subset dirs for metadata editor

        for dataset in train_dataset_group.datasets:
            is_dreambooth_dataset = isinstance(dataset, DreamBoothDataset)
            dataset_metadata = {
                "is_dreambooth": is_dreambooth_dataset,
                "batch_size_per_device": dataset.batch_size,
                "num_train_images": dataset.num_train_images,  # includes repeating
                "num_reg_images": dataset.num_reg_images,
                "resolution": (dataset.width, dataset.height),
                "enable_bucket": bool(dataset.enable_bucket),
                "min_bucket_reso": dataset.min_bucket_reso,
                "max_bucket_reso": dataset.max_bucket_reso,
                "tag_frequency": dataset.tag_frequency,
                "bucket_info": dataset.bucket_info,
                "resize_interpolation": dataset.resize_interpolation,
            }

            subsets_metadata = []
            for subset in dataset.subsets:
                subset_metadata = {
                    "img_count": subset.img_count,
                    "num_repeats": subset.num_repeats,
                    "color_aug": bool(subset.color_aug),
                    "flip_aug": bool(subset.flip_aug),
                    "random_crop": bool(subset.random_crop),
                    "random_crop_padding_percent": float(getattr(subset, "random_crop_padding_percent", 0.05)),
                    "shuffle_caption": bool(subset.shuffle_caption),
                    "keep_tokens": subset.keep_tokens,
                    "keep_tokens_separator": subset.keep_tokens_separator,
                    "secondary_separator": subset.secondary_separator,
                    "enable_wildcard": bool(subset.enable_wildcard),
                    "caption_prefix": subset.caption_prefix,
                    "caption_suffix": subset.caption_suffix,
                    "resize_interpolation": subset.resize_interpolation,
                }

                image_dir_or_metadata_file = None
                if subset.image_dir:
                    image_dir = os.path.basename(subset.image_dir)
                    subset_metadata["image_dir"] = image_dir
                    image_dir_or_metadata_file = image_dir

                if is_dreambooth_dataset:
                    subset_metadata["class_tokens"] = subset.class_tokens
                    subset_metadata["is_reg"] = subset.is_reg
                    if subset.is_reg:
                        image_dir_or_metadata_file = None  # not merging reg dataset
                else:
                    metadata_file = os.path.basename(subset.metadata_file)
                    subset_metadata["metadata_file"] = metadata_file
                    image_dir_or_metadata_file = metadata_file  # may overwrite

                subsets_metadata.append(subset_metadata)

                # merge dataset dir: not reg subset only
                # TODO update additional-network extension to show detailed dataset config from metadata
                if image_dir_or_metadata_file is not None:
                    # datasets may have a certain dir multiple times
                    v = image_dir_or_metadata_file
                    i = 2
                    while v in dataset_dirs_info:
                        v = image_dir_or_metadata_file + f" ({i})"
                        i += 1
                    image_dir_or_metadata_file = v

                    dataset_dirs_info[image_dir_or_metadata_file] = {
                        "n_repeats": subset.num_repeats,
                        "img_count": subset.img_count,
                    }

            dataset_metadata["subsets"] = subsets_metadata
            datasets_metadata.append(dataset_metadata)

            # merge tag frequency:
            for ds_dir_name, ds_freq_for_dir in dataset.tag_frequency.items():
                # あるディレクトリが複数のdatasetで使用されている場合、一度だけ数える
                # もともと繰り返し回数を指定しているので、キャプション内でのタグの出現回数と、それが学習で何度使われるかは一致しない
                # なので、ここで複数datasetの回数を合算してもあまり意味はない
                if ds_dir_name in tag_frequency:
                    continue
                tag_frequency[ds_dir_name] = ds_freq_for_dir

        metadata["ss_datasets"] = json.dumps(datasets_metadata)
        metadata["ss_tag_frequency"] = json.dumps(tag_frequency)
        metadata["ss_dataset_dirs"] = json.dumps(dataset_dirs_info)
    else:
        # conserving backward compatibility when using train_dataset_dir and reg_dataset_dir
        assert (
                len(train_dataset_group.datasets) == 1
        ), f"There should be a single dataset but {len(train_dataset_group.datasets)} found. This seems to be a bug. / データセットは1個だけ存在するはずですが、実際には{len(train_dataset_group.datasets)}個でした。プログラムのバグかもしれません。"

        dataset = train_dataset_group.datasets[0]

        dataset_dirs_info = {}
        reg_dataset_dirs_info = {}
        if use_dreambooth_method:
            for subset in dataset.subsets:
                info = reg_dataset_dirs_info if subset.is_reg else dataset_dirs_info
                info[os.path.basename(subset.image_dir)] = {"n_repeats": subset.num_repeats,
                                                            "img_count": subset.img_count}
        else:
            for subset in dataset.subsets:
                dataset_dirs_info[os.path.basename(subset.metadata_file)] = {
                    "n_repeats": subset.num_repeats,
                    "img_count": subset.img_count,
                }

        metadata.update(
            {
                "ss_batch_size_per_device": cfg.training.train_batch_size,
                "ss_total_batch_size": total_batch_size,
                "ss_resolution": cfg.dataset.resolution,
                "ss_color_aug": bool(cfg.dataset.color_aug),
                "ss_flip_aug": bool(cfg.dataset.flip_aug),
                "ss_random_crop": bool(cfg.dataset.random_crop),
                "ss_random_crop_padding_percent": float(getattr(cfg.dataset, "random_crop_padding_percent", 0.05)),
                "ss_shuffle_caption": bool(cfg.dataset.shuffle_caption),
                "ss_enable_bucket": bool(dataset.enable_bucket),
                "ss_bucket_no_upscale": bool(dataset.bucket_no_upscale),
                "ss_min_bucket_reso": dataset.min_bucket_reso,
                "ss_max_bucket_reso": dataset.max_bucket_reso,
                "ss_keep_tokens": cfg.dataset.keep_tokens,
                "ss_dataset_dirs": json.dumps(dataset_dirs_info),
                "ss_reg_dataset_dirs": json.dumps(reg_dataset_dirs_info),
                "ss_tag_frequency": json.dumps(dataset.tag_frequency),
                "ss_bucket_info": json.dumps(dataset.bucket_info),
            }
        )

    # add extra args
    if cfg.network.network_args:
        metadata["ss_network_args"] = json.dumps(net_kwargs)

    # model name and hash
    if cfg.model.pretrained_model_name_or_path is not None:
        sd_model_name = cfg.model.pretrained_model_name_or_path
        if os.path.exists(sd_model_name):
            metadata["ss_sd_model_hash"] = model_hash(sd_model_name)
            metadata["ss_new_sd_model_hash"] = calculate_sha256(sd_model_name)
            sd_model_name = os.path.basename(sd_model_name)
        metadata["ss_sd_model_name"] = sd_model_name

    if cfg.model.vae is not None:
        vae_name = cfg.model.vae
        if os.path.exists(vae_name):
            metadata["ss_vae_hash"] = model_hash(vae_name)
            metadata["ss_new_vae_hash"] = calculate_sha256(vae_name)
            vae_name = os.path.basename(vae_name)
        metadata["ss_vae_name"] = vae_name

    metadata = {k: str(v) for k, v in metadata.items()}

    # make minimum metadata for filtering
    minimum_metadata = {}
    for key in SS_METADATA_MINIMUM_KEYS:
        if key in metadata:
            minimum_metadata[key] = metadata[key]

    # calculate steps to skip when resuming or starting from a specific step
    initial_step = 0
    if cfg.training.initial_epoch is not None or cfg.training.initial_step is not None:
        # if initial_epoch or initial_step is specified, steps_from_state is ignored even when resuming
        if steps_from_state is not None:
            logger.warning(
                "steps from the state is ignored because initial_step is specified / initial_stepが指定されているため、stateからのステップ数は無視されます"
            )
        if cfg.training.initial_step is not None:
            initial_step = cfg.training.initial_step
        else:
            # num steps per epoch is calculated by num_processes and gradient_accumulation_steps
            initial_step = (cfg.training.initial_epoch - 1) * math.ceil(
                len(train_dataloader) / accelerator.num_processes / cfg.training.gradient_accumulation_steps
            )
    else:
        # if initial_epoch and initial_step are not specified, steps_from_state is used when resuming
        if steps_from_state is not None:
            initial_step = steps_from_state
            steps_from_state = None

    if initial_step > 0:
        assert (
                cfg.training.max_train_steps > initial_step
        ), f"max_train_steps should be greater than initial step / max_train_stepsは初期ステップより大きい必要があります: {cfg.training.max_train_steps} vs {initial_step}"

    epoch_to_start = 0
    if initial_step > 0:
        if cfg.training.skip_until_initial_step:
            # if skip_until_initial_step is specified, load data and discard it to ensure the same data is used
            if not cfg.saving.resume:
                logger.info(
                    f"initial_step is specified but not resuming. lr scheduler will be started from the beginning / initial_stepが指定されていますがresumeしていないため、lr schedulerは最初から始まります"
                )
            logger.info(f"skipping {initial_step} steps / {initial_step}ステップをスキップします")
            initial_step *= cfg.training.gradient_accumulation_steps

            # set epoch to start to make initial_step less than len(train_dataloader)
            epoch_to_start = initial_step // math.ceil(len(train_dataloader) / cfg.training.gradient_accumulation_steps)
        else:
            # if not, only epoch no is skipped for informative purpose
            epoch_to_start = initial_step // math.ceil(len(train_dataloader) / cfg.training.gradient_accumulation_steps)
            initial_step = 0  # do not skip

    global_step = 0

    noise_scheduler = self.get_noise_scheduler(cfg, accelerator.device)

    # --- LIVE PLOTTER & STATIC PLOT SETUP ---
    timestep_counts = None
    plotter_settings = None

    if is_main_process:
        # --- START: Comprehensive Settings Gathering ---
        # Determine the actual sampler being used
        sampler_type = cfg.timestep.timestep_sampling
        if self.la_sampler is not None:
            if isinstance(self.la_sampler, LogSNRUniformSampler):
                sampler_type = "log_snr_uniform"
            elif isinstance(self.la_sampler, TemperedAdaptiveSampler):
                sampler_type = "tempered_adaptive"
            # The default is mix_adaptive if la_sampler exists

        plotter_settings = {
            "Timestep Sampler": sampler_type,
            "Dynamic Schedule": "Enabled" if cfg.timestep.dynamic_timestep_schedule else "Disabled",
            "Min Timestep": cfg.timestep.min_timestep,
            "Max Timestep": cfg.timestep.max_timestep,
        }

        # Add sampler-specific settings
        if sampler_type == "mix_adaptive":
            plotter_settings.update({
                "Anneal": cfg.timestep.mix_adaptive_anneal,
                "Start/End P": f"{cfg.timestep.mix_adaptive_start_p} -> {cfg.timestep.mix_adaptive_end_p}",
                "Fixed P": cfg.timestep.mix_adaptive_fixed_p,
                "Num Bins": cfg.timestep.mix_adaptive_bins,
                "EMA Beta": cfg.timestep.mix_adaptive_ema_beta,
                "Small T Frac/Cap": f"{cfg.timestep.mix_adaptive_small_t_frac} / {cfg.timestep.mix_adaptive_small_t_cap}",
            })
        elif sampler_type == "tempered_adaptive":
            plotter_settings.update({
                "Num Bins": cfg.timestep.mix_adaptive_bins,
                "EMA Beta": cfg.timestep.mix_adaptive_ema_beta,
                "Temperature": cfg.timestep.mix_adaptive_temperature,
                "Prior Weight": cfg.timestep.mix_adaptive_prior_weight,
                "Min Prob": cfg.timestep.mix_adaptive_min_prob,
                "Warmup Steps": cfg.timestep.mix_adaptive_warmup_steps,
                "Prior Bias": cfg.timestep.mix_adaptive_prior_bias,
                "Entropy Floor": cfg.timestep.mix_adaptive_entropy_floor_ratio,
            })
        elif sampler_type == "gaussian_mid_snr":
            plotter_settings.update({
                "Num Bins": cfg.timestep.mix_adaptive_bins,
                "EMA Beta": cfg.timestep.mix_adaptive_ema_beta,
                "Temperature": cfg.timestep.mix_adaptive_temperature,
                "Min Prob": cfg.timestep.mix_adaptive_min_prob,
                "Entropy Floor": cfg.timestep.mix_adaptive_entropy_floor_ratio,
                "Uniform Mix When Low Entropy": cfg.timestep.mix_adaptive_uniform_mix_when_low_entropy,
                "Prior_Mu": cfg.timestep.mix_adaptive_prior_mu,
                "Prior Sigma": cfg.timestep.mix_adaptive_prior_sigma,
                "Prior Weight": cfg.timestep.mix_adaptive_prior_weight,
                "Warmup Steps": cfg.timestep.mix_adaptive_warmup_steps,
            })
        elif sampler_type == "snr_windowed":
            plotter_settings.update({
                "Num Bins": cfg.timestep.mix_adaptive_bins,
                "EMA Beta": cfg.timestep.mix_adaptive_ema_beta,
                "Temperature": cfg.timestep.mix_adaptive_temperature,
                "Min Prob": cfg.timestep.mix_adaptive_min_prob,
                "Entropy Floor": cfg.timestep.mix_adaptive_entropy_floor_ratio,
                "Uniform Mix": cfg.timestep.mix_adaptive_uniform_mix_when_low_entropy,
                "Center Mu": cfg.timestep.mix_adaptive_center_mu,
                "Half Width": cfg.timestep.mix_adaptive_half_width,
                "Widen To": cfg.timestep.mix_adaptive_widen_to,
                "Total Widen Steps": cfg.timestep.mix_adaptive_max_train_steps,
                "Cap Max T": cfg.timestep.mix_adaptive_cap_max_t,
            })
        elif sampler_type not in ["uniform", "log_snr_uniform"]:  # Legacy shifted sampler
            plotter_settings.update({
                "Shift": cfg.timestep.discrete_flow_shift,
                "Sigmoid Scale": cfg.timestep.sigmoid_scale,
            })

        # Setup for the live interactive plotter
        if cfg.logging.live_plot_port is not None:

            # Step 1: Find the script to run.
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            plotter_script_path = os.path.join(project_root, "tools", "visualization", "live_plotter.py")

            # Step 2: Check if the script actually exists. If not, disable the feature and continue.
            if not os.path.exists(plotter_script_path):
                logger.error(f"live_plotter.py not found at {plotter_script_path}. Live plotter disabled.")
            else:
                # Step 3: Launch live_plotter.py if it's not already running.
                if self.live_plotter_process is None or self.live_plotter_process.poll() is not None:
                    logger.info(f"Launching live plotter server on port {cfg.logging.live_plot_port}")
                    self.live_plotter_process = subprocess.Popen(
                        [sys.executable, plotter_script_path, "--port", str(cfg.logging.live_plot_port)],
                        stdin=subprocess.PIPE,
                    )

                # Step 4: Send the initial "handshake" data.
                reset_str = "RESET::\n"
                alphas_cumprod_np = noise_scheduler.alphas_cumprod.cpu().numpy()
                schedule_str = f"SCHEDULE::{','.join(map(str, alphas_cumprod_np))}\n"
                settings_str = f"SETTINGS::{json.dumps(plotter_settings)}\n"

                try:
                    self.live_plotter_process.stdin.write(reset_str.encode('utf-8'))
                    self.live_plotter_process.stdin.write(schedule_str.encode('utf-8'))
                    self.live_plotter_process.stdin.write(settings_str.encode('utf-8'))
                    self.live_plotter_process.stdin.flush()
                except (BrokenPipeError, OSError):
                    logger.error("Failed to send data to live plotter. It may have crashed.")
                    self.live_plotter_process = None

        # Setup for saving static plot images
        if cfg.logging.log_timestep_distribution_every_n_steps is not None:
            timestep_counts = np.zeros(noise_scheduler.config.num_train_timesteps, dtype=np.int64)

    # --- Custom Timestep Sampler Initialization ---
    # Inject sampler when specified. This block creates the sampler object.
    if cfg.timestep.timestep_sampling:
        if cfg.timestep.timestep_sampling == "log_snr_uniform":
            accelerator.print("Initializing LogSNRUniformSampler.")
            self.la_sampler = LogSNRUniformSampler(noise_scheduler, noise_scheduler.config.num_train_timesteps)
            cfg.timestep.timestep_sampling = "mix_adaptive"
        elif cfg.timestep.timestep_sampling == "tempered_adaptive":
            accelerator.print("Initializing TemperedAdaptiveSampler.")
            self.la_sampler = TemperedAdaptiveSampler(
                noise_scheduler,
                num_bins=cfg.timestep.mix_adaptive_bins,
                ema_beta=cfg.timestep.mix_adaptive_ema_beta,
                temperature=cfg.timestep.mix_adaptive_temperature,
                prior_weight=cfg.timestep.mix_adaptive_prior_weight,
                min_prob=cfg.timestep.mix_adaptive_min_prob,
                warmup_steps=cfg.timestep.mix_adaptive_warmup_steps,
                prior_bias=cfg.timestep.mix_adaptive_prior_bias,
                entropy_floor=cfg.timestep.mix_adaptive_entropy_floor_ratio,
            )
            cfg.timestep.timestep_sampling = "mix_adaptive"
        elif cfg.timestep.timestep_sampling == "gaussian_mid_snr":
            accelerator.print("Initializing GaussianMidSNRSampler.")
            self.la_sampler = GaussianMidSNRSampler(
                noise_scheduler,
                num_bins=cfg.timestep.mix_adaptive_bins,
                ema_beta=cfg.timestep.mix_adaptive_ema_beta,
                temperature=cfg.timestep.mix_adaptive_temperature,
                min_prob=cfg.timestep.mix_adaptive_min_prob,
                entropy_floor=cfg.timestep.mix_adaptive_entropy_floor_ratio,
                # uniform_mix_when_low_entropy=getattr(args, "uniform_mix_when_low_entropy", 0.1),
                prior_mu=cfg.timestep.mix_adaptive_prior_mu,
                prior_sigma=cfg.timestep.mix_adaptive_prior_sigma,
                prior_weight=cfg.timestep.mix_adaptive_prior_weight,
                warmup_steps=cfg.timestep.mix_adaptive_warmup_steps,
            )
            cfg.timestep.timestep_sampling = "mix_adaptive"
        elif cfg.timestep.timestep_sampling == "snr_windowed":
            accelerator.print("Initializing SNRWindowedSampler.")
            self.la_sampler = SNRWindowedSampler(
                noise_scheduler,
                num_bins=cfg.timestep.mix_adaptive_bins,
                ema_beta=cfg.timestep.mix_adaptive_ema_beta,
                temperature=cfg.timestep.mix_adaptive_temperature,
                min_prob=cfg.timestep.mix_adaptive_min_prob,
                entropy_floor=cfg.timestep.mix_adaptive_entropy_floor_ratio,

                center_mu=cfg.timestep.mix_adaptive_center_mu,
                half_width=cfg.timestep.mix_adaptive_half_width,
                widen_to=cfg.timestep.mix_adaptive_widen_to,
                total_widen_steps=cfg.timestep.mix_adaptive_max_train_steps,
                cap_max_t=cfg.timestep.mix_adaptive_cap_max_t,
            )
        elif cfg.timestep.timestep_sampling == "snr_windowed":
            accelerator.print("Initializing SNRWindowedSampler.")
            self.la_sampler = SNRWindowedSampler(
                noise_scheduler,
                num_bins=cfg.timestep.mix_adaptive_bins,
                ema_beta=cfg.timestep.mix_adaptive_ema_beta,
                temperature=cfg.timestep.mix_adaptive_temperature,
                min_prob=cfg.timestep.mix_adaptive_min_prob,
                entropy_floor=cfg.timestep.mix_adaptive_entropy_floor_ratio,

                center_mu=cfg.timestep.mix_adaptive_center_mu,
                half_width=cfg.timestep.mix_adaptive_half_width,
                widen_to=cfg.timestep.mix_adaptive_widen_to,
                total_widen_steps=cfg.timestep.mix_adaptive_max_train_steps,
                cap_max_t=cfg.timestep.mix_adaptive_cap_max_t,
            )
            cfg.timestep.timestep_sampling = "mix_adaptive"

        elif cfg.timestep.timestep_sampling == "mix_adaptive":
            accelerator.print("Initializing LossAwareTimestepSampler.")
            self.la_sampler = LossAwareTimestepSampler(
                num_train_timesteps=noise_scheduler.config.num_train_timesteps,
                num_bins=cfg.timestep.mix_adaptive_bins,
                ema_beta=cfg.timestep.mix_adaptive_ema_beta,
                small_t_frac=cfg.timestep.mix_adaptive_small_t_frac,
                small_t_cap=cfg.timestep.mix_adaptive_small_t_cap,
                start_p=cfg.timestep.mix_adaptive_start_p,
                end_p=cfg.timestep.mix_adaptive_end_p,
                anneal=cfg.timestep.mix_adaptive_anneal,
                fixed_p=cfg.timestep.mix_adaptive_fixed_p,
            )
            # No need to set args.la_sampler, we use self.la_sampler

        if cfg.timestep.timestep_sampling == "sigma" or cfg.timestep.timestep_sampling == "uniform":
            self.la_sampler = None
            cfg.timestep.timestep_sampling = "uniform"
            if cfg.timestep.timestep_sampling == "sigma":
                logger.warning("sigma sampling is not supported yet, using uniform sampling")
        elif cfg.timestep.timestep_sampling == "shift":
            self.la_sampler = None
            # shift sampling is handled in get_noise_noisy_latents_and_timesteps

    edm2_model, edm2_optimizer, edm2_lr_scheduler = prepare_edm2_loss_weighting(cfg.loss, cfg.training, noise_scheduler,
                                                                                accelerator)

    init_trackers(accelerator, cfg, "network_train")

    loss_recorder = EMARecorder()
    val_loss_recorder = EMARecorder()

    if cfg.loss.edm2_loss_weighting:
        loss_scaled_recorder = EMARecorder()

    del train_dataset_group
    if val_dataset_group is not None:
        del val_dataset_group

    # callback for step start
    if hasattr(accelerator.unwrap_model(network), "on_step_start"):
        on_step_start_for_network = accelerator.unwrap_model(network).on_step_start
    else:
        on_step_start_for_network = lambda *args, **kwargs: None

    # function for saving/removing
    def save_model(ckpt_name, unwrapped_nw, steps, epoch_no, force_sync_upload=False, dtype_override=None):
        os.makedirs(cfg.saving.output_dir, exist_ok=True)
        ckpt_file = os.path.join(cfg.saving.output_dir, ckpt_name)

        accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
        metadata["ss_training_finished_at"] = str(time.time())
        metadata["ss_steps"] = str(steps)
        metadata["ss_epoch"] = str(epoch_no)

        metadata_to_save = minimum_metadata if cfg.saving.no_metadata else metadata
        sai_metadata = self.get_sai_model_spec(cfg)
        metadata_to_save.update(sai_metadata)

        unwrapped_nw.save_weights(ckpt_file, dtype_override or save_dtype, metadata_to_save)
        if cfg.huggingface.huggingface_repo_id is not None:
            huggingface_util.upload(cfg.huggingface, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)

    def remove_model(old_ckpt_name):
        old_ckpt_file = os.path.join(cfg.saving.output_dir, old_ckpt_name)
        if os.path.exists(old_ckpt_file):
            accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
            os.remove(old_ckpt_file)

    # if text_encoder is not needed for training, delete it to save memory.
    # TODO this can be automated after SDXL sample prompt cache is implemented
    if self.is_text_encoder_not_needed_for_training(cfg):
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
    current_global_step_loss_scaled = 0.0 if cfg.loss.edm2_loss_weighting else None
    average_loss_scaled = 0.0 if cfg.loss.edm2_loss_weighting else None
    avr_loss = 0.0
    accumulation_counter = 0

    # For --sample_at_first
    if sample_images_check(cfg.sampling, 0, global_step) or calculate_val_loss_check(cfg.training, global_step, 0,
                                                                                     val_dataloader, train_dataloader):
        # Switch network to eval mode
        accelerator.unwrap_model(network).eval()
        optimizer_eval_fn()
        self.sample_images(accelerator, cfg, 0, global_step, accelerator.device, vae, tokenizers, text_encoder, unet)
        if calculate_val_loss_check(cfg.training, global_step, 0, val_dataloader, train_dataloader):
            current_val_loss, average_val_loss, val_logs = self.calculate_val_loss(
                global_step, 0, train_dataloader, val_loss_recorder, val_dataloader,
                cyclic_val_dataloader, network, tokenize_strategy,
                text_encoders, text_encoding_strategy, unet, vae, noise_scheduler,
                vae_dtype, weight_dtype, accelerator, cfg, 0, None, train_text_encoder)
        # Switch network to train mode
        optimizer_train_fn()
        accelerator.unwrap_model(network).train()

    if plot_edm2_loss_weighting_check(cfg.loss, cfg.training, global_step):
        plot_edm2_loss_weighting(cfg.loss, cfg.saving.output_name, global_step, edm2_model, 1000, accelerator.device)

    is_tracking = len(accelerator.trackers) > 0
    if is_tracking:
        logs = self.generate_step_logs(
            cfg,
            current_global_step_loss,
            avr_loss,
            lr_scheduler,
            lr_descriptions,
            optimizer,
            keys_scaled,
            mean_norm,
            maximum_norm,
            mean_grad_norm,
            mean_combined_norm,
            edm2_lr_scheduler,
            current_global_step_loss_scaled,
            average_loss_scaled,
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

    # --- Add this block for Dynamic Timestep Schedule ---
    # Parse the schedule from the command-line argument string
    dynamic_timestep_schedule = ast.literal_eval(
        cfg.timestep.dynamic_timestep_schedule) if cfg.timestep.dynamic_timestep_schedule else None
    if dynamic_timestep_schedule:
        # Sort the schedule by step number to be safe
        dynamic_timestep_schedule.sort(key=lambda x: x[0])
        accelerator.print(f"Using dynamic timestep schedule: {dynamic_timestep_schedule}")

    # Initialize the current range with the defaults
    current_min_timestep = 0 if cfg.timestep.min_timestep is None else cfg.timestep.min_timestep
    current_max_timestep = noise_scheduler.config.num_train_timesteps if cfg.timestep.max_timestep is None else cfg.timestep.max_timestep
    # ---------------------------------------------------

    clean_memory_on_device(accelerator.device)

    progress_bar = tqdm(
        range(cfg.training.max_train_steps - initial_step), smoothing=0, disable=not accelerator.is_local_main_process,
        desc="steps"
    )

    for epoch in range(epoch_to_start, num_train_epochs):
        current_epoch.value = epoch + 1
        accelerator.print(f"\nepoch {current_epoch.value}/{num_train_epochs}\n")

        metadata["ss_epoch"] = str(current_epoch.value)

        accelerator.unwrap_model(network).on_epoch_start(text_encoder, unet)  # network.train() is called here

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
                on_step_start_for_network(text_encoder, unet)

                accumulation_counter += 1

                # preprocess batch for each model
                self.on_step_start(cfg, accelerator, network, text_encoders, unet, batch, weight_dtype, is_train=True)

                loss, pre_scaling_loss, loss_scaled, timesteps = self.process_batch(
                    batch,
                    text_encoders,
                    unet,
                    network,
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
                    self.all_reduce_network(accelerator, network)  # sync DDP grad manually
                    if cfg.optimizer.max_grad_norm != 0.0:
                        params_to_clip = accelerator.unwrap_model(network).get_trainable_params()
                        accelerator.clip_grad_norm_(params_to_clip, cfg.optimizer.max_grad_norm)

                    # if hasattr(network, "update_grad_norms"):
                    #    network.update_grad_norms()
                    # if hasattr(network, "update_norms"):
                    #    network.update_norms()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if cfg.loss.edm2_loss_weighting:
                    edm2_optimizer.step()
                    edm2_lr_scheduler.step()
                    # swap to pre_scaling_loss for logging
                    edm2_optimizer.zero_grad(set_to_none=True)

            if cfg.network.scale_weight_norms and accelerator.sync_gradients:
                keys_scaled, mean_norm, maximum_norm = accelerator.unwrap_model(network).apply_max_norm_regularization(
                    cfg.network.scale_weight_norms, accelerator.device
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

                if (sample_images_check(cfg.sampling, None, global_step) or
                        calculate_val_loss_check(cfg.training, global_step, step, val_dataloader, train_dataloader) or
                        cfg.saving.save_every_n_steps is not None and global_step % cfg.saving.save_every_n_steps == 0):

                    accelerator.unwrap_model(network).eval()
                    optimizer_eval_fn()
                    self.sample_images(
                        accelerator, cfg, None, global_step, accelerator.device, vae, tokenizers, text_encoder, unet
                    )

                    if calculate_val_loss_check(cfg.training, global_step, step, val_dataloader, train_dataloader):
                        current_val_loss, average_val_loss, val_logs = self.calculate_val_loss(global_step, step,
                                                                                               skipped_dataloader or train_dataloader,
                                                                                               val_loss_recorder,
                                                                                               val_dataloader,
                                                                                               cyclic_val_dataloader,
                                                                                               network,
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
                    if cfg.saving.save_every_n_steps is not None and global_step % cfg.saving.save_every_n_steps == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            ckpt_name = get_step_ckpt_name(cfg.saving, "." + cfg.saving.save_model_as, global_step)
                            save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch)

                            if cfg.loss.edm2_loss_weighting:
                                loss_weights_ckpt_name = get_step_ckpt_name(cfg.saving, "." + cfg.saving.save_model_as,
                                                                            global_step, "_edm2_loss_weights")
                                save_model(loss_weights_ckpt_name, accelerator.unwrap_model(edm2_model), global_step,
                                           epoch, dtype_override=torch.float32)

                            if cfg.saving.save_state:
                                save_and_remove_state_stepwise(cfg.saving, accelerator, global_step)

                            remove_step_no = get_remove_step_no(cfg.saving, global_step)
                            if remove_step_no is not None:
                                remove_ckpt_name = get_step_ckpt_name(cfg.saving, "." + cfg.saving.save_model_as,
                                                                      remove_step_no)
                                remove_model(remove_ckpt_name)

                                if cfg.loss.edm2_loss_weighting:
                                    remove_loss_weights_ckpt_name = get_step_ckpt_name(cfg.saving,
                                                                                       "." + cfg.saving.save_model_as,
                                                                                       remove_step_no,
                                                                                       "_edm2_loss_weights")
                                    remove_model(remove_loss_weights_ckpt_name)

                    if plot_edm2_loss_weighting_check(cfg.loss, cfg.training, global_step):
                        plot_edm2_loss_weighting(cfg.loss, cfg.saving.output_name, global_step, edm2_model, 1000,
                                                 accelerator.device)
                    optimizer_train_fn()
                    accelerator.unwrap_model(network).train()

            current_global_step_loss += loss.detach().item()
            if cfg.loss.edm2_loss_weighting:
                current_global_step_loss_scaled += loss_scaled.detach().item()
            else:
                current_global_step_loss_scaled = None

            if accelerator.sync_gradients:
                loss_recorder.add(current_global_step_loss / accumulation_counter)
                if cfg.loss.edm2_loss_weighting:
                    loss_scaled_recorder.add(current_global_step_loss_scaled / accumulation_counter)
                avr_loss: float = loss_recorder.average
                logs = {"avr_loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**{**max_mean_logs, **logs})

                if is_tracking:
                    current_global_step_loss = (current_global_step_loss / accumulation_counter)
                    if cfg.loss.edm2_loss_weighting:
                        current_global_step_loss_scaled = (current_global_step_loss_scaled / accumulation_counter)
                        average_loss_scaled: float = loss_scaled_recorder.average
                    else:
                        current_global_step_loss_scaled = None
                        average_loss_scaled = None

                    logs = self.generate_step_logs(
                        cfg,
                        current_global_step_loss,
                        avr_loss,
                        lr_scheduler,
                        lr_descriptions,
                        optimizer,
                        keys_scaled,
                        mean_norm,
                        maximum_norm,
                        mean_grad_norm,
                        mean_combined_norm,
                        edm2_lr_scheduler,
                        current_global_step_loss_scaled,
                        average_loss_scaled,
                        current_val_loss=current_val_loss,
                        average_val_loss=average_val_loss,
                        timesteps=timesteps
                    )
                    self.step_logging(accelerator, logs, global_step, epoch + 1)

                current_global_step_loss = 0.0

                if cfg.loss.edm2_loss_weighting:
                    current_global_step_loss_scaled = 0.0

                accumulation_counter = 0

                # --- LIVE PLOTTER & STATIC PLOT UPDATE ---
                if is_main_process:
                    timesteps_np = timesteps.cpu().numpy()

                    # Send data to the live plotter
                    if self.live_plotter_process and self.live_plotter_process.poll() is None:
                        try:
                            self.live_plotter_process.stdin.write(
                                f"{','.join(map(str, timesteps_np))}\n".encode('utf-8'))
                            self.live_plotter_process.stdin.flush()
                        except (BrokenPipeError, OSError):
                            logger.error("Live plotter connection lost.")
                            self.live_plotter_process = None

                    # Update counts and save static plot if needed
                    if timestep_counts is not None:
                        unique, counts = np.unique(timesteps_np, return_counts=True)
                        timestep_counts[unique] += counts

                        if global_step % cfg.logging.log_timestep_distribution_every_n_steps == 0:
                            self.save_timestep_distribution_plot(cfg, global_step, timestep_counts, plotter_settings)

            if global_step >= cfg.training.max_train_steps:
                break

        # END OF EPOCH
        if is_tracking:
            logs = {"loss/epoch_average": loss_recorder.average}
            accelerator.log(logs, step=global_step)

        accelerator.wait_for_everyone()

        if (sample_images_check(cfg.sampling, current_epoch.value, global_step) or
                cfg.saving.save_every_n_epochs is not None):

            # 指定エポックごとにモデルを保存
            optimizer_eval_fn()
            accelerator.unwrap_model(network).eval()
            if cfg.saving.save_every_n_epochs is not None:
                saving = current_epoch.value % cfg.saving.save_every_n_epochs == 0 and current_epoch.value < num_train_epochs
                if is_main_process and saving:
                    ckpt_name = get_epoch_ckpt_name(cfg.saving, "." + cfg.saving.save_model_as, current_epoch.value)
                    save_model(ckpt_name, accelerator.unwrap_model(network), global_step, current_epoch.value)

                    if cfg.loss.edm2_loss_weighting:
                        loss_weights_ckpt_name = get_epoch_ckpt_name(cfg.saving, "." + cfg.saving.save_model_as,
                                                                     current_epoch.value, "_edm2_loss_weights")
                        save_model(loss_weights_ckpt_name, accelerator.unwrap_model(edm2_model), global_step,
                                   current_epoch.value, dtype_override=torch.float32)

                    remove_epoch_no = get_remove_epoch_no(cfg.saving, current_epoch.value)
                    if remove_epoch_no is not None:
                        remove_ckpt_name = get_epoch_ckpt_name(cfg.saving, "." + cfg.saving.save_model_as,
                                                               remove_epoch_no)
                        remove_model(remove_ckpt_name)

                        if cfg.loss.edm2_loss_weighting:
                            remove_loss_weights_ckpt_name = get_epoch_ckpt_name(cfg.saving,
                                                                                "." + cfg.saving.save_model_as,
                                                                                remove_epoch_no, "_edm2_loss_weights")
                            remove_model(remove_loss_weights_ckpt_name)

                    if cfg.saving.save_state:
                        save_and_remove_state_on_epoch_end(cfg.saving, accelerator, current_epoch.value)

            self.sample_images(accelerator, cfg, current_epoch.value, global_step, accelerator.device, vae, tokenizers,
                               text_encoder, unet)
            progress_bar.unpause()
            optimizer_train_fn()
            accelerator.unwrap_model(network).train()

        # end of epoch

    # metadata["ss_epoch"] = str(num_train_epochs)
    metadata["ss_training_finished_at"] = str(time.time())

    if is_main_process:
        network = accelerator.unwrap_model(network)

    accelerator.end_training()
    optimizer_eval_fn()

    if is_main_process and (cfg.saving.save_state or cfg.saving.save_state_on_train_end):
        save_state_on_train_end(cfg.saving, accelerator)

    if is_main_process:
        ckpt_name = get_last_ckpt_name(cfg.saving, "." + cfg.saving.save_model_as)
        save_model(ckpt_name, network, global_step, num_train_epochs, force_sync_upload=True)

        if cfg.loss.edm2_loss_weighting:
            loss_weights_ckpt_name = get_last_ckpt_name(cfg.saving, "." + cfg.saving.save_model_as,
                                                        "_edm2_loss_weights")
            save_model(loss_weights_ckpt_name, accelerator.unwrap_model(edm2_model), global_step, num_train_epochs,
                       force_sync_upload=True, dtype_override=torch.float32)

    logger.info("model saved.")
