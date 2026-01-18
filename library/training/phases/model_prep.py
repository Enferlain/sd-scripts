"""
Model Preparation Phase - Adapter creation and precision configuration.

These functions handle model loading, adapter creation, and precision
settings for training.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from typing import TYPE_CHECKING

import torch
from torch import nn

from library.adapters.lora_utils import resolve_adapter_kwargs

if TYPE_CHECKING:
    from library.training.trainers.peft_trainer import PeftTrainer

from library.utils.common_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def prepare_models(trainer: PeftTrainer) -> None:
    """Phase 3: Create adapter and configure precision.

    Updates trainer.adapter, trainer.net_kwargs, trainer.unet_weight_dtype,
    and trainer.te_weight_dtype.

    Args:
        trainer: PeftTrainer instance
    """
    create_adapter(trainer)
    configure_precision(trainer)


def create_adapter(trainer: PeftTrainer) -> None:
    """Create and configure the adapter (LoRA/LyCORIS) for training.

    Updates trainer.adapter and trainer.net_kwargs.

    Args:
        trainer: PeftTrainer instance
    """
    cfg = trainer.cfg
    accelerator = trainer.accelerator
    vae = trainer.vae
    unet = trainer.unet
    text_encoder = trainer._text_encoder  # TODO: Original reference for adapter API
    text_encoders = trainer.text_encoders
    weight_dtype = trainer.weight_dtype

    # Import adapter module dynamically
    sys.path.append(os.path.dirname(__file__))  # TODO: Maybe leftover from different location adapters, idk investigate
    accelerator.print("import peft module:", cfg.peft.adapter_module)
    adapter_module = importlib.import_module(cfg.peft.adapter_module)

    # Merge base weights if specified
    if cfg.peft.base_weights is not None:
        for i, weight_path in enumerate(cfg.peft.base_weights):
            if cfg.peft.base_weights_multiplier is None or len(cfg.peft.base_weights_multiplier) <= i:
                multiplier = 1.0
            else:
                multiplier = cfg.peft.base_weights_multiplier[i]

            accelerator.print(f"merging module: {weight_path} with multiplier {multiplier}")

            module, weights_sd = adapter_module.create_adapter_from_weights(
                multiplier, weight_path, vae, text_encoder, unet, for_inference=True
            )
            module.merge_to(text_encoder, unet, weights_sd, weight_dtype, accelerator.device if cfg.performance.memory.lowram else "cpu")

        accelerator.print(f"all weights merged: {', '.join(cfg.peft.base_weights)}")

    # Prepare adapter kwargs
    net_kwargs = {}
    if cfg.peft.adapter_args is not None:
        for net_arg in cfg.peft.adapter_args:
            key, value = net_arg.split("=", 1)
            net_kwargs[key] = value

    # Resolve explicit LoRA fields from config to kwargs
    resolve_adapter_kwargs(cfg.peft, net_kwargs)

    # Create adapter
    if cfg.peft.adapter_rank_from_weights:
        adapter, _ = adapter_module.create_adapter_from_weights(1, cfg.peft.adapter_weights, vae, text_encoder, unet, **net_kwargs)
    else:
        if "dropout" not in net_kwargs:
            # workaround for LyCORIS
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
        raise RuntimeError("Adapter creation returned None - check adapter module configuration")

    # Prepare adapter if method exists
    if hasattr(adapter, "prepare_adapter"):
        adapter.prepare_adapter(cfg)

    if cfg.peft.scale_weight_norms and not hasattr(adapter, "apply_max_norm_regularization"):
        logger.warning("warning: scale_weight_norms is specified but the peft does not support it")
        cfg.peft.scale_weight_norms = False

    trainer.strategies.post_process_adapter(cfg, accelerator, adapter, text_encoders, unet)

    # Apply adapter to unet and text_encoder
    train_unet = trainer.strategies.is_train_unet(cfg)
    train_text_encoder = trainer.strategies.is_train_text_encoder(cfg)
    adapter.apply_to(text_encoder, unet, train_text_encoder, train_unet)

    # Load weights if specified
    if cfg.peft.adapter_weights is not None:
        info = adapter.load_weights(cfg.peft.adapter_weights)
        accelerator.print(f"load peft weights from {cfg.peft.adapter_weights}: {info}")

    trainer.adapter = adapter
    trainer.net_kwargs = net_kwargs


def configure_precision(trainer: PeftTrainer) -> None:
    """Configure precision settings for UNet, text encoders, and adapter.

    Updates trainer.unet_weight_dtype and trainer.te_weight_dtype.

    Args:
        trainer: PeftTrainer instance
    """
    cfg = trainer.cfg
    unet = trainer.unet
    text_encoders = trainer.text_encoders
    adapter = trainer.adapter
    weight_dtype = trainer.weight_dtype
    accelerator = trainer.accelerator
    strategies = trainer.strategies

    # Full fp16/bf16 training - cast entire adapter
    if cfg.performance.precision.full_fp16:
        accelerator.print("enable full fp16 training.")
        adapter.to(weight_dtype)
    elif cfg.performance.precision.full_bf16:
        accelerator.print("enable full bf16 training.")
        adapter.to(weight_dtype)

    unet_weight_dtype = te_weight_dtype = weight_dtype

    # FP8 support for base model
    if cfg.performance.precision.fp8_base or cfg.performance.precision.fp8_base_unet:
        assert torch.__version__ >= "2.1.0", "fp8_base requires torch>=2.1.0"
        accelerator.print("enable fp8 training for U-Net.")
        unet_weight_dtype = torch.float8_e4m3fn

        if not cfg.performance.precision.fp8_base_unet:
            accelerator.print("enable fp8 training for Text Encoder.")
        te_weight_dtype = weight_dtype if cfg.performance.precision.fp8_base_unet else torch.float8_e4m3fn

        logger.info(f"set U-Net weight dtype to {unet_weight_dtype}")
        unet.to(dtype=unet_weight_dtype)

    # Configure UNet
    unet.requires_grad_(False)
    if strategies.cast_unet(cfg):
        unet.to(dtype=unet_weight_dtype)

    # Configure text encoders
    for i, t_enc in enumerate(text_encoders):
        t_enc.requires_grad_(False)

        if t_enc.device.type != "cpu" and strategies.cast_text_encoder(cfg):
            t_enc.to(dtype=te_weight_dtype)

            # nn.Embedding doesn't support FP8
            if te_weight_dtype != weight_dtype:
                strategies.prepare_text_encoder_fp8(i, t_enc, te_weight_dtype, weight_dtype)

    trainer.unet_weight_dtype = unet_weight_dtype
    trainer.te_weight_dtype = te_weight_dtype
