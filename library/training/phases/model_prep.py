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
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from library.adapters.lora_utils import resolve_adapter_kwargs

if TYPE_CHECKING:
    from accelerate import Accelerator
    from library.strategies.base.training import TrainingStrategy

logger = logging.getLogger(__name__)


def create_adapter(
    cfg: Any,
    vae: nn.Module,
    text_encoder: Any,  # TODO: Migrate adapter APIs to use text_encoders list instead
    unet: nn.Module,
    accelerator: Accelerator,
    strategies: TrainingStrategy,
    text_encoders: list[nn.Module],
    weight_dtype: torch.dtype,
) -> tuple[nn.Module | None, dict]:
    """
    Create and configure the adapter (LoRA/LyCORIS) for training.

    Args:
        cfg: Config object with PEFT settings
        vae: VAE model
        text_encoder: Text encoder(s) - original reference
        unet: UNet model
        accelerator: Accelerator instance
        strategies: Training strategy for post-processing
        text_encoders: List of text encoders
        weight_dtype: Weight dtype for merging

    Returns:
        Tuple of (adapter, net_kwargs), or (None, {}) if creation failed
    """
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
        logger.warning("Adapter creation returned None - check adapter module configuration")
        return None, {}

    # Prepare adapter if method exists
    if hasattr(adapter, "prepare_adapter"):
        adapter.prepare_adapter(cfg)

    if cfg.peft.scale_weight_norms and not hasattr(adapter, "apply_max_norm_regularization"):
        logger.warning("warning: scale_weight_norms is specified but the peft does not support it")
        cfg.peft.scale_weight_norms = False

    strategies.post_process_adapter(cfg, accelerator, adapter, text_encoders, unet)

    # Apply adapter to unet and text_encoder
    train_unet = strategies.is_train_unet(cfg)
    train_text_encoder = strategies.is_train_text_encoder(cfg)
    adapter.apply_to(text_encoder, unet, train_text_encoder, train_unet)

    # Load weights if specified
    if cfg.peft.adapter_weights is not None:
        info = adapter.load_weights(cfg.peft.adapter_weights)
        accelerator.print(f"load peft weights from {cfg.peft.adapter_weights}: {info}")

    return adapter, net_kwargs


def configure_precision(
    cfg: Any,
    unet: nn.Module,
    text_encoders: list[nn.Module],
    adapter: nn.Module,
    weight_dtype: torch.dtype,
    accelerator: Accelerator,
    strategies: TrainingStrategy,
) -> tuple[torch.dtype, torch.dtype]:
    """
    Configure precision settings for UNet, text encoders, and adapter.

    Handles fp8, fp16, bf16 precision configurations.

    Args:
        cfg: Config object with precision settings
        unet: UNet model
        text_encoders: List of text encoder models
        adapter: Adapter module
        weight_dtype: Base weight dtype
        accelerator: Accelerator instance
        strategies: Training strategy

    Returns:
        Tuple of (unet_weight_dtype, te_weight_dtype)
    """
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

    return unet_weight_dtype, te_weight_dtype
