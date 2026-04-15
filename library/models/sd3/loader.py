"""SD3 model loading helpers."""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Literal, cast

import torch
from accelerate import init_empty_weights
from transformers import CLIPTextConfig, CLIPTextModelWithProjection, T5Config, T5EncoderModel

from library.config.dataclasses.data import CachingConfig
from library.config.dataclasses.model import ModelConfig
from library.config.dataclasses.performance import MemoryConfig, PrecisionConfig
from library.models.sd3.mmdit import MMDiT, SD3Params, create_sd3_mmdit
from library.models.sd3.vae import SDVAE
from library.models.runtime_utils import set_padding_mode_for_vae_conv2d_modules
from library.utils.device_utils import clean_memory_on_device
from library.utils.safetensors_utils import load_safetensors
from library.utils.torch_utils import match_mixed_precision


logger = logging.getLogger(__name__)


def _pop_prefixed_state_dict(state_dict: dict[str, torch.Tensor], prefix: str):
    extracted: dict[str, torch.Tensor] = {}
    for key in list(state_dict.keys()):
        if key.startswith(prefix):
            extracted[key[len(prefix) :]] = state_dict.pop(key)
    return extracted


def analyze_state_dict_state(state_dict: dict[str, torch.Tensor], prefix: str = ""):
    logger.info("Analyzing state dict state...")

    patch_size = state_dict[f"{prefix}x_embedder.proj.weight"].shape[2]
    depth = state_dict[f"{prefix}x_embedder.proj.weight"].shape[0] // 64
    num_patches = state_dict[f"{prefix}pos_embed"].shape[1]
    pos_embed_max_size = round(math.sqrt(num_patches))
    adm_in_channels = state_dict[f"{prefix}y_embedder.mlp.0.weight"].shape[1]
    context_shape = state_dict[f"{prefix}context_embedder.weight"].shape
    qk_norm = "rms" if f"{prefix}joint_blocks.0.context_block.attn.ln_k.weight" in state_dict else None

    x_block_self_attn_layers: list[int] = []
    re_attn = re.compile(r"\.(\d+)\.x_block\.attn2\.ln_k\.weight")
    for key in state_dict:
        match = re_attn.search(key)
        if match:
            x_block_self_attn_layers.append(int(match.group(1)))

    context_embedder_in_features = context_shape[1]
    context_embedder_out_features = context_shape[0]

    if qk_norm is not None:
        model_type = "5-large" if len(x_block_self_attn_layers) == 0 else "5-medium"
    else:
        model_type = "medium"

    params = SD3Params(
        patch_size=patch_size,
        depth=depth,
        num_patches=num_patches,
        pos_embed_max_size=pos_embed_max_size,
        adm_in_channels=adm_in_channels,
        qk_norm=qk_norm,
        x_block_self_attn_layers=x_block_self_attn_layers,
        context_embedder_in_features=context_embedder_in_features,
        context_embedder_out_features=context_embedder_out_features,
        model_type=model_type,
    )
    logger.info("Analyzed state dict state: %s", params)
    return params


def load_mmdit(
    state_dict: dict[str, torch.Tensor],
    dtype: torch.dtype | None,
    device: str | torch.device,
    attn_mode: str = "torch",
) -> MMDiT:
    mmdit_sd = _pop_prefixed_state_dict(state_dict, "model.diffusion_model.")

    logger.info("Building MMDiT")
    params = analyze_state_dict_state(mmdit_sd)
    with init_empty_weights():
        mmdit = create_sd3_mmdit(params, attn_mode)

    logger.info("Loading state dict...")
    info = mmdit.load_state_dict(mmdit_sd, strict=False, assign=True)
    logger.info("Loaded MMDiT: %s", info)
    if dtype is not None:
        mmdit = mmdit.to(dtype=dtype)
    return mmdit.to(device)


def _materialize_meta_module(
    module: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    device: str | torch.device,
) -> tuple[list[str], list[str]]:
    """Materialize meta-initialized modules on the target device before loading weights.

    Hugging Face text models may keep non-persistent buffers off the state dict. If the module
    stays on the meta device until after `load_state_dict(..., assign=True)`, a later `.to(device)`
    will fail when those buffers remain meta. `to_empty()` gives every tensor real storage first,
    then `load_state_dict(..., assign=True)` can still swap in the checkpoint tensors without
    leaving those buffers on the meta device.
    """
    module.to_empty(device=device)
    info = module.load_state_dict(state_dict, strict=False, assign=True)
    return info.missing_keys, info.unexpected_keys


def load_clip_l(
    clip_l_path: str | None,
    dtype: torch.dtype | None,
    device: str | torch.device,
    disable_mmap: bool = False,
    state_dict: dict[str, torch.Tensor] | None = None,
):
    clip_l_sd = None
    if clip_l_path is None and state_dict is not None:
        if "text_encoders.clip_l.transformer.text_model.embeddings.position_embedding.weight" in state_dict:
            logger.info("clip_l is included in the checkpoint")
            clip_l_sd = _pop_prefixed_state_dict(state_dict, "text_encoders.clip_l.")
        else:
            logger.info("clip_l is not included in the checkpoint and clip_l_path is not provided")
            return None

    logger.info("Building CLIP-L")
    config = CLIPTextConfig(
        vocab_size=49408,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=77,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-05,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        model_type="clip_text_model",
        projection_dim=768,
    )
    with init_empty_weights():
        clip = CLIPTextModelWithProjection(config)

    if clip_l_sd is None:
        assert clip_l_path is not None
        logger.info("Loading state dict from %s", clip_l_path)
        clip_l_sd = load_safetensors(clip_l_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)

    if "text_projection.weight" not in clip_l_sd:
        logger.info("Adding text_projection.weight to CLIP-L state dict")
        clip_l_sd["text_projection.weight"] = torch.eye(768, dtype=dtype, device=device)

    missing_keys, unexpected_keys = _materialize_meta_module(clip, clip_l_sd, device)
    logger.info("Loaded CLIP-L: missing_keys=%s unexpected_keys=%s", missing_keys, unexpected_keys)
    return clip


def load_clip_g(
    clip_g_path: str | None,
    dtype: torch.dtype | None,
    device: str | torch.device,
    disable_mmap: bool = False,
    state_dict: dict[str, torch.Tensor] | None = None,
):
    clip_g_sd = None
    if state_dict is not None:
        if "text_encoders.clip_g.transformer.text_model.embeddings.position_embedding.weight" in state_dict:
            logger.info("clip_g is included in the checkpoint")
            clip_g_sd = _pop_prefixed_state_dict(state_dict, "text_encoders.clip_g.")
        elif clip_g_path is None:
            logger.info("clip_g is not included in the checkpoint and clip_g_path is not provided")
            return None

    logger.info("Building CLIP-G")
    config = CLIPTextConfig(
        vocab_size=49408,
        hidden_size=1280,
        intermediate_size=5120,
        num_hidden_layers=32,
        num_attention_heads=20,
        max_position_embeddings=77,
        hidden_act="gelu",
        layer_norm_eps=1e-05,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        model_type="clip_text_model",
        projection_dim=1280,
    )
    with init_empty_weights():
        clip = CLIPTextModelWithProjection(config)

    if clip_g_sd is None:
        assert clip_g_path is not None
        logger.info("Loading state dict from %s", clip_g_path)
        clip_g_sd = load_safetensors(clip_g_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)
    missing_keys, unexpected_keys = _materialize_meta_module(clip, clip_g_sd, device)
    logger.info("Loaded CLIP-G: missing_keys=%s unexpected_keys=%s", missing_keys, unexpected_keys)
    return clip


def load_t5xxl(
    t5xxl_path: str | None,
    dtype: torch.dtype | None,
    device: str | torch.device,
    disable_mmap: bool = False,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> T5EncoderModel | None:
    t5xxl_sd = None
    if state_dict is not None:
        if "text_encoders.t5xxl.transformer.encoder.block.0.layer.0.SelfAttention.k.weight" in state_dict:
            logger.info("t5xxl is included in the checkpoint")
            t5xxl_sd = _pop_prefixed_state_dict(state_dict, "text_encoders.t5xxl.")
        elif t5xxl_path is None:
            logger.info("t5xxl is not included in the checkpoint and t5xxl_path is not provided")
            return None

    t5_config = json.loads(
        """
{
  "architectures": ["T5EncoderModel"],
  "classifier_dropout": 0.0,
  "d_ff": 10240,
  "d_kv": 64,
  "d_model": 4096,
  "decoder_start_token_id": 0,
  "dense_act_fn": "gelu_new",
  "dropout_rate": 0.1,
  "eos_token_id": 1,
  "feed_forward_proj": "gated-gelu",
  "initializer_factor": 1.0,
  "is_encoder_decoder": true,
  "is_gated_act": true,
  "layer_norm_epsilon": 1e-06,
  "model_type": "t5",
  "num_decoder_layers": 24,
  "num_heads": 64,
  "num_layers": 24,
  "output_past": true,
  "pad_token_id": 0,
  "relative_attention_max_distance": 128,
  "relative_attention_num_buckets": 32,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.41.2",
  "use_cache": true,
  "vocab_size": 32128
}
"""
    )
    config = T5Config(**t5_config)
    with init_empty_weights():
        t5xxl = T5EncoderModel._from_config(config)

    if t5xxl_sd is None:
        assert t5xxl_path is not None
        logger.info("Loading state dict from %s", t5xxl_path)
        t5xxl_sd = load_safetensors(t5xxl_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)
    missing_keys, unexpected_keys = _materialize_meta_module(t5xxl, t5xxl_sd, device)
    logger.info("Loaded T5xxl: missing_keys=%s unexpected_keys=%s", missing_keys, unexpected_keys)
    return t5xxl


def load_vae(
    vae_path: str | None,
    vae_dtype: torch.dtype | None,
    device: str | torch.device | None,
    disable_mmap: bool = False,
    state_dict: dict[str, torch.Tensor] | None = None,
):
    if vae_path:
        logger.info("Loading VAE from %s...", vae_path)
        vae_sd = load_safetensors(vae_path, str(device) if device is not None else "cpu", disable_mmap, dtype=vae_dtype)
    else:
        assert state_dict is not None, "state_dict must be provided when vae_path is not set"
        vae_sd = _pop_prefixed_state_dict(state_dict, "first_stage_model.")

    logger.info("Building VAE")
    vae = SDVAE(vae_dtype or torch.float32, device)
    logger.info("Loading state dict...")
    info = vae.load_state_dict(vae_sd)
    logger.info("Loaded VAE: %s", info)
    return vae.to(device=device, dtype=vae_dtype)


def _resolve_sd3_model_attr(model_config: ModelConfig, name: str, default=None):
    """Read temporary SD3-specific config attributes without hard-coding dataclass fields yet."""
    return getattr(model_config, name, default)


def _maybe_cast_fp8(module: torch.nn.Module, module_name: str) -> torch.nn.Module:
    """Cast supported SD3 modules to FP8 when requested."""
    unsupported_fp8_dtypes = {
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    }
    if module.dtype in unsupported_fp8_dtypes:
        raise ValueError(f"Unsupported fp8 model dtype for {module_name}: {module.dtype}")
    if module.dtype == torch.float8_e4m3fn:
        logger.info("Loaded fp8 %s model", module_name)
        return module

    logger.info("Casting %s to fp8_e4m3fn", module_name)
    return module.to(torch.float8_e4m3fn)


def _prepare_scaled_positional_embeddings(
    mmdit: MMDiT,
    resolutions: list[tuple[int, int]] | None,
) -> None:
    """Enable scaled positional embeddings for the latent resolutions seen in the dataset."""
    if not resolutions:
        logger.warning("enable_scaled_pos_embed=True but no dataset resolutions were provided; leaving scaled positional embeddings disabled")
        return

    latent_sizes = [round(math.sqrt(height * width)) // 8 for height, width in resolutions]
    latent_sizes = sorted(set(latent_sizes))
    logger.info("Preparing scaled positional embeddings for resolutions=%s latent_sizes=%s", resolutions, latent_sizes)
    mmdit.enable_scaled_pos_embed(True, latent_sizes)


def _load_target_model(
    model_config: ModelConfig,
    memory_config: MemoryConfig,
    caching_config: CachingConfig,
    precision_config: PrecisionConfig,
    accelerator,
    weight_dtype: torch.dtype,
    device: str | torch.device = "cpu",
    resolutions: list[tuple[int, int]] | None = None,
) -> tuple[str, list[torch.nn.Module | None], SDVAE, MMDiT]:
    """Load SD3 model components from a unified checkpoint plus optional sidecars."""
    name_or_path = model_config.pretrained_model_name_or_path
    assert name_or_path is not None, "pretrained_model_name_or_path must be specified"

    model_dtype = match_mixed_precision(precision_config, weight_dtype)
    loading_dtype = None if precision_config.fp8_base else model_dtype

    logger.info("Loading SD3 state dict from %s", name_or_path)
    state_dict = load_safetensors(
        name_or_path,
        "cpu",
        disable_mmap=caching_config.disable_mmap_load_safetensors,
        dtype=loading_dtype,
    )

    mmdit = load_mmdit(state_dict, loading_dtype, device)
    mmdit.set_pos_emb_random_crop_rate(float(_resolve_sd3_model_attr(model_config, "pos_emb_random_crop_rate", 0.0)))

    if bool(_resolve_sd3_model_attr(model_config, "enable_scaled_pos_embed", False)):
        _prepare_scaled_positional_embeddings(mmdit, resolutions)

    blocks_to_swap = int(_resolve_sd3_model_attr(model_config, "blocks_to_swap", 0) or 0)
    if blocks_to_swap > 0:
        if memory_config.cpu_offload_checkpointing:
            raise ValueError("SD3 blocks_to_swap is not supported together with cpu_offload_checkpointing")
        mmdit.enable_block_swap(blocks_to_swap, accelerator.device)

    if precision_config.fp8_base:
        mmdit = _maybe_cast_fp8(mmdit, "SD3 MMDiT")

    clip_l = load_clip_l(
        _resolve_sd3_model_attr(model_config, "clip_l", None),
        model_dtype,
        device,
        disable_mmap=caching_config.disable_mmap_load_safetensors,
        state_dict=state_dict,
    )
    if clip_l is not None:
        clip_l.eval()

    clip_g = load_clip_g(
        _resolve_sd3_model_attr(model_config, "clip_g", None),
        model_dtype,
        device,
        disable_mmap=caching_config.disable_mmap_load_safetensors,
        state_dict=state_dict,
    )
    if clip_g is not None:
        clip_g.eval()

    t5_loading_dtype = None if precision_config.fp8_base and not precision_config.fp8_base_unet else model_dtype
    t5xxl = load_t5xxl(
        _resolve_sd3_model_attr(model_config, "t5xxl", None),
        t5_loading_dtype,
        device,
        disable_mmap=caching_config.disable_mmap_load_safetensors,
        state_dict=state_dict,
    )
    if t5xxl is not None:
        t5xxl.eval()
        if precision_config.fp8_base and not precision_config.fp8_base_unet:
            t5xxl = cast(T5EncoderModel, _maybe_cast_fp8(t5xxl, "SD3 T5-XXL"))

    vae = load_vae(
        model_config.vae,
        model_dtype,
        device,
        disable_mmap=caching_config.disable_mmap_load_safetensors,
        state_dict=state_dict,
    )
    if model_config.vae_conv2d_padding_mode is not None and model_config.vae_conv2d_padding_mode.lower() != "zeros":
        logger.info("Loading SD3 VAE with padding mode: %s", model_config.vae_conv2d_padding_mode)
        padding_mode = cast(
            Literal["zeros", "reflect", "replicate", "circular"],
            model_config.vae_conv2d_padding_mode,
        )
        set_padding_mode_for_vae_conv2d_modules(vae, padding_mode)

    return mmdit.model_type, [clip_l, clip_g, t5xxl], vae, mmdit


def load_target_model(
    model_config: ModelConfig,
    memory_config: MemoryConfig,
    caching_config: CachingConfig,
    precision_config: PrecisionConfig,
    accelerator,
    weight_dtype: torch.dtype,
    *,
    resolutions: list[tuple[int, int]] | None = None,
) -> tuple[str, list[torch.nn.Module | None], SDVAE, MMDiT]:
    """Load SD3 model components with distributed-process coordination."""
    model_version: str | None = None
    text_encoders: list[torch.nn.Module | None] | None = None
    vae: SDVAE | None = None
    mmdit: MMDiT | None = None

    assert accelerator.state.num_processes > 0, "num_processes must be greater than 0"
    for process_index in range(accelerator.state.num_processes):
        if process_index == accelerator.state.local_process_index:
            logger.info(
                "loading SD3 model for process %s/%s",
                accelerator.state.local_process_index,
                accelerator.state.num_processes,
            )
            load_device = accelerator.device if memory_config.lowram else "cpu"
            model_version, text_encoders, vae, mmdit = _load_target_model(
                model_config,
                memory_config,
                caching_config,
                precision_config,
                accelerator,
                weight_dtype,
                device=load_device,
                resolutions=resolutions,
            )

            if getattr(mmdit, "blocks_to_swap", 0):
                mmdit.move_to_device_except_swap_blocks(accelerator.device)
            if memory_config.lowram:
                for text_encoder in text_encoders:
                    if text_encoder is not None:
                        text_encoder.to(accelerator.device)
                vae.to(accelerator.device)
                if not getattr(mmdit, "blocks_to_swap", 0):
                    mmdit.to(accelerator.device)

            clean_memory_on_device(accelerator.device)
        accelerator.wait_for_everyone()

    assert model_version is not None and text_encoders is not None and vae is not None and mmdit is not None, "SD3 model loading failed"
    return model_version, text_encoders, vae, mmdit


__all__ = [
    "analyze_state_dict_state",
    "load_clip_g",
    "load_clip_l",
    "load_target_model",
    "load_mmdit",
    "load_t5xxl",
    "load_vae",
]
