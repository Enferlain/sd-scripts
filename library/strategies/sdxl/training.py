# SDXL PEFT Training Strategy implementation

import ast
import logging
import random
from typing import Any

import torch
from torch import nn
from tqdm import tqdm

import library.strategies.sdxl.caching
from library.strategies.sdxl.encoding import encode_sdxl_tokens, encode_sdxl_tokens_with_weights
from library.strategies.sdxl.tokenization import build_sdxl_tokenizers, tokenize_sdxl_text, tokenize_sdxl_text_with_weights

try:
    from ramtorch.helpers import replace_linear_with_ramtorch
except (ImportError, AssertionError):
    replace_linear_with_ramtorch = None  # type: ignore[assignment]

from library.strategies.sdxl.caching import SdxlConditioning
from library.strategies.base.training import TrainingStrategy
from library.constants import SDXL_VAE_LATENT_SCALE, MODEL_VERSION_SDXL_BASE_V1_0
from library.models.sd.tokenizer import tokenize_clip_captions
from library.models.sdxl.conversion import get_size_embeddings
from library.models.sdxl.text_encoder import encode_input_ids_sdxl
from library.models.sdxl.loader import load_target_model
from library.models.runtime_utils import replace_unet_modules
from library.training.sample_generation import sample_images_common
from library.pipelines.sdxl_lpw_stable_diffusion import SdxlStableDiffusionLongPromptWeightingPipeline
from library.utils.model_metadata import get_model_metadata_from_config
from library.training.diffusion import get_noise_noisy_latents_and_timesteps, prepare_latents
from library.training.trainer_utils import restore_rng_state, switch_rng_state

from library.losses.loss import get_huber_threshold_if_needed, conditional_loss
from library.losses.loss_weighting import apply_masked_loss, post_process_loss


logger = logging.getLogger(__name__)


class SdxlTrainingStrategy(TrainingStrategy):
    """
    SDXL implementation of PEFT training strategy.

    Extracted from legacy SDXL training script.
    """

    vae_latent_scale: float = SDXL_VAE_LATENT_SCALE

    # Instance state set during model loading
    _tokenizers: list[Any]
    load_stable_diffusion_format: bool = False
    logit_scale: Any = None
    ckpt_info: Any = None
    max_token_length: int = 0

    def __init__(self, cfg: Any):
        """Construct a fully initialized SDXL training strategy from config."""
        self._tokenizers, self.max_token_length = build_sdxl_tokenizers(
            cfg.training.max_token_length,
            cfg.model.tokenizer_cache_dir,
        )
        self.load_stable_diffusion_format = False
        self.logit_scale = None
        self.ckpt_info = None
        self.la_sampler = None
        self.live_plotter_process = None

    @property
    def tokenizers(self) -> list[Any]:
        """Return the tokenizer instances owned by this SDXL strategy."""
        return self._tokenizers

    def load_target_model(
        self, cfg: Any, weight_dtype: torch.dtype, accelerator: Any
    ) -> tuple[str, list[nn.Module], nn.Module, nn.Module | None]:
        """
        Load SDXL model components (dual text encoders, VAE, UNet).

        Args:
            cfg: Configuration object.
            weight_dtype: Weight data type.
            accelerator: Accelerator instance.

        Returns:
            Tuple of (model_version, text_encoders, vae, unet).
        """
        (
            load_stable_diffusion_format,
            text_encoder1,
            text_encoder2,
            vae,
            unet,
            logit_scale,
            ckpt_info,
        ) = load_target_model(
            cfg.model,
            cfg.performance.memory,
            cfg.data.caching,
            cfg.performance.precision,
            accelerator,
            MODEL_VERSION_SDXL_BASE_V1_0,
            weight_dtype,
        )

        # Store for later use in checkpointing
        self.load_stable_diffusion_format = load_stable_diffusion_format
        self.logit_scale = logit_scale
        self.ckpt_info = ckpt_info

        if cfg.performance.memory.use_ramtorch:
            if replace_linear_with_ramtorch is None:
                raise ImportError("RamTorch is not available. Please install it or set use_ramtorch to False.")
            logger.info("Applying RamTorch to SDXL UNet, VAE, and Text Encoders.")
            if isinstance(unet, torch.nn.Module):
                unet = replace_linear_with_ramtorch(unet, accelerator.device)
                logger.info("RamTorch applied to SDXL unet.")

            if isinstance(vae, torch.nn.Module):
                vae = replace_linear_with_ramtorch(vae, accelerator.device)
                logger.info("RamTorch applied to SDXL vae.")

            if isinstance(text_encoder1, torch.nn.Module):
                text_encoder1 = replace_linear_with_ramtorch(text_encoder1, accelerator.device)
                logger.info("RamTorch applied to SDXL Clip-L.")

            if isinstance(text_encoder2, torch.nn.Module):
                text_encoder2 = replace_linear_with_ramtorch(text_encoder2, accelerator.device)
                logger.info("RamTorch applied to SDXL Clip-G.")

        # Apply xformers / memory efficient attention
        replace_unet_modules(
            unet, cfg.performance.attention.mem_eff_attn, cfg.performance.attention.xformers, cfg.performance.attention.sdpa
        )
        if torch.__version__ >= "2.0.0":
            vae.set_use_memory_efficient_attention_xformers(cfg.performance.attention.xformers)

        return MODEL_VERSION_SDXL_BASE_V1_0, [text_encoder1, text_encoder2], vae, unet

    # --- ModelPreparationStrategy overrides (CLIP-specific) ---

    def prepare_text_encoder_grad_ckpt_workaround(self, index: int, text_encoder: Any) -> None:
        """Enable grad on CLIP embeddings so gradient checkpointing works."""
        text_encoder.text_model.embeddings.requires_grad_(True)

    def prepare_text_encoder_fp8(self, index: int, text_encoder: Any, te_weight_dtype: torch.dtype, weight_dtype: torch.dtype) -> None:
        """Cast CLIP embeddings back from FP8 — nn.Embedding doesn't support FP8."""
        text_encoder.text_model.embeddings.to(dtype=weight_dtype)

    def cast_text_encoder(self, cfg: Any) -> bool:
        """SDXL casts text encoders during shared precision setup."""
        return True

    def cast_vae(self, cfg: Any) -> bool:
        """SDXL casts the VAE during trainer setup."""
        return True

    def cast_denoiser(self, cfg: Any) -> bool:
        """SDXL casts the denoiser during shared precision setup."""
        return True

    def get_models_for_text_encoding(self, cfg: Any, accelerator: Any, text_encoders: list[Any]) -> list[Any]:
        """
        Return text encoders for encoding in SDXL.

        SDXL needs unwrapped text_encoder2 for pooled output.
        Returns: [text_encoder1, text_encoder2, unwrapped_text_encoder2]

        Args:
            cfg: Configuration object.
            accelerator: Accelerator instance.
            text_encoders: List of text encoders.

        Returns:
            List of models for text encoding.
        """
        return text_encoders + [accelerator.unwrap_model(text_encoders[-1])]

    # --- New pipeline caching methods ---

    def create_latent_caching_strategy(self, cfg: Any) -> library.strategies.sdxl.caching.SdxlLatentsPipelineStrategy:
        """
        Create SDXL latent caching strategy for the new CachingEngine pipeline.

        Args:
            cfg: Configuration object.

        Returns:
            SdxlLatentsPipelineStrategy instance.
        """
        latent_dtype = "fp32" if cfg.performance.precision.no_half_vae else "fp16"
        return library.strategies.sdxl.caching.SdxlLatentsPipelineStrategy(
            flip_aug=cfg.data.preprocessing.flip_aug,
            dtype=latent_dtype,
        )

    def create_te_caching_strategy(self, cfg: Any) -> library.strategies.sdxl.caching.SdxlTextEncoderPipelineStrategy:
        """
        Create SDXL text encoder caching strategy for the new CachingEngine pipeline.

        Args:
            cfg: Configuration object.

        Returns:
            SdxlTextEncoderPipelineStrategy instance.
        """
        return library.strategies.sdxl.caching.SdxlTextEncoderPipelineStrategy(
            max_token_length=cfg.training.max_token_length,
        )

    def get_token_cache_encoder_names(self) -> list[str]:
        """Return SDXL token-cache encoder names."""
        return ["clip_l", "clip_g"]

    def build_te_cache_model_bundle(self, cfg: Any, accelerator: Any, text_encoders: list[Any], tokenizers: list[Any]) -> Any:
        """Return the TE caching bundle for SDXL text encoding."""
        return (*text_encoders, *tokenizers)

    def tokenize_captions(self, tokenizers: list[Any], captions: list[str], max_token_length: int) -> list[torch.Tensor]:
        """
        Tokenize captions using SDXL dual CLIP tokenizers.

        Args:
            tokenizers: [clip_l_tokenizer, clip_g_tokenizer].
            captions: List of caption strings.
            max_token_length: Maximum token sequence length.

        Returns:
            List of [clip_l_tokens, clip_g_tokens] tensors.
        """
        return [
            tokenize_clip_captions(tokenizers[0], captions, max_token_length),
            tokenize_clip_captions(tokenizers[1], captions, max_token_length),
        ]

    def tokenize(self, text: str | list[str]) -> list[torch.Tensor]:
        """Tokenize SDXL prompts using strategy-owned tokenizer state."""
        if len(self._tokenizers) != 2:
            raise RuntimeError("SDXL strategy has no tokenizers configured.")
        return tokenize_sdxl_text(self._tokenizers[0], self._tokenizers[1], self.max_token_length, text)

    def tokenize_with_weights(self, text: str | list[str]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Tokenize SDXL prompts with per-token prompt weights."""
        if len(self._tokenizers) != 2:
            raise RuntimeError("SDXL strategy has no tokenizers configured.")
        return tokenize_sdxl_text_with_weights(self._tokenizers[0], self._tokenizers[1], self.max_token_length, text)

    def encode_tokens(self, models: list[Any], tokens: list[torch.Tensor]) -> list[torch.Tensor]:
        """Encode SDXL tokens using strategy-owned tokenizer runtime state."""
        if len(self._tokenizers) != 2:
            raise RuntimeError("SDXL strategy has no tokenizers configured.")
        return encode_sdxl_tokens(self._tokenizers, models, tokens)

    def encode_tokens_with_weights(
        self,
        models: list[Any],
        tokens: list[torch.Tensor],
        weights: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Encode SDXL tokens and apply prompt weights."""
        if len(self._tokenizers) != 2:
            raise RuntimeError("SDXL strategy has no tokenizers configured.")
        return encode_sdxl_tokens_with_weights(self._tokenizers, models, tokens, weights)

    def encode_te_outputs_in_memory(
        self,
        text_encoders: list[Any],
        tokenizers: list[Any],
        caption: str,
        max_token_length: int,
        device: Any,
    ) -> dict[str, torch.Tensor]:
        """
        Compute SDXL text encoder outputs for a single caption (in-memory caching).

        Args:
            text_encoders: [clip_l_encoder, clip_g_encoder].
            tokenizers: [clip_l_tokenizer, clip_g_tokenizer].
            caption: Single caption string.
            max_token_length: Maximum token sequence length.
            device: Device to run computation on.

        Returns:
            Dict with hidden_state1, hidden_state2, pool2 tensors on CPU.
        """
        input_ids1, input_ids2 = self.tokenize_captions(tokenizers, [caption], max_token_length)
        input_ids1 = input_ids1.to(device)
        input_ids2 = input_ids2.to(device)

        with torch.no_grad():
            hidden_state1, hidden_state2, pool2 = encode_input_ids_sdxl(
                input_ids1,
                input_ids2,
                tokenizers[0],
                tokenizers[1],
                text_encoders[0],
                text_encoders[1],
            )
            return {
                "hidden_state1": hidden_state1.squeeze(0).cpu(),
                "hidden_state2": hidden_state2.squeeze(0).cpu(),
                "pool2": pool2.squeeze(0).cpu(),
            }

    def call_denoiser(
        self,
        cfg: Any,
        accelerator: Any,
        denoiser: Any,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_conds: Any,
        batch: Any,
        weight_dtype: torch.dtype,
        **kwargs,
    ) -> torch.Tensor:
        """
        Call SDXL UNet with size embeddings.

        SDXL UNet signature includes vector_embedding (size/crop conditioning).

        Args:
            cfg: Configuration object.
            accelerator: Accelerator instance.
            denoiser: Denoiser model.
            noisy_latents: Noisy latents tensor.
            timesteps: Timesteps tensor.
            text_conds: Tuple of text conditioning (encoder_hidden_states1, encoder_hidden_states2, pool2).
            batch: Batch data.
            weight_dtype: Weight data type.
            **kwargs: Additional arguments.

        Returns:
            Noise prediction tensor.
        """
        indices = kwargs.get("indices")

        # Get size embeddings from conditioning objects
        conditionings = batch["conditionings"]
        orig_size, crop_size, target_size = self._extract_conditioning_tensors(conditionings, accelerator.device, weight_dtype)
        embs = get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(weight_dtype)

        # Concat text embeddings
        encoder_hidden_states1, encoder_hidden_states2, pool2 = text_conds

        # Debug: ensure batch sizes match
        if pool2.shape[0] != embs.shape[0]:
            raise RuntimeError(
                f"Batch size mismatch in call_unet: pool2 has {pool2.shape[0]} samples, "
                f"but conditionings has {len(conditionings)} items (embs shape: {embs.shape}). "
                f"batch latents shape: {batch['latents'].shape if 'latents' in batch else 'N/A'}, "
                f"captions: {len(batch.get('captions', []))}"
            )

        vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)
        text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)

        if indices is not None and len(indices) > 0:
            noisy_latents = noisy_latents[indices]
            timesteps = timesteps[indices]
            text_embedding = text_embedding[indices]
            vector_embedding = vector_embedding[indices]

        noise_pred = denoiser(noisy_latents, timesteps, text_embedding, vector_embedding)
        return noise_pred

    def call_unet(
        self,
        cfg: Any,
        accelerator: Any,
        unet: Any,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_conds: Any,
        batch: Any,
        weight_dtype: torch.dtype,
        **kwargs,
    ) -> torch.Tensor:
        """Backward-compatible SDXL alias for the denoiser call."""
        return self.call_denoiser(
            cfg,
            accelerator,
            unet,
            noisy_latents,
            timesteps,
            text_conds,
            batch,
            weight_dtype,
            **kwargs,
        )

    def sample_images(
        self,
        accelerator: Any,
        cfg: Any,
        epoch: int,
        global_step: int,
        device: torch.device,
        vae: Any,
        tokenizers: list[Any],
        text_encoders: list[Any],
        unet: Any,
    ) -> None:
        """
        Generate sample images for SDXL.

        Args:
            accelerator: Accelerator instance.
            cfg: Configuration object.
            epoch: Current epoch.
            global_step: Current global step.
            device: Device.
            vae: VAE model.
            tokenizers: List of tokenizers.
            text_encoders: List of text encoder models.
            unet: UNet model.
        """
        sample_images_common(
            SdxlStableDiffusionLongPromptWeightingPipeline,
            accelerator,
            cfg.output.sampling,
            cfg.training,
            cfg.output.saving,
            cfg.loss,
            epoch,
            global_step,
            device,
            vae,
            tokenizers,
            text_encoders,
            unet,
            strategy=self,
        )

    def update_metadata(self, metadata: dict, cfg: Any) -> None:
        """
        Add SDXL-specific metadata fields.
        SDXL doesn't add extra metadata beyond what get_model_metadata provides.

        Args:
            metadata: Metadata dictionary.
            cfg: Configuration object.
        """
        pass

    def get_model_metadata(self, cfg: Any) -> dict:
        """
        Get SAI model spec for SDXL.

        Args:
            cfg: Configuration object.

        Returns:
            Metadata dictionary.
        """
        return get_model_metadata_from_config(
            state_dict=None,  # Valid: function signature accepts dict | None
            metadata_config=cfg.output.metadata,
            is_sdxl=True,  # SDXL strategy is always SDXL
            is_v2=False,  # SDXL is not v2
            v_parameterization=cfg.loss.v_parameterization,
            is_lora=True,
            is_textual_inversion=False,
            resolution=cfg.data.preprocessing.resolution,
            min_timestep=cfg.timestep.min_timestep,
            max_timestep=cfg.timestep.max_timestep,
            clip_skip=cfg.training.clip_skip,
        )

    def save_model_checkpoint(
        self,
        trainer: Any,
        ckpt_name: str,
        step: int,
        epoch: int,
        metadata: dict[str, str],
        save_dtype: torch.dtype,
        force_sync_upload: bool = False,
    ) -> None:
        """Serialize an SDXL full-model checkpoint.

        Handles both SD-format (.safetensors/.ckpt) and Diffusers-format saves.
        Uses strategy-owned state (logit_scale, ckpt_info) set during
        ``load_target_model``, so mode never needs to know about them.
        """
        import os

        from library.models.sdxl.conversion import (
            save_diffusers_checkpoint,
            save_stable_diffusion_checkpoint,
        )

        cfg = trainer.cfg

        # Determine save format from config
        save_model_as = cfg.output.saving.save_model_as
        save_stable_diffusion_format = save_model_as in ("safetensors", "ckpt")
        use_safetensors = save_model_as == "safetensors"

        # Unwrap models from accelerator
        assert trainer.unet is not None, "UNet must be set before save_model_checkpoint"
        unet = trainer.accelerator.unwrap_model(trainer.unet)
        text_encoder1 = trainer.accelerator.unwrap_model(trainer.text_encoders[0])
        text_encoder2 = trainer.accelerator.unwrap_model(trainer.text_encoders[1]) if len(trainer.text_encoders) > 1 else None
        vae = trainer.vae

        os.makedirs(cfg.output.saving.output_dir, exist_ok=True)
        ckpt_file = os.path.join(cfg.output.saving.output_dir, ckpt_name)

        if save_stable_diffusion_format:
            # Generate SD-format metadata with is_lora=False for full-model
            modelspec_metadata = get_model_metadata_from_config(
                state_dict=None,
                metadata_config=cfg.output.metadata,
                is_sdxl=True,
                is_v2=False,
                v_parameterization=cfg.loss.v_parameterization,
                is_lora=False,
                is_textual_inversion=False,
                is_stable_diffusion_ckpt=True,
            )

            # Merge runtime metadata (ss_* fields from trainer) with modelspec.
            # Runtime metadata is the base; modelspec keys overlay on top.
            merged_metadata = {**metadata, **modelspec_metadata}

            trainer.accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
            save_stable_diffusion_checkpoint(
                ckpt_file,
                text_encoder1,
                text_encoder2,
                unet,
                epoch,
                step,
                self.ckpt_info,  # Strategy-owned state from load_target_model
                vae,
                self.logit_scale,  # Strategy-owned state from load_target_model
                merged_metadata,
                save_dtype,
            )
        else:
            # Diffusers format — ckpt_name is a directory path
            out_dir = ckpt_file
            os.makedirs(out_dir, exist_ok=True)

            src_path = cfg.model.pretrained_model_name_or_path

            trainer.accelerator.print(f"\nsaving model: {out_dir}")
            save_diffusers_checkpoint(
                out_dir,
                text_encoder1,
                text_encoder2,
                unet,
                src_path,
                vae,
                use_safetensors=use_safetensors,
                save_dtype=save_dtype,
            )

        # Upload to HuggingFace if configured
        if cfg.output.huggingface is not None and cfg.output.huggingface.huggingface_repo_id is not None:
            from library.utils import huggingface_util

            huggingface_util.upload(
                cfg.output.huggingface,
                ckpt_file,  # Works for both formats: file or directory path
                "/" + ckpt_name,
                force_sync_upload=force_sync_upload,
            )

    def post_process_trainable(self, cfg: Any, accelerator: Any, trainable_model: Any, text_encoders: list[Any], unet: Any) -> None:
        """SDXL-specific post-processing: freeze TE1 last layer and final_layer_norm.

        This prevents training instability in SDXL by freezing the last
        encoder layer and the final layer norm of the first text encoder
        (CLIP-L). Only applies when TE1 is being trained (has grad enabled).
        """
        if len(text_encoders) < 1:
            return

        te1 = text_encoders[0]

        # Only freeze if TE1 is actually being trained
        if not any(p.requires_grad for p in te1.parameters()):
            return

        # Freeze TE1's last encoder layer
        if hasattr(te1, "text_model"):
            text_model = te1.text_model
            if hasattr(text_model, "encoder") and hasattr(text_model.encoder, "layers"):
                last_layer = text_model.encoder.layers[-1]
                last_layer.requires_grad_(False)
            if hasattr(text_model, "final_layer_norm"):
                text_model.final_layer_norm.requires_grad_(False)

    def _extract_conditioning_tensors(
        self,
        conditionings: list[SdxlConditioning],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract SDXL micro-conditioning tensors from batch conditionings.

        Args:
            conditionings: List of SdxlConditioning objects from batch.
            device: Target device for tensors.
            dtype: Target dtype for tensors.

        Returns:
            Tuple of (original_sizes, crop_top_lefts, target_sizes) tensors.
        """
        orig_sizes = []
        crop_top_lefts = []
        target_sizes = []

        for cond in conditionings:
            orig_sizes.append(cond.original_size_hw)
            crop_top_lefts.append(cond.crop_top_left)
            target_sizes.append(cond.target_size_hw)

        return (
            torch.tensor(orig_sizes, device=device, dtype=dtype),
            torch.tensor(crop_top_lefts, device=device, dtype=dtype),
            torch.tensor(target_sizes, device=device, dtype=dtype),
        )

    def _get_text_cond(
        self, cfg: Any, accelerator: Any, batch: Any, text_encoders: list[Any], weight_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get SDXL text conditioning from batch.

        Args:
            cfg: Configuration object.
            accelerator: Accelerator instance.
            batch: Batch data.
            text_encoders: List of text encoders.
            weight_dtype: Weight data type.

        Returns:
            Tuple of (encoder_hidden_states1, encoder_hidden_states2, pool2).
        """
        # Check for cached TE outputs (new pipeline format)
        te_outputs = batch.get("text_encoder_outputs")
        if te_outputs is not None:
            return (
                te_outputs["hidden_state1"].to(accelerator.device, dtype=weight_dtype),
                te_outputs["hidden_state2"].to(accelerator.device, dtype=weight_dtype),
                te_outputs["pool2"].to(accelerator.device, dtype=weight_dtype),
            )

        # Encode on-the-fly using tokenized inputs or tokenize from captions
        input_ids = batch.get("input_ids")

        # Determine device for encoding: use TE device (may be CPU when offloading)
        te_device = text_encoders[0].device

        # Fallback: tokenize captions on-the-fly if no cached tokens
        if input_ids is None:
            captions = batch.get("captions", [])
            if not captions:
                raise ValueError("Batch has neither 'input_ids' nor 'captions' - cannot encode text")

            input_ids1, input_ids2 = self.tokenize(captions)
            input_ids1 = input_ids1.to(te_device)
            input_ids2 = input_ids2.to(te_device)
        else:
            input_ids1 = input_ids["clip_l"].to(te_device)
            input_ids2 = input_ids["clip_g"].to(te_device)

        with torch.enable_grad():
            encoder_hidden_states1, encoder_hidden_states2, pool2 = encode_input_ids_sdxl(
                input_ids1,
                input_ids2,
                self._tokenizers[0],
                self._tokenizers[1],
                text_encoders[0],
                text_encoders[1],
                weight_dtype=None if not cfg.performance.precision.full_fp16 else weight_dtype,
                unwrapped_text_encoder2=accelerator.unwrap_model(text_encoders[1]),
            )

        # Move outputs to training device (may be different from TE device when offloading)
        return (
            encoder_hidden_states1.to(accelerator.device, dtype=weight_dtype),
            encoder_hidden_states2.to(accelerator.device, dtype=weight_dtype),
            pool2.to(accelerator.device, dtype=weight_dtype),
        )

    def get_noise_pred_and_target(
        self,
        cfg: Any,
        accelerator: Any,
        noise_scheduler: Any,
        latents: torch.Tensor,
        batch: Any,
        text_encoder_conds: Any,
        unet: Any,
        trainable_model: Any,
        weight_dtype: torch.dtype,
        train_unet: bool,
        fixed_timesteps: torch.Tensor | None = None,
        is_train: bool = True,
        min_timestep_override: int | None = None,
        max_timestep_override: int | None = None,
        global_step: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Sample noise, call UNet, get noise prediction target.

        Args:
            cfg: Configuration object.
            accelerator: Accelerator instance.
            noise_scheduler: Noise scheduler.
            latents: Latents tensor.
            batch: Batch data.
            text_encoder_conds: Text conditioning.
            unet: UNet model.
            trainable_model: The trainable model.
            weight_dtype: Weight data type.
            train_unet: Boolean indicating if UNet is trained.
            fixed_timesteps: Optional fixed timesteps.
            is_train: Boolean indicating training mode.
            min_timestep_override: Optional minimum timesteps override.
            max_timestep_override: Optional maximum timesteps override.
            global_step: Current global step.

        Returns:
            Tuple of (noise_pred, target, timesteps, weighting).
        """
        noise, noisy_latents, timesteps = get_noise_noisy_latents_and_timesteps(
            cfg.loss.regularization,
            cfg.timestep,
            cfg.training,
            noise_scheduler,
            latents,
            la_sampler=self.la_sampler,
            global_step=global_step,
            fixed_timesteps=fixed_timesteps,
            is_train=is_train,
            min_timestep_override=min_timestep_override,
            max_timestep_override=max_timestep_override,
            output_dtype=weight_dtype,
        )

        if is_train and cfg.performance.memory.gradient_checkpointing:
            for x in noisy_latents:
                x.requires_grad_(True)
            # For SDXL, text_encoder_conds is a tuple, handle differently
            if isinstance(text_encoder_conds, (list, tuple)):
                for t in text_encoder_conds:
                    if t is not None and hasattr(t, "requires_grad_"):
                        t.requires_grad_(True)

        with torch.set_grad_enabled(is_train), accelerator.autocast():
            noise_pred = self.call_unet(
                cfg, accelerator, unet, noisy_latents.requires_grad_(train_unet), timesteps, text_encoder_conds, batch, weight_dtype
            )

        if cfg.loss.v_parameterization:
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise

        # differential output preservation
        if "custom_attributes" in batch:
            diff_output_pr_indices = []
            for i, custom_attributes in enumerate(batch["custom_attributes"]):
                if "diff_output_preservation" in custom_attributes and custom_attributes["diff_output_preservation"]:
                    diff_output_pr_indices.append(i)

            if len(diff_output_pr_indices) > 0 and hasattr(trainable_model, "set_multiplier"):
                trainable_model.set_multiplier(0.0)
                with torch.no_grad(), accelerator.autocast():
                    noise_pred_prior = self.call_unet(
                        cfg,
                        accelerator,
                        unet,
                        noisy_latents,
                        timesteps,
                        text_encoder_conds,
                        batch,
                        weight_dtype,
                        indices=diff_output_pr_indices,
                    )
                trainable_model.set_multiplier(1.0)
                target[diff_output_pr_indices] = noise_pred_prior.to(target.dtype)

        return noise_pred, target, timesteps, None

    def process_batch(
        self,
        batch: Any,
        text_encoders: list[Any],
        unet: Any,
        trainable_model: Any,
        vae: Any,
        noise_scheduler: Any,
        vae_dtype: torch.dtype,
        weight_dtype: torch.dtype,
        accelerator: Any,
        cfg: Any,
        is_train: bool = True,
        train_text_encoder: bool = True,
        train_unet: bool = True,
        edm2_model: Any | None = None,
        min_timestep_override: int | None = None,
        max_timestep_override: int | None = None,
        global_step: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """
        Process a batch for SDXL training.

        Args:
            batch: Batch data.
            text_encoders: List of text encoders.
            unet: UNet model.
            trainable_model: The trainable model.
            vae: VAE model.
            noise_scheduler: Noise scheduler.
            vae_dtype: VAE data type.
            weight_dtype: Weight data type.
            accelerator: Accelerator instance.
            cfg: Configuration object.
            is_train: Training mode flag.
            train_text_encoder: Train text encoder flag.
            train_unet: Train UNet flag.
            edm2_model: EDM2 model (optional).
            min_timestep_override: Minimum timesteps override.
            max_timestep_override: Maximum timesteps override.
            global_step: Global step.

        Returns:
            Tuple of (loss, pre_scaling_loss, loss_scaled, timesteps).
        """
        with torch.no_grad():
            latents = prepare_latents(
                batch,
                cfg.data.caching,
                accelerator.device,
                vae,
                vae_dtype,
                self.vae_latent_scale,
                log_fn=accelerator.print,
            )

        # SDXL text conditioning - use cached outputs or encode on the fly
        text_encoder_conds = self._get_text_cond(cfg, accelerator, batch, text_encoders, weight_dtype)

        noise_pred, target, timesteps, weighting = self.get_noise_pred_and_target(
            cfg,
            accelerator,
            noise_scheduler,
            latents,
            batch,
            text_encoder_conds,
            unet,
            trainable_model,
            weight_dtype,
            train_unet,
            is_train=is_train,
            min_timestep_override=min_timestep_override,
            max_timestep_override=max_timestep_override,
            global_step=global_step,
        )

        if is_train:
            huber_c = get_huber_threshold_if_needed(cfg.loss, cfg.loss.huber, timesteps, noise_scheduler)
            loss = conditional_loss(
                noise_pred.float(), target.float(), cfg.loss.loss_type, "none", huber_c, scale=float(cfg.loss.loss_scale)
            )
            if weighting is not None:
                loss = loss * weighting
            if cfg.loss.masked.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                # Fail fast if user explicitly requested masked loss but no masks available
                if cfg.loss.masked.masked_loss:
                    has_cond = "conditioning_images" in batch
                    has_alpha = "alpha_masks" in batch and batch["alpha_masks"] is not None
                    if not has_cond and not has_alpha:
                        raise ValueError(
                            "cfg.loss.masked.masked_loss=True but no masks found in batch. "
                            "Ensure your dataset has alpha channels or conditioning images. "
                            "Set cfg.loss.masked.masked_loss=False if masking is not intended."
                        )
                loss = apply_masked_loss(loss, batch)
        else:
            loss = conditional_loss(noise_pred.float(), target.float(), "l2", "none", None)

        per_sample_loss = loss.mean([1, 2, 3])

        if is_train and self.la_sampler is not None and hasattr(self.la_sampler, "update"):
            self.la_sampler.update(timesteps.detach(), per_sample_loss.detach())

        loss = per_sample_loss
        if is_train:
            loss = loss * batch["loss_weights"].to(loss.device)
            loss = post_process_loss(loss, cfg, timesteps, noise_scheduler)

        if is_train and cfg.loss.loss_multiplier:
            loss.mul_(float(cfg.loss.loss_multiplier) if cfg.loss.loss_multiplier is not None else 1.0)

        pre_scaling_loss = loss.mean()

        if is_train and cfg.loss.edm2.edm2_loss_weighting and edm2_model is not None:
            loss, loss_scaled = edm2_model(loss, timesteps)
            loss_scaled = loss_scaled.mean()
        else:
            loss_scaled = None

        return loss.mean(), pre_scaling_loss, loss_scaled, timesteps

    def process_val_batch(
        self,
        batch: Any,
        text_encoders: list[Any],
        unet: Any,
        trainable_model: Any,
        vae: Any,
        noise_scheduler: Any,
        vae_dtype: torch.dtype,
        weight_dtype: torch.dtype,
        accelerator: Any,
        cfg: Any,
        train_text_encoder: bool = True,
        train_unet: bool = True,
        timesteps_list: list[int] | None = None,
    ) -> torch.Tensor:
        """
        Process a batch for SDXL validation loss.

        Args:
            batch: Batch data.
            text_encoders: List of text encoders.
            unet: UNet model.
            trainable_model: The trainable model.
            vae: VAE model.
            noise_scheduler: Noise scheduler.
            vae_dtype: VAE data type.
            weight_dtype: Weight data type.
            accelerator: Accelerator instance.
            cfg: Configuration object.
            train_text_encoder: Train text encoder flag.
            train_unet: Train UNet flag.
            timesteps_list: List of timesteps for validation.

        Returns:
            Validation loss.
        """
        if timesteps_list is None:
            timesteps_list = [50, 350, 500, 650, 950]
        with torch.autograd.grad_mode.inference_mode(mode=True):
            latents = prepare_latents(
                batch,
                cfg.data.caching,
                accelerator.device,
                vae,
                vae_dtype,
                self.vae_latent_scale,
                log_fn=accelerator.print,
            )
            total_loss = torch.zeros(1, device=latents.device)

            # SDXL text conditioning
            text_encoder_conds = self._get_text_cond(cfg, accelerator, batch, text_encoders, weight_dtype)

            batch_size = latents.shape[0]
            for fixed_timestep_value in timesteps_list:
                fixed_timesteps = torch.full((batch_size,), fixed_timestep_value, dtype=torch.long, device=latents.device)
                noise_pred, target, _, _ = self.get_noise_pred_and_target(
                    cfg,
                    accelerator,
                    noise_scheduler,
                    latents,
                    batch,
                    text_encoder_conds,
                    unet,
                    trainable_model,
                    weight_dtype,
                    train_unet,
                    fixed_timesteps,
                    is_train=False,
                )

                loss = conditional_loss(noise_pred.float(), target.float(), "l2", "none", None)
                loss = loss.mean([1, 2, 3]).mean()
                total_loss += loss

        return total_loss / len(timesteps_list)

    def calculate_val_loss(
        self,
        global_step: int,
        epoch_step: int,
        train_dataloader: Any,
        val_loss_recorder: Any,
        val_dataloader: Any,
        cyclic_val_dataloader: Any,
        trainable_model: Any,
        text_encoders: list[Any],
        unet: Any,
        vae: Any,
        noise_scheduler: Any,
        vae_dtype: torch.dtype,
        weight_dtype: torch.dtype,
        accelerator: Any,
        cfg: Any,
        epoch: int,
        batch: Any | None = None,
        train_text_encoder: bool = True,
    ) -> tuple[float | None, float | None]:
        """
        Calculate validation loss for SDXL.

        Args:
            global_step: Global step.
            epoch_step: Epoch step.
            train_dataloader: Training dataloader.
            val_loss_recorder: Validation loss recorder.
            val_dataloader: Validation dataloader.
            cyclic_val_dataloader: Cyclic validation dataloader.
            trainable_model: The trainable model.
            text_encoders: List of text encoders.
            unet: UNet model.
            vae: VAE model.
            noise_scheduler: Noise scheduler.
            vae_dtype: VAE data type.
            weight_dtype: Weight data type.
            accelerator: Accelerator instance.
            cfg: Configuration object.
            epoch: Current epoch.
            batch: Optional batch.
            train_text_encoder: Train text encoder flag.

        Returns:
            Tuple of (current_val_loss, average_val_loss).
        """
        if batch is not None:
            self.on_step_start(cfg, accelerator, trainable_model, text_encoders, unet, batch, weight_dtype, is_train=False)

        rng_states = switch_rng_state(int(cfg.validation.validation_seed) if cfg.validation.validation_seed else 23, accelerator)
        timesteps_list = ast.literal_eval(cfg.validation.validation_timesteps)

        accelerator.print("")
        accelerator.print("Validating...")
        total_loss = 0.0
        with torch.no_grad():
            validation_steps = (
                min(int(cfg.validation.max_validation_steps), len(val_dataloader))
                if cfg.validation.max_validation_steps is not None
                else len(val_dataloader)
            )
            val_dataloader_seed = random.randint(global_step, 0x7FFFFFFF)
            val_dataloader_state = random.Random(val_dataloader_seed).getstate()
            for _val_step in tqdm(range(validation_steps), desc="Validation Steps"):
                val_original_state = random.getstate()
                random.setstate(val_dataloader_state)
                batch = next(cyclic_val_dataloader)
                val_dataloader_state = random.getstate()
                random.setstate(val_original_state)
                loss = self.process_val_batch(
                    batch,
                    text_encoders,
                    unet,
                    trainable_model,
                    vae,
                    noise_scheduler,
                    vae_dtype,
                    weight_dtype,
                    accelerator,
                    cfg,
                    train_text_encoder=train_text_encoder,
                    timesteps_list=timesteps_list,
                )
                total_loss += loss.detach().item()
            current_val_loss = total_loss / validation_steps
            val_loss_recorder.add(current_val_loss)

        average_val_loss: float = val_loss_recorder.average

        restore_rng_state(rng_states, accelerator)

        return current_val_loss, average_val_loss
