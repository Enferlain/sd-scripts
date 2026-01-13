# SDXL PEFT Training Strategy implementation

import ast
import logging
import random
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from tqdm import tqdm

import library.strategies.base.encoding
import library.strategies.base.tokenization
import library.strategies.sd.caching
import library.strategies.sdxl.caching
import library.strategies.sdxl.encoding
import library.strategies.sdxl.tokenization

try:
    from ramtorch.helpers import replace_linear_with_ramtorch
except (ImportError, AssertionError):
    replace_linear_with_ramtorch = None  # type: ignore[assignment]

from library.strategies.sdxl.caching import SdxlConditioning
from library.strategies.base.training import TrainingStrategy
from library.constants import SDXL_VAE_LATENT_SCALE, MODEL_VERSION_SDXL_BASE_V1_0
from library.models.sdxl.conversion import get_size_embeddings
from library.models.sdxl.text_encoder import get_hidden_states_sdxl
from library.models.sdxl.loader import load_target_model
from library.models.runtime_utils import replace_unet_modules
from library.training.sample_generation import sample_images_common
from library.pipelines.sdxl_lpw_stable_diffusion import SdxlStableDiffusionLongPromptWeightingPipeline
from library.utils.model_metadata import get_model_metadata_from_config
from library.training.diffusion import get_noise_noisy_latents_and_timesteps
from library.training.trainer_utils import calculate_val_loss_check
from library.config.config_validation import validate_sdxl_peft
from library.utils.common_utils import setup_logging
from library.utils.device_utils import clean_memory_on_device
from library.losses.loss import get_huber_threshold_if_needed, conditional_loss
from library.losses.loss_weighting import apply_masked_loss

setup_logging()
logger = logging.getLogger(__name__)


def tokenize_sdxl_captions(
    tokenizer1: Any, tokenizer2: Any, captions: list[str], max_token_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize captions on-the-fly for SDXL (dual CLIP encoders).

    Handles 77+ token sequences by chunking into multiple 77-token segments,
    matching the legacy TokenizeStrategy._get_input_ids behavior.

    Args:
        tokenizer1: CLIP-L tokenizer.
        tokenizer2: CLIP-G tokenizer.
        captions: List of caption strings.
        max_token_length: Maximum token sequence length (e.g., 225 for 3 chunks).

    Returns:
        Tuple of (clip_l_tokens, clip_g_tokens):
        - If max_token_length <= 77: shape [batch_size, 77]
        - If max_token_length > 77: shape [batch_size, n_chunks, 77]
    """
    # Match legacy behavior: SdxlTokenizeStrategy adds +2 for BOS/EOS
    effective_max_len = max_token_length + 2 if max_token_length is not None else None

    def _tokenize_and_chunk(tokenizer: Any, texts: list[str], max_len: int) -> torch.Tensor:
        """Tokenize and optionally chunk into 77-token segments."""
        model_max = tokenizer.model_max_length  # 77 for CLIP

        if max_len is None or max_len <= model_max:
            # Simple case: just tokenize with padding/truncation to 77
            return tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=model_max,
                return_tensors="pt",
            ).input_ids

        # Long sequence case: tokenize to full length, then chunk
        # Request max_len tokens (will be padded/truncated)
        raw_tokens = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).input_ids  # [batch, max_len]

        # Chunk each sample into [n_chunks, 77] segments
        batch_chunks = []
        for input_ids in raw_tokens:
            # input_ids: [max_len]
            chunks = []
            # Step through in increments of 75 (77 - BOS - EOS)
            for i in range(1, max_len - model_max + 2, model_max - 2):
                # Build chunk: <BOS> + 75 tokens + <EOS/PAD>
                chunk = torch.cat(
                    [
                        input_ids[0:1],  # BOS
                        input_ids[i : i + model_max - 2],  # 75 content tokens
                        input_ids[-1:],  # last token (EOS or PAD)
                    ]
                )

                # Fix chunk endings for v2/SDXL tokenizers (pad_token != eos_token)
                if tokenizer.pad_token_id != tokenizer.eos_token_id:
                    # If end is "x <non-EOS/PAD>", change last to EOS
                    if chunk[-2] != tokenizer.eos_token_id and chunk[-2] != tokenizer.pad_token_id:
                        chunk[-1] = tokenizer.eos_token_id
                    # If beginning is "<BOS> <PAD> ...", change to "<BOS> <EOS> ..."
                    if chunk[1] == tokenizer.pad_token_id:
                        chunk[1] = tokenizer.eos_token_id

                chunks.append(chunk)

            batch_chunks.append(torch.stack(chunks))  # [n_chunks, 77]

        return torch.stack(batch_chunks)  # [batch, n_chunks, 77]

    tokens1 = _tokenize_and_chunk(tokenizer1, captions, effective_max_len)
    tokens2 = _tokenize_and_chunk(tokenizer2, captions, effective_max_len)

    return tokens1, tokens2


@dataclass
class SdxlTrainingStrategy(TrainingStrategy):
    """
    SDXL implementation of PEFT training strategy.

    Extracted from SDXLPeftTrainer class methods.
    """

    vae_latent_scale: float = SDXL_VAE_LATENT_SCALE

    # Instance state set during model loading
    load_stable_diffusion_format: bool = False
    logit_scale: Any = None
    ckpt_info: Any = None

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

    def get_tokenize_strategy(self, cfg: Any) -> Any:
        """
        Return SDXL tokenize strategy (dual tokenizers).

        Args:
            cfg: Configuration object.

        Returns:
            SdxlTokenizeStrategy instance.
        """
        return library.strategies.sdxl.tokenization.SdxlTokenizeStrategy(cfg.training.max_token_length, cfg.model.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: library.strategies.sdxl.tokenization.SdxlTokenizeStrategy) -> list[Any]:
        """
        Return both tokenizers for SDXL.

        Args:
            tokenize_strategy: SdxlTokenizeStrategy instance.

        Returns:
            List containing two tokenizers.
        """
        return [tokenize_strategy.tokenizer1, tokenize_strategy.tokenizer2]

    def get_latents_caching_strategy(self, cfg: Any) -> Any:
        """
        Return SD/SDXL latents caching strategy (shared implementation).

        Args:
            cfg: Configuration object.

        Returns:
            SdSdxlLatentsCachingStrategy instance.
        """
        return library.strategies.sd.caching.SdSdxlLatentsCachingStrategy(
            False, cfg.data.caching.cache_latents_to_disk, cfg.data.caching.vae_batch_size, cfg.data.caching.skip_cache_check
        )

    def get_text_encoding_strategy(self, cfg: Any) -> Any:
        """
        Return SDXL text encoding strategy.

        Args:
            cfg: Configuration object.

        Returns:
            SdxlTextEncodingStrategy instance.
        """
        return library.strategies.sdxl.encoding.SdxlTextEncodingStrategy()

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

    def get_text_encoder_outputs_caching_strategy(self, cfg: Any) -> Any | None:
        """
        Return SDXL text encoder outputs caching strategy if enabled.

        Args:
            cfg: Configuration object.

        Returns:
            SdxlTextEncoderOutputsCachingStrategy instance or None.
        """
        if cfg.data.caching.cache_text_encoder_outputs:
            return library.strategies.sdxl.caching.SdxlTextEncoderOutputsCachingStrategy(
                cfg.data.caching.cache_text_encoder_outputs_to_disk,
                None,  # batch_size: not used for text encoder outputs caching
                cfg.data.caching.skip_cache_check,
                is_weighted=cfg.data.caption.weighted_captions,
            )
        else:
            return None

    def cache_text_encoder_outputs_if_needed(
        self, cfg: Any, accelerator: Any, unet: Any, vae: Any, text_encoders: list[Any], dataset: Any, weight_dtype: torch.dtype
    ) -> None:
        """
        Cache text encoder outputs for SDXL (dual encoders, more complex than SD).

        Args:
            cfg: Configuration object.
            accelerator: Accelerator instance.
            unet: UNet model.
            vae: VAE model.
            text_encoders: List of text encoders.
            dataset: Dataset object.
            weight_dtype: Weight data type.
        """
        if cfg.data.caching.cache_text_encoder_outputs:
            org_vae_device = vae.device
            org_unet_device = unet.device
            if not cfg.performance.memory.lowram:
                # Save memory by moving vae and unet to cpu
                logger.info("move vae and unet to cpu to save memory")
                vae.to("cpu")
                unet.to("cpu")
                clean_memory_on_device(accelerator.device)

            # When TE is not be trained, it will not be prepared so we need to use explicit autocast
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            text_encoders[1].to(accelerator.device, dtype=weight_dtype)
            with accelerator.autocast():
                dataset.new_cache_text_encoder_outputs(text_encoders + [accelerator.unwrap_model(text_encoders[-1])], accelerator)
            accelerator.wait_for_everyone()

            text_encoders[0].to("cpu", dtype=torch.float32)  # Text Encoder doesn't work with fp16 on CPU
            text_encoders[1].to("cpu", dtype=torch.float32)
            clean_memory_on_device(accelerator.device)

            if not cfg.performance.memory.lowram:
                logger.info("move vae and unet back to original device")
                vae.to(org_vae_device)  # Defined above in this same `if not lowram` block
                unet.to(org_unet_device)  # Defined above in this same `if not lowram` block
        else:
            # Get text encoder outputs at each training step, so keep on GPU
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            text_encoders[1].to(accelerator.device, dtype=weight_dtype)

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
        """
        Call SDXL UNet with size embeddings.

        SDXL UNet signature includes vector_embedding (size/crop conditioning).

        Args:
            cfg: Configuration object.
            accelerator: Accelerator instance.
            unet: UNet model.
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

        noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)
        return noise_pred

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
        )

    def validate_extra_config(self, cfg: Any, train_dataset_group: Any, val_dataset_group: Any) -> None:
        """
        Run SDXL-specific config validation.

        Args:
            cfg: Configuration object.
            train_dataset_group: Training dataset group.
            val_dataset_group: Validation dataset group.
        """
        validate_sdxl_peft(cfg, train_dataset_group, val_dataset_group)

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

    # region SDXL-specific conditioning extraction

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

    # endregion

    # region SDXL-specific text conditioning

    def _get_text_cond(
        self, cfg: Any, accelerator: Any, batch: Any, tokenizers: list[Any], text_encoders: list[Any], weight_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get SDXL text conditioning from batch.

        Args:
            cfg: Configuration object.
            accelerator: Accelerator instance.
            batch: Batch data.
            tokenizers: List of tokenizers.
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

        # DEBUG: Log TE device placement and training status (remove after testing)
        te1_training = any(p.requires_grad for p in text_encoders[0].parameters())
        te2_training = any(p.requires_grad for p in text_encoders[1].parameters())
        logger.info(
            f"[DEBUG] _get_text_cond: TE device={te_device}, TE1 trainable={te1_training}, TE2 trainable={te2_training}"
        )  # DEBUG: remove

        # Fallback: tokenize captions on-the-fly if no cached tokens
        if input_ids is None:
            captions = batch.get("captions", [])
            if not captions:
                raise ValueError("Batch has neither 'input_ids' nor 'captions' - cannot encode text")

            # Tokenize using the tokenize_fn if available, otherwise use tokenizers directly
            input_ids1, input_ids2 = tokenize_sdxl_captions(tokenizers[0], tokenizers[1], captions, cfg.training.max_token_length)
            input_ids1 = input_ids1.to(te_device)
            input_ids2 = input_ids2.to(te_device)
        else:
            input_ids1 = input_ids["clip_l"].to(te_device)
            input_ids2 = input_ids["clip_g"].to(te_device)

        with torch.enable_grad():
            encoder_hidden_states1, encoder_hidden_states2, pool2 = get_hidden_states_sdxl(
                cfg.training.max_token_length,
                input_ids1,
                input_ids2,
                tokenizers[0],
                tokenizers[1],
                text_encoders[0],
                text_encoders[1],
                None if not cfg.performance.precision.full_fp16 else weight_dtype,
                accelerator=accelerator,
            )

        # DEBUG: Log output grad status before device transfer (remove after testing)
        logger.info(
            f"[DEBUG] TE outputs: h1.requires_grad={encoder_hidden_states1.requires_grad}, h1.device={encoder_hidden_states1.device}"
        )  # DEBUG: remove

        # Move outputs to training device (may be different from TE device when offloading)
        result = (
            encoder_hidden_states1.to(accelerator.device, dtype=weight_dtype),
            encoder_hidden_states2.to(accelerator.device, dtype=weight_dtype),
            pool2.to(accelerator.device, dtype=weight_dtype),
        )

        # DEBUG: Log output device after transfer (remove after testing)
        logger.info(f"[DEBUG] After .to(): h1.requires_grad={result[0].requires_grad}, h1.device={result[0].device}")  # DEBUG: remove

        return result

    # endregion

    # region Training batch processing methods

    def get_noise_pred_and_target(
        self,
        cfg: Any,
        accelerator: Any,
        noise_scheduler: Any,
        latents: torch.Tensor,
        batch: Any,
        text_encoder_conds: Any,
        unet: Any,
        adapter: Any,
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
            adapter: Adapter model.
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

            if len(diff_output_pr_indices) > 0:
                adapter.set_multiplier(0.0)
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
                adapter.set_multiplier(1.0)
                target[diff_output_pr_indices] = noise_pred_prior.to(target.dtype)

        return noise_pred, target, timesteps, None

    def process_batch(
        self,
        batch: Any,
        text_encoders: list[Any],
        unet: Any,
        adapter: Any,
        vae: Any,
        noise_scheduler: Any,
        vae_dtype: torch.dtype,
        weight_dtype: torch.dtype,
        accelerator: Any,
        cfg: Any,
        text_encoding_strategy: library.strategies.base.encoding.TextEncodingStrategy,
        tokenize_strategy: library.strategies.base.tokenization.TokenizeStrategy,
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
            adapter: Adapter model.
            vae: VAE model.
            noise_scheduler: Noise scheduler.
            vae_dtype: VAE data type.
            weight_dtype: Weight data type.
            accelerator: Accelerator instance.
            cfg: Configuration object.
            text_encoding_strategy: Text encoding strategy.
            tokenize_strategy: Tokenize strategy.
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
            latents = self._prepare_latents(batch, cfg, accelerator, vae, vae_dtype)

        # SDXL text conditioning - use cached outputs or encode on the fly
        tokenizers = self.get_tokenizers(tokenize_strategy)  # type: ignore[arg-type]  # Caller ensures correct strategy type
        text_encoder_conds = self._get_text_cond(cfg, accelerator, batch, tokenizers, text_encoders, weight_dtype)

        noise_pred, target, timesteps, weighting = self.get_noise_pred_and_target(
            cfg,
            accelerator,
            noise_scheduler,
            latents,
            batch,
            text_encoder_conds,
            unet,
            adapter,
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
            loss = self.post_process_loss(loss, cfg, timesteps, noise_scheduler)

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
        adapter: Any,
        vae: Any,
        noise_scheduler: Any,
        vae_dtype: torch.dtype,
        weight_dtype: torch.dtype,
        accelerator: Any,
        cfg: Any,
        text_encoding_strategy: library.strategies.base.encoding.TextEncodingStrategy,
        tokenize_strategy: library.strategies.base.tokenization.TokenizeStrategy,
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
            adapter: Adapter model.
            vae: VAE model.
            noise_scheduler: Noise scheduler.
            vae_dtype: VAE data type.
            weight_dtype: Weight data type.
            accelerator: Accelerator instance.
            cfg: Configuration object.
            text_encoding_strategy: Text encoding strategy.
            tokenize_strategy: Tokenize strategy.
            train_text_encoder: Train text encoder flag.
            train_unet: Train UNet flag.
            timesteps_list: List of timesteps for validation.

        Returns:
            Validation loss.
        """
        if timesteps_list is None:
            timesteps_list = [50, 350, 500, 650, 950]
        total_loss: torch.Tensor = torch.tensor(0.0)
        with torch.autograd.grad_mode.inference_mode(mode=True):
            latents = self._prepare_latents(batch, cfg, accelerator, vae, vae_dtype)

            # SDXL text conditioning
            tokenizers = self.get_tokenizers(tokenize_strategy)  # type: ignore[arg-type]  # Caller ensures correct strategy type
            text_encoder_conds = self._get_text_cond(cfg, accelerator, batch, tokenizers, text_encoders, weight_dtype)

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
                    adapter,
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
        adapter: Any,
        tokenize_strategy: Any,
        text_encoders: list[Any],
        text_encoding_strategy: Any,
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
            adapter: Adapter model.
            tokenize_strategy: Tokenize strategy.
            text_encoders: List of text encoders.
            text_encoding_strategy: Text encoding strategy.
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
        if not calculate_val_loss_check(cfg.validation, cfg.training, global_step, epoch_step, val_dataloader, train_dataloader):
            return None, None

        if batch is not None:
            self.on_step_start(cfg, accelerator, adapter, text_encoders, unet, batch, weight_dtype, is_train=False)

        rng_states = self.switch_rng_state(int(cfg.validation.validation_seed) if cfg.validation.validation_seed else 23, accelerator)
        timesteps_list = ast.literal_eval(cfg.validation.validation_timesteps)

        accelerator.print("")
        accelerator.print("Validating バリデーション処理...")
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
                    adapter,
                    vae,
                    noise_scheduler,
                    vae_dtype,
                    weight_dtype,
                    accelerator,
                    cfg,
                    text_encoding_strategy,
                    tokenize_strategy,
                    train_text_encoder=train_text_encoder,
                    timesteps_list=timesteps_list,
                )
                total_loss += loss.detach().item()
            current_val_loss = total_loss / validation_steps
            val_loss_recorder.add(current_val_loss)

        average_val_loss: float = val_loss_recorder.average

        self.restore_rng_state(rng_states, accelerator)

        return current_val_loss, average_val_loss

    # endregion
