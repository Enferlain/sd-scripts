# SDXL PEFT Training Strategy implementation

import ast
import logging
import random
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from tqdm import tqdm

try:
    from ramtorch.helpers import replace_linear_with_ramtorch
except (ImportError, AssertionError):
    replace_linear_with_ramtorch = None  # type: ignore[assignment]

from library.strategies import strategy_sdxl, strategy_sd, strategy_base
from library.strategies.peft_strategy_base import PeftTrainingStrategy
from library.constants import SDXL_VAE_LATENT_SCALE, MODEL_VERSION_SDXL_BASE_V1_0
from library.models.sdxl.conversion import get_size_embeddings
from library.models.sdxl.text_encoder import get_hidden_states_sdxl
from library.models.sdxl.loader import load_target_model
from library.models.model_prep import replace_unet_modules
from library.training.sdxl_sample_generation import sample_images
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


@dataclass
class SdxlPeftStrategy(PeftTrainingStrategy):
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
            cfg.performance.caching,
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
        return strategy_sdxl.SdxlTokenizeStrategy(cfg.training.max_token_length, cfg.model.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_sdxl.SdxlTokenizeStrategy) -> list[Any]:
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
        return strategy_sd.SdSdxlLatentsCachingStrategy(
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
        return strategy_sdxl.SdxlTextEncodingStrategy()

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
        if cfg.performance.caching.cache_text_encoder_outputs:
            return strategy_sdxl.SdxlTextEncoderOutputsCachingStrategy(
                cfg.performance.caching.cache_text_encoder_outputs_to_disk,
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
        if cfg.performance.caching.cache_text_encoder_outputs:
            if not cfg.performance.memory.lowram:
                # Save memory by moving vae and unet to cpu
                logger.info("move vae and unet to cpu to save memory")
                org_vae_device = vae.device
                org_unet_device = unet.device
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

        # Get size embeddings
        orig_size = batch["original_sizes_hw"]
        crop_size = batch["crop_top_lefts"]
        target_size = batch["target_sizes_hw"]
        embs = get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(weight_dtype)

        # Concat text embeddings
        encoder_hidden_states1, encoder_hidden_states2, pool2 = text_conds
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
        sample_images(
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
        if "text_encoder_outputs1_list" not in batch or batch["text_encoder_outputs1_list"] is None:
            input_ids1 = batch["input_ids"]
            input_ids2 = batch["input_ids2"]
            with torch.enable_grad():
                input_ids1 = input_ids1.to(accelerator.device)
                input_ids2 = input_ids2.to(accelerator.device)
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
        else:
            encoder_hidden_states1 = batch["text_encoder_outputs1_list"].to(accelerator.device).to(weight_dtype)
            encoder_hidden_states2 = batch["text_encoder_outputs2_list"].to(accelerator.device).to(weight_dtype)
            pool2 = batch["text_encoder_pool2_list"].to(accelerator.device).to(weight_dtype)

        return encoder_hidden_states1, encoder_hidden_states2, pool2

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
        text_encoding_strategy: strategy_base.TextEncodingStrategy,
        tokenize_strategy: strategy_base.TokenizeStrategy,
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
            huber_c = get_huber_threshold_if_needed(cfg.loss, timesteps, noise_scheduler)
            loss = conditional_loss(
                noise_pred.float(), target.float(), cfg.loss.loss_type, "none", huber_c, scale=float(cfg.loss.loss_scale)
            )
            if weighting is not None:
                loss = loss * weighting
            if cfg.loss.masked or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                loss = apply_masked_loss(loss, batch)
        else:
            loss = conditional_loss(noise_pred.float(), target.float(), "l2", "none", None)

        per_sample_loss = loss.mean([1, 2, 3])

        if is_train and self.la_sampler is not None and hasattr(self.la_sampler, "update"):
            self.la_sampler.update(timesteps.detach(), per_sample_loss.detach())

        loss = per_sample_loss
        if is_train:
            loss = loss * batch["loss_weights"]
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
        text_encoding_strategy: strategy_base.TextEncodingStrategy,
        tokenize_strategy: strategy_base.TokenizeStrategy,
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
    ) -> tuple[float | None, float | None, dict | None]:
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
            Tuple of (current_val_loss, average_val_loss, logs).
        """
        if not calculate_val_loss_check(cfg.validation, cfg.training, global_step, epoch_step, val_dataloader, train_dataloader):
            return None, None, None

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
        logs = {"loss/current_val_loss": current_val_loss, "loss/average_val_loss": average_val_loss}

        self.restore_rng_state(rng_states, accelerator)

        return current_val_loss, average_val_loss, logs

    # endregion
