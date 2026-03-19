# SD1.5/2 PEFT Training Strategy implementation

import ast
import logging
import random
from typing import Any

import torch
from torch import nn
from tqdm import tqdm

import library.strategies.sd.caching
from library.strategies.sd.encoding import encode_sd_tokens, encode_sd_tokens_with_weights
from library.strategies.sd.tokenization import build_sd_tokenizers, tokenize_sd_captions, tokenize_sd_text, tokenize_sd_text_with_weights

try:
    from ramtorch.helpers import replace_linear_with_ramtorch
except (ImportError, AssertionError):
    replace_linear_with_ramtorch = None  # type: ignore[assignment]

import library.models.sd.conversion
from library.constants import SD_VAE_LATENT_SCALE
from library.strategies.base.training import TrainingStrategy
from library.models.runtime_utils import replace_unet_modules
from library.models.sd.loader import load_target_model
from library.models.sd.text_encoder import get_hidden_states_sd
from library.training.sample_generation import sample_images_common
from library.pipelines.lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline
from library.utils.model_metadata import get_model_metadata_from_config
from library.training.diffusion import get_noise_noisy_latents_and_timesteps, prepare_latents
from library.training.trainer_utils import restore_rng_state, switch_rng_state

from library.losses.loss import get_huber_threshold_if_needed, conditional_loss
from library.losses.loss_weighting import apply_masked_loss, post_process_loss
logger = logging.getLogger(__name__)


class SdTrainingStrategy(TrainingStrategy):
    """
    SD1.5/2 implementation of PEFT training strategy.

    Extracted from legacy SD training script.
    """

    vae_latent_scale: float = SD_VAE_LATENT_SCALE
    _tokenizers: list[Any]

    def load_target_model(
        self, cfg: Any, weight_dtype: torch.dtype, accelerator: Any
    ) -> tuple[str, nn.Module, nn.Module, nn.Module | None]:
        """
        Load SD1.5/2 model components.

        Args:
            cfg: Configuration object.
            weight_dtype: Weight data type.
            accelerator: Accelerator instance.

        Returns:
            Tuple of (model_version, text_encoder, vae, unet).
        """
        text_encoder, vae, unet, _ = load_target_model(cfg.model, cfg.performance.memory, weight_dtype, accelerator)

        if cfg.performance.memory.use_ramtorch:
            if replace_linear_with_ramtorch is None:
                raise ImportError("RamTorch is not available. Please install it or set use_ramtorch to False.")
            logger.info("Applying RamTorch to SD UNet, VAE, and Clip-L.")
            if isinstance(unet, torch.nn.Module):
                unet = replace_linear_with_ramtorch(unet, accelerator.device)
                logger.info("RamTorch applied to SD unet.")

            if isinstance(text_encoder, torch.nn.Module):  # SD uses single text_encoder, SDXL uses list
                text_encoder = replace_linear_with_ramtorch(text_encoder, accelerator.device)
                logger.info("RamTorch applied to SD Clip-L.")

            if isinstance(vae, torch.nn.Module):
                vae = replace_linear_with_ramtorch(vae, accelerator.device)
                logger.info("RamTorch applied to SD VAE.")

        # Apply xformers / memory efficient attention
        replace_unet_modules(
            unet, cfg.performance.attention.mem_eff_attn, cfg.performance.attention.xformers, cfg.performance.attention.sdpa
        )
        if torch.__version__ >= "2.0.0":
            vae.set_use_memory_efficient_attention_xformers(cfg.performance.attention.xformers)

        return (
            library.models.sd_model_util.get_model_version_str_for_sd1_sd2(cfg.model.model_type == "sd2", cfg.loss.v_parameterization),
            text_encoder,
            vae,
            unet,
        )

    # --- ModelPreparationStrategy overrides (CLIP-specific) ---

    def prepare_text_encoder_grad_ckpt_workaround(self, index: int, text_encoder: Any) -> None:
        """Enable grad on CLIP embeddings so gradient checkpointing works."""
        text_encoder.text_model.embeddings.requires_grad_(True)

    def prepare_text_encoder_fp8(self, index: int, text_encoder: Any, te_weight_dtype: torch.dtype, weight_dtype: torch.dtype) -> None:
        """Cast CLIP embeddings back from FP8 — nn.Embedding doesn't support FP8."""
        text_encoder.text_model.embeddings.to(dtype=weight_dtype)

    def cast_text_encoder(self, cfg: Any) -> bool:
        """SD casts text encoders during shared precision setup."""
        return True

    def cast_vae(self, cfg: Any) -> bool:
        """SD casts the VAE during trainer setup."""
        return True

    def cast_denoiser(self, cfg: Any) -> bool:
        """SD casts the denoiser during shared precision setup."""
        return True

    def post_process_trainable(
        self,
        cfg: Any,
        accelerator: Any,
        trainable_model: Any,
        text_encoders: list[Any],
        denoiser: Any,
    ) -> None:
        """SD currently requires no post-processing after trainable setup."""
        return None

    max_token_length: int = 0
    clip_skip: int | None = None

    def __init__(self, cfg: Any):
        """Construct a fully initialized SD training strategy from config."""
        self._tokenizers, self.max_token_length = build_sd_tokenizers(
            cfg.model.model_type == "sd2",
            cfg.training.max_token_length,
            cfg.model.tokenizer_cache_dir,
        )
        self.clip_skip = cfg.training.clip_skip
        self.la_sampler = None
        self.live_plotter_process = None

    @property
    def tokenizers(self) -> list[Any]:
        """Return the tokenizer instances owned by this SD strategy."""
        return self._tokenizers

    # --- New pipeline caching methods ---
    def create_latent_caching_strategy(self, cfg: Any) -> Any:
        """Create SD latent caching strategy for the new CachingEngine pipeline."""
        latent_dtype = "fp32" if cfg.performance.precision.no_half_vae else "fp16"
        return library.strategies.sd.caching.SdLatentsPipelineStrategy(
            flip_aug=cfg.data.preprocessing.flip_aug,
            dtype=latent_dtype,
        )

    def create_te_caching_strategy(self, cfg: Any) -> Any:
        """Create SD text encoder caching strategy for the new CachingEngine pipeline."""
        return library.strategies.sd.caching.SdTextEncoderPipelineStrategy(
            clip_skip=cfg.training.clip_skip,
            max_token_length=cfg.training.max_token_length,
        )

    def get_token_cache_encoder_names(self) -> list[str]:
        """Return SD token-cache encoder names."""
        return ["clip"]

    def build_te_cache_model_bundle(self, cfg: Any, accelerator: Any, text_encoders: list[Any], tokenizers: list[Any]) -> Any:
        """Return the TE caching bundle for SD-style text encoding."""
        return (*text_encoders, *tokenizers)

    def tokenize_captions(self, tokenizers: list[Any], captions: list[str], max_token_length: int) -> list[torch.Tensor]:
        """Tokenize captions using SD's single CLIP tokenizer."""
        return [tokenize_sd_captions(tokenizers[0], captions, max_token_length)]

    def tokenize(self, text: str | list[str]) -> list[torch.Tensor]:
        """Tokenize SD prompts using strategy-owned tokenizer state."""
        if not self._tokenizers:
            raise RuntimeError("SD strategy has no tokenizers configured.")
        return tokenize_sd_text(self._tokenizers[0], self.max_token_length, text)

    def tokenize_with_weights(self, text: str | list[str]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Tokenize SD prompts with per-token prompt weights."""
        if not self._tokenizers:
            raise RuntimeError("SD strategy has no tokenizers configured.")
        return tokenize_sd_text_with_weights(self._tokenizers[0], self.max_token_length, text)

    def encode_tokens(self, models: list[Any], tokens: list[torch.Tensor]) -> list[torch.Tensor]:
        """Encode SD tokens using strategy-owned tokenizer/runtime state."""
        if not self._tokenizers:
            raise RuntimeError("SD strategy has no tokenizers configured.")
        return encode_sd_tokens(self._tokenizers[0], self.clip_skip, models, tokens)

    def _prepare_latents(self, batch: Any, cfg: Any, accelerator: Any, vae: Any, vae_dtype: torch.dtype) -> torch.Tensor:
        """Delegate SD latent preparation to the shared diffusion helper."""
        return prepare_latents(batch, cfg, accelerator, vae, vae_dtype, self.vae_latent_scale)

    def encode_tokens_with_weights(
        self,
        models: list[Any],
        tokens: list[torch.Tensor],
        weights: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Encode SD tokens and apply prompt weights."""
        if not self._tokenizers:
            raise RuntimeError("SD strategy has no tokenizers configured.")
        return encode_sd_tokens_with_weights(self._tokenizers[0], self.clip_skip, models, tokens, weights)

    def encode_te_outputs_in_memory(
        self,
        text_encoders: list[Any],
        tokenizers: list[Any],
        caption: str,
        max_token_length: int,
        device: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute SD text encoder outputs for a single caption (in-memory caching)."""
        input_ids = tokenize_sd_captions(tokenizers[0], [caption], max_token_length).to(device)

        with torch.no_grad():
            hidden_state = get_hidden_states_sd(
                input_ids,
                tokenizers[0],
                text_encoders[0],
            )
            return {"hidden_state": hidden_state.squeeze(0).cpu()}

    def get_models_for_text_encoding(self, cfg: Any, accelerator: Any, text_encoders: list[Any]) -> list[Any]:
        """
        Return text encoders for encoding (SD uses single encoder).

        Args:
            cfg: Configuration object.
            accelerator: Accelerator instance.
            text_encoders: List of text encoders.

        Returns:
            List of text encoders.
        """
        return text_encoders

    def _get_text_conds(
        self,
        batch: Any,
        text_encoders: list[Any],
        accelerator: Any,
        cfg: Any,
        train_text_encoder: bool,
        is_train: bool,
        weight_dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        """Get SD text conditioning from cached outputs, cached tokens, or live captions."""
        te_outputs = batch.get("text_encoder_outputs")
        text_encoder_conds = (
            [te_outputs["hidden_state"].to(accelerator.device, dtype=weight_dtype)]
            if te_outputs is not None
            else []
        )

        if len(text_encoder_conds) == 0 or text_encoder_conds[0] is None or train_text_encoder:
            with torch.set_grad_enabled(is_train and train_text_encoder), accelerator.autocast():
                if cfg.data.caption.weighted_captions:
                    input_ids_list, weights_list = self.tokenize_with_weights(batch["captions"])
                    encoded_text_encoder_conds = self.encode_tokens_with_weights(
                        self.get_models_for_text_encoding(cfg, accelerator, text_encoders),
                        input_ids_list,
                        weights_list,
                    )
                else:
                    input_ids_dict = batch.get("input_ids")
                    if input_ids_dict is not None:
                        input_ids = [input_ids_dict["clip"].to(accelerator.device)]
                    else:
                        input_ids = [ids.to(accelerator.device) for ids in self.tokenize(batch["captions"])]
                    encoded_text_encoder_conds = self.encode_tokens(
                        self.get_models_for_text_encoding(cfg, accelerator, text_encoders),
                        input_ids,
                    )

                if cfg.performance.precision.full_fp16:
                    encoded_text_encoder_conds = [c.to(weight_dtype) for c in encoded_text_encoder_conds]

            if len(text_encoder_conds) == 0:
                text_encoder_conds = encoded_text_encoder_conds
            else:
                for i in range(len(encoded_text_encoder_conds)):
                    if encoded_text_encoder_conds[i] is not None:
                        text_encoder_conds[i] = encoded_text_encoder_conds[i]

        return text_encoder_conds

    def call_denoiser(
        self,
        cfg: Any,
        accelerator: Any,
        denoiser: Any,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_conds: list[torch.Tensor],
        batch: Any,
        weight_dtype: torch.dtype,
        **kwargs,
    ) -> torch.Tensor:
        """
        Call SD UNet with simple signature.

        Args:
            cfg: Configuration object.
            accelerator: Accelerator instance.
            denoiser: Denoiser model.
            noisy_latents: Noisy latents tensor.
            timesteps: Timesteps tensor.
            text_conds: List of text conditioning tensors.
            batch: Batch data.
            weight_dtype: Weight data type.
            **kwargs: Additional arguments.

        Returns:
            Noise prediction tensor.
        """
        noise_pred = denoiser(noisy_latents, timesteps, text_conds[0]).sample
        return noise_pred

    def call_unet(
        self,
        cfg: Any,
        accelerator: Any,
        unet: Any,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_conds: list[torch.Tensor],
        batch: Any,
        weight_dtype: torch.dtype,
        **kwargs,
    ) -> torch.Tensor:
        """Backward-compatible SD alias for the denoiser call."""
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
        Generate sample images for SD.

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
            StableDiffusionLongPromptWeightingPipeline,
            accelerator,
            cfg.output.sampling,
            cfg.training,
            cfg.output.saving,
            cfg.loss,
            epoch,
            global_step,
            device,
            vae,
            tokenizers[0],
            text_encoders[0],
            unet,
            strategy=self,
        )

    def update_metadata(self, metadata: dict, cfg: Any) -> None:
        """
        SD doesn't add extra metadata.

        Args:
            metadata: Metadata dictionary.
            cfg: Configuration object.
        """
        pass

    def get_model_metadata(self, cfg: Any) -> dict:
        """
        Get SAI model spec for SD.

        Args:
            cfg: Configuration object.

        Returns:
            Metadata dictionary.
        """
        return get_model_metadata_from_config(
            state_dict=None,  # Valid: function signature accepts dict | None
            metadata_config=cfg.output.metadata,
            is_sdxl=False,  # SD strategy is never used for SDXL
            is_v2=cfg.model.model_type == "sd2",
            v_parameterization=cfg.loss.v_parameterization,
            is_lora=True,
            is_textual_inversion=False,
            resolution=cfg.data.preprocessing.resolution,
            min_timestep=cfg.timestep.min_timestep,
            max_timestep=cfg.timestep.max_timestep,
            clip_skip=cfg.training.clip_skip,
        )

    def get_noise_pred_and_target(
        self,
        cfg: Any,
        accelerator: Any,
        noise_scheduler: Any,
        latents: torch.Tensor,
        batch: Any,
        text_encoder_conds: list[Any],
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
            for t in text_encoder_conds:
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
        Process a batch for training.

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
            latents = self._prepare_latents(batch, cfg, accelerator, vae, vae_dtype)

        text_encoder_conds = self._get_text_conds(
            batch=batch,
            text_encoders=text_encoders,
            accelerator=accelerator,
            cfg=cfg,
            train_text_encoder=train_text_encoder,
            is_train=is_train,
            weight_dtype=weight_dtype,
        )

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
                loss = apply_masked_loss(loss, batch)
        else:
            loss = conditional_loss(noise_pred.float(), target.float(), "l2", "none", None)

        per_sample_loss = loss.mean([1, 2, 3])

        if is_train and self.la_sampler is not None and hasattr(self.la_sampler, "update"):
            self.la_sampler.update(timesteps.detach(), per_sample_loss.detach())

        loss = per_sample_loss
        if is_train:
            loss = loss * batch["loss_weights"]
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
        Process a batch for validation loss.

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
            latents = prepare_latents(batch, cfg, accelerator, vae, vae_dtype, self.vae_latent_scale)
            total_loss = torch.zeros(1, device=latents.device)

            text_encoder_conds = self._get_text_conds(
                batch=batch,
                text_encoders=text_encoders,
                accelerator=accelerator,
                cfg=cfg,
                train_text_encoder=train_text_encoder,
                is_train=False,
                weight_dtype=weight_dtype,
            )

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
    ) -> tuple[float | None, float | None, dict | None]:
        """
        Calculate validation loss.

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
            Tuple of (current_val_loss, average_val_loss, logs).
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
        logs = {"loss/current_val_loss": current_val_loss, "loss/average_val_loss": average_val_loss}

        restore_rng_state(rng_states, accelerator)

        return current_val_loss, average_val_loss, logs
