# PEFT Training Strategy interfaces
# Follows the pattern from PEFT_REFACTORING_PLAN.md

import logging
import torch

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


from library.optimizers.optimizer_utils import should_train_text_encoder, should_train_unet


logger = logging.getLogger(__name__)


class ModelLoadingStrategy(ABC):
    """Strategy for loading model components (text encoders, VAE, UNet)."""

    @abstractmethod
    def load_target_model(self, cfg: Any, weight_dtype: torch.dtype, accelerator: Any) -> tuple[str, Any, Any, Any]:
        """
        Load model components for this architecture.

        Args:
            cfg: Configuration object containing model settings.
            weight_dtype: Data type for model weights (e.g., torch.float16).
            accelerator: Accelerator instance for handling device placement.

        Returns:
            Tuple of (model_version, text_encoder, vae, unet):
            - model_version: String identifier for the model version.
            - text_encoder: A single text encoder model or a list of text encoders.
            - vae: The VAE model.
            - unet: The UNet model, or None if loaded lazily.
        """
        raise NotImplementedError

    def load_unet_lazily(self, cfg: Any, weight_dtype: torch.dtype, accelerator: Any, text_encoders: list[Any]) -> Any:
        """
        Load UNet lazily if not loaded in ``load_target_model``.

        Args:
            cfg: Configuration object.
            weight_dtype: Weight data type.
            accelerator: Accelerator instance.
            text_encoders: List of text encoders.

        Returns:
            Loaded UNet model.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("load_unet_lazily is not implemented for this architecture")


class TokenizationStrategy(ABC):
    """Strategy for tokenization setup in PEFT training."""

    @abstractmethod
    def get_tokenize_strategy(self, cfg: Any) -> Any:
        """
        Return the appropriate TokenizeStrategy for this architecture.

        Args:
            cfg: Configuration object containing tokenizer settings.

        Returns:
            A TokenizeStrategy instance suitable for the model architecture.
        """
        raise NotImplementedError

    @abstractmethod
    def get_tokenizers(self, tokenize_strategy: Any) -> list[Any] | Any:
        """
        Return tokenizer(s) from the strategy.

        Args:
            tokenize_strategy: The strategy object created by `get_tokenize_strategy`.

        Returns:
            A single tokenizer or a list/tuple of tokenizers.
        """
        raise NotImplementedError


class CachingStrategy(ABC):
    """Strategy for latents and text encoder caching."""

    @abstractmethod
    def get_latents_caching_strategy(self, cfg: Any) -> Any:
        """
        Return the LatentsCachingStrategy for this architecture.

        Args:
            cfg: Configuration object containing caching settings.

        Returns:
            A LatentsCachingStrategy instance.
        """
        raise NotImplementedError

    @abstractmethod
    def get_text_encoding_strategy(self, cfg: Any) -> Any:
        """
        Return the TextEncodingStrategy for this architecture.

        Args:
            cfg: Configuration object containing text encoding settings.

        Returns:
            A TextEncodingStrategy instance.
        """
        raise NotImplementedError

    def get_text_encoder_outputs_caching_strategy(self, cfg: Any) -> Any | None:
        """
        Return the legacy TextEncoderOutputsCachingStrategy.

        This hook exists only for deprecated dataset/script paths that still
        use the old per-dataset TE caching flow. The active shared runner does
        not use it, so the default implementation is ``None``.

        Args:
            cfg: Configuration object containing caching settings.

        Returns:
            A TextEncoderOutputsCachingStrategy instance, or None if not supported/enabled.
        """
        return None

    # --- New/shared-pipeline caching methods ---
    # These create strategies for the CachingEngine pipeline
    # (library/data/caching_engine.py). The legacy TE-output hooks above are
    # deprecated-path compatibility surface. `get_latents_caching_strategy()`
    # remains active transitional surface while the current runner still uses
    # LatentsCachingStrategy singleton wiring during setup.

    @abstractmethod
    def create_latent_caching_strategy(self, cfg: Any) -> Any:
        """
        Create a new-pipeline CachingStrategy for VAE latent caching.

        Returns an instance compatible with library.data.CachingEngine.

        Args:
            cfg: Configuration object.

        Returns:
            A CachingStrategy (library.data.caching_engine.CachingStrategy) instance.
        """
        raise NotImplementedError

    @abstractmethod
    def create_te_caching_strategy(self, cfg: Any) -> Any:
        """
        Create a new-pipeline CachingStrategy for text encoder output caching.

        Returns an instance compatible with library.data.CachingEngine, or None
        if disk-based TE caching is not applicable.

        Args:
            cfg: Configuration object.

        Returns:
            A CachingStrategy instance, or None.
        """
        raise NotImplementedError

    def get_token_cache_encoder_names(self) -> list[str]:
        """
        Return encoder names used when persisting per-epoch token caches.

        Shared phase code should not hardcode model-family token names.

        Returns:
            Ordered encoder names matching ``tokenize_captions()`` output order.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define token cache encoder names")

    def build_te_cache_model_bundle(self, cfg: Any, accelerator: Any, text_encoders: list[Any], tokenizers: list[Any]) -> Any:
        """
        Return the model bundle passed to the TE caching engine.

        Shared phase code should not know how a model family packs text
        encoders, tokenizers, or unwrapped helpers for TE caching.

        Args:
            cfg: Configuration object.
            accelerator: Accelerator instance.
            text_encoders: Text encoder models for the current trainer.
            tokenizers: Tokenizers for the current trainer.

        Returns:
            Model-family-specific bundle consumed by the TE caching strategy.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define a TE cache model bundle")

    @abstractmethod
    def tokenize_captions(self, tokenizers: list[Any], captions: list[str], max_token_length: int) -> list[torch.Tensor]:
        """
        Tokenize captions using model-family-specific tokenization.

        Args:
            tokenizers: List of tokenizer instances for this architecture.
            captions: List of caption strings to tokenize.
            max_token_length: Maximum token sequence length.

        Returns:
            List of token tensors, one per tokenizer.
        """
        raise NotImplementedError

    @abstractmethod
    def encode_te_outputs_in_memory(
        self,
        text_encoders: list[Any],
        tokenizers: list[Any],
        caption: str,
        max_token_length: int,
        device: Any,
    ) -> dict[str, torch.Tensor]:
        """
        Compute text encoder outputs for a single caption (in-memory caching path).

        Used when TE outputs are cached in memory rather than to disk.

        Args:
            text_encoders: List of text encoder models.
            tokenizers: List of tokenizer instances.
            caption: Single caption string.
            max_token_length: Maximum token sequence length.
            device: Device to run computation on.

        Returns:
            Dict of output name -> tensor (CPU), e.g. {"hidden_state1": ..., "pool2": ...}.
        """
        raise NotImplementedError

    def cache_text_encoder_outputs_if_needed(
        self, cfg: Any, accelerator: Any, unet: Any, vae: Any, text_encoders: list[Any], dataset: Any, weight_dtype: torch.dtype
    ) -> None:
        """
        Handle deprecated-path text encoder output caching if needed.

        The active shared runner no longer uses this hook. Legacy SD script
        paths still call it, so the default behavior is to ensure text
        encoders are moved to the accelerator for live encoding when no
        specialized TE-output caching strategy exists.

        Args:
            cfg: Configuration object.
            accelerator: Accelerator instance.
            unet: The UNet model.
            vae: The VAE model.
            text_encoders: List of text encoder models.
            dataset: The dataset to cache outputs for.
            weight_dtype: Data type for calculations.
        """
        for text_encoder in text_encoders:
            text_encoder.to(accelerator.device, dtype=weight_dtype)

    @abstractmethod
    def get_models_for_text_encoding(self, cfg: Any, accelerator: Any, text_encoders: list[Any]) -> list[Any]:
        """
        Return models to use for text encoding during training.

        SDXL may return wrapped/unwrapped models differently.

        Args:
            cfg: Configuration object.
            accelerator: Accelerator instance.
            text_encoders: List of available text encoder models.

        Returns:
            List of models properly prepared for encoding.
        """
        raise NotImplementedError


class SampleGenerationStrategy(ABC):
    """Strategy for generating sample images during training."""

    @abstractmethod
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
        Generate sample images for the current training step.

        Args:
            accelerator: Accelerator instance.
            cfg: Configuration object.
            epoch: Current epoch number.
            global_step: Current global step number.
            device: Device to run generation on.
            vae: The VAE model.
            tokenizers: List of tokenizers.
            text_encoders: List of text encoders.
            unet: The UNet model.
        """
        raise NotImplementedError


class CheckpointingStrategy(ABC):
    """Strategy for model-specific checkpointing and metadata."""

    @abstractmethod
    def update_metadata(self, metadata: dict, cfg: Any) -> None:
        """
        Add model-specific metadata fields.

        Args:
            metadata: The metadata dictionary to update.
            cfg: Configuration object.
        """
        raise NotImplementedError

    @abstractmethod
    def get_model_metadata(self, cfg: Any) -> dict:
        """
        Get model spec metadata.

        Args:
            cfg: Configuration object.

        Returns:
            Dictionary containing model metadata.
        """
        raise NotImplementedError

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
        """Serialize a full-model checkpoint for this model family.

        Mode decides when to save (lifecycle); this method decides how
        to serialize using architecture-specific conversion logic.

        The default implementation raises ``NotImplementedError``; model
        families that support full-model checkpointing (e.g. SDXL)
        override this with their own conversion/save logic.

        Policy (save cadence, retention, pruning) is not part of this
        method — that stays in Trainer/phases.

        Args:
            trainer: The Trainer instance (provides access to models,
                accelerator, cfg, and any state needed for saving).
            ckpt_name: Exact checkpoint filename or directory name.
            step: Current training step.
            epoch: Current training epoch.
            metadata: Training metadata dict (ss_* keys, etc.).
            save_dtype: Data type for saved weights.
            force_sync_upload: If True, block until HF upload completes.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement save_model_checkpoint. Full-model saving requires a strategy override."
        )


class ValidationStrategy(ABC):
    """Strategy for model-specific validation."""

    @abstractmethod
    def validate_extra_config(self, cfg: Any, train_dataset_group: Any, val_dataset_group: Any) -> None:
        """
        Perform model-specific config validation.

        Args:
            cfg: Configuration object.
            train_dataset_group: Training dataset group configuration.
            val_dataset_group: Validation dataset group configuration.
        """
        raise NotImplementedError

    @abstractmethod
    def calculate_val_loss(
        self,
        global_step: int,
        epoch_step: int,
        train_dataloader: Any,
        val_loss_recorder: Any,
        val_dataloader: Any,
        cyclic_val_dataloader: Any,
        trainable_model: Any,
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
        Calculate validation loss.

        Returns:
            Tuple of (current_val_loss, average_val_loss).
        """
        raise NotImplementedError


class DiffusionTrainingStrategy(ABC):
    """Strategy for diffusion-specific batch processing behavior."""

    def encode_images_to_latents(self, cfg: Any, vae: Any, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latents using the VAE.

        Args:
            cfg: Configuration object.
            vae: VAE model instance.
            images: Batch of images to encode.

        Returns:
            Encoded latents.
        """
        return vae.encode(images).latent_dist.sample()

    def shift_scale_latents(self, cfg: Any, latents: torch.Tensor) -> torch.Tensor:
        """
        Apply the model-family VAE latent scale factor.

        Args:
            cfg: Configuration object.
            latents: Latents tensor to scale.

        Returns:
            Scaled latents.
        """
        return latents * self.vae_latent_scale  # Defined in concrete strategies.

    def _prepare_latents(self, batch: Any, cfg: Any, accelerator: Any, vae: Any, vae_dtype: torch.dtype) -> torch.Tensor:
        """
        Prepare latents from batch data or cached entries.

        Shared between training and validation batch processing.

        Args:
            batch: Batch data containing either cached latents or images.
            cfg: Configuration object.
            accelerator: Accelerator instance.
            vae: VAE model for encoding images.
            vae_dtype: Data type for VAE operations.

        Returns:
            Prepared and scaled latents tensor.
        """
        import typing

        if "latents" in batch and batch["latents"] is not None:
            latents = typing.cast(torch.FloatTensor, batch["latents"].to(accelerator.device))
        else:
            vae_batch_size = cfg.data.caching.vae_batch_size
            if vae_batch_size is None or len(batch["images"]) <= vae_batch_size:
                latents = self.encode_images_to_latents(cfg, vae, batch["images"].to(accelerator.device, dtype=vae_dtype))
            else:
                chunks = [batch["images"][i : i + vae_batch_size] for i in range(0, len(batch["images"]), vae_batch_size)]
                list_latents = []
                for chunk in chunks:
                    with torch.no_grad():
                        chunk_latents = self.encode_images_to_latents(cfg, vae, chunk.to(accelerator.device, dtype=vae_dtype))
                        list_latents.append(chunk_latents)
                latents = torch.cat(list_latents, dim=0)

            if torch.any(torch.isnan(latents)):
                accelerator.print("NaN found in latents, replacing with zeros")
                latents = typing.cast(torch.FloatTensor, torch.nan_to_num(latents, 0, out=latents))

            latents = self.shift_scale_latents(cfg, latents)

        return latents

    @abstractmethod
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
        text_encoding_strategy: Any,
        tokenize_strategy: Any,
        is_train: bool = True,
        train_text_encoder: bool = True,
        train_unet: bool = True,
        edm2_model: Any | None = None,
        min_timestep_override: int | None = None,
        max_timestep_override: int | None = None,
        global_step: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """
        Process a batch for training or validation.

        Returns:
            Tuple of (loss, pre_scaling_loss, loss_scaled, timesteps).
        """
        raise NotImplementedError


class UNetCallingStrategy(ABC):
    """Strategy for calling UNet during training."""

    @abstractmethod
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
        Call UNet with architecture-specific arguments.

        SDXL adds added_cond_kwargs for size/crop conditioning.

        Args:
            cfg: Configuration object.
            accelerator: Accelerator instance.
            unet: The UNet model.
            noisy_latents: Input latents with noise.
            timesteps: Timesteps for denoising.
            text_conds: Text conditioning embeddings.
            batch: The current data batch.
            weight_dtype: Data type for calculations.
            **kwargs: Additional architecture-specific arguments.

        Returns:
            The noise prediction tensor.
        """
        raise NotImplementedError


class ModelPreparationStrategy:
    """Strategy hooks used while preparing trainable models and precision."""

    def get_text_encoders_train_flags(self, cfg: Any, text_encoders: list[Any]) -> list[bool]:
        """
        Return per-text-encoder training flags.

        Args:
            cfg: Configuration object.
            text_encoders: List of text encoders.

        Returns:
            List of boolean flags indicating training status for each encoder.
        """
        return [True] * len(text_encoders) if self.is_train_text_encoder(cfg) else [False] * len(text_encoders)

    def is_train_text_encoder(self, cfg: Any) -> bool:
        """
        Check if text encoder should be trained based on LR config.

        Args:
            cfg: Configuration object.

        Returns:
            True if text encoder should be trained, False otherwise.
        """
        return should_train_text_encoder(cfg.optimizer.learning_rates)

    def is_train_unet(self, cfg: Any) -> bool:
        """
        Check if UNet should be trained based on LR config.

        Args:
            cfg: Configuration object.

        Returns:
            True if UNet should be trained, False otherwise.
        """
        return should_train_unet(cfg.optimizer.learning_rates)

    def cast_text_encoder(self, cfg: Any) -> bool:
        """
        Determine if text encoder should be cast to a specific dtype.

        Args:
            cfg: Configuration object.

        Returns:
            True by default.
        """
        return True

    def cast_vae(self, cfg: Any) -> bool:
        """
        Determine if VAE should be cast to a specific dtype.

        Args:
            cfg: Configuration object.

        Returns:
            True by default.
        """
        return True

    def cast_unet(self, cfg: Any) -> bool:
        """
        Determine if UNet should be cast to a specific dtype.

        Args:
            cfg: Configuration object.

        Returns:
            True by default.
        """
        return True

    def is_text_encoder_not_needed_for_training(self, cfg: Any) -> bool:
        """
        Check if text encoder is unnecessary for the training loop.

        Returns True when TE caching is enabled AND TEs are not being trained.
        ``offload_text_encoders`` keeps TEs in memory (on CPU), so this returns
        False in that case.

        Args:
            cfg: Configuration object.

        Returns:
            True if TEs can be deleted from memory, False otherwise.
        """
        return cfg.data.caching.cache_text_encoder_outputs and not self.is_train_text_encoder(cfg)

    def prepare_text_encoder_grad_ckpt_workaround(self, index: int, text_encoder: Any) -> None:
        """
        Set up gradient checkpointing for a text encoder.

        Args:
            index: Index of the text encoder.
            text_encoder: The text encoder model.
        """
        text_encoder.text_model.embeddings.requires_grad_(True)

    def prepare_text_encoder_fp8(self, index: int, text_encoder: Any, te_weight_dtype: torch.dtype, weight_dtype: torch.dtype) -> None:
        """
        Prepare text encoder modules for FP8 training.

        Args:
            index: Index of the text encoder.
            text_encoder: The text encoder model.
            te_weight_dtype: Target weight dtype for text encoder.
            weight_dtype: General weight dtype.
        """
        text_encoder.text_model.embeddings.to(dtype=weight_dtype)

    def prepare_unet_with_accelerator(self, cfg: Any, accelerator: Any, unet: Any) -> Any:
        """
        Prepare UNet with the accelerator.

        Args:
            cfg: Configuration object.
            accelerator: Accelerator instance.
            unet: The UNet model.

        Returns:
            Prepared UNet model.
        """
        return accelerator.prepare(unet)

    def post_process_trainable(self, cfg: Any, accelerator: Any, trainable_model: Any, text_encoders: list[Any], unet: Any) -> None:
        """
        Post-process the trainable model after creation.

        Args:
            cfg: Configuration object.
            accelerator: Accelerator instance.
            trainable_model: The trainable model.
            text_encoders: List of text encoders.
            unet: The UNet model.
        """
        return None


class TrainingRuntimeStrategy:
    """Strategy hooks used by the shared training loop runtime."""

    def on_step_start(
        self,
        cfg: Any,
        accelerator: Any,
        trainable_model: Any,
        text_encoders: list[Any],
        unet: Any,
        batch: Any,
        weight_dtype: torch.dtype,
        is_train: bool = True,
    ) -> None:
        """
        Hook called at the start of each training or validation step.

        Args:
            cfg: Configuration object.
            accelerator: Accelerator instance.
            trainable_model: The trainable model.
            text_encoders: List of text encoders.
            unet: The UNet model.
            batch: The current data batch.
            weight_dtype: Weight data type.
            is_train: Boolean indicating training mode.
        """
        return None

    def on_validation_step_end(
        self, cfg: Any, accelerator: Any, trainable_model: Any, text_encoders: list[Any], unet: Any, batch: Any, weight_dtype: torch.dtype
    ) -> None:
        """
        Hook called after each validation step.

        Args:
            cfg: Configuration object.
            accelerator: Accelerator instance.
            trainable_model: The trainable model.
            text_encoders: List of text encoders.
            unet: The UNet model.
            batch: The current data batch.
            weight_dtype: Weight data type.
        """
        return None


@dataclass
class TrainingStrategy(
    ModelLoadingStrategy,
    TokenizationStrategy,
    CachingStrategy,
    SampleGenerationStrategy,
    CheckpointingStrategy,
    ValidationStrategy,
    DiffusionTrainingStrategy,
    UNetCallingStrategy,
    ModelPreparationStrategy,
    TrainingRuntimeStrategy,
):
    """
    Combined interface for all training strategies.

    Implementations inherit from this and provide model-specific implementations.
    """

    # Instance state (set during training)
    la_sampler: Any = field(default=None, init=False, repr=False)
    live_plotter_process: Any = field(default=None, init=False, repr=False)
