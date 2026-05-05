"""Trainer-facing model-family strategy contracts and minimal shared defaults."""

from abc import ABC, abstractmethod
from typing import Any

import torch

from library.losses.loss_modifiers import BatchLossOutput
from library.objectives.base import ObjectiveRuntime
from library.strategies.base.context import StrategyPhase


class ModelConditioning(ABC):  # noqa: B024 - Marker class, no abstract methods
    """
    Base class for model-specific conditioning payloads.

    Concrete strategy families define their own conditioning dataclasses and
    thread them through cache loading / batch assembly as needed. The data
    pipeline only transports these values; ownership lives with strategies.
    """

    pass


class ModelLoadingStrategy(ABC):
    """Strategy for loading model components (text encoders, VAE, denoiser)."""

    @abstractmethod
    def load_target_model(self, cfg: Any, weight_dtype: torch.dtype, accelerator: Any) -> tuple[str, Any, Any, Any]:
        """
        Load model components for this architecture.

        Args:
            cfg: Configuration object containing model settings.
            weight_dtype: Data type for model weights (e.g., torch.float16).
            accelerator: Accelerator instance for handling device placement.

        Returns:
            Tuple of (model_version, text_encoder, vae, denoiser):
            - model_version: String identifier for the model version.
            - text_encoder: A single text encoder model or a list of text encoders.
            - vae: The VAE model.
            - denoiser: The denoiser model, or None if loaded lazily.
        """
        raise NotImplementedError

    def load_denoiser_lazily(self, cfg: Any, weight_dtype: torch.dtype, accelerator: Any, text_encoders: list[Any]) -> Any:
        """
        Load the denoiser lazily if not loaded in ``load_target_model``.

        Args:
            cfg: Configuration object.
            weight_dtype: Weight data type.
            accelerator: Accelerator instance.
            text_encoders: List of text encoders.

        Returns:
            Loaded denoiser model.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("load_denoiser_lazily is not implemented for this architecture")


class TokenizationStrategy(ABC):
    """Facet for model-family tokenization behavior."""

    @property
    @abstractmethod
    def tokenizers(self) -> list[Any]:
        """Return tokenizer instances owned by this strategy."""
        raise NotImplementedError

    @abstractmethod
    def tokenize(self, text: str | list[str]) -> list[torch.Tensor]:
        """
        Tokenize text into model-family token tensors.

        Args:
            text: Text or list of text to tokenize.

        Returns:
            List of token tensors.
        """
        raise NotImplementedError

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


class TextEncodingStrategy(ABC):
    """Facet for model-family text-encoding behavior."""

    @abstractmethod
    def encode_tokens(self, models: list[Any], tokens: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Encode token tensors into model-family text-conditioning outputs.
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


class ConditioningStrategy(ABC):
    """Facet for model-family conditioning resolution used by diffusion/validation."""

    @abstractmethod
    def resolve_conditioning(
        self,
        batch: Any,
        text_encoders: list[Any],
        accelerator: Any,
        cfg: Any,
        train_text_encoder: bool,
        is_train: bool,
        weight_dtype: torch.dtype,
    ) -> Any:
        """
        Resolve the family-specific conditioning payload for a batch.

        Implementations may combine cached text-encoder outputs, cached token
        tensors, live caption encoding, and any family-specific merge policy
        needed before diffusion/denoiser calls.
        """
        raise NotImplementedError


class CachingStrategy(ABC):
    """Strategy for latents and text encoder caching."""

    # --- New/shared-pipeline caching methods ---
    # These create strategies for the CachingEngine pipeline
    # (library/data/caching_engine.py).

    @abstractmethod
    def create_latent_caching_strategy(self, cfg: Any) -> Any:
        """
        Create a new-pipeline CacheBackend for VAE latent caching.

        Returns an instance compatible with library.data.CachingEngine.

        Args:
            cfg: Configuration object.

        Returns:
            A CacheBackend (library.data.caching_engine.CacheBackend) instance.
        """
        raise NotImplementedError

    @abstractmethod
    def create_te_caching_strategy(self, cfg: Any) -> Any:
        """
        Create a new-pipeline CacheBackend for text encoder output caching.

        Returns an instance compatible with library.data.CachingEngine, or None
        if disk-based TE caching is not applicable.

        Args:
            cfg: Configuration object.

        Returns:
            A CacheBackend instance, or None.
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
        denoiser: Any,
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
            denoiser: The denoiser model.
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
        denoiser: Any,
        vae: Any,
        objective_runtime: ObjectiveRuntime,
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


DiffusionBatchLossOutput = BatchLossOutput


class DiffusionTrainingStrategy(ABC):
    """Contract for diffusion-specific batch processing behavior."""

    @abstractmethod
    def process_batch(
        self,
        batch: Any,
        text_encoders: list[Any],
        denoiser: Any,
        trainable_model: Any,
        vae: Any,
        objective_runtime: ObjectiveRuntime,
        vae_dtype: torch.dtype,
        weight_dtype: torch.dtype,
        accelerator: Any,
        cfg: Any,
        is_train: bool = True,
        train_text_encoder: bool = True,
        train_denoiser: bool = True,
        global_step: int = 0,
    ) -> BatchLossOutput:
        """
        Process a batch for training or validation.

        Returns:
            Base per-sample loss and timestep data before trainer-owned modifiers.
        """
        raise NotImplementedError


class DenoiserCallingStrategy(ABC):
    """Strategy for calling the denoiser during training."""

    @abstractmethod
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
        *,
        phase: StrategyPhase,
        global_step: int,
        is_train: bool,
        train_denoiser: bool = True,
        sample_indices: tuple[int, ...] | None = None,
        enable_grad: bool | None = None,
    ) -> torch.Tensor:
        """
        Call the denoiser with architecture-specific arguments.

        Args:
            cfg: Configuration object.
            accelerator: Accelerator instance.
            denoiser: The denoiser model.
            noisy_latents: Input latents with noise.
            timesteps: Timesteps for denoising.
            text_conds: Text conditioning embeddings.
            batch: The current data batch.
            weight_dtype: Data type for calculations.
            phase: Current strategy phase for scoped context publication.
            global_step: Active training step associated with this denoiser call.
            is_train: Whether the surrounding strategy execution is training.
            train_denoiser: Whether gradients should flow to the denoiser input.
            sample_indices: Optional selected sample indices for indexed passes.
            enable_grad: Optional override for grad-enabled execution.

        Returns:
            The noise prediction tensor.
        """
        raise NotImplementedError


class ModelPreparationStrategy(ABC):
    """Strategy hooks used while preparing trainable models and precision."""

    @abstractmethod
    def cast_text_encoder(self, cfg: Any) -> bool:
        """
        Determine if text encoder should be cast to a specific dtype.

        Args:
            cfg: Configuration object.

        Returns:
            True if the text encoder should be cast.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement cast_text_encoder")

    @abstractmethod
    def cast_vae(self, cfg: Any) -> bool:
        """
        Determine if VAE should be cast to a specific dtype.

        Args:
            cfg: Configuration object.

        Returns:
            True if the VAE should be cast.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement cast_vae")

    @abstractmethod
    def cast_denoiser(self, cfg: Any) -> bool:
        """
        Determine if denoiser should be cast to a specific dtype.

        Args:
            cfg: Configuration object.

        Returns:
            True if the denoiser should be cast.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement cast_denoiser")

    def prepare_text_encoder_grad_ckpt_workaround(self, index: int, text_encoder: Any) -> None:
        """
        Set up gradient checkpointing workaround for a text encoder.

        Model families must override this with architecture-specific logic
        (e.g. CLIP needs ``text_model.embeddings.requires_grad_(True)``).

        Args:
            index: Index of the text encoder.
            text_encoder: The text encoder model.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement prepare_text_encoder_grad_ckpt_workaround")

    def prepare_text_encoder_fp8(self, index: int, text_encoder: Any, te_weight_dtype: torch.dtype, weight_dtype: torch.dtype) -> None:
        """
        Prepare text encoder embedding modules for FP8 training.

        ``nn.Embedding`` does not support FP8, so model families must cast the
        embedding layer back to the base weight dtype.  Override with
        architecture-specific logic.

        Args:
            index: Index of the text encoder.
            text_encoder: The text encoder model.
            te_weight_dtype: Target weight dtype for text encoder.
            weight_dtype: General weight dtype.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement prepare_text_encoder_fp8")

    @abstractmethod
    def post_process_trainable(
        self,
        cfg: Any,
        accelerator: Any,
        trainable_model: Any,
        text_encoders: list[Any],
        denoiser: Any,
    ) -> None:
        """
        Post-process the trainable model after creation.

        Args:
            cfg: Configuration object.
            accelerator: Accelerator instance.
            trainable_model: The trainable model.
            text_encoders: List of text encoders.
            denoiser: The denoiser model.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement post_process_trainable")


class TrainingStrategy(
    ModelLoadingStrategy,
    TokenizationStrategy,
    TextEncodingStrategy,
    ConditioningStrategy,
    CachingStrategy,
    SampleGenerationStrategy,
    CheckpointingStrategy,
    ValidationStrategy,
    DiffusionTrainingStrategy,
    DenoiserCallingStrategy,
    ModelPreparationStrategy,
):
    """
    Combined interface for all training strategies.

    Implementations inherit from this and provide model-specific implementations.

    Concrete strategies are expected to be fully formed when instantiated.
    Runner code accesses ``strategy.tokenizers`` and calls strategy methods on
    the strategy itself without a separate initialization phase.
    """
