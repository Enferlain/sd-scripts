# PEFT Training Strategy interfaces
# Follows the pattern from PEFT_REFACTORING_PLAN.md

import logging
import random
import numpy as np
import torch

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from diffusers import DDPMScheduler
from typing import Any, List, Tuple, Union, Optional

from library.utils.common_utils import setup_logging
from library.optimizers.optimizer_utils import should_train_text_encoder, should_train_unet
from library.training.noise_utils import (
    prepare_scheduler_for_custom_training,
    fix_noise_scheduler_betas_for_zero_terminal_snr,
)
from library.losses.loss_weighting import (
    apply_snr_weight,
    scale_v_prediction_loss_like_noise_prediction,
    add_v_prediction_like_loss,
    apply_debiased_estimation,
)

setup_logging()
logger = logging.getLogger(__name__)


class ModelLoadingStrategy(ABC):
    """Strategy for loading model components (text encoders, VAE, UNet)."""
    
    @abstractmethod
    def load_target_model(self, cfg: Any, weight_dtype: torch.dtype, accelerator: Any) -> Tuple[str, Any, Any, Any]:
        """
        Load model components for this architecture.
        
        Args:
            cfg: Configuration object containing model settings.
            weight_dtype: Data type for model weights (e.g., torch.float16).
            accelerator: Accelerator instance for handling device placement.

        Returns:
            Tuple of (model_version, text_encoder, vae, unet):
            - model_version: String identifier for the model version.
            - text_encoder: A single text encoder model or a list of text encoders (for SDXL).
            - vae: The VAE model.
            - unet: The UNet model, or None if loaded lazily.
        """
        raise NotImplementedError


class TokenizationPeftStrategy(ABC):
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
    def get_tokenizers(self, tokenize_strategy: Any) -> Union[List[Any], Any]:
        """
        Return tokenizer(s) from the strategy.

        Args:
            tokenize_strategy: The strategy object created by `get_tokenize_strategy`.

        Returns:
            A single tokenizer or a list/tuple of tokenizers.
        """
        raise NotImplementedError


class CachingPeftStrategy(ABC):
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
    
    @abstractmethod
    def get_text_encoder_outputs_caching_strategy(self, cfg: Any) -> Optional[Any]:
        """
        Return the TextEncoderOutputsCachingStrategy.

        Args:
            cfg: Configuration object containing caching settings.

        Returns:
            A TextEncoderOutputsCachingStrategy instance, or None if not supported/enabled.
        """
        raise NotImplementedError
    
    @abstractmethod
    def cache_text_encoder_outputs_if_needed(
        self, cfg: Any, accelerator: Any, unet: Any, vae: Any, text_encoders: List[Any], dataset: Any, weight_dtype: torch.dtype
    ) -> None:
        """
        Cache text encoder outputs if caching is enabled.

        Args:
            cfg: Configuration object.
            accelerator: Accelerator instance.
            unet: The UNet model.
            vae: The VAE model.
            text_encoders: List of text encoder models.
            dataset: The dataset to cache outputs for.
            weight_dtype: Data type for calculations.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_models_for_text_encoding(self, cfg: Any, accelerator: Any, text_encoders: List[Any]) -> List[Any]:
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


class UNetCallingStrategy(ABC):
    """Strategy for calling UNet during training."""
    
    @abstractmethod
    def call_unet(
        self, cfg: Any, accelerator: Any, unet: Any, noisy_latents: torch.Tensor, timesteps: torch.Tensor,
        text_conds: Any, batch: Any, weight_dtype: torch.dtype, **kwargs
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


class SampleGenerationPeftStrategy(ABC):
    """Strategy for generating sample images during training."""
    
    @abstractmethod
    def sample_images(
        self, accelerator: Any, cfg: Any, epoch: int, global_step: int,
        device: torch.device, vae: Any, tokenizers: List[Any], text_encoders: List[Any], unet: Any
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


class CheckpointingPeftStrategy(ABC):
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


class ValidationPeftStrategy(ABC):
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


@dataclass
class PeftTrainingStrategy(
    ModelLoadingStrategy,
    TokenizationPeftStrategy,
    CachingPeftStrategy,
    UNetCallingStrategy,
    SampleGenerationPeftStrategy,
    CheckpointingPeftStrategy,
    ValidationPeftStrategy,
):
    """
    Combined interface for all PEFT training strategies.
    
    Implementations inherit from this and provide model-specific implementations.
    """
    
    # Instance state (set during training)
    la_sampler: Any = field(default=None, init=False, repr=False)
    live_plotter_process: Any = field(default=None, init=False, repr=False)
    
    # --- Shared methods (identical across SD/SDXL) ---
    
    def get_noise_scheduler(self, cfg: Any, device: torch.device) -> Any:
        """
        Create noise scheduler. Same for SD and SDXL.

        Args:
            cfg: Configuration object.
            device: Device to place the scheduler on.

        Returns:
            Initialized DDPMScheduler.
        """
        noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", 
            num_train_timesteps=1000, clip_sample=False
        )

        if cfg.loss.regularization.zero_terminal_snr:
            fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)

        prepare_scheduler_for_custom_training(noise_scheduler, device)
        return noise_scheduler

    def encode_images_to_latents(self, cfg: Any, vae: Any, images: torch.FloatTensor) -> torch.FloatTensor:
        """
        Encode images to latents using VAE.

        Args:
            cfg: Configuration object.
            vae: VAE model instance.
            images: Batch of images to encode.

        Returns:
            Encoded latents.
        """
        return vae.encode(images).latent_dist.sample()

    def shift_scale_latents(self, cfg: Any, latents: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply VAE scale factor to latents. Uses self.vae_latent_scale from child class.

        Args:
            cfg: Configuration object.
            latents: Latents tensor to scale.

        Returns:
            Scaled latents.
        """
        return latents * self.vae_latent_scale  # Child class must define vae_latent_scale

    def post_process_loss(self, loss: torch.Tensor, cfg: Any, timesteps: torch.IntTensor, noise_scheduler: Any) -> torch.FloatTensor:
        """
        Apply SNR weighting, v-pred scaling, debiased estimation etc.

        Args:
            loss: Raw loss tensor.
            cfg: Configuration object.
            timesteps: Timesteps associated with the loss.
            noise_scheduler: Noise scheduler instance.

        Returns:
            Processed loss tensor.
        """
        if cfg.loss.snr.min_snr_gamma:
            loss = apply_snr_weight(loss, timesteps, noise_scheduler, cfg.loss.snr.min_snr_gamma, cfg.loss.v_parameterization)
        if cfg.loss.snr.scale_v_pred_loss_like_noise_pred:
            loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
        if cfg.loss.snr.v_pred_like_loss:
            loss = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, cfg.loss.snr.v_pred_like_loss)
        if cfg.loss.snr.debiased_estimation_loss:
            loss = apply_debiased_estimation(loss, timesteps, noise_scheduler, cfg.loss.v_parameterization)
        return loss

    # --- Additional methods that may need strategy ---
    
    def get_text_encoders_train_flags(self, cfg: Any, text_encoders: List[Any]) -> List[bool]:
        """
        Return list of flags for whether each text encoder should be trained.

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
        return should_train_text_encoder(cfg.optimizer)
    
    def is_train_unet(self, cfg: Any) -> bool:
        """
        Check if UNet should be trained based on LR config.

        Args:
            cfg: Configuration object.

        Returns:
            True if UNet should be trained, False otherwise.
        """
        return should_train_unet(cfg.optimizer)
    
    def cast_text_encoder(self, cfg: Any) -> bool:
        """
        Determine if text encoder should be cast to a specific dtype.

        Args:
            cfg: Configuration object.

        Returns:
            True (default implementation).
        """
        return True
    
    def cast_vae(self, cfg: Any) -> bool:
        """
        Determine if VAE should be cast to a specific dtype.

        Args:
            cfg: Configuration object.

        Returns:
            True (default implementation).
        """
        return True
    
    def cast_unet(self, cfg: Any) -> bool:
        """
        Determine if UNet should be cast to a specific dtype.

        Args:
            cfg: Configuration object.

        Returns:
            True (default implementation).
        """
        return True
    
    def is_text_encoder_not_needed_for_training(self, cfg: Any) -> bool:
        """
        Check if text encoder is unnecessary for training.

        Args:
            cfg: Configuration object.

        Returns:
            False (default implementation).
        """
        return False
    
    def prepare_text_encoder_grad_ckpt_workaround(self, index: int, text_encoder: Any) -> None:
        """
        Set up gradient checkpointing for text encoder.

        Args:
            index: Index of the text encoder.
            text_encoder: The text encoder model.
        """
        text_encoder.text_model.embeddings.requires_grad_(True)
    
    def prepare_text_encoder_fp8(self, index: int, text_encoder: Any, te_weight_dtype: torch.dtype, weight_dtype: torch.dtype) -> None:
        """
        Prepare text encoder for FP8 training.

        Args:
            index: Index of the text encoder.
            text_encoder: The text encoder model.
            te_weight_dtype: Target weight dtype for text encoder.
            weight_dtype: General weight dtype.
        """
        text_encoder.text_model.embeddings.to(dtype=weight_dtype)
    
    def prepare_unet_with_accelerator(self, cfg: Any, accelerator: Any, unet: Any) -> Any:
        """
        Prepare UNet with accelerator.

        Args:
            cfg: Configuration object.
            accelerator: Accelerator instance.
            unet: The UNet model.

        Returns:
            Prepared UNet model.
        """
        return accelerator.prepare(unet)
    
    def post_process_adapter(self, cfg: Any, accelerator: Any, adapter: Any, text_encoders: List[Any], unet: Any) -> None:
        """
        Post-process adapter after creation. Override for model-specific behavior.

        Args:
            cfg: Configuration object.
            accelerator: Accelerator instance.
            adapter: The adapter model.
            text_encoders: List of text encoders.
            unet: The UNet model.
        """
        pass
    
    def on_step_start(self, cfg: Any, accelerator: Any, adapter: Any, text_encoders: List[Any], unet: Any, batch: Any, weight_dtype: torch.dtype, is_train: bool = True) -> None:
        """
        Hook called at the start of each training step.

        Args:
            cfg: Configuration object.
            accelerator: Accelerator instance.
            adapter: The adapter model.
            text_encoders: List of text encoders.
            unet: The UNet model.
            batch: The current data batch.
            weight_dtype: Weight data type.
            is_train: Boolean indicating training mode.
        """
        pass

    def on_validation_step_end(self, cfg: Any, accelerator: Any, adapter: Any, text_encoders: List[Any], unet: Any, batch: Any, weight_dtype: torch.dtype) -> None:
        """
        Hook called after each validation step.

        Args:
            cfg: Configuration object.
            accelerator: Accelerator instance.
            adapter: The adapter model.
            text_encoders: List of text encoders.
            unet: The UNet model.
            batch: The current data batch.
            weight_dtype: Weight data type.
        """
        pass

    def load_unet_lazily(self, cfg: Any, weight_dtype: torch.dtype, accelerator: Any, text_encoders: List[Any]) -> Any:
        """
        Load UNet lazily if not loaded in load_target_model. Not used by SD.

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
    
    def all_reduce_adapter(self, accelerator: Any, adapter: Any) -> None:
        """
        Sync DDP gradients manually.

        Args:
            accelerator: Accelerator instance.
            adapter: The adapter model containing parameters to sync.
        """
        for param in adapter.parameters():
            if param.grad is not None:
                param.grad = accelerator.reduce(param.grad, reduction="mean")

    def switch_rng_state(self, val_seed: int, accelerator: Any) -> Tuple[Any, Any, Any, Any]:
        """
        Store current RNG states and set new seed for validation.

        Args:
            val_seed: Seed to use for validation.
            accelerator: Accelerator instance.

        Returns:
            Tuple containing CPU, GPU, Python, and Numpy RNG states.
        """
        cpu_rng_state = torch.get_rng_state()
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        
        gpu_rng_state = None
        if accelerator.device.type == "cuda":
            gpu_rng_state = torch.cuda.get_rng_state()
        elif accelerator.device.type == "xpu":
            gpu_rng_state = torch.xpu.get_rng_state()

        random.seed(val_seed)
        np.random.seed(val_seed)
        torch.manual_seed(val_seed)
        if accelerator.device.type == "cuda":
            torch.cuda.manual_seed_all(val_seed)

        return (cpu_rng_state, gpu_rng_state, python_rng_state, numpy_rng_state)

    def restore_rng_state(self, rng_states: Tuple[Any, Any, Any, Any], accelerator: Any) -> None:
        """
        Restore RNG states after validation.

        Args:
            rng_states: Tuple of RNG states returned by switch_rng_state.
            accelerator: Accelerator instance.
        """
        cpu_rng_state, gpu_rng_state, python_rng_state, numpy_rng_state = rng_states
        
        torch.set_rng_state(cpu_rng_state)
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        
        if gpu_rng_state is not None:
            if accelerator.device.type == "cuda":
                torch.cuda.set_rng_state(gpu_rng_state)
            elif accelerator.device.type == "xpu":
                torch.xpu.set_rng_state(gpu_rng_state)
