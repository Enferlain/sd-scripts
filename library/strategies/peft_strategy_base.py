# PEFT Training Strategy interfaces
# Follows the pattern from PEFT_REFACTORING_PLAN.md

import logging
import random
import numpy as np
import torch

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union

from library.utils.common_utils import setup_logging
from library.training.optimizer import should_train_text_encoder, should_train_unet

setup_logging()
logger = logging.getLogger(__name__)


class ModelLoadingStrategy(ABC):
    """Strategy for loading model components (text encoders, VAE, UNet)."""
    
    @abstractmethod
    def load_target_model(self, cfg, weight_dtype, accelerator) -> Tuple[str, Any, Any, Any]:
        """
        Load model components for this architecture.
        
        Returns:
            Tuple of (model_version, text_encoder, vae, unet)
            - text_encoder can be a single model or a list (for SDXL)
            - unet can be None for lazy loading
        """
        raise NotImplementedError


class TokenizationPeftStrategy(ABC):
    """Strategy for tokenization setup in PEFT training."""
    
    @abstractmethod
    def get_tokenize_strategy(self, cfg):
        """Return the appropriate TokenizeStrategy for this architecture."""
        raise NotImplementedError
    
    @abstractmethod
    def get_tokenizers(self, tokenize_strategy):
        """Return tokenizer(s) from the strategy. May be a single tokenizer or tuple."""
        raise NotImplementedError


class CachingPeftStrategy(ABC):
    """Strategy for latents and text encoder caching."""
    
    @abstractmethod
    def get_latents_caching_strategy(self, cfg):
        """Return the LatentsCachingStrategy for this architecture."""
        raise NotImplementedError
    
    @abstractmethod
    def get_text_encoding_strategy(self, cfg):
        """Return the TextEncodingStrategy for this architecture."""
        raise NotImplementedError
    
    @abstractmethod
    def get_text_encoder_outputs_caching_strategy(self, cfg):
        """Return the TextEncoderOutputsCachingStrategy. May be None."""
        raise NotImplementedError
    
    @abstractmethod
    def cache_text_encoder_outputs_if_needed(
        self, cfg, accelerator, unet, vae, text_encoders: List, dataset, weight_dtype
    ):
        """Cache text encoder outputs if caching is enabled."""
        raise NotImplementedError
    
    @abstractmethod
    def get_models_for_text_encoding(self, cfg, accelerator, text_encoders: List) -> List:
        """
        Return models to use for text encoding during training.
        SDXL may return wrapped/unwrapped models differently.
        """
        raise NotImplementedError


class UNetCallingStrategy(ABC):
    """Strategy for calling UNet during training."""
    
    @abstractmethod
    def call_unet(
        self, cfg, accelerator, unet, noisy_latents, timesteps, 
        text_conds, batch, weight_dtype, **kwargs
    ):
        """
        Call UNet with architecture-specific arguments.
        
        SDXL adds added_cond_kwargs for size/crop conditioning.
        """
        raise NotImplementedError


class SampleGenerationPeftStrategy(ABC):
    """Strategy for generating sample images during training."""
    
    @abstractmethod
    def sample_images(
        self, accelerator, cfg, epoch: int, global_step: int,
        device, vae, tokenizers, text_encoders, unet
    ):
        """Generate sample images for the current training step."""
        raise NotImplementedError


class CheckpointingPeftStrategy(ABC):
    """Strategy for model-specific checkpointing and metadata."""
    
    @abstractmethod
    def update_metadata(self, metadata: dict, cfg):
        """Add model-specific metadata fields."""
        raise NotImplementedError
    
    @abstractmethod
    def get_sai_model_spec(self, cfg) -> dict:
        """Get SAI model spec metadata."""
        raise NotImplementedError


class ValidationPeftStrategy(ABC):
    """Strategy for model-specific validation."""
    
    @abstractmethod
    def validate_extra_config(self, cfg, train_dataset_group, val_dataset_group):
        """Perform model-specific config validation."""
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
    
    # Model-specific constants
    vae_scale_factor: float = 0.18215
    is_sdxl: bool = False
    
    # Instance state (set during training)
    la_sampler: Any = field(default=None, init=False, repr=False)
    live_plotter_process: Any = field(default=None, init=False, repr=False)
    
    # --- Additional methods that may need strategy ---
    
    def get_text_encoders_train_flags(self, cfg, text_encoders) -> List[bool]:
        """Return list of flags for whether each text encoder should be trained."""
        return [True] * len(text_encoders) if self.is_train_text_encoder(cfg) else [False] * len(text_encoders)
    
    def is_train_text_encoder(self, cfg) -> bool:
        """Check if text encoder should be trained based on LR config."""
        return should_train_text_encoder(cfg.optimizer)
    
    def is_train_unet(self, cfg) -> bool:
        """Check if UNet should be trained based on LR config."""
        return should_train_unet(cfg.optimizer)
    
    def cast_text_encoder(self, cfg) -> bool:
        return True
    
    def cast_vae(self, cfg) -> bool:
        return True
    
    def cast_unet(self, cfg) -> bool:
        return True
    
    def is_text_encoder_not_needed_for_training(self, cfg) -> bool:
        return False
    
    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        """Set up gradient checkpointing for text encoder."""
        text_encoder.text_model.embeddings.requires_grad_(True)
    
    def prepare_text_encoder_fp8(self, index, text_encoder, te_weight_dtype, weight_dtype):
        """Prepare text encoder for FP8 training."""
        text_encoder.text_model.embeddings.to(dtype=weight_dtype)
    
    def prepare_unet_with_accelerator(self, cfg, accelerator, unet):
        """Prepare UNet with accelerator."""
        return accelerator.prepare(unet)
    
    def post_process_network(self, cfg, accelerator, network, text_encoders, unet):
        """Post-process peft after creation. Override for model-specific behavior."""
        pass
    
    def on_step_start(self, cfg, accelerator, network, text_encoders, unet, batch, weight_dtype, is_train: bool = True):
        """Hook called at the start of each training step."""
        pass

    def on_validation_step_end(self, cfg, accelerator, network, text_encoders, unet, batch, weight_dtype):
        """Hook called after each validation step."""
        pass

    def load_unet_lazily(self, cfg, weight_dtype, accelerator, text_encoders):
        """Load UNet lazily if not loaded in load_target_model. Not used by SD."""
        raise NotImplementedError("load_unet_lazily is not implemented for this architecture")
    
    def all_reduce_network(self, accelerator, network):
        """Sync DDP gradients manually."""
        for param in network.parameters():
            if param.grad is not None:
                param.grad = accelerator.reduce(param.grad, reduction="mean")

    def switch_rng_state(self, val_seed: int, accelerator):
        """Store current RNG states and set new seed for validation."""
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

    def restore_rng_state(self, rng_states, accelerator):
        """Restore RNG states after validation."""
        cpu_rng_state, gpu_rng_state, python_rng_state, numpy_rng_state = rng_states
        
        torch.set_rng_state(cpu_rng_state)
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        
        if gpu_rng_state is not None:
            if accelerator.device.type == "cuda":
                torch.cuda.set_rng_state(gpu_rng_state)
            elif accelerator.device.type == "xpu":
                torch.xpu.set_rng_state(gpu_rng_state)
