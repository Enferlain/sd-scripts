"""
PEFT Trainer - Orchestrates LoRA/adapter training.

This trainer handles the training loop structure while delegating
model-specific operations to TrainingStrategy classes.
"""

from __future__ import annotations

import logging
import math
import os
import time
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from types import SimpleNamespace

import torch
from torch import nn, Tensor
from tqdm import tqdm

import library.strategies.base.tokenization
import library.strategies.base.caching
from library.performance import deepspeed_utils
from library.training.trainer_utils import prepare_accelerator
from library.utils.common_utils import setup_logging
from library.utils.torch_utils import set_torch_cuda_reduced_precision, set_seed_from_config, prepare_dtype
from library.data import create_manifest_from_config, get_or_create_manifest, DatasetManifest, Bucket


if TYPE_CHECKING:
    from accelerate import Accelerator
    from library.data import DatasetManifest, EpochManifest
    from library.strategies.base.training import TrainingStrategy


setup_logging()
logger = logging.getLogger(__name__)


@dataclass
class StepOutput:
    """Output from a single training step.

    Separates model math from loop policy - the trainer returns this
    and the loop decides what to do with it (checkpoint, log, etc.)
    """

    loss: float  # Pre-scaling loss value for logging
    timesteps: Tensor  # Timesteps used in this batch
    did_sync: bool  # True if gradients synced (global_step increments)
    metrics: dict = field(default_factory=dict)  # Optional extra metrics


class PeftTrainer:
    """
    Trainer for PEFT (LoRA/adapter) training.

    Orchestrates the training loop while delegating model-specific
    operations to the provided TrainingStrategy.

    Usage:
        strategies = SdxlTrainingStrategy()
        trainer = PeftTrainer(cfg, strategies)
        trainer.train()
    """

    def __init__(self, cfg: Any, strategies: TrainingStrategy):
        """
        Initialize the trainer.

        Args:
            cfg: Hydra config object (e.g., SDXLPeftConfig)
            strategies: Model-specific training strategy
        """
        self.cfg = cfg
        self.strategies = strategies

        # Will be set during setup()
        self.accelerator: Accelerator | None = None
        self.device: torch.device | None = None
        self.weight_dtype: torch.dtype | None = None
        self.save_dtype: torch.dtype | None = None
        self.vae_dtype: torch.dtype | None = None
        self.tokenizers: list[Any] = []

        # Will be set during manifest creation
        self.train_manifest: DatasetManifest | None = None
        self.val_manifest: DatasetManifest | None = None

        # Will be set during prepare_models()
        self.unet: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.text_encoders: list[nn.Module] = []
        self.adapter: nn.Module | None = None
        self.noise_scheduler: Any = None

        # Will be set during prepare_optimizer()
        self.optimizer: Any = None
        self.lr_scheduler: Any = None
        self.optimizer_train_fn: Any = None
        self.optimizer_eval_fn: Any = None

        # Training state
        self.global_step: int = 0
        self.current_epoch: int = 0
        self.max_train_steps: int = 0
        self.num_train_epochs: int = 0
        self.epoch_to_start: int = 0

        # Session info
        self.session_id: int = random.randint(0, 2**32)
        self.training_started_at: float = time.time()

    def train(self) -> None:
        """Main training entry point. Orchestrates all phases."""
        self.setup()
        self.run_caching()
        self.prepare_models()
        self.prepare_optimizer()

        self._log_training_info()
        self._maybe_sample_at_start()

        self.run_training_loop()

        self._finalize_training()

    # =========================================================================
    # Phase Methods - Override in subclasses if needed
    # =========================================================================

    def setup(self) -> None:
        """Phase 1: Initialize accelerator, tokenizers, and create dataset manifests."""

        self.strategies.la_sampler = None

        set_torch_cuda_reduced_precision(self.cfg.performance.precision)
        deepspeed_utils.prepare_deepspeed_config(self.cfg.performance.deepspeed, self.cfg.data.loader)
        setup_logging(self.cfg.output.logging, reset=True)

        # Validate required config fields
        if not self.cfg.output.saving.save_model_as:
            raise ValueError("save_model_as must be specified (safetensors, ckpt, or diffusers)")

        self._cache_latents = self.cfg.data.caching.cache_latents
        self._use_dreambooth_method = self.cfg.data.source.in_json is None

        set_seed_from_config(self.cfg.training)

        tokenize_strategy = self.strategies.get_tokenize_strategy(self.cfg)
        library.strategies.base.tokenization.TokenizeStrategy.set_strategy(tokenize_strategy)
        self.tokenizers = self.strategies.get_tokenizers(tokenize_strategy)
        self._tokenize_strategy = tokenize_strategy

        # prepare caching strategy: this must be set before preparing dataset
        latents_caching_strategy = self.strategies.get_latents_caching_strategy(self.cfg)
        library.strategies.base.caching.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

        # Prepare accelerator first (needed for distributed caching)
        logger.info("preparing accelerator")
        self.accelerator = prepare_accelerator(
            self.cfg.performance.precision,
            self.cfg.performance.compilation,
            self.cfg.performance.distributed,
            self.cfg.performance.deepspeed,
            self.cfg.output.logging,
            self.cfg.training,
        )
        self.device = self.accelerator.device

        # Track current epoch/step for checkpointing
        self._current_epoch_state = getattr(self.accelerator.state, "epoch", None) or SimpleNamespace(value=0)
        self._current_step_state = getattr(self.accelerator.state, "step", None) or SimpleNamespace(value=0)

        # Create dataset manifest
        logger.info("Preparing dataset manifest")
        self._latent_dtype = "fp32" if self.cfg.performance.precision.no_half_vae else "fp16"
        self._cache_dir = self.cfg.data.caching.cache_dir or self.cfg.data.source.train_data_dir

        if self.cfg.data.source.val_data_dir:
            # Separate validation directory - create train manifest without val split
            self.train_manifest = create_manifest_from_config(
                data_config=self.cfg.data,
                cache_dir=self._cache_dir,
                latent_dtype=self._latent_dtype,
                validation=False,
            )
            self.val_manifest = create_manifest_from_config(
                data_config=self.cfg.data,
                cache_dir=self._cache_dir,
                latent_dtype=self._latent_dtype,
                validation=True,
            )
        else:
            # Use get_or_create_manifest for persistence and validation split
            self.train_manifest, self.val_manifest = get_or_create_manifest(
                data_config=self.cfg.data,
                cache_dir=self._cache_dir,
                latent_dtype=self._latent_dtype,
                validation_split=self.cfg.validation.validation_split,
                validation_seed=self.cfg.validation.validation_seed,
            )

            # If validation split was used, filter train entries
            if self.val_manifest is not None:
                train_entries = {k: v for k, v in self.train_manifest.entries.items() if v.split == "train"}
                train_buckets = {}
                for bucket_key, bucket in self.train_manifest.buckets.items():
                    train_ids = [img_id for img_id in bucket.image_ids if img_id in train_entries]
                    if train_ids:
                        train_buckets[bucket_key] = Bucket(resolution=bucket.resolution, image_ids=train_ids)
                self.train_manifest = DatasetManifest(
                    version=self.train_manifest.version,
                    created_at=self.train_manifest.created_at,
                    base_resolution=self.train_manifest.base_resolution,
                    bucket_reso_steps=self.train_manifest.bucket_reso_steps,
                    min_bucket_reso=self.train_manifest.min_bucket_reso,
                    max_bucket_reso=self.train_manifest.max_bucket_reso,
                    latent_channels=self.train_manifest.latent_channels,
                    latent_scale_factor=self.train_manifest.latent_scale_factor,
                    latent_dtype=self.train_manifest.latent_dtype,
                    entries=train_entries,
                    buckets=train_buckets,
                )

        # Prepare dtypes
        self.weight_dtype, self.save_dtype = prepare_dtype(self.cfg.performance.precision, self.cfg.output.saving)
        self.vae_dtype = (
            (torch.float32 if self.cfg.performance.precision.no_half_vae else self.weight_dtype)
            if self.strategies.cast_vae(self.cfg)
            else None
        )

        # Load target models: unet may be None for lazy loading
        self._model_version, text_encoder, self.vae, self.unet = self.strategies.load_target_model(
            self.cfg, self.weight_dtype, self.accelerator
        )

        if self.vae_dtype is None:
            self.vae_dtype = self.vae.dtype
            logger.info(f"vae_dtype is set to {self.vae_dtype} by the model since cast_vae() is false")

        # text_encoder is List[CLIPTextModel] or CLIPTextModel
        self.text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]
        self._text_encoder = text_encoder  # Keep original reference for compatibility

    def run_caching(self) -> None:
        """Phase 2: Cache latents and optionally text encoder outputs."""
        # TODO: Extract from sdxl_peft.py lines ~220-400
        raise NotImplementedError("run_caching() not yet implemented - extract from sdxl_peft.py")

    def prepare_models(self) -> None:
        """Phase 3: Load UNet, VAE, text encoders and adapter. Configure for training."""
        # TODO: Extract from sdxl_peft.py lines ~400-660
        raise NotImplementedError("prepare_models() not yet implemented - extract from sdxl_peft.py")

    def prepare_optimizer(self) -> None:
        """Phase 4: Create optimizer, LR scheduler, and calculate training steps."""
        # TODO: Extract from sdxl_peft.py lines ~500-680
        raise NotImplementedError("prepare_optimizer() not yet implemented - extract from sdxl_peft.py")

    def run_training_loop(self) -> None:
        """Phase 5: Execute the main training loop."""
        # TODO: Extract from sdxl_peft.py lines ~900-1240
        raise NotImplementedError("run_training_loop() not yet implemented - extract from sdxl_peft.py")

    def train_step(self, batch: dict, step: int) -> StepOutput:
        """Execute a single training step. Returns StepOutput for loop decisions."""
        # TODO: Extract core step logic from sdxl_peft.py
        raise NotImplementedError("train_step() not yet implemented - extract from sdxl_peft.py")

    # =========================================================================
    # Lifecycle Hooks
    # =========================================================================

    def on_epoch_start(self, epoch: int) -> None:
        """Called at the start of each epoch."""
        self.current_epoch = epoch + 1
        self.accelerator.print(f"\nepoch {self.current_epoch}/{self.num_train_epochs}")
        self._emit("on_epoch_start", epoch=epoch)

    def on_epoch_end(self, epoch: int) -> None:
        """Called at the end of each epoch."""
        self._emit("on_epoch_end", epoch=epoch)

    def save_checkpoint(self, step: int, epoch: int, final: bool = False) -> None:
        """Save model checkpoint."""
        # TODO: Extract save_model closure from sdxl_peft.py
        self._emit("on_checkpoint", step=step, epoch=epoch, final=final)
        raise NotImplementedError("save_checkpoint() not yet implemented")

    # =========================================================================
    # Event System (for future callback extensibility)
    # =========================================================================

    def _emit(self, event: str, **kwargs) -> None:
        """Internal event hook. No-op by default, override for extensions."""
        pass  # Future: dispatch to registered callbacks

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _log_training_info(self) -> None:
        """Log training configuration summary."""
        # TODO: Extract logging from sdxl_peft.py lines ~690-705
        pass

    def _maybe_sample_at_start(self) -> None:
        """Handle --sample_at_first if configured."""
        # TODO: Extract from sdxl_peft.py lines ~819-851
        pass

    def _finalize_training(self) -> None:
        """Cleanup and final save after training completes."""
        # TODO: Extract from sdxl_peft.py lines ~1307-1334
        pass

    def _create_epoch_dataloader(self, epoch: int) -> Any:
        """Create DataLoader for a specific epoch."""
        # TODO: Extract from sdxl_peft.py epoch manifest + dataloader creation
        raise NotImplementedError("_create_epoch_dataloader() not yet implemented")

    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.accelerator.is_main_process if self.accelerator else True
