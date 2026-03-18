"""
Trainer - Orchestrates model training.

This trainer handles the training loop structure while delegating
model-specific operations to TrainingStrategy classes and
mode-specific operations to TrainingMode plugins.
"""

from __future__ import annotations

import logging
import math
import os
import time
import random
from typing import TYPE_CHECKING, Any
from types import SimpleNamespace

import torch
from torch import nn

from library.logging.resource_monitor import create_resource_monitor
from library.performance import deepspeed_utils
from library.training.noise_utils import get_noise_scheduler
from library.training.trainer_utils import prepare_accelerator
from library.utils.common_utils import setup_logging, suppress_non_main_process_logging
from library.utils.hash_utils import get_git_is_dirty, get_git_revision_hash

from library.utils.torch_utils import set_torch_cuda_reduced_precision, set_seed_from_config, prepare_dtype
from library.data import create_manifest_from_config, get_or_create_manifest, DatasetManifest, Bucket

if TYPE_CHECKING:
    from accelerate import Accelerator
    from library.data.caching_engine import CacheHandler
    from library.strategies.base.training import TrainingStrategy
    from library.training.modes.base import TrainingMode


logger = logging.getLogger(__name__)


# @dataclass
# class StepOutput:
#     """Output from a single training step.

#     Separates model math from loop policy - the trainer returns this
#     and the loop decides what to do with it (checkpoint, log, etc.)
#     """

#     loss: float  # Pre-scaling loss value for logging
#     timesteps: Tensor  # Timesteps used in this batch
#     did_sync: bool  # True if gradients synced (global_step increments)
#     metrics: dict = field(default_factory=dict)  # Optional extra metrics


class Trainer:
    """
    Trainer for model training (PEFT, fine-tune, etc.).

    Orchestrates the training loop while delegating model-specific
    operations to the provided TrainingStrategy and mode-specific
    operations to the provided TrainingMode.

    Usage:
        strategies = SdxlTrainingStrategy(cfg)
        mode = PeftMode()
        trainer = Trainer(cfg, strategies, mode)
        trainer.train()
    """

    def __init__(self, cfg: Any, strategies: TrainingStrategy, mode: TrainingMode):
        """
        Initialize the trainer.

        Args:
            cfg: Hydra config object (e.g., SDXLPeftConfig)
            strategies: Model-specific training strategy
            mode: Training mode plugin (e.g., PeftMode)
        """
        self.cfg = cfg
        self.strategies = strategies
        self.mode = mode

        # Will be set during setup()
        self._accelerator: Accelerator | None = None
        self.device: torch.device | None = None
        self.weight_dtype: torch.dtype | None = None
        self.save_dtype: torch.dtype | None = None
        self.vae_dtype: torch.dtype | None = None
        self.tokenizers: list[Any] = []

        # Will be set during manifest creation
        self.train_manifest: DatasetManifest | None = None
        self.val_manifest: DatasetManifest | None = None

        # Will be set during model loading (in setup)
        self.denoiser: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.text_encoders: list[nn.Module] = []
        self._text_encoder: Any = None  # Original reference for adapter API compatibility

        # Will be set during prepare_models()
        self.adapter: nn.Module | None = None
        self.net_kwargs: dict = {}
        self.denoiser_weight_dtype: torch.dtype | None = None
        self.te_weight_dtype: torch.dtype | None = None

        # Will be set during run_caching()
        self.latent_cache_handler: CacheHandler | None = None
        self.te_cache_handler: CacheHandler | None = None

        # Will be set during prepare_optimizer()
        self.optimizer: Any = None
        self.optimizer_name: str = ""
        self.optimizer_args: dict = {}
        self.optimizer_train_fn: Any = None
        self.optimizer_eval_fn: Any = None
        self.lr_descriptions: list = []
        self.lr_scheduler: Any = None

        # Training state
        self.global_step: int = 0
        self.current_epoch: int = 0
        self.max_train_steps: int = 0
        self.num_train_epochs: int = 0
        self.epoch_to_start: int = 0
        self.num_batches_per_epoch: int = 0

        # Session info
        self.session_id: int = random.randint(0, 2**32)
        self.training_started_at: float = time.time()

        # Metadata for checkpoints (set during setup)
        self._metadata: dict = {}
        self._minimum_metadata: dict = {}

        # Internal state
        self._model_version: str = ""
        self._cache_latents: bool = False
        self._use_dreambooth_method: bool = False
        self._cache_dir: str | None = None
        self._latent_dtype: str = "fp16"
        self._n_workers: int = 0
        self._initial_step: int = 0

        # Training loop state (set before run_training_loop)
        self.noise_scheduler: Any = None
        self._progress_bar: Any = None
        self._loss_recorder: Any = None
        self._loss_scaled_recorder: Any = None
        self._val_loss_recorder: Any = None
        self._is_tracking: bool = False
        self._accumulation_counter: int = 0
        self._current_global_step_loss: float = 0.0
        self._current_global_step_loss_scaled: float | None = 0.0
        self._current_val_loss: float | None = None
        self._average_val_loss: float | None = None

        # Validation scheduler (created during _log_training_info)
        self._validation_scheduler: Any = None
        self._resource_monitor: Any = None

        # Validation state
        self._val_dataloader: Any = None
        self._cyclic_val_dataloader: Any = None
        self._train_text_encoder: bool = False
        self._train_denoiser: bool = True
        self._grad_sync_handle: Any = None  # Object passed to accelerator.accumulate() for grad sync
        self._primary_trainable: nn.Module | None = None  # Semantic trainable model (set by mode)

        # EDM2 state
        self._edm2_model: Any = None
        self._edm2_optimizer: Any = None
        self._edm2_lr_scheduler: Any = None

        # Dynamic timestep schedule
        self._dynamic_timestep_schedule: list | None = None
        self._current_min_timestep: int | None = None
        self._current_max_timestep: int | None = None

        # Live plotter state
        self._timestep_counts: Any = None
        self._plotter_settings: Any = None

        # SimpleNamespace state containers for cross-function state sharing
        self._current_epoch_state: SimpleNamespace = SimpleNamespace(value=0)
        self._current_step_state: SimpleNamespace = SimpleNamespace(value=0)

    def train(self) -> None:
        """Main training entry point. Orchestrates all phases."""
        try:
            self.setup()
            self.run_caching()
            self.prepare_models()
            self.prepare_optimizer()

            self._log_training_info()
            self._maybe_sample_at_start()

            self.run_training_loop()

            self._finalize_training()
        finally:
            if self._resource_monitor is not None:
                self._resource_monitor.end_session()

    # =========================================================================
    # Phase Methods - Delegate to phase functions
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

        self.tokenizers = self.strategies.tokenizers

        # Prepare accelerator first (needed for distributed caching)
        logger.info("preparing accelerator")
        self._accelerator = prepare_accelerator(
            self.cfg.performance.precision,
            self.cfg.performance.compilation,
            self.cfg.performance.distributed,
            self.cfg.performance.deepspeed,
            self.cfg.output.logging,
            self.cfg.training,
        )
        self.device = self.accelerator.device
        suppress_non_main_process_logging(self.accelerator.is_main_process)
        self._resource_monitor = create_resource_monitor(
            accelerator=self.accelerator,
            resource_monitor_config=self.cfg.output.logging.resource_monitor,
            output_dir=self.cfg.output.saving.output_dir,
            run_id=self.session_id,
            config_name=self.cfg.output.saving.output_name,
            git_sha=get_git_revision_hash(),
            git_dirty=get_git_is_dirty(),
        )
        self._resource_monitor.start_session()

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

        # Calculate batches per epoch for step calculations
        train_image_count = sum(e.num_repeats for e in self.train_manifest.entries.values() if not e.is_reg)
        self.num_batches_per_epoch = math.ceil(train_image_count / self.cfg.training.train_batch_size)

        # Prepare dtypes
        self.weight_dtype, self.save_dtype = prepare_dtype(self.cfg.performance.precision, self.cfg.output.saving)
        self.vae_dtype = (
            (torch.float32 if self.cfg.performance.precision.no_half_vae else self.weight_dtype)
            if self.strategies.cast_vae(self.cfg)
            else None
        )

        # Load target models: denoiser may be None for lazy loading
        self._model_version, text_encoder, self.vae, self.denoiser = self.strategies.load_target_model(
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
        from library.training.phases.caching import run_caching

        run_caching(self)

    def prepare_models(self) -> None:
        """Phase 3: Create adapter and configure precision."""
        from library.training.phases.model_prep import prepare_models

        prepare_models(self)

    def prepare_optimizer(self) -> None:
        """Phase 4: Create optimizer, LR scheduler, and calculate training steps."""
        from library.training.phases.optimizer import prepare_optimizer

        prepare_optimizer(self)

    def run_training_loop(self) -> None:
        """Execute the main training loop."""
        from library.training.phases.training_loop import run_training_loop

        run_training_loop(self)

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

    def save_checkpoint(
        self,
        ckpt_name: str,
        target_model: nn.Module,
        step: int,
        epoch: int,
        force_sync_upload: bool = False,
        dtype_override: torch.dtype | None = None,
    ) -> None:
        """Save model checkpoint by delegating to the training mode.

        Args:
            ckpt_name: Checkpoint filename.
            target_model: The model to save. For standard adapter saves
                this is the unwrapped adapter; for EDM2 loss weights
                this is ``_edm2_model``.
            step: Current training step.
            epoch: Current epoch number.
            force_sync_upload: Force synchronous HuggingFace upload.
            dtype_override: Override save dtype (e.g. float32 for EDM2).
        """
        metadata_to_save = self._minimum_metadata.copy() if self.cfg.output.saving.no_metadata else self._metadata.copy()
        modelspec_metadata = self.strategies.get_model_metadata(self.cfg)
        metadata_to_save.update(modelspec_metadata)

        self.mode.save_checkpoint(
            self,
            ckpt_name=ckpt_name,
            step=step,
            epoch=epoch,
            metadata=metadata_to_save,
            force_sync_upload=force_sync_upload,
            dtype_override=dtype_override,
            target_model=target_model,
        )

        self._emit("on_checkpoint", step=step, epoch=epoch)

    def remove_checkpoint(self, old_ckpt_name: str) -> None:
        """Remove old checkpoint file or directory (diffusers format)."""
        import shutil

        old_ckpt_path = os.path.join(self.cfg.output.saving.output_dir, old_ckpt_name)
        if os.path.isdir(old_ckpt_path):
            self.accelerator.print(f"removing old checkpoint directory: {old_ckpt_path}")
            shutil.rmtree(old_ckpt_path)
        elif os.path.isfile(old_ckpt_path):
            self.accelerator.print(f"removing old checkpoint: {old_ckpt_path}")
            os.remove(old_ckpt_path)

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
        """Log training configuration, create metadata, init noise scheduler, and loss recorders."""
        from tqdm import tqdm

        from library.losses.edm2_loss_utils import prepare_edm2_loss_weighting
        from library.losses.loss import EMARecorder
        from library.logging.step_logging import init_trackers
        from library.logging.training_plots import setup_live_plotter
        from library.timesteps.timestep_utils import init_timestep_sampler, parse_dynamic_timestep_schedule
        from library.training.training_metadata import create_training_metadata
        from library.utils.device_utils import clean_memory_on_device

        cfg = self.cfg

        # Calculate total batch size
        total_batch_size = cfg.training.train_batch_size * self.accelerator.num_processes * cfg.training.gradient_accumulation_steps

        # Calculate stats from manifest
        assert self.train_manifest is not None, "train_manifest must be set before _log_training_info"
        num_train_images = sum(e.num_repeats for e in self.train_manifest.entries.values() if not e.is_reg)
        num_reg_images = sum(e.num_repeats for e in self.train_manifest.entries.values() if e.is_reg)
        num_val_images = sum(e.num_repeats for e in self.val_manifest.entries.values()) if self.val_manifest else 0

        # Log training stats
        self.accelerator.print("running training")
        self.accelerator.print(f"  num train images * repeats: {num_train_images}")
        self.accelerator.print(f"  num validation images * repeats: {num_val_images}")
        self.accelerator.print(f"  num reg images: {num_reg_images}")
        self.accelerator.print(f"  num batches per epoch: {self.num_batches_per_epoch}")
        self.accelerator.print(f"  num epochs: {self.num_train_epochs}")
        self.accelerator.print(f"  batch size per device: {cfg.training.train_batch_size}")
        self.accelerator.print(f"  gradient accumulation steps: {cfg.training.gradient_accumulation_steps}")
        self.accelerator.print(f"  total optimization steps: {self.max_train_steps}")

        # --- Training diagnostics block ---
        from library.training.trainer_utils import log_training_diagnostics

        diag_components, diag_aliases = self.mode.get_diagnostics_components(self)

        log_training_diagnostics(
            accelerator=self.accelerator,
            cfg=cfg,
            mode=self.mode,
            strategies=self.strategies,
            components=diag_components,
            optimizer=self.optimizer,
            optimizer_name=self.optimizer_name,
            lr_descriptions=self.lr_descriptions,
            aliases=diag_aliases,
        )
        self._resource_monitor.emit_startup_component_memory(
            diag_components,
            self.optimizer_name,
            deepspeed_enabled=cfg.performance.deepspeed.deepspeed,
            deepspeed_zero_stage=cfg.performance.deepspeed.zero_stage,
        )

        # Create training metadata
        # Convert optimizer_args to a formatted string for metadata (may already be str)
        if isinstance(self.optimizer_args, dict):
            optimizer_args_str = ", ".join(f"{k}={v}" for k, v in self.optimizer_args.items())
        else:
            optimizer_args_str = str(self.optimizer_args) if self.optimizer_args else ""
        self._metadata, self._minimum_metadata = create_training_metadata(
            cfg=cfg,
            manifest=self.train_manifest,
            val_manifest=self.val_manifest,
            session_id=self.session_id,
            training_started_at=self.training_started_at,
            model_version=self._model_version,
            num_train_epochs=self.num_train_epochs,
            optimizer_name=self.optimizer_name,
            optimizer_args=optimizer_args_str,
            net_kwargs=self.net_kwargs,
            num_batches_per_epoch=self.num_batches_per_epoch,
            total_batch_size=total_batch_size,
            use_dreambooth_method=self._use_dreambooth_method,
        )
        self.strategies.update_metadata(self._metadata, cfg)

        # Noise scheduler
        self.noise_scheduler = get_noise_scheduler(cfg, self.accelerator.device)

        # Timestep sampler
        self.strategies.la_sampler = init_timestep_sampler(cfg.timestep, self.noise_scheduler, self.accelerator)

        # Live plotter setup
        self._timestep_counts = None
        self._plotter_settings = None
        if self.is_main_process:
            self._timestep_counts, self._plotter_settings = setup_live_plotter(
                cfg, self.noise_scheduler, self.strategies.la_sampler, self.strategies
            )

        # EDM2 loss weighting
        self._edm2_model, self._edm2_optimizer, self._edm2_lr_scheduler = prepare_edm2_loss_weighting(
            cfg.loss.edm2, cfg.training, self.noise_scheduler, self.accelerator
        )

        # Init trackers
        init_trackers(self.accelerator, cfg.output.logging, "training")

        # Loss recorders
        self._loss_recorder = EMARecorder()
        self._val_loss_recorder = EMARecorder()
        self._loss_scaled_recorder = EMARecorder() if cfg.loss.edm2.edm2_loss_weighting else None

        # Init loss tracking state
        self._current_global_step_loss = 0.0
        self._current_global_step_loss_scaled = 0.0 if cfg.loss.edm2.edm2_loss_weighting else None
        self._current_val_loss = None
        self._average_val_loss = None
        self._accumulation_counter = 0
        self._is_tracking = len(self.accelerator.trackers) > 0

        # Dynamic timestep schedule
        self._dynamic_timestep_schedule, self._current_min_timestep, self._current_max_timestep = parse_dynamic_timestep_schedule(
            cfg.timestep, self.noise_scheduler, self.accelerator
        )

        clean_memory_on_device(self.accelerator.device)

        # Progress bar
        self._progress_bar = tqdm(
            range(self.max_train_steps - self._initial_step), smoothing=0, disable=not self.accelerator.is_local_main_process, desc="steps"
        )

        # Validation scheduler (single source of truth for trigger decisions)
        from library.training.phases.validation import ValidationScheduler

        self._validation_scheduler = ValidationScheduler(cfg.validation)

    def _maybe_sample_at_start(self) -> None:
        """Handle --sample_at_first and run_at_start validation if configured."""
        from library.training.sample_generation import sample_images_check
        from library.training.phases.validation import ValidationStepContext

        cfg = self.cfg

        assert self._validation_scheduler is not None, "_validation_scheduler must be initialized before _maybe_sample_at_start"

        should_sample = sample_images_check(cfg.output.sampling, 0, self.global_step)
        val_ctx = ValidationStepContext(
            global_step=self.global_step,
            epoch_step=0,
            current_epoch=0,
            is_last_step_in_epoch=False,
            is_training_start=True,
            is_training_end=False,
            has_validation_data=self._val_dataloader is not None,
        )
        should_validate = self._validation_scheduler.should_run(val_ctx)

        if should_sample or should_validate:
            # Switch to eval mode
            self.mode.set_eval(self)
            self.optimizer_eval_fn()

            # Sample images (independent of validation)
            if should_sample:
                self.strategies.sample_images(
                    self.accelerator,
                    cfg,
                    0,
                    self.global_step,
                    self.accelerator.device,
                    self.vae,
                    self.tokenizers,
                    self._text_encoder,
                    self.denoiser,
                )

            # Validate (independent of sampling)
            if should_validate:
                assert self.vae_dtype is not None, "vae_dtype must be set"
                assert self.weight_dtype is not None, "weight_dtype must be set"
                self._current_val_loss, self._average_val_loss = self.strategies.calculate_val_loss(
                    self.global_step,
                    0,
                    self.num_batches_per_epoch,
                    self._val_loss_recorder,
                    self._val_dataloader,
                    self._cyclic_val_dataloader,
                    self.trainable_model,
                    self.text_encoders,
                    self.denoiser,
                    self.vae,
                    self.noise_scheduler,
                    self.vae_dtype,
                    self.weight_dtype,
                    self.accelerator,
                    cfg,
                    0,
                    None,
                    self._train_text_encoder,
                )
                self.accelerator.print(f"  val_loss: {self._current_val_loss:.4f}  (avg: {self._average_val_loss:.4f})")

            # Switch back to train mode
            self.optimizer_train_fn()
            self.mode.set_train(self)

            # Ensure VRAM is clean before resuming training
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def _finalize_training(self) -> None:
        """Cleanup and final save after training completes."""
        from library.training.checkpointing import get_last_ckpt_name, save_state_on_train_end

        cfg = self.cfg

        # Update metadata
        self._metadata["ss_training_finished_at"] = str(time.time())

        self.accelerator.end_training()
        self.optimizer_eval_fn()

        # Save state if configured
        if self.is_main_process and (cfg.output.saving.save_state or cfg.output.saving.save_state_on_train_end):
            save_state_on_train_end(cfg.output.saving, self.accelerator)

        # Save final checkpoint
        if self.is_main_process:
            import torch

            assert self.trainable_model is not None, "trainable_model must be set before finalizing"
            unwrapped = self.accelerator.unwrap_model(self.trainable_model)
            ckpt_name = get_last_ckpt_name(cfg.output.saving, "." + cfg.output.saving.save_model_as)
            self.save_checkpoint(ckpt_name, unwrapped, self.global_step, self.num_train_epochs, force_sync_upload=True)

            if cfg.loss.edm2.edm2_loss_weighting:
                loss_weights_ckpt_name = get_last_ckpt_name(cfg.output.saving, "." + cfg.output.saving.save_model_as, "_edm2_loss_weights")
                self.save_checkpoint(
                    loss_weights_ckpt_name,
                    self.accelerator.unwrap_model(self._edm2_model),
                    self.global_step,
                    self.num_train_epochs,
                    force_sync_upload=True,
                    dtype_override=torch.float32,
                )

        logger.info("model saved.")

    @property
    def trainable_model(self) -> nn.Module | None:
        """The primary semantic trainable model (set by mode).

        For PEFT: the adapter module.
        For fine-tune: the denoiser.

        Distinct from ``_grad_sync_handle`` which is the grad-sync wrapper
        """
        return self._primary_trainable

    @property
    def accelerator(self) -> Accelerator:
        """Get accelerator, asserting it was initialized via setup()."""
        if self._accelerator is None:
            raise RuntimeError("Accelerator not initialized. Call setup() first.")
        return self._accelerator

    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self._accelerator.is_main_process if self._accelerator else True
