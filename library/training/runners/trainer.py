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
from pathlib import Path
from typing import TYPE_CHECKING, Any
from types import SimpleNamespace

import torch
from torch import nn

from library.losses.loss_modifiers import LossModifier, NoOpLossModifier
from library.logging.console import MainProcessConsole
from library.logging.phase_tags import (
    EVENT_CHECKPOINT_SAVED,
    PHASE_CHECKPOINT_SAVE,
    PHASE_STARTUP_ACCELERATOR,
    PHASE_STARTUP_DATASET_MANIFEST,
    PHASE_STARTUP_METADATA,
    PHASE_STARTUP_RUNTIME,
    PHASE_STARTUP_SUMMARY,
)
from library.logging.runtime_trace import RuntimeTrace
from library.logging.resource_monitor import create_resource_monitor
from library.logging.reports import is_benchmark_report_enabled, write_run_report
from library.logging.summaries import build_startup_memory_rows, build_trainer_diagnostic_rows, build_training_startup_summary
from library.models import (
    LoadedModelComponent,
    build_component_module_pairs,
    find_loaded_components,
    get_loaded_component_module,
    get_loaded_component_modules,
    update_loaded_component_module,
    update_loaded_component_modules_by_role,
)
from library.objectives import ObjectiveDefinition, build_objective
from library.objectives.base import ObjectiveRuntime
from library.optimization.optimizer_utils import apply_optimizer_runtime_mode
from library.optimization.types import OptimizationPlan
from library.performance import deepspeed_utils
from library.metadata.records import MetadataValue
from library.training.phases.orchestration_helpers import monitored_phase
from library.training.trainer_utils import prepare_accelerator
from library.utils.common_utils import setup_logging, suppress_non_main_process_logging
from library.utils.hash_utils import get_git_is_dirty, get_git_revision_hash

from library.utils.torch_utils import set_torch_cuda_reduced_precision, set_seed_from_config, prepare_dtype
from library.data import create_manifest_from_config, get_or_create_manifest, DatasetManifest, Bucket

if TYPE_CHECKING:
    from accelerate import Accelerator
    from library.data.caching_engine import CacheBackend
    from library.strategies.base.contracts import TrainingStrategy
    from library.training.metadata import TrainingMetadataState
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
    Trainer for model training (adapter, fine-tune, etc.).

    Orchestrates the training loop while delegating model-specific
    operations to the provided TrainingStrategy and mode-specific
    operations to the provided TrainingMode.

    Usage:
        strategies = SdxlTrainingStrategy(cfg)
        mode = AdapterMode()
        trainer = Trainer(cfg, strategies, mode)
        trainer.train()
    """

    def __init__(
        self,
        cfg: Any,
        strategies: TrainingStrategy,
        mode: TrainingMode,
        *,
        process_launched_perf: float | None = None,
    ):
        """
        Initialize the trainer.

        Args:
            cfg: Hydra config object (e.g., RunConfig)
            strategies: Model-specific training strategy
            mode: Training mode plugin (e.g., AdapterMode)
        """
        self.cfg = cfg
        self.strategies = strategies
        self.mode = mode
        self.objective: ObjectiveDefinition = build_objective(cfg)

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
        # Primary top-level model representation: family-declared loaded components.
        self.loaded_components: tuple[LoadedModelComponent, ...] = ()
        self._text_encoder: Any = None  # Original reference for adapter API compatibility

        # Will be set during prepare_models()
        self.adapter: nn.Module | None = None
        self.adapter_method_name: str | None = None
        self.net_kwargs: dict = {}
        self.denoiser_weight_dtype: torch.dtype | None = None
        self.te_weight_dtype: torch.dtype | None = None

        # Will be set during run_caching()
        self.latent_cache_backend: CacheBackend | None = None
        self.te_cache_backend: CacheBackend | None = None

        # Will be set during prepare_optimizer()
        self.optimizer: Any = None
        self.optimizer_name: str = ""
        self.optimizer_args: dict = {}
        self.optimizer_train_fn: Any = None  # Legacy tuple-path compatibility only
        self.optimizer_eval_fn: Any = None  # Legacy tuple-path compatibility only
        self.lr_descriptions: list = []
        self.optimization_plan: OptimizationPlan | None = None
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
        launched_perf = process_launched_perf if process_launched_perf is not None else time.perf_counter()
        self.runtime_trace = RuntimeTrace(launched_perf)
        self._runtime_trace_first_step_started = False
        self._runtime_trace_first_step_synced = False

        # Metadata state for checkpoints (set during setup)
        self._metadata_state: TrainingMetadataState | None = None

        # Internal state
        self._model_version: str = ""
        self._cache_latents: bool = False
        self._use_dreambooth_method: bool = False
        self._cache_dir: str | None = None
        self._latent_dtype: str = "fp16"
        self._n_workers: int = 0
        self._initial_step: int = 0

        # Training loop state (set before run_training_loop)
        self.objective_runtime: ObjectiveRuntime | None = None
        self._progress_bar: Any = None
        self._loss_recorder: Any = None
        self._val_loss_recorder: Any = None
        self._loss_modifier_metric_recorders: dict[str, Any] = {}
        self._is_tracking: bool = False
        self._accumulation_counter: int = 0
        self._current_global_step_loss: float = 0.0
        self._current_loss_modifier_metrics: dict[str, float] = {}
        self._current_val_loss: float | None = None
        self._average_val_loss: float | None = None

        # Validation scheduler (created during _log_training_info)
        self._validation_scheduler: Any = None
        self._resource_monitor: Any = None
        self._console: MainProcessConsole | None = None
        self._observer: Any = None

        # Validation state
        self._val_dataloader: Any = None
        self._cyclic_val_dataloader: Any = None
        self._train_text_encoder: bool = False
        self._train_denoiser: bool = True
        self._grad_sync_handle: Any = None  # Object passed to accelerator.accumulate() for grad sync
        self._primary_trainable: nn.Module | None = None  # Semantic trainable model (set by mode)

        # Optional post-loss modifier runtime state
        self._loss_modifier_runtime = NoOpLossModifier()

        # Live plotter state
        self._timestep_counts: Any = None
        self._plotter_settings: Any = None

        # SimpleNamespace state containers for cross-function state sharing
        self._current_epoch_state: SimpleNamespace = SimpleNamespace(value=0)
        self._current_step_state: SimpleNamespace = SimpleNamespace(value=0)

    def train(self) -> None:
        """Main training entry point. Orchestrates all phases."""
        succeeded = False
        error_message: str | None = None
        try:
            self.setup()
            self.run_caching()
            self.prepare_models()
            self.prepare_optimizer()

            self._initialize_training_run_state()
            self._run_startup_eval_actions()

            self.run_training_loop()

            self._finalize_training()
            succeeded = True
        except Exception as exc:
            error_message = str(exc)
            raise
        finally:
            # Progress-bar/resource cleanup must run for both ordinary failures and
            # real interrupts. The double-Ctrl+C guard only decides when to raise;
            # this unconditional cleanup keeps terminal/report state sane afterward.
            if self._progress_bar is not None:
                self._progress_bar.close()
                self._progress_bar = None
            if self._resource_monitor is not None:
                self._resource_monitor.end_session()
            self.runtime_trace.finish()
            if self.is_main_process and is_benchmark_report_enabled(self.cfg):
                try:
                    report_path = write_run_report(self, succeeded=succeeded, error_message=error_message)
                    self._log_benchmark_report_artifacts(report_path)
                except Exception as exc:  # pragma: no cover - best-effort reporting
                    logger.warning("Failed to write benchmark report: %s", exc)
            self._finish_observer_run(succeeded=succeeded, error_message=error_message)

    # =========================================================================
    # Phase Methods - Delegate to phase functions
    # =========================================================================

    def setup(self) -> None:
        """Phase 1: Initialize accelerator, tokenizers, and create dataset manifests."""

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
        logger.info("[%s] preparing accelerator", PHASE_STARTUP_ACCELERATOR)
        with monitored_phase(self, PHASE_STARTUP_ACCELERATOR):
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
        self._console = MainProcessConsole(is_main_process=self.accelerator.is_main_process)
        from library.logging.metrics import LoggingTrainingObserver, resolve_hydra_config_name

        self._observer = LoggingTrainingObserver(console=self._console)
        self._resource_monitor = create_resource_monitor(
            accelerator=self.accelerator,
            resource_monitor_config=self.cfg.output.logging.resource_monitor,
            output_dir=self.cfg.output.saving.output_dir,
            run_identifier=self.session_id,
            config_name=resolve_hydra_config_name(),
            git_sha=get_git_revision_hash(),
            git_dirty=get_git_is_dirty(),
            metadata_runtime=self._observer.metadata_runtime,
        )
        self._resource_monitor.start_session()

        # Track current epoch/step for checkpointing
        self._current_epoch_state = getattr(self.accelerator.state, "epoch", None) or SimpleNamespace(value=0)
        self._current_step_state = getattr(self.accelerator.state, "step", None) or SimpleNamespace(value=0)

        # Create dataset manifest
        logger.info("[%s] preparing dataset manifest", PHASE_STARTUP_DATASET_MANIFEST)
        self._latent_dtype = "fp32" if self.cfg.performance.precision.no_half_vae else "fp16"
        self._cache_dir = self.cfg.data.caching.cache_dir or self.cfg.data.source.train_data_dir

        with monitored_phase(self, PHASE_STARTUP_DATASET_MANIFEST):
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
        self._model_version, self.loaded_components = self.strategies.load_target_model(self.cfg, self.weight_dtype, self.accelerator)
        self.sync_component_views()

        if self.vae_dtype is None:
            assert self.vae is not None, "vae must be loaded before inferring vae dtype"
            self.vae_dtype = self.vae.dtype
            logger.info(f"vae_dtype is set to {self.vae_dtype} by the model since cast_vae() is false")

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
                this is ``edm2.model``.
            step: Current training step.
            epoch: Current epoch number.
            force_sync_upload: Force synchronous HuggingFace upload.
            dtype_override: Override save dtype (e.g. float32 for EDM2).
        """
        ckpt_path = Path(self.cfg.output.saving.output_dir) / ckpt_name
        logger.info("[%s] saving checkpoint: %s", PHASE_CHECKPOINT_SAVE, str(ckpt_path))
        with monitored_phase(self, PHASE_CHECKPOINT_SAVE):
            self.mode.save_checkpoint(
                self,
                ckpt_name=ckpt_name,
                step=step,
                epoch=epoch,
                metadata=self._build_checkpoint_metadata(ckpt_name=ckpt_name, step=step, epoch=epoch),
                force_sync_upload=force_sync_upload,
                dtype_override=dtype_override,
                target_model=target_model,
            )
            self.runtime_trace.event(EVENT_CHECKPOINT_SAVED)

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

    def _build_checkpoint_metadata(
        self,
        *,
        ckpt_name: str = "checkpoint",
        step: int | None = None,
        epoch: int | None = None,
    ) -> dict[str, str]:
        """Build checkpoint metadata for the main model and any sidecars."""
        metadata_state = self._require_metadata_state()

        modelspec_metadata = self.strategies.get_model_metadata(self.cfg)
        return metadata_state.build_checkpoint_metadata(
            model_metadata=modelspec_metadata,
            no_metadata=self.cfg.output.saving.no_metadata,
            artifact_identifier=ckpt_name,
            step=step,
            epoch=epoch,
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _compute_total_batch_size(self) -> int:
        """Compute effective total batch size across devices and accumulation."""
        cfg = self.cfg
        return cfg.training.train_batch_size * self.accelerator.num_processes * cfg.training.gradient_accumulation_steps

    def log_progress_message(
        self,
        message: str,
        *,
        tag: str | None = None,
        level: str | int = "info",
        stacklevel: int = 2,
    ) -> None:
        """Emit a progress-safe lifecycle log line with a normal logger fallback."""
        if self._console is not None:
            self._console.log_external(message, tag=tag, level=level, stacklevel=stacklevel + 1)
            return
        rendered = f"[{tag}] {message}" if tag is not None else message
        resolved_level = level if isinstance(level, int) else getattr(logging, level.upper())
        logger.log(resolved_level, rendered, stacklevel=stacklevel)

    def print_progress_message(self, message: str) -> None:
        """Emit a progress-safe lifecycle line with the accelerator print fallback."""
        if self._console is not None:
            self._console.print_external(message)
            return
        if self._accelerator is not None:
            self.accelerator.print(message)
            return
        print(message)

    def _log_benchmark_report_artifacts(self, markdown_path: Path | None) -> None:
        """Register generated benchmark report files with the training observer."""
        if markdown_path is None or self._observer is None:
            return

        markdown_path = Path(markdown_path)
        self._observer.log_artifact(
            str(markdown_path),
            kind="benchmark_report",
            metadata={"format": "markdown"},
        )

        json_path = markdown_path.with_suffix(".json")
        if json_path.exists():
            self._observer.log_artifact(
                str(json_path),
                kind="benchmark_report",
                metadata={"format": "json"},
            )

    def _finish_observer_run(self, *, succeeded: bool, error_message: str | None) -> None:
        """Finish observer bookkeeping after final artifacts are registered.

        External tracker shutdown remains owned by Accelerator.end_training()
        during successful training finalization.
        """
        if self._observer is None:
            return
        try:
            self._observer.finish_run(
                status="finished" if succeeded else "failed",
                error_message=error_message,
                global_step=self.global_step,
                epoch=self.current_epoch,
            )
        except Exception as exc:  # pragma: no cover - best-effort logging lifecycle
            logger.warning("Failed to finish logging observer run: %s", exc)

    def _order_memory_components(
        self,
        components: list[tuple[str, nn.Module]],
        component_rows: list[Any],
    ) -> list[tuple[str, nn.Module]]:
        """Align startup resource rows to the same producer-owned component order as diagnostics."""
        if not components or not component_rows:
            return components

        component_by_label = dict(components)
        ordered_labels: list[str] = []
        seen_labels: set[str] = set()

        for row in component_rows:
            label = getattr(row, "label", None)
            if not isinstance(label, str) or label in seen_labels:
                continue
            if label in component_by_label:
                ordered_labels.append(label)
                seen_labels.add(label)

        ordered_components = [(label, component_by_label[label]) for label in ordered_labels]
        ordered_components.extend((label, module) for label, module in components if label not in seen_labels)
        return ordered_components

    def _emit_training_startup_summary(self) -> None:
        """Log dataset/runtime summary and emit startup diagnostics."""
        assert self.train_manifest is not None, "train_manifest must be set before _emit_training_startup_summary"
        component_rows, aliases = build_trainer_diagnostic_rows(self)
        summary = build_training_startup_summary(trainer=self, component_rows=component_rows, aliases=aliases)
        assert self._observer is not None, "observer must be initialized before startup summary"
        self._observer.log_startup_summary(summary)
        memory_components = build_component_module_pairs(self.loaded_components)
        memory_components = self._order_memory_components(memory_components, component_rows)
        memory_rows = build_startup_memory_rows(memory_components, component_rows)
        self._resource_monitor.emit_startup_component_memory(
            memory_rows,
            self.optimizer_name,
            deepspeed_enabled=self.cfg.performance.deepspeed.deepspeed,
            deepspeed_zero_stage=self.cfg.performance.deepspeed.zero_stage,
        )

    def _format_optimizer_args_for_metadata(self) -> str:
        """Format optimizer args into the string metadata form used in checkpoints."""
        if isinstance(self.optimizer_args, dict):
            return ", ".join(f"{k}={v}" for k, v in self.optimizer_args.items())
        return str(self.optimizer_args) if self.optimizer_args else ""

    def _set_training_metadata_fact(self, key: str, value: MetadataValue) -> None:
        """Update training-run metadata facts after initialization."""
        self._metadata_state = self._require_metadata_state().with_fact(key, value)

    def _require_metadata_state(self) -> TrainingMetadataState:
        """Return initialized training metadata state or fail close to misuse."""
        if self._metadata_state is None:
            raise RuntimeError("Training metadata state must be initialized before checkpoint metadata is used")
        return self._metadata_state

    def _initialize_training_metadata(self, *, total_batch_size: int) -> None:
        """Build trainer metadata and let strategies append model-specific fields."""
        from library.metadata.emitters.run import (
            TrainingMetadataBuildContext,
            TrainingMetadataState,
            build_training_metadata_bundle,
        )

        assert self.train_manifest is not None, "train_manifest must be set before metadata initialization"
        bundle = build_training_metadata_bundle(
            TrainingMetadataBuildContext(
                cfg=self.cfg,
                manifest=self.train_manifest,
                val_manifest=self.val_manifest,
                session_id=self.session_id,
                training_started_at=self.training_started_at,
                model_version=self._model_version,
                num_train_epochs=self.num_train_epochs,
                optimizer_name=self.optimizer_name,
                optimizer_args=self._format_optimizer_args_for_metadata(),
                num_batches_per_epoch=self.num_batches_per_epoch,
                total_batch_size=total_batch_size,
                objective=self.objective,
            )
        )
        metadata = dict(bundle.full.compatibility_metadata)
        self.strategies.update_metadata(metadata, self.cfg)
        self._metadata_state = TrainingMetadataState.from_bundle(bundle).with_compatibility_metadata(metadata)

    def _initialize_training_runtime(self) -> None:
        """Initialize runtime helpers that depend on the optimizer and scheduler state."""
        from library.logging.training_plots import setup_live_plotter

        cfg = self.cfg

        objective_runtime = self.objective.build_runtime(cfg, self.accelerator)
        self.objective_runtime = objective_runtime
        self._loss_modifier_runtime = objective_runtime.loss_modifier

        self._timestep_counts = None
        self._plotter_settings = None
        if self.is_main_process:
            self._timestep_counts, self._plotter_settings = setup_live_plotter(
                cfg, objective_runtime, self.strategies
            )

    def _initialize_tracking_state(self) -> None:
        """Initialize trackers, recorders, validation scheduler, and progress state."""
        from library.losses.loss import EMARecorder
        from library.logging.metrics import (
            AccelerateMetricsSink,
            build_tracker_config,
            init_trackers,
            resolve_hydra_config_name,
            resolve_tracker_name,
        )
        from library.training.phases.validation import ValidationScheduler
        from library.utils.device_utils import clean_memory_on_device

        cfg = self.cfg

        init_trackers(self.accelerator, cfg.output.logging, "training")
        if self._observer is not None:
            self._observer.metrics_sink = AccelerateMetricsSink(self.accelerator)
            self._observer.start_run(
                resolve_tracker_name(cfg.output.logging, "training"),
                build_tracker_config(cfg.output.logging),
                run_identifier=str(self.session_id),
                mode_name=type(self.mode).__name__,
                strategy_name=type(self.strategies).__name__,
                optimizer_name=self.optimizer_name or None,
                config_name=resolve_hydra_config_name(),
                global_step=self.global_step,
                epoch=self.current_epoch,
            )

        self._loss_recorder = EMARecorder()
        self._val_loss_recorder = EMARecorder()
        self._loss_modifier_metric_recorders = {}

        self._current_global_step_loss = 0.0
        self._current_loss_modifier_metrics = {}
        self._current_val_loss = None
        self._average_val_loss = None
        self._accumulation_counter = 0
        self._is_tracking = len(self.accelerator.trackers) > 0

        clean_memory_on_device(self.accelerator.device)
        self._validation_scheduler = ValidationScheduler(cfg.validation)

    def _initialize_training_run_state(self) -> None:
        """Initialize the shared trainer runtime state before entering the loop."""
        total_batch_size = self._compute_total_batch_size()
        self._initialize_tracking_state()
        logger.info("[%s] emitting training startup summary", PHASE_STARTUP_SUMMARY)
        with monitored_phase(self, PHASE_STARTUP_SUMMARY):
            self._emit_training_startup_summary()
        logger.info("[%s] assembling training metadata", PHASE_STARTUP_METADATA)
        with monitored_phase(self, PHASE_STARTUP_METADATA):
            self._initialize_training_metadata(total_batch_size=total_batch_size)
        logger.info("[%s] initializing training runtime helpers", PHASE_STARTUP_RUNTIME)
        with monitored_phase(self, PHASE_STARTUP_RUNTIME):
            self._initialize_training_runtime()

    def _compute_startup_eval_actions(self) -> tuple[bool, bool]:
        """Return whether startup sampling and startup validation should run."""
        from library.training.sample_generation import sample_images_check
        from library.training.phases.validation import ValidationStepContext

        cfg = self.cfg

        assert self._validation_scheduler is not None, "_validation_scheduler must be initialized before startup eval"

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
        return should_sample, self._validation_scheduler.should_run(val_ctx)

    def _cleanup_after_startup_eval(self) -> None:
        """Free transient eval memory before training resumes."""
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _run_startup_eval_actions(self) -> None:
        """Run startup sampling/validation if configured by current trigger policy."""
        from library.training.phases.orchestration_helpers import run_sampling_and_validation, temporarily_in_eval_mode

        should_sample, should_validate = self._compute_startup_eval_actions()
        if should_sample or should_validate:
            with temporarily_in_eval_mode(self):
                run_sampling_and_validation(
                    self,
                    should_sample=should_sample,
                    should_validate=should_validate,
                    sample_epoch=0,
                    validation_step=0,
                    validation_epoch=0,
                    validation_batch=None,
                )
            self._cleanup_after_startup_eval()

    def _save_final_state_if_enabled(self) -> None:
        """Persist accelerator state at train end when configured."""
        from library.training.checkpointing import save_state_on_train_end

        cfg = self.cfg
        if self.is_main_process and (cfg.output.saving.save_state or cfg.output.saving.save_state_on_train_end):
            save_state_on_train_end(cfg.output.saving, self.accelerator)

    def _save_final_checkpoint_artifacts(self) -> None:
        """Save final model checkpoint and optional loss-modifier sidecar."""
        from library.training.checkpointing import get_last_ckpt_name

        cfg = self.cfg

        if not self.is_main_process:
            return

        assert self.trainable_model is not None, "trainable_model must be set before finalizing"
        unwrapped = self.accelerator.unwrap_model(self.trainable_model)
        ckpt_name = get_last_ckpt_name(cfg.output.saving, "." + cfg.output.saving.save_model_as)
        self.save_checkpoint(ckpt_name, unwrapped, self.global_step, self.num_train_epochs, force_sync_upload=True)

        if self.loss_modifier.is_enabled and self.loss_modifier.sidecar_suffix:
            loss_weights_ckpt_name = get_last_ckpt_name(
                cfg.output.saving,
                "." + cfg.output.saving.save_model_as,
                self.loss_modifier.sidecar_suffix,
            )
            sidecar_path = os.path.join(cfg.output.saving.output_dir, loss_weights_ckpt_name)
            self.loss_modifier.save_sidecar(
                sidecar_path,
                self._build_checkpoint_metadata(
                    ckpt_name=loss_weights_ckpt_name,
                    step=self.global_step,
                    epoch=self.num_train_epochs,
                ),
            )

    def _finalize_training(self) -> None:
        """Cleanup and final save after training completes."""

        # Update metadata
        self._set_training_metadata_fact("ss_training_finished_at", time.time())

        self.accelerator.end_training()
        if self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None
        apply_optimizer_runtime_mode(self.optimizer, self.optimization_plan, training=False)
        self._save_final_state_if_enabled()
        self._save_final_checkpoint_artifacts()

    def sync_component_views(self) -> None:
        """Refresh compatibility-era component projections from loaded components."""
        text_encoders = self.text_encoders
        if len(text_encoders) > 1:
            self._text_encoder = text_encoders
        else:
            self._text_encoder = text_encoders[0] if text_encoders else None

    def get_loaded_components(
        self,
        *,
        role: str | None = None,
        capability: str | None = None,
        include_unloaded: bool = False,
    ) -> list[LoadedModelComponent]:
        """Return trainer-owned loaded components filtered by declared semantics."""
        return find_loaded_components(
            self.loaded_components,
            role=role,
            capability=capability,
            include_unloaded=include_unloaded,
        )

    @property
    def text_encoders(self) -> list[Any]:
        """Return loaded text-encoder modules in family-declared order."""
        return get_loaded_component_modules(self.loaded_components, role="text_encoder", include_unloaded=True)

    @text_encoders.setter
    def text_encoders(self, modules: list[Any]) -> None:
        self.loaded_components = update_loaded_component_modules_by_role(
            self.loaded_components,
            role="text_encoder",
            modules=modules,
        )
        self.sync_component_views()

    @property
    def vae(self) -> Any | None:
        """Return the first loaded VAE-like module."""
        return get_loaded_component_module(self.loaded_components, role="vae")

    @vae.setter
    def vae(self, module: Any | None) -> None:
        self.loaded_components = update_loaded_component_module(self.loaded_components, role="vae", module=module)

    @property
    def denoiser(self) -> Any | None:
        """Return the first loaded denoiser-like module."""
        return get_loaded_component_module(self.loaded_components, role="denoiser")

    @denoiser.setter
    def denoiser(self, module: Any | None) -> None:
        self.loaded_components = update_loaded_component_module(self.loaded_components, role="denoiser", module=module)

    @property
    def trainable_model(self) -> nn.Module | None:
        """The primary semantic trainable model (set by mode).

        For PEFT: the adapter module.
        For fine-tune: the denoiser.

        Distinct from ``_grad_sync_handle`` which is the grad-sync wrapper
        """
        return self._primary_trainable

    @property
    def loss_modifier(self) -> LossModifier:
        """Trainer-owned post-loss modifier runtime."""
        return self._loss_modifier_runtime

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
