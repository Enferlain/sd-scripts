from __future__ import annotations

import time
import logging
from dataclasses import asdict, dataclass, field
from typing import Protocol, runtime_checkable

import torch
from accelerate import Accelerator
from omegaconf import OmegaConf

from library.metadata.backends import MetadataSnapshot
from library.metadata.dataclasses.observability import (
    AnalyticsSnapshotFacts,
    LoggedArtifactFacts as LoggedArtifact,
    RunLifecycleFacts,
)
from library.metadata.runtime import MetadataRuntime, MetadataRuntimeItem
from library.config.dataclasses.output import LoggingConfig
from library.logging.console import MainProcessConsole
from library.logging.summaries import TrainingStartupSummary

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LearningRateMetric:
    """A labeled optimizer LR value, plus optional optimizer-derived effective LR."""

    label: str
    value: float
    derived_value: float | None = None


@dataclass(frozen=True, slots=True)
class SamplerStepMetrics:
    """Structured sampler facts rendered into the flat tracker key namespace."""

    ema_loss_mean: float | None = None
    ema_loss_std: float | None = None
    ema_loss_bins: tuple[float, ...] = ()
    entropy_ratio: float | None = None
    timestep_histogram: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class StepMetricsEvent:
    """Typed runtime event for one training-step metrics emission."""

    current_loss: float
    average_loss: float
    learning_rates: tuple[LearningRateMetric, ...] = ()
    modifier_lrs: tuple[tuple[str, float], ...] = ()
    current_loss_scaled: float | None = None
    average_loss_scaled: float | None = None
    keys_scaled: float | None = None
    maximum_norm: float | None = None
    mean_norm: float | None = None
    mean_grad_norm: float | None = None
    mean_combined_norm: float | None = None
    current_val_loss: float | None = None
    average_val_loss: float | None = None
    sampler: SamplerStepMetrics | None = None
    schedule_free_derived_lr: float | None = None


@runtime_checkable
class MetricsSink(Protocol):
    def start_run(self, run_name: str, config: dict[str, object]) -> None: ...

    def log_metrics(self, metrics: dict[str, float], *, step: int, epoch: int | None = None) -> None: ...

    def finish_run(self) -> None: ...


@runtime_checkable
class TrainingObserver(Protocol):
    def start_run(
        self,
        run_name: str,
        config: dict[str, object],
        *,
        run_identifier: str | None = None,
        mode_name: str | None = None,
        strategy_name: str | None = None,
        optimizer_name: str | None = None,
        config_name: str | None = None,
        global_step: int | None = None,
        epoch: int | None = None,
    ) -> None: ...

    def log_metrics(self, step: int, metrics: dict[str, float], *, epoch: int | None = None) -> None: ...

    def log_console(self, message: str, *, level: str = "info", tag: str | None = None) -> None: ...

    def log_startup_summary(self, summary: TrainingStartupSummary) -> None: ...

    def log_artifact(self, path: str, *, kind: str, metadata: dict[str, object] | None = None) -> None: ...

    def finish_run(
        self,
        *,
        status: str = "finished",
        error_message: str | None = None,
        global_step: int | None = None,
        epoch: int | None = None,
    ) -> None: ...


@dataclass(slots=True)
class AccelerateMetricsSink:
    accelerator: Accelerator

    def start_run(self, run_name: str, config: dict[str, object]) -> None:
        """Record no external side effect; tracker init is still owned by init_trackers()."""
        del run_name, config

    def log_metrics(self, metrics: dict[str, float], *, step: int, epoch: int | None = None) -> None:
        log_metrics_to_trackers(self.accelerator, metrics, step, step, epoch if epoch is not None else 0)

    def finish_run(self) -> None:
        """Finish observer bookkeeping only; Accelerator.end_training() remains trainer-owned."""
        return None


@dataclass(slots=True)
class LoggingTrainingObserver:
    console: MainProcessConsole
    metrics_sink: MetricsSink | None = None
    metadata_runtime: MetadataRuntime = field(default_factory=MetadataRuntime)
    artifacts: list[LoggedArtifact] = field(default_factory=list)
    run_name: str | None = None
    run_config: dict[str, object] = field(default_factory=dict)
    run_identifier: str | None = None
    mode_name: str | None = None
    strategy_name: str | None = None
    optimizer_name: str | None = None
    config_name: str | None = None
    _has_started: bool = field(default=False, init=False, repr=False)
    _run_active: bool = field(default=False, init=False, repr=False)
    _run_started_at: float | None = field(default=None, init=False, repr=False)

    def start_run(
        self,
        run_name: str,
        config: dict[str, object],
        *,
        run_identifier: str | None = None,
        mode_name: str | None = None,
        strategy_name: str | None = None,
        optimizer_name: str | None = None,
        config_name: str | None = None,
        global_step: int | None = None,
        epoch: int | None = None,
    ) -> None:
        if self._has_started:
            raise RuntimeError("LoggingTrainingObserver is single-run and cannot be reused.")
        if not run_name:
            raise ValueError("LoggingTrainingObserver.start_run() requires a non-empty run_name.")

        self._has_started = True
        self.run_name = run_name
        self.run_identifier = run_identifier or run_name
        self.run_config = dict(config)
        self.mode_name = mode_name
        self.strategy_name = strategy_name
        self.optimizer_name = optimizer_name
        self.config_name = config_name
        self._run_active = True
        self._run_started_at = time.perf_counter()
        self._file_metadata_item(
            RunLifecycleFacts(
                run_identifier=self.run_identifier,
                event_type="run_started",
                run_name=run_name,
                status="running",
                mode_name=self.mode_name,
                strategy_name=self.strategy_name,
                optimizer_name=self.optimizer_name,
                config_name=self.config_name,
                global_step=global_step,
                epoch=epoch,
            )
        )
        if self.metrics_sink is not None:
            self.metrics_sink.start_run(run_name, self.run_config)

    def log_metrics(self, step: int, metrics: dict[str, float], *, epoch: int | None = None) -> None:
        if self.metrics_sink is None:
            return
        self.metrics_sink.log_metrics(metrics, step=step, epoch=epoch)

    def log_console(self, message: str, *, level: str = "info", tag: str | None = None) -> None:
        self.console.log(message, level=level, tag=tag, stacklevel=4)

    def log_startup_summary(self, summary: TrainingStartupSummary) -> None:
        self.console.log_startup_summary(summary, stacklevel=4)
        if self.run_identifier is None:
            logger.warning("Skipping startup summary metadata filing because no run identifier is available yet.")
            return
        self._file_metadata_item(
            AnalyticsSnapshotFacts(
                snapshot_identifier=f"startup-summary:{self.run_identifier}",
                snapshot_kind="training_startup_summary",
                source="library.logging.metrics.LoggingTrainingObserver.log_startup_summary",
                payload=asdict(summary),
                run_identifier=self.run_identifier,
            )
        )

    def log_artifact(self, path: str, *, kind: str, metadata: dict[str, object] | None = None) -> None:
        artifact = LoggedArtifact(path=path, kind=kind, metadata=dict(metadata or {}))
        self.artifacts.append(artifact)
        self._file_metadata_item(artifact)

    def finish_run(
        self,
        *,
        status: str = "finished",
        error_message: str | None = None,
        global_step: int | None = None,
        epoch: int | None = None,
    ) -> None:
        """Finish the observer-local run lifecycle and notify the metrics sink if present."""
        if not self._run_active:
            return
        self._run_active = False
        duration_ms = None
        if self._run_started_at is not None:
            duration_ms = (time.perf_counter() - self._run_started_at) * 1000.0
        if self.run_identifier is not None:
            self._file_metadata_item(
                RunLifecycleFacts(
                    run_identifier=self.run_identifier,
                    event_type="run_finished",
                    run_name=self.run_name,
                    status=status,
                    mode_name=self.mode_name,
                    strategy_name=self.strategy_name,
                    optimizer_name=self.optimizer_name,
                    config_name=self.config_name,
                    global_step=global_step,
                    epoch=epoch,
                    duration_ms=duration_ms,
                    error_message=error_message,
                )
            )
        self._run_started_at = None
        if self.metrics_sink is not None:
            self.metrics_sink.finish_run()

    def metadata_snapshot(self) -> MetadataSnapshot:
        """Return the observer's collected observability metadata snapshot."""
        return self.metadata_runtime.snapshot()

    def _file_metadata_item(self, item: MetadataRuntimeItem) -> None:
        """File one accepted typed metadata item through the shared metadata runtime."""
        self.metadata_runtime.file(item)


def _resolve_lr_descriptions(lr_descriptions: list[str] | None, optimization_plan=None) -> list[str]:
    """Resolve trainer-facing LR labels from plan metadata when available."""
    if optimization_plan is None:
        return list(lr_descriptions or [])
    return list(getattr(optimization_plan, "lr_descriptions", lr_descriptions or []))


def _validate_lr_label_count(labels: list[str], lrs: list[float]) -> None:
    if len(labels) < len(lrs):
        raise ValueError(
            f"learning-rate labels has {len(labels)} elements but lr_scheduler has {len(lrs)} learning rates. "
            "Ensure labels match the number of parameter groups."
        )


def _optimizer_type_name(cfg) -> str:
    return cfg.optimizer.optimizer_type.lower()


def _build_learning_rate_metrics(
    cfg,
    lr_scheduler,
    labels: list[str],
) -> tuple[LearningRateMetric, ...]:
    lrs = list(lr_scheduler.get_last_lr())
    _validate_lr_label_count(labels, lrs)

    optimizer_type = _optimizer_type_name(cfg)
    metrics: list[LearningRateMetric] = []
    for index, lr in enumerate(lrs):
        derived_value = None
        if optimizer_type.startswith("dadapt") or optimizer_type == "prodigy":
            derived_value = (
                lr_scheduler.optimizers[-1].param_groups[index]["d"]
                * lr_scheduler.optimizers[-1].param_groups[index]["lr"]
            )
        metrics.append(LearningRateMetric(label=labels[index], value=lr, derived_value=derived_value))

    return tuple(metrics)


def _build_sampler_step_metrics(timestep_runtime, timesteps: torch.Tensor | None) -> SamplerStepMetrics | None:
    sampler = getattr(timestep_runtime, "sampler", None)
    if sampler is None or timesteps is None:
        return None

    ema_loss_mean = None
    ema_loss_std = None
    ema_loss_bins: tuple[float, ...] = ()
    if hasattr(sampler, "bin_loss_ema"):
        ema_loss_mean = sampler.bin_loss_ema.mean().item()
        ema_loss_std = sampler.bin_loss_ema.std().item()
        ema_loss_bins = tuple(loss_val.item() for loss_val in sampler.bin_loss_ema)

    entropy_ratio = sampler.last_entropy_ratio if hasattr(sampler, "last_entropy_ratio") else None

    timestep_histogram: tuple[float, ...] = ()
    if hasattr(sampler, "num_bins") and hasattr(sampler, "T"):
        hist = torch.histogram(
            timesteps.float().cpu(),
            bins=sampler.num_bins,
            range=(0, sampler.T),
        )
        timestep_histogram = tuple(count.item() for count in hist.hist)

    return SamplerStepMetrics(
        ema_loss_mean=ema_loss_mean,
        ema_loss_std=ema_loss_std,
        ema_loss_bins=ema_loss_bins,
        entropy_ratio=entropy_ratio,
        timestep_histogram=timestep_histogram,
    )


def _build_schedule_free_derived_lr(cfg, optimizer) -> float | None:
    if _optimizer_type_name(cfg).endswith("prodigyplusschedulefree") and optimizer is not None:
        return optimizer.param_groups[0]["d"] * optimizer.param_groups[0]["lr"]
    return None


def build_step_metrics_event(
    cfg,
    current_loss,
    avr_loss,
    lr_scheduler,
    lr_descriptions: list[str] | None = None,
    optimization_plan=None,
    timestep_runtime=None,
    optimizer=None,
    keys_scaled=None,
    mean_norm=None,
    maximum_norm=None,
    mean_grad_norm=None,
    mean_combined_norm=None,
    modifier_lrs: dict[str, float] | None = None,
    current_loss_scaled=None,
    average_loss_scaled=None,
    current_val_loss=None,
    average_val_loss=None,
    timesteps: torch.Tensor | None = None,
):
    """Build the typed runtime event for one step-metric emission."""
    lr_names = _resolve_lr_descriptions(lr_descriptions, optimization_plan)

    return StepMetricsEvent(
        current_loss=current_loss,
        average_loss=avr_loss,
        learning_rates=_build_learning_rate_metrics(cfg, lr_scheduler, lr_names),
        modifier_lrs=tuple((modifier_name, modifier_lr) for modifier_name, modifier_lr in (modifier_lrs or {}).items()),
        current_loss_scaled=current_loss_scaled,
        average_loss_scaled=average_loss_scaled,
        keys_scaled=keys_scaled,
        maximum_norm=maximum_norm,
        mean_norm=mean_norm,
        mean_grad_norm=mean_grad_norm,
        mean_combined_norm=mean_combined_norm,
        current_val_loss=current_val_loss,
        average_val_loss=average_val_loss,
        sampler=_build_sampler_step_metrics(timestep_runtime, timesteps),
        schedule_free_derived_lr=_build_schedule_free_derived_lr(cfg, optimizer),
    )


def render_step_metrics_event(event: StepMetricsEvent) -> dict[str, float]:
    """Render a typed step event into the existing flat tracker key namespace."""
    logs = {"loss/current": event.current_loss, "loss/average": event.average_loss}

    if event.current_loss_scaled is not None:
        logs["loss/current_scaled"] = event.current_loss_scaled
        logs["loss/average_scaled"] = event.average_loss_scaled

    if event.keys_scaled is not None:
        logs["max_norm/keys_scaled"] = event.keys_scaled
        logs["max_norm/max_key_norm"] = event.maximum_norm
    if event.mean_norm is not None:
        logs["norm/avg_key_norm"] = event.mean_norm
    if event.mean_grad_norm is not None:
        logs["norm/avg_grad_norm"] = event.mean_grad_norm
    if event.mean_combined_norm is not None:
        logs["norm/avg_combined_norm"] = event.mean_combined_norm

    if event.current_val_loss is not None:
        logs["loss/current_val_loss"] = event.current_val_loss
        logs["loss/average_val_loss"] = event.average_val_loss

    for lr_metric in event.learning_rates:
        logs[f"lr/{lr_metric.label}"] = lr_metric.value
        if lr_metric.derived_value is not None:
            logs[f"lr/d*lr/{lr_metric.label}"] = lr_metric.derived_value

    if event.schedule_free_derived_lr is not None:
        logs["lr/d*lr"] = event.schedule_free_derived_lr

    for modifier_name, modifier_lr in event.modifier_lrs:
        logs[f"lr/{modifier_name}"] = modifier_lr

    if event.sampler is not None:
        if event.sampler.ema_loss_mean is not None:
            logs["sampler/ema_loss_mean"] = event.sampler.ema_loss_mean
        if event.sampler.ema_loss_std is not None:
            logs["sampler/ema_loss_std"] = event.sampler.ema_loss_std
        for index, loss_value in enumerate(event.sampler.ema_loss_bins):
            logs[f"sampler_ema_loss_bins/bin_{index}"] = loss_value

        if event.sampler.entropy_ratio is not None:
            logs["sampler/entropy_ratio"] = event.sampler.entropy_ratio

        for index, count in enumerate(event.sampler.timestep_histogram):
            logs[f"sampler_timestep_hist/bin_{index}"] = count

    return logs


def generate_step_logs(
    cfg,
    current_loss,
    avr_loss,
    lr_scheduler,
    lr_descriptions: list[str] | None = None,
    optimization_plan=None,
    timestep_runtime=None,
    optimizer=None,
    keys_scaled=None,
    mean_norm=None,
    maximum_norm=None,
    mean_grad_norm=None,
    mean_combined_norm=None,
    modifier_lrs: dict[str, float] | None = None,
    current_loss_scaled=None,
    average_loss_scaled=None,
    current_val_loss=None,
    average_val_loss=None,
    timesteps: torch.Tensor | None = None,
):
    """Generate flat tracker logs for training progress tracking."""
    event = build_step_metrics_event(
        cfg,
        current_loss,
        avr_loss,
        lr_scheduler,
        lr_descriptions=lr_descriptions,
        optimization_plan=optimization_plan,
        timestep_runtime=timestep_runtime,
        optimizer=optimizer,
        keys_scaled=keys_scaled,
        mean_norm=mean_norm,
        maximum_norm=maximum_norm,
        mean_grad_norm=mean_grad_norm,
        mean_combined_norm=mean_combined_norm,
        modifier_lrs=modifier_lrs,
        current_loss_scaled=current_loss_scaled,
        average_loss_scaled=average_loss_scaled,
        current_val_loss=current_val_loss,
        average_val_loss=average_val_loss,
        timesteps=timesteps,
    )
    return render_step_metrics_event(event)


def step_logging(accelerator: Accelerator, logs: dict, global_step: int, epoch: int):
    """Log metrics at each step."""
    log_metrics_to_trackers(accelerator, logs, global_step, global_step, epoch)


def epoch_logging(accelerator: Accelerator, logs: dict, global_step: int, epoch: int):
    """Log metrics at epoch end."""
    log_metrics_to_trackers(accelerator, logs, epoch, global_step, epoch)


def log_metrics_to_trackers(accelerator: Accelerator, logs: dict, step_value: int, global_step: int, epoch: int):
    """
    Log metrics to trackers.
    step_value is for tensorboard, other values are for wandb.
    """
    tensorboard_tracker = None
    wandb_tracker = None
    other_trackers = []
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            tensorboard_tracker = accelerator.get_tracker("tensorboard")
        elif tracker.name == "wandb":
            wandb_tracker = accelerator.get_tracker("wandb")
        else:
            other_trackers.append(accelerator.get_tracker(tracker.name))

    if tensorboard_tracker is not None:
        tensorboard_tracker.log(logs, step=step_value)

    if wandb_tracker is not None:
        wandb_logs = dict(logs)
        wandb_logs["global_step"] = global_step
        wandb_logs["epoch"] = epoch
        wandb_tracker.log(wandb_logs)  # Uses payload fields for W&B step info.

    for tracker in other_trackers:
        tracker.log(logs, step=step_value)


accelerator_logging = log_metrics_to_trackers


def _flatten_for_hparams(obj, prefix=""):
    items = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                items.update(_flatten_for_hparams(value, new_key))
            elif isinstance(value, (list, tuple)):
                items[new_key] = str(value)
            elif value is None:
                items[new_key] = "None"
            elif isinstance(value, (int, float, str, bool)):
                items[new_key] = value
            else:
                items[new_key] = str(value)
    return items


def build_tracker_config(logging_config: LoggingConfig) -> dict[str, object]:
    """Build the sanitized flattened config payload used by tracker-style sinks."""
    if hasattr(logging_config, "__dataclass_fields__"):
        config_to_log = asdict(logging_config)
    else:
        config_to_log = OmegaConf.to_container(logging_config, resolve=True)

    sensitive_keys = ["wandb_api_key", "huggingface_token"]
    if isinstance(config_to_log, dict):
        for key in sensitive_keys:
            if key in config_to_log:
                config_to_log[key] = "*****"

    return _flatten_for_hparams(config_to_log)


def resolve_tracker_name(logging_config: LoggingConfig, default_tracker_name: str) -> str:
    """Return the project/run namespace name used for tracker initialization."""
    return logging_config.log_tracker_name if logging_config.log_tracker_name else default_tracker_name


def init_trackers(accelerator: Accelerator, logging_config: LoggingConfig, default_tracker_name: str):
    """
    Initialize experiment trackers with tracker specific behaviors.

    Note: This function only executes on the main process.

    Args:
        accelerator: Accelerator instance.
        logging_config: LoggingConfig with tracker settings.
        default_tracker_name: Default name for the tracker.
    """
    if accelerator.is_main_process:
        init_kwargs = {}
        if logging_config.wandb_run_name:
            init_kwargs.setdefault("wandb", {})["name"] = logging_config.wandb_run_name
        if logging_config.log_tracker_config:
            for key, val in logging_config.log_tracker_config.items():
                if isinstance(val, dict) and isinstance(init_kwargs.get(key), dict):
                    init_kwargs[key].update(val)
                else:
                    init_kwargs[key] = val

        tracker_name = resolve_tracker_name(logging_config, default_tracker_name)
        config_to_log = build_tracker_config(logging_config)

        accelerator.init_trackers(
            tracker_name,
            config=config_to_log,
            init_kwargs=init_kwargs,
        )


def append_lr_to_logs_with_names(logs: dict, lr_scheduler, optimizer_type: str, names: list[str]):
    """
    Append learning rate information to the logs with specific parameter group names.

    Args:
        logs: The dictionary of logs to update.
        lr_scheduler: The learning rate scheduler.
        optimizer_type: The type name of the optimizer.
        names: A list of names corresponding to the parameter groups.
    """
    lrs = lr_scheduler.get_last_lr()

    if len(names) < len(lrs):
        raise ValueError(
            f"names list has {len(names)} elements but lr_scheduler has {len(lrs)} learning rates. "
            "Ensure names list matches the number of parameter groups."
        )

    for lr_index in range(len(lrs)):
        name = names[lr_index]
        logs["lr/" + name] = float(lrs[lr_index])

        if optimizer_type.lower().startswith("DAdapt".lower()) or optimizer_type.lower() == "Prodigy".lower():
            logs["lr/d*lr/" + name] = (
                lr_scheduler.optimizers[-1].param_groups[lr_index]["d"] * lr_scheduler.optimizers[-1].param_groups[lr_index]["lr"]
            )
