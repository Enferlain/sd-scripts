from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol, runtime_checkable

import torch
from accelerate import Accelerator
from omegaconf import OmegaConf

from library.config.dataclasses.output import LoggingConfig
from library.logging.console import MainProcessConsole
from library.logging.summaries import TrainingStartupSummary


@dataclass(frozen=True, slots=True)
class LoggedArtifact:
    """A file produced by the run and registered with the logging layer."""

    path: str
    kind: str
    metadata: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class MetricsSink(Protocol):
    def start_run(self, run_name: str, config: dict[str, object]) -> None: ...

    def log_metrics(self, metrics: dict[str, float], *, step: int, epoch: int | None = None) -> None: ...

    def finish_run(self) -> None: ...


@runtime_checkable
class TrainingObserver(Protocol):
    def start_run(self, run_name: str, config: dict[str, object]) -> None: ...

    def log_metrics(self, step: int, metrics: dict[str, float], *, epoch: int | None = None) -> None: ...

    def log_console(self, message: str, *, level: str = "info", tag: str | None = None) -> None: ...

    def log_startup_summary(self, summary: TrainingStartupSummary) -> None: ...

    def log_artifact(self, path: str, *, kind: str, metadata: dict[str, object] | None = None) -> None: ...

    def finish_run(self) -> None: ...


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
    artifacts: list[LoggedArtifact] = field(default_factory=list)
    run_name: str | None = None
    run_config: dict[str, object] = field(default_factory=dict)
    _run_active: bool = field(default=False, init=False, repr=False)

    def start_run(self, run_name: str, config: dict[str, object]) -> None:
        self.run_name = run_name
        self.run_config = dict(config)
        self._run_active = True
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

    def log_artifact(self, path: str, *, kind: str, metadata: dict[str, object] | None = None) -> None:
        self.artifacts.append(LoggedArtifact(path=path, kind=kind, metadata=dict(metadata or {})))

    def finish_run(self) -> None:
        """Finish the observer-local run lifecycle and notify the metrics sink if present."""
        if not self._run_active:
            return
        self._run_active = False
        if self.metrics_sink is not None:
            self.metrics_sink.finish_run()


def _resolve_lr_descriptions(lr_descriptions: list[str] | None, optimization_plan=None) -> list[str]:
    """Resolve trainer-facing LR labels from plan metadata when available."""
    if optimization_plan is None:
        return list(lr_descriptions or [])
    return list(getattr(optimization_plan, "lr_descriptions", lr_descriptions or []))


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
    """Generate step logs for training progress tracking."""
    logs = {"loss/current": current_loss, "loss/average": avr_loss}

    if current_loss_scaled is not None:
        logs["loss/current_scaled"] = current_loss_scaled
        logs["loss/average_scaled"] = average_loss_scaled

    if keys_scaled is not None:
        logs["max_norm/keys_scaled"] = keys_scaled
        logs["max_norm/max_key_norm"] = maximum_norm
    if mean_norm is not None:
        logs["norm/avg_key_norm"] = mean_norm
    if mean_grad_norm is not None:
        logs["norm/avg_grad_norm"] = mean_grad_norm
    if mean_combined_norm is not None:
        logs["norm/avg_combined_norm"] = mean_combined_norm

    if current_val_loss is not None:
        logs["loss/current_val_loss"] = current_val_loss
        logs["loss/average_val_loss"] = average_val_loss

    lrs = lr_scheduler.get_last_lr()
    lr_names = _resolve_lr_descriptions(lr_descriptions, optimization_plan)

    for i, lr in enumerate(lrs):
        lr_desc = lr_names[i]

        logs[f"lr/{lr_desc}"] = lr

        if cfg.optimizer.optimizer_type.lower().startswith("dadapt") or cfg.optimizer.optimizer_type.lower() == "prodigy":
            logs[f"lr/d*lr/{lr_desc}"] = (
                lr_scheduler.optimizers[-1].param_groups[i]["d"] * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
            )
        if cfg.optimizer.optimizer_type.lower().endswith("prodigyplusschedulefree") and optimizer is not None:
            logs["lr/d*lr"] = optimizer.param_groups[0]["d"] * optimizer.param_groups[0]["lr"]

    if modifier_lrs is not None:
        for modifier_name, modifier_lr in modifier_lrs.items():
            logs[f"lr/{modifier_name}"] = modifier_lr

    sampler = getattr(timestep_runtime, "sampler", None)
    if sampler is not None and timesteps is not None:
        if hasattr(sampler, "bin_loss_ema"):
            logs["sampler/ema_loss_mean"] = sampler.bin_loss_ema.mean().item()
            logs["sampler/ema_loss_std"] = sampler.bin_loss_ema.std().item()

            for i, loss_val in enumerate(sampler.bin_loss_ema):
                logs[f"sampler_ema_loss_bins/bin_{i}"] = loss_val.item()

        if hasattr(sampler, "last_entropy_ratio"):
            logs["sampler/entropy_ratio"] = sampler.last_entropy_ratio

        if hasattr(sampler, "num_bins") and hasattr(sampler, "T"):
            hist = torch.histogram(
                timesteps.float().cpu(),
                bins=sampler.num_bins,
                range=(0, sampler.T),
            )
            for i, count in enumerate(hist.hist):
                logs[f"sampler_timestep_hist/bin_{i}"] = count.item()  # hist is a Tensor, .item() is valid

    return logs


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
        logs["global_step"] = global_step
        logs["epoch"] = epoch
        wandb_tracker.log(logs)  # Uses logs dict for step info (global_step/epoch added above)

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
