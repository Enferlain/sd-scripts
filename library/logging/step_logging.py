import torch
from accelerate import Accelerator
from omegaconf import OmegaConf

from library.config.dataclasses.output import LoggingConfig


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
    accelerator_logging(accelerator, logs, global_step, global_step, epoch)


def epoch_logging(accelerator: Accelerator, logs: dict, global_step: int, epoch: int):
    """Log metrics at epoch end."""
    accelerator_logging(accelerator, logs, epoch, global_step, epoch)


def accelerator_logging(accelerator: Accelerator, logs: dict, step_value: int, global_step: int, epoch: int):
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

        # sanitize config for logging - convert to dict if needed
        if hasattr(logging_config, "__dataclass_fields__"):
            from dataclasses import asdict

            config_to_log = asdict(logging_config)
        else:
            config_to_log = OmegaConf.to_container(logging_config, resolve=True)

        sensitive_keys = ["wandb_api_key", "huggingface_token"]
        if isinstance(config_to_log, dict):
            for key in sensitive_keys:
                if key in config_to_log:
                    config_to_log[key] = "*****"

        tracker_name = logging_config.log_tracker_name if logging_config.log_tracker_name else default_tracker_name

        # Sanitize config values for TensorBoard hparams (only accepts int, float, str, bool, Tensor)
        # Need to flatten nested dicts to dot-notation keys
        def flatten_for_hparams(obj, prefix=""):
            items = {}
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_key = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, dict):
                        items.update(flatten_for_hparams(v, new_key))
                    elif isinstance(v, (list, tuple)):
                        items[new_key] = str(v)
                    elif v is None:
                        items[new_key] = "None"
                    elif isinstance(v, (int, float, str, bool)):
                        items[new_key] = v
                    else:
                        items[new_key] = str(v)
            return items

        config_to_log = flatten_for_hparams(config_to_log)

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
