# PEFT Training Common Utilities
# Shared utility functions for PEFT training that are model-agnostic

import logging
import os
from typing import Optional

import torch
from accelerate import Accelerator

from library.utils.common_utils import setup_logging

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

setup_logging()
logger = logging.getLogger(__name__)


def generate_step_logs(
    cfg,
    current_loss,
    avr_loss,
    lr_scheduler,
    lr_descriptions,
    la_sampler=None,
    optimizer=None,
    keys_scaled=None,
    mean_norm=None,
    maximum_norm=None,
    mean_grad_norm=None,
    mean_combined_norm=None,
    edm2_lr_scheduler=None,
    current_loss_scaled=None,
    average_loss_scaled=None,
    current_val_loss=None,
    average_val_loss=None,
    timesteps: Optional[torch.Tensor] = None,
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
    for i, lr in enumerate(lrs):
        if lr_descriptions is not None:
            lr_desc = lr_descriptions[i]
        else:
            idx = i - (0 if cfg.network.network_train_unet_only else -1)
            if idx == -1:
                lr_desc = "textencoder"
            else:
                if len(lrs) > 2:
                    lr_desc = f"group{idx}"
                else:
                    lr_desc = "unet"

        logs[f"lr/{lr_desc}"] = lr

        if cfg.optimizer.optimizer_type.lower().startswith("DAdapt".lower()) or cfg.optimizer.optimizer_type.lower() == "Prodigy".lower():
            logs[f"lr/d*lr/{lr_desc}"] = (
                lr_scheduler.optimizers[-1].param_groups[i]["d"] * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
            )
        if cfg.optimizer.optimizer_type.lower().endswith("ProdigyPlusScheduleFree".lower()) and optimizer is not None:
            logs["lr/d*lr"] = optimizer.param_groups[0]["d"] * optimizer.param_groups[0]["lr"]
    else:
        idx = 0
        if not cfg.network.network_train_unet_only:
            logs["lr/textencoder"] = float(lrs[0])
            idx = 1

        for i in range(idx, len(lrs)):
            logs[f"lr/group{i}"] = float(lrs[i])
            if cfg.optimizer.optimizer_type.lower().startswith("DAdapt".lower()) or cfg.optimizer.optimizer_type.lower() == "Prodigy".lower():
                logs[f"lr/d*lr/group{i}"] = (
                    lr_scheduler.optimizers[-1].param_groups[i]["d"] * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
                )
            if cfg.optimizer.optimizer_type.lower().endswith("ProdigyPlusScheduleFree".lower()) and optimizer is not None:
                logs[f"lr/d*lr/group{i}"] = optimizer.param_groups[i]["d"] * optimizer.param_groups[i]["lr"]

    if edm2_lr_scheduler is not None:
        logs["lr/edm2"] = edm2_lr_scheduler.get_last_lr()[0]

    if cfg.timestep.timestep_sampling == "mix_adaptive" and la_sampler is not None and timesteps is not None:
        if hasattr(la_sampler, "last_mix_p"):
            logs["sampler/mix_p"] = la_sampler.last_mix_p
        if hasattr(la_sampler, "last_small_t_frac"):
            logs["sampler/small_t_frac"] = la_sampler.last_small_t_frac

        if hasattr(la_sampler, "bin_loss_ema"):
            logs["sampler/ema_loss_mean"] = la_sampler.bin_loss_ema.mean().item()
            logs["sampler/ema_loss_std"] = la_sampler.bin_loss_ema.std().item()

            for i, loss_val in enumerate(la_sampler.bin_loss_ema):
                logs[f"sampler_ema_loss_bins/bin_{i}"] = loss_val.item()

        if hasattr(la_sampler, "num_bins") and hasattr(la_sampler, "T"):
            hist = torch.histogram(
                timesteps.float().cpu(),
                bins=la_sampler.num_bins,
                range=(0, la_sampler.T),
            )
            for i, count in enumerate(hist.hist):
                logs[f"sampler_timestep_hist/bin_{i}"] = count.item()

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
        wandb_tracker.log(logs)

    for tracker in other_trackers:
        tracker.log(logs, step=step_value)


def save_timestep_distribution_plot(cfg, global_step, timestep_counts, settings_dict=None):
    """Save timestep distribution plot to disk."""
    if plt is None:
        logger.warning("Matplotlib is not installed. Cannot save timestep distribution plot.")
        return

    output_dir = os.path.join(cfg.saving.output_dir, "timestep_plots")
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(15, 7))
    plt.bar(range(len(timestep_counts)), timestep_counts, width=1.0)
    plt.title(f"Timestep Distribution at Step {global_step}")
    plt.xlabel("Timestep")
    plt.ylabel("Accumulated Count")
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)

    if settings_dict:
        settings_text = "\n".join([f"{key}: {value}" for key, value in settings_dict.items() if value is not None])
        plt.figtext(0.01, 0.01, settings_text, wrap=True, horizontalalignment='left', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.1))

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    filename = os.path.join(output_dir, f"step_{global_step:06d}.png")
    plt.savefig(filename)
    plt.close()


def close_live_plotter(live_plotter_process):
    """Clean up live plotter subprocess."""
    import subprocess
    
    if live_plotter_process is not None:
        logger.info("Shutting down live plotter server...")
        try:
            if live_plotter_process.stdin:
                live_plotter_process.stdin.close()
            if live_plotter_process.poll() is None:
                live_plotter_process.terminate()
                live_plotter_process.wait(timeout=5)
            logger.info("Live plotter server shut down.")
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Could not shut down live plotter server cleanly, killing: {e}")
            live_plotter_process.kill()

