"""
Training visualization utilities.

Functions for plotting timesteps distributions and managing the live plotter.
"""

import json
import os
import logging
import subprocess
import sys

import numpy as np
from library.timesteps.samplers.log_snr_sampler import LogSNRUniformSampler
from library.timesteps.samplers.tempered_adaptive_sampler import TemperedAdaptiveSampler

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

logger = logging.getLogger(__name__)


def save_timestep_distribution_plot(
    cfg, global_step: int, timestep_counts: np.ndarray, settings_dict: dict = None
):
    """
    Save the timesteps distribution plot to disk.

    Args:
        cfg: The training configuration object.
        global_step: The current global step of training.
        timestep_counts: An array containing the counts of each timesteps.
        settings_dict: A dictionary of settings to display on the plot.
    """
    if plt is None:
        logger.warning(
            "Matplotlib is not installed. Cannot save timesteps distribution plot."
        )
        return

    output_dir = os.path.join(cfg.output.saving.output_dir, "timestep_plots")
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

    plt.tight_layout(rect=[0, 0.1, 1, 1])  # TODO: Expected type 'tuple[float, float, float, float] | None', got 'list[int | float]' instead
    filename = os.path.join(output_dir, f"step_{global_step:06d}.png")
    plt.savefig(filename)
    plt.close()


def close_live_plotter(live_plotter_process: subprocess.Popen):
    """
    Clean up the live plotter subprocess.

    Args:
        live_plotter_process: The subprocess object for the live plotter server.
    """
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


def get_plotter_settings(cfg, la_sampler) -> dict:
    """
    Gather plotter settings based on the sampler type and config.

    Args:
        cfg: The training configuration object.
        la_sampler: The loss-aware sampler object, or None.

    Returns:
        A dictionary containing the plotter settings to be displayed.
    """
    # Determine the actual sampler being used
    sampler_type = cfg.timestep.timestep_sampling
    if la_sampler is not None:
        if isinstance(la_sampler, LogSNRUniformSampler):
            sampler_type = "log_snr_uniform"
        elif isinstance(la_sampler, TemperedAdaptiveSampler):
            sampler_type = "tempered_adaptive"

    plotter_settings = {
        "Timestep Sampler": sampler_type,
        "Dynamic Schedule": "Enabled" if cfg.timestep.dynamic_timestep_schedule else "Disabled",
        "Min Timestep": cfg.timestep.min_timestep,
        "Max Timestep": cfg.timestep.max_timestep,
    }

    # Add sampler-specific settings
    if sampler_type == "mix_adaptive":
        mc = cfg.timestep.mix_adaptive
        plotter_settings.update({
            "Anneal": mc.anneal,
            "Start/End P": f"{mc.start_p} -> {mc.end_p}",
            "Fixed P": mc.fixed_p,
            "Num Bins": mc.bins,
            "EMA Beta": mc.ema_beta,
            "Small T Frac/Cap": f"{mc.small_t_frac} / {mc.small_t_cap}",
        })
    elif sampler_type == "tempered_adaptive":
        tc = cfg.timestep.tempered_adaptive
        plotter_settings.update({
            "Num Bins": tc.bins,
            "EMA Beta": tc.ema_beta,
            "Temperature": tc.temperature,
            "Prior Weight": tc.prior_weight,
            "Min Prob": tc.min_prob,
            "Warmup Steps": tc.warmup_steps,
            "Prior Bias": tc.prior_bias,
            "Entropy Floor": tc.entropy_floor,
        })
    elif sampler_type == "gaussian_mid_snr":
        gc = cfg.timestep.gaussian_mid_snr
        plotter_settings.update({
            "Num Bins": gc.bins,
            "EMA Beta": gc.ema_beta,
            "Temperature": gc.temperature,
            "Min Prob": gc.min_prob,
            "Entropy Floor": gc.entropy_floor,
            "Uniform Mix When Low Entropy": gc.uniform_mix_when_low_entropy,
            "Prior_Mu": gc.prior_mu,
            "Prior Sigma": gc.prior_sigma,
            "Prior Weight": gc.prior_weight,
            "Warmup Steps": gc.warmup_steps,
        })
    elif sampler_type == "snr_windowed":
        sc = cfg.timestep.snr_windowed
        plotter_settings.update({
            "Num Bins": sc.bins,
            "EMA Beta": sc.ema_beta,
            "Temperature": sc.temperature,
            "Min Prob": sc.min_prob,
            "Entropy Floor": sc.entropy_floor,
            "Uniform Mix": sc.uniform_mix_when_low_entropy,
            "Center Mu": sc.center_mu,
            "Half Width": sc.half_width,
            "Widen To": sc.widen_to,
            "Total Widen Steps": sc.max_train_steps,
            "Cap Max T": sc.cap_max_t,
        })
    elif sampler_type not in ["uniform", "log_snr_uniform"]:
        plotter_settings.update({
            "Shift": cfg.timestep.discrete_flow_shift,
            "Sigmoid Scale": cfg.timestep.sigmoid_scale,
        })

    return plotter_settings


def setup_live_plotter(cfg, noise_scheduler, la_sampler, strategy):
    """
    Setup the live plotter subprocess and initialize static plot timesteps tracking.

    Args:
        cfg: The training configuration object.
        noise_scheduler: The noise scheduler from Diffusers.
        la_sampler: The loss-aware timesteps sampler (or None).
        strategy: The training strategy object, used to store the plotter process handle.

    Returns:
        A tuple containing:
            - timestep_counts: An array for tracking timesteps counts (or None).
            - plotter_settings: A dictionary of settings for the plotter (or None).
    """
    timestep_counts = None
    plotter_settings = None

    # Get plotter settings
    plotter_settings = get_plotter_settings(cfg, la_sampler)
    
    # Setup for the live interactive plotter
    if cfg.output.logging.live_plot_port is not None:
        plotter_script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "live_plotter", "server.py"
        )

        if not os.path.exists(plotter_script_path):
            logger.error(f"server.py not found at {plotter_script_path}. Live plotter disabled.")
        else:
            if strategy.live_plotter_process is None or strategy.live_plotter_process.poll() is not None:
                logger.info(f"Launching live plotter server on port {cfg.output.logging.live_plot_port}")
                strategy.live_plotter_process = subprocess.Popen(
                    [sys.executable, plotter_script_path, "--port", str(cfg.output.logging.live_plot_port)],
                    stdin=subprocess.PIPE,
                )

            # Send the initial "handshake" data
            reset_str = "RESET::\n"
            alphas_cumprod_np = noise_scheduler.alphas_cumprod.cpu().numpy()
            schedule_str = f"SCHEDULE::{','.join(map(str, alphas_cumprod_np))}\n"
            settings_str = f"SETTINGS::{json.dumps(plotter_settings)}\n"

            try:
                strategy.live_plotter_process.stdin.write(reset_str.encode('utf-8'))
                strategy.live_plotter_process.stdin.write(schedule_str.encode('utf-8'))
                strategy.live_plotter_process.stdin.write(settings_str.encode('utf-8'))
                strategy.live_plotter_process.stdin.flush()
            except (BrokenPipeError, OSError):
                logger.error("Failed to send data to live plotter. It may have crashed.")
                strategy.live_plotter_process = None

    # Setup for saving static plot images
    if cfg.output.logging.log_timestep_distribution_every_n_steps is not None:
        timestep_counts = np.zeros(noise_scheduler.config.num_train_timesteps, dtype=np.int64)

    return timestep_counts, plotter_settings
