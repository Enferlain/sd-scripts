"""Compatibility wrappers around the trainer-owned timestep runtime."""

from typing import Any

from library.config.dataclasses.timestep import TimestepConfig
from library.timesteps.runtime import build_timestep_runtime


def parse_dynamic_timestep_schedule(
    timestep_config: TimestepConfig, noise_scheduler: Any, accelerator: Any
) -> tuple[list[tuple[int, int, int]] | None, int, int]:
    """
    Parse dynamic timesteps schedule from config.

    Args:
        timestep_config: Training configuration
        noise_scheduler: Diffusers noise scheduler
        accelerator: HuggingFace Accelerator (for printing)

    Returns:
        Tuple of (schedule_list, current_min_timestep, current_max_timestep)
        schedule_list is None if no dynamic schedule is configured.
    """
    runtime = build_timestep_runtime(
        timestep_config,
        noise_scheduler,
        accelerator,
        allow_sampler_construction=False,
    )
    dynamic_timestep_schedule = runtime.dynamic_schedule or None
    return dynamic_timestep_schedule, runtime.current_min_timestep, runtime.current_max_timestep


def init_timestep_sampler(timestep_config: TimestepConfig, noise_scheduler: Any, accelerator: Any) -> Any:
    """
    Initialize the appropriate timesteps sampler based on config.

    Args:
        timestep_config: Training configuration
        noise_scheduler: Diffusers noise scheduler
        accelerator: HuggingFace Accelerator

    Returns:
        Timestep sampler instance or None for stateless modes.
    """
    runtime = build_timestep_runtime(
        timestep_config,
        noise_scheduler,
        accelerator,
        include_dynamic_schedule=False,
    )
    return runtime.sampler
