from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

import torch

from library.config.dataclasses.timestep import TimestepConfig
from library.timesteps.continuous_sampling import apply_training_shift, sample_continuous_timesteps
from library.timesteps.samplers.adaptive_log_snr_sampler import AdaptiveLogSNRSampler
from library.timesteps.samplers.log_snr_sampler import LogSNRUniformSampler


def _runtime_print(accelerator: Any | None, message: str) -> None:
    """Emit a runtime status message when an accelerator logger is available."""
    if accelerator is not None and hasattr(accelerator, "print"):
        accelerator.print(message)


def resolve_timestep_sampling_mode(requested_mode: str | None) -> tuple[str, str]:
    """Resolve the requested timestep mode into the effective runtime mode."""
    normalized_mode = requested_mode or "uniform"
    if normalized_mode in {"uniform", "log_snr_uniform", "adaptive_log_snr", "logit_normal", "cosine_shaped"}:
        return normalized_mode, normalized_mode

    raise ValueError(
        f"Unsupported timestep_sampling={normalized_mode!r}. "
        "Supported modes are: uniform, log_snr_uniform, adaptive_log_snr, logit_normal, cosine_shaped."
    )


def _parse_dynamic_schedule(raw_schedule: str | None) -> list[tuple[int, int, int]]:
    """Parse a dynamic timestep schedule string into sorted step/min/max tuples."""
    if not raw_schedule:
        return []

    parsed_schedule = ast.literal_eval(raw_schedule)
    if not isinstance(parsed_schedule, list):
        raise ValueError("dynamic_timestep_schedule must evaluate to a list of (step, min_timestep, max_timestep) tuples.")

    normalized_schedule: list[tuple[int, int, int]] = []
    for item in parsed_schedule:
        if not isinstance(item, list | tuple) or len(item) != 3:
            raise ValueError("dynamic_timestep_schedule entries must be 3-item (step, min_timestep, max_timestep) tuples.")
        step, min_timestep, max_timestep = item
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in (step, min_timestep, max_timestep)):
            raise ValueError("dynamic_timestep_schedule entries must contain integers.")
        normalized_schedule.append((step, min_timestep, max_timestep))

    normalized_schedule.sort(key=lambda item: item[0])
    return normalized_schedule


def _build_adaptive_sampler(
    timestep_config: TimestepConfig,
    noise_scheduler: Any,
    accelerator: Any | None,
) -> tuple[Any | None, str, str]:
    """Create the adaptive sampler and resolve requested vs effective sampling mode."""
    requested_mode = timestep_config.timestep_sampling or "uniform"

    if requested_mode == "log_snr_uniform":
        _runtime_print(accelerator, "Initializing LogSNRUniformSampler.")
        return (
            LogSNRUniformSampler(noise_scheduler, noise_scheduler.config.num_train_timesteps),
            requested_mode,
            requested_mode,
        )

    if requested_mode == "adaptive_log_snr":
        _runtime_print(accelerator, "Initializing AdaptiveLogSNRSampler.")
        adaptive_cfg = timestep_config.adaptive_log_snr
        return (
            AdaptiveLogSNRSampler(
                noise_scheduler,
                num_bins=adaptive_cfg.bins,
                ema_beta=adaptive_cfg.ema_beta,
                temperature=adaptive_cfg.temperature,
                prior_weight=adaptive_cfg.prior_weight,
                min_prob=adaptive_cfg.min_prob,
                warmup_steps=adaptive_cfg.warmup_steps,
                entropy_floor_ratio=adaptive_cfg.entropy_floor,
                uniform_mix_when_low_entropy=adaptive_cfg.uniform_mix_when_low_entropy,
            ),
            requested_mode,
            requested_mode,
        )

    if requested_mode in {"uniform", "logit_normal", "cosine_shaped"}:
        return None, requested_mode, requested_mode

    raise ValueError(
        f"Unsupported timestep_sampling={requested_mode!r}. "
        "Supported modes are: uniform, log_snr_uniform, adaptive_log_snr, logit_normal, cosine_shaped."
    )


def _resolve_sampler_runtime(
    timestep_config: TimestepConfig,
    noise_scheduler: Any,
    accelerator: Any | None,
    *,
    sampler_override: Any | None,
    allow_sampler_construction: bool,
) -> tuple[Any | None, str, str]:
    """Resolve sampler instance plus requested/effective modes for a timestep runtime."""
    if sampler_override is not None:
        requested_mode, effective_mode = resolve_timestep_sampling_mode(timestep_config.timestep_sampling)
        return sampler_override, requested_mode, effective_mode

    if allow_sampler_construction:
        return _build_adaptive_sampler(timestep_config, noise_scheduler, accelerator)

    requested_mode, effective_mode = resolve_timestep_sampling_mode(timestep_config.timestep_sampling)
    return None, requested_mode, effective_mode


@dataclass
class TimestepRuntime:
    """Trainer-owned runtime state for timestep scheduling and adaptive sampling."""

    requested_mode: str
    effective_mode: str
    current_min_timestep: int
    current_max_timestep: int
    dynamic_schedule: list[tuple[int, int, int]]
    sampler: Any | None = None

    @property
    def is_adaptive(self) -> bool:
        """Return whether timestep selection uses a stateful adaptive sampler."""
        return self.sampler is not None and hasattr(self.sampler, "update")

    def advance_to_step(self, global_step: int) -> tuple[int, int] | None:
        """Advance the active timestep range when a dynamic schedule threshold is reached."""
        updated_range: tuple[int, int] | None = None
        while self.dynamic_schedule and global_step >= self.dynamic_schedule[0][0]:
            _, new_min, new_max = self.dynamic_schedule.pop(0)
            self.current_min_timestep = new_min
            self.current_max_timestep = new_max
            updated_range = (new_min, new_max)
        return updated_range

    def observe(self, timesteps: torch.Tensor, per_sample_loss: torch.Tensor) -> None:
        """Update adaptive timestep state from the losses observed for a batch."""
        if self.sampler is not None and hasattr(self.sampler, "update"):
            self.sampler.update(timesteps.detach(), per_sample_loss.detach())

    def sample_timesteps(
        self,
        *,
        timestep_config: TimestepConfig,
        training_config: Any,
        noise_scheduler: Any,
        batch_size: int,
        device: torch.device,
        global_step: int = 0,
        fixed_timesteps: torch.Tensor | None = None,
        is_train: bool = True,
    ) -> torch.Tensor:
        """Sample timesteps using the active runtime mode and range state."""
        if fixed_timesteps is not None:
            return fixed_timesteps

        min_timestep = self.current_min_timestep
        max_timestep = self.current_max_timestep

        if is_train and hasattr(noise_scheduler, "edm2_laplace_weights"):
            return torch.multinomial(noise_scheduler.edm2_laplace_weights, num_samples=batch_size, replacement=True).to(
                dtype=torch.long,
                device=device,
            )

        if is_train and hasattr(noise_scheduler, "laplace_weights"):
            return torch.multinomial(noise_scheduler.laplace_weights, num_samples=batch_size, replacement=True).to(
                dtype=torch.long,
                device=device,
            )

        if is_train and self.sampler is not None:
            if self.sampler is None:
                raise ValueError(
                    f"effective timestep mode is '{self.effective_mode}' but no sampler runtime is configured."
                )

            t_local = self.sampler.sample(
                batch_size,
                device,
                global_step,
                training_config.max_train_steps,
            )
            return t_local.clamp(min_timestep, max_timestep - 1).to(dtype=torch.long, device=device)

        if is_train and self.effective_mode in {"logit_normal", "cosine_shaped"}:
            u = sample_continuous_timesteps(
                timestep_sampling=self.effective_mode,
                batch_size=batch_size,
                logit_mean=float(timestep_config.logit_mean),
                logit_std=float(timestep_config.logit_std),
                cosine_shape_scale=float(timestep_config.cosine_shape_scale),
            )
            u = apply_training_shift(u, float(timestep_config.training_shift))
            return min_timestep + (u * (max_timestep - min_timestep)).to(
                dtype=torch.long, device=device
            )

        if self.effective_mode != "uniform":
            raise ValueError(f"Unsupported timestep runtime mode: {self.effective_mode}")

        if min_timestep < max_timestep:
            timesteps = torch.randint(min_timestep, max_timestep, (batch_size,), device="cpu")
        else:
            timesteps = torch.full((batch_size,), max_timestep, device="cpu")
        return timesteps.long().to(device)


def build_timestep_runtime(
    timestep_config: TimestepConfig,
    noise_scheduler: Any,
    accelerator: Any | None = None,
    *,
    sampler_override: Any | None = None,
    min_timestep_override: int | None = None,
    max_timestep_override: int | None = None,
    global_step: int | None = None,
    include_dynamic_schedule: bool = True,
    allow_sampler_construction: bool = True,
) -> TimestepRuntime:
    """Build the active timestep runtime without mutating the config object."""
    sampler, requested_mode, effective_mode = _resolve_sampler_runtime(
        timestep_config,
        noise_scheduler,
        accelerator,
        sampler_override=sampler_override,
        allow_sampler_construction=allow_sampler_construction,
    )
    dynamic_schedule = _parse_dynamic_schedule(timestep_config.dynamic_timestep_schedule) if include_dynamic_schedule else []
    if dynamic_schedule:
        _runtime_print(accelerator, f"Using dynamic timesteps schedule: {dynamic_schedule}")

    current_min_timestep = (
        min_timestep_override if min_timestep_override is not None else (0 if timestep_config.min_timestep is None else timestep_config.min_timestep)
    )
    current_max_timestep = (
        max_timestep_override
        if max_timestep_override is not None
        else (noise_scheduler.config.num_train_timesteps if timestep_config.max_timestep is None else timestep_config.max_timestep)
    )

    runtime = TimestepRuntime(
        requested_mode=requested_mode,
        effective_mode=effective_mode,
        current_min_timestep=current_min_timestep,
        current_max_timestep=current_max_timestep,
        dynamic_schedule=dynamic_schedule,
        sampler=sampler,
    )
    if global_step is not None and include_dynamic_schedule:
        runtime.advance_to_step(global_step)
    return runtime
