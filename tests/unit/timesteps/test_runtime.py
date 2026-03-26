"""Unit tests for trainer-owned timestep runtime behavior."""

from unittest.mock import MagicMock

import pytest
import torch

from library.config.dataclasses.timestep import TimestepConfig
from library.timesteps.runtime import TimestepRuntime, build_timestep_runtime
from library.timesteps.timestep_utils import init_timestep_sampler, parse_dynamic_timestep_schedule


class MockSchedulerConfig:
    """Minimal scheduler config stub."""

    num_train_timesteps = 1000


class MockNoiseScheduler:
    """Minimal noise scheduler stub for timestep runtime tests."""

    def __init__(self):
        self.config = MockSchedulerConfig()
        self.alphas_cumprod = torch.linspace(0.999, 0.001, self.config.num_train_timesteps)


class StrictSampler:
    """Sampler stub that only accepts the shared runtime sampling inputs."""

    def __init__(self):
        self.called_args = None

    def sample(self, bsz: int, device: torch.device, global_step: int, max_steps: int) -> torch.Tensor:
        self.called_args = (bsz, device, global_step, max_steps)
        return torch.full((bsz,), 5, dtype=torch.long, device=device)


@pytest.fixture
def mock_accelerator():
    """Create a lightweight accelerator stub with printable output."""
    accelerator = MagicMock()
    accelerator.print = MagicMock()
    return accelerator


@pytest.fixture
def mock_noise_scheduler():
    """Create a lightweight noise scheduler stub."""
    return MockNoiseScheduler()


@pytest.mark.training
@pytest.mark.unit
class TestBuildTimestepRuntime:
    """Test construction and ownership semantics of the timestep runtime."""

    def test_build_preserves_requested_mode_without_mutating_config(self, mock_noise_scheduler, mock_accelerator):
        cfg = TimestepConfig(timestep_sampling="log_snr_uniform")

        runtime = build_timestep_runtime(cfg, mock_noise_scheduler, mock_accelerator)

        assert cfg.timestep_sampling == "log_snr_uniform"
        assert runtime.requested_mode == "log_snr_uniform"
        assert runtime.effective_mode == "log_snr_uniform"
        assert runtime.sampler is not None

    def test_build_creates_new_adaptive_log_snr_sampler(self, mock_noise_scheduler, mock_accelerator):
        cfg = TimestepConfig(timestep_sampling="adaptive_log_snr")

        runtime = build_timestep_runtime(cfg, mock_noise_scheduler, mock_accelerator)

        assert runtime.requested_mode == "adaptive_log_snr"
        assert runtime.effective_mode == "adaptive_log_snr"
        assert runtime.is_adaptive is True

    def test_build_parses_dynamic_schedule(self, mock_noise_scheduler, mock_accelerator):
        cfg = TimestepConfig(dynamic_timestep_schedule="[(30, 100, 700), (10, 50, 900)]")

        runtime = build_timestep_runtime(cfg, mock_noise_scheduler, mock_accelerator)

        assert runtime.dynamic_schedule == [(10, 50, 900), (30, 100, 700)]
        mock_accelerator.print.assert_called_once()

    def test_compat_init_sampler_does_not_mutate_config(self, mock_noise_scheduler, mock_accelerator):
        cfg = TimestepConfig(timestep_sampling="log_snr_uniform")

        sampler = init_timestep_sampler(cfg, mock_noise_scheduler, mock_accelerator)

        assert sampler is not None
        assert cfg.timestep_sampling == "log_snr_uniform"

    def test_compat_parse_dynamic_schedule_uses_runtime_shape(self, mock_noise_scheduler, mock_accelerator):
        cfg = TimestepConfig(dynamic_timestep_schedule="[(30, 100, 700), (10, 50, 900)]")

        schedule, current_min, current_max = parse_dynamic_timestep_schedule(cfg, mock_noise_scheduler, mock_accelerator)

        assert schedule == [(10, 50, 900), (30, 100, 700)]
        assert current_min == 0
        assert current_max == 1000


@pytest.mark.training
@pytest.mark.unit
class TestTimestepRuntime:
    """Test timestep runtime scheduling and update behavior."""

    def test_advance_to_step_applies_all_due_schedule_entries(self):
        runtime = TimestepRuntime(
            requested_mode="uniform",
            effective_mode="uniform",
            current_min_timestep=0,
            current_max_timestep=1000,
            dynamic_schedule=[(10, 100, 900), (20, 200, 800)],
        )

        updated_range = runtime.advance_to_step(20)

        assert updated_range == (200, 800)
        assert runtime.current_min_timestep == 200
        assert runtime.current_max_timestep == 800
        assert runtime.dynamic_schedule == []

    def test_observe_updates_sampler_with_detached_inputs(self):
        sampler = MagicMock()
        runtime = TimestepRuntime(
            requested_mode="adaptive_log_snr",
            effective_mode="adaptive_log_snr",
            current_min_timestep=0,
            current_max_timestep=1000,
            dynamic_schedule=[],
            sampler=sampler,
        )
        timesteps = torch.tensor([10, 20], requires_grad=False)
        sampling_loss = torch.tensor([0.2, 0.4], requires_grad=True)

        runtime.observe(timesteps, sampling_loss)

        observed_timesteps, observed_loss = sampler.update.call_args.args
        assert not observed_timesteps.requires_grad
        assert not observed_loss.requires_grad

    def test_shift_sampling_uses_runtime_range(self):
        runtime = TimestepRuntime(
            requested_mode="shift",
            effective_mode="shift",
            current_min_timestep=200,
            current_max_timestep=400,
            dynamic_schedule=[],
        )
        cfg = TimestepConfig(timestep_sampling="shift", sigmoid_scale=1.0, discrete_flow_shift=1.0)
        training_cfg = MagicMock()
        training_cfg.max_train_steps = 1000

        timesteps = runtime.sample_timesteps(
            timestep_config=cfg,
            training_config=training_cfg,
            noise_scheduler=MockNoiseScheduler(),
            batch_size=64,
            device=torch.device("cpu"),
            is_train=True,
        )

        assert timesteps.min() >= 200
        assert timesteps.max() < 400

    def test_sampler_runtime_passes_only_shared_sampling_inputs(self):
        sampler = StrictSampler()
        runtime = TimestepRuntime(
            requested_mode="log_snr_uniform",
            effective_mode="log_snr_uniform",
            current_min_timestep=0,
            current_max_timestep=1000,
            dynamic_schedule=[],
            sampler=sampler,
        )
        cfg = TimestepConfig(timestep_sampling="log_snr_uniform")
        training_cfg = MagicMock()
        training_cfg.max_train_steps = 250

        timesteps = runtime.sample_timesteps(
            timestep_config=cfg,
            training_config=training_cfg,
            noise_scheduler=MockNoiseScheduler(),
            batch_size=4,
            device=torch.device("cpu"),
            global_step=12,
            is_train=True,
        )

        assert sampler.called_args == (4, torch.device("cpu"), 12, 250)
        assert torch.equal(timesteps, torch.full((4,), 5, dtype=torch.long))
