"""Unit tests for library.logging.step_logging.

Tests correctness of LR metric emission and tracker initialization,
specifically the bugs fixed in Phase 0:
1. Duplicate LR keys (removed dead fallback code, lr_descriptions now required)
2. wandb_run_name overwritten by log_tracker_config
"""

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Lightweight stubs - avoid importing heavy deps (torch, accelerate)
# ---------------------------------------------------------------------------


@dataclass
class _OptimizerConfig:
    optimizer_type: str = "AdamW"


@dataclass
class _TimestepConfig:
    timestep_sampling: str = "uniform"


@dataclass
class _StubCfg:
    optimizer: _OptimizerConfig = field(default_factory=_OptimizerConfig)
    timestep: _TimestepConfig = field(default_factory=_TimestepConfig)


def _make_lr_scheduler(lrs: list[float]):
    """Create a minimal LR scheduler mock returning *lrs* from get_last_lr."""
    sched = MagicMock()
    sched.get_last_lr.return_value = lrs
    return sched


# ===================================================================
# generate_step_logs – LR key correctness
# ===================================================================


class TestGenerateStepLogsLrKeys:
    """Assert LR keys match the provided lr_descriptions with no duplicates."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from library.logging.step_logging import generate_step_logs

        self.generate_step_logs = generate_step_logs

    def test_single_group(self):
        logs = self.generate_step_logs(
            cfg=_StubCfg(),
            current_loss=0.1,
            avr_loss=0.2,
            lr_scheduler=_make_lr_scheduler([1e-4]),
            lr_descriptions=["denoiser"],
        )
        lr_keys = [k for k in logs if k.startswith("lr/")]
        assert lr_keys == ["lr/denoiser"]

    def test_two_groups_te_and_unet(self):
        logs = self.generate_step_logs(
            cfg=_StubCfg(),
            current_loss=0.1,
            avr_loss=0.2,
            lr_scheduler=_make_lr_scheduler([5e-5, 1e-4]),
            lr_descriptions=["textencoder", "denoiser"],
        )
        lr_keys = [k for k in logs if k.startswith("lr/")]
        assert lr_keys == ["lr/textencoder", "lr/denoiser"]
        assert len(lr_keys) == len(set(lr_keys)), f"Duplicate LR keys: {lr_keys}"

    def test_multiple_text_encoders(self):
        """SDXL-style: TE1, TE2, UNet each with own LR."""
        logs = self.generate_step_logs(
            cfg=_StubCfg(),
            current_loss=0.1,
            avr_loss=0.2,
            lr_scheduler=_make_lr_scheduler([5e-5, 3e-5, 1e-4]),
            lr_descriptions=["text_encoder1", "text_encoder2", "denoiser"],
        )
        lr_keys = [k for k in logs if k.startswith("lr/")]
        assert lr_keys == ["lr/text_encoder1", "lr/text_encoder2", "lr/denoiser"]
        assert len(lr_keys) == len(set(lr_keys)), f"Duplicate LR keys: {lr_keys}"

    def test_custom_descriptions(self):
        logs = self.generate_step_logs(
            cfg=_StubCfg(),
            current_loss=0.1,
            avr_loss=0.2,
            lr_scheduler=_make_lr_scheduler([1e-4, 2e-4]),
            lr_descriptions=["my_unet", "my_te"],
        )
        lr_keys = [k for k in logs if k.startswith("lr/")]
        assert lr_keys == ["lr/my_unet", "lr/my_te"]

    def test_no_duplicate_keys(self):
        """Ensure exactly one key per param group, no extra keys from stale fallback logic."""
        logs = self.generate_step_logs(
            cfg=_StubCfg(),
            current_loss=0.1,
            avr_loss=0.2,
            lr_scheduler=_make_lr_scheduler([3e-4]),
            lr_descriptions=["denoiser"],
        )
        lr_keys = [k for k in logs if k.startswith("lr/")]
        assert len(lr_keys) == 1, f"Expected exactly 1 LR key, got {lr_keys}"


# ===================================================================
# init_trackers – W&B run name retention
# ===================================================================


@dataclass
class _LoggingConfigStub:
    """Minimal LoggingConfig for init_trackers tests."""

    wandb_run_name: str | None = None
    log_tracker_config: dict | None = field(default_factory=dict)
    log_tracker_name: str | None = None
    console_log_level: str | None = None
    console_log_file: str | None = None
    console_log_simple: bool = False
    wandb_api_key: str | None = None


class TestInitTrackers:
    """Assert init_trackers preserves wandb_run_name through deep-merge."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from library.logging.step_logging import init_trackers

        self.init_trackers = init_trackers

    def _make_accelerator(self):
        acc = MagicMock()
        acc.is_main_process = True
        return acc

    def test_only_wandb_run_name(self):
        acc = self._make_accelerator()
        cfg = _LoggingConfigStub(wandb_run_name="my-run")
        self.init_trackers(acc, cfg, "default_project")

        call_kwargs = acc.init_trackers.call_args
        init_kwargs = call_kwargs.kwargs.get("init_kwargs") or call_kwargs[1].get("init_kwargs")
        assert init_kwargs["wandb"]["name"] == "my-run"

    def test_only_log_tracker_config(self):
        acc = self._make_accelerator()
        cfg = _LoggingConfigStub(
            log_tracker_config={"wandb": {"entity": "my-team"}},
        )
        self.init_trackers(acc, cfg, "default_project")

        call_kwargs = acc.init_trackers.call_args
        init_kwargs = call_kwargs.kwargs.get("init_kwargs") or call_kwargs[1].get("init_kwargs")
        assert init_kwargs["wandb"]["entity"] == "my-team"
        # No name key since wandb_run_name is not set
        assert "name" not in init_kwargs["wandb"]

    def test_both_wandb_run_name_and_tracker_config_preserves_name(self):
        """wandb_run_name should survive when log_tracker_config also has wandb keys."""
        acc = self._make_accelerator()
        cfg = _LoggingConfigStub(
            wandb_run_name="my-run",
            log_tracker_config={"wandb": {"entity": "my-team", "tags": ["test"]}},
        )
        self.init_trackers(acc, cfg, "default_project")

        call_kwargs = acc.init_trackers.call_args
        init_kwargs = call_kwargs.kwargs.get("init_kwargs") or call_kwargs[1].get("init_kwargs")
        # Both name and entity should exist
        assert init_kwargs["wandb"]["name"] == "my-run", "wandb_run_name was overwritten!"
        assert init_kwargs["wandb"]["entity"] == "my-team"
        assert init_kwargs["wandb"]["tags"] == ["test"]

    def test_tracker_config_without_wandb_key(self):
        """Non-wandb tracker config should not affect wandb run name."""
        acc = self._make_accelerator()
        cfg = _LoggingConfigStub(
            wandb_run_name="my-run",
            log_tracker_config={"tensorboard": {"log_dir": "/tmp/tb"}},
        )
        self.init_trackers(acc, cfg, "default_project")

        call_kwargs = acc.init_trackers.call_args
        init_kwargs = call_kwargs.kwargs.get("init_kwargs") or call_kwargs[1].get("init_kwargs")
        assert init_kwargs["wandb"]["name"] == "my-run"
        assert init_kwargs["tensorboard"]["log_dir"] == "/tmp/tb"

    def test_custom_tracker_name(self):
        acc = self._make_accelerator()
        cfg = _LoggingConfigStub(log_tracker_name="my-project")
        self.init_trackers(acc, cfg, "default_project")

        call_kwargs = acc.init_trackers.call_args
        # First positional arg is tracker_name
        assert call_kwargs[0][0] == "my-project"

    def test_default_tracker_name_when_none(self):
        acc = self._make_accelerator()
        cfg = _LoggingConfigStub()
        self.init_trackers(acc, cfg, "fallback_name")

        call_kwargs = acc.init_trackers.call_args
        assert call_kwargs[0][0] == "fallback_name"

    def test_not_called_on_non_main_process(self):
        acc = self._make_accelerator()
        acc.is_main_process = False
        cfg = _LoggingConfigStub(wandb_run_name="my-run")
        self.init_trackers(acc, cfg, "default_project")

        acc.init_trackers.assert_not_called()
