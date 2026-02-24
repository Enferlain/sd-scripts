"""
Unit tests for ValidationScheduler and validation cadence normalization.

Tests the validation trigger matrix: start/end/step/epoch combinations,
default epoch-end behavior, and cadence normalization via prepare_config.
"""

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

from library.training.phases.validation import (
    ValidationScheduler,
    ValidationStepContext,
)


# =============================================================================
# Helpers
# =============================================================================


@dataclass
class MockValidationConfig:
    """Minimal stand-in for ValidationConfig with same fields."""

    run_at_start: bool = True
    run_at_end: bool = True
    validate_every_n_steps: int | None = None
    validate_every_n_epochs: int | None = None
    max_validation_steps: int | None = None
    validation_split: float = 0.0
    validation_seed: int | None = None
    validation_timesteps: str = field(default="[50, 350, 500, 650, 950]")


def _ctx(
    global_step: int = 50,
    epoch_step: int = 10,
    current_epoch: int = 1,
    is_last_step_in_epoch: bool = False,
    is_training_start: bool = False,
    is_training_end: bool = False,
    has_validation_data: bool = True,
) -> ValidationStepContext:
    """Convenience builder for ValidationStepContext."""
    return ValidationStepContext(
        global_step=global_step,
        epoch_step=epoch_step,
        current_epoch=current_epoch,
        is_last_step_in_epoch=is_last_step_in_epoch,
        is_training_start=is_training_start,
        is_training_end=is_training_end,
        has_validation_data=has_validation_data,
    )


# =============================================================================
# ValidationScheduler Tests
# =============================================================================


@pytest.mark.training
@pytest.mark.unit
class TestValidationScheduler:
    """Test ValidationScheduler.should_run() trigger logic."""

    # --- No validation data ---

    def test_no_validation_data_always_false(self):
        """No validation data means never validate, regardless of config."""
        cfg = MockValidationConfig(run_at_start=True, run_at_end=True, validate_every_n_steps=10)
        sched = ValidationScheduler(cfg)

        assert sched.should_run(_ctx(has_validation_data=False, is_training_start=True)) is False
        assert sched.should_run(_ctx(has_validation_data=False, is_training_end=True)) is False
        assert sched.should_run(_ctx(has_validation_data=False, global_step=10)) is False

    # --- run_at_start ---

    def test_run_at_start_enabled(self):
        """run_at_start triggers at is_training_start=True."""
        cfg = MockValidationConfig(run_at_start=True)
        sched = ValidationScheduler(cfg)

        assert sched.should_run(_ctx(is_training_start=True)) is True

    def test_run_at_start_disabled(self):
        """run_at_start=False skips training start."""
        cfg = MockValidationConfig(run_at_start=False)
        sched = ValidationScheduler(cfg)

        assert sched.should_run(_ctx(is_training_start=True, global_step=0)) is False

    def test_run_at_start_does_not_trigger_mid_training(self):
        """run_at_start only works when is_training_start is True."""
        cfg = MockValidationConfig(run_at_start=True)
        sched = ValidationScheduler(cfg)

        assert sched.should_run(_ctx(is_training_start=False, global_step=0)) is False

    # --- run_at_end ---

    def test_run_at_end_enabled(self):
        """run_at_end triggers at is_training_end=True."""
        cfg = MockValidationConfig(run_at_end=True)
        sched = ValidationScheduler(cfg)

        assert sched.should_run(_ctx(is_training_end=True)) is True

    def test_run_at_end_disabled(self):
        """run_at_end=False skips training end."""
        cfg = MockValidationConfig(run_at_end=False)
        sched = ValidationScheduler(cfg)

        assert sched.should_run(_ctx(is_training_end=True)) is False

    # --- every_n_steps ---

    def test_step_cadence_triggers_on_match(self):
        """Step cadence triggers when global_step is divisible by n."""
        cfg = MockValidationConfig(run_at_start=False, run_at_end=False, validate_every_n_steps=100)
        sched = ValidationScheduler(cfg)

        assert sched.should_run(_ctx(global_step=100)) is True
        assert sched.should_run(_ctx(global_step=200)) is True
        assert sched.should_run(_ctx(global_step=300)) is True

    def test_step_cadence_skips_between_intervals(self):
        """Step cadence returns False between intervals."""
        cfg = MockValidationConfig(run_at_start=False, run_at_end=False, validate_every_n_steps=100)
        sched = ValidationScheduler(cfg)

        assert sched.should_run(_ctx(global_step=50)) is False
        assert sched.should_run(_ctx(global_step=150)) is False
        assert sched.should_run(_ctx(global_step=99)) is False

    def test_step_cadence_excludes_step_zero(self):
        """Step cadence never triggers at step 0 (that's run_at_start's job)."""
        cfg = MockValidationConfig(run_at_start=False, run_at_end=False, validate_every_n_steps=100)
        sched = ValidationScheduler(cfg)

        assert sched.should_run(_ctx(global_step=0)) is False

    # --- every_n_epochs ---

    def test_epoch_cadence_triggers_on_match(self):
        """Epoch cadence triggers at last step of a matching epoch."""
        cfg = MockValidationConfig(run_at_start=False, run_at_end=False, validate_every_n_epochs=2)
        sched = ValidationScheduler(cfg)

        assert sched.should_run(_ctx(current_epoch=2, is_last_step_in_epoch=True)) is True
        assert sched.should_run(_ctx(current_epoch=4, is_last_step_in_epoch=True)) is True

    def test_epoch_cadence_skips_non_matching_epochs(self):
        """Epoch cadence returns False on non-matching epochs."""
        cfg = MockValidationConfig(run_at_start=False, run_at_end=False, validate_every_n_epochs=2)
        sched = ValidationScheduler(cfg)

        assert sched.should_run(_ctx(current_epoch=1, is_last_step_in_epoch=True)) is False
        assert sched.should_run(_ctx(current_epoch=3, is_last_step_in_epoch=True)) is False

    def test_epoch_cadence_requires_last_step_in_epoch(self):
        """Epoch cadence only triggers on last step of the epoch."""
        cfg = MockValidationConfig(run_at_start=False, run_at_end=False, validate_every_n_epochs=1)
        sched = ValidationScheduler(cfg)

        assert sched.should_run(_ctx(current_epoch=1, is_last_step_in_epoch=False)) is False

    # --- Default: epoch-end fallback ---

    def test_default_epoch_end_when_no_cadence_set(self):
        """When neither step nor epoch cadence is set, validate at epoch end."""
        cfg = MockValidationConfig(
            run_at_start=False,
            run_at_end=False,
            validate_every_n_steps=None,
            validate_every_n_epochs=None,
        )
        sched = ValidationScheduler(cfg)

        assert sched.should_run(_ctx(is_last_step_in_epoch=True)) is True

    def test_default_epoch_end_does_not_trigger_mid_epoch(self):
        """Default epoch-end fallback only triggers at last step."""
        cfg = MockValidationConfig(
            run_at_start=False,
            run_at_end=False,
            validate_every_n_steps=None,
            validate_every_n_epochs=None,
        )
        sched = ValidationScheduler(cfg)

        assert sched.should_run(_ctx(is_last_step_in_epoch=False)) is False

    def test_default_epoch_end_not_active_when_step_cadence_set(self):
        """Epoch-end fallback is suppressed when step cadence is set."""
        cfg = MockValidationConfig(
            run_at_start=False,
            run_at_end=False,
            validate_every_n_steps=100,
            validate_every_n_epochs=None,
        )
        sched = ValidationScheduler(cfg)

        assert sched.should_run(_ctx(global_step=50, is_last_step_in_epoch=True)) is False

    def test_default_epoch_end_not_active_when_epoch_cadence_set(self):
        """Epoch-end fallback is suppressed when epoch cadence is set."""
        cfg = MockValidationConfig(
            run_at_start=False,
            run_at_end=False,
            validate_every_n_steps=None,
            validate_every_n_epochs=3,
        )
        sched = ValidationScheduler(cfg)

        assert sched.should_run(_ctx(current_epoch=1, is_last_step_in_epoch=True)) is False

    # --- Both step and epoch cadences set (OR semantics) ---

    def test_both_cadences_step_matches(self):
        """When both cadences set, step match alone triggers."""
        cfg = MockValidationConfig(
            run_at_start=False,
            run_at_end=False,
            validate_every_n_steps=50,
            validate_every_n_epochs=3,
        )
        sched = ValidationScheduler(cfg)

        assert sched.should_run(_ctx(global_step=50, current_epoch=1)) is True

    def test_both_cadences_epoch_matches(self):
        """When both cadences set, epoch match alone triggers."""
        cfg = MockValidationConfig(
            run_at_start=False,
            run_at_end=False,
            validate_every_n_steps=50,
            validate_every_n_epochs=3,
        )
        sched = ValidationScheduler(cfg)

        assert sched.should_run(_ctx(global_step=77, current_epoch=3, is_last_step_in_epoch=True)) is True

    def test_both_cadences_neither_matches(self):
        """When both cadences set, neither matching → False."""
        cfg = MockValidationConfig(
            run_at_start=False,
            run_at_end=False,
            validate_every_n_steps=50,
            validate_every_n_epochs=3,
        )
        sched = ValidationScheduler(cfg)

        assert sched.should_run(_ctx(global_step=77, current_epoch=1, is_last_step_in_epoch=False)) is False


# =============================================================================
# Cadence Normalization Tests (via prepare_config)
# =============================================================================


@pytest.mark.training
@pytest.mark.unit
class TestCadenceNormalization:
    """Test that prepare_config normalizes invalid validation cadence values."""

    @staticmethod
    def _make_cfg_with_validation(steps=None, epochs=None):
        """Build a minimal mock cfg with validation cadence fields."""
        cfg = MagicMock()
        cfg.validation.validate_every_n_steps = steps
        cfg.validation.validate_every_n_epochs = epochs
        # Satisfy other prepare_config paths
        cfg.data.caching.cache_latents_to_disk = False
        cfg.data.caching.cache_latents = False
        cfg.data.caption.caption_extention = None
        cfg.performance = None  # skip TE caching path
        cfg.optimizer.use_8bit_adam = False
        cfg.optimizer.use_lion_optimizer = False
        cfg.output.sampling.sample_every_n_epochs = None
        cfg.output.sampling.sample_every_n_steps = None
        return cfg

    def test_negative_step_cadence_normalized_to_none(self):
        """Negative validate_every_n_steps is normalized to None."""
        from library.config.config_validation import prepare_config

        cfg = self._make_cfg_with_validation(steps=-1)
        prepare_config(cfg)
        assert cfg.validation.validate_every_n_steps is None

    def test_zero_step_cadence_normalized_to_none(self):
        """Zero validate_every_n_steps is normalized to None."""
        from library.config.config_validation import prepare_config

        cfg = self._make_cfg_with_validation(steps=0)
        prepare_config(cfg)
        assert cfg.validation.validate_every_n_steps is None

    def test_negative_epoch_cadence_normalized_to_none(self):
        """Negative validate_every_n_epochs is normalized to None."""
        from library.config.config_validation import prepare_config

        cfg = self._make_cfg_with_validation(epochs=-5)
        prepare_config(cfg)
        assert cfg.validation.validate_every_n_epochs is None

    def test_valid_cadences_unchanged(self):
        """Valid positive cadence values are not modified."""
        from library.config.config_validation import prepare_config

        cfg = self._make_cfg_with_validation(steps=10, epochs=2)
        prepare_config(cfg)
        assert cfg.validation.validate_every_n_steps == 10
        assert cfg.validation.validate_every_n_epochs == 2

    def test_none_cadences_unchanged(self):
        """None cadence values (disabled) are not modified."""
        from library.config.config_validation import prepare_config

        cfg = self._make_cfg_with_validation(steps=None, epochs=None)
        prepare_config(cfg)
        assert cfg.validation.validate_every_n_steps is None
        assert cfg.validation.validate_every_n_epochs is None
