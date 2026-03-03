"""
Unit tests for training trigger policy helpers.
"""

from unittest.mock import MagicMock

import pytest

from library.config.dataclasses.output import SamplingConfig, SavingConfig
from library.training.phases.triggers import (
    EpochEndTriggerContext,
    StepTriggerContext,
    compute_epoch_end_actions,
    compute_step_actions,
)


@pytest.mark.training
@pytest.mark.unit
class TestStepTriggers:
    """Test step-level trigger decision helpers."""

    def test_compute_step_actions_uses_scheduler_sampling_and_step_save(self):
        """Step actions combine validation, sampling, and step-save policy."""
        scheduler = MagicMock()
        scheduler.should_run.return_value = True

        sampling = SamplingConfig(sample_every_n_steps=10)
        saving = SavingConfig(save_every_n_steps=5)
        ctx = StepTriggerContext(
            global_step=10,
            epoch_step=3,
            current_epoch=1,
            num_steps_in_epoch=8,
            max_train_steps=100,
            has_validation_data=True,
        )

        actions = compute_step_actions(
            validation_scheduler=scheduler,
            sampling_config=sampling,
            saving_config=saving,
            ctx=ctx,
        )

        assert actions.should_validate is True
        assert actions.should_sample is True
        assert actions.should_save_step is True
        assert actions.should_enter_eval_mode is True

    def test_compute_step_actions_builds_validation_context_flags(self):
        """Validation context flags reflect end-of-epoch and training-end state."""
        scheduler = MagicMock()
        scheduler.should_run.return_value = False

        sampling = SamplingConfig(sample_every_n_steps=None)
        saving = SavingConfig(save_every_n_steps=None)
        ctx = StepTriggerContext(
            global_step=20,
            epoch_step=4,
            current_epoch=2,
            num_steps_in_epoch=5,
            max_train_steps=20,
            has_validation_data=False,
        )

        compute_step_actions(
            validation_scheduler=scheduler,
            sampling_config=sampling,
            saving_config=saving,
            ctx=ctx,
        )

        called_ctx = scheduler.should_run.call_args.args[0]
        assert called_ctx.is_last_step_in_epoch is True
        assert called_ctx.is_training_end is True
        assert called_ctx.has_validation_data is False


@pytest.mark.training
@pytest.mark.unit
class TestEpochEndTriggers:
    """Test epoch-end trigger decision helpers."""

    def test_epoch_end_save_boundary_and_last_epoch_guard(self):
        """Epoch-end save should respect interval and skip final epoch."""
        sampling = SamplingConfig(sample_every_n_epochs=None)
        saving = SavingConfig(save_every_n_epochs=2)

        save_epoch_2 = compute_epoch_end_actions(
            sampling_config=sampling,
            saving_config=saving,
            ctx=EpochEndTriggerContext(current_epoch=2, num_train_epochs=4, global_step=20),
        )
        assert save_epoch_2.should_save_epoch is True

        skip_last_epoch = compute_epoch_end_actions(
            sampling_config=sampling,
            saving_config=saving,
            ctx=EpochEndTriggerContext(current_epoch=4, num_train_epochs=4, global_step=40),
        )
        assert skip_last_epoch.should_save_epoch is False

    def test_epoch_end_eval_mode_preserves_existing_save_config_behavior(self):
        """Epoch-end eval mode is entered when epoch save config exists."""
        sampling = SamplingConfig(sample_every_n_epochs=None)
        saving = SavingConfig(save_every_n_epochs=3)

        actions = compute_epoch_end_actions(
            sampling_config=sampling,
            saving_config=saving,
            ctx=EpochEndTriggerContext(current_epoch=1, num_train_epochs=5, global_step=10),
            sample_images_check_fn=lambda *_: False,
        )

        assert actions.should_sample is False
        assert actions.should_save_epoch is False
        assert actions.should_enter_eval_mode is True

    def test_epoch_end_no_eval_when_no_sampling_and_no_save_config(self):
        """Epoch-end eval mode is skipped when no sample/save trigger exists."""
        sampling = SamplingConfig(sample_every_n_epochs=None)
        saving = SavingConfig(save_every_n_epochs=None)

        actions = compute_epoch_end_actions(
            sampling_config=sampling,
            saving_config=saving,
            ctx=EpochEndTriggerContext(current_epoch=1, num_train_epochs=5, global_step=10),
            sample_images_check_fn=lambda *_: False,
        )

        assert actions.should_enter_eval_mode is False
