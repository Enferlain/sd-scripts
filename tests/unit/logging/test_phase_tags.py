"""Unit tests for structured runtime trace phase-tag helpers."""

from __future__ import annotations

from library.logging.phase_tags import is_training_epoch_phase, training_epoch_phase


def test_is_training_epoch_phase_accepts_numeric_epoch_tags_only():
    assert is_training_epoch_phase(training_epoch_phase(0))
    assert is_training_epoch_phase(training_epoch_phase(12))


def test_is_training_epoch_phase_rejects_non_numeric_or_non_epoch_tags():
    assert not is_training_epoch_phase(None)
    assert not is_training_epoch_phase("")
    assert not is_training_epoch_phase("training.epoch.")
    assert not is_training_epoch_phase("training.epoch.setup")
    assert not is_training_epoch_phase("training.epoch.1.summary")
    assert not is_training_epoch_phase("training.epoch.-1")
    assert not is_training_epoch_phase("startup.metadata")
