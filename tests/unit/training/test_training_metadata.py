"""Unit tests for shared training metadata helpers."""
from types import SimpleNamespace

import pytest

from library.metadata.dataclasses import RunMetadataFacts
from library.metadata.emitters.run import TrainingMetadataState, build_objective_ss_metadata


@pytest.mark.training
@pytest.mark.unit
def test_build_objective_ss_metadata_adds_rf_fields_for_rectified_flow() -> None:
    cfg = SimpleNamespace(
        model=SimpleNamespace(model_type="sd3"),
        timestep=SimpleNamespace(
            timestep_sampling="cosine_shaped",
            rf_loss_weighting_scheme="cosmap",
            training_shift=2.0,
            logit_mean=0.1,
            logit_std=1.2,
            cosine_shape_scale=1.5,
        ),
    )

    metadata = build_objective_ss_metadata(cfg, "rectified_flow")

    assert metadata["ss_timestep_sampling"] == "cosine_shaped"
    assert metadata["ss_rf_loss_weighting_scheme"] == "cosmap"
    assert metadata["ss_training_shift"] == 2.0
    assert metadata["ss_logit_mean"] == 0.1
    assert metadata["ss_logit_std"] == 1.2
    assert metadata["ss_cosine_shape_scale"] == 1.5


@pytest.mark.training
@pytest.mark.unit
def test_build_objective_ss_metadata_skips_non_rf_objectives() -> None:
    cfg = SimpleNamespace(
        model=SimpleNamespace(model_type="sdxl"),
        timestep=SimpleNamespace(
            timestep_sampling="cosine_shaped",
            rf_loss_weighting_scheme="cosmap",
            training_shift=2.0,
            logit_mean=0.1,
            logit_std=1.2,
            cosine_shape_scale=1.5,
        ),
    )

    metadata = build_objective_ss_metadata(cfg, "ddpm")

    assert metadata == {}


@pytest.mark.training
@pytest.mark.unit
def test_training_metadata_state_updates_full_and_minimum_facts() -> None:
    state = TrainingMetadataState(
        full=RunMetadataFacts(run_identifier="run", compatibility_metadata={"ss_seed": "42"}),
        minimum=RunMetadataFacts(run_identifier="run", compatibility_metadata={}),
    )

    updated = state.with_fact("ss_adapter_rank", 16).with_fact("ss_epoch", 3)

    assert updated.full.compatibility_metadata["ss_adapter_rank"] == "16"
    assert updated.full.compatibility_metadata["ss_epoch"] == "3"
    assert updated.minimum.compatibility_metadata["ss_adapter_rank"] == "16"
    assert "ss_epoch" not in updated.minimum.compatibility_metadata
