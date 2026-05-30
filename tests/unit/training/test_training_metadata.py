"""Unit tests for shared training metadata helpers."""
from types import SimpleNamespace

import pytest

from library.metadata.dataclasses import RunMetadataFacts
from library.metadata.emitters.run import TrainingMetadataState, build_objective_run_metadata


@pytest.mark.training
@pytest.mark.unit
def test_build_objective_run_metadata_adds_rf_fields_for_rectified_flow() -> None:
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

    metadata = build_objective_run_metadata(cfg, "rectified_flow")

    assert metadata["timestep_sampling"] == "cosine_shaped"
    assert metadata["rf_loss_weighting_scheme"] == "cosmap"
    assert metadata["training_shift"] == 2.0
    assert metadata["logit_mean"] == 0.1
    assert metadata["logit_std"] == 1.2
    assert metadata["cosine_shape_scale"] == 1.5


@pytest.mark.training
@pytest.mark.unit
def test_build_objective_run_metadata_skips_non_rf_objectives() -> None:
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

    metadata = build_objective_run_metadata(cfg, "ddpm")

    assert metadata == {}


@pytest.mark.training
@pytest.mark.unit
def test_training_metadata_state_updates_full_facts() -> None:
    state = TrainingMetadataState(
        full=RunMetadataFacts(run_identifier="run", metadata={"seed": "42"}),
    )

    updated = state.with_fact("adapter_rank", 16).with_fact("epoch", 3)

    assert updated.full.metadata["adapter_rank"] == "16"
    assert updated.full.metadata["epoch"] == "3"
