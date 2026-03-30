"""Unit tests for shared training metadata helpers."""

from types import SimpleNamespace

import pytest

from library.training.training_metadata import append_objective_metadata


@pytest.mark.training
@pytest.mark.unit
def test_append_objective_metadata_adds_rf_fields_for_sd3() -> None:
    metadata: dict[str, object] = {}
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

    append_objective_metadata(metadata, cfg, "rectified_flow")

    assert metadata["ss_timestep_sampling"] == "cosine_shaped"
    assert metadata["ss_rf_loss_weighting_scheme"] == "cosmap"
    assert metadata["ss_training_shift"] == 2.0
    assert metadata["ss_logit_mean"] == 0.1
    assert metadata["ss_logit_std"] == 1.2
    assert metadata["ss_cosine_shape_scale"] == 1.5


@pytest.mark.training
@pytest.mark.unit
def test_append_objective_metadata_skips_non_rf_model_families() -> None:
    metadata: dict[str, object] = {}
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

    append_objective_metadata(metadata, cfg, "ddpm")

    assert metadata == {}
