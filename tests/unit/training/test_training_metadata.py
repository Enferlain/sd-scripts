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
        timestep=SimpleNamespace(weighting_scheme="mode", logit_mean=0.1, logit_std=1.2, mode_scale=1.5),
    )

    append_objective_metadata(metadata, cfg)

    assert metadata["ss_weighting_scheme"] == "mode"
    assert metadata["ss_logit_mean"] == 0.1
    assert metadata["ss_logit_std"] == 1.2
    assert metadata["ss_mode_scale"] == 1.5


@pytest.mark.training
@pytest.mark.unit
def test_append_objective_metadata_skips_non_rf_model_families() -> None:
    metadata: dict[str, object] = {}
    cfg = SimpleNamespace(
        model=SimpleNamespace(model_type="sdxl"),
        timestep=SimpleNamespace(weighting_scheme="mode", logit_mean=0.1, logit_std=1.2, mode_scale=1.5),
    )

    append_objective_metadata(metadata, cfg)

    assert metadata == {}
