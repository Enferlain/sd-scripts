"""Pre-migration parity fixtures for active model-family checkpoint metadata."""

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from library.config.dataclasses.output import MetadataConfig
from library.metadata.dataclasses import ModelSpecFacts, RunMetadataFacts
from library.metadata.emitters.checkpoint import build_checkpoint_metadata
from library.utils import model_metadata


def _load_checkpointing_module(family: str) -> ModuleType:
    """Load a narrow facet without importing each family's eager package facade."""
    module_path = Path(__file__).parents[3] / "library" / "strategies" / family / "checkpointing.py"
    spec = importlib.util.spec_from_file_location(f"_parity_{family}_checkpointing", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SdCheckpointingStrategy = _load_checkpointing_module("sd").SdCheckpointingStrategy
Sd3CheckpointingStrategy = _load_checkpointing_module("sd3").Sd3CheckpointingStrategy
SdxlCheckpointingStrategy = _load_checkpointing_module("sdxl").SdxlCheckpointingStrategy


FIXED_TIMESTAMP = 946684800.0
FIXED_DATE = "2000-01-01T00:00:00"
FIXED_IMPLEMENTATION_VERSION = "sd-scripts/parity-fixture"

# CHANGELOG_HISTORY records these as the compatibility names retained for SD3.
# The immediate pre-migration checkpoint path no longer emits them; the focused
# test below captures both that historical contract and the current observable
# ``kuro.run.*`` projection so the migration can repair it intentionally.
HISTORICAL_SD3_ATTENTION_MASK_COMPATIBILITY = {
    "ss_apply_lg_attn_mask": "True",
    "ss_apply_t5_attn_mask": "False",
}


class _FrozenDateTime:
    @classmethod
    def fromtimestamp(cls, timestamp: float) -> "_FrozenDateTime":
        assert timestamp == FIXED_TIMESTAMP
        return cls()

    def isoformat(self) -> str:
        return FIXED_DATE


@pytest.fixture(autouse=True)
def _freeze_model_metadata_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_metadata.time, "time", lambda: FIXED_TIMESTAMP)
    monkeypatch.setattr(model_metadata, "datetime", SimpleNamespace(datetime=_FrozenDateTime))
    monkeypatch.setattr(model_metadata, "get_implementation_version", lambda: FIXED_IMPLEMENTATION_VERSION)


def _family_cfg(
    *,
    model_type: str,
    objective_path: str,
    prediction: str,
    resolution: tuple[int, int],
    clip_skip: int | None = 2,
    sd3_type: str = "medium",
    apply_lg_attn_mask: bool = False,
    apply_t5_attn_mask: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            model_type=model_type,
            sd3_type=sd3_type,
            apply_lg_attn_mask=apply_lg_attn_mask,
            apply_t5_attn_mask=apply_t5_attn_mask,
        ),
        objective=SimpleNamespace(path=objective_path, prediction=prediction),
        output=SimpleNamespace(
            metadata=MetadataConfig(
                metadata_title="Parity Artifact",
                metadata_author="Metadata Tests",
            )
        ),
        data=SimpleNamespace(preprocessing=SimpleNamespace(resolution=resolution)),
        timestep=SimpleNamespace(min_timestep=10, max_timestep=900),
        training=SimpleNamespace(clip_skip=clip_skip),
    )


def _expected_modelspec(
    *,
    architecture: str,
    implementation: str,
    resolution: str,
    prediction_type: str | None,
    encoder_layer: str | None = "2",
) -> dict[str, str]:
    expected = {
        "modelspec.architecture": architecture,
        "modelspec.implementation": implementation,
        "modelspec.title": "Parity Artifact",
        "modelspec.resolution": resolution,
        "modelspec.sai_model_spec": "1.0.1",
        "modelspec.author": "Metadata Tests",
        "modelspec.date": FIXED_DATE,
        "modelspec.implementation_version": FIXED_IMPLEMENTATION_VERSION,
        "modelspec.timestep_range": "10,900",
    }
    if prediction_type is not None:
        expected["modelspec.prediction_type"] = prediction_type
    if encoder_layer is not None:
        expected["modelspec.encoder_layer"] = encoder_layer
    return expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model_type", "prediction", "expected_architecture", "expected_prediction"),
    [
        ("sd1", "epsilon", "stable-diffusion-v1/lora", "epsilon"),
        ("sd2", "epsilon", "stable-diffusion-v2-512/lora", "epsilon"),
        ("sd2", "v_prediction", "stable-diffusion-v2-768-v/lora", "v_prediction"),
    ],
)
def test_sd_adapter_modelspec_output_parity(
    model_type: str,
    prediction: str,
    expected_architecture: str,
    expected_prediction: str,
) -> None:
    cfg = _family_cfg(
        model_type=model_type,
        objective_path="ddpm",
        prediction=prediction,
        resolution=(768, 512),
    )

    metadata = SdCheckpointingStrategy().get_model_metadata(cfg)

    assert metadata == _expected_modelspec(
        architecture=expected_architecture,
        implementation="diffusers",
        resolution="768x512",
        prediction_type=expected_prediction,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("objective_path", "prediction", "expected_prediction"),
    [
        ("ddpm", "epsilon", "epsilon"),
        ("ddpm", "v_prediction", "v_prediction"),
        ("rectified_flow", "flow", None),
    ],
)
def test_sdxl_adapter_modelspec_output_parity(
    objective_path: str,
    prediction: str,
    expected_prediction: str | None,
) -> None:
    cfg = _family_cfg(
        model_type="sdxl",
        objective_path=objective_path,
        prediction=prediction,
        resolution=(1024, 1024),
    )

    metadata = SdxlCheckpointingStrategy().get_model_metadata(cfg)

    assert metadata == _expected_modelspec(
        architecture="stable-diffusion-xl-v1-base/lora",
        implementation="https://github.com/Stability-AI/generative-models",
        resolution="1024x1024",
        prediction_type=expected_prediction,
    )


@pytest.mark.unit
def test_sdxl_full_model_role_has_checkpoint_architecture_and_implementation() -> None:
    metadata = model_metadata.get_model_metadata_from_config(
        state_dict=None,
        metadata_config=MetadataConfig(
            metadata_title="Parity Artifact",
            metadata_author="Metadata Tests",
        ),
        is_sdxl=True,
        is_v2=False,
        v_parameterization=False,
        prediction_type="epsilon",
        is_lora=False,
        is_textual_inversion=False,
        resolution=(1024, 1024),
        min_timestep=10,
        max_timestep=900,
        clip_skip=2,
        is_stable_diffusion_ckpt=True,
    )

    assert metadata == _expected_modelspec(
        architecture="stable-diffusion-xl-v1-base",
        implementation="https://github.com/Stability-AI/generative-models",
        resolution="1024x1024",
        prediction_type="epsilon",
    )


@pytest.mark.unit
def test_sd3_full_model_modelspec_output_parity() -> None:
    cfg = _family_cfg(
        model_type="sd3",
        objective_path="rectified_flow",
        prediction="flow",
        resolution=(1024, 768),
        clip_skip=None,
        sd3_type="medium",
    )

    metadata = Sd3CheckpointingStrategy().get_model_metadata(cfg)

    assert metadata == _expected_modelspec(
        architecture="stable-diffusion-3-medium",
        implementation="https://github.com/Stability-AI/generative-models",
        resolution="1024x768",
        prediction_type=None,
        encoder_layer=None,
    )


@pytest.mark.unit
def test_user_modelspec_extensions_currently_override_standard_fields_on_collision() -> None:
    metadata = model_metadata.get_model_metadata_from_config(
        state_dict=None,
        metadata_config=MetadataConfig(metadata_title="Configured Title"),
        is_sdxl=False,
        is_v2=False,
        v_parameterization=False,
        is_lora=True,
        is_textual_inversion=False,
        optional_metadata={
            "architecture": "user-architecture",
            "title": "User Extension Title",
            "sai_model_spec": "user-version",
            "custom_field": "custom-value",
        },
    )

    assert metadata == {
        "modelspec.architecture": "user-architecture",
        "modelspec.implementation": "diffusers",
        "modelspec.title": "User Extension Title",
        "modelspec.resolution": "512x512",
        "modelspec.sai_model_spec": "user-version",
        "modelspec.date": FIXED_DATE,
        "modelspec.implementation_version": FIXED_IMPLEMENTATION_VERSION,
        "modelspec.prediction_type": "epsilon",
        "modelspec.custom_field": "custom-value",
    }


@pytest.mark.unit
def test_sd3_attention_masks_use_current_kuro_checkpoint_projection() -> None:
    cfg = _family_cfg(
        model_type="sd3",
        objective_path="rectified_flow",
        prediction="flow",
        resolution=(1024, 1024),
        apply_lg_attn_mask=True,
        apply_t5_attn_mask=False,
    )
    run_metadata: dict[str, str] = {}
    strategy = Sd3CheckpointingStrategy()
    strategy.update_metadata(run_metadata, cfg)

    metadata = build_checkpoint_metadata(
        training_facts=RunMetadataFacts(run_identifier="run-7", metadata=run_metadata),
        model_facts=ModelSpecFacts.from_modelspec_metadata(strategy.get_model_metadata(cfg)),
        no_metadata=False,
        artifact_identifier="sd3.safetensors",
        step=12,
        epoch=3,
    )

    assert run_metadata == {
        "apply_lg_attn_mask": "True",
        "apply_t5_attn_mask": "False",
    }
    assert metadata["kuro.run.apply_lg_attn_mask"] == "True"
    assert metadata["kuro.run.apply_t5_attn_mask"] == "False"
    assert HISTORICAL_SD3_ATTENTION_MASK_COMPATIBILITY.keys().isdisjoint(metadata)
    assert "apply_lg_attn_mask" not in metadata
    assert "apply_t5_attn_mask" not in metadata


@pytest.mark.unit
def test_no_metadata_preserves_modelspec_and_model_identity_only() -> None:
    modelspec = _expected_modelspec(
        architecture="stable-diffusion-v1/lora",
        implementation="diffusers",
        resolution="512x512",
        prediction_type="epsilon",
    )

    metadata = build_checkpoint_metadata(
        training_facts=RunMetadataFacts(
            run_identifier="run-8",
            metadata={"seed": "42", "apply_lg_attn_mask": "True"},
        ),
        model_facts=ModelSpecFacts.from_modelspec_metadata(modelspec),
        no_metadata=True,
        artifact_identifier="adapter.safetensors",
        step=99,
        epoch=4,
    )

    assert metadata == {
        "kuro.schema_version": "1",
        "kuro.model.id": "active-model",
        "kuro.model.schema_version": "1",
        "kuro.model.producer": "model.modelspec",
        "kuro.model.architecture": "stable-diffusion-v1/lora",
        "kuro.model.implementation": "diffusers",
        "kuro.model.prediction_type": "epsilon",
        **modelspec,
    }
