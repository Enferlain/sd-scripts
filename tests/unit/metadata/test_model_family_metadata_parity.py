"""Pre-migration parity fixtures for active model-family checkpoint metadata."""

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from library.config.dataclasses.output import MetadataConfig
from library.metadata import (
    MetadataRuntime,
    ModelArtifactPresentation,
    ModelArtifactResolutionContext,
    ModelRealizationFacts,
    ModelSpecCompatibilityProjection,
    SafetensorsMetadataProjection,
    SsCompatibilityProjection,
)
from library.metadata.dataclasses import RunMetadataFacts
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


def _artifact_context(
    cfg: SimpleNamespace,
    *,
    family_identifier: str,
    model_version: str,
    artifact_identifier: str,
    artifact_role: str,
    realization_identifier: str | None = None,
) -> ModelArtifactResolutionContext:
    prediction_type = None if cfg.objective.path == "rectified_flow" else cfg.objective.prediction
    return ModelArtifactResolutionContext(
        family_identifier=family_identifier,
        model_version=model_version,
        artifact_identifier=artifact_identifier,
        artifact_role=artifact_role,
        serialization_format="safetensors",
        resolution=cfg.data.preprocessing.resolution,
        created_at=FIXED_TIMESTAMP,
        presentation=ModelArtifactPresentation(
            title=cfg.output.metadata.metadata_title,
            author=cfg.output.metadata.metadata_author,
        ),
        realization_identifier=realization_identifier,
        implementation_version=FIXED_IMPLEMENTATION_VERSION,
        prediction_type=prediction_type,
        timestep_range=(cfg.timestep.min_timestep, cfg.timestep.max_timestep),
        encoder_layer=cfg.training.clip_skip,
    )


def _project_resolved_modelspec(strategy, context: ModelArtifactResolutionContext) -> dict[str, str]:
    runtime = MetadataRuntime()
    runtime.file(strategy.resolve_model_artifact_facts(context))
    result = SafetensorsMetadataProjection.from_sequence(
        (ModelSpecCompatibilityProjection(artifact_identifier=context.artifact_identifier),)
    ).project(runtime.snapshot())
    return {key: str(value) for key, value in result.metadata.items()}


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
    ("model_type", "prediction", "expected_architecture", "expected_implementation", "expected_prediction"),
    [
        (
            "sd1",
            "epsilon",
            "stable-diffusion-v1/lora",
            "https://github.com/CompVis/stable-diffusion",
            "epsilon",
        ),
        (
            "sd2",
            "epsilon",
            "stable-diffusion-v2-512/lora",
            "https://github.com/Stability-AI/stablediffusion",
            "epsilon",
        ),
        (
            "sd2",
            "v_prediction",
            "stable-diffusion-v2-768-v/lora",
            "https://github.com/Stability-AI/stablediffusion",
            "v_prediction",
        ),
    ],
)
def test_sd_typed_modelspec_projection_matches_parity_except_corrected_implementation(
    model_type: str,
    prediction: str,
    expected_architecture: str,
    expected_implementation: str,
    expected_prediction: str,
) -> None:
    cfg = _family_cfg(
        model_type=model_type,
        objective_path="ddpm",
        prediction=prediction,
        resolution=(768, 512),
    )
    context = _artifact_context(
        cfg,
        family_identifier="sd",
        model_version=model_type,
        artifact_identifier=f"{model_type}-adapter.safetensors",
        artifact_role="adapter",
    )

    metadata = _project_resolved_modelspec(SdCheckpointingStrategy(), context)

    assert metadata == _expected_modelspec(
        architecture=expected_architecture,
        implementation=expected_implementation,
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
@pytest.mark.parametrize(
    ("objective_path", "prediction", "expected_prediction"),
    [
        ("ddpm", "epsilon", "epsilon"),
        ("ddpm", "v_prediction", "v_prediction"),
        ("rectified_flow", "flow", None),
    ],
)
def test_sdxl_typed_adapter_projection_matches_parity(
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
    context = _artifact_context(
        cfg,
        family_identifier="sdxl",
        model_version="sdxl",
        artifact_identifier=f"sdxl-{objective_path}-adapter.safetensors",
        artifact_role="adapter",
    )

    metadata = _project_resolved_modelspec(SdxlCheckpointingStrategy(), context)

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
def test_sdxl_typed_full_model_projection_matches_parity() -> None:
    cfg = _family_cfg(
        model_type="sdxl",
        objective_path="ddpm",
        prediction="epsilon",
        resolution=(1024, 1024),
    )
    context = _artifact_context(
        cfg,
        family_identifier="sdxl",
        model_version="sdxl",
        artifact_identifier="sdxl-full.safetensors",
        artifact_role="full_model",
    )

    metadata = _project_resolved_modelspec(SdxlCheckpointingStrategy(), context)

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
def test_sd3_typed_full_model_projection_uses_corrected_reference_implementation() -> None:
    cfg = _family_cfg(
        model_type="sd3",
        objective_path="rectified_flow",
        prediction="flow",
        resolution=(1024, 768),
        clip_skip=None,
        sd3_type="medium",
    )
    context = _artifact_context(
        cfg,
        family_identifier="sd3",
        model_version="medium",
        artifact_identifier="sd3-full.safetensors",
        artifact_role="full_model",
    )

    metadata = _project_resolved_modelspec(Sd3CheckpointingStrategy(), context)

    assert metadata == _expected_modelspec(
        architecture="stable-diffusion-3-medium",
        implementation="https://github.com/Stability-AI/sd3.5",
        resolution="1024x768",
        prediction_type=None,
        encoder_layer=None,
    )


@pytest.mark.unit
def test_sd3_typed_family_projection_restores_historical_attention_mask_keys() -> None:
    cfg = _family_cfg(
        model_type="sd3",
        objective_path="rectified_flow",
        prediction="flow",
        resolution=(1024, 1024),
        clip_skip=None,
        apply_lg_attn_mask=True,
        apply_t5_attn_mask=False,
    )
    realization = ModelRealizationFacts.for_run(
        run_identifier="run-sd3-parity",
        realization_key="training-target",
        family_identifier="sd3",
        model_version="medium",
    )
    context = _artifact_context(
        cfg,
        family_identifier="sd3",
        model_version="medium",
        artifact_identifier="sd3-family.safetensors",
        artifact_role="full_model",
        realization_identifier=realization.realization_identifier,
    )
    strategy = Sd3CheckpointingStrategy()
    runtime = MetadataRuntime()
    runtime.file_many(
        (
            realization,
            strategy.resolve_model_artifact_facts(context),
            strategy.resolve_model_family_metadata(
                cfg,
                run_identifier=realization.run_identifier,
                realization_identifier=realization.realization_identifier,
            ),
        )
    )

    metadata = (
        SafetensorsMetadataProjection.from_sequence((SsCompatibilityProjection(artifact_identifier=context.artifact_identifier),))
        .project(runtime.snapshot())
        .metadata
    )

    assert metadata == HISTORICAL_SD3_ATTENTION_MASK_COMPATIBILITY


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
def test_sd3_attention_masks_use_central_checkpoint_projections() -> None:
    cfg = _family_cfg(
        model_type="sd3",
        objective_path="rectified_flow",
        prediction="flow",
        resolution=(1024, 1024),
        apply_lg_attn_mask=True,
        apply_t5_attn_mask=False,
    )
    strategy = Sd3CheckpointingStrategy()
    realization = ModelRealizationFacts.for_run(
        run_identifier="run-7",
        realization_key="training-target",
        family_identifier="sd3",
        model_version="medium",
    )
    context = _artifact_context(
        cfg,
        family_identifier="sd3",
        model_version="medium",
        artifact_identifier="sd3.safetensors",
        artifact_role="full_model",
        realization_identifier=realization.realization_identifier,
    )
    runtime = MetadataRuntime()
    runtime.file_many(
        (
            realization,
            strategy.resolve_model_artifact_facts(context),
            strategy.resolve_model_family_metadata(
                cfg,
                run_identifier=realization.run_identifier,
                realization_identifier=realization.realization_identifier,
            ),
        )
    )

    metadata = build_checkpoint_metadata(
        snapshot=runtime.snapshot(),
        training_facts=RunMetadataFacts(run_identifier="run-7", metadata={}),
        no_metadata=False,
        artifact_identifier="sd3.safetensors",
        step=12,
        epoch=3,
    )

    assert {key: metadata[key] for key in HISTORICAL_SD3_ATTENTION_MASK_COMPATIBILITY} == (HISTORICAL_SD3_ATTENTION_MASK_COMPATIBILITY)
    assert metadata["kuro.model.family.apply_lg_attn_mask"] == "True"
    assert metadata["kuro.model.family.apply_t5_attn_mask"] == "False"
    assert "apply_lg_attn_mask" not in metadata
    assert "apply_t5_attn_mask" not in metadata


@pytest.mark.unit
def test_no_metadata_preserves_modelspec_and_model_identity_only() -> None:
    cfg = _family_cfg(
        model_type="sd1",
        objective_path="ddpm",
        prediction="epsilon",
        resolution=(512, 512),
    )
    context = _artifact_context(
        cfg,
        family_identifier="sd",
        model_version="sd1",
        artifact_identifier="adapter.safetensors",
        artifact_role="adapter",
    )
    runtime = MetadataRuntime()
    runtime.file(SdCheckpointingStrategy().resolve_model_artifact_facts(context))

    metadata = build_checkpoint_metadata(
        snapshot=runtime.snapshot(),
        training_facts=RunMetadataFacts(
            run_identifier="run-8",
            metadata={"seed": "42", "apply_lg_attn_mask": "True"},
        ),
        no_metadata=True,
        artifact_identifier="adapter.safetensors",
        step=99,
        epoch=4,
    )

    assert metadata == {
        "kuro.schema_version": "1",
        "kuro.model.artifact.id": "adapter.safetensors",
        "kuro.model.artifact.schema_version": "1",
        "kuro.model.artifact.producer": "model.artifact",
        "kuro.model.artifact.namespace": "kuro",
        "kuro.model.artifact.label": "Parity Artifact",
        "kuro.model.artifact.architecture": "stable-diffusion-v1/lora",
        "kuro.model.artifact.artifact_format": "safetensors",
        "kuro.model.artifact.artifact_role": "adapter",
        "kuro.model.artifact.author": "Metadata Tests",
        "kuro.model.artifact.date": FIXED_DATE,
        "kuro.model.artifact.encoder_layer": "2",
        "kuro.model.artifact.extension_fields": "{}",
        "kuro.model.artifact.family_identifier": "sd",
        "kuro.model.artifact.implementation": "https://github.com/CompVis/stable-diffusion",
        "kuro.model.artifact.implementation_version": FIXED_IMPLEMENTATION_VERSION,
        "kuro.model.artifact.prediction_type": "epsilon",
        "kuro.model.artifact.resolution": "512x512",
        "kuro.model.artifact.timestep_range": "10,900",
        "kuro.model.artifact.title": "Parity Artifact",
        **_expected_modelspec(
            architecture="stable-diffusion-v1/lora",
            implementation="https://github.com/CompVis/stable-diffusion",
            resolution="512x512",
            prediction_type="epsilon",
        ),
    }
