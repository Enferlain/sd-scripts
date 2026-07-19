"""Family-owned canonical model metadata resolver contracts."""

import importlib.util

from dataclasses import fields, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from library.metadata.runtime import MetadataRuntime
from library.strategies.base.contracts import CheckpointingStrategy
from library.strategies.base.features import ModelFamilyMetadataStrategy

from library.metadata import (
    ModelArtifactFacts,
    ModelArtifactPresentation,
    ModelArtifactResolutionContext,
    ModelFamilyMetadataContribution,
)


def _load_checkpointing_module(family: str) -> ModuleType:
    """Load a narrow facet without importing each family's eager package facade."""
    module_path = Path(__file__).parents[3] / "library" / "strategies" / family / "checkpointing.py"
    spec = importlib.util.spec_from_file_location(f"_typed_{family}_checkpointing", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SdCheckpointingStrategy = _load_checkpointing_module("sd").SdCheckpointingStrategy
Sd3CheckpointingStrategy = _load_checkpointing_module("sd3").Sd3CheckpointingStrategy
SdxlCheckpointingStrategy = _load_checkpointing_module("sdxl").SdxlCheckpointingStrategy


def _context(
    *,
    family: str,
    version: str,
    role: str = "adapter",
    prediction_type: str | None = "epsilon",
    serialization_format: str = "safetensors",
    resolution: tuple[int, int] = (1024, 768),
) -> ModelArtifactResolutionContext:
    return ModelArtifactResolutionContext(
        family_identifier=family,
        model_version=version,
        artifact_identifier=f"{family}-{role}.safetensors",
        artifact_role=role,
        serialization_format=serialization_format,
        resolution=resolution,
        created_at=946684800.0,
        presentation=ModelArtifactPresentation(
            title="Canonical Artifact",
            author="Metadata Tests",
            extension_fields={"custom_field": "custom-value"},
        ),
        implementation_version="sd-scripts/test",
        prediction_type=prediction_type,
        timestep_range=(10, 900),
        encoder_layer=2,
    )


def _assert_canonical(facts: ModelArtifactFacts) -> None:
    assert all(not field.name.startswith(("modelspec.", "ss_")) for field in fields(facts))
    assert all(not key.startswith(("modelspec.", "ss_")) for key in facts.extension_fields)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("version", "prediction_type", "expected_architecture", "expected_implementation"),
    [
        (
            "sd_v1",
            "epsilon",
            "stable-diffusion-v1/lora",
            "https://github.com/CompVis/stable-diffusion",
        ),
        (
            "sd_v2",
            "epsilon",
            "stable-diffusion-v2-512/lora",
            "https://github.com/Stability-AI/stablediffusion",
        ),
        (
            "sd_v2_v",
            "v_prediction",
            "stable-diffusion-v2-768-v/lora",
            "https://github.com/Stability-AI/stablediffusion",
        ),
    ],
)
def test_sd_resolver_returns_canonical_adapter_facts(
    version: str,
    prediction_type: str,
    expected_architecture: str,
    expected_implementation: str,
) -> None:
    context = _context(family="sd", version=version, prediction_type=prediction_type)

    facts = SdCheckpointingStrategy().resolve_model_artifact_facts(context)

    assert facts.architecture == expected_architecture
    assert facts.implementation == expected_implementation
    assert facts.prediction_type == prediction_type
    assert facts.resolution == "1024x768"
    assert facts.timestep_range == "10,900"
    assert facts.encoder_layer == "2"
    assert facts.implementation_version == "sd-scripts/test"
    _assert_canonical(facts)


@pytest.mark.unit
def test_sd_resolver_distinguishes_full_model_artifact_semantics() -> None:
    context = _context(
        family="sd",
        version="sd_v1",
        role="full_model",
        serialization_format="safetensors",
    )

    facts = SdCheckpointingStrategy().resolve_model_artifact_facts(context)

    assert facts.architecture == "stable-diffusion-v1"
    assert facts.implementation == "https://github.com/CompVis/stable-diffusion"
    assert facts.artifact_role == "full_model"
    _assert_canonical(facts)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("version", "serialization_format", "expected_architecture"),
    [
        ("sd_v1", "safetensors", "stable-diffusion-v1/textual-inversion"),
        ("sd_v2_v", "pt", "stable-diffusion-v2-768-v/textual-inversion"),
    ],
)
def test_sd_resolver_supports_textual_inversion_artifacts(
    version: str,
    serialization_format: str,
    expected_architecture: str,
) -> None:
    context = replace(
        _context(
            family="sd",
            version=version,
            role="textual_inversion",
            prediction_type="v_prediction" if version == "sd_v2_v" else "epsilon",
        ),
        serialization_format=serialization_format,
        presentation=ModelArtifactPresentation(),
    )

    facts = SdCheckpointingStrategy().resolve_model_artifact_facts(context)

    assert facts.architecture == expected_architecture
    assert facts.artifact_role == "textual_inversion"
    assert facts.title == "TextualInversion@946684800.0"
    _assert_canonical(facts)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("context", "error"),
    [
        (_context(family="sd", version="sd_v1", prediction_type="invalid"), "prediction type"),
        (
            _context(family="sd", version="sd_v1", serialization_format="pickle"),
            "serialization format",
        ),
    ],
)
def test_sd_resolver_rejects_unsupported_semantics(
    context: ModelArtifactResolutionContext,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        SdCheckpointingStrategy().resolve_model_artifact_facts(context)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("role", "serialization_format", "prediction_type", "architecture", "implementation"),
    [
        (
            "adapter",
            "safetensors",
            "epsilon",
            "stable-diffusion-xl-v1-base/lora",
            "https://github.com/Stability-AI/generative-models",
        ),
        (
            "full_model",
            "safetensors",
            "epsilon",
            "stable-diffusion-xl-v1-base",
            "https://github.com/Stability-AI/generative-models",
        ),
        (
            "full_model",
            "diffusers",
            "epsilon",
            "stable-diffusion-xl-v1-base",
            "https://github.com/Stability-AI/generative-models",
        ),
        (
            "adapter",
            "safetensors",
            None,
            "stable-diffusion-xl-v1-base/lora",
            "https://github.com/Stability-AI/generative-models",
        ),
    ],
)
def test_sdxl_resolver_handles_artifact_role_and_rf_omission(
    role: str,
    serialization_format: str,
    prediction_type: str | None,
    architecture: str,
    implementation: str,
) -> None:
    context = _context(
        family="sdxl",
        version="base",
        role=role,
        prediction_type=prediction_type,
        serialization_format=serialization_format,
        resolution=(1024, 1024),
    )

    facts = SdxlCheckpointingStrategy().resolve_model_artifact_facts(context)

    assert facts.architecture == architecture
    assert facts.implementation == implementation
    assert facts.prediction_type == prediction_type
    assert facts.resolution == "1024x1024"
    _assert_canonical(facts)


@pytest.mark.unit
def test_sdxl_resolver_supports_textual_inversion_artifacts() -> None:
    context = replace(
        _context(
            family="sdxl",
            version="sdxl_base_v1-0",
            role="textual_inversion",
            prediction_type="epsilon",
        ),
        presentation=ModelArtifactPresentation(),
    )

    facts = SdxlCheckpointingStrategy().resolve_model_artifact_facts(context)

    assert facts.architecture == "stable-diffusion-xl-v1-base/textual-inversion"
    assert facts.artifact_role == "textual_inversion"
    assert facts.title == "TextualInversion@946684800.0"
    _assert_canonical(facts)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("context", "error"),
    [
        (_context(family="sdxl", version="base", prediction_type="invalid"), "prediction type"),
        (
            _context(family="sdxl", version="base", serialization_format="pickle"),
            "serialization format",
        ),
    ],
)
def test_sdxl_resolver_rejects_unsupported_semantics(
    context: ModelArtifactResolutionContext,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        SdxlCheckpointingStrategy().resolve_model_artifact_facts(context)


@pytest.mark.unit
def test_sd3_resolver_returns_canonical_artifact_and_attention_mask_facts() -> None:
    strategy = Sd3CheckpointingStrategy()
    context = _context(
        family="sd3",
        version="medium",
        role="full_model",
        prediction_type=None,
    )
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            apply_lg_attn_mask=True,
            apply_t5_attn_mask=False,
        )
    )

    artifact_facts = strategy.resolve_model_artifact_facts(context)
    contribution = strategy.resolve_model_family_metadata(
        cfg,
        run_identifier="run-sd3",
        realization_identifier="run/run-sd3/model/target",
    )

    assert artifact_facts.architecture == "stable-diffusion-3-medium"
    assert artifact_facts.implementation == "https://github.com/Stability-AI/sd3.5"
    assert artifact_facts.prediction_type is None
    assert isinstance(contribution, ModelFamilyMetadataContribution)
    assert contribution.contribution_namespace == "sd3.checkpointing"
    assert {field.name: field.value for field in contribution.fields} == {
        "apply_lg_attn_mask": True,
        "apply_t5_attn_mask": False,
    }
    assert isinstance(strategy, ModelFamilyMetadataStrategy)
    _assert_canonical(artifact_facts)
    assert all(not field.name.startswith(("modelspec.", "ss_")) for field in contribution.fields)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("version", "architecture"),
    [
        ("medium", "stable-diffusion-3-medium"),
        ("5-medium", "stable-diffusion-3-5-medium"),
        ("5-large", "stable-diffusion-3-5-large"),
    ],
)
def test_sd3_resolver_uses_sd3_reference_for_supported_loaded_versions(
    version: str,
    architecture: str,
) -> None:
    facts = Sd3CheckpointingStrategy().resolve_model_artifact_facts(
        _context(
            family="sd3",
            version=version,
            role="full_model",
            prediction_type=None,
        )
    )

    assert facts.architecture == architecture
    assert facts.implementation == "https://github.com/Stability-AI/sd3.5"


@pytest.mark.unit
def test_sd3_resolver_rejects_unsupported_serialization_format() -> None:
    context = _context(
        family="sd3",
        version="medium",
        role="full_model",
        prediction_type=None,
        serialization_format="diffusers",
    )

    with pytest.raises(ValueError, match="serialization format"):
        Sd3CheckpointingStrategy().resolve_model_artifact_facts(context)


class _FutureCheckpointingStrategy(CheckpointingStrategy):
    def resolve_model_artifact_facts(self, context: ModelArtifactResolutionContext) -> ModelArtifactFacts:
        return ModelArtifactFacts.from_resolution_context(
            context,
            architecture="future-v1/adapter",
            implementation="future-runtime",
            default_title="Future Artifact",
        )


@pytest.mark.unit
def test_future_family_only_needs_typed_resolver_for_central_runtime() -> None:
    strategy = _FutureCheckpointingStrategy()
    facts = strategy.resolve_model_artifact_facts(
        _context(
            family="future-family",
            version="v1",
        )
    )
    runtime = MetadataRuntime()

    runtime.file(facts)
    record = runtime.snapshot().record_for(
        entity_type="model_artifact",
        identifier=facts.artifact_identifier,
    )

    assert record is not None
    assert record.facts["family_identifier"] == "future-family"
    assert record.facts["architecture"] == "future-v1/adapter"
    assert not hasattr(strategy, "get_model_metadata")
    assert not hasattr(strategy, "update_metadata")
