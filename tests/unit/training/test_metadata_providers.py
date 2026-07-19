"""Unit tests for active checkpoint metadata assembly."""

import pytest

from library.metadata import (
    MetadataRuntime,
    ModelArtifactFacts,
    ModelRealizationFacts,
    RunMetadataFacts,
)
from library.metadata.emitters.checkpoint import build_checkpoint_metadata, build_model_artifact_export_metadata


def _accepted_model_snapshot(*, artifact_identifier: str, title: str = "LoRA"):
    runtime = MetadataRuntime()
    realization = ModelRealizationFacts.for_run(
        run_identifier="123",
        realization_key="training-target",
        family_identifier="sdxl",
        model_version="sdxl_base_v1-0",
    )
    runtime.file_many(
        (
            realization,
            ModelArtifactFacts(
                artifact_identifier=artifact_identifier,
                family_identifier="sdxl",
                artifact_role="adapter",
                artifact_format="safetensors",
                architecture="stable-diffusion-xl-v1-base/lora",
                implementation="https://github.com/Stability-AI/generative-models",
                title=title,
                resolution="1024x1024",
                realization_identifier=realization.realization_identifier,
                prediction_type="epsilon",
            ),
        )
    )
    return runtime.snapshot()


@pytest.mark.training
@pytest.mark.unit
def test_build_checkpoint_metadata_projects_accepted_artifact_and_run_facts() -> None:
    artifact_identifier = "adapter.safetensors"
    metadata = build_checkpoint_metadata(
        snapshot=_accepted_model_snapshot(artifact_identifier=artifact_identifier),
        training_facts=RunMetadataFacts(
            run_identifier="123",
            metadata={
                "session_id": "123",
                "seed": "42",
                "adapter_method": "lora",
                "adapter_rank": "16",
                "adapter_alpha": "8.0",
                "adapter_neuron_dropout": "0.1",
                "scale_weight_norms": "1.0",
            },
        ),
        no_metadata=False,
        artifact_identifier=artifact_identifier,
        artifact_format="safetensors",
        step=12,
        epoch=3,
    )

    assert metadata["modelspec.title"] == "LoRA"
    assert metadata["modelspec.architecture"] == "stable-diffusion-xl-v1-base/lora"
    assert metadata["kuro.schema_version"] == "1"
    assert metadata["kuro.run.id"] == "123"
    assert metadata["kuro.run.seed"] == "42"
    assert metadata["kuro.run.adapter_method"] == "lora"
    assert metadata["kuro.run.adapter_rank"] == "16"
    assert metadata["kuro.run.adapter_alpha"] == "8.0"
    assert metadata["kuro.run.adapter_neuron_dropout"] == "0.1"
    assert metadata["kuro.run.scale_weight_norms"] == "1.0"
    assert metadata["kuro.artifact.id"] == artifact_identifier
    assert metadata["kuro.artifact.format"] == "safetensors"
    assert metadata["kuro.artifact.step"] == "12"
    assert metadata["kuro.artifact.epoch"] == "3"
    assert metadata["kuro.model.artifact.id"] == artifact_identifier
    assert metadata["kuro.model.artifact.architecture"] == "stable-diffusion-xl-v1-base/lora"
    assert metadata["kuro.model.artifact.prediction_type"] == "epsilon"
    assert metadata["kuro.model.realization.id"] == "run/123/model/training-target"


@pytest.mark.training
@pytest.mark.unit
@pytest.mark.parametrize(
    ("artifact_identifier", "title"),
    [
        ("adapter.safetensors", "LoRA"),
        ("model.safetensors", "Full Model"),
    ],
)
def test_build_checkpoint_metadata_preserves_model_only_output_when_no_metadata_is_requested(
    artifact_identifier: str,
    title: str,
) -> None:
    metadata = build_checkpoint_metadata(
        snapshot=_accepted_model_snapshot(
            artifact_identifier=artifact_identifier,
            title=title,
        ),
        training_facts=RunMetadataFacts(
            run_identifier="123",
            metadata={"session_id": "123", "seed": "42"},
        ),
        no_metadata=True,
        artifact_identifier=artifact_identifier,
    )

    assert metadata["modelspec.title"] == title
    assert metadata["kuro.schema_version"] == "1"
    assert metadata["kuro.model.artifact.id"] == artifact_identifier
    assert "kuro.run.id" not in metadata
    assert "kuro.artifact.id" not in metadata
    assert "kuro.model.realization.id" not in metadata
    assert not any(key.startswith("ss_") for key in metadata)


@pytest.mark.training
@pytest.mark.unit
def test_standalone_model_artifact_export_projects_textual_inversion_facts() -> None:
    metadata = build_model_artifact_export_metadata(
        ModelArtifactFacts(
            artifact_identifier="embedding.safetensors",
            family_identifier="sdxl",
            artifact_role="textual_inversion",
            artifact_format="safetensors",
            architecture="stable-diffusion-xl-v1-base/textual-inversion",
            implementation="https://github.com/Stability-AI/generative-models",
            title="TextualInversion@946684800.0",
            resolution="1024x1024",
            prediction_type="epsilon",
        )
    )

    assert metadata["modelspec.sai_model_spec"] == "1.0.1"
    assert metadata["modelspec.architecture"] == "stable-diffusion-xl-v1-base/textual-inversion"
    assert metadata["modelspec.title"] == "TextualInversion@946684800.0"
    assert metadata["kuro.model.artifact.id"] == "embedding.safetensors"
    assert metadata["kuro.model.artifact.artifact_role"] == "textual_inversion"
