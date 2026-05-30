"""Unit tests for active checkpoint metadata assembly."""

import pytest

from library.metadata.dataclasses import ModelSpecFacts, RunMetadataFacts
from library.metadata.emitters.checkpoint import build_checkpoint_metadata


@pytest.mark.training
@pytest.mark.unit
def test_build_checkpoint_metadata_projects_full_checkpoint_metadata() -> None:
    metadata = build_checkpoint_metadata(
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
        model_facts=ModelSpecFacts.from_modelspec_metadata(
            {
                "modelspec.title": "LoRA",
                "modelspec.architecture": "stable-diffusion-xl-v1-base/lora",
                "modelspec.implementation": "sgm",
                "modelspec.prediction_type": "epsilon",
            }
        ),
        no_metadata=False,
        artifact_identifier="adapter.safetensors",
        step=12,
        epoch=3,
    )

    assert metadata["modelspec.title"] == "LoRA"
    assert metadata["kuro.schema_version"] == "1"
    assert metadata["kuro.run.id"] == "123"
    assert metadata["kuro.run.seed"] == "42"
    assert metadata["kuro.run.adapter_method"] == "lora"
    assert metadata["kuro.run.adapter_rank"] == "16"
    assert metadata["kuro.run.adapter_alpha"] == "8.0"
    assert metadata["kuro.run.adapter_neuron_dropout"] == "0.1"
    assert metadata["kuro.run.scale_weight_norms"] == "1.0"
    assert metadata["kuro.artifact.id"] == "adapter.safetensors"
    assert metadata["kuro.artifact.step"] == "12"
    assert metadata["kuro.artifact.epoch"] == "3"
    assert metadata["kuro.model.architecture"] == "stable-diffusion-xl-v1-base/lora"
    assert metadata["kuro.model.implementation"] == "sgm"
    assert metadata["kuro.model.prediction_type"] == "epsilon"
    assert "kuro.adapter.adapter_rank" not in metadata


@pytest.mark.training
@pytest.mark.unit
def test_build_checkpoint_metadata_omits_training_run_metadata_when_no_metadata_is_requested() -> None:
    metadata = build_checkpoint_metadata(
        training_facts=RunMetadataFacts(
            run_identifier="123",
            metadata={
                "session_id": "123",
                "seed": "42",
                "adapter_method": "lora",
                "adapter_rank": "16",
                "adapter_alpha": "8.0",
            },
        ),
        model_facts=ModelSpecFacts.from_modelspec_metadata({"modelspec.title": "LoRA"}),
        no_metadata=True,
        artifact_identifier="adapter.safetensors",
    )

    assert metadata["modelspec.title"] == "LoRA"
    assert metadata["kuro.schema_version"] == "1"
    assert "kuro.run.id" not in metadata
    assert "kuro.artifact.id" not in metadata
    assert "kuro.run.seed" not in metadata


@pytest.mark.training
@pytest.mark.unit
def test_build_checkpoint_metadata_omits_training_run_metadata_for_finetune_like_no_metadata_run() -> None:
    metadata = build_checkpoint_metadata(
        training_facts=RunMetadataFacts(
            run_identifier="123",
            metadata={
                "session_id": "123",
                "seed": "42",
                "is_v2": "False",
                "base_model_version": "sdxl-base",
            },
        ),
        model_facts=ModelSpecFacts.from_modelspec_metadata({"modelspec.title": "Full Model"}),
        no_metadata=True,
        artifact_identifier="model.safetensors",
    )

    assert metadata["modelspec.title"] == "Full Model"
    assert metadata["kuro.schema_version"] == "1"
    assert "kuro.run.id" not in metadata
    assert "kuro.artifact.id" not in metadata
    assert "kuro.run.seed" not in metadata
