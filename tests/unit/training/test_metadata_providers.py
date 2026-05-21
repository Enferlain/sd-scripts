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
            compatibility_metadata={
                "ss_session_id": "123",
                "ss_seed": "42",
                "ss_adapter_module": "networks.lora",
                "ss_adapter_rank": "16",
                "ss_adapter_alpha": "8.0",
                "ss_adapter_neuron_dropout": "0.1",
                "ss_scale_weight_norms": "1.0",
            },
        ),
        minimum_training_facts=RunMetadataFacts(
            run_identifier="123",
            compatibility_metadata={"ss_adapter_module": "networks.lora"},
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

    assert metadata["ss_seed"] == "42"
    assert metadata["ss_adapter_module"] == "networks.lora"
    assert metadata["ss_adapter_rank"] == "16"
    assert metadata["ss_adapter_alpha"] == "8.0"
    assert metadata["ss_adapter_neuron_dropout"] == "0.1"
    assert metadata["ss_scale_weight_norms"] == "1.0"
    assert metadata["modelspec.title"] == "LoRA"
    assert metadata["kuro.schema_version"] == "1"
    assert metadata["kuro.run.id"] == "123"
    assert metadata["kuro.artifact.id"] == "adapter.safetensors"
    assert metadata["kuro.artifact.step"] == "12"
    assert metadata["kuro.artifact.epoch"] == "3"
    assert metadata["kuro.model.architecture"] == "stable-diffusion-xl-v1-base/lora"
    assert metadata["kuro.model.implementation"] == "sgm"
    assert metadata["kuro.model.prediction_type"] == "epsilon"
    assert "kuro.adapter.adapter_rank" not in metadata


@pytest.mark.training
@pytest.mark.unit
def test_build_checkpoint_metadata_preserves_no_metadata_minimum_policy() -> None:
    metadata = build_checkpoint_metadata(
        training_facts=RunMetadataFacts(
            run_identifier="123",
            compatibility_metadata={
                "ss_session_id": "123",
                "ss_seed": "42",
                "ss_adapter_module": "networks.lora",
                "ss_adapter_rank": "16",
            },
        ),
        minimum_training_facts=RunMetadataFacts(
            run_identifier="123",
            compatibility_metadata={"ss_adapter_module": "networks.lora"},
        ),
        model_facts=ModelSpecFacts.from_modelspec_metadata({"modelspec.title": "LoRA"}),
        no_metadata=True,
        artifact_identifier="adapter.safetensors",
    )

    assert metadata == {
        "ss_adapter_module": "networks.lora",
        "modelspec.title": "LoRA",
    }
