from typing import Any

from library.objectives.ddpm import DDPM_PREDICTION_TYPE_V, resolve_ddpm_prediction_type
from library.strategies.base.contracts import CheckpointingStrategy
from library.utils.model_metadata import get_model_metadata_from_config

from library.metadata.dataclasses.model import (
    ModelArtifactFacts,
    ModelArtifactResolutionContext,
)


SD1_REFERENCE_IMPLEMENTATION = "https://github.com/CompVis/stable-diffusion"
SD2_REFERENCE_IMPLEMENTATION = "https://github.com/Stability-AI/stablediffusion"


class SdCheckpointingStrategy(CheckpointingStrategy):
    """Checkpointing facet for SD 1.5/2.0 training strategies."""

    def resolve_model_artifact_facts(self, context: ModelArtifactResolutionContext) -> ModelArtifactFacts:
        """Resolve SD1/SD2 artifact semantics without rendering compatibility keys."""
        if context.family_identifier not in {"sd", "sd1", "sd2"}:
            raise ValueError(f"SD artifact resolver received family {context.family_identifier!r}.")
        if context.artifact_role not in {"adapter", "full_model"}:
            raise ValueError(f"Unsupported SD artifact role: {context.artifact_role!r}.")
        if context.serialization_format not in {"safetensors", "ckpt", "diffusers", "diffusers_safetensors"}:
            raise ValueError(f"Unsupported SD serialization format: {context.serialization_format!r}.")

        is_v2 = context.model_version in {"sd2", "sd_v2", "sd_v2_v"}
        if context.model_version not in {"sd1", "sd2", "sd_v1", "sd_v2", "sd_v2_v"}:
            raise ValueError(f"Unsupported SD model version: {context.model_version!r}.")
        prediction_type = (
            None if context.prediction_type is None else resolve_ddpm_prediction_type(context.prediction_type)
        )
        is_v_prediction = prediction_type == DDPM_PREDICTION_TYPE_V or context.model_version == "sd_v2_v"
        architecture = "stable-diffusion-v2-768-v" if is_v2 and is_v_prediction else (
            "stable-diffusion-v2-512" if is_v2 else "stable-diffusion-v1"
        )

        is_adapter = context.artifact_role == "adapter"
        if is_adapter:
            architecture += "/lora"
        implementation = SD2_REFERENCE_IMPLEMENTATION if is_v2 else SD1_REFERENCE_IMPLEMENTATION
        default_title = f"{'LoRA' if is_adapter else 'Checkpoint'}@{context.created_at}"
        return ModelArtifactFacts.from_resolution_context(
            context,
            architecture=architecture,
            implementation=implementation,
            default_title=default_title,
        )

    def update_metadata(self, metadata: dict, cfg: Any) -> None:
        """SD currently adds no extra metadata fields."""

    def get_model_metadata(self, cfg: Any) -> dict:
        """Get SAI model spec metadata for SD."""
        prediction_type = resolve_ddpm_prediction_type(cfg.objective.prediction)
        return get_model_metadata_from_config(
            state_dict=None,
            metadata_config=cfg.output.metadata,
            is_sdxl=False,
            is_v2=cfg.model.model_type == "sd2",
            v_parameterization=prediction_type == DDPM_PREDICTION_TYPE_V,
            prediction_type=prediction_type,
            is_lora=True,
            is_textual_inversion=False,
            resolution=cfg.data.preprocessing.resolution,
            min_timestep=cfg.timestep.min_timestep,
            max_timestep=cfg.timestep.max_timestep,
            clip_skip=cfg.training.clip_skip,
        )
