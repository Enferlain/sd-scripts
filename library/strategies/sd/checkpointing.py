from dataclasses import replace
from library.objectives.ddpm import DDPM_PREDICTION_TYPE_V, resolve_ddpm_prediction_type
from library.strategies.base.contracts import CheckpointingStrategy

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
        if context.artifact_role not in {"adapter", "full_model", "textual_inversion"}:
            raise ValueError(f"Unsupported SD artifact role: {context.artifact_role!r}.")
        supported_formats = {"safetensors", "ckpt", "diffusers", "diffusers_safetensors"}
        if context.artifact_role == "textual_inversion":
            supported_formats.add("pt")
        if context.serialization_format not in supported_formats:
            raise ValueError(f"Unsupported SD serialization format: {context.serialization_format!r}.")

        is_v2 = context.model_version in {"sd2", "sd_v2", "sd_v2_v"}
        if context.model_version not in {"sd1", "sd2", "sd_v1", "sd_v2", "sd_v2_v"}:
            raise ValueError(f"Unsupported SD model version: {context.model_version!r}.")
        prediction_type = None if context.prediction_type is None else resolve_ddpm_prediction_type(context.prediction_type)
        is_v_prediction = prediction_type == DDPM_PREDICTION_TYPE_V or context.model_version == "sd_v2_v"
        architecture = (
            "stable-diffusion-v2-768-v" if is_v2 and is_v_prediction else ("stable-diffusion-v2-512" if is_v2 else "stable-diffusion-v1")
        )

        if context.artifact_role == "adapter":
            architecture += "/lora"
            artifact_title = "LoRA"
        elif context.artifact_role == "textual_inversion":
            architecture += "/textual-inversion"
            artifact_title = "TextualInversion"
        else:
            artifact_title = "Checkpoint"
        implementation = SD2_REFERENCE_IMPLEMENTATION if is_v2 else SD1_REFERENCE_IMPLEMENTATION
        return ModelArtifactFacts.from_resolution_context(
            replace(context, prediction_type=prediction_type),
            architecture=architecture,
            implementation=implementation,
            default_title=f"{artifact_title}@{context.created_at}",
        )
