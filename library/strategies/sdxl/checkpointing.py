from dataclasses import replace
from typing import Any

import torch

from library.metadata.dataclasses.model import (
    ModelArtifactFacts,
    ModelArtifactResolutionContext,
)
from library.objectives.ddpm import resolve_ddpm_prediction_type
from library.strategies.base.contracts import CheckpointingStrategy


SDXL_REFERENCE_IMPLEMENTATION = "https://github.com/Stability-AI/generative-models"


class SdxlCheckpointingStrategy(CheckpointingStrategy):
    """Checkpointing facet for SDXL training strategies."""

    def resolve_model_artifact_facts(self, context: ModelArtifactResolutionContext) -> ModelArtifactFacts:
        """Resolve SDXL adapter/full-model facts and preserve RF omission."""
        if context.family_identifier != "sdxl":
            raise ValueError(f"SDXL artifact resolver received family {context.family_identifier!r}.")
        if context.artifact_role not in {"adapter", "full_model", "textual_inversion"}:
            raise ValueError(f"Unsupported SDXL artifact role: {context.artifact_role!r}.")
        supported_formats = {"safetensors", "ckpt", "diffusers", "diffusers_safetensors"}
        if context.artifact_role == "textual_inversion":
            supported_formats.add("pt")
        if context.serialization_format not in supported_formats:
            raise ValueError(f"Unsupported SDXL serialization format: {context.serialization_format!r}.")
        prediction_type = None if context.prediction_type is None else resolve_ddpm_prediction_type(context.prediction_type)

        architecture = "stable-diffusion-xl-v1-base"
        if context.artifact_role == "adapter":
            architecture += "/lora"
            artifact_title = "LoRA"
        elif context.artifact_role == "textual_inversion":
            architecture += "/textual-inversion"
            artifact_title = "TextualInversion"
        else:
            artifact_title = "Checkpoint"
        return ModelArtifactFacts.from_resolution_context(
            replace(context, prediction_type=prediction_type),
            architecture=architecture,
            implementation=SDXL_REFERENCE_IMPLEMENTATION,
            default_title=f"{artifact_title}@{context.created_at}",
        )

    def save_model_checkpoint(
        self,
        trainer: Any,
        ckpt_name: str,
        step: int,
        epoch: int,
        metadata: dict[str, str],
        save_dtype: torch.dtype,
        force_sync_upload: bool = False,
    ) -> None:
        """Serialize an SDXL full-model checkpoint."""
        import os

        from library.models.sdxl.conversion import (
            save_diffusers_checkpoint,
            save_stable_diffusion_checkpoint,
        )

        cfg = trainer.cfg

        save_model_as = cfg.output.saving.save_model_as
        save_stable_diffusion_format = save_model_as in ("safetensors", "ckpt")
        use_safetensors = save_model_as == "safetensors"

        assert trainer.denoiser is not None, "Denoiser must be set before save_model_checkpoint"
        unet = trainer.accelerator.unwrap_model(trainer.denoiser)
        text_encoder1 = trainer.accelerator.unwrap_model(trainer.text_encoders[0])
        text_encoder2 = trainer.accelerator.unwrap_model(trainer.text_encoders[1]) if len(trainer.text_encoders) > 1 else None
        vae = trainer.vae

        os.makedirs(cfg.output.saving.output_dir, exist_ok=True)
        ckpt_file = os.path.join(cfg.output.saving.output_dir, ckpt_name)

        if save_stable_diffusion_format:
            save_stable_diffusion_checkpoint(
                ckpt_file,
                text_encoder1,
                text_encoder2,
                unet,
                epoch,
                step,
                self.ckpt_info,
                vae,
                self.logit_scale,
                metadata,
                save_dtype,
            )
        else:
            out_dir = ckpt_file
            os.makedirs(out_dir, exist_ok=True)

            src_path = cfg.model.pretrained_model_name_or_path

            save_diffusers_checkpoint(
                out_dir,
                text_encoder1,
                text_encoder2,
                unet,
                src_path,
                vae,
                use_safetensors=use_safetensors,
                save_dtype=save_dtype,
            )

        if cfg.output.huggingface is not None and cfg.output.huggingface.huggingface_repo_id is not None:
            from library.utils import huggingface_util

            huggingface_util.upload(
                cfg.output.huggingface,
                ckpt_file,
                "/" + ckpt_name,
                force_sync_upload=force_sync_upload,
            )
