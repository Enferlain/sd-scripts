from __future__ import annotations

import os

from typing import Any

import torch

from library.models.sd3.conversion import save_models
from library.strategies.base.contracts import CheckpointingStrategy
from library.strategies.base.features import ModelFamilyMetadataStrategy

from library.metadata.dataclasses.model import (
    ModelArtifactFacts,
    ModelArtifactResolutionContext,
    ModelFamilyMetadataContribution,
    ModelFamilyMetadataField,
)


SD3_REFERENCE_IMPLEMENTATION = "https://github.com/Stability-AI/sd3.5"


class Sd3CheckpointingStrategy(CheckpointingStrategy, ModelFamilyMetadataStrategy):
    """Checkpointing facet for SD3 training strategies."""

    def resolve_model_artifact_facts(self, context: ModelArtifactResolutionContext) -> ModelArtifactFacts:
        """Resolve SD3 full-model facts with DDPM prediction claims omitted."""
        if context.family_identifier != "sd3":
            raise ValueError(f"SD3 artifact resolver received family {context.family_identifier!r}.")
        if context.artifact_role != "full_model":
            raise ValueError(f"Unsupported SD3 artifact role: {context.artifact_role!r}.")
        if context.serialization_format != "safetensors":
            raise ValueError(f"Unsupported SD3 serialization format: {context.serialization_format!r}.")
        if context.prediction_type is not None:
            raise ValueError("SD3 artifact facts must omit DDPM prediction_type.")

        return ModelArtifactFacts.from_resolution_context(
            context,
            architecture=f"stable-diffusion-3-{context.model_version}",
            implementation=SD3_REFERENCE_IMPLEMENTATION,
            default_title=f"Checkpoint@{context.created_at}",
        )

    def resolve_model_family_metadata(
        self,
        cfg: Any,
        *,
        run_identifier: str,
        realization_identifier: str,
    ) -> ModelFamilyMetadataContribution:
        """Resolve SD3 attention-mask settings as canonical family-local facts."""
        return ModelFamilyMetadataContribution.for_realization(
            run_identifier=run_identifier,
            realization_identifier=realization_identifier,
            contribution_namespace="sd3.checkpointing",
            contribution_version="1",
            fields=(
                ModelFamilyMetadataField(
                    name="apply_lg_attn_mask",
                    value=bool(getattr(cfg.model, "apply_lg_attn_mask", False)),
                ),
                ModelFamilyMetadataField(
                    name="apply_t5_attn_mask",
                    value=bool(getattr(cfg.model, "apply_t5_attn_mask", False)),
                ),
            ),
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
        """Serialize an SD3 full-model checkpoint."""
        del step, epoch

        cfg = trainer.cfg
        save_model_as = cfg.output.saving.save_model_as
        if save_model_as != "safetensors":
            raise NotImplementedError("SD3 full-model checkpoint saving currently supports only save_model_as='safetensors'")

        assert trainer.denoiser is not None, "Denoiser must be set before save_model_checkpoint"

        mmdit = trainer.accelerator.unwrap_model(trainer.denoiser)
        clip_l = trainer.accelerator.unwrap_model(trainer.text_encoders[0]) if len(trainer.text_encoders) > 0 else None
        clip_g = trainer.accelerator.unwrap_model(trainer.text_encoders[1]) if len(trainer.text_encoders) > 1 else None
        t5xxl = trainer.accelerator.unwrap_model(trainer.text_encoders[2]) if len(trainer.text_encoders) > 2 else None
        vae = trainer.vae

        os.makedirs(cfg.output.saving.output_dir, exist_ok=True)
        ckpt_file = os.path.join(cfg.output.saving.output_dir, ckpt_name)

        saved_paths = save_models(
            ckpt_file,
            mmdit=mmdit,
            vae=vae,
            clip_l=clip_l,
            clip_g=clip_g,
            t5xxl=t5xxl,
            metadata=metadata,
            save_dtype=save_dtype,
        )

        if cfg.output.huggingface is not None and cfg.output.huggingface.huggingface_repo_id is not None:
            from library.utils import huggingface_util

            for saved_path in saved_paths:
                file_name = os.path.basename(saved_path)
                huggingface_util.upload(
                    cfg.output.huggingface,
                    saved_path,
                    "/" + file_name,
                    force_sync_upload=force_sync_upload,
                )


__all__ = ["Sd3CheckpointingStrategy"]
