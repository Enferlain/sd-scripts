from __future__ import annotations

import os
from typing import Any

import torch

from library.models.sd3.conversion import save_models
from library.strategies.base.contracts import CheckpointingStrategy
from library.utils.model_metadata import get_model_metadata_from_config


class Sd3CheckpointingStrategy(CheckpointingStrategy):
    """Checkpointing facet for SD3 training strategies."""

    def update_metadata(self, metadata: dict, cfg: Any) -> None:
        """Add SD3-specific runtime metadata fields."""
        metadata["ss_apply_lg_attn_mask"] = str(bool(getattr(cfg.model, "apply_lg_attn_mask", False)))
        metadata["ss_apply_t5_attn_mask"] = str(bool(getattr(cfg.model, "apply_t5_attn_mask", False)))

    def get_model_metadata(self, cfg: Any) -> dict:
        """Get the SAI model spec metadata for SD3."""
        return get_model_metadata_from_config(
            state_dict=None,
            metadata_config=cfg.output.metadata,
            is_sdxl=False,
            is_v2=False,
            v_parameterization=False,
            prediction_type=None,
            is_lora=False,
            is_textual_inversion=False,
            resolution=cfg.data.preprocessing.resolution,
            min_timestep=cfg.timestep.min_timestep,
            max_timestep=cfg.timestep.max_timestep,
            clip_skip=cfg.training.clip_skip,
            is_stable_diffusion_ckpt=True,
            sd3_type=str(getattr(self, "_model_version", getattr(cfg.model, "sd3_type", "medium"))),
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
