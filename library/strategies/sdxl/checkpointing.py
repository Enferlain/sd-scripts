import logging
from typing import Any

import torch

from library.objectives.ddpm import DDPM_PREDICTION_TYPE_V, resolve_ddpm_prediction_type
from library.strategies.base.contracts import CheckpointingStrategy
from library.utils.model_metadata import get_model_metadata_from_config

logger = logging.getLogger(__name__)


def resolve_sdxl_modelspec_prediction(cfg: Any) -> tuple[bool, str | None]:
    """Resolve SDXL model-spec prediction metadata for the active objective path."""
    if cfg.objective.path == "rectified_flow":
        return False, None

    prediction_type = resolve_ddpm_prediction_type(cfg.objective.prediction)
    return prediction_type == DDPM_PREDICTION_TYPE_V, prediction_type


class SdxlCheckpointingStrategy(CheckpointingStrategy):
    """Checkpointing facet for SDXL training strategies."""

    def update_metadata(self, metadata: dict, cfg: Any) -> None:
        """
        Add SDXL-specific metadata fields.

        SDXL does not currently add extra runtime metadata beyond the model
        spec metadata produced by ``get_model_metadata``.
        """
        return None

    def get_model_metadata(self, cfg: Any) -> dict:
        """Get the SAI model spec metadata for SDXL."""
        v_parameterization, prediction_type = resolve_sdxl_modelspec_prediction(cfg)
        return get_model_metadata_from_config(
            state_dict=None,
            metadata_config=cfg.output.metadata,
            is_sdxl=True,
            is_v2=False,
            v_parameterization=v_parameterization,
            prediction_type=prediction_type,
            is_lora=True,
            is_textual_inversion=False,
            resolution=cfg.data.preprocessing.resolution,
            min_timestep=cfg.timestep.min_timestep,
            max_timestep=cfg.timestep.max_timestep,
            clip_skip=cfg.training.clip_skip,
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
            v_parameterization, prediction_type = resolve_sdxl_modelspec_prediction(cfg)
            modelspec_metadata = get_model_metadata_from_config(
                state_dict=None,
                metadata_config=cfg.output.metadata,
                is_sdxl=True,
                is_v2=False,
                v_parameterization=v_parameterization,
                prediction_type=prediction_type,
                is_lora=False,
                is_textual_inversion=False,
                is_stable_diffusion_ckpt=True,
            )

            merged_metadata = {**metadata, **modelspec_metadata}

            logger.info("[checkpoint] saving checkpoint: %s", ckpt_file)
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
                merged_metadata,
                save_dtype,
            )
        else:
            out_dir = ckpt_file
            os.makedirs(out_dir, exist_ok=True)

            src_path = cfg.model.pretrained_model_name_or_path

            logger.info("[checkpoint] saving model: %s", out_dir)
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
