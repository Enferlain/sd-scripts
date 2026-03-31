from typing import Any

from library.objectives.ddpm import DDPM_PREDICTION_TYPE_V, resolve_ddpm_prediction_type
from library.strategies.base.contracts import CheckpointingStrategy
from library.utils.model_metadata import get_model_metadata_from_config


class SdCheckpointingStrategy(CheckpointingStrategy):
    """Checkpointing facet for SD 1.5/2.0 training strategies."""

    def update_metadata(self, metadata: dict, cfg: Any) -> None:
        """SD currently adds no extra metadata fields."""
        return None

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
