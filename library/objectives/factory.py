from __future__ import annotations

from typing import Any

from library.objectives.base import ObjectiveDefinition
from library.objectives.ddpm import DDPMObjective
from library.objectives.rectified_flow import RectifiedFlowObjective


def build_objective(cfg: Any) -> ObjectiveDefinition:
    """Resolve the active objective owner from the current config.

    The repo does not yet expose a dedicated objective config root, so the
    current resolution still infers the objective from the active model family.
    """
    if cfg.model.model_type == "sd3":
        return RectifiedFlowObjective()
    return DDPMObjective()
