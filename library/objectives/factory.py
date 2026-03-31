from __future__ import annotations

from typing import Any

from library.objectives.base import ObjectiveDefinition
from library.objectives.ddpm import DDPMObjective
from library.objectives.rectified_flow import RectifiedFlowObjective


def build_objective(cfg: Any) -> ObjectiveDefinition:
    """Resolve the active objective owner from the current config."""
    objective_path = getattr(getattr(cfg, "objective", None), "path", None)
    if objective_path == "rectified_flow":
        return RectifiedFlowObjective()
    if objective_path == "ddpm":
        return DDPMObjective()
    raise ValueError(f"Unsupported objective.path={objective_path!r}")
