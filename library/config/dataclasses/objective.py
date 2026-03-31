from dataclasses import dataclass, field


@dataclass
class ObjectiveConfig:
    """Objective/runtime configuration."""

    path: str = field(
        default="ddpm",
        metadata={"help": "Training-state construction path: ddpm or rectified_flow"},
    )
    prediction: str = field(
        default="epsilon",
        metadata={"help": "Prediction target convention: epsilon, v_prediction, or flow"},
    )
