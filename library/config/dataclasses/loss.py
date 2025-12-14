from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LossConfig:
    loss_type: str = "l2"
    huber_schedule: str = "snr"
    huber_c: float = 0.1
    huber_scale: float = 1.0
    loss_scale: float = 1.0
    prior_loss_weight: float = 1.0
    loss_multiplier: Optional[float] = None
    min_snr_gamma: Optional[float] = None
    scale_v_pred_loss_like_noise_pred: bool = False
    v_pred_like_loss: Optional[float] = None
    debiased_estimation_loss: bool = False
