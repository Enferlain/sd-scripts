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

    # EDM2
    edm2_loss_weighting: bool = field(default=False, metadata={"help": "Use EDM2 loss weighting."})
    edm2_loss_weighting_laplace: bool = field(default=False, metadata={"help": "Use EDM2 loss weighting to calculate timestep sampling using laplace."})
    edm2_loss_weighting_optimizer: str = field(default="torch.optim.AdamW", metadata={"help": "Fully qualified optimizer class name to use with the edm2 loss weighting optimizer."})
    edm2_loss_weighting_optimizer_lr: float = field(default=2e-2, metadata={"help": "Learning rate as a float for the edm2 loss weighting optimizer."})
    edm2_loss_weighting_optimizer_args: str = field(default="{'weight_decay': 0, 'betas': (0.9,0.999)}", metadata={"help": "A JSON object as a string of optimizer args for the edm2 loss weighting optimizer."})
    edm2_loss_weighting_lr_scheduler: bool = field(default=False, metadata={"help": "Use lr scheduler with EDM2 loss weighting optimizer."})
    edm2_loss_weighting_lr_scheduler_warmup_percent: float = field(default=0.1, metadata={"help": "Percent of training steps to use for warmup."})
    edm2_loss_weighting_lr_scheduler_constant_percent: float = field(default=0.1, metadata={"help": "Percent of training steps to maintain constant LR before decay."})
    edm2_loss_weighting_generate_graph: bool = field(default=False, metadata={"help": "Enable generation of graph images that show the loss weighting per timestep."})
    edm2_loss_weighting_generate_graph_every_x_steps: int = field(default=20, metadata={"help": "Every x steps generate a graph image."})
    edm2_loss_weighting_generate_graph_output_dir: Optional[str] = field(default=None, metadata={"help": "The parent directory where loss weighting graph images should be stored"})
    edm2_loss_weighting_generate_graph_y_limit: Optional[int] = field(default=None, metadata={"help": "Set the max limit of the y axis"})
    edm2_loss_weighting_generate_graph_y_scale: str = field(default="linear", metadata={"help": "Select between linear or log scaling for the y-axis."})
    edm2_loss_weighting_num_channels: int = field(default=128, metadata={"help": "The number of channels used by for the loss weighting module."})
    edm2_loss_weighting_initial_weights: Optional[str] = field(default=None, metadata={"help": "The full filepath to initial weights and state of edm2 weighting model to use instead of random."})
    edm2_loss_weighting_lr_scheduler_decay_scaling: float = field(default=1.0, metadata={"help": "A scaling factor to apply to the decay rate of the edm2_loss_weighting_lr_scheduler"})
    edm2_loss_weighting_importance_weighting: bool = field(default=False, metadata={"help": "If edm2 loss scaling weights are weighted by importance"})
    edm2_loss_weighting_importance_weighting_max: float = field(default=10.0, metadata={"help": "The max loss weighting/scaling to apply when using edm2 importance weighting"})
    edm2_loss_weighting_importance_min_snr_gamma: float = field(default=1.0, metadata={"help": "The min snr gamma used for edm2 importance weighting as a heuristic"})
    edm2_loss_weighting_importance_weighting_safety_override: bool = field(default=False, metadata={"help": "At your own risk, you may set this to true to ALLOW stacking debiased loss and/or typical min snr gamma with EDM2 using importance weighting."})
