from dataclasses import dataclass, field
from typing import Optional, List, Union, Any

@dataclass
class NetworkConfig:
    network_dim: Optional[int] = field(default=None, metadata={"help": "network dimensions (depends on each network)"})
    network_alpha: float = field(default=1.0, metadata={"help": "alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version)"})
    network_dropout: Optional[float] = field(default=None, metadata={"help": "Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons)"})
    network_weights: Optional[str] = field(default=None, metadata={"help": "pretrained weights for network"})
    network_module: Optional[str] = field(default=None, metadata={"help": "network module to train"})
    network_args: Optional[List[str]] = field(default=None, metadata={"help": "additional arguments for network (key=value)"})
    network_train_unet_only: bool = field(default=False, metadata={"help": "only training U-Net part"})
    network_train_text_encoder_only: bool = field(default=False, metadata={"help": "only training Text Encoder part"})
    dim_from_weights: bool = field(default=False, metadata={"help": "automatically determine dim (rank) from network_weights"})
    scale_weight_norms: Optional[float] = field(default=None, metadata={"help": "Scale the weight of each key pair to help prevent overtraing via exploding gradients. (1 is a good starting point)"})
    base_weights: Optional[List[str]] = field(default=None, metadata={"help": "network weights to merge into the model before training"})
    base_weights_multiplier: Optional[List[float]] = field(default=None, metadata={"help": "multiplier for network weights to merge into the model before training"})
    training_comment: Optional[str] = field(default=None, metadata={"help": "arbitrary comment string stored in metadata"})
    unet_lr: Optional[float] = field(default=None, metadata={"help": "learning rate for U-Net"})
    # NOTE: Type is Any due to OmegaConf limitation (Union of primitives and containers not supported).
    # Semantically this is Optional[Union[float, List[float]]] - single LR or per-TE LRs.
    # Future improvement: Unify all TE LR configs (text_encoder_lr, learning_rate_te1/te2) into a single
    # list-based field that works for models with 1, 2, or more text encoders.
    text_encoder_lr: Optional[Any] = field(default=None, metadata={"help": "learning rate for Text Encoder(s). Can be float or list of floats for multiple TEs."})
    orthograd_targets: Optional[List[str]] = field(default=None, metadata={"help": "A list of strings to determine which named parameters should subject to orthgrad"})

