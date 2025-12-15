from dataclasses import dataclass, field
from typing import Optional, List, Union

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

    # These were in train_network.py args but seem network related
    use_ramtorch: bool = field(default=False, metadata={"help": "Use RamTorch to reduce GPU memory usage by keeping model weights on CPU."})
    direct_ramtorch: bool = field(default=False, metadata={"help": "Train orig weights in lyco full module and save diff instead of keep both."})

    unet_lr: Optional[float] = field(default=None, metadata={"help": "learning rate for U-Net"})
    text_encoder_lr: Optional[Union[float, List[float]]] = field(default=None, metadata={"help": "learning rate for Text Encoder, can be multiple"})
    orthograd_targets: Optional[List[str]] = field(default=None, metadata={"help": "A list of strings to determine which named parameters should subject to orthgrad"})
