from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class ModelConfig:
    """
    Model configuration.
    """

    model_type: str | None = field(default=MISSING, metadata={"help": "Model architecture type: sd15, sd2, sdxl, sd3, flux"})
    pretrained_model_name_or_path: str | None = field(default=None, metadata={"help": "pretrained model to train"})
    vae: str | None = field(default=None, metadata={"help": "path to VAE model to use"})
    vae_conv2d_padding_mode: str = field(default="zeros", metadata={"help": "Adjusts the padding for Conv2d modules in the VAE"})
