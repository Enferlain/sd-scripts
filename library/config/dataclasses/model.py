from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """
    Model configuration.
    
    model_type values: sd15, sd2, sdxl, flux
    """
    model_type: str = field(default="sdxl", metadata={"help": "Model architecture type: sd15, sd2, sdxl, flux"})
    pretrained_model_name_or_path: Optional[str] = field(default=None, metadata={"help": "pretrained model to train"})
    tokenizer_cache_dir: Optional[str] = field(default=None, metadata={"help": "directory for caching Tokenizer"})
    vae: Optional[str] = field(default=None, metadata={"help": "path to VAE model to use"})
    vae_conv2d_padding_mode: str = field(default="zeros", metadata={"help": "Adjusts the padding for Conv2d modules in the VAE"})
