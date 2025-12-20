from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelLoadingConfig:
    pretrained_model_name_or_path: Optional[str] = field(default=None, metadata={"help": "pretrained model to train"})
    tokenizer_cache_dir: Optional[str] = field(default=None, metadata={"help": "directory for caching Tokenizer"})
    vae: Optional[str] = field(default=None, metadata={"help": "path to VAE model to use"})
    vae_conv2d_padding_mode: str = field(default="zeros", metadata={"help": "Adjusts the padding for Conv2d modules in the VAE"})

@dataclass
class SDModelsConfig(ModelLoadingConfig):
    v2: bool = field(default=False, metadata={"help": "load Stable Diffusion v2.0 model"})
