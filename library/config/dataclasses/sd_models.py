from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SDModelsConfig:
    v2: bool = field(default=False, metadata={"help": "load Stable Diffusion v2.0 model"})
    v_parameterization: bool = field(default=False, metadata={"help": "enable v-parameterization training"})
    pretrained_model_name_or_path: Optional[str] = field(default=None, metadata={"help": "pretrained model to train"})
    tokenizer_cache_dir: Optional[str] = field(default=None, metadata={"help": "directory for caching Tokenizer"})
