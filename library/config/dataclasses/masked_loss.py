from dataclasses import dataclass, field
from typing import Optional

@dataclass
class MaskedLossConfig:
    conditioning_data_dir: Optional[str] = field(default=None, metadata={"help": "conditioning data directory"})
    masked_loss: bool = field(default=False, metadata={"help": "apply mask for calculating loss"})
