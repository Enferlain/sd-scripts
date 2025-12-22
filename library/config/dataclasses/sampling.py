from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class SamplingConfig:
    sample_every_n_steps: Optional[int] = None
    sample_at_first: bool = False
    sample_every_n_epochs: Optional[int] = None
    sample_prompts: Optional[str] = None
    sample_sampler: str = "ddim"
