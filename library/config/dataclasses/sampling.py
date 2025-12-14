from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class SamplingConfig:
    sample_every_n_steps: Optional[int] = None
    sample_at_first: bool = False
    sample_every_n_epochs: Optional[int] = None
    sample_prompts: Optional[List[str]] = None
    sample_sampler: str = "ddim"

    def __post_init__(self):
        if self.sample_every_n_epochs is not None and self.sample_every_n_epochs <= 0:
            self.sample_every_n_epochs = None

        if self.sample_every_n_steps is not None and self.sample_every_n_steps <= 0:
            self.sample_every_n_steps = None
