from dataclasses import dataclass, field
from typing import Optional
from .training import TrainingConfig
from .optimizer import OptimizerConfig
from .data import DataConfig
from .model import ModelConfig
from .performance import PerformanceConfig
from .loss import LossConfig
from .timestep import TimestepConfig
from .output import OutputConfig
from .validation import ValidationConfig


@dataclass
class TextualInversionSpecificConfig:
    weights: Optional[str] = None
    num_vectors_per_token: int = 1
    token_string: Optional[str] = None
    init_word: Optional[str] = None
    use_object_template: bool = False
    use_style_template: bool = False


@dataclass
class TextualInversionConfig:
    textual_inversion: TextualInversionSpecificConfig = field(default_factory=TextualInversionSpecificConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    timestep: TimestepConfig = field(default_factory=TimestepConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
