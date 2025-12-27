from dataclasses import dataclass, field
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
class SDFineTuneConfig:
    """Root configuration for SD fine-tuning training.
    
    Note: train_text_encoder and learning_rate_te have been removed.
    Text encoder training is now controlled via optimizer.learning_rates.text_encoders
    (LR-based control per Schema 1).
    """
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    timestep: TimestepConfig = field(default_factory=TimestepConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
