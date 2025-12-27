from dataclasses import dataclass, field
from .training import TrainingConfig
from .optimizer import OptimizerConfig
from .dataset import DatasetConfig
from .model import ModelConfig
from .sdxl import SDXLConfig
from .performance import PerformanceConfig
from .loss import LossConfig
from .timestep import TimestepConfig
from .buckets import BucketsConfig
from .output import OutputConfig
from .validation import ValidationConfig


@dataclass
class SDXLFineTuneConfig:
    """Root configuration for SDXL fine-tuning training.
    
    This is the main config used by scripts/sdxl_finetune.py.
    """
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sdxl: SDXLConfig = field(default_factory=SDXLConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    timestep: TimestepConfig = field(default_factory=TimestepConfig)
    buckets: BucketsConfig = field(default_factory=BucketsConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
