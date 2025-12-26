from dataclasses import dataclass, field
from .model import ModelConfig
from .buckets import BucketsConfig
from .training import TrainingConfig
from .optimizer import OptimizerConfig
from .dataset import DatasetConfig
from .peft import PeftConfig
from .sdxl import SDXLConfig
from .performance import PerformanceConfig
from .loss import LossConfig
from .regularization import RegularizationConfig
from .timestep import TimestepConfig
from .masked_loss import MaskedLossConfig
from .output import OutputConfig


@dataclass
class SDXLPeftConfig:
    """Root configuration for SDXL PEFT/LoRA training.
    
    This is the main config used by scripts/sdxl_peft.py.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    buckets: BucketsConfig = field(default_factory=BucketsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    peft: PeftConfig = field(default_factory=PeftConfig)
    sdxl: SDXLConfig = field(default_factory=SDXLConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    timestep: TimestepConfig = field(default_factory=TimestepConfig)
    masked_loss: MaskedLossConfig = field(default_factory=MaskedLossConfig)
