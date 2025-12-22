from dataclasses import dataclass, field
from .model import ModelConfig
from .buckets import BucketsConfig
from .training import TrainingConfig
from .optimizer import OptimizerConfig
from .dataset import DatasetConfig
from .network import NetworkConfig
from .sdxl import SDXLConfig
from .saving import SavingConfig
from .logging import LoggingConfig
from .performance import PerformanceConfig
from .loss import LossConfig
from .regularization import RegularizationConfig
from .timestep import TimestepConfig
from .sampling import SamplingConfig
from .masked_loss import MaskedLossConfig
from .metadata import MetadataConfig
from .huggingface import HuggingFaceConfig


@dataclass
class SDXLPeftConfig:
    """Root configuration for SDXL PEFT/LoRA training.
    
    This is the main config used by scripts/sdxl_peft.py.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    buckets: BucketsConfig = field(default_factory=BucketsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    sdxl: SDXLConfig = field(default_factory=SDXLConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    saving: SavingConfig = field(default_factory=SavingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    timestep: TimestepConfig = field(default_factory=TimestepConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    masked_loss: MaskedLossConfig = field(default_factory=MaskedLossConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
