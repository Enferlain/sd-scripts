from dataclasses import dataclass, field
from typing import Optional
from .training import TrainingConfig
from .optimizer import OptimizerConfig
from .dataset import DatasetConfig
from .buckets import BucketsConfig
from .sd_models import SDModelsConfig
from .saving import SavingConfig
from .huggingface import HuggingFaceConfig
from .logging import LoggingConfig
from .performance import PerformanceConfig
from .loss import LossConfig
from .regularization import RegularizationConfig
from .timestep import TimestepConfig
from .sampling import SamplingConfig
from .masked_loss import MaskedLossConfig
from .metadata import MetadataConfig

@dataclass
class FineTuneSpecificConfig:
    """Fine-tuning specific configuration."""
    diffusers_xformers: bool = False
    train_text_encoder: bool = False
    learning_rate_te: Optional[float] = None

@dataclass
class FineTuneConfig:
    """Root configuration for fine-tuning training."""
    fine_tune: FineTuneSpecificConfig = field(default_factory=FineTuneSpecificConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    buckets: BucketsConfig = field(default_factory=BucketsConfig)
    sd_models: SDModelsConfig = field(default_factory=SDModelsConfig)
    saving: SavingConfig = field(default_factory=SavingConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    timestep: TimestepConfig = field(default_factory=TimestepConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    masked_loss: MaskedLossConfig = field(default_factory=MaskedLossConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
