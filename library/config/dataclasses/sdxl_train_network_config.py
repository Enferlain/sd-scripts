from dataclasses import dataclass, field
from library.config.dataclasses.sd_models import ModelLoadingConfig
from library.config.dataclasses.training import TrainingConfig
from library.config.dataclasses.optimizer import OptimizerConfig
from library.config.dataclasses.dataset import DatasetConfig
from library.config.dataclasses.network import NetworkConfig
from library.config.dataclasses.sdxl_training import SDXLTrainingConfig
from library.config.dataclasses.saving import SavingConfig
from library.config.dataclasses.logging import LoggingConfig
from library.config.dataclasses.performance import PerformanceConfig
from library.config.dataclasses.loss import LossConfig
from library.config.dataclasses.regularization import RegularizationConfig
from library.config.dataclasses.timestep import TimestepConfig
from library.config.dataclasses.sampling import SamplingConfig
from library.config.dataclasses.masked_loss import MaskedLossConfig
from library.config.dataclasses.metadata import MetadataConfig
from library.config.dataclasses.huggingface import HuggingFaceConfig

@dataclass
class SDXLTrainNetworkConfig:
    sd_models: ModelLoadingConfig = field(default_factory=ModelLoadingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    sdxl_training: SDXLTrainingConfig = field(default_factory=SDXLTrainingConfig)
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
