from dataclasses import dataclass, field
from .saving import SavingConfig
from .logging import LoggingConfig
from .huggingface import HuggingFaceConfig
from .sampling import SamplingConfig
from .metadata import MetadataConfig


@dataclass
class OutputConfig:
    """Output configuration with organized subcategories for saving, logging, and publishing."""
    saving: SavingConfig = field(default_factory=SavingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
