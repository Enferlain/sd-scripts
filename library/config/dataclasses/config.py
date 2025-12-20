# This module provides backwards-compatible imports for config classes.
# New code should import directly from the specific module files.

# Re-export SDXLFineTuningConfig from its new canonical location
from .sdxl_train import SDXLFineTuningConfig, SDXLTrainConfig

__all__ = ['SDXLFineTuningConfig', 'SDXLTrainConfig']
