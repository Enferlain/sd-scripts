"""
Backward compatibility re-exports for DataConfig.

The new structure is in data.py. This file provides aliases for code that may still
import from dataset.py.
"""

from library.config.dataclasses.data import (
    DataConfig,
    SourceConfig,
    PreprocessingConfig,
    CaptionConfig,
    BucketingConfig,
    CachingConfig,
)

# Deprecated alias - use DataConfig instead
DatasetConfig = DataConfig

__all__ = [
    "DataConfig",
    "DatasetConfig",  # deprecated
    "SourceConfig",
    "PreprocessingConfig",
    "CaptionConfig",
    "BucketingConfig",
    "CachingConfig",
]
