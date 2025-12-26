"""
Pytest configuration and shared fixtures for sd-scripts tests.

This module provides common fixtures and utilities for testing the library modules
after the Hydra/dataclass migration.
"""

import pytest
import tempfile
import shutil
import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, field

# Conditional torch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

# Import all config dataclasses for fixture creation
from library.config.dataclasses.optimizer import OptimizerConfig
from library.config.dataclasses.dataset import DatasetConfig
from library.config.dataclasses.training import TrainingConfig
from library.config.dataclasses.peft import PeftConfig
from library.config.dataclasses.buckets import BucketsConfig
from library.config.dataclasses.model import ModelConfig
from library.config.dataclasses.output import SavingConfig
from library.config.dataclasses.output import LoggingConfig
from library.config.dataclasses.output import MetadataConfig


# ============================================================================
# Hydra Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_hydra():
    """Automatically clean up Hydra GlobalHydra instance after each test."""
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


@pytest.fixture
def config_path():
    """Return the absolute path to the configs directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs"))


@pytest.fixture
def hydra_ctx(config_path):
    """Context manager for Hydra initialization."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=config_path):
        yield
    GlobalHydra.instance().clear()


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================

@pytest.fixture
def tmp_output_dir():
    """Create a temporary output directory for tests."""
    tmp_dir = tempfile.mkdtemp(prefix="sd_scripts_test_")
    yield tmp_dir
    # Cleanup
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def tmp_dataset_dir(tmp_output_dir):
    """Create a temporary dataset directory with basic structure."""
    dataset_dir = os.path.join(tmp_output_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Create a simple subdirectory structure
    train_dir = os.path.join(dataset_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    
    yield dataset_dir


# ============================================================================
# Mock Config Fixtures
# ============================================================================

@pytest.fixture
def mock_optimizer_config():
    """Create a basic OptimizerConfig for testing."""
    return OptimizerConfig(
        optimizer_type="AdamW",
        learning_rate=1e-4,
        lr_scheduler="constant",
        lr_warmup_steps=0,
    )


@pytest.fixture
def mock_dataset_config(tmp_dataset_dir):
    """Create a basic DatasetConfig for testing."""
    return DatasetConfig(
        train_data_dir=tmp_dataset_dir,
        resolution="512,512",
        batch_size=1,
        max_token_length=225,
    )


@pytest.fixture
def mock_training_config():
    """Create a basic TrainingConfig for testing."""
    return TrainingConfig(
        max_train_epochs=10,
        train_batch_size=1,
        mixed_precision="fp16",
        gradient_accumulation_steps=1,
    )


@pytest.fixture
def mock_adapter_config():
    """Create a basic PeftConfig for testing."""
    return PeftConfig(
        module="adapters.lora",
        dim=4,
        alpha=1.0,
    )


@pytest.fixture
def mock_buckets_config():
    """Create a basic BucketsConfig for testing."""
    return BucketsConfig(
        enable_bucket=True,
        min_bucket_reso=256,
        max_bucket_reso=1024,
        bucket_reso_steps=64,
    )


@pytest.fixture
def mock_model_config():
    """Create a basic ModelConfig for testing."""
    return ModelConfig(
        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
    )


@pytest.fixture
def mock_saving_config(tmp_output_dir):
    """Create a basic SavingConfig for testing."""
    return SavingConfig(
        output_dir=tmp_output_dir,
        output_name="test_model",
        save_model_as="safetensors",
    )


@pytest.fixture
def mock_logging_config(tmp_output_dir):
    """Create a basic LoggingConfig for testing."""
    return LoggingConfig(
        logging_dir=os.path.join(tmp_output_dir, "logs"),
        log_prefix="test_",
    )


@pytest.fixture
def mock_metadata_config():
    """Create a basic MetadataConfig for testing."""
    return MetadataConfig(
        metadata_title="Test Model",
        metadata_author="Test Author",
    )


# ============================================================================
# Mock Model/Tensor Fixtures
# ============================================================================

@pytest.fixture
def mock_model_parameters():
    """Create mock model parameters for optimizer testing."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")
    
    # Create a simple module with parameters
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Linear(10, 10)
            self.layer2 = torch.nn.Linear(10, 10)
    
    model = MockModel()
    return list(model.parameters())


@pytest.fixture
def mock_accelerator():
    """Create a mock Accelerator object for testing."""
    class MockAccelerator:
        def __init__(self):
            self.state = type('obj', (object,), {'num_processes': 1})()
            self.num_processes = 1
            
        def wait_for_everyone(self):
            pass
            
        def save_state(self, output_dir):
            pass
            
        def unwrap_model(self, model):
            return model
    
    return MockAccelerator()


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")
    
    class MockTokenizer:
        def __init__(self):
            self.model_max_length = 77
            
        def __call__(self, text, **kwargs):
            # Return mock input_ids
            return {"input_ids": torch.randint(0, 1000, (1, 77))}
    
    return MockTokenizer()


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_image_tensor():
    """Create a sample image tensor for testing."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")
    
    # Create a 512x512 RGB image tensor
    return torch.randn(3, 512, 512)


@pytest.fixture
def sample_latent_tensor():
    """Create a sample latent tensor for testing."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")
    
    # Create a 64x64 latent (typical for 512x512 image with VAE)
    return torch.randn(4, 64, 64)


@pytest.fixture
def sample_caption():
    """Return a sample caption for testing."""
    return "a beautiful landscape with mountains and a lake"


# ============================================================================
# Utility Functions
# ============================================================================

def assert_config_valid(config: Any):
    """
    Helper function to validate that a config object is properly instantiated.
    
    Args:
        config: Any dataclass config object
    """
    assert config is not None
    assert dataclass(config) is not None  # Verify it's a dataclass
