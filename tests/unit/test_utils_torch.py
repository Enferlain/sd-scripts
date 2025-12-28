"""
Unit tests for library/utils/torch_utils.py

Tests PyTorch utility functions for dtype handling and precision.
"""

import pytest
import torch

from library.utils.torch_utils import (
    prepare_dtype,
    match_mixed_precision,
)
from library.config.dataclasses.performance import PerformanceConfig
from library.config.dataclasses.output import SavingConfig
from unittest.mock import patch, MagicMock
import random
import numpy as np


# =============================================================================
# prepare_dtype Tests
# =============================================================================

@pytest.mark.training
@pytest.mark.unit
class TestPrepareDtype:
    """Test prepare_dtype function with proper config types."""
    
    def test_default_returns_fp32(self):
        """Test that default mixed_precision='no' returns float32."""
        perf_cfg = PerformanceConfig()
        
        weight_dtype, save_dtype = prepare_dtype(perf_cfg)
        
        assert weight_dtype == torch.float32
        assert save_dtype is None
        
    def test_fp16_mixed_precision(self):
        """Test fp16 mixed precision returns float16 weight dtype."""
        perf_cfg = PerformanceConfig(mixed_precision="fp16")
        
        weight_dtype, _ = prepare_dtype(perf_cfg)
        
        assert weight_dtype == torch.float16
        
    def test_bf16_mixed_precision(self):
        """Test bf16 mixed precision returns bfloat16 weight dtype."""
        perf_cfg = PerformanceConfig(mixed_precision="bf16")
        
        weight_dtype, _ = prepare_dtype(perf_cfg)
        
        assert weight_dtype == torch.bfloat16
        
    def test_save_precision_fp16(self):
        """Test save_precision='fp16' returns float16 save dtype."""
        perf_cfg = PerformanceConfig()
        save_cfg = SavingConfig(save_precision="fp16")
        
        _, save_dtype = prepare_dtype(perf_cfg, save_cfg)
        
        assert save_dtype == torch.float16
        
    def test_save_precision_bf16(self):
        """Test save_precision='bf16' returns bfloat16 save dtype."""
        perf_cfg = PerformanceConfig()
        save_cfg = SavingConfig(save_precision="bf16")
        
        _, save_dtype = prepare_dtype(perf_cfg, save_cfg)
        
        assert save_dtype == torch.bfloat16
        
    def test_save_precision_float(self):
        """Test save_precision='float' returns float32 save dtype."""
        perf_cfg = PerformanceConfig()
        save_cfg = SavingConfig(save_precision="float")
        
        _, save_dtype = prepare_dtype(perf_cfg, save_cfg)
        
        assert save_dtype == torch.float32
        
    def test_no_saving_config_returns_none_save_dtype(self):
        """Test that without saving_config, save_dtype is None."""
        perf_cfg = PerformanceConfig(mixed_precision="fp16")
        
        weight_dtype, save_dtype = prepare_dtype(perf_cfg)
        
        assert weight_dtype == torch.float16
        assert save_dtype is None


# =============================================================================
# match_mixed_precision Tests
# =============================================================================

@pytest.mark.training
@pytest.mark.unit
class TestMatchMixedPrecision:
    """Test match_mixed_precision function."""
    
    def test_full_fp16_with_matching_dtype(self):
        """Test full_fp16=True returns weight_dtype when matched."""
        from library.config.dataclasses.performance import PrecisionConfig
        precision_cfg = PrecisionConfig(full_fp16=True)
        weight_dtype = torch.float16
        
        result = match_mixed_precision(precision_cfg, weight_dtype)
        
        assert result == torch.float16
        
    def test_full_bf16_with_matching_dtype(self):
        """Test full_bf16=True returns weight_dtype when matched."""
        from library.config.dataclasses.performance import PrecisionConfig
        precision_cfg = PrecisionConfig(full_bf16=True)
        weight_dtype = torch.bfloat16
        
        result = match_mixed_precision(precision_cfg, weight_dtype)
        
        assert result == torch.bfloat16
        
    def test_full_fp16_requires_fp16_dtype(self):
        """Test that full_fp16 raises if weight_dtype is not float16."""
        from library.config.dataclasses.performance import PrecisionConfig
        precision_cfg = PrecisionConfig(full_fp16=True)
        weight_dtype = torch.float32
        
        with pytest.raises(AssertionError):
            match_mixed_precision(precision_cfg, weight_dtype)
            
    def test_full_bf16_requires_bf16_dtype(self):
        """Test that full_bf16 raises if weight_dtype is not bfloat16."""
        from library.config.dataclasses.performance import PrecisionConfig
        precision_cfg = PrecisionConfig(full_bf16=True)
        weight_dtype = torch.float32
        
        with pytest.raises(AssertionError):
            match_mixed_precision(precision_cfg, weight_dtype)
            
    def test_no_full_precision_returns_none(self):
        """Test that without full_fp16/bf16, returns None."""
        from library.config.dataclasses.performance import PrecisionConfig
        precision_cfg = PrecisionConfig(full_fp16=False, full_bf16=False)
        weight_dtype = torch.float16
        
        result = match_mixed_precision(precision_cfg, weight_dtype)
        
        assert result is None


# =============================================================================
# set_seed_from_config Tests
# =============================================================================

from library.utils.torch_utils import set_seed_from_config

@pytest.mark.training
@pytest.mark.unit
class TestSetSeedFromConfig:
    """Test set_seed_from_config function."""
    
    @patch("library.utils.torch_utils.set_seed")
    def test_seed_is_set_when_present(self, mock_set_seed):
        """Test that set_seed is called when config has seed."""
        # Using a dummy config object with a seed attribute
        class DummyConfig:
            seed = 42
            
        config = DummyConfig()

        set_seed_from_config(config)
        
        mock_set_seed.assert_called_with(42)
        
    @patch("library.utils.torch_utils.set_seed")
    def test_no_seed_in_config_generates_random(self, mock_set_seed):
        """Test that random seed is generated if seed is None."""
        class DummyConfig:
            seed = None
            
        config = DummyConfig()

        set_seed_from_config(config)
        
        # Verify set_seed was called with SOME integer
        assert mock_set_seed.called
        call_arg = mock_set_seed.call_args[0][0]
        assert isinstance(call_arg, int)
        assert config.seed is not None  # content should be updated


