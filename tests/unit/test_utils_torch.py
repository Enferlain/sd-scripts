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
from library.config.dataclasses.saving import SavingConfig


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
        cfg = PerformanceConfig(full_fp16=True)
        weight_dtype = torch.float16
        
        result = match_mixed_precision(cfg, weight_dtype)
        
        assert result == torch.float16
        
    def test_full_bf16_with_matching_dtype(self):
        """Test full_bf16=True returns weight_dtype when matched."""
        cfg = PerformanceConfig(full_bf16=True)
        weight_dtype = torch.bfloat16
        
        result = match_mixed_precision(cfg, weight_dtype)
        
        assert result == torch.bfloat16
        
    def test_full_fp16_requires_fp16_dtype(self):
        """Test that full_fp16 raises if weight_dtype is not float16."""
        cfg = PerformanceConfig(full_fp16=True)
        weight_dtype = torch.float32
        
        with pytest.raises(AssertionError):
            match_mixed_precision(cfg, weight_dtype)
            
    def test_full_bf16_requires_bf16_dtype(self):
        """Test that full_bf16 raises if weight_dtype is not bfloat16."""
        cfg = PerformanceConfig(full_bf16=True)
        weight_dtype = torch.float32
        
        with pytest.raises(AssertionError):
            match_mixed_precision(cfg, weight_dtype)
            
    def test_no_full_precision_returns_none(self):
        """Test that without full_fp16/bf16, returns None."""
        cfg = PerformanceConfig(full_fp16=False, full_bf16=False)
        weight_dtype = torch.float16
        
        result = match_mixed_precision(cfg, weight_dtype)
        
        assert result is None
