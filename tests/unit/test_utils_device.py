"""
Unit tests for library/utils/device_utils.py

Tests for device utility functions using strict mocking to avoid hardware dependencies.
"""

import sys
import torch
import pytest
from unittest.mock import patch, MagicMock

from library.utils.device_utils import (
    clean_memory,
    clean_memory_on_device,
    synchronize_device,
    get_preferred_device,
    init_ipex,
)


@pytest.fixture
def mock_torch():
    """Patches torch in device_utils to a MagicMock."""
    with patch("library.utils.device_utils.torch") as mock:
        yield mock


# -------------------------
# clean_memory
# -------------------------

def test_clean_memory_calls_gc_and_backends(mock_torch):
    with patch("library.utils.device_utils.gc.collect") as mock_gc, \
         patch("library.utils.device_utils.HAS_CUDA", True), \
         patch("library.utils.device_utils.HAS_XPU", True), \
         patch("library.utils.device_utils.HAS_MPS", True):

        clean_memory()

        mock_gc.assert_called_once()
        mock_torch.cuda.empty_cache.assert_called_once()
        mock_torch.xpu.empty_cache.assert_called_once()
        mock_torch.mps.empty_cache.assert_called_once()


# -------------------------
# clean_memory_on_device
# -------------------------

def test_clean_memory_on_device_none(mock_torch):
    with patch("library.utils.device_utils.gc.collect") as mock_gc:
        clean_memory_on_device(None)
        mock_gc.assert_called_once()

def test_clean_memory_on_device_cuda_str(mock_torch):
    # Setup mock device type
    mock_torch.device.return_value.type = "cuda"
    
    with patch("library.utils.device_utils.gc.collect"):
        clean_memory_on_device("cuda")
        mock_torch.cuda.empty_cache.assert_called_once()

def test_clean_memory_on_device_cuda_device(mock_torch):
    # Pass a specific mock as device
    dev_mock = MagicMock()
    dev_mock.type = "cuda"
    
    with patch("library.utils.device_utils.gc.collect"):
        clean_memory_on_device(dev_mock)
        mock_torch.cuda.empty_cache.assert_called_once()

def test_clean_memory_on_device_xpu_and_mps(mock_torch):
    with patch("library.utils.device_utils.gc.collect"):
        
        # Test XPU
        mock_torch.device.return_value.type = "xpu"
        clean_memory_on_device("xpu")
        mock_torch.xpu.empty_cache.assert_called_once()

        # Test MPS
        mock_torch.device.return_value.type = "mps"
        clean_memory_on_device("mps")
        mock_torch.mps.empty_cache.assert_called_once()


# -------------------------
# synchronize_device
# -------------------------

def test_synchronize_device_branches(mock_torch):
    # Test CUDA
    mock_torch.device.return_value.type = "cuda"
    synchronize_device("cuda")
    mock_torch.cuda.synchronize.assert_called_once()

    # Test XPU
    mock_torch.device.return_value.type = "xpu"
    synchronize_device("xpu")
    mock_torch.xpu.synchronize.assert_called_once()

    # Test MPS
    mock_torch.device.return_value.type = "mps"
    synchronize_device("mps")
    mock_torch.mps.synchronize.assert_called_once()

    # None should do nothing
    synchronize_device(None)
    # Verify no extra calls (counts should still be 1)
    mock_torch.cuda.synchronize.assert_called_once()


# -------------------------
# get_preferred_device
# -------------------------

def test_get_preferred_device_priority(mock_torch):
    """
    Test priority: CUDA > XPU > MPS > CPU
    Using explicit device type return values for the mock.
    """
    
    # Helper to reset cache and mock return
    def check_priority(has_cuda, has_xpu, has_mps, expected_type):
        get_preferred_device.cache_clear()
        
        # Configure what torch.device("stuff") returns
        # We need it to return an object with .type = expected_type
        # But wait, logic is: if HAS_CUDA: device = torch.device("cuda")
        # So we must ensure torch.device("cuda") returns something with .type="cuda"
        
        def device_side_effect(name):
            m = MagicMock()
            m.type = name
            # Special case for logging in the function
            m.__str__.return_value = name 
            return m
        
        mock_torch.device.side_effect = device_side_effect

        with patch("library.utils.device_utils.HAS_CUDA", has_cuda), \
             patch("library.utils.device_utils.HAS_XPU", has_xpu), \
             patch("library.utils.device_utils.HAS_MPS", has_mps):
            
            dev = get_preferred_device()
            # We assume the implementation passes the correct string
            # verify the calls were made with expected string if needed, 
            # but checking returned mock type is enough
            assert dev.type == expected_type

    # 1. CUDA preferred
    check_priority(True, False, False, "cuda")
    
    # 2. XPU preferred
    check_priority(False, True, False, "xpu")
    
    # 3. MPS preferred
    check_priority(False, False, True, "mps")
    
    # 4. CPU fallback
    check_priority(False, False, False, "cpu")


def test_get_preferred_device_is_cached(mock_torch):
    get_preferred_device.cache_clear()
    
    with patch("library.utils.device_utils.HAS_CUDA", False), \
         patch("library.utils.device_utils.HAS_XPU", False), \
         patch("library.utils.device_utils.HAS_MPS", False):

        d1 = get_preferred_device()
        d2 = get_preferred_device()
        
        assert d1 is d2
        # Should be called once for d1, d2 is from cache
        assert mock_torch.device.call_count == 1
        mock_torch.device.assert_called_with("cpu")


# -------------------------
# init_ipex
# -------------------------

def test_init_ipex_calls_ipex_init_when_xpu():
    # Use patch.dict to inject a mock module for library.performance.ipex
    mock_ipex_module = MagicMock()
    mock_ipex_module.ipex_init.return_value = (True, "")
    
    with patch.dict(sys.modules, {"library.performance.ipex": mock_ipex_module}):
        with patch("library.utils.device_utils.HAS_XPU", True):
            init_ipex()
            
    mock_ipex_module.ipex_init.assert_called_once()

def test_init_ipex_handles_failure(capsys):
    mock_ipex_module = MagicMock()
    mock_ipex_module.ipex_init.return_value = (False, "some_error")
    
    with patch.dict(sys.modules, {"library.performance.ipex": mock_ipex_module}):
        with patch("library.utils.device_utils.HAS_XPU", True):
            init_ipex()
    
    captured = capsys.readouterr()
    assert "failed to initialize ipex: some_error" in captured.out

def test_init_ipex_no_xpu():
    # Even if module exists, it shouldn't be touched if HAS_XPU is False
    mock_ipex_module = MagicMock()
    
    with patch.dict(sys.modules, {"library.performance.ipex": mock_ipex_module}):
        with patch("library.utils.device_utils.HAS_XPU", False):
            init_ipex()
            
    mock_ipex_module.ipex_init.assert_not_called()

def test_init_ipex_catches_exceptions(capsys):
    mock_ipex_module = MagicMock()
    mock_ipex_module.ipex_init.side_effect = RuntimeError("boom")
    
    with patch.dict(sys.modules, {"library.performance.ipex": mock_ipex_module}):
        with patch("library.utils.device_utils.HAS_XPU", True):
            init_ipex()

    captured = capsys.readouterr()
    assert "failed to initialize ipex:" in captured.out
