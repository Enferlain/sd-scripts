"""
Unit tests for library/data/caching.py

Tests disk cache validation and text encoder output file I/O functions.
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from library.data.caching import (
    is_disk_cached_latents_is_expected,
    save_text_encoder_outputs_to_disk,
    load_text_encoder_outputs_from_disk,
)


# =============================================================================
# is_disk_cached_latents_is_expected Tests
# =============================================================================

@pytest.mark.unit
class TestIsDiskCachedLatentsIsExpected:
    """Test latent cache validation logic."""
    
    def test_returns_false_if_file_not_exists(self, tmp_path):
        """Non-existent file should return False."""
        npz_path = str(tmp_path / "nonexistent.npz")
        
        result = is_disk_cached_latents_is_expected(
            reso=(512, 512), npz_path=npz_path, flip_aug=False, alpha_mask=False
        )
        
        assert result is False
    
    def test_returns_false_if_missing_required_keys(self, tmp_path):
        """File without required keys should return False."""
        npz_path = tmp_path / "incomplete.npz"
        np.savez(npz_path, some_other_key=np.zeros((1, 64, 64)))
        
        result = is_disk_cached_latents_is_expected(
            reso=(512, 512), npz_path=str(npz_path), flip_aug=False, alpha_mask=False
        )
        
        assert result is False
    
    def test_returns_true_for_valid_cache(self, tmp_path):
        """Valid cache with correct shape should return True."""
        npz_path = tmp_path / "valid.npz"
        # reso is WxH, latents are HxW // 8
        # reso=(512, 512) -> expected_latents_size=(64, 64)
        latents = np.zeros((4, 64, 64))  # C, H, W
        np.savez(
            npz_path,
            latents=latents,
            original_size=np.array([512, 512]),
            crop_ltrb=np.array([0, 0, 0, 0]),
        )
        
        result = is_disk_cached_latents_is_expected(
            reso=(512, 512), npz_path=str(npz_path), flip_aug=False, alpha_mask=False
        )
        
        assert result is True
    
    def test_returns_false_for_wrong_latent_size(self, tmp_path):
        """Wrong latent dimensions should return False."""
        npz_path = tmp_path / "wrong_size.npz"
        # reso=(512, 512) expects latents of shape (C, 64, 64), but we provide (C, 32, 32)
        latents = np.zeros((4, 32, 32))
        np.savez(
            npz_path,
            latents=latents,
            original_size=np.array([256, 256]),
            crop_ltrb=np.array([0, 0, 0, 0]),
        )
        
        result = is_disk_cached_latents_is_expected(
            reso=(512, 512), npz_path=str(npz_path), flip_aug=False, alpha_mask=False
        )
        
        assert result is False
    
    def test_flip_aug_requires_flipped_latents(self, tmp_path):
        """With flip_aug=True, must have latents_flipped key."""
        npz_path = tmp_path / "no_flip.npz"
        latents = np.zeros((4, 64, 64))
        np.savez(
            npz_path,
            latents=latents,
            original_size=np.array([512, 512]),
            crop_ltrb=np.array([0, 0, 0, 0]),
        )
        
        result = is_disk_cached_latents_is_expected(
            reso=(512, 512), npz_path=str(npz_path), flip_aug=True, alpha_mask=False
        )
        
        assert result is False
    
    def test_flip_aug_valid_with_flipped_latents(self, tmp_path):
        """With flip_aug=True and latents_flipped present, should return True."""
        npz_path = tmp_path / "with_flip.npz"
        latents = np.zeros((4, 64, 64))
        latents_flipped = np.zeros((4, 64, 64))
        np.savez(
            npz_path,
            latents=latents,
            latents_flipped=latents_flipped,
            original_size=np.array([512, 512]),
            crop_ltrb=np.array([0, 0, 0, 0]),
        )
        
        result = is_disk_cached_latents_is_expected(
            reso=(512, 512), npz_path=str(npz_path), flip_aug=True, alpha_mask=False
        )
        
        assert result is True
    
    def test_alpha_mask_required_when_flag_true(self, tmp_path):
        """With alpha_mask=True, must have alpha_mask key."""
        npz_path = tmp_path / "no_alpha.npz"
        latents = np.zeros((4, 64, 64))
        np.savez(
            npz_path,
            latents=latents,
            original_size=np.array([512, 512]),
            crop_ltrb=np.array([0, 0, 0, 0]),
        )
        
        result = is_disk_cached_latents_is_expected(
            reso=(512, 512), npz_path=str(npz_path), flip_aug=False, alpha_mask=True
        )
        
        assert result is False
    
    def test_alpha_mask_invalid_when_flag_false_but_present(self, tmp_path):
        """With alpha_mask=False but alpha_mask present, should return False."""
        npz_path = tmp_path / "unexpected_alpha.npz"
        latents = np.zeros((4, 64, 64))
        alpha = np.zeros((512, 512))  # HxW
        np.savez(
            npz_path,
            latents=latents,
            original_size=np.array([512, 512]),
            crop_ltrb=np.array([0, 0, 0, 0]),
            alpha_mask=alpha,
        )
        
        result = is_disk_cached_latents_is_expected(
            reso=(512, 512), npz_path=str(npz_path), flip_aug=False, alpha_mask=False
        )
        
        assert result is False


# =============================================================================
# save/load_text_encoder_outputs Tests
# =============================================================================

@pytest.mark.unit
class TestTextEncoderOutputsIO:
    """Test text encoder output save/load functions."""
    
    def test_save_and_load_roundtrip(self, tmp_path):
        """Saved outputs should be loadable."""
        npz_path = tmp_path / "te_outputs.npz"
        
        hidden_state1 = torch.randn(2, 77, 768)
        hidden_state2 = torch.randn(2, 77, 1280)
        pool2 = torch.randn(2, 1280)
        
        save_text_encoder_outputs_to_disk(str(npz_path), hidden_state1, hidden_state2, pool2)
        
        loaded_hs1, loaded_hs2, loaded_pool2 = load_text_encoder_outputs_from_disk(str(npz_path))
        
        assert torch.allclose(hidden_state1, loaded_hs1, atol=1e-5)
        assert torch.allclose(hidden_state2, loaded_hs2, atol=1e-5)
        assert torch.allclose(pool2, loaded_pool2, atol=1e-5)
    
    def test_load_returns_tensors(self, tmp_path):
        """Loaded outputs should be PyTorch tensors."""
        npz_path = tmp_path / "te_outputs.npz"
        
        hidden_state1 = torch.ones(1, 10, 64)
        hidden_state2 = torch.ones(1, 10, 128)
        pool2 = torch.ones(1, 128)
        
        save_text_encoder_outputs_to_disk(str(npz_path), hidden_state1, hidden_state2, pool2)
        
        loaded_hs1, loaded_hs2, loaded_pool2 = load_text_encoder_outputs_from_disk(str(npz_path))
        
        assert isinstance(loaded_hs1, torch.Tensor)
        assert isinstance(loaded_hs2, torch.Tensor)
        assert isinstance(loaded_pool2, torch.Tensor)
    
    def test_load_handles_missing_hidden_state2(self, tmp_path):
        """Load should return None for missing hidden_state2."""
        npz_path = tmp_path / "te_outputs_partial.npz"
        
        # Manually create npz with only hidden_state1
        np.savez(
            npz_path,
            hidden_state1=np.ones((1, 10, 64), dtype=np.float32),
        )
        
        loaded_hs1, loaded_hs2, loaded_pool2 = load_text_encoder_outputs_from_disk(str(npz_path))
        
        assert loaded_hs1 is not None
        assert loaded_hs2 is None
        assert loaded_pool2 is None
    
    def test_save_converts_to_float32(self, tmp_path):
        """Save should convert to float32 for storage efficiency."""
        npz_path = tmp_path / "te_outputs.npz"
        
        # Create float16 tensors
        hidden_state1 = torch.ones(1, 10, 64, dtype=torch.float16)
        hidden_state2 = torch.ones(1, 10, 128, dtype=torch.float16)
        pool2 = torch.ones(1, 128, dtype=torch.float16)
        
        save_text_encoder_outputs_to_disk(str(npz_path), hidden_state1, hidden_state2, pool2)
        
        # Check saved dtype
        with np.load(npz_path) as f:
            assert f["hidden_state1"].dtype == np.float32
            assert f["hidden_state2"].dtype == np.float32
            assert f["pool2"].dtype == np.float32
