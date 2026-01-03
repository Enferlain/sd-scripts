"""
Unit tests for library/data/caching.py

Tests disk cache validation and text encoder output file I/O functions.
"""

import pytest
import numpy as np
import torch

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

        result = is_disk_cached_latents_is_expected(reso=(512, 512), npz_path=npz_path, flip_aug=False, alpha_mask=False)

        assert result is False

    def test_returns_false_if_missing_required_keys(self, tmp_path):
        """File without required keys should return False."""
        npz_path = tmp_path / "incomplete.npz"
        np.savez(npz_path, some_other_key=np.zeros((1, 64, 64)))

        result = is_disk_cached_latents_is_expected(reso=(512, 512), npz_path=str(npz_path), flip_aug=False, alpha_mask=False)

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

        result = is_disk_cached_latents_is_expected(reso=(512, 512), npz_path=str(npz_path), flip_aug=False, alpha_mask=False)

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

        result = is_disk_cached_latents_is_expected(reso=(512, 512), npz_path=str(npz_path), flip_aug=False, alpha_mask=False)

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

        result = is_disk_cached_latents_is_expected(reso=(512, 512), npz_path=str(npz_path), flip_aug=True, alpha_mask=False)

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

        result = is_disk_cached_latents_is_expected(reso=(512, 512), npz_path=str(npz_path), flip_aug=True, alpha_mask=False)

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

        result = is_disk_cached_latents_is_expected(reso=(512, 512), npz_path=str(npz_path), flip_aug=False, alpha_mask=True)

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

        result = is_disk_cached_latents_is_expected(reso=(512, 512), npz_path=str(npz_path), flip_aug=False, alpha_mask=False)

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


# =============================================================================
# Heavy Mocking Tests - load_images_and_masks_for_caching
# =============================================================================

from unittest.mock import Mock, patch
from library.data.caching import (
    load_images_and_masks_for_caching,
    cache_batch_latents,
    cache_batch_text_encoder_outputs,
)


@pytest.mark.unit
class TestLoadImagesAndMasksForCaching:
    """Test load_images_and_masks_for_caching with mocked image loading."""

    @pytest.fixture
    def mock_image_info(self):
        """Create a mock ImageInfo object."""
        info = Mock()
        info.absolute_path = "/path/to/image.png"
        info.image = None
        info.bucket_reso = (512, 512)
        info.resized_size = (512, 512)
        info.resize_interpolation = None
        return info

    @patch("library.data.caching.load_image")
    @patch("library.data.caching.trim_and_resize_if_required")
    def test_returns_correct_tuple_structure(self, mock_trim, mock_load, mock_image_info):
        """Test that function returns (tensor, masks, sizes, crops) tuple."""
        # Setup mocks
        mock_load.return_value = np.zeros((512, 512, 3), dtype=np.uint8)
        mock_trim.return_value = (
            np.zeros((512, 512, 3), dtype=np.uint8),
            (512, 512),  # original_size
            (0, 0, 512, 512),  # crop_ltrb
        )

        result = load_images_and_masks_for_caching([mock_image_info], use_alpha_mask=False, random_crop=False)

        assert len(result) == 4
        img_tensor, alpha_masks, original_sizes, crop_ltrbs = result
        assert isinstance(img_tensor, torch.Tensor)
        assert len(alpha_masks) == 1
        assert len(original_sizes) == 1
        assert len(crop_ltrbs) == 1

    @patch("library.data.caching.load_image")
    @patch("library.data.caching.trim_and_resize_if_required")
    def test_extracts_alpha_mask_when_requested(self, mock_trim, mock_load, mock_image_info):
        """Test that alpha mask is extracted when use_alpha_mask=True."""
        # Image with alpha channel
        rgba_image = np.zeros((512, 512, 4), dtype=np.uint8)
        rgba_image[:, :, 3] = 128  # 50% alpha

        mock_load.return_value = rgba_image
        mock_trim.return_value = (rgba_image, (512, 512), (0, 0, 512, 512))

        _, alpha_masks, _, _ = load_images_and_masks_for_caching([mock_image_info], use_alpha_mask=True, random_crop=False)

        assert alpha_masks[0] is not None
        assert isinstance(alpha_masks[0], torch.Tensor)

    @patch("library.data.caching.load_image")
    @patch("library.data.caching.trim_and_resize_if_required")
    def test_alpha_mask_is_none_when_not_requested(self, mock_trim, mock_load, mock_image_info):
        """Test that alpha mask is None when use_alpha_mask=False."""
        mock_load.return_value = np.zeros((512, 512, 3), dtype=np.uint8)
        mock_trim.return_value = (np.zeros((512, 512, 3), dtype=np.uint8), (512, 512), (0, 0, 512, 512))

        _, alpha_masks, _, _ = load_images_and_masks_for_caching([mock_image_info], use_alpha_mask=False, random_crop=False)

        assert alpha_masks[0] is None

    @patch("library.data.caching.load_image")
    @patch("library.data.caching.trim_and_resize_if_required")
    def test_uses_preloaded_image_if_available(self, mock_trim, mock_load, mock_image_info):
        """Test that preloaded image is used instead of loading from disk."""
        mock_image_info.image = np.zeros((512, 512, 3), dtype=np.uint8)
        mock_trim.return_value = (np.zeros((512, 512, 3), dtype=np.uint8), (512, 512), (0, 0, 512, 512))

        load_images_and_masks_for_caching([mock_image_info], use_alpha_mask=False, random_crop=False)

        # Should not call load_image when image is preloaded
        mock_load.assert_not_called()


# =============================================================================
# Heavy Mocking Tests - cache_batch_latents
# =============================================================================


@pytest.mark.unit
class TestCacheBatchLatents:
    """Test cache_batch_latents with mocked VAE."""

    @pytest.fixture
    def mock_vae(self):
        """Create a mock VAE that returns fake latents."""
        vae = Mock()
        vae.device = torch.device("cpu")
        vae.dtype = torch.float32

        # Mock encode chain: vae.encode(x).latent_dist.sample()
        latent_dist = Mock()
        latent_dist.sample.return_value = torch.randn(1, 4, 64, 64)
        encode_result = Mock()
        encode_result.latent_dist = latent_dist
        vae.encode.return_value = encode_result

        return vae

    @pytest.fixture
    def mock_image_info(self):
        """Create a mock ImageInfo for caching."""
        info = Mock()
        info.absolute_path = "/path/to/image.png"
        info.image = np.zeros((512, 512, 3), dtype=np.uint8)
        info.bucket_reso = (512, 512)
        info.resized_size = (512, 512)
        info.resize_interpolation = None
        info.latents_npz = "/path/to/latents.npz"
        return info

    @patch("library.data.caching.clean_memory_on_device")
    @patch("library.data.caching.trim_and_resize_if_required")
    def test_sets_latents_on_info_object(self, mock_trim, mock_clean, mock_vae, mock_image_info):
        """Test that latents are set on info object when cache_to_disk=False."""
        mock_trim.return_value = (np.zeros((512, 512, 3), dtype=np.uint8), (512, 512), (0, 0, 512, 512))

        cache_batch_latents(
            mock_vae, cache_to_disk=False, image_infos=[mock_image_info], flip_aug=False, use_alpha_mask=False, random_crop=False
        )

        assert mock_image_info.latents is not None

    @patch("library.data.caching.clean_memory_on_device")
    @patch("library.data.caching.trim_and_resize_if_required")
    def test_sets_flipped_latents_when_flip_aug(self, mock_trim, mock_clean, mock_vae, mock_image_info):
        """Test that flipped latents are set when flip_aug=True."""
        mock_trim.return_value = (np.zeros((512, 512, 3), dtype=np.uint8), (512, 512), (0, 0, 512, 512))

        cache_batch_latents(
            mock_vae, cache_to_disk=False, image_infos=[mock_image_info], flip_aug=True, use_alpha_mask=False, random_crop=False
        )

        assert mock_image_info.latents_flipped is not None

    @patch("library.data.caching.clean_memory_on_device")
    @patch("library.data.caching.trim_and_resize_if_required")
    def test_calls_vae_encode(self, mock_trim, mock_clean, mock_vae, mock_image_info):
        """Test that VAE encode is called."""
        mock_trim.return_value = (np.zeros((512, 512, 3), dtype=np.uint8), (512, 512), (0, 0, 512, 512))

        cache_batch_latents(
            mock_vae, cache_to_disk=False, image_infos=[mock_image_info], flip_aug=False, use_alpha_mask=False, random_crop=False
        )

        mock_vae.encode.assert_called_once()

    @patch("library.data.caching.clean_memory_on_device")
    @patch("library.data.caching.trim_and_resize_if_required")
    def test_cleans_memory_after_caching(self, mock_trim, mock_clean, mock_vae, mock_image_info):
        """Test that memory cleanup is called after caching."""
        mock_trim.return_value = (np.zeros((512, 512, 3), dtype=np.uint8), (512, 512), (0, 0, 512, 512))

        cache_batch_latents(
            mock_vae, cache_to_disk=False, image_infos=[mock_image_info], flip_aug=False, use_alpha_mask=False, random_crop=False
        )

        mock_clean.assert_called_once()


# =============================================================================
# Heavy Mocking Tests - cache_batch_text_encoder_outputs
# =============================================================================


@pytest.mark.unit
class TestCacheBatchTextEncoderOutputs:
    """Test cache_batch_text_encoder_outputs with mocked text encoders."""

    @pytest.fixture
    def mock_tokenizers(self):
        """Create mock tokenizers."""
        tok1 = Mock()
        tok2 = Mock()
        return [tok1, tok2]

    @pytest.fixture
    def mock_text_encoders(self):
        """Create mock text encoders."""
        enc1 = Mock()
        enc1.device = torch.device("cpu")
        enc2 = Mock()
        enc2.device = torch.device("cpu")
        return [enc1, enc2]

    @pytest.fixture
    def mock_image_info(self, tmp_path):
        """Create mock ImageInfo for text encoder caching."""
        info = Mock()
        info.text_encoder_outputs_npz = str(tmp_path / "te_outputs.npz")
        return info

    @patch("library.data.caching.get_hidden_states_sdxl")
    def test_sets_outputs_on_info_when_cache_to_disk_false(self, mock_get_hidden, mock_tokenizers, mock_text_encoders, mock_image_info):
        """Test that outputs are set on info object when cache_to_disk=False."""
        mock_get_hidden.return_value = (torch.randn(1, 77, 768), torch.randn(1, 77, 1280), torch.randn(1, 1280))

        input_ids1 = torch.randint(0, 1000, (1, 77))
        input_ids2 = torch.randint(0, 1000, (1, 77))

        cache_batch_text_encoder_outputs(
            [mock_image_info],
            mock_tokenizers,
            mock_text_encoders,
            max_token_length=77,
            cache_to_disk=False,
            input_ids1=input_ids1,
            input_ids2=input_ids2,
            dtype=torch.float32,
        )

        assert mock_image_info.text_encoder_outputs1 is not None
        assert mock_image_info.text_encoder_outputs2 is not None
        assert mock_image_info.text_encoder_pool2 is not None

    @patch("library.data.caching.get_hidden_states_sdxl")
    def test_saves_to_disk_when_cache_to_disk_true(self, mock_get_hidden, mock_tokenizers, mock_text_encoders, mock_image_info, tmp_path):
        """Test that outputs are saved to disk when cache_to_disk=True."""
        mock_get_hidden.return_value = (torch.randn(1, 77, 768), torch.randn(1, 77, 1280), torch.randn(1, 1280))

        input_ids1 = torch.randint(0, 1000, (1, 77))
        input_ids2 = torch.randint(0, 1000, (1, 77))

        cache_batch_text_encoder_outputs(
            [mock_image_info],
            mock_tokenizers,
            mock_text_encoders,
            max_token_length=77,
            cache_to_disk=True,
            input_ids1=input_ids1,
            input_ids2=input_ids2,
            dtype=torch.float32,
        )

        # Check file was created
        import os

        assert os.path.exists(mock_image_info.text_encoder_outputs_npz)

    @patch("library.data.caching.get_hidden_states_sdxl")
    def test_calls_get_hidden_states_sdxl(self, mock_get_hidden, mock_tokenizers, mock_text_encoders, mock_image_info):
        """Test that get_hidden_states_sdxl is called with correct args."""
        mock_get_hidden.return_value = (torch.randn(1, 77, 768), torch.randn(1, 77, 1280), torch.randn(1, 1280))

        input_ids1 = torch.randint(0, 1000, (1, 77))
        input_ids2 = torch.randint(0, 1000, (1, 77))

        cache_batch_text_encoder_outputs(
            [mock_image_info],
            mock_tokenizers,
            mock_text_encoders,
            max_token_length=77,
            cache_to_disk=False,
            input_ids1=input_ids1,
            input_ids2=input_ids2,
            dtype=torch.float32,
        )

        mock_get_hidden.assert_called_once()
