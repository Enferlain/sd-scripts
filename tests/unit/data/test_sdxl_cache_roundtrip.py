"""
Comprehensive tests for SDXL cache save/load roundtrip.

Verifies that latents and text encoder outputs are correctly:
1. Encoded to tensors
2. Saved to .safetensors files
3. Loaded back with identical values
4. Metadata is preserved
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from library.data.structures import CacheEntry
from library.strategies.sdxl.caching import (
    SdxlLatentsPipelineStrategy,
    SdxlTextEncoderPipelineStrategy,
    get_crop_ltrb,
)


@pytest.fixture
def sample_entry(tmp_path: Path) -> CacheEntry:
    """Create a sample cache entry for testing (with cache paths pre-set)."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    entry_id = "test_image_001"
    return CacheEntry(
        id=entry_id,
        image_path="/path/to/image.jpg",
        original_size=(1920, 1080),
        bucket_reso=(1024, 576),
        resized_size=(1024, 576),
        caption="a beautiful landscape with mountains",
        latent_cache_path=str(cache_dir / f"{entry_id}_latent.safetensors"),
        te_cache_path=str(cache_dir / f"{entry_id}_te.safetensors"),
    )


@pytest.fixture
def mock_vae():
    """Create a mock VAE that returns properly shaped latents."""
    vae = MagicMock()
    vae.device = torch.device("cpu")
    vae.dtype = torch.float32

    def mock_encode(images):
        b, c, h, w = images.shape
        latent_h, latent_w = h // 8, w // 8
        mock_output = MagicMock()
        # Return deterministic values for testing (not random)
        mock_output.latent_dist.sample.return_value = (
            torch.arange(b * 4 * latent_h * latent_w, dtype=torch.float32).reshape(b, 4, latent_h, latent_w) / 1000.0
        )
        return mock_output

    vae.encode = mock_encode
    return vae


class TestSdxlLatentsSaveLoad:
    """Test SDXL latent caching save/load roundtrip."""

    def test_save_and_load_preserves_latents(self, sample_entry: CacheEntry, mock_vae, tmp_path: Path):
        """Test that saved latents can be loaded back with identical values."""
        strategy = SdxlLatentsPipelineStrategy(dtype="fp32")
        cache_path = Path(sample_entry.latent_cache_path)

        # Create fake image tensor matching bucket resolution
        images = torch.randn(1, 3, 576, 1024)  # [B, C, H, W]

        # Encode
        results = strategy.encode_batch(images, mock_vae, [sample_entry])
        assert len(results) == 1
        original_data = results[0]

        # Save
        strategy.save_cache(original_data, cache_path)
        assert cache_path.exists()

        # Load
        cache_data = strategy.load_cache(cache_path)

        # Verify latents are identical
        assert cache_data.latents is not None
        assert torch.allclose(original_data["latents"], cache_data.latents)

    def test_save_and_load_with_flip_aug(self, sample_entry: CacheEntry, mock_vae, tmp_path: Path):
        """Test that flipped latents are saved and loaded correctly."""
        strategy = SdxlLatentsPipelineStrategy(dtype="fp32", flip_aug=True)
        cache_path = Path(sample_entry.latent_cache_path)

        images = torch.randn(1, 3, 576, 1024)
        results = strategy.encode_batch(images, mock_vae, [sample_entry])
        original_data = results[0]

        # Should have both regular and flipped
        assert "latents" in original_data
        assert "latents_flipped" in original_data

        strategy.save_cache(original_data, cache_path)
        cache_data = strategy.load_cache(cache_path)

        assert torch.allclose(original_data["latents"], cache_data.latents)
        assert torch.allclose(original_data["latents_flipped"], cache_data.latents_flipped)

    def test_metadata_is_preserved(self, sample_entry: CacheEntry, mock_vae, tmp_path: Path):
        """Test that metadata is saved and can be read back."""
        from safetensors import safe_open

        strategy = SdxlLatentsPipelineStrategy(dtype="fp32")
        cache_path = Path(sample_entry.latent_cache_path)

        images = torch.randn(1, 3, 576, 1024)
        results = strategy.encode_batch(images, mock_vae, [sample_entry])
        strategy.save_cache(results[0], cache_path)

        # Read metadata directly
        with safe_open(str(cache_path), framework="pt") as f:
            metadata = f.metadata()

        assert metadata is not None
        assert "original_size" in metadata
        assert metadata["original_size"] == "1920,1080"
        assert "bucket_reso" in metadata
        assert metadata["bucket_reso"] == "1024,576"
        assert "crop_ltrb" in metadata

    def test_dtype_fp16_roundtrip(self, sample_entry: CacheEntry, mock_vae, tmp_path: Path):
        """Test fp16 latents save/load correctly."""
        strategy = SdxlLatentsPipelineStrategy(dtype="fp16")
        cache_path = Path(sample_entry.latent_cache_path)

        images = torch.randn(1, 3, 576, 1024)
        results = strategy.encode_batch(images, mock_vae, [sample_entry])
        original_latents = results[0]["latents"]

        # Verify dtype
        assert original_latents.dtype == torch.float16

        strategy.save_cache(results[0], cache_path)
        cache_data = strategy.load_cache(cache_path)

        assert cache_data.latents.dtype == torch.float16
        assert torch.allclose(original_latents, cache_data.latents)

    def test_is_cache_valid_returns_true_for_valid_cache(self, sample_entry: CacheEntry, mock_vae, tmp_path: Path):
        """Test validation passes for properly cached files."""
        strategy = SdxlLatentsPipelineStrategy(dtype="fp32")
        cache_path = Path(sample_entry.latent_cache_path)

        images = torch.randn(1, 3, 576, 1024)
        results = strategy.encode_batch(images, mock_vae, [sample_entry])
        strategy.save_cache(results[0], cache_path)

        # Should be valid
        assert strategy.is_cache_valid(cache_path, sample_entry, flip_aug=False)

    def test_is_cache_valid_returns_false_for_missing_flip(self, sample_entry: CacheEntry, mock_vae, tmp_path: Path):
        """Test validation fails when flip_aug is required but not cached."""
        strategy = SdxlLatentsPipelineStrategy(dtype="fp32", flip_aug=False)
        cache_path = Path(sample_entry.latent_cache_path)

        images = torch.randn(1, 3, 576, 1024)
        results = strategy.encode_batch(images, mock_vae, [sample_entry])
        strategy.save_cache(results[0], cache_path)

        # Should fail when flip_aug is required
        assert not strategy.is_cache_valid(cache_path, sample_entry, flip_aug=True)


class TestSdxlTextEncoderSaveLoad:
    """Test SDXL text encoder caching save/load roundtrip."""

    @pytest.fixture
    def mock_text_encoders(self):
        """Create mock text encoders."""
        te1 = MagicMock()
        te1.device = torch.device("cpu")

        te2 = MagicMock()
        te2.device = torch.device("cpu")

        tokenizer1 = MagicMock()
        tokenizer1.model_max_length = 77
        tokenizer1.return_value = MagicMock(input_ids=torch.zeros(1, 77, dtype=torch.long))

        tokenizer2 = MagicMock()
        tokenizer2.model_max_length = 77
        tokenizer2.return_value = MagicMock(input_ids=torch.zeros(1, 77, dtype=torch.long))

        # Mock encoder outputs
        enc1_out = MagicMock()
        enc1_out.hidden_states = [torch.randn(1, 77, 768) for _ in range(12)]
        te1.return_value = enc1_out

        enc2_out = MagicMock()
        enc2_out.hidden_states = [torch.randn(1, 77, 1280) for _ in range(24)]
        enc2_out.text_embeds = torch.randn(1, 1280)
        te2.return_value = enc2_out

        return te1, te2, tokenizer1, tokenizer2

    def test_save_and_load_preserves_embeddings(self, sample_entry: CacheEntry, mock_text_encoders, tmp_path: Path):
        """Test that TE embeddings are saved and loaded correctly."""
        strategy = SdxlTextEncoderPipelineStrategy(dtype="fp32")
        cache_path = Path(sample_entry.latent_cache_path)

        # Encode (images not used for TE)
        dummy_images = torch.empty(0)
        results = strategy.encode_batch(dummy_images, mock_text_encoders, [sample_entry])
        original_data = results[0]

        assert "hidden_state1" in original_data
        assert "hidden_state2" in original_data
        assert "pool2" in original_data

        # Save and load
        strategy.save_cache(original_data, cache_path)
        cache_data = strategy.load_cache(cache_path)

        assert torch.allclose(original_data["hidden_state1"], cache_data.aux["hidden_state1"])
        assert torch.allclose(original_data["hidden_state2"], cache_data.aux["hidden_state2"])
        assert torch.allclose(original_data["pool2"], cache_data.aux["pool2"])

    def test_te_metadata_includes_caption_hash(self, sample_entry: CacheEntry, mock_text_encoders, tmp_path: Path):
        """Test that caption hash is stored for change detection."""
        from safetensors import safe_open

        from library.utils.hash_utils import stable_string_hash

        strategy = SdxlTextEncoderPipelineStrategy(dtype="fp32")
        cache_path = Path(sample_entry.latent_cache_path)

        dummy_images = torch.empty(0)
        results = strategy.encode_batch(dummy_images, mock_text_encoders, [sample_entry])
        strategy.save_cache(results[0], cache_path)

        with safe_open(str(cache_path), framework="pt") as f:
            metadata = f.metadata()

        assert "caption_hash" in metadata
        expected_hash = str(stable_string_hash(sample_entry.caption))
        assert metadata["caption_hash"] == expected_hash

    def test_is_cache_valid_detects_caption_change(self, sample_entry: CacheEntry, mock_text_encoders, tmp_path: Path):
        """Test that validation fails when caption has changed."""
        strategy = SdxlTextEncoderPipelineStrategy(dtype="fp32")
        cache_path = Path(sample_entry.latent_cache_path)

        dummy_images = torch.empty(0)
        results = strategy.encode_batch(dummy_images, mock_text_encoders, [sample_entry])
        strategy.save_cache(results[0], cache_path)

        # Should be valid with original caption
        assert strategy.is_cache_valid(cache_path, sample_entry)

        # Change caption
        modified_entry = CacheEntry(
            id=sample_entry.id,
            image_path=sample_entry.image_path,
            original_size=sample_entry.original_size,
            bucket_reso=sample_entry.bucket_reso,
            resized_size=sample_entry.resized_size,
            caption="a completely different caption",  # Changed!
        )

        # Should fail validation
        assert not strategy.is_cache_valid(cache_path, modified_entry)


class TestCropCoordinates:
    """Test crop coordinate calculation for SDXL micro-conditioning."""

    def test_no_crop_for_matching_aspect(self):
        """Perfect aspect ratio match = no cropping."""
        crop = get_crop_ltrb((1024, 576), (1920, 1080))
        assert crop == (0, 0, 1024, 576)

    def test_crop_for_wider_image(self):
        """Image wider than bucket = vertical crop."""
        crop = get_crop_ltrb((1024, 576), (2000, 1000))
        # 2:1 aspect fitting into 1.78:1 bucket
        assert crop[0] == 0  # left
        assert crop[1] > 0  # top (some crop)
        assert crop[2] == 1024  # right
        assert crop[3] < 576  # bottom

    def test_crop_for_taller_image(self):
        """Image taller than bucket = horizontal crop."""
        crop = get_crop_ltrb((1024, 576), (1000, 2000))
        # 0.5:1 aspect fitting into 1.78:1 bucket
        assert crop[0] > 0  # left (some crop)
        assert crop[1] == 0  # top
        assert crop[2] < 1024  # right
        assert crop[3] == 576  # bottom

    def test_square_to_square(self):
        """Square image to square bucket = no cropping."""
        crop = get_crop_ltrb((1024, 1024), (800, 800))
        assert crop == (0, 0, 1024, 1024)
