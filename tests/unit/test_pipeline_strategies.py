"""
Unit tests for pipeline caching strategies.

Tests SdLatentsPipelineStrategy, SdxlLatentsPipelineStrategy, and SdxlTextEncoderPipelineStrategy.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from library.data.pipeline.dataclasses import CacheEntry
from library.strategies.pipeline_sd import SdLatentsPipelineStrategy, SD_VAE_LATENT_SCALE
from library.strategies.pipeline_sdxl import (
    SdxlLatentsPipelineStrategy,
    SdxlTextEncoderPipelineStrategy,
    SDXL_VAE_LATENT_SCALE,
)


@pytest.fixture
def sample_entry() -> CacheEntry:
    """Create a sample CacheEntry for testing."""
    return CacheEntry(
        id="test_image_001",
        image_path="/data/images/test.png",
        original_size=(1920, 1080),
        bucket_reso=(1024, 576),
        resized_size=(1024, 576),
        caption="a test image for caching",
    )


@pytest.fixture
def sample_entries(sample_entry: CacheEntry) -> list[CacheEntry]:
    """Create a batch of sample entries."""
    entries = [sample_entry]
    for i in range(2, 4):
        entries.append(
            CacheEntry(
                id=f"test_image_{i:03d}",
                image_path=f"/data/images/test{i}.png",
                original_size=(1920, 1080),
                bucket_reso=(1024, 576),
                resized_size=(1024, 576),
                caption=f"test caption {i}",
            )
        )
    return entries


@pytest.fixture
def mock_vae():
    """Create a mock VAE model."""
    vae = MagicMock()
    vae.device = torch.device("cpu")
    vae.dtype = torch.float32

    # Mock encoding: return latent distribution with sample method
    latent_dist = MagicMock()
    # Return latents of shape [B, 4, H/8, W/8]
    latent_dist.sample.return_value = torch.randn(3, 4, 72, 128)
    vae.encode.return_value.latent_dist = latent_dist

    return vae


class TestSdLatentsPipelineStrategy:
    """Tests for SD latent caching strategy."""

    def test_get_cache_path(self, sample_entry: CacheEntry, tmp_path: Path):
        """Test cache path generation."""
        strategy = SdLatentsPipelineStrategy()
        path = strategy.get_cache_path(sample_entry, tmp_path)

        assert path == tmp_path / "test_image_001_sd_latents.safetensors"

    def test_get_cache_path_custom_suffix(self, sample_entry: CacheEntry, tmp_path: Path):
        """Test cache path with custom suffix."""
        strategy = SdLatentsPipelineStrategy(cache_suffix="_custom.safetensors")
        path = strategy.get_cache_path(sample_entry, tmp_path)

        assert path == tmp_path / "test_image_001_custom.safetensors"

    def test_encode_batch(self, sample_entries: list[CacheEntry], mock_vae):
        """Test batch encoding produces correct output structure."""
        strategy = SdLatentsPipelineStrategy()
        images = torch.randn(3, 3, 576, 1024)  # [B, C, H, W]

        results = strategy.encode_batch(images, mock_vae, sample_entries)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert "latents" in result
            assert "metadata" in result
            assert result["latents"].shape == (4, 72, 128)
            assert "original_size" in result["metadata"]
            assert "bucket_reso" in result["metadata"]

    def test_encode_batch_with_flip_aug(self, sample_entries: list[CacheEntry], mock_vae):
        """Test that flip augmentation produces flipped latents."""
        strategy = SdLatentsPipelineStrategy(flip_aug=True)
        images = torch.randn(3, 3, 576, 1024)

        results = strategy.encode_batch(images, mock_vae, sample_entries)

        for result in results:
            assert "latents_flipped" in result
            assert result["latents_flipped"].shape == (4, 72, 128)

    def test_save_and_load_cache(self, sample_entry: CacheEntry, tmp_path: Path):
        """Test save/load roundtrip."""
        strategy = SdLatentsPipelineStrategy()
        cache_path = strategy.get_cache_path(sample_entry, tmp_path)

        # Create fake cache data
        data = {
            "latents": torch.randn(4, 72, 128, dtype=torch.float16),
            "metadata": {
                "original_size": "1920,1080",
                "bucket_reso": "1024,576",
            },
        }

        # Save
        strategy.save_cache(data, cache_path)
        assert cache_path.exists()

        # Load
        loaded = strategy.load_cache(cache_path)
        assert "latents" in loaded
        assert loaded["latents"].shape == (4, 72, 128)

    def test_preprocess_image(self):
        """Test image preprocessing."""
        from PIL import Image

        strategy = SdLatentsPipelineStrategy()
        image = Image.new("RGB", (1920, 1080), color=(128, 128, 128))

        tensor = strategy.preprocess_image(image, (1024, 576))

        assert tensor.shape == (3, 576, 1024)  # [C, H, W]
        assert tensor.min() >= -1.0
        assert tensor.max() <= 1.0


class TestSdxlLatentsPipelineStrategy:
    """Tests for SDXL latent caching strategy."""

    def test_get_cache_path(self, sample_entry: CacheEntry, tmp_path: Path):
        """Test cache path generation."""
        strategy = SdxlLatentsPipelineStrategy()
        path = strategy.get_cache_path(sample_entry, tmp_path)

        assert path == tmp_path / "test_image_001_sdxl_latents.safetensors"

    def test_encode_batch(self, sample_entries: list[CacheEntry], mock_vae):
        """Test batch encoding produces correct output structure."""
        strategy = SdxlLatentsPipelineStrategy()
        images = torch.randn(3, 3, 576, 1024)

        results = strategy.encode_batch(images, mock_vae, sample_entries)

        assert len(results) == 3
        for result in results:
            assert "latents" in result
            assert "metadata" in result

    def test_scale_factor_differs_from_sd(self):
        """Verify SDXL uses different scale factor than SD."""
        assert SDXL_VAE_LATENT_SCALE != SD_VAE_LATENT_SCALE
        assert SDXL_VAE_LATENT_SCALE == 0.13025
        assert SD_VAE_LATENT_SCALE == 0.18215


class TestSdxlTextEncoderPipelineStrategy:
    """Tests for SDXL text encoder caching strategy."""

    @pytest.fixture
    def mock_text_encoders(self):
        """Create mock SDXL text encoders and tokenizers."""
        # Mock tokenizers
        tokenizer1 = MagicMock()
        tokenizer1.model_max_length = 77
        tokenizer1.return_value.input_ids = torch.zeros(3, 77, dtype=torch.long)

        tokenizer2 = MagicMock()
        tokenizer2.model_max_length = 77
        tokenizer2.return_value.input_ids = torch.zeros(3, 77, dtype=torch.long)

        # Mock text encoder 1
        text_encoder1 = MagicMock()
        text_encoder1.device = torch.device("cpu")
        enc1_out = MagicMock()
        enc1_out.hidden_states = [torch.randn(3, 77, 768) for _ in range(13)]
        text_encoder1.return_value = enc1_out

        # Mock text encoder 2
        text_encoder2 = MagicMock()
        text_encoder2.device = torch.device("cpu")
        enc2_out = MagicMock()
        enc2_out.hidden_states = [torch.randn(3, 77, 1280) for _ in range(33)]
        enc2_out.text_embeds = torch.randn(3, 1280)
        text_encoder2.return_value = enc2_out

        return (text_encoder1, text_encoder2, tokenizer1, tokenizer2)

    def test_get_cache_path(self, sample_entry: CacheEntry, tmp_path: Path):
        """Test cache path generation."""
        strategy = SdxlTextEncoderPipelineStrategy()
        path = strategy.get_cache_path(sample_entry, tmp_path)

        assert path == tmp_path / "test_image_001_sdxl_te.safetensors"

    def test_encode_batch(self, sample_entries: list[CacheEntry], mock_text_encoders):
        """Test text encoding produces correct output structure."""
        strategy = SdxlTextEncoderPipelineStrategy()
        dummy_images = torch.empty(0)  # Not used for TE caching

        results = strategy.encode_batch(dummy_images, mock_text_encoders, sample_entries)

        assert len(results) == 3
        for result in results:
            assert "hidden_state1" in result
            assert "hidden_state2" in result
            assert "pool2" in result
            assert "metadata" in result

    def test_save_and_load_cache(self, sample_entry: CacheEntry, tmp_path: Path):
        """Test save/load roundtrip."""
        strategy = SdxlTextEncoderPipelineStrategy()
        cache_path = strategy.get_cache_path(sample_entry, tmp_path)

        # Create fake cache data
        data = {
            "hidden_state1": torch.randn(77, 768, dtype=torch.float16),
            "hidden_state2": torch.randn(77, 1280, dtype=torch.float16),
            "pool2": torch.randn(1280, dtype=torch.float16),
            "metadata": {"caption_hash": "12345678"},
        }

        # Save
        strategy.save_cache(data, cache_path)
        assert cache_path.exists()

        # Load
        loaded = strategy.load_cache(cache_path)
        assert "hidden_state1" in loaded
        assert "hidden_state2" in loaded
        assert "pool2" in loaded
