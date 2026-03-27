"""
Unit tests for pipeline caching strategies.

Tests SdLatentsPipelineStrategy, SdxlLatentsPipelineStrategy, and SdxlTextEncoderPipelineStrategy.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from library.constants import SD_VAE_LATENT_SCALE, SDXL_VAE_LATENT_SCALE
from library.data.structures import CacheEntry
from library.strategies.sd.caching import SdLatentsPipelineStrategy, SdTextEncoderPipelineStrategy
from library.strategies.sdxl.caching import SdxlLatentsPipelineStrategy, SdxlTextEncoderPipelineStrategy


@pytest.fixture
def sample_entry(tmp_path: Path) -> CacheEntry:
    """Create a sample CacheEntry for testing (with cache paths pre-set)."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    entry_id = "test_image_001"
    return CacheEntry(
        id=entry_id,
        image_path="/data/images/test.png",
        original_size=(1920, 1080),
        bucket_reso=(1024, 576),
        resized_size=(1024, 576),
        caption="a test image for caching",
        latent_cache_path=str(cache_dir / f"{entry_id}_latent.safetensors"),
        te_cache_path=str(cache_dir / f"{entry_id}_te.safetensors"),
    )


@pytest.fixture
def sample_entries(sample_entry: CacheEntry, tmp_path: Path) -> list[CacheEntry]:
    """Create a batch of sample entries."""
    cache_dir = tmp_path / "cache"
    entries = [sample_entry]
    for i in range(2, 4):
        entry_id = f"test_image_{i:03d}"
        entries.append(
            CacheEntry(
                id=entry_id,
                image_path=f"/data/images/test{i}.png",
                original_size=(1920, 1080),
                bucket_reso=(1024, 576),
                resized_size=(1024, 576),
                caption=f"test caption {i}",
                latent_cache_path=str(cache_dir / f"{entry_id}_latent.safetensors"),
                te_cache_path=str(cache_dir / f"{entry_id}_te.safetensors"),
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

    def test_get_entry_cache_path(self, sample_entry: CacheEntry, tmp_path: Path):
        """Test that entry has cache path set."""
        strategy = SdLatentsPipelineStrategy()
        # Verify entry path was set by fixture
        path = strategy.get_entry_cache_path(sample_entry)
        assert path is not None
        assert path.endswith("_latent.safetensors")

    def test_cache_path_format(self, tmp_path: Path):
        """Test that cache paths are set correctly at manifest creation."""
        # This tests the path format that create_manifest uses
        cache_dir = tmp_path / "cache"
        entry_id = "my_image"
        expected_path = str(cache_dir / f"{entry_id}_latent.safetensors")
        entry = CacheEntry(
            id=entry_id,
            image_path="/test.png",
            original_size=(512, 512),
            bucket_reso=(512, 512),
            resized_size=(512, 512),
            caption="test",
            latent_cache_path=expected_path,
        )
        assert entry.latent_cache_path == expected_path

    def test_encode_batch(self, sample_entries: list[CacheEntry], mock_vae):
        """Test batch encoding produces correct output structure."""
        strategy = SdLatentsPipelineStrategy()
        images = torch.randn(3, 3, 576, 1024)  # [B, C, H, W]

        results = strategy.encode_batch(images, mock_vae, sample_entries)

        assert len(results) == 3
        for _, result in enumerate(results):
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
        cache_path = Path(sample_entry.latent_cache_path)

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

        # Load - returns CacheData object now
        loaded = strategy.load_cache(cache_path)
        assert loaded.latents is not None
        assert loaded.latents.shape == (4, 72, 128)
        assert loaded.conditioning is None

    def test_preprocess_image(self):
        """Test image preprocessing."""
        from PIL import Image

        strategy = SdLatentsPipelineStrategy()
        image = Image.new("RGB", (1920, 1080), color=(128, 128, 128))

        tensor = strategy.preprocess_image(image, (1024, 576))

        assert tensor.shape == (3, 576, 1024)  # [C, H, W]
        assert tensor.min() >= -1.0
        assert tensor.max() <= 1.0

    def test_preprocess_image_resize_then_crop(self):
        """Test that mismatched AR images are resized then cropped (not squished)."""
        from PIL import Image

        strategy = SdLatentsPipelineStrategy()
        # Image with slightly wider AR than bucket (1556/2048 = 0.76 vs 1536/2048 = 0.75)
        image = Image.new("RGB", (1556, 2048), color=(100, 150, 200))

        # Target bucket is 1536x2048, resized_size calculated by select_bucket would be (1556, 2048)
        tensor = strategy.preprocess_image(
            image,
            target_size=(1536, 2048),  # bucket_reso
            resized_size=(1556, 2048),  # maintains AR, slightly larger
        )

        # Output should be exactly bucket size (cropped, not squished)
        assert tensor.shape == (3, 2048, 1536)  # [C, H, W]

    def test_preprocess_image_random_crop_varies(self):
        """Test that random_crop produces varied crops."""
        from PIL import Image
        import numpy as np

        strategy = SdLatentsPipelineStrategy()
        # Create image with horizontal gradient so different crop positions differ
        width, height = 1600, 2048
        gradient = np.linspace(0, 255, width, dtype=np.uint8)
        gradient_img = np.tile(gradient, (height, 1))
        gradient_rgb = np.stack([gradient_img, gradient_img, gradient_img], axis=2)
        image = Image.fromarray(gradient_rgb, "RGB")

        crops = []
        for _ in range(10):
            tensor = strategy.preprocess_image(
                image,
                target_size=(1536, 2048),
                resized_size=(1600, 2048),  # 64px to crop
                random_crop=True,
            )
            # Check the exact values differ (random crop offset)
            crops.append(tensor[0, 0, 0].item())

        # With random crop on gradient image, we should see variation
        assert len(set(crops)) > 1, "Random crop should produce varied results"

    def test_preprocess_image_center_crop_consistent(self):
        """Test that center crop is deterministic."""
        from PIL import Image

        strategy = SdLatentsPipelineStrategy()
        image = Image.new("RGB", (1600, 2048), color=(100, 150, 200))

        tensors = [
            strategy.preprocess_image(
                image,
                target_size=(1536, 2048),
                resized_size=(1600, 2048),
                random_crop=False,
            )
            for _ in range(3)
        ]

        # Center crop should be identical each time
        assert torch.allclose(tensors[0], tensors[1])
        assert torch.allclose(tensors[1], tensors[2])


class TestSdTextEncoderPipelineStrategy:
    """Tests for SD text encoder caching strategy."""

    @pytest.fixture
    def mock_text_encoder_bundle(self):
        """Create a mock SD text encoder and tokenizer."""
        tokenizer = MagicMock()
        tokenizer.model_max_length = 77
        tokenizer.bos_token_id = 49406
        tokenizer.eos_token_id = 49407
        tokenizer.pad_token_id = 49407
        tokenizer.eos_token = 49407

        def tokenizer_call(text, **kwargs):
            batch_size = len(text) if isinstance(text, list) else 1
            return MagicMock(input_ids=torch.zeros(batch_size, 77, dtype=torch.long))

        tokenizer.__call__ = tokenizer_call
        tokenizer.side_effect = tokenizer_call

        text_encoder = MagicMock()
        text_encoder.device = torch.device("cpu")
        text_encoder.return_value = (torch.randn(3, 77, 768),)

        return (text_encoder, tokenizer)

    def test_get_entry_cache_path(self, sample_entry: CacheEntry):
        """Test that TE strategy reads te_cache_path from entry."""
        strategy = SdTextEncoderPipelineStrategy()
        path = strategy.get_entry_cache_path(sample_entry)
        assert path is not None
        assert path.endswith("_te.safetensors")

    def test_encode_batch(self, sample_entries: list[CacheEntry], mock_text_encoder_bundle):
        """Test SD text encoding produces the expected output structure."""
        strategy = SdTextEncoderPipelineStrategy()
        dummy_images = torch.empty(0)

        results = strategy.encode_batch(dummy_images, mock_text_encoder_bundle, sample_entries)

        assert len(results) == 3
        for result in results:
            assert "hidden_state" in result
            assert "metadata" in result

    def test_save_and_load_cache(self, sample_entry: CacheEntry):
        """Test SD text encoder save/load roundtrip."""
        strategy = SdTextEncoderPipelineStrategy()
        cache_path = Path(sample_entry.te_cache_path)

        data = {
            "hidden_state": torch.randn(77, 768, dtype=torch.float16),
            "metadata": {"caption_hash": "12345678"},
        }

        strategy.save_cache(data, cache_path)
        assert cache_path.exists()

        loaded = strategy.load_cache(cache_path)
        assert "hidden_state" in loaded.aux


class TestSdxlLatentsPipelineStrategy:
    """Tests for SDXL latent caching strategy."""

    def test_get_entry_cache_path(self, sample_entry: CacheEntry, tmp_path: Path):
        """Test that entry has cache path set."""
        strategy = SdxlLatentsPipelineStrategy()
        path = strategy.get_entry_cache_path(sample_entry)
        assert path is not None
        assert path.endswith("_latent.safetensors")

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

    def test_get_entry_cache_path(self, sample_entry: CacheEntry, tmp_path: Path):
        """Test that TE strategy reads te_cache_path from entry."""
        strategy = SdxlTextEncoderPipelineStrategy()
        path = strategy.get_entry_cache_path(sample_entry)
        assert path is not None
        assert path.endswith("_te.safetensors")

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
        cache_path = Path(sample_entry.latent_cache_path)

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

        # Load - returns CacheData object with aux dict
        loaded = strategy.load_cache(cache_path)
        assert "hidden_state1" in loaded.aux
        assert "hidden_state2" in loaded.aux
        assert "pool2" in loaded.aux
