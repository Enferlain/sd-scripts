"""
Integration tests for the complete data pipeline.

Tests the flow: scan_directory → create_manifest → CachingEngine → strategy.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from library.data.pipeline.dataset_scanner import scan_directory, create_manifest
from library.data.pipeline.caching_engine import CachingEngine
from library.data.pipeline.dataclasses import DatasetManifest
from library.strategies.sd_caching import SdLatentsPipelineStrategy
from library.strategies.sdxl_caching import SdxlLatentsPipelineStrategy


# Path to test assets
TEST_IMAGES_DIR = Path(__file__).parent.parent / "assets" / "images"


@pytest.fixture
def mock_vae():
    """Create a mock VAE that returns properly shaped latents."""
    vae = MagicMock()
    vae.device = torch.device("cpu")
    vae.dtype = torch.float32

    def mock_encode(images):
        # Return latents at 1/8 resolution
        b, c, h, w = images.shape
        latent_h, latent_w = h // 8, w // 8
        mock_output = MagicMock()
        mock_output.latent_dist.sample.return_value = torch.randn(b, 4, latent_h, latent_w)
        return mock_output

    vae.encode = mock_encode
    return vae


@pytest.fixture
def mock_accelerator():
    """Create a mock accelerator for single-GPU testing."""
    accelerator = MagicMock()
    accelerator.process_index = 0
    accelerator.num_processes = 1
    accelerator.wait_for_everyone = MagicMock()
    return accelerator


class TestPipelineIntegration:
    """Integration tests for the data pipeline."""

    @pytest.mark.skipif(not TEST_IMAGES_DIR.exists(), reason="Test images not available")
    def test_scan_real_images(self):
        """Test scanning real test images directory."""
        scanned = scan_directory(str(TEST_IMAGES_DIR))

        assert len(scanned) == 5, f"Expected 5 images, found {len(scanned)}"

        for img in scanned:
            assert img.caption, f"Image {img.path} missing caption"
            assert img.width > 0
            assert img.height > 0

    @pytest.mark.skipif(not TEST_IMAGES_DIR.exists(), reason="Test images not available")
    def test_create_manifest_from_real_images(self):
        """Test creating manifest from real images."""
        scanned = scan_directory(str(TEST_IMAGES_DIR))

        manifest = create_manifest(
            scanned_images=scanned,
            base_resolution=(1024, 1024),
            bucket_reso_steps=64,
        )

        assert isinstance(manifest, DatasetManifest)
        assert len(manifest.entries) == 5
        assert len(manifest.buckets) > 0

    @pytest.mark.skipif(not TEST_IMAGES_DIR.exists(), reason="Test images not available")
    def test_sd_caching_integration(self, mock_vae, mock_accelerator, tmp_path):
        """Test full SD caching pipeline with real images."""
        # Step 1: Scan directory
        scanned = scan_directory(str(TEST_IMAGES_DIR))

        # Step 2: Create manifest
        manifest = create_manifest(
            scanned_images=scanned,
            base_resolution=(1024, 1024),
            bucket_reso_steps=64,
        )

        # Step 3: Create strategy and engine
        strategy = SdLatentsPipelineStrategy(dtype="fp32")
        engine = CachingEngine(strategy, batch_size=2, num_workers=2)

        # Step 4: Cache the dataset
        cache_dir = tmp_path / "cache"
        updated_manifest = engine.cache_dataset(
            manifest=manifest,
            model=mock_vae,
            accelerator=mock_accelerator,
            cache_dir=cache_dir,
            skip_existing=True,
            show_progress=False,
        )

        # Step 5: Verify cache files created
        cache_files = list(cache_dir.glob("*.safetensors"))
        assert len(cache_files) == 5, f"Expected 5 cache files, found {len(cache_files)}"

        # Verify entries have cache paths
        for entry in updated_manifest.entries.values():
            assert entry.latent_cache_path is not None
            assert Path(entry.latent_cache_path).exists()

    @pytest.mark.skipif(not TEST_IMAGES_DIR.exists(), reason="Test images not available")
    def test_sdxl_caching_integration(self, mock_vae, mock_accelerator, tmp_path):
        """Test full SDXL caching pipeline with real images."""
        # Step 1: Scan and manifest
        scanned = scan_directory(str(TEST_IMAGES_DIR))
        manifest = create_manifest(
            scanned_images=scanned,
            base_resolution=(1024, 1024),
            bucket_reso_steps=64,
        )

        # Step 2: Create SDXL strategy and engine
        strategy = SdxlLatentsPipelineStrategy(dtype="fp32")
        engine = CachingEngine(strategy, batch_size=2, num_workers=2)

        # Step 3: Cache
        cache_dir = tmp_path / "sdxl_cache"
        engine.cache_dataset(
            manifest=manifest,
            model=mock_vae,
            accelerator=mock_accelerator,
            cache_dir=cache_dir,
            show_progress=False,
        )

        # Step 4: Verify
        cache_files = list(cache_dir.glob("*_sdxl_latents.safetensors"))
        assert len(cache_files) == 5

    @pytest.mark.skipif(not TEST_IMAGES_DIR.exists(), reason="Test images not available")
    def test_skip_existing_caches(self, mock_vae, mock_accelerator, tmp_path):
        """Test that existing cache files are skipped."""
        scanned = scan_directory(str(TEST_IMAGES_DIR))
        manifest = create_manifest(scanned_images=scanned, base_resolution=(1024, 1024))

        strategy = SdLatentsPipelineStrategy(dtype="fp32")
        engine = CachingEngine(strategy, batch_size=2)
        cache_dir = tmp_path / "cache"

        # First run: create caches
        engine.cache_dataset(manifest, mock_vae, mock_accelerator, cache_dir, show_progress=False)

        # Get modification times
        cache_files = list(cache_dir.glob("*.safetensors"))
        mtimes_before = {f: f.stat().st_mtime for f in cache_files}

        # Second run: should skip existing
        engine.cache_dataset(manifest, mock_vae, mock_accelerator, cache_dir, show_progress=False)

        # Verify files were not modified
        for f in cache_files:
            assert f.stat().st_mtime == mtimes_before[f], f"{f.name} was modified on second run"

    @pytest.mark.skipif(not TEST_IMAGES_DIR.exists(), reason="Test images not available")
    def test_cache_can_be_loaded(self, mock_vae, mock_accelerator, tmp_path):
        """Test that cached files can be loaded back."""
        scanned = scan_directory(str(TEST_IMAGES_DIR))
        manifest = create_manifest(scanned_images=scanned, base_resolution=(1024, 1024))

        strategy = SdLatentsPipelineStrategy(dtype="fp32")
        engine = CachingEngine(strategy, batch_size=2)
        cache_dir = tmp_path / "cache"

        engine.cache_dataset(manifest, mock_vae, mock_accelerator, cache_dir, show_progress=False)

        # Load each cache file
        for entry in manifest.entries.values():
            cache_path = strategy.get_cache_path(entry, cache_dir)
            loaded = strategy.load_cache(cache_path)

            assert loaded.latents is not None
            assert loaded.latents.ndim == 3  # [C, H, W]
            assert loaded.latents.shape[0] == 4  # 4 latent channels
