"""
Integration tests with real VAE model.

These tests use actual model files from tests/assets/ to verify
the caching pipeline produces correct outputs, not just that it runs.

Marked with @pytest.mark.slow - exclude from fast CI runs.
"""

from pathlib import Path

import pytest
import torch

from library.data import scan_directory, create_manifest
from library.data.caching_engine import CachingEngine
from library.strategies.sdxl.caching import SdxlLatentsPipelineStrategy


# Paths to test assets
TEST_ASSETS_DIR = Path(__file__).parent.parent / "assets"
TEST_IMAGES_DIR = TEST_ASSETS_DIR / "images"
SDXL_VAE_PATH = TEST_ASSETS_DIR / "sdxl_vae.safetensors"


def load_sdxl_vae(device: str = "cpu") -> torch.nn.Module:
    """Load the real SDXL VAE from test assets."""
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_single_file(
        str(SDXL_VAE_PATH),
        torch_dtype=torch.float32,
    )
    vae = vae.to(device)
    vae.eval()
    return vae


class MockAccelerator:
    """Simple mock accelerator for single-process testing."""

    def __init__(self):
        self.process_index = 0
        self.num_processes = 1

    def wait_for_everyone(self):
        pass


@pytest.mark.slow
class TestRealVAEIntegration:
    """Integration tests using real VAE model."""

    @pytest.fixture
    def real_vae(self):
        """Load real SDXL VAE."""
        if not SDXL_VAE_PATH.exists():
            pytest.skip(f"SDXL VAE not found at {SDXL_VAE_PATH}")
        return load_sdxl_vae(device="cpu")

    @pytest.fixture
    def accelerator(self):
        return MockAccelerator()

    @pytest.mark.skipif(not TEST_IMAGES_DIR.exists(), reason="Test images not available")
    @pytest.mark.skipif(not SDXL_VAE_PATH.exists(), reason="SDXL VAE not available")
    def test_real_vae_encoding_produces_valid_latents(self, real_vae, accelerator, tmp_path):
        """
        Test that encoding with real VAE produces valid latent tensors.

        This verifies:
        - Latents have correct shape (4 channels, 1/8 resolution)
        - Latents have reasonable statistics (not all zeros, not NaN)
        - Latents are scaled correctly
        """
        # Scan just one image for speed
        scanned = scan_directory(str(TEST_IMAGES_DIR))
        scanned = scanned[:1]  # Just first image

        cache_dir = tmp_path / "cache"
        manifest = create_manifest(
            scanned_images=scanned,
            base_resolution=(1024, 1024),
            bucket_reso_steps=64,
            cache_dir=str(cache_dir),
        )

        strategy = SdxlLatentsPipelineStrategy(dtype="fp32")
        engine = CachingEngine(strategy, batch_size=1, num_workers=1)
        engine.cache_dataset(
            manifest=manifest,
            model=real_vae,
            accelerator=accelerator,
            cache_dir=cache_dir,
            show_progress=False,
        )

        # Load the cached latent (path was set by caching engine)
        entry = list(manifest.entries.values())[0]
        cache_path = Path(entry.latent_cache_path)
        loaded = strategy.load_cache(cache_path)

        latents = loaded.latents

        # Verify shape: [4, H/8, W/8]
        assert latents.ndim == 3, f"Expected 3D tensor, got {latents.ndim}D"
        assert latents.shape[0] == 4, f"Expected 4 channels, got {latents.shape[0]}"

        # Bucket resolution was assigned during manifest creation
        bucket_w, bucket_h = entry.bucket_reso
        expected_latent_h = bucket_h // 8
        expected_latent_w = bucket_w // 8
        assert latents.shape[1] == expected_latent_h, f"Height mismatch: {latents.shape[1]} vs {expected_latent_h}"
        assert latents.shape[2] == expected_latent_w, f"Width mismatch: {latents.shape[2]} vs {expected_latent_w}"

        # Verify values are valid (not NaN or all zeros)
        assert not torch.isnan(latents).any(), "Latents contain NaN values"
        assert not torch.isinf(latents).any(), "Latents contain Inf values"
        assert latents.abs().sum() > 0, "Latents are all zeros"

        # Verify reasonable statistics (VAE latents typically have std around 1-5 after scaling)
        std = latents.std().item()
        assert 0.1 < std < 20, f"Latent std={std} seems unreasonable"

    @pytest.mark.skipif(not TEST_IMAGES_DIR.exists(), reason="Test images not available")
    @pytest.mark.skipif(not SDXL_VAE_PATH.exists(), reason="SDXL VAE not available")
    def test_different_images_produce_different_latents(self, real_vae, accelerator, tmp_path):
        """
        Test that different images produce different latent tensors.

        This catches bugs where preprocessing or encoding ignores the input.
        """
        scanned = scan_directory(str(TEST_IMAGES_DIR))
        scanned = scanned[:2]  # Two different images

        cache_dir = tmp_path / "cache"
        manifest = create_manifest(
            scanned_images=scanned,
            base_resolution=(1024, 1024),
            bucket_reso_steps=64,
            cache_dir=str(cache_dir),
        )

        strategy = SdxlLatentsPipelineStrategy(dtype="fp32")
        engine = CachingEngine(strategy, batch_size=1, num_workers=1)
        engine.cache_dataset(manifest, real_vae, accelerator, cache_dir, show_progress=False)

        # Load both latents (paths were set by caching engine)
        entries = list(manifest.entries.values())
        latents1 = strategy.load_cache(Path(entries[0].latent_cache_path)).latents
        latents2 = strategy.load_cache(Path(entries[1].latent_cache_path)).latents

        # They should be different (unless the images happen to be identical)
        # Use a tolerance - they won't be exactly equal even for similar images
        if latents1.shape == latents2.shape:
            diff = (latents1 - latents2).abs().mean().item()
            assert diff > 0.01, f"Latents are suspiciously similar (mean diff={diff})"

    @pytest.mark.skipif(not TEST_IMAGES_DIR.exists(), reason="Test images not available")
    @pytest.mark.skipif(not SDXL_VAE_PATH.exists(), reason="SDXL VAE not available")
    def test_encoding_consistency(self, real_vae, accelerator, tmp_path):
        """
        Test that encoding produces consistent (though not bit-exact) latents.

        Note: VAE.encode().latent_dist.sample() is stochastic by design.
        Each call samples from the Gaussian, so results differ slightly.
        For cached latents this is fine - we cache once and reuse.
        """
        scanned = scan_directory(str(TEST_IMAGES_DIR))
        scanned = scanned[:1]

        manifest = create_manifest(scanned_images=scanned, base_resolution=(1024, 1024))

        strategy = SdxlLatentsPipelineStrategy(dtype="fp32")
        engine = CachingEngine(strategy, batch_size=1, num_workers=1)

        # First run - set paths for cache_dir1
        cache_dir1 = tmp_path / "cache1"
        for entry in manifest.entries.values():
            entry.latent_cache_path = str(cache_dir1 / f"{entry.id}_latent.safetensors")
        engine.cache_dataset(manifest, real_vae, accelerator, cache_dir1, show_progress=False)

        # Store first run's latent path
        entry = list(manifest.entries.values())[0]
        latent_path1 = Path(entry.latent_cache_path)

        # Second run - set paths for cache_dir2
        cache_dir2 = tmp_path / "cache2"
        for entry in manifest.entries.values():
            entry.latent_cache_path = str(cache_dir2 / f"{entry.id}_latent.safetensors")
        engine.cache_dataset(manifest, real_vae, accelerator, cache_dir2, show_progress=False)

        # Load and compare
        entry = list(manifest.entries.values())[0]
        latent_path2 = Path(entry.latent_cache_path)

        latents1 = strategy.load_cache(latent_path1).latents
        latents2 = strategy.load_cache(latent_path2).latents

        # VAE sample() is stochastic, so we expect small differences
        # Typical diff is ~0.0003, we allow up to 0.01 (still very similar)
        diff = (latents1 - latents2).abs().max().item()
        assert diff < 0.01, f"Latents differ too much: max diff={diff}"

        # But they should be highly correlated (same structure, tiny noise)
        correlation = torch.corrcoef(torch.stack([latents1.flatten(), latents2.flatten()]))[0, 1]
        assert correlation > 0.999, f"Latents not correlated enough: r={correlation}"
