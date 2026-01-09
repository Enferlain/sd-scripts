"""
Benchmark tests for the data pipeline.

These tests measure performance characteristics of prepare_epoch and DataLoader.
Run with: pytest tests/unit/data/test_pipeline_benchmark.py -v -s
"""

import time
import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock

from library.data.structures import DatasetManifest, CacheEntry
from library.data.epoch_preparation import prepare_epoch
from library.data.dataloader import create_training_dataloader
from library.data.caching_engine import CachingStrategy


class MockCachingStrategy(CachingStrategy):
    """Mock strategy that returns dummy tensors without disk I/O."""

    def __init__(self, latent_dim=(4, 64, 64)):
        self.latent_dim = latent_dim

    def load_cache(self, path):
        mock_data = MagicMock()
        mock_data.latents = torch.randn(self.latent_dim)
        mock_data.latents_flipped = torch.randn(self.latent_dim)
        mock_data.alpha_mask = None
        mock_data.conditioning = MagicMock()
        mock_data.conditioning.original_size_hw = (1024, 1024)
        mock_data.conditioning.crop_top_left = (0, 0)
        mock_data.conditioning.target_size_hw = (1024, 1024)
        mock_data.aux = {"hidden_state1": torch.randn(1, 77, 768)}
        return mock_data

    def get_entry_cache_path(self, entry):
        return Path("/tmp/mock")

    def is_cache_valid(self, *args, **kwargs):
        return True

    def encode_batch(self, *args, **kwargs):
        return []

    def save_cache(self, *args, **kwargs):
        pass


@pytest.fixture
def manifest_1k():
    """Generate a 1k image manifest."""
    return _generate_manifest(1_000)


@pytest.fixture
def manifest_10k():
    """Generate a 10k image manifest."""
    return _generate_manifest(10_000)


def _generate_manifest(num_images: int) -> DatasetManifest:
    """Generate a synthetic dataset manifest."""
    entries = {}
    buckets = [(1024, 1024), (768, 1024), (1024, 768)]

    for i in range(num_images):
        img_id = f"img_{i:06d}"
        bucket = buckets[i % len(buckets)]
        entry = CacheEntry(
            id=img_id,
            image_path=f"/tmp/dataset/{img_id}.png",
            original_size=bucket,
            bucket_reso=bucket,
            resized_size=bucket,
            caption=f"a photo of image {i}, highly detailed, 8k",
            num_repeats=1,
            is_reg=False,
            split="train",
            latent_cache_path=f"/tmp/cache/{img_id}.safetensors",
            te_cache_path=f"/tmp/cache/{img_id}_te.safetensors",
            has_flipped=True,
            has_alpha_mask=False,
        )
        entries[img_id] = entry

    return DatasetManifest(entries=entries)


@pytest.mark.benchmark
class TestPipelineBenchmark:
    """Benchmark tests for data pipeline performance."""

    def test_prepare_epoch_1k(self, manifest_1k):
        """prepare_epoch should complete in <100ms for 1k images."""
        start = time.perf_counter()
        epoch_manifest = prepare_epoch(manifest=manifest_1k, epoch=1, seed=42, batch_size=4, shuffle=True)
        duration = time.perf_counter() - start

        assert duration < 0.1, f"prepare_epoch took {duration:.3f}s (expected <0.1s)"
        assert epoch_manifest.num_batches > 0

    def test_prepare_epoch_10k(self, manifest_10k):
        """prepare_epoch should complete in <500ms for 10k images."""
        start = time.perf_counter()
        epoch_manifest = prepare_epoch(manifest=manifest_10k, epoch=1, seed=42, batch_size=4, shuffle=True)
        duration = time.perf_counter() - start

        assert duration < 0.5, f"prepare_epoch took {duration:.3f}s (expected <0.5s)"
        assert epoch_manifest.num_batches > 0

    def test_dataloader_throughput(self, manifest_1k):
        """DataLoader should achieve >50 batches/second with mocked I/O."""
        epoch_manifest = prepare_epoch(manifest=manifest_1k, epoch=1, seed=42, batch_size=4)

        latent_strategy = MockCachingStrategy()
        te_strategy = MockCachingStrategy()

        dataloader = create_training_dataloader(
            dataset_manifest=manifest_1k,
            epoch_manifest=epoch_manifest,
            latent_strategy=latent_strategy,
            te_strategy=te_strategy,
            num_workers=0,
        )

        # Time 50 batches
        iterator = iter(dataloader)
        start = time.perf_counter()
        count = 0
        for _ in range(50):
            try:
                next(iterator)
                count += 1
            except StopIteration:
                break
        duration = time.perf_counter() - start

        throughput = count / duration if duration > 0 else 0
        assert throughput > 50, f"Throughput {throughput:.1f} batch/s (expected >50)"

    def test_first_batch_latency(self, manifest_1k):
        """First batch should be available in <50ms."""
        epoch_manifest = prepare_epoch(manifest=manifest_1k, epoch=1, seed=42, batch_size=4)

        latent_strategy = MockCachingStrategy()
        dataloader = create_training_dataloader(
            dataset_manifest=manifest_1k,
            epoch_manifest=epoch_manifest,
            latent_strategy=latent_strategy,
            num_workers=0,
        )

        start = time.perf_counter()
        batch = next(iter(dataloader))
        latency = time.perf_counter() - start

        assert latency < 0.05, f"First batch latency {latency:.3f}s (expected <0.05s)"
        assert "latents" in batch
