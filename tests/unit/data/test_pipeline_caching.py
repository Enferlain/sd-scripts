"""
Unit tests for the caching engine module.

Tests Phase 2 of the data pipeline: caching latents and text encoder outputs.
"""

import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock

import torch
from PIL import Image

from library.data.pipeline.caching_engine import CachingEngine, CachingStrategy
from library.data.pipeline.dataclasses import CacheEntry, DatasetManifest, Bucket


class MockCachingStrategy(CachingStrategy):
    """Mock strategy for testing."""

    def __init__(self):
        self.encode_calls = []
        self.save_calls = []

    def get_entry_cache_path(self, entry: CacheEntry) -> str | None:
        """Return pre-set latent cache path from entry."""
        return entry.latent_cache_path

    def encode_batch(self, images, model, entries):
        self.encode_calls.append((images.shape, len(entries)))
        # Return fake encoded data for each entry
        return [{"latents": torch.randn(4, 64, 64)} for _ in entries]

    def save_cache(self, data, path):
        self.save_calls.append(path)
        # Create empty file to simulate saving
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def load_cache(self, path):
        return {"latents": torch.randn(4, 64, 64)}

    def is_cache_valid(self, path, entry, flip_aug=False, alpha_mask=False):
        # For tests using skip_validity_check=True, this won't be called
        # For basic testing, just return True if file exists
        return path.exists()


@pytest.fixture
def mock_accelerator():
    """Create a mock accelerator for single-GPU testing."""
    acc = Mock()
    acc.process_index = 0
    acc.num_processes = 1
    acc.wait_for_everyone = Mock()
    return acc


@pytest.fixture
def temp_dataset_dir():
    """Create a temporary directory with test images and manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        images_dir = tmpdir / "images"
        images_dir.mkdir()

        # Create test images
        for i in range(5):
            img = Image.new("RGB", (512, 512), color=(i * 50, i * 50, i * 50))
            img.save(images_dir / f"image_{i:03d}.png")

        yield tmpdir


@pytest.fixture
def sample_manifest(temp_dataset_dir):
    """Create a sample manifest with entries (cache paths pre-set)."""
    images_dir = temp_dataset_dir / "images"
    cache_dir = temp_dataset_dir / "cache"

    entries = {}
    for i in range(5):
        entry = CacheEntry(
            id=f"img_{i:03d}",
            image_path=str(images_dir / f"image_{i:03d}.png"),
            original_size=(512, 512),
            bucket_reso=(512, 512),
            resized_size=(512, 512),
            caption=f"test caption {i}",
            latent_cache_path=str(cache_dir / f"img_{i:03d}.safetensors"),
        )
        entries[entry.id] = entry

    buckets = {
        "512x512": Bucket(
            resolution=(512, 512),
            image_ids=list(entries.keys()),
        )
    }

    return DatasetManifest(
        entries=entries,
        buckets=buckets,
        base_resolution=(512, 512),
        cache_dir=str(cache_dir),
    )


# =============================================================================
# CachingEngine Tests
# =============================================================================


@pytest.mark.unit
class TestCachingEngine:
    """Test CachingEngine functionality."""

    def test_batch_entries_by_bucket(self, sample_manifest):
        """Should group entries by bucket resolution."""
        strategy = MockCachingStrategy()
        engine = CachingEngine(strategy, batch_size=2)

        entries = list(sample_manifest.entries.values())
        batches = engine._batch_entries_by_bucket(entries)

        assert "512x512" in batches
        # 5 entries with batch_size=2 = 3 batches (2+2+1)
        assert len(batches["512x512"]) == 3

    def test_split_for_single_gpu(self, sample_manifest, mock_accelerator):
        """Single GPU should get all entries."""
        strategy = MockCachingStrategy()
        engine = CachingEngine(strategy)

        entries = list(sample_manifest.entries.values())
        my_entries = engine._split_for_gpu(entries, mock_accelerator)

        assert len(my_entries) == len(entries)

    def test_split_for_multi_gpu(self, sample_manifest):
        """Multi-GPU should split entries by modulo."""
        strategy = MockCachingStrategy()
        engine = CachingEngine(strategy)

        entries = list(sample_manifest.entries.values())

        # Simulate 2 GPUs
        acc0 = Mock(process_index=0, num_processes=2)
        acc1 = Mock(process_index=1, num_processes=2)

        entries_gpu0 = engine._split_for_gpu(entries, acc0)
        entries_gpu1 = engine._split_for_gpu(entries, acc1)

        # Should split evenly (5 entries = 3 + 2)
        assert len(entries_gpu0) + len(entries_gpu1) == len(entries)
        assert len(entries_gpu0) == 3
        assert len(entries_gpu1) == 2

    def test_cache_dataset_creates_files(self, sample_manifest, mock_accelerator, temp_dataset_dir):
        """Should create cache files for all entries."""
        strategy = MockCachingStrategy()
        engine = CachingEngine(strategy, batch_size=2)

        cache_dir = temp_dataset_dir / "cache"
        mock_model = Mock()

        engine.cache_dataset(
            sample_manifest,
            mock_model,
            mock_accelerator,
            cache_dir,
            show_progress=False,
        )

        # Check that strategy was called
        assert len(strategy.encode_calls) == 3  # 3 batches
        assert len(strategy.save_calls) == 5  # 5 entries

        # Check cache files exist
        for entry in sample_manifest.entries.values():
            assert entry.latent_cache_path is not None
            assert Path(entry.latent_cache_path).exists()

    def test_skip_existing_caches(self, sample_manifest, mock_accelerator, temp_dataset_dir):
        """Should skip entries with existing cache files."""
        strategy = MockCachingStrategy()
        engine = CachingEngine(strategy, batch_size=2)

        cache_dir = temp_dataset_dir / "cache"
        cache_dir.mkdir()
        mock_model = Mock()

        # Pre-create cache for first 2 entries (paths already set on entries)
        for entry in list(sample_manifest.entries.values())[:2]:
            cache_path = Path(entry.latent_cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.touch()

        engine.cache_dataset(
            sample_manifest,
            mock_model,
            mock_accelerator,
            cache_dir,
            skip_existing=True,
            show_progress=False,
        )

        # Only 3 new entries should be cached
        assert len(strategy.save_calls) == 3

    def test_all_cached_returns_early(self, sample_manifest, mock_accelerator, temp_dataset_dir):
        """Should return early if all entries are already cached."""
        strategy = MockCachingStrategy()
        engine = CachingEngine(strategy)

        cache_dir = temp_dataset_dir / "cache"
        cache_dir.mkdir()
        mock_model = Mock()

        # Pre-create cache for all entries (paths already set on entries)
        for entry in sample_manifest.entries.values():
            cache_path = Path(entry.latent_cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.touch()

        engine.cache_dataset(
            sample_manifest,
            mock_model,
            mock_accelerator,
            cache_dir,
            show_progress=False,
        )

        # No encoding should happen
        assert len(strategy.encode_calls) == 0

    def test_cache_invalidation_bucket_change(self, sample_manifest, mock_accelerator, temp_dataset_dir):
        """Should re-cache when cache is invalid (e.g. bucket change)."""
        strategy = MockCachingStrategy()

        # Override is_cache_valid to fail for one entry
        def side_effect(path, entry, flip_aug=False, alpha_mask=False):
            if entry.id == "img_000":
                return False
            return path.exists()

        strategy.is_cache_valid = side_effect  # type: ignore[method-assign]

        engine = CachingEngine(strategy, batch_size=2)

        cache_dir = temp_dataset_dir / "cache"
        cache_dir.mkdir()
        mock_model = Mock()

        # Pre-create cache for all entries (paths already set on entries)
        for entry in sample_manifest.entries.values():
            cache_path = Path(entry.latent_cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.touch()

        engine.cache_dataset(
            sample_manifest,
            mock_model,
            mock_accelerator,
            cache_dir,
            skip_existing=True,
            skip_validity_check=False,  # Must be False to trigger check
            show_progress=False,
        )

        # img_000 should be re-cached (saved again)
        # Check that save_cache was called exactly once
        assert len(strategy.save_calls) == 1
        assert "img_000.safetensors" in str(strategy.save_calls[0])
