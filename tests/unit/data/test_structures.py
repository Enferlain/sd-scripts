"""
Unit tests for library/data/structures.py

Tests data class logic and properties.
"""

import pytest
from library.data.structures import Bucket, BatchInfo, EpochManifest, DatasetManifest, CacheEntry

@pytest.mark.unit
class TestBucket:
    def test_memory_per_image(self):
        """Test memory calculation for different dtypes and scales."""
        # 1024x1024 bucket
        bucket = Bucket(resolution=(1024, 1024))

        # SDXL defaults: 4 channels, scale 8, fp16 (2 bytes)
        # Latent: 1024/8 = 128 -> 128x128
        # Elements: 128 * 128 * 4 = 65536
        # Bytes: 65536 * 2 = 131072
        mem_fp16 = bucket.memory_per_image(latent_channels=4, latent_scale_factor=8, latent_dtype="fp16")
        assert mem_fp16 == 128 * 128 * 4 * 2

        # Flux: 16 channels, fp32 (4 bytes)
        mem_flux = bucket.memory_per_image(latent_channels=16, latent_scale_factor=8, latent_dtype="fp32")
        assert mem_flux == 128 * 128 * 16 * 4

    def test_count(self):
        bucket = Bucket(resolution=(512, 512), image_ids=["a", "b", "c"])
        assert bucket.count == 3


@pytest.mark.unit
class TestBatchInfo:
    def test_batch_size(self):
        batch = BatchInfo(image_ids=["a", "b"], bucket_reso=(512, 512))
        assert batch.batch_size == 2

    def test_get_sample_key_no_repeats(self):
        """Test sample key generation without repeats."""
        batch = BatchInfo(image_ids=["img1", "img2"], bucket_reso=(512, 512))
        # Default behavior: no repeat indices -> just image id
        assert batch.get_sample_key(0) == "img1"
        assert batch.get_sample_key(1) == "img2"

    def test_get_sample_key_with_repeats(self):
        """Test sample key generation with repeats."""
        # img1 repeated twice
        batch = BatchInfo(
            image_ids=["img1", "img1"],
            bucket_reso=(512, 512),
            repeat_indices=[0, 1]
        )
        assert batch.get_sample_key(0) == "img1"          # repeat 0 uses base ID
        assert batch.get_sample_key(1) == "img1#1"        # repeat 1 uses suffix


@pytest.mark.unit
class TestEpochManifest:
    def test_properties(self):
        b1 = BatchInfo(image_ids=["a", "b"], bucket_reso=(512, 512))
        b2 = BatchInfo(image_ids=["c"], bucket_reso=(512, 512))

        manifest = EpochManifest(epoch=1, seed=42, batches=[b1, b2])

        assert manifest.num_batches == 2
        assert manifest.num_images == 3


@pytest.mark.unit
class TestDatasetManifest:
    def test_lookup_methods(self):
        entry = CacheEntry(
            id="img1", image_path="path", original_size=(100,100),
            bucket_reso=(100,100), resized_size=(100,100), caption="cap"
        )
        bucket = Bucket(resolution=(100, 100), image_ids=["img1"])

        manifest = DatasetManifest(
            entries={"img1": entry},
            buckets={"100x100": bucket}
        )

        assert manifest.get_entry("img1") == entry
        assert manifest.get_entry("missing") is None

        assert manifest.get_bucket((100, 100)) == bucket
        assert manifest.get_bucket((200, 200)) is None

    def test_counts(self):
        e1 = CacheEntry(
            id="1", image_path="", original_size=(0,0), bucket_reso=(0,0), resized_size=(0,0), caption="cap1"
        )
        e2 = CacheEntry(
            id="2", image_path="", original_size=(0,0), bucket_reso=(0,0), resized_size=(0,0), caption=""
        )

        manifest = DatasetManifest(entries={"1": e1, "2": e2})

        assert manifest.image_count == 2
        assert manifest.caption_count == 1

    def test_bucket_summary(self):
        b1 = Bucket(resolution=(512, 512), image_ids=["a", "b"])
        b2 = Bucket(resolution=(1024, 1024), image_ids=["c"])

        manifest = DatasetManifest(buckets={"512x512": b1, "1024x1024": b2})

        summary = manifest.bucket_summary
        assert summary["512x512"] == 2
        assert summary["1024x1024"] == 1
