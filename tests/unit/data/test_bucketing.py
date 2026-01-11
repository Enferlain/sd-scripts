"""
Unit tests for the bucketing module.

Tests bucket resolution generation and bucket selection logic.
"""

import pytest
from library.data.bucketing import make_bucket_resolutions, select_bucket


# =============================================================================
# make_bucket_resolutions Tests
# =============================================================================


@pytest.mark.unit
class TestMakeBucketResolutions:
    """Test bucket resolution generation."""

    def test_includes_square_bucket(self):
        """Should include a square bucket."""
        resos = make_bucket_resolutions((1024, 1024))
        # Should have a 1024x1024 bucket
        assert (1024, 1024) in resos

    def test_returns_sorted_list(self):
        """Result should be sorted."""
        resos = make_bucket_resolutions((1024, 1024))
        assert resos == sorted(resos)

    def test_includes_symmetric_pairs(self):
        """Non-square buckets should have their transpose included."""
        resos = make_bucket_resolutions((1024, 1024))
        for w, h in resos:
            if w != h:
                assert (h, w) in resos, f"Missing symmetric pair for ({w}, {h})"

    def test_respects_divisibility(self):
        """All dimensions should be divisible by divisible param."""
        resos = make_bucket_resolutions((1024, 1024), divisible=64)
        for w, h in resos:
            assert w % 64 == 0, f"Width {w} not divisible by 64"
            assert h % 64 == 0, f"Height {h} not divisible by 64"


# =============================================================================
# select_bucket Tests
# =============================================================================


@pytest.mark.unit
class TestSelectBucket:
    """Test bucket selection for images."""

    def test_square_image_gets_square_bucket(self):
        """Square image should get a square or near-square bucket."""
        resos = make_bucket_resolutions((1024, 1024))
        bucket, resized = select_bucket(1024, 1024, resos)
        assert bucket[0] == bucket[1]  # Square bucket

    def test_landscape_image_gets_landscape_bucket(self):
        """Landscape image should get a landscape bucket."""
        resos = make_bucket_resolutions((1024, 1024))
        bucket, resized = select_bucket(1920, 1080, resos)
        assert bucket[0] > bucket[1]  # Width > height

    def test_portrait_image_gets_portrait_bucket(self):
        """Portrait image should get a portrait bucket."""
        resos = make_bucket_resolutions((1024, 1024))
        bucket, resized = select_bucket(1080, 1920, resos)
        assert bucket[0] < bucket[1]  # Width < height

    def test_no_upscale_mode(self):
        """no_upscale mode should not enlarge small images."""
        resos = make_bucket_resolutions((1024, 1024))
        # Small image that would need upscaling
        bucket, resized = select_bucket(256, 256, resos, no_upscale=True, max_area=1024 * 1024)
        # Bucket should be <= original size (rounded to steps)
        assert bucket[0] <= 256
        assert bucket[1] <= 256
