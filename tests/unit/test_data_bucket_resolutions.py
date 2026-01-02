import pytest
from library.data.data_structures import make_bucket_resolutions


# =============================================================================
# make_bucket_resolutions Tests
# =============================================================================

@pytest.mark.unit
class TestMakeBucketResolutions:
    """Test bucket resolution generation for training."""
    
    def test_includes_square_bucket(self):
        """Should always include a square bucket."""
        resos = make_bucket_resolutions((512, 512))
        assert (512, 512) in resos
        
    def test_returns_sorted_list(self):
        """Result should be sorted."""
        resos = make_bucket_resolutions((512, 512))
        assert resos == sorted(resos)
        
    def test_respects_min_size(self):
        """All dimensions should be >= min_size."""
        resos = make_bucket_resolutions((512, 512), min_size=256, max_size=1024)
        for w, h in resos:
            assert w >= 256, f"Width {w} < min_size 256"
            assert h >= 256, f"Height {h} < min_size 256"
            
    def test_respects_max_size(self):
        """All dimensions should be <= max_size."""
        resos = make_bucket_resolutions((512, 512), min_size=256, max_size=768)
        for w, h in resos:
            assert w <= 768, f"Width {w} > max_size 768"
            assert h <= 768, f"Height {h} > max_size 768"
            
    def test_all_divisible_by_divisor(self):
        """All dimensions should be divisible by divisible param."""
        resos = make_bucket_resolutions((512, 512), divisible=64)
        for w, h in resos:
            assert w % 64 == 0, f"Width {w} not divisible by 64"
            assert h % 64 == 0, f"Height {h} not divisible by 64"
            
    def test_includes_symmetric_pairs(self):
        """Non-square buckets should have their transpose included."""
        resos = make_bucket_resolutions((512, 512))
        for w, h in resos:
            if w != h:
                assert (h, w) in resos, f"Missing symmetric pair for ({w}, {h})"
                
    def test_respects_max_area(self):
        """All buckets should respect max area constraint."""
        max_reso = (512, 768)
        max_area = 512 * 768
        resos = make_bucket_resolutions(max_reso)
        for w, h in resos:
            # Area should be at or close to max_area when dimensions fit
            assert w * h <= max_area * 1.1, f"Area {w*h} exceeds max area {max_area}"
