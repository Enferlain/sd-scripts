import pytest
import numpy as np
from library.data import data_structures as ds

# ============================================================================
# BucketManager Tests
# ============================================================================


class TestBucketManager:
    @pytest.fixture
    def manager(self):
        # Default manageable setup
        # max_reso (512,512), min_size 256, max_size 1024, step 64
        bm = ds.BucketManager(no_upscale=False, max_reso=(512, 512), min_size=256, max_size=1024, reso_steps=64)
        # Avoid model_util dependency by setting predefined directly
        bm.set_predefined_resos([(512, 512), (256, 512), (512, 256)])
        return bm

    def test_init_assertions(self):
        # max_size < max_reso
        with pytest.raises(AssertionError):
            ds.BucketManager(False, (512, 512), 256, 400, 64)

        # max_size < min_size
        with pytest.raises(AssertionError):
            ds.BucketManager(False, (512, 512), 1200, 1024, 64)

    def test_round_to_steps(self, manager):
        # "round to int, then floor to step"
        assert manager.round_to_steps(64) == 64
        assert manager.round_to_steps(65) == 64
        assert manager.round_to_steps(95) == 64
        assert manager.round_to_steps(127) == 64
        assert manager.round_to_steps(128) == 128

    def test_select_bucket_no_upscale_false(self, manager):
        # Input: 512x512. Should match (512,512) bucket exact.
        reso, resized, ar_err = manager.select_bucket(512, 512)
        assert reso == (512, 512)
        assert resized == (512, 512)
        assert ar_err == 0.0

        # Input: 256x256. Should upscale to (512,512) bucket.
        reso, resized, ar_err = manager.select_bucket(256, 256)
        assert reso == (512, 512)
        assert resized == (512, 512)

        # Input: 100x200 (1:2). Matches (256, 512)
        reso, resized, ar_err = manager.select_bucket(100, 200)
        assert reso == (256, 512)
        # Scale: 200->512 (2.56x). 100->256.
        assert resized == (256, 512)

    def test_select_bucket_no_upscale_true(self):
        bm = ds.BucketManager(no_upscale=True, max_reso=(512, 512), min_size=256, max_size=1024, reso_steps=64)

        # Case 1: Large image, needs downscale
        # 2048x2048. Max area 512*512=262144. Sqrt=512.
        reso, resized, error = bm.select_bucket(2048, 2048)
        assert reso == (512, 512)
        assert resized == (512, 512)

        # Case 2: Small image, no upscale
        # 300x300.
        # 300 - 300%64 = 256.
        reso, resized, error = bm.select_bucket(300, 300)
        assert resized == (300, 300)
        assert reso == (256, 256)

    def test_get_crop_ltrb(self):
        # Exact match
        l, t, r, b = ds.BucketManager.get_crop_ltrb((512, 512), (512, 512))
        assert (l, t, r, b) == (0, 0, 512, 512)

        # Case 1: Image wider than bucket (needs horizontal crop)
        # Bucket: 512x512 (AR=1). Image: 1024x512 (AR=2).
        # bucket_ar (1) < image_ar (2). Else branch.
        l, t, r, b = ds.BucketManager.get_crop_ltrb((512, 512), (1024, 512))
        assert (l, t, r, b) == (0, 128, 512, 384)
        assert t == (512 - (b - t)) // 2  # Centered vertically

        # Case 2: Bucket wider than image (Portrait image in Square bucket)
        # Bucket: 512x512 (AR=1). Image: 512x1024 (AR=0.5).
        # bucket_ar (1) > image_ar (0.5). If branch.
        # resized_width = bucket_h * image_ar = 512 * 0.5 = 256.
        # resized_height = bucket_h = 512.
        # crop_left = (512 - 256) // 2 = 128.
        # crop_top = (512 - 512) // 2 = 0.
        l, t, r, b = ds.BucketManager.get_crop_ltrb((512, 512), (512, 1024))

        # Verify
        # Width should be 256, Height 512.
        assert r - l == 256
        assert b - t == 512
        # Should be centered horizontally
        assert l == 128
        assert t == 0
        assert r == 384
        assert b == 512

    def test_select_bucket_registers_new_reso(self):
        # Test that select_bucket with no_upscale=True adds new resolutions to internal state
        bm = ds.BucketManager(no_upscale=True, max_reso=(512, 512), min_size=256, max_size=1024, reso_steps=64)

        # Initial state checks
        assert len(bm.resos) == 0

        # Select for odd size: 301x301
        # 301 - 301%64 = 256. Reso becomes (256, 256).
        reso, _, _ = bm.select_bucket(301, 301)

        assert reso == (256, 256)

        # Verify side effects
        assert (256, 256) in bm.resos
        assert (256, 256) in bm.reso_to_id
        assert len(bm.buckets) == 1  # A new bucket list created

    def test_add_image_invalid_reso(self, manager):
        # Case 3: Adding image with unregistered resolution should raise KeyError
        with pytest.raises(KeyError):
            manager.add_image((123, 456), "img_invalid")

    def test_bucket_sort_and_shuffle(self, manager, monkeypatch):
        # Add some dummy images to predefined buckets
        r1 = (512, 512)
        r2 = (256, 256)

        manager.add_if_new_reso(r1)  # Id 0
        manager.add_if_new_reso(r2)  # Id 1

        # Add items
        manager.add_image(r1, "img1")
        manager.add_image(r1, "img2")
        manager.add_image(r2, "img3")

        # Test Shuffle: Mock random.shuffle to be deterministic (reverse)
        def mock_shuffle(x):
            x.reverse()

        monkeypatch.setattr(ds.random, "shuffle", mock_shuffle)

        manager.shuffle()

        b1 = manager.buckets[manager.reso_to_id[r1]]
        # Original append order: img1, img2. Reversed: img2, img1.
        assert b1 == ["img2", "img1"]
        assert len(b1) == 2

        # Test Sort: Should sort resos and reorder buckets
        manager.sort()

        assert manager.resos[0] == (256, 256)
        assert manager.resos[1] == (512, 512)

        # Check mapping updated
        assert manager.reso_to_id[(256, 256)] == 0
        assert manager.reso_to_id[(512, 512)] == 1

        # Check buckets moved correctly
        assert manager.buckets[0] == ["img3"]
        assert manager.buckets[1] == ["img2", "img1"]  # Preserves previous shuffled state

    def test_make_buckets_calls_make_bucket_resolutions(self):
        """make_buckets should call make_bucket_resolutions and set predefined resos."""
        bm = ds.BucketManager(no_upscale=False, max_reso=(512, 512), min_size=256, max_size=1024, reso_steps=64)
        bm.make_buckets()

        # Verify predefined_resos were set (using real make_bucket_resolutions)
        assert len(bm.predefined_resos) > 0
        assert (512, 512) in bm.predefined_resos  # Square bucket should exist
        # All resos should be divisible by 64
        for w, h in bm.predefined_resos:
            assert w % 64 == 0
            assert h % 64 == 0


# ============================================================================
# AugHelper Tests
# ============================================================================


class TestAugHelper:
    def test_color_aug_skip(self, monkeypatch):
        pytest.importorskip("cv2")
        helper = ds.AugHelper()
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        # Random > 0.33, verify no change
        monkeypatch.setattr(ds.random, "random", lambda: 0.5)
        res = helper.color_aug(img)
        assert res["image"] is img  # Should be identity

    def test_color_aug_hue(self, monkeypatch):
        pytest.importorskip("cv2")
        helper = ds.AugHelper()
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        # Force color aug (<=0.33) and hue shift (>0.5)
        # random calls: 1. check aug (0.3), 2. check hue vs gamma (0.6)
        random_vals = [0.3, 0.6]

        def mock_random():
            if random_vals:
                return random_vals.pop(0)
            return 0.5

        monkeypatch.setattr(ds.random, "random", mock_random)
        monkeypatch.setattr(ds.random, "uniform", lambda a, b: 5.0)

        res = helper.color_aug(img)
        assert res["image"].shape == (100, 100, 3)

    def test_color_aug_gamma(self, monkeypatch):
        pytest.importorskip("cv2")
        helper = ds.AugHelper()
        img = np.full((100, 100, 3), 100, dtype=np.uint8)  # Gray

        # Force color aug (<=0.33) and gamma (<=0.5)
        random_vals = [0.3, 0.2]

        def mock_random():
            if random_vals:
                return random_vals.pop(0)
            return 0.9

        monkeypatch.setattr(ds.random, "random", mock_random)

        # Use realistic gamma 1.05 (brighter)
        # x^1.05 > x for x > 1
        monkeypatch.setattr(ds.random, "uniform", lambda a, b: 1.05)

        res = helper.color_aug(img)
        out_img = res["image"]

        assert out_img.shape == (100, 100, 3)
        assert out_img.dtype == np.uint8
        # Verify values changed: 100^1.05 approx 125 > 100
        assert out_img[0, 0, 0] > 100
        assert out_img[0, 0, 0] == 125  # 100**1.05 = 125.89 -> 125 (int cast in python? no, np.clip(..).astype)


# ============================================================================
# Subset Tests
# ============================================================================


def make_subset_args(**kwargs):
    defaults = dict(
        image_dir="/tmp/img",
        is_reg=False,
        class_tokens="dog",
        caption_extension=".txt",
        metadata_file="/tmp/meta.json",
        conditioning_data_dir="/tmp/cond",
        cache_info=True,
        alpha_mask=False,
        num_repeats=1,
        shuffle_caption=False,
        caption_separator=",",
        keep_tokens=0,
        keep_tokens_separator="",
        secondary_separator=None,
        enable_wildcard=False,
        color_aug=False,
        flip_aug=False,
        face_crop_aug_range=None,
        random_crop=False,
        random_crop_padding_percent=0.0,
        caption_dropout_rate=0.0,
        caption_dropout_every_n_epochs=0,
        caption_tag_dropout_rate=0.0,
        caption_prefix=None,
        caption_suffix=None,
        token_warmup_min=0,
        token_warmup_step=0,
    )
    # Filter for specific class args if needed, but python kwargs handle extras if careful
    # We'll just construct individually to be safe
    return defaults


def test_subset_initialization_normalization():
    # Caption extension normalization "txt" -> ".txt"
    sub = ds.DreamBoothSubset(
        image_dir="/tmp/img",
        is_reg=False,
        class_tokens="dog",
        caption_extension="txt",
        cache_info=True,
        alpha_mask=False,
        num_repeats=1,
        shuffle_caption=False,
        caption_separator=",",
        keep_tokens=0,
        keep_tokens_separator="",
        secondary_separator=None,
        enable_wildcard=False,
        color_aug=False,
        flip_aug=False,
        face_crop_aug_range=None,
        random_crop=False,
        random_crop_padding_percent=0.0,
        caption_dropout_rate=0.0,
        caption_dropout_every_n_epochs=0,
        caption_tag_dropout_rate=0.0,
        caption_prefix=None,
        caption_suffix=None,
        token_warmup_min=0,
        token_warmup_step=0,
    )
    assert sub.caption_extension == ".txt"


def test_dreambooth_subset_equality():
    # Equality depends only on image_dir
    kwargs = dict(
        is_reg=False,
        class_tokens="dog",
        caption_extension=".txt",
        cache_info=True,
        alpha_mask=False,
        num_repeats=1,
        shuffle_caption=False,
        caption_separator=",",
        keep_tokens=0,
        keep_tokens_separator="",
        secondary_separator=None,
        enable_wildcard=False,
        color_aug=False,
        flip_aug=False,
        face_crop_aug_range=None,
        random_crop=False,
        random_crop_padding_percent=0.0,
        caption_dropout_rate=0.0,
        caption_dropout_every_n_epochs=0,
        caption_tag_dropout_rate=0.0,
        caption_prefix=None,
        caption_suffix=None,
        token_warmup_min=0,
        token_warmup_step=0,
    )
    s1 = ds.DreamBoothSubset(image_dir="/dir/A", **kwargs)
    s2 = ds.DreamBoothSubset(image_dir="/dir/A", **kwargs)
    s3 = ds.DreamBoothSubset(image_dir="/dir/B", **kwargs)

    assert s1 == s2
    assert s1 != s3


def test_finetuning_subset_equality():
    # Equality depends only on metadata_file
    kwargs = dict(
        image_dir="/tmp/img",
        alpha_mask=False,
        num_repeats=1,
        shuffle_caption=False,
        caption_separator=",",
        keep_tokens=0,
        keep_tokens_separator="",
        secondary_separator=None,
        enable_wildcard=False,
        color_aug=False,
        flip_aug=False,
        face_crop_aug_range=None,
        random_crop=False,
        random_crop_padding_percent=0.0,
        caption_dropout_rate=0.0,
        caption_dropout_every_n_epochs=0,
        caption_tag_dropout_rate=0.0,
        caption_prefix=None,
        caption_suffix=None,
        token_warmup_min=0,
        token_warmup_step=0,
    )
    s1 = ds.FineTuningSubset(metadata_file="/meta.json", **kwargs)
    s2 = ds.FineTuningSubset(metadata_file="/meta.json", **kwargs)
    s3 = ds.FineTuningSubset(metadata_file="/other.json", **kwargs)

    assert s1 == s2
    assert s1 != s3


def test_controlnet_subset_equality():
    # Equality depends on image_dir AND conditioning_data_dir
    kwargs = dict(
        caption_extension=".txt",
        cache_info=True,
        num_repeats=1,
        shuffle_caption=False,
        caption_separator=",",
        keep_tokens=0,
        keep_tokens_separator="",
        secondary_separator=None,
        enable_wildcard=False,
        color_aug=False,
        flip_aug=False,
        face_crop_aug_range=None,
        random_crop=False,
        random_crop_padding_percent=0.0,
        caption_dropout_rate=0.0,
        caption_dropout_every_n_epochs=0,
        caption_tag_dropout_rate=0.0,
        caption_prefix=None,
        caption_suffix=None,
        token_warmup_min=0,
        token_warmup_step=0,
    )
    s1 = ds.ControlNetSubset(image_dir="/dir/A", conditioning_data_dir="/cond/A", **kwargs)
    s2 = ds.ControlNetSubset(image_dir="/dir/A", conditioning_data_dir="/cond/A", **kwargs)
    s3 = ds.ControlNetSubset(image_dir="/dir/A", conditioning_data_dir="/cond/B", **kwargs)
    s4 = ds.ControlNetSubset(image_dir="/dir/B", conditioning_data_dir="/cond/A", **kwargs)

    assert s1 == s2
    assert s1 != s3
    assert s1 != s4


def test_image_info_construction():
    info = ds.ImageInfo("key", 1, "caption", False, "/abs/path")
    assert info.image_key == "key"
    assert info.num_repeats == 1
    assert info.latents is None
    # Check other defaults
    assert info.image_size is None
    assert info.bucket_reso is None
    assert info.image is None
