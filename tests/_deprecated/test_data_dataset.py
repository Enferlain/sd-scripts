import types
import torch
import pytest
from unittest.mock import MagicMock
from library.data._deprecated import dataset as ds, dataset_utils


# ============================================================================
# Fixtures & Helpers
# ============================================================================


@pytest.fixture
def base_ds():
    """Create a basic BaseDataset instance for testing methods that don't need heavy setup."""
    d = ds.BaseDataset(resolution=(512, 512), adapter_multiplier=1.0, debug_dataset=False)
    d.tokenizer_max_length = 77
    d.max_train_steps = 100
    return d


def make_subset(**overrides):
    """Create a simple namespace acting as a subset config."""
    defaults = dict(
        caption_prefix=None,
        caption_suffix=None,
        caption_dropout_rate=0.0,
        caption_dropout_every_n_epochs=0,
        enable_wildcard=False,
        shuffle_caption=False,
        token_warmup_step=0,
        token_warmup_min=1,
        caption_tag_dropout_rate=0.0,
        caption_separator=", ",
        keep_tokens=0,
        keep_tokens_separator=None,
        secondary_separator=None,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


# ============================================================================
# Bucket Resolution Tests
# ============================================================================


def test_adjust_min_max_bucket_reso_by_steps(base_ds, caplog):
    # Case 1: Adjustment needed
    with caplog.at_level("WARNING"):
        min_r, max_r = base_ds.adjust_min_max_bucket_reso_by_steps((512, 512), 250, 1030, 64)

    assert (min_r, max_r) == (192, 1088)
    assert "min_bucket_reso is adjusted" in caplog.text
    assert "max_bucket_reso is adjusted" in caplog.text

    # Case 2: No adjustment needed
    caplog.clear()
    min_r, max_r = base_ds.adjust_min_max_bucket_reso_by_steps((512, 512), 256, 1024, 64)
    assert (min_r, max_r) == (256, 1024)
    assert "adjusted" not in caplog.text


def test_adjust_min_max_bucket_reso_invalid(base_ds):
    # Min bucket larger than resolution
    with pytest.raises(AssertionError):
        base_ds.adjust_min_max_bucket_reso_by_steps((512, 512), 600, 1024, 64)

    # Max bucket smaller than resolution
    with pytest.raises(AssertionError):
        base_ds.adjust_min_max_bucket_reso_by_steps((512, 512), 256, 400, 64)


# ============================================================================
# Caption Processing Tests
# ============================================================================


def test_process_caption_simple(base_ds):
    subset = make_subset()
    assert base_ds.process_caption(subset, "hello world") == "hello world"


def test_process_caption_prefix_suffix(base_ds):
    subset = make_subset(caption_prefix="prefix", caption_suffix="suffix")
    assert base_ds.process_caption(subset, "world") == "prefix world suffix"


def test_process_caption_dropout(monkeypatch, base_ds):
    # Ensure every_n_epochs doesn't trigger unexpectedly
    subset = make_subset(caption_dropout_rate=0.5, caption_dropout_every_n_epochs=0)

    # Force random.random() to 0.0 (always drop)
    monkeypatch.setattr(ds.random, "random", lambda: 0.0)
    assert base_ds.process_caption(subset, "hello") == ""

    # Force random.random() to 1.0 (never drop)
    monkeypatch.setattr(ds.random, "random", lambda: 1.0)
    assert base_ds.process_caption(subset, "hello") == "hello"


def test_process_caption_wildcard(monkeypatch, base_ds):
    subset = make_subset(enable_wildcard=True)

    # Mock random.choice to return "b"
    def mock_choice(seq):
        if "b" in seq:
            return "b"
        return seq[0]

    monkeypatch.setattr(ds.random, "choice", mock_choice)

    # Caption with wildcard
    caption = "look at {a|b}"
    assert base_ds.process_caption(subset, caption) == "look at b"


def test_process_caption_multiline_wildcard(monkeypatch, base_ds):
    subset = make_subset(enable_wildcard=True)

    # If wildcards enabled, it picks a random line
    monkeypatch.setattr(ds.random, "choice", lambda seq: seq[1] if len(seq) > 1 else seq[0])

    caption = "line1\nline2\nline3"
    # Should pick line2 based on our mock
    assert base_ds.process_caption(subset, caption) == "line2"


def test_process_caption_multiline_no_wildcard(base_ds):
    subset = make_subset(enable_wildcard=False)
    caption = "line1\nline2"
    # Should always take first line
    assert base_ds.process_caption(subset, caption) == "line1"


def test_process_caption_replacements(base_ds):
    subset = make_subset()
    base_ds.add_replacement("dog", "cat")
    assert base_ds.process_caption(subset, "a cute dog") == "a cute cat"

    # Replace all case (empty string key)
    base_ds.add_replacement("", "all replaced")
    assert base_ds.process_caption(subset, "anything") == "all replaced"


# ============================================================================
# Train/Val Split Tests
# ============================================================================


@pytest.mark.parametrize(
    "split_ratio, expected_train_len",
    [
        (0.0, 10),
        (0.2, 8),
        (0.5, 5),
        (1.0, 0),
    ],
)
def test_split_train_val_boundaries(split_ratio, expected_train_len):
    paths = [str(i) for i in range(10)]
    sizes = [None] * 10
    train_paths, _ = dataset_utils.split_train_val(paths, sizes, True, split_ratio, 123)
    assert len(train_paths) == expected_train_len


def test_split_train_val_deterministic():
    paths = [str(i) for i in range(10)]
    sizes = [None] * 10

    # With fixed seed, outputs should be identical
    a_train, _ = dataset_utils.split_train_val(paths, sizes, True, 0.2, 123)
    b_train, _ = dataset_utils.split_train_val(paths, sizes, True, 0.2, 123)

    assert a_train == b_train


# ============================================================================
# Arbitrary Dataset Loading Tests
# ============================================================================


def test_load_arbitrary_dataset(monkeypatch):
    fake_module = types.SimpleNamespace()
    fake_ctor_args = {}

    class FakeDataset:
        def __init__(self, tok, max_len, res, debug):
            fake_ctor_args["args"] = (tok, max_len, res, debug)

    fake_module.MyDataset = FakeDataset

    # Patch importlib.import_module on the library.data.dataset_utils module
    from library.data import dataset_utils

    monkeypatch.setattr(dataset_utils.importlib, "import_module", lambda _: fake_module)

    # Create nested data_config structure
    data_config = types.SimpleNamespace(
        source=types.SimpleNamespace(dataset_class="my.module.MyDataset"),
        preprocessing=types.SimpleNamespace(
            resolution=(512, 512),
            debug_dataset=False,
        ),
    )
    max_token_length = 75

    out = dataset_utils.load_arbitrary_dataset(data_config, max_token_length, tokenizer="mock_tokenizer")

    assert isinstance(out, FakeDataset)
    assert fake_ctor_args["args"] == ("mock_tokenizer", 75, (512, 512), False)


# ============================================================================
# Input IDs Tests
# ============================================================================


class MockTokenizer:
    def __init__(self, model_max_length=77, pad_token_id=0, eos_token_id=1):
        self.model_max_length = model_max_length
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    def __call__(self, text, padding=None, truncation=None, max_length=None, return_tensors=None):
        # Return sequential ids to easily verify slicing matches
        size = max_length if max_length else self.model_max_length
        ids = torch.arange(size, dtype=torch.long).unsqueeze(0)  # [1, size]
        return types.SimpleNamespace(input_ids=ids)


def test_get_input_ids_standard(base_ds):
    tokenizer = MockTokenizer()
    base_ds.tokenizer_max_length = 77

    ids = base_ds.get_input_ids("test", tokenizer)
    assert ids.shape == (1, 77)
    assert torch.equal(ids, torch.arange(77, dtype=torch.long).unsqueeze(0))


def test_get_input_ids_long_v1(base_ds):
    # V1 case where pad_token_id == eos_token_id
    tokenizer = MockTokenizer(pad_token_id=1, eos_token_id=1)
    base_ds.tokenizer_max_length = 227  # 75*3 + 2

    ids = base_ds.get_input_ids("long caption", tokenizer)

    # Should result in 3 chunks of 77
    assert ids.shape == (3, 77)

    # Chunk 1: [BOS(0), 1..75, EOS(226 if untruncated, but check slice)]
    # Input ids mock: 0..226.

    # Logic verify
    c1 = ids[0]
    assert c1[0] == 0  # BOS
    assert torch.equal(c1[1:76], torch.arange(1, 76, dtype=torch.long))
    assert c1[-1] == 226  # EOS

    c2 = ids[1]
    assert c2[0] == 0
    assert torch.equal(c2[1:76], torch.arange(76, 151, dtype=torch.long))
    assert c2[-1] == 226

    c3 = ids[2]  # Added assertion for 3rd chunk per feedback
    assert c3[0] == 0
    assert torch.equal(c3[1:76], torch.arange(151, 226, dtype=torch.long))  # 151 + 75 = 226
    assert c3[-1] == 226


def test_get_input_ids_long_v2_fixups(base_ds):
    # V2/SDXL case: pad != eos
    tokenizer = MockTokenizer(pad_token_id=0, eos_token_id=1)
    base_ds.tokenizer_max_length = 227

    # We want to test logic:
    # if ids_chunk[-2] != eos and != pad: ids_chunk[-1] = eos
    # if ids_chunk[1] == pad: ids_chunk[1] = eos

    # Standard mock returns sequential:
    # Chunk 1 (1..75). Last is 75 (idx 76). Not 0, not 1. Last token forced to 1.
    ids = base_ds.get_input_ids("test", tokenizer)

    # Check fixup 1: Forced EOS at end of chunk 1
    assert ids[0][-1] == 1

    # Test fixup 2: Start with PAD
    class PadHeavyTokenizer:
        def __init__(self):
            self.model_max_length = 77
            self.pad_token_id = 0
            self.eos_token_id = 1

        def __call__(self, *args, **kwargs):
            t = torch.zeros((1, 227), dtype=torch.long)
            return types.SimpleNamespace(input_ids=t)

    pad_heavy_tokenizer = PadHeavyTokenizer()
    ids_pad = base_ds.get_input_ids("pad test", pad_heavy_tokenizer)

    # Chunk 1: [0(BOS), 0(PAD? no, index 1 of tensor), ..., 0(EOS?)]
    # Original tensor is all 0.
    # ids_chunk[1] comes from input_ids[1] -> 0.
    # Since ids_chunk[1] == 0 (PAD), it should switch to EOS (1).
    assert ids_pad[0][1] == 1


# ============================================================================
# Register Image Tests
# ============================================================================


def test_register_image_populates_dicts(base_ds):
    """Verify register_image populates image_data and image_to_subset."""
    from library.data._deprecated.data_structures import ImageInfo

    info = ImageInfo(image_key="test_key", num_repeats=1, caption="test caption", is_reg=False, absolute_path="/fake/path/image.png")
    subset = make_subset()

    base_ds.register_image(info, subset)

    assert "test_key" in base_ds.image_data
    assert base_ds.image_data["test_key"] is info
    assert "test_key" in base_ds.image_to_subset
    assert base_ds.image_to_subset["test_key"] is subset


def test_register_image_multiple(base_ds):
    """Verify multiple images can be registered to different subsets."""
    from library.data._deprecated.data_structures import ImageInfo

    info1 = ImageInfo("key1", 1, "cap1", False, "/path1.png")
    info2 = ImageInfo("key2", 2, "cap2", True, "/path2.png")
    subset1 = make_subset()
    subset2 = make_subset(caption_prefix="reg")

    base_ds.register_image(info1, subset1)
    base_ds.register_image(info2, subset2)

    assert len(base_ds.image_data) == 2
    assert base_ds.image_to_subset["key1"] is subset1
    assert base_ds.image_to_subset["key2"] is subset2


# ============================================================================
# Cacheability Check Tests
# ============================================================================


def test_is_latent_cacheable_true(base_ds):
    """Latent caching is possible when no augmentation requires runtime changes."""
    subset1 = make_subset(color_aug=False, random_crop=False)
    subset2 = make_subset(color_aug=False, random_crop=False)
    # Monkey-patch required attributes
    subset1.color_aug = False
    subset1.random_crop = False
    subset2.color_aug = False
    subset2.random_crop = False
    base_ds.subsets = [subset1, subset2]

    assert base_ds.is_latent_cacheable() is True


def test_is_latent_cacheable_false_color_aug(base_ds):
    """Color augmentation prevents latent caching."""
    subset = make_subset()
    subset.color_aug = True
    subset.random_crop = False
    base_ds.subsets = [subset]

    assert base_ds.is_latent_cacheable() is False


def test_is_latent_cacheable_false_random_crop(base_ds):
    """Random crop prevents latent caching."""
    subset = make_subset()
    subset.color_aug = False
    subset.random_crop = True
    base_ds.subsets = [subset]

    assert base_ds.is_latent_cacheable() is False


def test_is_text_encoder_output_cacheable_true(base_ds):
    """TE output caching is possible when no caption variation enabled."""
    subset = make_subset()
    subset.caption_dropout_rate = 0
    subset.shuffle_caption = False
    subset.token_warmup_step = 0
    subset.caption_tag_dropout_rate = 0
    base_ds.subsets = [subset]

    assert base_ds.is_text_encoder_output_cacheable() is True


def test_is_text_encoder_output_cacheable_false_shuffle_caption(base_ds):
    """Shuffle caption prevents TE output caching."""
    subset = make_subset()
    subset.caption_dropout_rate = 0
    subset.shuffle_caption = True
    subset.token_warmup_step = 0
    subset.caption_tag_dropout_rate = 0
    base_ds.subsets = [subset]

    assert base_ds.is_text_encoder_output_cacheable() is False


def test_is_text_encoder_output_cacheable_false_dropout(base_ds):
    """Caption dropout prevents TE output caching."""
    subset = make_subset()
    subset.caption_dropout_rate = 0.1
    subset.shuffle_caption = False
    subset.token_warmup_step = 0
    subset.caption_tag_dropout_rate = 0
    base_ds.subsets = [subset]

    assert base_ds.is_text_encoder_output_cacheable() is False


def test_is_text_encoder_output_cacheable_false_warmup(base_ds):
    """Token warmup prevents TE output caching."""
    subset = make_subset()
    subset.caption_dropout_rate = 0
    subset.shuffle_caption = False
    subset.token_warmup_step = 100
    subset.caption_tag_dropout_rate = 0
    base_ds.subsets = [subset]

    assert base_ds.is_text_encoder_output_cacheable() is False


def test_is_text_encoder_output_cacheable_false_tag_dropout(base_ds):
    """Tag dropout prevents TE output caching."""
    subset = make_subset()
    subset.caption_dropout_rate = 0
    subset.shuffle_caption = False
    subset.token_warmup_step = 0
    subset.caption_tag_dropout_rate = 0.1
    base_ds.subsets = [subset]

    assert base_ds.is_text_encoder_output_cacheable() is False


# ============================================================================
# Shuffle Buckets Tests
# ============================================================================


def test_shuffle_buckets_deterministic(base_ds):
    """Shuffle with same seed produces same order."""
    from library.data._deprecated.data_structures import BucketManager, BucketBatchIndex

    base_ds.seed = 42
    base_ds.current_epoch = 0
    base_ds.bucket_manager = BucketManager(False, (512, 512), None, None, None)
    base_ds.bucket_manager.set_predefined_resos([(512, 512)])
    base_ds.bucket_manager.add_if_new_reso((512, 512))  # Register reso in reso_to_id
    # Add some images
    for i in range(10):
        base_ds.bucket_manager.add_image((512, 512), f"img_{i}")

    base_ds.buckets_indices = [BucketBatchIndex(0, 2, i) for i in range(5)]

    # Shuffle once
    base_ds.shuffle_buckets()
    order1 = [idx.batch_index for idx in base_ds.buckets_indices]

    # Reset and shuffle again with same seed
    base_ds.buckets_indices = [BucketBatchIndex(0, 2, i) for i in range(5)]
    base_ds.shuffle_buckets()
    order2 = [idx.batch_index for idx in base_ds.buckets_indices]

    assert order1 == order2


def test_shuffle_buckets_different_epochs(base_ds):
    """Shuffle produces different order for different epochs."""
    from library.data._deprecated.data_structures import BucketManager, BucketBatchIndex

    base_ds.seed = 42
    base_ds.bucket_manager = BucketManager(False, (512, 512), None, None, None)
    base_ds.bucket_manager.set_predefined_resos([(512, 512)])
    base_ds.bucket_manager.add_if_new_reso((512, 512))  # Register reso in reso_to_id
    for i in range(20):
        base_ds.bucket_manager.add_image((512, 512), f"img_{i}")

    base_ds.buckets_indices = [BucketBatchIndex(0, 2, i) for i in range(10)]

    base_ds.current_epoch = 0
    base_ds.shuffle_buckets()
    order_epoch0 = [idx.batch_index for idx in base_ds.buckets_indices]

    base_ds.buckets_indices = [BucketBatchIndex(0, 2, i) for i in range(10)]
    base_ds.current_epoch = 1
    base_ds.shuffle_buckets()
    order_epoch1 = [idx.batch_index for idx in base_ds.buckets_indices]

    # Different epochs should produce different orders (with high probability)
    assert order_epoch0 != order_epoch1


# ============================================================================
# Get Image Size Tests
# ============================================================================


def test_get_image_size_jxl_path(base_ds, monkeypatch):
    """JXL files use dedicated size function."""
    monkeypatch.setattr(ds, "get_jxl_size", lambda p: (1024, 768))

    result = base_ds.get_image_size("/path/to/image.jxl")
    assert result == (1024, 768)


def test_get_image_size_jxl_uppercase(base_ds, monkeypatch):
    """JXL extension check is case-insensitive."""
    monkeypatch.setattr(ds, "get_jxl_size", lambda p: (800, 600))

    result = base_ds.get_image_size("/path/to/IMAGE.JXL")
    assert result == (800, 600)


def test_get_image_size_regular_image(base_ds, monkeypatch):
    """Regular images use imagesize module."""
    monkeypatch.setattr(ds.imagesize, "get", lambda p: (512, 512))

    result = base_ds.get_image_size("/path/to/image.png")
    assert result == (512, 512)


def test_get_image_size_pil_fallback(base_ds, monkeypatch, tmp_path):
    """Falls back to PIL when imagesize returns invalid size."""
    from PIL import Image

    # Create a real test image
    img_path = tmp_path / "test.png"
    img = Image.new("RGB", (256, 128))
    img.save(img_path)

    # imagesize returns invalid result
    monkeypatch.setattr(ds.imagesize, "get", lambda p: (-1, -1))

    result = base_ds.get_image_size(str(img_path))
    assert result == (256, 128)


# ============================================================================
# Cache Latents Tests (Heavy Mocking)
# ============================================================================

from unittest.mock import patch


@patch("library.data.dataset.cache_batch_latents")
@patch("library.data.dataset.is_disk_cached_latents_is_expected")
def test_cache_latents_groups_by_condition(mock_cache_check, mock_batch_fn, base_ds):
    """Verify images with same conditions are batched together."""
    from library.data._deprecated.data_structures import ImageInfo

    # Setup dataset with images
    info1 = ImageInfo("key1", 1, "cap1", False, "/path1.png")
    info1.bucket_reso = (512, 512)
    info1.resized_size = (512, 512)
    info1.image_size = (512, 512)

    info2 = ImageInfo("key2", 1, "cap2", False, "/path2.png")
    info2.bucket_reso = (512, 512)
    info2.resized_size = (512, 512)
    info2.image_size = (512, 512)

    subset = make_subset()
    subset.flip_aug = False
    subset.alpha_mask = False
    subset.random_crop = False
    subset.random_crop_padding_percent = 0.0

    base_ds.image_data = {"key1": info1, "key2": info2}
    base_ds.image_to_subset = {"key1": subset, "key2": subset}

    mock_cache_check.return_value = False  # No cached latents

    mock_vae = MagicMock()
    base_ds.cache_latents(mock_vae, vae_batch_size=2, cache_to_disk=False)

    # Verify batch was called (images grouped together by condition)
    assert mock_batch_fn.called


@patch("library.data.dataset.cache_batch_latents")
@patch("library.data.dataset.is_disk_cached_latents_is_expected")
def test_cache_latents_skips_cached(mock_cache_check, mock_batch_fn, base_ds):
    """Images with valid disk cache are skipped."""
    from library.data._deprecated.data_structures import ImageInfo

    info = ImageInfo("key1", 1, "cap1", False, "/path1.png")
    info.bucket_reso = (512, 512)
    info.resized_size = (512, 512)
    info.image_size = (512, 512)

    subset = make_subset()
    subset.flip_aug = False
    subset.alpha_mask = False
    subset.random_crop = False
    subset.random_crop_padding_percent = 0.0

    base_ds.image_data = {"key1": info}
    base_ds.image_to_subset = {"key1": subset}

    # Return True = cache exists and is valid
    mock_cache_check.return_value = True

    mock_vae = MagicMock()
    base_ds.cache_latents(mock_vae, vae_batch_size=1, cache_to_disk=True)

    # No batch call since cache exists
    mock_batch_fn.assert_not_called()


@patch("library.data.dataset.cache_batch_latents")
def test_cache_latents_non_main_process_early_exit(mock_batch_fn, base_ds):
    """Non-main process exits early when caching to disk."""
    from library.data._deprecated.data_structures import ImageInfo

    info = ImageInfo("key1", 1, "cap1", False, "/path1.png")
    info.bucket_reso = (512, 512)
    info.resized_size = (512, 512)
    info.image_size = (512, 512)

    subset = make_subset()
    subset.flip_aug = False
    subset.alpha_mask = False
    subset.random_crop = False
    subset.random_crop_padding_percent = 0.0

    base_ds.image_data = {"key1": info}
    base_ds.image_to_subset = {"key1": subset}

    mock_vae = MagicMock()
    # cache_to_disk=True, is_main_process=False -> should return early
    base_ds.cache_latents(mock_vae, vae_batch_size=1, cache_to_disk=True, is_main_process=False)

    # info.latents_npz should be set, but no batch call
    assert info.latents_npz is not None
    mock_batch_fn.assert_not_called()


@patch("library.data.dataset.cache_batch_latents")
@patch("library.data.dataset.is_disk_cached_latents_is_expected")
def test_cache_latents_skips_finetuning_with_npz(mock_cache_check, mock_batch_fn, base_ds):
    """Images that already have latents_npz set (fine-tuning) are skipped."""
    from library.data._deprecated.data_structures import ImageInfo

    info = ImageInfo("key1", 1, "cap1", False, "/path1.png")
    info.bucket_reso = (512, 512)
    info.resized_size = (512, 512)
    info.image_size = (512, 512)
    info.latents_npz = "/already/cached.npz"  # Pre-set from fine-tuning metadata

    subset = make_subset()
    subset.flip_aug = False
    subset.alpha_mask = False
    subset.random_crop = False
    subset.random_crop_padding_percent = 0.0

    base_ds.image_data = {"key1": info}
    base_ds.image_to_subset = {"key1": subset}

    mock_vae = MagicMock()
    base_ds.cache_latents(mock_vae, vae_batch_size=1, cache_to_disk=False)

    # Cache check not called because latents_npz was already set
    mock_cache_check.assert_not_called()
    mock_batch_fn.assert_not_called()


@patch("library.data.dataset.cache_batch_latents")
@patch("library.data.dataset.is_disk_cached_latents_is_expected")
def test_cache_latents_batches_by_vae_batch_size(mock_cache_check, mock_batch_fn, base_ds):
    """Verify batches respect vae_batch_size parameter."""
    from library.data._deprecated.data_structures import ImageInfo

    # Create 5 images with same condition
    infos = []
    for i in range(5):
        info = ImageInfo(f"key{i}", 1, f"cap{i}", False, f"/path{i}.png")
        info.bucket_reso = (512, 512)
        info.resized_size = (512, 512)
        info.image_size = (512, 512)
        infos.append(info)

    subset = make_subset()
    subset.flip_aug = False
    subset.alpha_mask = False
    subset.random_crop = False
    subset.random_crop_padding_percent = 0.0

    base_ds.image_data = {info.image_key: info for info in infos}
    base_ds.image_to_subset = {info.image_key: subset for info in infos}

    mock_cache_check.return_value = False

    mock_vae = MagicMock()
    base_ds.cache_latents(mock_vae, vae_batch_size=2, cache_to_disk=False)

    # With 5 images and batch_size=2, should have 3 batches (2, 2, 1)
    assert mock_batch_fn.call_count == 3
