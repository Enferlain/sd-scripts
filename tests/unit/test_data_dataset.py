import types
import torch
import pytest
import importlib
from unittest.mock import MagicMock
from library.data import dataset as ds

# ============================================================================
# Fixtures & Helpers
# ============================================================================

@pytest.fixture
def base_ds():
    """Create a basic BaseDataset instance for testing methods that don't need heavy setup."""
    d = ds.BaseDataset(resolution=(512, 512), network_multiplier=1.0, debug_dataset=False)
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
        if "b" in seq: return "b"
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

@pytest.mark.parametrize("split_ratio, expected_train_len", [
    (0.0, 10),
    (0.2, 8),
    (0.5, 5),
    (1.0, 0),
])
def test_split_train_val_boundaries(split_ratio, expected_train_len):
    paths = [str(i) for i in range(10)]
    sizes = [None] * 10
    train_paths, _ = ds.split_train_val(paths, sizes, True, split_ratio, 123)
    assert len(train_paths) == expected_train_len

def test_split_train_val_deterministic():
    paths = [str(i) for i in range(10)]
    sizes = [None] * 10
    
    # With fixed seed, outputs should be identical
    a_train, _ = ds.split_train_val(paths, sizes, True, 0.2, 123)
    b_train, _ = ds.split_train_val(paths, sizes, True, 0.2, 123)
    
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
    
    # Patch importlib.import_module on the library.data.dataset module as suggested
    monkeypatch.setattr(ds.importlib, "import_module", lambda _: fake_module)

    cfg = types.SimpleNamespace(
        dataset_class="my.module.MyDataset",
        max_token_length=75,
        resolution=(512, 512),
        debug_dataset=False,
    )
    
    out = ds.load_arbitrary_dataset(cfg, tokenizer="mock_tokenizer")
    
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
        ids = torch.arange(size, dtype=torch.long).unsqueeze(0) # [1, size]
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
    base_ds.tokenizer_max_length = 227 # 75*3 + 2
    
    ids = base_ds.get_input_ids("long caption", tokenizer)
    
    # Should result in 3 chunks of 77
    assert ids.shape == (3, 77)
    
    # Chunk 1: [BOS(0), 1..75, EOS(226 if untruncated, but check slice)]
    # Input ids mock: 0..226.
    
    # Logic verify
    c1 = ids[0]
    assert c1[0] == 0 # BOS
    assert torch.equal(c1[1:76], torch.arange(1, 76, dtype=torch.long))
    assert c1[-1] == 226 # EOS
    
    c2 = ids[1]
    assert c2[0] == 0 
    assert torch.equal(c2[1:76], torch.arange(76, 151, dtype=torch.long))
    assert c2[-1] == 226

    c3 = ids[2] # Added assertion for 3rd chunk per feedback
    assert c3[0] == 0
    assert torch.equal(c3[1:76], torch.arange(151, 226, dtype=torch.long)) # 151 + 75 = 226
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
