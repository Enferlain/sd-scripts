import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import torch
import random
from safetensors.torch import save_file

from library.data.pipeline.dataclasses import DatasetManifest, CacheEntry, EpochManifest, BatchInfo
from library.data.pipeline.epoch_preparation import prepare_epoch, prepare_validation_epoch, tokenize_epoch_manifest, load_epoch_tokens
from library.data.pipeline.caption_processor import CaptionConfig


@pytest.fixture
def mock_manifest():
    manifest = DatasetManifest()

    # Create entries with different resolutions
    entries = {
        "img1": CacheEntry(
            id="img1", image_path="/tmp/img1.jpg", original_size=(1024, 1024),
            bucket_reso=(1024, 1024), resized_size=(1024, 1024), caption="caption 1",
            num_repeats=1
        ),
        "img2": CacheEntry(
            id="img2", image_path="/tmp/img2.jpg", original_size=(512, 512),
            bucket_reso=(512, 512), resized_size=(512, 512), caption="caption 2",
            num_repeats=1
        ),
        "img3": CacheEntry(
            id="img3", image_path="/tmp/img3.jpg", original_size=(512, 512),
            bucket_reso=(512, 512), resized_size=(512, 512), caption="caption 3",
            num_repeats=1
        ),
        "img4": CacheEntry(
            id="img4", image_path="/tmp/img4.jpg", original_size=(1024, 1024),
            bucket_reso=(1024, 1024), resized_size=(1024, 1024), caption="caption 4",
            num_repeats=2
        ),
    }
    manifest.entries = entries
    return manifest

def test_prepare_epoch_shuffle():
    """Different seed+epoch produces different order"""
    # Create a manifest with many items to avoid random collision
    manifest = DatasetManifest()
    entries = {}
    for i in range(20):
        entries[f"img{i}"] = CacheEntry(
            id=f"img{i}", image_path=f"/tmp/img{i}.jpg", original_size=(512, 512),
            bucket_reso=(512, 512), resized_size=(512, 512), caption=f"caption {i}",
            num_repeats=1
        )
    manifest.entries = entries

    epoch1 = prepare_epoch(manifest, epoch=1, seed=42, shuffle=True, warmup_largest_first=False)
    epoch2 = prepare_epoch(manifest, epoch=2, seed=42, shuffle=True, warmup_largest_first=False)

    # Check seeds are different
    assert epoch1.seed == 43
    assert epoch2.seed == 44

    # Extract order of image IDs
    order1 = [b.image_ids[0] for b in epoch1.batches]
    order2 = [b.image_ids[0] for b in epoch2.batches]

    # With 20 items, probability of collision is infinitesimal
    assert order1 != order2

    # Verify that the same seed+epoch produces the same order
    epoch1_again = prepare_epoch(manifest, epoch=1, seed=42, shuffle=True, warmup_largest_first=False)
    order1_again = [b.image_ids[0] for b in epoch1_again.batches]
    assert order1 == order1_again

def test_prepare_epoch_warmup(mock_manifest):
    """Largest buckets first"""
    # mock_manifest has (1024, 1024) and (512, 512) buckets.
    # 1024x1024 is larger.

    epoch = prepare_epoch(mock_manifest, epoch=1, seed=42, warmup_largest_first=True, warmup_batches=2)

    # First batch should be 1024x1024
    assert epoch.batches[0].bucket_reso == (1024, 1024)

    # There are 3 images with 1024x1024 (img1, img4*2) and 2 with 512x512 (img2, img3).
    # Since batch_size defaults to 1, we have 3 batches of 1024x1024 and 2 of 512x512.
    # warmup_batches=2 means the first 2 should be guaranteed to be the largest available (1024x1024).

    # assert epoch.batches[0].bucket_reso == (1024, 1024) # Removed duplicate assertion
    assert epoch.batches[1].bucket_reso == (1024, 1024)

    # The rest are shuffled, but we can verify that we indeed prioritize largest

def test_prepare_epoch_repeats(mock_manifest):
    """num_repeats expands images correctly"""
    # img4 has num_repeats=2
    epoch = prepare_epoch(mock_manifest, epoch=1, seed=42, batch_size=1, shuffle=False)

    # Count occurrences of img4
    img4_count = 0
    for batch in epoch.batches:
        for img_id in batch.image_ids:
            if img_id == "img4":
                img4_count += 1

    assert img4_count == 2

    # img1 has num_repeats=1
    img1_count = sum(1 for batch in epoch.batches for img_id in batch.image_ids if img_id == "img1")
    assert img1_count == 1

def test_caption_processing(mock_manifest):
    """Shuffle, dropout, prefix/suffix applied"""
    caption_config = CaptionConfig(
        caption_dropout_rate=0.0,
        shuffle_caption=True,
        token_warmup_step=0 # disable warmup for deterministic test
    )

    # Modify an entry to have comma separated tags
    mock_manifest.entries["img1"].caption = "tag1, tag2, tag3"

    epoch = prepare_epoch(mock_manifest, epoch=1, seed=42, caption_config=caption_config)

    # Find batch with img1
    found = False
    for batch in epoch.batches:
        if "img1" in batch.image_ids:
            processed = batch.processed_captions[0]
            # Verify tags are present
            assert "tag1" in processed
            assert "tag2" in processed
            assert "tag3" in processed

            # Verify that shuffle actually changed the order
            # With seed 42 and default hashing, the order should be different
            # Original: "tag1, tag2, tag3"
            assert processed != "tag1, tag2, tag3"

            found = True
            break

    assert found

def test_tokenize_epoch_manifest(tmp_path, mock_manifest):
    """Saves tokens to safetensors with correct shapes"""
    epoch = prepare_epoch(mock_manifest, epoch=1, seed=42)

    # Mock tokenize function
    # return list of tensors. Assume 2 encoders (like SDXL)
    def mock_tokenize(captions):
        num_captions = len(captions)
        # return two tensors: one for clip_l, one for clip_g
        # shape [num_captions, 77]
        return [
            torch.zeros((num_captions, 77), dtype=torch.long),
            torch.ones((num_captions, 77), dtype=torch.long)
        ]

    output_path = tmp_path / "tokens.safetensors"

    path = tokenize_epoch_manifest(
        epoch,
        mock_tokenize,
        output_path,
        encoder_names=["clip_l", "clip_g"]
    )

    assert path.exists()

    # Verify content using safetensors
    from safetensors.torch import load_file
    tensors = load_file(path)

    assert "clip_l" in tensors
    assert "clip_g" in tensors

    # Total images: img1(1) + img2(1) + img3(1) + img4(2) = 5
    assert tensors["clip_l"].shape == (5, 77)
    assert tensors["clip_g"].shape == (5, 77)

    # check metadata
    # We can't easily check metadata with load_file, need safe_open or load_epoch_tokens to check that.

def test_load_epoch_tokens(tmp_path, mock_manifest):
    """Loads tokens, validates manifest hash"""
    epoch = prepare_epoch(mock_manifest, epoch=1, seed=42)

    def mock_tokenize(captions):
        num_captions = len(captions)
        return [torch.zeros((num_captions, 77), dtype=torch.long)]

    output_path = tmp_path / "tokens_load.safetensors"
    tokenize_epoch_manifest(epoch, mock_tokenize, output_path, encoder_names=["clip"])

    tensors, metadata = load_epoch_tokens(output_path)

    assert "clip" in tensors
    assert tensors["clip"].shape == (5, 77)

    assert "epoch" in metadata
    assert metadata["epoch"] == "1"
    assert "seed" in metadata
    assert metadata["seed"] == str(epoch.seed)
    assert "manifest_hash" in metadata

def test_prepare_validation_epoch(mock_manifest):
    """Verify validation epoch preparation"""
    # Create validation entries
    mock_manifest.entries["val1"] = CacheEntry(
        id="val1", image_path="/tmp/val1.jpg", original_size=(512, 512),
        bucket_reso=(512, 512), resized_size=(512, 512), caption="val 1",
        split="val"
    )
    mock_manifest.entries["val2"] = CacheEntry(
        id="val2", image_path="/tmp/val2.jpg", original_size=(512, 512),
        bucket_reso=(512, 512), resized_size=(512, 512), caption="val 2",
        split="val"
    )

    epoch = prepare_validation_epoch(mock_manifest, batch_size=1)

    # Should only have validation images
    assert epoch.num_batches == 2
    ids = sorted([b.image_ids[0] for b in epoch.batches])
    assert ids == ["val1", "val2"]

    # Should be deterministic
    epoch2 = prepare_validation_epoch(mock_manifest, batch_size=1)
    ids2 = sorted([b.image_ids[0] for b in epoch2.batches])
    assert ids == ids2
