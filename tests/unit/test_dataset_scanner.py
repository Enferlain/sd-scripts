
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import json
import tempfile
import shutil

from library.data.pipeline.dataset_scanner import (
    scan_directory,
    scan_metadata_file,
    create_manifest,
    create_manifest_from_config,
    make_bucket_resolutions,
    select_bucket,
    ScannedImage
)
from library.data.pipeline.dataclasses import DatasetManifest
from library.config.dataclasses.data import (
    DataConfig, SourceConfig, PreprocessingConfig,
    CaptionConfig, BucketingConfig
)

# Helpers
@pytest.fixture
def temp_dir():
    dir_path = tempfile.mkdtemp()
    yield Path(dir_path)
    shutil.rmtree(dir_path)

@pytest.fixture
def mock_get_image_size():
    with patch("library.data.pipeline.dataset_scanner.get_image_size") as mock:
        mock.return_value = (512, 512)
        yield mock

@pytest.fixture
def mock_check_has_alpha():
    with patch("library.data.pipeline.dataset_scanner.check_has_alpha") as mock:
        mock.return_value = False
        yield mock

def create_dummy_image(path: Path):
    path.touch()

def create_dummy_caption(path: Path, content: str):
    path.write_text(content, encoding="utf-8")

# Tests

def test_scan_directory_basic(temp_dir, mock_get_image_size, mock_check_has_alpha):
    """Finds images, reads captions, returns ScannedImage list"""
    img1 = temp_dir / "img1.png"
    cap1 = temp_dir / "img1.txt"
    img2 = temp_dir / "img2.jpg"
    cap2 = temp_dir / "img2.caption"

    create_dummy_image(img1)
    create_dummy_caption(cap1, "caption 1")
    create_dummy_image(img2)
    create_dummy_caption(cap2, "caption 2")

    # Configure mock to return different sizes if needed, or same
    mock_get_image_size.side_effect = [(512, 512), (1024, 768)]

    scanned = scan_directory(
        image_dir=temp_dir,
        caption_extension=".txt",
        is_reg=False
    )

    assert len(scanned) == 2
    # Sort order is deterministic (by path)
    # img1.png comes before img2.jpg
    assert scanned[0].path.name == "img1.png"
    assert scanned[0].caption == "caption 1"
    assert scanned[0].width == 512
    assert scanned[0].height == 512

    assert scanned[1].path.name == "img2.jpg"
    assert scanned[1].caption == "caption 2" # Should find .caption as fallback
    assert scanned[1].width == 1024
    assert scanned[1].height == 768

def test_scan_directory_class_tokens(temp_dir, mock_get_image_size, mock_check_has_alpha):
    """Uses class_tokens as fallback when no caption file"""
    img1 = temp_dir / "img1.png"
    create_dummy_image(img1)

    scanned = scan_directory(
        image_dir=temp_dir,
        class_tokens="style of x",
        require_caption=False
    )

    assert len(scanned) == 1
    assert scanned[0].caption == "style of x"

def test_scan_directory_is_reg(temp_dir, mock_get_image_size, mock_check_has_alpha):
    """Sets is_reg=True correctly, doesn't require captions"""
    img1 = temp_dir / "img1.png"
    create_dummy_image(img1)

    scanned = scan_directory(
        image_dir=temp_dir,
        is_reg=True,
        require_caption=False
    )

    assert len(scanned) == 1
    assert scanned[0].is_reg is True
    assert scanned[0].caption == ""

def test_scan_directory_validation_split(temp_dir, mock_get_image_size, mock_check_has_alpha):
    """Deterministically splits train/val with seed"""
    for i in range(10):
        create_dummy_image(temp_dir / f"img{i:02d}.png")
        create_dummy_caption(temp_dir / f"img{i:02d}.txt", f"cap{i}")

    scanned = scan_directory(
        image_dir=temp_dir,
        validation_split=0.2,
        validation_seed=42
    )

    assert len(scanned) == 10
    val_count = sum(1 for s in scanned if s.split == "val")
    train_count = sum(1 for s in scanned if s.split == "train")

    assert val_count == 2
    assert train_count == 8

    # Verify deterministic behavior
    scanned2 = scan_directory(
        image_dir=temp_dir,
        validation_split=0.2,
        validation_seed=42
    )

    val_indices_1 = [i for i, s in enumerate(scanned) if s.split == "val"]
    val_indices_2 = [i for i, s in enumerate(scanned2) if s.split == "val"]
    assert val_indices_1 == val_indices_2

def test_scan_metadata_file(temp_dir, mock_get_image_size, mock_check_has_alpha):
    """Parses JSON metadata, resolves image paths"""
    img_real_name = "my_image"
    img_path = temp_dir / f"{img_real_name}.png"
    create_dummy_image(img_path)

    metadata = {
        img_real_name: {
            "caption": "metadata caption",
            "tags": "tag1, tag2"
        }
    }

    meta_file = temp_dir / "meta.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f)

    scanned = scan_metadata_file(
        metadata_file=meta_file,
        image_dir=temp_dir
    )

    assert len(scanned) == 1
    assert scanned[0].path == img_path
    assert scanned[0].caption == "metadata caption"
    assert scanned[0].width == 512

def test_create_manifest():
    """Assigns images to buckets, generates correct manifest"""
    scanned = [
        ScannedImage(Path("/a/b.png"), 512, 512, "cap1", 1),
        ScannedImage(Path("/a/c.png"), 1024, 512, "cap2", 1),
    ]

    manifest = create_manifest(
        scanned_images=scanned,
        base_resolution=(512, 512),
        bucket_reso_steps=64
    )

    assert isinstance(manifest, DatasetManifest)
    assert len(manifest.entries) == 2
    assert len(manifest.buckets) >= 1

    entries = list(manifest.entries.values())
    assert entries[0].caption == "cap1"
    assert entries[0].original_size == (512, 512)
    assert entries[0].bucket_reso == (512, 512)

def test_create_manifest_from_config(temp_dir, mock_get_image_size, mock_check_has_alpha):
    """Handles train_data_dir, reg_data_dir, in_json, subsets"""
    # Create directory structure
    train_dir = temp_dir / "train"
    train_dir.mkdir()
    create_dummy_image(train_dir / "img1.png")
    create_dummy_caption(train_dir / "img1.txt", "train cap")

    reg_dir = temp_dir / "reg"
    reg_dir.mkdir()
    create_dummy_image(reg_dir / "reg1.png")

    # Config
    config = DataConfig()
    config.source.train_data_dir = str(train_dir)
    config.source.reg_data_dir = str(reg_dir)
    config.preprocessing.resolution = "512,512"

    manifest = create_manifest_from_config(config)

    assert len(manifest.entries) == 2

    # Verify train image
    train_entries = [e for e in manifest.entries.values() if not e.is_reg]
    assert len(train_entries) == 1
    assert train_entries[0].caption == "train cap"

    # Verify reg image
    reg_entries = [e for e in manifest.entries.values() if e.is_reg]
    assert len(reg_entries) == 1
    assert reg_entries[0].is_reg is True

def test_bucket_resolution_generation():
    """Matches legacy bucket resolution output"""
    resolutions = make_bucket_resolutions(
        max_reso=(512, 512),
        min_size=256,
        max_size=1024,
        divisible=64
    )

    # Basic checks
    assert (512, 512) in resolutions

    # Check bounds
    for w, h in resolutions:
        assert w % 64 == 0
        assert h % 64 == 0
        assert w >= 256
        assert h >= 256

def test_bucket_selection():
    """Selects correct bucket for various aspect ratios"""
    bucket_resos = [
        (512, 512),
        (768, 384), # 2.0
        (384, 768), # 0.5
    ]

    # Square image
    bucket, resized = select_bucket(1024, 1024, bucket_resos)
    assert bucket == (512, 512)
    assert resized == (512, 512)

    # Wide image (exact AR match)
    bucket, resized = select_bucket(2000, 1000, bucket_resos)
    assert bucket == (768, 384)
    assert resized == (768, 384)

    # Wide image (needs resize to fit)
    bucket, resized = select_bucket(1536, 768, bucket_resos)
    assert bucket == (768, 384)
    assert resized == (768, 384)

    # Slight mismatch AR
    bucket, resized = select_bucket(1000, 1000, bucket_resos)
    assert bucket == (512, 512)
