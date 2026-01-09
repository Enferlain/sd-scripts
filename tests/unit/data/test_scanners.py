"""
Unit tests for the dataset scanner module.

Tests Phase 1 of the data pipeline: scanning directories.
"""

import tempfile
import pytest
from pathlib import Path
from PIL import Image

from library.data.scanners import (
    scan_directory,
    scan_metadata_file,
)


@pytest.fixture
def temp_image_dir():
    """Create a temporary directory with test images and captions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create some test images
        for i in range(3):
            img = Image.new("RGB", (512, 512), color=(i * 50, i * 50, i * 50))
            img_path = tmpdir / f"image_{i:03d}.png"
            img.save(img_path)

            # Create caption file
            caption_path = tmpdir / f"image_{i:03d}.txt"
            caption_path.write_text(f"test caption {i}, tag1, tag2")

        # Create one image without caption
        img = Image.new("RGB", (768, 512), color=(100, 100, 100))
        img.save(tmpdir / "no_caption.png")

        # Create a subdirectory with images
        subdir = tmpdir / "subdir"
        subdir.mkdir()
        img = Image.new("RGB", (1024, 768), color=(200, 200, 200))
        img.save(subdir / "sub_image.jpg")
        (subdir / "sub_image.txt").write_text("subdirectory image caption")

        yield tmpdir


# =============================================================================
# scan_directory Tests
# =============================================================================


@pytest.mark.unit
class TestScanDirectory:
    """Test directory scanning functionality."""

    def test_finds_images(self, temp_image_dir):
        """Should find all images in directory."""
        scanned = scan_directory(temp_image_dir, require_caption=False)
        assert len(scanned) == 5  # 3 + 1 + 1 in subdir

    def test_reads_captions(self, temp_image_dir):
        """Should read captions from .txt files."""
        scanned = scan_directory(temp_image_dir, require_caption=False)
        with_captions = [s for s in scanned if s.caption]
        assert len(with_captions) == 4  # 3 in root + 1 in subdir

    def test_non_recursive(self, temp_image_dir):
        """Non-recursive scan should skip subdirectories."""
        scanned = scan_directory(temp_image_dir, recursive=False, require_caption=False)
        assert len(scanned) == 4  # Only root images

    def test_validation_split(self, temp_image_dir):
        """Validation split should mark some images as 'val'."""
        scanned = scan_directory(temp_image_dir, validation_split=0.4, validation_seed=42, require_caption=False)
        val_count = sum(1 for s in scanned if s.split == "val")
        train_count = sum(1 for s in scanned if s.split == "train")
        assert val_count > 0
        assert train_count > 0
        assert val_count + train_count == len(scanned)

    def test_require_caption_error(self, temp_image_dir):
        """Should raise error when require_caption=True and images are missing captions."""
        with pytest.raises(ValueError, match="Missing captions"):
            scan_directory(temp_image_dir, require_caption=True)

    def test_require_caption_allows_reg(self, temp_image_dir):
        """Regularization images should not trigger caption error."""
        # is_reg=True should allow missing captions
        scanned = scan_directory(temp_image_dir, is_reg=True, require_caption=True)
        assert len(scanned) == 5


# =============================================================================
# scan_metadata_file Tests
# =============================================================================


@pytest.fixture
def temp_metadata_dir():
    """Create a temporary directory with images and a metadata JSON file."""
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create some test images
        for i in range(3):
            img = Image.new("RGB", (512, 768), color=(i * 50, i * 50, i * 50))
            img.save(tmpdir / f"image_{i:03d}.png")

        # Create metadata file
        metadata = {
            "image_000": {"caption": "first image caption", "train_resolution": [512, 768]},
            "image_001": {"tags": "tag1, tag2, tag3"},  # tags instead of caption
            "image_002": {"caption": "third image", "tags": "aux, tags"},
        }
        (tmpdir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

        yield tmpdir


@pytest.mark.unit
class TestScanMetadataFile:
    """Test JSON metadata file scanning."""

    def test_loads_metadata(self, temp_metadata_dir):
        """Should load images from metadata file."""

        scanned = scan_metadata_file(
            temp_metadata_dir / "metadata.json",
            image_dir=temp_metadata_dir,
        )
        assert len(scanned) == 3

    def test_reads_caption_and_tags(self, temp_metadata_dir):
        """Should read captions and fall back to tags."""
        scanned = scan_metadata_file(
            temp_metadata_dir / "metadata.json",
            image_dir=temp_metadata_dir,
        )
        captions = {s.path.stem: s.caption for s in scanned}
        assert "first image caption" in captions["image_000"]
        assert "tag1" in captions["image_001"]  # Fell back to tags

    def test_uses_train_resolution(self, temp_metadata_dir):
        """Should use train_resolution from metadata if present."""
        scanned = scan_metadata_file(
            temp_metadata_dir / "metadata.json",
            image_dir=temp_metadata_dir,
        )
        img_000 = next(s for s in scanned if s.path.stem == "image_000")
        assert img_000.width == 512
        assert img_000.height == 768

    def test_require_caption_error(self, temp_metadata_dir):
        """Should raise error when require_caption=True and entries are missing captions."""
        import json

        # Create metadata with missing caption
        metadata = {"image_000": {}}  # No caption or tags
        (temp_metadata_dir / "no_caption.json").write_text(json.dumps(metadata), encoding="utf-8")

        with pytest.raises(ValueError, match="Missing captions"):
            scan_metadata_file(
                temp_metadata_dir / "no_caption.json",
                image_dir=temp_metadata_dir,
                require_caption=True,
            )


# =============================================================================
# class_tokens Tests
# =============================================================================


@pytest.mark.unit
class TestClassTokens:
    """Test class_tokens fallback for images without captions."""

    def test_class_tokens_fallback(self, temp_image_dir):
        """Should use class_tokens as caption when no caption file exists."""
        scanned = scan_directory(
            temp_image_dir,
            class_tokens="a photo of sks dog",
            require_caption=False,
        )
        # Images without captions should get the class_tokens
        no_caption_img = next(s for s in scanned if s.path.name == "no_caption.png")
        assert no_caption_img.caption == "a photo of sks dog"

    def test_class_tokens_does_not_override_existing(self, temp_image_dir):
        """Should not override existing captions with class_tokens."""
        scanned = scan_directory(
            temp_image_dir,
            class_tokens="default caption",
            require_caption=False,
        )
        # Images with captions should keep their original
        img_with_caption = next(s for s in scanned if s.path.stem == "image_000")
        assert "test caption 0" in img_with_caption.caption
