"""
Unit tests for the dataset scanner module.

Tests Phase 1 of the data pipeline: scanning directories and creating manifests.
"""

import tempfile
import pytest
from pathlib import Path

from PIL import Image

from library.data.pipeline.dataset_scanner import (
    scan_directory,
    make_bucket_resolutions,
    select_bucket,
    create_manifest,
)
from library.data.pipeline import save_dataset_manifest, load_dataset_manifest


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
# create_manifest Tests
# =============================================================================


@pytest.mark.unit
class TestCreateManifest:
    """Test manifest creation from scanned images."""

    def test_creates_entries_for_all_images(self, temp_image_dir):
        """Should create an entry for each scanned image."""
        scanned = scan_directory(temp_image_dir, require_caption=False)
        manifest = create_manifest(scanned, base_dir=temp_image_dir)
        assert len(manifest.entries) == len(scanned)

    def test_assigns_to_buckets(self, temp_image_dir):
        """Should assign images to buckets."""
        scanned = scan_directory(temp_image_dir, require_caption=False)
        manifest = create_manifest(scanned, base_dir=temp_image_dir)
        assert len(manifest.buckets) > 0
        # Total images in buckets should match entries
        total_in_buckets = sum(len(b.image_ids) for b in manifest.buckets.values())
        assert total_in_buckets == len(manifest.entries)

    def test_parses_tags(self, temp_image_dir):
        """Should parse comma-separated tags from captions."""
        scanned = scan_directory(temp_image_dir, require_caption=False)
        manifest = create_manifest(scanned)
        # Find an entry with tags
        entry_with_tags = next((e for e in manifest.entries.values() if e.tags), None)
        assert entry_with_tags is not None
        assert len(entry_with_tags.tags) > 1


# =============================================================================
# Manifest Roundtrip Tests
# =============================================================================


@pytest.mark.unit
class TestManifestRoundtrip:
    """Test saving and loading manifests."""

    def test_save_and_load(self, temp_image_dir):
        """Manifest should survive save/load roundtrip."""
        scanned = scan_directory(temp_image_dir, require_caption=False)
        manifest = create_manifest(scanned, base_dir=temp_image_dir)

        # Save
        manifest_path = temp_image_dir / "dataset.json"
        save_dataset_manifest(manifest, manifest_path)
        assert manifest_path.exists()

        # Load
        loaded = load_dataset_manifest(manifest_path)

        # Verify
        assert len(loaded.entries) == len(manifest.entries)
        assert len(loaded.buckets) == len(manifest.buckets)
        assert loaded.base_resolution == manifest.base_resolution
        assert loaded.latent_channels == manifest.latent_channels
        assert loaded.latent_dtype == manifest.latent_dtype


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
        from library.data.pipeline.dataset_scanner import scan_metadata_file

        scanned = scan_metadata_file(
            temp_metadata_dir / "metadata.json",
            image_dir=temp_metadata_dir,
        )
        assert len(scanned) == 3

    def test_reads_caption_and_tags(self, temp_metadata_dir):
        """Should read captions and fall back to tags."""
        from library.data.pipeline.dataset_scanner import scan_metadata_file

        scanned = scan_metadata_file(
            temp_metadata_dir / "metadata.json",
            image_dir=temp_metadata_dir,
        )
        captions = {s.path.stem: s.caption for s in scanned}
        assert "first image caption" in captions["image_000"]
        assert "tag1" in captions["image_001"]  # Fell back to tags

    def test_uses_train_resolution(self, temp_metadata_dir):
        """Should use train_resolution from metadata if present."""
        from library.data.pipeline.dataset_scanner import scan_metadata_file

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
        from library.data.pipeline.dataset_scanner import scan_metadata_file

        # Create metadata with missing caption
        metadata = {"image_000": {}}  # No caption or tags
        (temp_metadata_dir / "no_caption.json").write_text(json.dumps(metadata), encoding="utf-8")

        with pytest.raises(ValueError, match="Missing captions"):
            scan_metadata_file(
                temp_metadata_dir / "no_caption.json",
                image_dir=temp_metadata_dir,
                require_caption=True,
            )
