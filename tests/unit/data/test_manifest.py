"""
Unit tests for the manifest module.

Tests manifest creation, saving, loading, and config integration.
"""

import pytest
import tempfile
from pathlib import Path
from PIL import Image

from library.data.manifest import (
    create_manifest,
    create_manifest_from_config,
    save_dataset_manifest,
    load_dataset_manifest,
)
from library.data.scanners import scan_directory
from library.config.dataclasses.data import DataConfig


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
# create_manifest_from_config Tests
# =============================================================================


@pytest.mark.unit
class TestCreateManifestFromConfig:
    """Test high-level config-driven manifest creation."""

    def test_handles_train_data_dir(self, temp_image_dir):
        """Should scan train_data_dir from config (only captioned images)."""
        # Add caption for no_caption.png so all images are captioned
        (temp_image_dir / "no_caption.txt").write_text("added caption")

        config = DataConfig()
        config.source.train_data_dir = str(temp_image_dir)
        config.preprocessing.resolution = "512,512"
        config.caption.caption_extension = ".txt"

        manifest = create_manifest_from_config(config)
        assert len(manifest.entries) == 5  # All images in temp_image_dir

    def test_handles_reg_data_dir(self, temp_image_dir):
        """Should scan reg_data_dir as is_reg=True."""
        config = DataConfig()
        config.source.reg_data_dir = str(temp_image_dir)
        config.preprocessing.resolution = "512,512"

        manifest = create_manifest_from_config(config)
        # All should be marked as regularization images
        assert all(e.is_reg for e in manifest.entries.values())
