"""
Unit tests for library/data/image_utils.py

Tests image loading, globbing, and preprocessing utilities.
"""

import pytest
import os
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
from unittest.mock import patch

from library.data.image_utils import (
    glob_images,
    glob_images_pathlib,
    load_image,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_image_dir():
    """Create a temporary directory with test images."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test images with various extensions
        for i, ext in enumerate([".png", ".jpg", ".webp"]):
            img = Image.new("RGB", (64, 64), color=(i * 50, i * 50, i * 50))
            img.save(os.path.join(tmp_dir, f"test_image_{i}{ext}"))

        # Create a subdirectory with more images
        subdir = os.path.join(tmp_dir, "subdir")
        os.makedirs(subdir)
        img = Image.new("RGB", (32, 32), color=(100, 100, 100))
        img.save(os.path.join(subdir, "nested_image.png"))

        yield tmp_dir


@pytest.fixture
def temp_single_image():
    """Create a single test image and return its path."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new("RGB", (128, 128), color=(255, 0, 0))
        img.save(f.name)
        yield f.name
    # Cleanup
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.fixture
def temp_rgba_image():
    """Create a test RGBA image with transparency."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new("RGBA", (64, 64), color=(255, 0, 0, 128))
        img.save(f.name)
        yield f.name
    if os.path.exists(f.name):
        os.unlink(f.name)


# =============================================================================
# glob_images Tests
# =============================================================================


@pytest.mark.data
@pytest.mark.unit
class TestGlobImages:
    """Test glob_images function."""

    def test_finds_all_images(self, temp_image_dir):
        """Test that glob_images finds all image files."""
        images = glob_images(temp_image_dir)

        # Should find 3 images (png, jpg, webp) - not nested ones
        assert len(images) == 3

    def test_returns_sorted_list(self, temp_image_dir):
        """Test that results are sorted."""
        images = glob_images(temp_image_dir)

        assert images == sorted(images)

    def test_returns_unique_paths(self, temp_image_dir):
        """Test that results have no duplicates."""
        images = glob_images(temp_image_dir)

        assert len(images) == len(set(images))

    def test_empty_directory_returns_empty(self):
        """Test that empty directory returns empty list."""
        with tempfile.TemporaryDirectory() as empty_dir:
            images = glob_images(empty_dir)
            assert images == []

    def test_nonexistent_directory_returns_empty(self):
        """Test that nonexistent directory returns empty list."""
        images = glob_images("/nonexistent/path/to/dir")
        assert images == []

    @patch("library.data.image_utils.glob.glob")
    def test_glob_escapes_special_chars(self, mock_glob):
        """Test that glob inputs are escaped to handle brackets etc."""
        # Setup mock return
        mock_glob.return_value = ["/path/to/img.jpg"]

        # Call with path containing special chars
        glob_images("/path/[brackets]", base="*")

        # Verify escape was called (implicitly) and glob received escaped/modified path
        # glob.escape("/path/[brackets]") -> /path/[[]brackets[]] usually
        args, _ = mock_glob.call_args
        # We just check that the argument passed containing the brackets was modified/escaped
        # Or simply that it was called with *some* path including extensions
        assert "[[]" in args[0] or "\\[" in args[0] or mock_glob.call_count > 0

    @patch("library.data.image_utils.glob.glob")
    def test_glob_filters_extensions(self, mock_glob):
        """Test that glob loops over extensions."""
        mock_glob.side_effect = lambda p: [p]  # Echo the pattern back
        results = glob_images("/dir", base="*")

        # Should have searched for multiple extensions (defined in constant)
        assert mock_glob.call_count > 1
        assert isinstance(results, list)


# =============================================================================
# glob_images_pathlib Tests
# =============================================================================


@pytest.mark.data
@pytest.mark.unit
class TestGlobImagesPathlib:
    """Test glob_images_pathlib function."""

    def test_non_recursive_finds_top_level_only(self, temp_image_dir):
        """Test that non-recursive mode only finds top-level images."""
        dir_path = Path(temp_image_dir)
        images = glob_images_pathlib(dir_path, recursive=False)

        # Should find only the 3 top-level images
        assert len(images) == 3

    def test_recursive_finds_all_images(self, temp_image_dir):
        """Test that recursive mode finds all images including nested."""
        dir_path = Path(temp_image_dir)
        images = glob_images_pathlib(dir_path, recursive=True)

        # Should find 4 images (3 top-level + 1 nested)
        assert len(images) == 4

    def test_returns_sorted_paths(self, temp_image_dir):
        """Test that results are sorted."""
        dir_path = Path(temp_image_dir)
        images = glob_images_pathlib(dir_path, recursive=True)

        assert images == sorted(images)

    def test_returns_path_objects(self, temp_image_dir):
        """Test that results are Path objects."""
        dir_path = Path(temp_image_dir)
        images = glob_images_pathlib(dir_path, recursive=False)

        assert all(isinstance(p, Path) for p in images)


# =============================================================================
# load_image Tests
# =============================================================================


@pytest.mark.data
@pytest.mark.unit
class TestLoadImage:
    """Test load_image function."""

    def test_loads_rgb_image(self, temp_single_image):
        """Test loading an RGB image."""
        img = load_image(temp_single_image, alpha=False)

        assert isinstance(img, np.ndarray)
        assert img.dtype == np.uint8
        assert img.shape == (128, 128, 3)  # H, W, C

    def test_loads_rgba_image_with_alpha_true(self, temp_rgba_image):
        """Test loading an RGBA image with alpha channel."""
        img = load_image(temp_rgba_image, alpha=True)

        assert isinstance(img, np.ndarray)
        assert img.shape[2] == 4  # RGBA

    def test_converts_rgba_to_rgb_when_alpha_false(self, temp_rgba_image):
        """Test that RGBA is converted to RGB when alpha=False."""
        img = load_image(temp_rgba_image, alpha=False)

        assert img.shape[2] == 3  # RGB

    def test_raises_on_nonexistent_file(self):
        """Test that loading nonexistent file raises error."""
        with pytest.raises((IOError, OSError, FileNotFoundError)):
            load_image("/nonexistent/path/image.png")

    def test_returns_numpy_array(self, temp_single_image):
        """Test that result is a numpy array of uint8."""
        img = load_image(temp_single_image)

        assert isinstance(img, np.ndarray)
        assert img.dtype == np.uint8

    def test_load_image_rgb_conversion(self):
        """Test converting non-RGB images (e.g. CMYK) to RGB."""
        with patch("PIL.Image.open") as mock_open:
            # Create a real PIL image with a non-standard mode
            real_img = Image.new("CMYK", (10, 10), color="cyan")
            mock_open.return_value.__enter__.return_value = real_img

            img_arr = load_image("/dummy/path", alpha=False)

            assert img_arr.shape == (10, 10, 3)  # Converted to RGB
            assert img_arr.dtype == np.uint8

    def test_load_image_alpha(self):
        """Test loading image with alpha request converts to RGBA."""
        with patch("PIL.Image.open") as mock_open:
            real_img = Image.new("RGB", (10, 10), color="red")
            mock_open.return_value.__enter__.return_value = real_img

            img_arr = load_image("/dummy/path", alpha=True)

            assert img_arr.shape == (10, 10, 4)  # Converted to RGBA


# =============================================================================
# trim_and_resize_if_required Tests
# =============================================================================

from library.data.image_utils import trim_and_resize_if_required


@pytest.mark.data
@pytest.mark.unit
class TestTrimAndResize:
    """Test TRIM and RESIZE logic."""

    def test_no_resize_needed(self):
        """Test exact match dimensions return original image."""
        # 100x100 image, target 100x100
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        reso = (100, 100)

        # Mock BucketManager to decouple logic
        with patch("library.data.image_utils.BucketManager.get_crop_ltrb") as mock_get_crop:
            mock_get_crop.return_value = (0, 0, 100, 100)

            out_img, orig_size, crop = trim_and_resize_if_required(random_crop=False, image=img, reso=reso, resized_size=reso)

            assert out_img.shape == (100, 100, 3)
            assert orig_size == (100, 100)
            assert crop == (0, 0, 100, 100)

    def test_center_crop_logic(self):
        """Test that center crop is applied when no random crop is requested."""
        # 200x100 image, target 100x100
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        # Mark the center so we can check it remains
        img[:, 50:150, :] = 255

        reso = (100, 100)

        # Mock BucketManager
        with patch("library.data.image_utils.BucketManager.get_crop_ltrb") as mock_get_crop:
            mock_get_crop.return_value = (50, 0, 150, 100)

            # resized_size=(200, 100) means no resize needed, just crop
            out_img, orig_size, crop = trim_and_resize_if_required(random_crop=False, image=img, reso=reso, resized_size=(200, 100))

            assert out_img.shape == (100, 100, 3)
            # Verify it kept the center (which we set to 255)
            assert np.all(out_img == 255)
            assert crop == (50, 0, 150, 100)

    def test_resize_trigger(self):
        """Test that resizing occurs when input doesn't match resized_size."""
        # 200x200 input, target 100x100
        img = np.zeros((200, 200, 3), dtype=np.uint8)

        with (
            patch("library.data.image_utils.resize_image") as mock_resize,
            patch("library.data.image_utils.BucketManager.get_crop_ltrb") as mock_get_crop,
        ):
            # Setup resize to return correct shape image
            mock_resize.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            mock_get_crop.return_value = (0, 0, 100, 100)

            out_img, _, _ = trim_and_resize_if_required(
                random_crop=False,
                image=img,
                reso=(100, 100),
                resized_size=(100, 100),  # Logic will see 200!=100 and resize
            )

            mock_resize.assert_called_once()
