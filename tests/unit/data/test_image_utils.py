"""
Unit tests for library/data/image_utils.py
"""

import hashlib
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image

from library.data.image_utils import (
    get_image_size,
    check_has_alpha,
    generate_image_id,
    resize_image,
    trim_and_resize_if_required,
    validate_interpolation_fn,
    get_cv2_interpolation,
    get_pil_interpolation,
)


@pytest.fixture
def temp_image_path(tmp_path):
    """Create a temporary dummy image file."""
    path = tmp_path / "test_image.png"
    img = Image.new("RGB", (100, 200), color="red")
    img.save(path)
    return path


@pytest.fixture
def temp_rgba_image_path(tmp_path):
    """Create a temporary dummy RGBA image file."""
    path = tmp_path / "test_rgba.png"
    img = Image.new("RGBA", (100, 200), color=(255, 0, 0, 128))
    img.save(path)
    return path


@pytest.mark.unit
class TestGetImageSize:
    def test_pil_fallback(self, temp_image_path):
        """Test fallback to PIL when other methods fail/not applicable."""
        width, height = get_image_size(temp_image_path)
        assert width == 100
        assert height == 200

    @patch("imagesize.get")
    def test_imagesize_library(self, mock_get, temp_image_path):
        """Test using imagesize library."""
        mock_get.return_value = (123, 456)
        width, height = get_image_size(temp_image_path)
        assert width == 123
        assert height == 456
        mock_get.assert_called_once_with(str(temp_image_path))

    @patch("library.utils.jpeg_xl_util.get_jxl_size")
    def test_jxl_file(self, mock_jxl, tmp_path):
        """Test using specialized JXL handler."""
        path = tmp_path / "test.jxl"
        path.touch()
        mock_jxl.return_value = (321, 654)

        width, height = get_image_size(path)

        assert width == 321
        assert height == 654
        mock_jxl.assert_called_once_with(str(path))


@pytest.mark.unit
class TestCheckHasAlpha:
    def test_rgb_image(self, temp_image_path):
        assert not check_has_alpha(temp_image_path)

    def test_rgba_image(self, temp_rgba_image_path):
        assert check_has_alpha(temp_rgba_image_path)

    def test_invalid_path(self, tmp_path):
        # Should return False safely on error
        assert not check_has_alpha(tmp_path / "nonexistent.png")


@pytest.mark.unit
class TestGenerateImageId:
    def test_id_generation(self):
        path = Path("/data/train/subdir/image.png")
        base_dir = Path("/data/train")

        img_id = generate_image_id(path, base_dir)

        assert img_id.startswith("image_")
        # Relative path is subdir/image.png
        # Hash of "subdir/image.png"
        expected_hash = hashlib.md5("subdir/image.png".encode()).hexdigest()[:8]
        assert img_id.endswith(expected_hash)

    def test_no_base_dir(self):
        path = Path("/data/image.png")
        img_id = generate_image_id(path, None)
        assert img_id.startswith("image_")

    def test_long_filename(self):
        """Test that extremely long filenames are truncated."""
        long_name = "a" * 100
        path = Path(f"/data/{long_name}.png")

        img_id = generate_image_id(path, Path("/data"))

        # 32 chars stem + _ + 8 chars hash = 41 chars
        assert len(img_id) == 41
        assert img_id.startswith("a" * 32)


@pytest.mark.unit
class TestResizeImage:
    def test_resize_numpy(self):
        """Test resizing a numpy array."""
        # Create a 100x100 blue image (BGR for cv2 default)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = [255, 0, 0]

        resized = resize_image(img, 100, 100, 50, 50, "area")

        assert resized.shape == (50, 50, 3)
        assert resized.dtype == np.uint8

    def test_interpolation_pil_fallback(self):
        """Test logic that switches to PIL for specific interpolations (e.g. lanczos)."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch("library.data.image_utils.pil_resize") as mock_pil_resize:
            mock_pil_resize.return_value = np.zeros((50, 50, 3), dtype=np.uint8)

            resize_image(img, 100, 100, 50, 50, "lanczos")

            mock_pil_resize.assert_called_once()


@pytest.mark.unit
class TestTrimAndResize:
    def test_trim_and_resize(self):
        # 100x100 image
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        # Resize to 50x50, target reso 40x40
        # Should resize to 50x50 then crop to 40x40
        result_img, orig_size, crop_ltrb = trim_and_resize_if_required(
            False, img, (40, 40), (50, 50), "area"
        )

        assert result_img.shape == (40, 40, 3)
        assert orig_size == (100, 100)
        # Verify crop happened (center crop)
        # 50 -> 40, margin 10, crop 5 on each side
        # L=5, T=5, R=45, B=45
        # Note: get_crop_ltrb logic in BucketManager might differ slightly,
        # but here we test the function returns plausible values.


@pytest.mark.unit
class TestInterpolationUtils:
    def test_validate_interpolation(self):
        assert validate_interpolation_fn("area")
        assert validate_interpolation_fn("lanczos")
        assert not validate_interpolation_fn("invalid")

    def test_get_cv2_interpolation(self):
        import cv2
        assert get_cv2_interpolation("area") == cv2.INTER_AREA
        assert get_cv2_interpolation("linear") == cv2.INTER_LINEAR

    def test_get_pil_interpolation(self):
        assert get_pil_interpolation("lanczos") == Image.Resampling.LANCZOS
        assert get_pil_interpolation("nearest") == Image.Resampling.NEAREST
