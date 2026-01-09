"""
Unit tests for library/data/image_utils.py
"""

import hashlib
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image
import cv2

from library.data.image_utils import (
    get_image_size,
    check_has_alpha,
    generate_image_id,
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
        # Use Path to get relative path string for cross-platform compatibility
        rel_path_str = str(Path("subdir") / "image.png")
        expected_hash = hashlib.md5(rel_path_str.encode()).hexdigest()[:8]
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
