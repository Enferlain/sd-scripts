import pytest
import tempfile
from pathlib import Path
from PIL import Image


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
