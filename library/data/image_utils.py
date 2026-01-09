import hashlib
import logging
from pathlib import Path

from PIL import Image

from library.utils.common_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Caption file extensions to try, in priority order
CAPTION_EXTENSIONS = (".txt", ".caption")


def get_image_size(path: Path) -> tuple[int, int]:
    """
    Get image dimensions without loading the full image.

    Uses specialized handlers for JXL (fast header parsing),
    imagesize library if available, otherwise falls back to PIL.

    Args:
        path: Path to image file.

    Returns:
        Tuple of (width, height).
    """
    suffix = path.suffix.lower()

    # Use specialized JXL parser for up to 200x speedup
    if suffix == ".jxl":
        try:
            from library.utils.jpeg_xl_util import get_jxl_size

            return get_jxl_size(str(path))
        except Exception:
            pass  # Fall through to other methods

    # Try imagesize library
    try:
        import imagesize

        width, height = imagesize.get(str(path))
        if width > 0 and height > 0:
            return width, height
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback to PIL
    with Image.open(path) as img:
        return img.size


def check_has_alpha(path: Path) -> bool:
    """
    Check if an image has an alpha channel.

    Opens the image briefly to check its mode.

    Args:
        path: Path to image file.

    Returns:
        True if image has alpha channel (RGBA, LA, PA modes).
    """
    try:
        with Image.open(path) as img:
            return img.mode in ("RGBA", "LA", "PA")
    except Exception:
        return False


def generate_image_id(path: Path, base_dir: Path | None = None) -> str:
    """
    Generate a unique ID for an image.

    Uses a short hash of the relative path for uniqueness while keeping IDs readable.

    Args:
        path: Absolute path to image.
        base_dir: Base directory for relative path calculation.

    Returns:
        Unique identifier string.
    """
    if base_dir:
        rel_path = path.relative_to(base_dir)
    else:
        rel_path = path

    # Create hash from relative path
    path_hash = hashlib.md5(str(rel_path).encode()).hexdigest()[:8]
    stem = path.stem[:32]  # Limit stem length
    return f"{stem}_{path_hash}"
