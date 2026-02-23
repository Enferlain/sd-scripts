import hashlib
import logging
from pathlib import Path
import numpy as np
import cv2

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


# DEPRECATED HELPERS TO AVOID LEGACY CRASHES


def load_image(image_path, alpha=False):
    try:
        with Image.open(image_path) as image:
            if alpha:
                if not image.mode == "RGBA":
                    image = image.convert("RGBA")
            else:
                if not image.mode == "RGB":
                    image = image.convert("RGB")
            img = np.array(image, np.uint8)
            return img
    except OSError as e:
        logger.error(f"Error loading file: {image_path}")
        raise e


# 画像を読み込む。戻り値はnumpy.ndarray,(original width, original height),(crop left, crop top, crop right, crop bottom)
def trim_and_resize_if_required(
    random_crop: bool,
    image: np.ndarray,
    reso,
    resized_size: tuple[int, int],
    resize_interpolation: str | None = None,
    random_crop_padding_percent: float = 0.05,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int, int, int]]:
    image_height, image_width = image.shape[0:2]
    original_size = (image_width, image_height)  # size before resize

    if random_crop:
        resized_size = (
            int(resized_size[0] * (1.0 + random_crop_padding_percent)),
            int(resized_size[1] * (1.0 + random_crop_padding_percent)),
        )

    if image_width != resized_size[0] or image_height != resized_size[1]:
        image = resize_image(image, image_width, image_height, resized_size[0], resized_size[1], resize_interpolation)

    image_height, image_width = image.shape[0:2]

    if image_width > reso[0]:
        trim_size = image_width - reso[0]
        p = trim_size // 2 if not random_crop else random.randint(0, trim_size)
        # logger.info(f"w {trim_size} {p}")
        image = image[:, p : p + reso[0]]
    if image_height > reso[1]:
        trim_size = image_height - reso[1]
        p = trim_size // 2 if not random_crop else random.randint(0, trim_size)
        # logger.info(f"h {trim_size} {p})
        image = image[p : p + reso[1]]

    # random cropの場合のcropされた値をどうcrop left/topに反映するべきか全くアイデアがない
    # I have no idea how to reflect the cropped value in crop left/top in the case of random crop

    crop_ltrb = BucketManager.get_crop_ltrb(reso, original_size)

    assert image.shape[0] == reso[1] and image.shape[1] == reso[0], f"internal error, illegal trimmed size: {image.shape}, {reso}"
    return image, original_size, crop_ltrb


def pil_resize(image, size, interpolation):
    has_alpha = image.shape[2] == 4 if len(image.shape) == 3 else False

    if has_alpha:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
    else:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    resized_pil = pil_image.resize(size, resample=interpolation)

    # Convert back to cv2 format
    if has_alpha:
        resized_cv2 = cv2.cvtColor(np.array(resized_pil), cv2.COLOR_RGBA2BGRA)
    else:
        resized_cv2 = cv2.cvtColor(np.array(resized_pil), cv2.COLOR_RGB2BGR)

    return resized_cv2


def resize_image(
    image: np.ndarray,
    width: int,
    height: int,
    resized_width: int,
    resized_height: int,
    resize_interpolation: str | None = None,
):
    """
    Resize image with resize interpolation. Default interpolation to AREA if image is smaller, else LANCZOS.

    Args:
        image: numpy.ndarray
        width: int Original image width
        height: int Original image height
        resized_width: int Resized image width
        resized_height: int Resized image height
        resize_interpolation: Optional[str] Resize interpolation method "lanczos", "area", "bilinear", "bicubic", "nearest", "box"

    Returns:
        image
    """

    # Ensure all size parameters are actual integers
    width = int(width)
    height = int(height)
    resized_width = int(resized_width)
    resized_height = int(resized_height)

    if resize_interpolation is None:
        if width >= resized_width and height >= resized_height:
            resize_interpolation = "area"
        else:
            resize_interpolation = "lanczos"

    # we use PIL for lanczos (for backward compatibility) and box, cv2 for others
    use_pil = resize_interpolation in ["lanczos", "lanczos4", "box"]

    resized_size = (resized_width, resized_height)
    if use_pil:
        interpolation = get_pil_interpolation(resize_interpolation)
        image = pil_resize(image, resized_size, interpolation=interpolation)
        logger.debug(f"resize image using {resize_interpolation} (PIL)")
    else:
        interpolation = get_cv2_interpolation(resize_interpolation)
        image = cv2.resize(image, resized_size, interpolation=interpolation)
        logger.debug(f"resize image using {resize_interpolation} (cv2)")

    return image


def get_cv2_interpolation(interpolation: str | None) -> int | None:
    """
    Convert interpolation value to cv2 interpolation integer

    https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121
    """
    if interpolation is None:
        return None

    if interpolation == "lanczos" or interpolation == "lanczos4":
        # Lanczos interpolation over 8x8 neighborhood
        return cv2.INTER_LANCZOS4
    elif interpolation == "nearest":
        # Bit exact nearest neighbor interpolation. This will produce same results as the nearest neighbor method in PIL, scikit-image or Matlab.
        return cv2.INTER_NEAREST_EXACT
    elif interpolation == "bilinear" or interpolation == "linear":
        # bilinear interpolation
        return cv2.INTER_LINEAR
    elif interpolation == "bicubic" or interpolation == "cubic":
        # bicubic interpolation
        return cv2.INTER_CUBIC
    elif interpolation == "area" or interpolation == "box":
        # resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire'-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
        return cv2.INTER_AREA
    else:
        return None


def get_pil_interpolation(interpolation: str | None) -> Image.Resampling | None:
    """
    Convert interpolation value to PIL interpolation

    https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-filters
    """
    if interpolation is None:
        return None

    if interpolation == "lanczos":
        return Image.Resampling.LANCZOS
    elif interpolation == "nearest":
        # Pick one nearest pixel from the input image. Ignore all other input pixels.
        return Image.Resampling.NEAREST
    elif interpolation == "bilinear" or interpolation == "linear":
        # For resize calculate the output pixel value using linear interpolation on all pixels that may contribute to the output value. For other transformations linear interpolation over a 2x2 environment in the input image is used.
        return Image.Resampling.BILINEAR
    elif interpolation == "bicubic" or interpolation == "cubic":
        # For resize calculate the output pixel value using cubic interpolation on all pixels that may contribute to the output value. For other transformations cubic interpolation over a 4x4 environment in the input image is used.
        return Image.Resampling.BICUBIC
    elif interpolation == "area":
        # Image.Resampling.BOX may be more appropriate if upscaling
        # Area interpolation is related to cv2.INTER_AREA
        # Produces a sharper image than Resampling.BILINEAR, doesn’t have dislocations on local level like with Resampling.BOX.
        return Image.Resampling.HAMMING
    elif interpolation == "box":
        # Each pixel of source image contributes to one pixel of the destination image with identical weights. For upscaling is equivalent of Resampling.NEAREST.
        return Image.Resampling.BOX
    else:
        return None


def validate_interpolation_fn(interpolation_str: str) -> bool:
    """
    Check if a interpolation function is supported
    """
    return interpolation_str in ["lanczos", "nearest", "bilinear", "linear", "bicubic", "cubic", "area", "box"]
