import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from library.constants import IMAGE_EXTENSIONS
from library.data.caption_processor import read_caption
from library.data.image_utils import get_image_size, check_has_alpha
from library.utils.common_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@dataclass
class ScannedImage:
    """Intermediate representation of a scanned image before manifest creation."""

    path: Path
    """Absolute path to the image file."""

    width: int
    """Image width in pixels."""

    height: int
    """Image height in pixels."""

    caption: str = ""
    """Caption text."""

    num_repeats: int = 1
    """Number of times to repeat this image per epoch."""

    is_reg: bool = False
    """Whether this is a regularization image."""

    split: str = "train"
    """Dataset split: 'train' or 'val'."""

    has_alpha: bool = False
    """Whether this image has an alpha channel (RGBA)."""


def scan_directory(
    image_dir: str | Path,
    caption_extension: str = ".txt",
    is_reg: bool = False,
    num_repeats: int = 1,
    recursive: bool = True,
    validation_split: float = 0.0,
    validation_seed: int | None = None,
    require_caption: bool = True,
    alpha_mask: bool = False,
    class_tokens: str | None = None,
) -> list[ScannedImage]:
    """
    Scan a directory for images and their captions (DreamBooth style).

    Args:
        image_dir: Directory to scan.
        caption_extension: File extension for caption files.
        is_reg: Whether these are regularization images.
        num_repeats: Times to repeat each image per epoch.
        recursive: Whether to scan subdirectories.
        validation_split: Fraction of images to use for validation (0.0-1.0).
        validation_seed: Seed for deterministic validation split.
        require_caption: If True, raise error when non-reg images have no caption.
        alpha_mask: If True, check images for alpha channel (slower but enables mask training).
        class_tokens: Default caption to use when no caption file exists.

    Returns:
        List of ScannedImage objects.

    Raises:
        ValueError: If require_caption is True and images are missing captions.
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise ValueError(f"Directory does not exist: {image_dir}")

    # Find all image files - IMAGE_EXTENSIONS is a list from constants, normalize to lowercase for matching
    image_extensions_lower = {ext.lower() for ext in IMAGE_EXTENSIONS}
    if recursive:
        image_files = [f for f in image_dir.rglob("*") if f.suffix.lower() in image_extensions_lower]
    else:
        image_files = [f for f in image_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions_lower]

    image_files.sort()  # Deterministic ordering
    logger.info(f"Found {len(image_files)} images in {image_dir}")

    # Determine validation split
    val_indices = set()
    if validation_split > 0:
        import random

        rng = random.Random(validation_seed)
        num_val = int(len(image_files) * validation_split)
        val_indices = set(rng.sample(range(len(image_files)), num_val))
        logger.info(f"Validation split: {num_val} images ({validation_split * 100:.1f}%)")

    # Scan images in parallel
    scanned = []
    missing_captions = []

    def process_image(idx: int, path: Path) -> ScannedImage | None:
        try:
            width, height = get_image_size(path)
            caption = read_caption(path, caption_extension)
            # Use class_tokens as fallback if no caption file found
            if not caption and class_tokens:
                caption = class_tokens
            split = "val" if idx in val_indices else "train"
            has_alpha = check_has_alpha(path) if alpha_mask else False
            return ScannedImage(
                path=path,
                width=width,
                height=height,
                caption=caption,
                num_repeats=num_repeats,
                is_reg=is_reg,
                split=split,
                has_alpha=has_alpha,
            )
        except Exception as e:
            logger.warning(f"Failed to process {path}: {e}")
            return None
            return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_image, i, f): f for i, f in enumerate(image_files)}
        for future in as_completed(futures):
            result = future.result()
            if result:
                scanned.append(result)
                # Track images without captions
                if not result.caption and not result.is_reg:
                    missing_captions.append(result.path)

    # Validate captions
    if require_caption and missing_captions and not is_reg:
        raise ValueError(
            f"Missing captions for {len(missing_captions)} images. "
            f"First 5: {[str(p) for p in missing_captions[:5]]}. "
            f"Use require_caption=False to allow images without captions, "
            f"or set is_reg=True for regularization images."
        )

    # Sort by path for deterministic output
    scanned.sort(key=lambda x: x.path)
    return scanned


def scan_metadata_file(
    metadata_file: str | Path,
    image_dir: str | Path | None = None,
    num_repeats: int = 1,
    validation_split: float = 0.0,
    validation_seed: int | None = None,
    require_caption: bool = True,
    alpha_mask: bool = False,
) -> list[ScannedImage]:
    """
    Scan a JSON metadata file for images and captions (FineTuning style).

    The JSON format is: {image_key: {"caption": "...", "tags": "...", "train_resolution": [w, h]}, ...}

    Args:
        metadata_file: Path to JSON metadata file.
        image_dir: Optional base directory for relative image paths.
        num_repeats: Times to repeat each image per epoch.
        validation_split: Fraction of images to use for validation (0.0-1.0).
        validation_seed: Seed for deterministic validation split.
        require_caption: If True, raise error when images have no caption/tags.
        alpha_mask: If True, check images for alpha channel (slower but enables mask training).

    Returns:
        List of ScannedImage objects.

    Raises:
        ValueError: If metadata file doesn't exist or images are missing.
        ValueError: If require_caption is True and images are missing captions.
    """
    import json

    metadata_file = Path(metadata_file)
    if not metadata_file.exists():
        raise ValueError(f"Metadata file does not exist: {metadata_file}")

    with open(metadata_file, encoding="utf-8") as f:
        metadata = json.load(f)

    if not metadata:
        raise ValueError(f"Metadata file is empty: {metadata_file}")

    logger.info(f"Loaded metadata with {len(metadata)} entries from {metadata_file}")

    # Determine validation split
    image_keys = list(metadata.keys())
    image_keys.sort()  # Deterministic ordering

    val_indices = set()
    if validation_split > 0:
        import random

        rng = random.Random(validation_seed)
        num_val = int(len(image_keys) * validation_split)
        val_indices = set(rng.sample(range(len(image_keys)), num_val))
        logger.info(f"Validation split: {num_val} images ({validation_split * 100:.1f}%)")

    scanned = []
    missing_images = []
    missing_captions = []

    for idx, image_key in enumerate(image_keys):
        img_md = metadata[image_key]

        # Resolve image path
        abs_path = _resolve_image_path(image_key, Path(image_dir) if image_dir else None)
        if abs_path is None:
            missing_images.append(image_key)
            continue

        # Get caption (caption takes priority, fallback to tags)
        caption = img_md.get("caption") or img_md.get("tags") or ""

        # Get resolution if available
        train_reso = img_md.get("train_resolution")
        if train_reso:
            width, height = train_reso[0], train_reso[1]
        else:
            # Need to read from file
            width, height = get_image_size(abs_path)

        split = "val" if idx in val_indices else "train"
        has_alpha = check_has_alpha(abs_path) if alpha_mask else False

        scanned.append(
            ScannedImage(
                path=abs_path,
                width=width,
                height=height,
                caption=caption,
                num_repeats=num_repeats,
                is_reg=False,
                split=split,
                has_alpha=has_alpha,
            )
        )

        if not caption:
            missing_captions.append(image_key)

    if missing_images:
        logger.warning(f"Could not find {len(missing_images)} images: {missing_images[:5]}...")

    # Validate captions
    if require_caption and missing_captions:
        raise ValueError(
            f"Missing captions for {len(missing_captions)} images in metadata. "
            f"First 5: {missing_captions[:5]}. "
            f"Use require_caption=False to allow images without captions."
        )

    logger.info(f"Scanned {len(scanned)} images from metadata")
    return scanned


def _resolve_image_path(image_key: str, image_dir: Path | None) -> Path | None:
    """
    Resolve an image key to an absolute path.

    Tries:
    1. image_key as absolute path
    2. image_key relative to image_dir
    3. Glob search in image_dir

    Args:
        image_key: Image key from metadata (can be path or filename).
        image_dir: Optional base directory.

    Returns:
        Absolute path to image, or None if not found.
    """
    # Try as absolute path
    if Path(image_key).exists():
        return Path(image_key)

    if image_dir is None:
        return None

    image_dir = Path(image_dir)

    # Try as relative path
    rel_path = image_dir / image_key
    if rel_path.exists():
        return rel_path

    # Try adding common extensions
    for ext in IMAGE_EXTENSIONS:
        with_ext = image_dir / f"{image_key}{ext}"
        if with_ext.exists():
            return with_ext

    # Try glob search
    matches = list(image_dir.glob(f"{image_key}.*"))
    if matches:
        return matches[0]

    return None
