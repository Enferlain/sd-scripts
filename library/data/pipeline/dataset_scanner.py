"""
Dataset scanner for generating DatasetManifest from image directories.

This is Phase 1 of the data pipeline - scanning directories, reading captions,
computing bucket assignments, and generating a manifest for training.
"""

import hashlib
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from library.constants import IMAGE_EXTENSIONS
from library.data.pipeline.dataclasses import CacheEntry, Bucket, DatasetManifest
from library.config.dataclasses.data import DataConfig
from library.utils.common_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Caption file extensions to try, in priority order
CAPTION_EXTENSIONS = (".txt", ".caption")


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
    from PIL import Image

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
    from PIL import Image

    try:
        with Image.open(path) as img:
            return img.mode in ("RGBA", "LA", "PA")
    except Exception:
        return False


def read_caption(image_path: Path, caption_extension: str = ".txt") -> str:
    """
    Read caption for an image from associated text file.

    Tries the specified extension first, then falls back to other extensions.

    Args:
        image_path: Path to the image file.
        caption_extension: Primary caption file extension to try.

    Returns:
        Caption text, or empty string if not found.
    """
    stem = image_path.stem
    parent = image_path.parent

    # Try specified extension first
    extensions_to_try = [caption_extension] + [e for e in CAPTION_EXTENSIONS if e != caption_extension]

    for ext in extensions_to_try:
        caption_path = parent / f"{stem}{ext}"
        if caption_path.exists():
            try:
                return caption_path.read_text(encoding="utf-8").strip()
            except Exception as e:
                logger.warning(f"Failed to read caption {caption_path}: {e}")

    return ""


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


def make_bucket_resolutions(
    max_reso: tuple[int, int],
    min_size: int = 256,
    max_size: int = 1024,
    divisible: int = 64,
) -> list[tuple[int, int]]:
    """
    Generate bucket resolutions for training with aspect ratio bucketing.

    This is the same algorithm used in the _deprecated BucketManager.

    Args:
        max_reso: Tuple of (max_width, max_height) defining the maximum resolution.
        min_size: Minimum dimension size.
        max_size: Maximum dimension size.
        divisible: All dimensions must be divisible by this.

    Returns:
        Sorted list of (width, height) tuples representing bucket resolutions.
    """
    max_width, max_height = max_reso
    max_area = max_width * max_height

    resos = set()

    # Add square bucket
    width = int(math.sqrt(max_area) // divisible) * divisible
    resos.add((width, width))

    # Generate aspect ratio buckets
    width = min_size
    while width <= max_size:
        height = min(max_size, int((max_area // width) // divisible) * divisible)
        if height >= min_size:
            resos.add((width, height))
            resos.add((height, width))
        width += divisible

    resos_list = list(resos)
    resos_list.sort()
    return resos_list


def select_bucket(
    image_width: int,
    image_height: int,
    bucket_resos: list[tuple[int, int]],
    no_upscale: bool = False,
    max_area: int | None = None,
    reso_steps: int = 64,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Select the best bucket for an image.

    This is ported from BucketManager.select_bucket() for compatibility.

    Args:
        image_width: Original image width.
        image_height: Original image height.
        bucket_resos: List of available bucket resolutions.
        no_upscale: If True, never upscale images, only shrink.
        max_area: Maximum pixel area (for no_upscale mode).
        reso_steps: Resolution step size for rounding.

    Returns:
        Tuple of (bucket_reso, resized_size).
    """
    aspect_ratio = image_width / image_height

    if not no_upscale:
        # Standard mode: match to predefined bucket by aspect ratio
        bucket_resos_set = set(bucket_resos)
        reso = (image_width, image_height)

        if reso not in bucket_resos_set:
            # Find bucket with closest aspect ratio
            bucket_ars = [(w / h, (w, h)) for w, h in bucket_resos]
            best_match = min(bucket_ars, key=lambda x: abs(x[0] - aspect_ratio))
            reso = best_match[1]

        ar_reso = reso[0] / reso[1]
        if aspect_ratio > ar_reso:
            # Image is wider, match height
            scale = reso[1] / image_height
        else:
            scale = reso[0] / image_width

        resized_size = (int(image_width * scale + 0.5), int(image_height * scale + 0.5))
    else:
        # No upscale mode: only shrink, create custom bucket
        if max_area is None:
            max_area = max(w * h for w, h in bucket_resos) if bucket_resos else 1024 * 1024

        if image_width * image_height > max_area:
            # Shrink maintaining aspect ratio
            resized_width = math.sqrt(max_area * aspect_ratio)
            resized_height = max_area / resized_width

            # Round to reso_steps
            def round_to_steps(x):
                x = int(x + 0.5)
                return x - x % reso_steps

            b_width_rounded = round_to_steps(resized_width)
            b_height_in_wr = round_to_steps(b_width_rounded / aspect_ratio)
            ar_width_rounded = b_width_rounded / b_height_in_wr if b_height_in_wr else 0

            b_height_rounded = round_to_steps(resized_height)
            b_width_in_hr = round_to_steps(b_height_rounded * aspect_ratio)
            ar_height_rounded = b_width_in_hr / b_height_rounded if b_height_rounded else 0

            if abs(ar_width_rounded - aspect_ratio) < abs(ar_height_rounded - aspect_ratio):
                resized_size = (b_width_rounded, int(b_width_rounded / aspect_ratio + 0.5))
            else:
                resized_size = (int(b_height_rounded * aspect_ratio + 0.5), b_height_rounded)
        else:
            resized_size = (image_width, image_height)

        # Bucket size is resized size rounded down to reso_steps
        bucket_width = resized_size[0] - resized_size[0] % reso_steps
        bucket_height = resized_size[1] - resized_size[1] % reso_steps
        reso = (bucket_width, bucket_height)

    return reso, resized_size


def create_manifest(
    scanned_images: list[ScannedImage],
    base_dir: Path | None = None,
    base_resolution: tuple[int, int] = (1024, 1024),
    bucket_reso_steps: int = 64,
    min_bucket_reso: int = 256,
    max_bucket_reso: int = 2048,
    no_upscale: bool = False,
    latent_channels: int = 4,
    latent_scale_factor: int = 8,
    latent_dtype: str = "fp16",
) -> DatasetManifest:
    """
    Create a DatasetManifest from scanned images.

    Args:
        scanned_images: List of ScannedImage from scan_directory().
        base_dir: Base directory for relative path calculation in IDs.
        base_resolution: Base training resolution.
        bucket_reso_steps: Step size for bucket resolutions.
        min_bucket_reso: Minimum bucket dimension.
        max_bucket_reso: Maximum bucket dimension.
        no_upscale: If True, never upscale images.
        latent_channels: Number of VAE latent channels.
        latent_scale_factor: VAE spatial downscale factor.
        latent_dtype: Data type for cached latents.

    Returns:
        DatasetManifest ready to save or use for caching.
    """
    # Generate bucket resolutions
    bucket_resos = make_bucket_resolutions(
        max_reso=base_resolution,
        min_size=min_bucket_reso,
        max_size=max_bucket_reso,
        divisible=bucket_reso_steps,
    )
    max_area = base_resolution[0] * base_resolution[1]

    logger.info(f"Generated {len(bucket_resos)} bucket resolutions")

    # Process images and assign to buckets
    entries: dict[str, CacheEntry] = {}
    buckets: dict[str, Bucket] = {}

    for scanned in scanned_images:
        # Select bucket
        bucket_reso, resized_size = select_bucket(
            scanned.width,
            scanned.height,
            bucket_resos,
            no_upscale=no_upscale,
            max_area=max_area,
            reso_steps=bucket_reso_steps,
        )

        # Generate ID
        image_id = generate_image_id(scanned.path, base_dir)

        # Create entry
        entry = CacheEntry(
            id=image_id,
            image_path=str(scanned.path),
            original_size=(scanned.width, scanned.height),
            bucket_reso=bucket_reso,
            resized_size=resized_size,
            caption=scanned.caption,
            tags=[t.strip() for t in scanned.caption.split(",") if t.strip()] if scanned.caption else [],
            num_repeats=scanned.num_repeats,
            is_reg=scanned.is_reg,
            split=scanned.split,
            has_alpha_mask=scanned.has_alpha,
        )
        entries[image_id] = entry

        # Add to bucket
        bucket_key = f"{bucket_reso[0]}x{bucket_reso[1]}"
        if bucket_key not in buckets:
            buckets[bucket_key] = Bucket(resolution=bucket_reso)
        buckets[bucket_key].image_ids.append(image_id)

    # Create manifest
    manifest = DatasetManifest(
        version="2.0",
        created_at=datetime.now().isoformat(),
        base_resolution=base_resolution,
        bucket_reso_steps=bucket_reso_steps,
        min_bucket_reso=min_bucket_reso,
        max_bucket_reso=max_bucket_reso,
        latent_channels=latent_channels,
        latent_scale_factor=latent_scale_factor,
        latent_dtype=latent_dtype,
        entries=entries,
        buckets=buckets,
    )

    # Log statistics
    train_count = sum(1 for e in entries.values() if e.split == "train")
    val_count = sum(1 for e in entries.values() if e.split == "val")
    logger.info(f"Created manifest: {len(entries)} entries ({train_count} train, {val_count} val), {len(buckets)} buckets")

    return manifest


def create_manifest_from_config(
    data_config: DataConfig,
    cache_dir: str | Path | None = None,
    latent_channels: int = 4,
    latent_scale_factor: int = 8,
    latent_dtype: str = "fp16",
) -> DatasetManifest:
    """
    Create a DatasetManifest from DataConfig, handling all dataset sources.

    Supports:
    - train_data_dir: Main training images
    - reg_data_dir: Regularization images (is_reg=True)
    - in_json: FineTuning style metadata file
    - subsets: Multiple directories with individual settings

    Args:
        data_config: DataConfig containing source, preprocessing, caption, bucketing settings.
        cache_dir: Directory to store cache files.
        latent_channels: Number of VAE latent channels.
        latent_scale_factor: VAE spatial downscale factor.
        latent_dtype: Data type for cached latents.

    Returns:
        DatasetManifest ready for caching and training.
    """
    all_scanned: list[ScannedImage] = []
    caption_ext = data_config.caption.caption_extension or ".txt"

    # Handle train_data_dir (simple DreamBooth style)
    if data_config.source.train_data_dir:
        logger.info(f"Scanning train_data_dir: {data_config.source.train_data_dir}")
        scanned = scan_directory(
            data_config.source.train_data_dir,
            caption_extension=caption_ext,
            is_reg=False,
            num_repeats=data_config.source.dataset_repeats,
            alpha_mask=data_config.preprocessing.alpha_mask,
            require_caption=True,
        )
        all_scanned.extend(scanned)

    # Handle reg_data_dir (regularization images)
    if data_config.source.reg_data_dir:
        logger.info(f"Scanning reg_data_dir: {data_config.source.reg_data_dir}")
        scanned = scan_directory(
            data_config.source.reg_data_dir,
            caption_extension=caption_ext,
            is_reg=True,
            num_repeats=1,  # Reg images typically not repeated
            alpha_mask=data_config.preprocessing.alpha_mask,
            require_caption=False,  # Reg often uses class_tokens instead
        )
        all_scanned.extend(scanned)

    # Handle in_json (FineTuning style metadata)
    if data_config.source.in_json:
        logger.info(f"Scanning metadata file: {data_config.source.in_json}")
        scanned = scan_metadata_file(
            data_config.source.in_json,
            image_dir=data_config.source.train_data_dir,  # Use train_data_dir as base
            num_repeats=data_config.source.dataset_repeats,
            alpha_mask=data_config.preprocessing.alpha_mask,
            require_caption=True,
        )
        all_scanned.extend(scanned)

    # Handle subsets
    for subset in data_config.source.subsets:
        image_dir = subset.get("image_dir")
        if not image_dir:
            logger.warning("Subset missing image_dir, skipping")
            continue

        logger.info(f"Scanning subset: {image_dir}")
        scanned = scan_directory(
            image_dir,
            caption_extension=subset.get("caption_extension", caption_ext),
            is_reg=subset.get("is_reg", False),
            num_repeats=subset.get("num_repeats", data_config.source.dataset_repeats),
            alpha_mask=subset.get("alpha_mask", data_config.preprocessing.alpha_mask),
            class_tokens=subset.get("class_tokens"),
            require_caption=not subset.get("is_reg", False),
        )
        all_scanned.extend(scanned)

    if not all_scanned:
        raise ValueError("No images found. Specify at least one of: train_data_dir, reg_data_dir, in_json, or subsets")

    # Parse resolution
    base_resolution = (1024, 1024)
    if data_config.preprocessing.resolution:
        parts = data_config.preprocessing.resolution.replace("x", ",").split(",")
        if len(parts) == 2:
            base_resolution = (int(parts[0]), int(parts[1]))
        else:
            side = int(parts[0])
            base_resolution = (side, side)

    # Determine base_dir for ID generation
    base_dir = None
    if data_config.source.train_data_dir:
        base_dir = Path(data_config.source.train_data_dir).parent

    return create_manifest(
        all_scanned,
        base_dir=base_dir,
        base_resolution=base_resolution,
        bucket_reso_steps=data_config.bucketing.bucket_reso_steps,
        min_bucket_reso=data_config.bucketing.min_bucket_reso,
        max_bucket_reso=data_config.bucketing.max_bucket_reso,
        no_upscale=data_config.bucketing.bucket_no_upscale,
        latent_channels=latent_channels,
        latent_scale_factor=latent_scale_factor,
        latent_dtype=latent_dtype,
    )
