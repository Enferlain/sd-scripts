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

from library.data.pipeline.dataclasses import CacheEntry, Bucket, DatasetManifest
from library.utils.common_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Supported image extensions (webp and jxl are popular for large datasets)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".jxl", ".bmp", ".gif", ".tiff", ".tif")

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


def get_image_size(path: Path) -> tuple[int, int]:
    """
    Get image dimensions without loading the full image.

    Uses imagesize library if available, falls back to PIL.

    Args:
        path: Path to image file.

    Returns:
        Tuple of (width, height).
    """
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
) -> list[ScannedImage]:
    """
    Scan a directory for images and their captions.

    Args:
        image_dir: Directory to scan.
        caption_extension: File extension for caption files.
        is_reg: Whether these are regularization images.
        num_repeats: Times to repeat each image per epoch.
        recursive: Whether to scan subdirectories.
        validation_split: Fraction of images to use for validation (0.0-1.0).
        validation_seed: Seed for deterministic validation split.

    Returns:
        List of ScannedImage objects.
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise ValueError(f"Directory does not exist: {image_dir}")

    # Find all image files
    if recursive:
        image_files = [f for f in image_dir.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS]
    else:
        image_files = [f for f in image_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]

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

    def process_image(idx: int, path: Path) -> ScannedImage | None:
        try:
            width, height = get_image_size(path)
            caption = read_caption(path, caption_extension)
            split = "val" if idx in val_indices else "train"
            return ScannedImage(
                path=path,
                width=width,
                height=height,
                caption=caption,
                num_repeats=num_repeats,
                is_reg=is_reg,
                split=split,
            )
        except Exception as e:
            logger.warning(f"Failed to process {path}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_image, i, f): f for i, f in enumerate(image_files)}
        for future in as_completed(futures):
            result = future.result()
            if result:
                scanned.append(result)

    # Sort by path for deterministic output
    scanned.sort(key=lambda x: x.path)
    return scanned


def make_bucket_resolutions(
    max_reso: tuple[int, int],
    min_size: int = 256,
    max_size: int = 1024,
    divisible: int = 64,
) -> list[tuple[int, int]]:
    """
    Generate bucket resolutions for training with aspect ratio bucketing.

    This is the same algorithm used in the legacy BucketManager.

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
