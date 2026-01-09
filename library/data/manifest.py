"""
Manifest creation, reading, and writing utilities.

Handles:
- DatasetManifest and EpochManifest creation from scanned images
- JSON serialization/deserialization
- Config hash computation for cache invalidation
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from library.config.dataclasses.data import DataConfig
from library.data.bucketing import make_bucket_resolutions, select_bucket
from library.data.caption_processor import _parse_tags
from library.data.structures import DatasetManifest, CacheEntry, Bucket, EpochManifest, BatchInfo
from library.data.image_utils import generate_image_id
from library.data.scanners import ScannedImage, scan_directory, scan_metadata_file
from library.utils.common_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


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
    caption_separator: str = ", ",
    keep_tokens_separator: str = "",
    cache_dir: str | None = None,
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
        caption_separator: Separator for splitting caption into tags (default ", ").
        keep_tokens_separator: Separator marking fixed token regions (e.g. "|||").
        cache_dir: Directory for cache files. If set, entry paths are computed at creation.

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

        # Parse tags respecting keep_tokens_separator
        tags = _parse_tags(scanned.caption, caption_separator, keep_tokens_separator)

        # Create entry with cache paths if cache_dir provided
        entry = CacheEntry(
            id=image_id,
            image_path=str(scanned.path),
            original_size=(scanned.width, scanned.height),
            bucket_reso=bucket_reso,
            resized_size=resized_size,
            caption=scanned.caption,
            tags=tags,
            num_repeats=scanned.num_repeats,
            is_reg=scanned.is_reg,
            split=scanned.split,
            has_alpha_mask=scanned.has_alpha,
        )

        # Set cache paths upfront if cache_dir is known
        if cache_dir:
            entry.latent_cache_path = f"{cache_dir}/{image_id}_latent.safetensors"
            entry.te_cache_path = f"{cache_dir}/{image_id}_te.safetensors"

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
        cache_dir=cache_dir or "",
        total_images=len(entries),
        total_captions=sum(1 for e in entries.values() if e.caption),
        entries=entries,
        buckets=buckets,
    )

    # Log statistics
    train_count = sum(1 for e in entries.values() if e.split == "train")
    val_count = sum(1 for e in entries.values() if e.split == "val")
    logger.info(f"Created manifest: {len(entries)} entries ({train_count} train, {val_count} val), {len(buckets)} buckets")

    # Log per-bucket details (like legacy dataset.py)
    if buckets:
        logger.info(f"Bucket distribution ({len(buckets)} filled out of {len(bucket_resos)}):")
        sorted_buckets = sorted(buckets.items(), key=lambda x: (x[1].resolution[0], x[1].resolution[1]))
        for i, (_, bucket) in enumerate(sorted_buckets):
            logger.info(f"  bucket {i}: resolution {bucket.resolution}, count: {len(bucket.image_ids)}")

        # Calculate mean aspect ratio error
        ar_errors = []
        for entry in entries.values():
            original_ar = entry.original_size[0] / entry.original_size[1]
            bucket_ar = entry.bucket_reso[0] / entry.bucket_reso[1]
            ar_errors.append(abs(original_ar - bucket_ar))
        if ar_errors:
            mean_ar_error = sum(ar_errors) / len(ar_errors)
            logger.info(f"  mean ar error (without repeats): {mean_ar_error:.6f}")

    return manifest


def create_manifest_from_config(
    data_config: DataConfig,
    cache_dir: str | Path | None = None,
    latent_channels: int = 4,
    latent_scale_factor: int = 8,
    latent_dtype: str = "fp16",
    validation: bool = False,
    validation_split: float = 0.0,
    validation_seed: int | None = None,
) -> DatasetManifest:
    """
    Create a DatasetManifest from DataConfig, handling all dataset sources.

    Supports:
    - train_data_dir: Main training images
    - reg_data_dir: Regularization images (is_reg=True)
    - val_data_dir: Separate validation images (used when validation=True)
    - in_json: FineTuning style metadata file
    - subsets: Multiple directories with individual settings
    - validation_split: Split training data for validation (if val_data_dir not set)

    Args:
        data_config: DataConfig containing source, preprocessing, caption, bucketing settings.
        cache_dir: Directory to store cache files.
        latent_channels: Number of VAE latent channels.
        latent_scale_factor: VAE spatial downscale factor.
        latent_dtype: Data type for cached latents.
        validation: If True, create manifest for validation data only.
        validation_split: Fraction of training data to use for validation (0.0-1.0).
        validation_seed: Seed for deterministic validation split.

    Returns:
        DatasetManifest ready for caching and training.

    Raises:
        ValueError: If validation=True but no val_data_dir is configured.
    """
    all_scanned: list[ScannedImage] = []
    caption_ext = data_config.caption.caption_extension or ".txt"

    # Validation mode: only scan val_data_dir
    if validation:
        if not data_config.source.val_data_dir:
            raise ValueError("validation=True but val_data_dir is not configured in data_config.source")
        logger.info(f"Scanning val_data_dir: {data_config.source.val_data_dir}")
        scanned = scan_directory(
            data_config.source.val_data_dir,
            caption_extension=caption_ext,
            is_reg=False,
            num_repeats=1,  # Validation images not repeated
            alpha_mask=data_config.preprocessing.alpha_mask,
            require_caption=True,
        )
        # Mark all as validation split
        for s in scanned:
            s.split = "val"
        all_scanned.extend(scanned)
    else:
        # Training mode: scan train_data_dir, reg_data_dir, in_json, subsets

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
                validation_split=validation_split,
                validation_seed=validation_seed,
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
                validation_split=validation_split,
                validation_seed=validation_seed,
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
                validation_split=validation_split if not subset.get("is_reg", False) else 0.0,
                validation_seed=validation_seed,
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
    elif validation and data_config.source.val_data_dir:
        base_dir = Path(data_config.source.val_data_dir).parent

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
        caption_separator=data_config.caption.caption_separator,
        keep_tokens_separator=data_config.caption.keep_tokens_separator,
        cache_dir=str(cache_dir) if cache_dir else None,
    )


def compute_config_hash(
    train_data_dir: str,
    cache_dir: str,
    resolution: tuple[int, int],
    bucket_reso_steps: int,
    max_token_length: int | None,
    enable_bucket: bool = True,
) -> str:
    """
    Compute a hash of settings that affect manifest creation.

    If this hash changes, the manifest needs to be rebuilt.

    Args:
        train_data_dir: Source data directory.
        cache_dir: Directory for cache files.
        resolution: Base training resolution.
        bucket_reso_steps: Bucket resolution step size.
        max_token_length: Max token length for TE caching.
        enable_bucket: Whether bucketing is enabled.

    Returns:
        Hex string hash (16 characters).
    """
    relevant = {
        "train_data_dir": train_data_dir,
        "cache_dir": cache_dir,
        "resolution": resolution,
        "bucket_reso_steps": bucket_reso_steps,
        "max_token_length": max_token_length,
        "enable_bucket": enable_bucket,
    }
    data = json.dumps(relevant, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def get_or_create_manifest(
    data_config: Any,
    cache_dir: str | Path,
    latent_dtype: str = "fp16",
    validation_split: float = 0.0,
    validation_seed: int | None = None,
) -> tuple[DatasetManifest, DatasetManifest | None]:
    """
    Load existing manifest if config unchanged, otherwise create new.

    This enables fast resume - unchanged runs skip manifest rebuild.

    Args:
        data_config: DataConfig with source, preprocessing, bucketing settings.
        cache_dir: Directory for cache files and manifest storage.
        latent_dtype: Data type for latents.
        validation_split: Fraction for validation split.
        validation_seed: Seed for validation split.

    Returns:
        Tuple of (train_manifest, val_manifest or None).
    """
    # Import here to avoid circular dependency

    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "dataset_manifest.json"
    val_manifest_path = cache_dir / "val_manifest.json"

    # Compute hash of current config
    current_hash = compute_config_hash(
        train_data_dir=str(data_config.source.train_data_dir),
        cache_dir=str(cache_dir),
        resolution=(data_config.preprocessing.resolution, data_config.preprocessing.resolution),
        bucket_reso_steps=data_config.bucketing.bucket_reso_steps,
        max_token_length=getattr(data_config, "max_token_length", None),
        enable_bucket=data_config.bucketing.enable_bucket,
    )

    # Try to load existing manifest
    if manifest_path.exists():
        try:
            existing = load_dataset_manifest(manifest_path)

            # Check if config hash matches
            if existing.config_hash and existing.config_hash == current_hash:
                # Quick check: count images in source dir to detect additions/removals
                from library.constants import IMAGE_EXTENSIONS

                source_dir = Path(data_config.source.train_data_dir)
                current_image_count = sum(1 for f in source_dir.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)

                if current_image_count != existing.image_count:
                    logger.info(f"Dataset changed ({existing.image_count} -> {current_image_count} images), rebuilding manifest")
                else:
                    logger.info(
                        f"Loaded existing manifest: {existing.image_count} images, "
                        f"{existing.caption_count} captions, {len(existing.buckets)} buckets "
                        f"(hash: {current_hash[:8]}...)"
                    )

                    # Load validation manifest if exists
                    val_manifest = None
                    if val_manifest_path.exists():
                        val_manifest = load_dataset_manifest(val_manifest_path)
                        logger.info(f"Loaded validation manifest: {val_manifest.image_count} images")

                    return existing, val_manifest
            else:
                old_hash = existing.config_hash[:8] if existing.config_hash else "none"
                logger.info(f"Config changed (hash: {old_hash}... -> {current_hash[:8]}...), rebuilding manifest")
        except Exception as e:
            logger.warning(f"Failed to load existing manifest: {e}, will recreate")

    # Create new manifest
    logger.info("Creating new dataset manifest...")
    train_manifest = create_manifest_from_config(
        data_config=data_config,
        cache_dir=cache_dir,
        latent_dtype=latent_dtype,
        validation=False,
        validation_split=validation_split,
        validation_seed=validation_seed,
    )

    # Set config hash for future validation
    train_manifest.config_hash = current_hash

    # Split into train/val if validation_split > 0
    val_manifest = None
    if validation_split > 0:
        train_entries = {k: v for k, v in train_manifest.entries.items() if v.split == "train"}
        val_entries = {k: v for k, v in train_manifest.entries.items() if v.split == "val"}

        if val_entries:
            # Create separate val manifest
            val_manifest = DatasetManifest(
                version=train_manifest.version,
                created_at=train_manifest.created_at,
                base_resolution=train_manifest.base_resolution,
                bucket_reso_steps=train_manifest.bucket_reso_steps,
                min_bucket_reso=train_manifest.min_bucket_reso,
                max_bucket_reso=train_manifest.max_bucket_reso,
                latent_channels=train_manifest.latent_channels,
                latent_scale_factor=train_manifest.latent_scale_factor,
                latent_dtype=train_manifest.latent_dtype,
                cache_dir=train_manifest.cache_dir,
                entries=val_entries,
                buckets={},  # Will recompute if needed
            )

            # Update train manifest to only have train entries
            train_manifest.entries = train_entries

    # Save manifests
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_dataset_manifest(train_manifest, manifest_path)
    if val_manifest:
        save_dataset_manifest(val_manifest, val_manifest_path)

    logger.info(
        f"Created new manifest: {train_manifest.image_count} images, "
        f"{train_manifest.caption_count} captions, {len(train_manifest.buckets)} buckets "
        f"(hash: {current_hash[:8]}...)"
    )

    return train_manifest, val_manifest


def save_dataset_manifest(manifest: DatasetManifest, path: str | Path) -> None:
    """
    Save a DatasetManifest to JSON file.

    Args:
        manifest: The manifest to save.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclasses to dicts
    # Build bucket distribution summary (sorted by resolution)
    sorted_buckets = sorted(manifest.buckets.items(), key=lambda x: (x[1].resolution[0], x[1].resolution[1]))
    bucket_distribution = [{"resolution": list(bucket.resolution), "count": len(bucket.image_ids)} for _, bucket in sorted_buckets]

    data = {
        "version": manifest.version,
        "created_at": manifest.created_at or datetime.now().isoformat(),
        "summary": {
            "total_images": manifest.total_images or len(manifest.entries),
            "total_captions": manifest.total_captions or sum(1 for e in manifest.entries.values() if e.caption),
            "num_buckets": len(manifest.buckets),
        },
        "config": {
            "base_resolution": list(manifest.base_resolution),
            "bucket_reso_steps": manifest.bucket_reso_steps,
            "min_bucket_reso": manifest.min_bucket_reso,
            "max_bucket_reso": manifest.max_bucket_reso,
            "latent_channels": manifest.latent_channels,
            "latent_scale_factor": manifest.latent_scale_factor,
            "latent_dtype": manifest.latent_dtype,
            "cache_dir": manifest.cache_dir,
            "config_hash": manifest.config_hash,
        },
        "bucket_distribution": bucket_distribution,
        "entries": {id: _entry_to_dict(entry) for id, entry in manifest.entries.items()},
        "buckets": {key: _bucket_to_dict(bucket) for key, bucket in manifest.buckets.items()},
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved dataset manifest to {path} ({len(manifest.entries)} entries)")


def load_dataset_manifest(path: str | Path) -> DatasetManifest:
    """
    Load a DatasetManifest from JSON file.

    Args:
        path: Path to the manifest JSON file.

    Returns:
        Loaded DatasetManifest.

    Raises:
        FileNotFoundError: If manifest file doesn't exist.
        ValueError: If manifest format is invalid.
    """
    path = Path(path)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Parse config
    config = data.get("config", {})

    # Parse entries
    entries = {}
    for id, entry_data in data.get("entries", {}).items():
        entries[id] = _dict_to_entry(id, entry_data)

    # Parse buckets
    buckets = {}
    for key, bucket_data in data.get("buckets", {}).items():
        buckets[key] = _dict_to_bucket(bucket_data)

    # Load summary stats (optional, may not exist in older manifests)
    summary = data.get("summary", {})

    manifest = DatasetManifest(
        version=data.get("version", "2.0"),
        created_at=data.get("created_at", ""),
        base_resolution=tuple(config.get("base_resolution", [1024, 1024])),
        bucket_reso_steps=config.get("bucket_reso_steps", 64),
        min_bucket_reso=config.get("min_bucket_reso", 256),
        max_bucket_reso=config.get("max_bucket_reso", 2048),
        latent_channels=config.get("latent_channels", 4),
        latent_scale_factor=config.get("latent_scale_factor", 8),
        latent_dtype=config.get("latent_dtype", "fp16"),
        cache_dir=config.get("cache_dir", ""),
        config_hash=config.get("config_hash", ""),
        total_images=summary.get("total_images", 0),
        total_captions=summary.get("total_captions", 0),
        entries=entries,
        buckets=buckets,
    )

    logger.info(f"Loaded dataset manifest from {path} ({len(entries)} entries, {len(buckets)} buckets)")
    return manifest


def save_epoch_manifest(manifest: EpochManifest, path: str | Path) -> None:
    """
    Save an EpochManifest to JSON file.

    Args:
        manifest: The epoch manifest to save.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "epoch": manifest.epoch,
        "seed": manifest.seed,
        "batches": [_batch_info_to_dict(batch) for batch in manifest.batches],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    logger.debug(f"Saved epoch manifest to {path} ({manifest.num_batches} batches)")


def load_epoch_manifest(path: str | Path) -> EpochManifest:
    """
    Load an EpochManifest from JSON file.

    Args:
        path: Path to the epoch manifest JSON file.

    Returns:
        Loaded EpochManifest.
    """
    path = Path(path)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return EpochManifest(
        epoch=data["epoch"],
        seed=data["seed"],
        batches=[_dict_to_batch_info(b) for b in data["batches"]],
    )


# --- Serialization helpers ---


def _entry_to_dict(entry: CacheEntry) -> dict:
    """Convert CacheEntry to JSON-serializable dict."""
    return {
        "path": entry.image_path,
        "original_size": list(entry.original_size),
        "bucket_reso": list(entry.bucket_reso),
        "resized_size": list(entry.resized_size),
        "caption": entry.caption,
        "tags": entry.tags,
        "num_repeats": entry.num_repeats,
        "is_reg": entry.is_reg,
        "split": entry.split,
        "latent_cache": entry.latent_cache_path,
        "te_cache": entry.te_cache_path,
        "has_flipped": entry.has_flipped,
        "has_alpha_mask": entry.has_alpha_mask,
    }


def _dict_to_entry(id: str, data: dict) -> CacheEntry:
    """Convert dict to CacheEntry."""
    return CacheEntry(
        id=id,
        image_path=data["path"],
        original_size=tuple(data["original_size"]),
        bucket_reso=tuple(data["bucket_reso"]),
        resized_size=tuple(data["resized_size"]),
        caption=data.get("caption", ""),
        tags=data.get("tags", []),
        num_repeats=data.get("num_repeats", 1),
        is_reg=data.get("is_reg", False),
        split=data.get("split", "train"),
        latent_cache_path=data.get("latent_cache"),
        te_cache_path=data.get("te_cache"),
        has_flipped=data.get("has_flipped", False),
        has_alpha_mask=data.get("has_alpha_mask", False),
    )


def _bucket_to_dict(bucket: Bucket) -> dict:
    """Convert Bucket to JSON-serializable dict."""
    return {
        "resolution": list(bucket.resolution),
        "image_ids": bucket.image_ids,
        "recommended_batch_size": bucket.recommended_batch_size,
    }


def _dict_to_bucket(data: dict) -> Bucket:
    """Convert dict to Bucket."""
    return Bucket(
        resolution=tuple(data["resolution"]),
        image_ids=data.get("image_ids", []),
        recommended_batch_size=data.get("recommended_batch_size", 1),
    )


def _batch_info_to_dict(batch: BatchInfo) -> dict:
    """Convert BatchInfo to JSON-serializable dict."""
    return {
        "image_ids": batch.image_ids,
        "bucket_reso": list(batch.bucket_reso),
        "processed_captions": batch.processed_captions,
        "input_ids": batch.input_ids,  # Dict keyed by encoder name
    }


def _dict_to_batch_info(data: dict) -> BatchInfo:
    """Convert dict to BatchInfo."""
    return BatchInfo(
        image_ids=data["image_ids"],
        bucket_reso=tuple(data["bucket_reso"]),
        processed_captions=data.get("processed_captions", []),
        input_ids=data.get("input_ids", {}),
    )
