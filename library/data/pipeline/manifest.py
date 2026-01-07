"""
Manifest reading and writing utilities.

Handles JSON serialization/deserialization of DatasetManifest and EpochManifest.
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from library.data.pipeline.dataclasses import DatasetManifest, CacheEntry, Bucket, EpochManifest, BatchInfo
from library.utils.common_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def compute_config_hash(
    cache_dir: str,
    resolution: tuple[int, int],
    bucket_reso_steps: int,
    max_token_length: int | None,
    image_count: int,
) -> str:
    """
    Compute a hash of settings that affect cache validity.

    If this hash changes, the manifest needs to be rebuilt.

    Args:
        cache_dir: Directory for cache files.
        resolution: Base training resolution.
        bucket_reso_steps: Bucket resolution step size.
        max_token_length: Max token length for TE caching.
        image_count: Number of images in dataset.

    Returns:
        Hex string hash.
    """
    relevant = {
        "cache_dir": cache_dir,
        "resolution": resolution,
        "bucket_reso_steps": bucket_reso_steps,
        "max_token_length": max_token_length,
        "image_count": image_count,
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
    from library.data.pipeline.dataset_scanner import create_manifest_from_config

    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "dataset_manifest.json"
    val_manifest_path = cache_dir / "val_manifest.json"

    # Try to load existing manifest
    if manifest_path.exists():
        try:
            existing = load_dataset_manifest(manifest_path)
            # TODO: Compute current hash and compare with existing.config_hash
            # For now, just load if present (user can delete to force rebuild)
            logger.info(f"Loaded existing manifest with {len(existing.entries)} entries")

            # Load validation manifest if exists
            val_manifest = None
            if val_manifest_path.exists():
                val_manifest = load_dataset_manifest(val_manifest_path)
                logger.info(f"Loaded existing validation manifest with {len(val_manifest.entries)} entries")

            return existing, val_manifest
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
    data = {
        "version": manifest.version,
        "created_at": manifest.created_at or datetime.now().isoformat(),
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
