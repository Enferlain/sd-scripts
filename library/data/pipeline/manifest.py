"""
Manifest reading and writing utilities.

Handles JSON serialization/deserialization of DatasetManifest and EpochManifest.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from library.data.pipeline.dataclasses import DatasetManifest, CacheEntry, Bucket, EpochManifest
from library.utils.common_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


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
        "batches": manifest.batches,
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
        batches=data["batches"],
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
    }


def _dict_to_bucket(data: dict) -> Bucket:
    """Convert dict to Bucket."""
    return Bucket(
        resolution=tuple(data["resolution"]),
        image_ids=data.get("image_ids", []),
    )
