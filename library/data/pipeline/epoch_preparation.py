"""
Epoch preparation utilities.

Generates EpochManifest from DatasetManifest with:
- Shuffling
- Bucketed batch organization
- Memory-aware ordering
- Repeat handling
"""

import logging
import random

from library.data.pipeline.dataclasses import DatasetManifest, EpochManifest
from library.utils.common_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def prepare_epoch(
    manifest: DatasetManifest,
    epoch: int,
    seed: int | None = None,
    batch_size: int = 1,
    shuffle: bool = True,
    memory_order: bool = True,
    drop_last: bool = False,
) -> EpochManifest:
    """
    Prepare an EpochManifest for training.

    This is Phase 3 of the pipeline - converting the static dataset manifest
    into a shuffled, batched order for one epoch of training.

    Args:
        manifest: The dataset manifest.
        epoch: Epoch number (used for seed if seed not provided).
        seed: Random seed for shuffling. If None, uses epoch number.
        batch_size: Target batch size.
        shuffle: Whether to shuffle within buckets.
        memory_order: Whether to order buckets by memory (smallest first).
        drop_last: Whether to drop incomplete final batches.

    Returns:
        EpochManifest with pre-computed batch order.
    """
    seed = seed if seed is not None else epoch
    rng = random.Random(seed)

    # Collect all image IDs with repeats
    all_ids = []
    for entry in manifest.entries.values():
        if entry.split != "train":
            continue
        for _ in range(entry.num_repeats):
            all_ids.append(entry.id)

    # Group by bucket
    bucket_ids: dict[str, list[str]] = {}
    for id in all_ids:
        entry = manifest.get_entry(id)
        if entry is None:
            continue
        bucket_key = f"{entry.bucket_reso[0]}x{entry.bucket_reso[1]}"
        if bucket_key not in bucket_ids:
            bucket_ids[bucket_key] = []
        bucket_ids[bucket_key].append(id)

    # Shuffle within each bucket
    if shuffle:
        for ids in bucket_ids.values():
            rng.shuffle(ids)

    # Create batches per bucket
    all_batches: list[tuple[int, list[str]]] = []  # (memory_estimate, batch)

    for bucket_key, ids in bucket_ids.items():
        parts = bucket_key.split("x")
        bucket = manifest.get_bucket((int(parts[0]), int(parts[1])))
        memory_per_batch = bucket.memory_per_image * batch_size if bucket else 0

        for i in range(0, len(ids), batch_size):
            batch = ids[i : i + batch_size]
            if drop_last and len(batch) < batch_size:
                continue
            all_batches.append((memory_per_batch, batch))

    # Order batches
    if memory_order:
        # Sort by memory (smallest buckets first) for predictable memory usage
        all_batches.sort(key=lambda x: x[0])
    else:
        # Shuffle batches across buckets
        if shuffle:
            rng.shuffle(all_batches)

    # Extract just the batch IDs
    batches = [batch for _, batch in all_batches]

    epoch_manifest = EpochManifest(
        epoch=epoch,
        seed=seed,
        batches=batches,
    )

    logger.info(f"Prepared epoch {epoch}: {epoch_manifest.num_batches} batches, {epoch_manifest.num_images} images")

    return epoch_manifest


def prepare_validation_epoch(
    manifest: DatasetManifest,
    batch_size: int = 1,
    seed: int = 42,
) -> EpochManifest:
    """
    Prepare an EpochManifest for validation (no shuffling, deterministic).

    Args:
        manifest: The dataset manifest.
        batch_size: Batch size for validation.
        seed: Fixed seed for deterministic ordering.

    Returns:
        EpochManifest for validation.
    """
    # Get validation entries
    val_ids = [entry.id for entry in manifest.entries.values() if entry.split == "val"]

    # Sort for deterministic order
    val_ids.sort()

    # Create batches
    batches = []
    for i in range(0, len(val_ids), batch_size):
        batches.append(val_ids[i : i + batch_size])

    return EpochManifest(
        epoch=0,  # Validation doesn't have epochs
        seed=seed,
        batches=batches,
    )
