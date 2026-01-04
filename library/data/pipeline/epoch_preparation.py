"""
Epoch preparation utilities.

Generates EpochManifest from DatasetManifest with:
- Shuffling
- Bucketed batch organization
- Memory-aware ordering (largest first for warmup)
- Repeat handling
"""

import logging
import random

from library.data.pipeline.dataclasses import DatasetManifest, EpochManifest, BatchInfo
from library.utils.common_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def prepare_epoch(
    manifest: DatasetManifest,
    epoch: int,
    seed: int | None = None,
    batch_size: int = 1,
    shuffle: bool = True,
    warmup_largest_first: bool = True,
    warmup_batches: int = 10,
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
        warmup_largest_first: If True, place largest-resolution batches first
            to establish CUDA memory allocation upfront, preventing OOMs later.
        warmup_batches: Number of largest batches to place at start for warmup.
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

    # Create BatchInfo objects per bucket
    all_batches: list[tuple[int, BatchInfo]] = []  # (pixel_count, BatchInfo)

    for bucket_key, ids in bucket_ids.items():
        parts = bucket_key.split("x")
        bucket_reso = (int(parts[0]), int(parts[1]))
        pixel_count = bucket_reso[0] * bucket_reso[1]

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            if drop_last and len(batch_ids) < batch_size:
                continue

            batch_info = BatchInfo(
                image_ids=batch_ids,
                bucket_reso=bucket_reso,
                # processed_captions will be populated later in Phase 3 implementation
            )
            all_batches.append((pixel_count, batch_info))

    # Order batches
    if warmup_largest_first:
        # Sort by resolution (largest first) for warmup
        all_batches.sort(key=lambda x: -x[0])
        # Take warmup batches, shuffle the rest
        warmup = [batch for _, batch in all_batches[:warmup_batches]]
        remaining = [batch for _, batch in all_batches[warmup_batches:]]
        if shuffle:
            rng.shuffle(remaining)
        batches = warmup + remaining
    else:
        # Just shuffle all batches
        batches = [batch for _, batch in all_batches]
        if shuffle:
            rng.shuffle(batches)

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
    # Get validation entries grouped by bucket for proper batching
    bucket_entries: dict[str, list[str]] = {}
    for entry in manifest.entries.values():
        if entry.split != "val":
            continue
        bucket_key = f"{entry.bucket_reso[0]}x{entry.bucket_reso[1]}"
        if bucket_key not in bucket_entries:
            bucket_entries[bucket_key] = []
        bucket_entries[bucket_key].append(entry.id)

    # Sort each bucket for deterministic order
    for ids in bucket_entries.values():
        ids.sort()

    # Create BatchInfo objects
    batches = []
    for bucket_key, ids in sorted(bucket_entries.items()):
        parts = bucket_key.split("x")
        bucket_reso = (int(parts[0]), int(parts[1]))

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            batches.append(
                BatchInfo(
                    image_ids=batch_ids,
                    bucket_reso=bucket_reso,
                )
            )

    return EpochManifest(
        epoch=0,  # Validation doesn't have epochs
        seed=seed,
        batches=batches,
    )
