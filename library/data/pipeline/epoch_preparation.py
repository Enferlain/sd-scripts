"""
Epoch preparation utilities.

Generates EpochManifest from DatasetManifest with:
- Shuffling
- Bucketed batch organization
- Memory-aware ordering (largest first for warmup)
- Repeat handling
- Caption processing (shuffle, dropout, wildcards)
"""

import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from library.data.pipeline.caption_processor import CaptionConfig, process_caption
from library.data.pipeline.dataclasses import DatasetManifest, EpochManifest, BatchInfo
from library.utils.common_utils import setup_logging
from library.utils.hash_utils import stable_string_hash

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
    caption_config: CaptionConfig | None = None,
    current_step: int = 0,
    max_train_steps: int = 0,
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
        caption_config: Optional caption processing configuration.
        current_step: Current training step (for token warmup).
        max_train_steps: Total training steps (for warmup calculation).

    Returns:
        EpochManifest with pre-computed batch order.
    """
    seed = seed if seed is not None else epoch
    rng = random.Random(seed)

    # Collect all sample keys with repeats
    # Each repeat is a unique "sample instance" with key "img_id#repeat_idx"
    all_sample_keys: list[str] = []
    for entry in manifest.entries.values():
        if entry.split != "train":
            continue
        for r in range(entry.num_repeats):
            sample_key = f"{entry.id}#{r}" if entry.num_repeats > 1 else entry.id
            all_sample_keys.append(sample_key)

    # Group by bucket
    bucket_keys: dict[str, list[str]] = {}
    for sample_key in all_sample_keys:
        # Extract original image id from sample_key
        img_id = sample_key.split("#")[0] if "#" in sample_key else sample_key
        entry = manifest.get_entry(img_id)
        if entry is None:
            continue
        bucket_key = f"{entry.bucket_reso[0]}x{entry.bucket_reso[1]}"
        if bucket_key not in bucket_keys:
            bucket_keys[bucket_key] = []
        bucket_keys[bucket_key].append(sample_key)

    # Shuffle within each bucket
    if shuffle:
        for keys in bucket_keys.values():
            rng.shuffle(keys)

    # Create BatchInfo objects per bucket
    all_batches: list[tuple[int, BatchInfo]] = []  # (pixel_count, BatchInfo)

    for bucket_key, sample_keys in bucket_keys.items():
        parts = bucket_key.split("x")
        bucket_reso = (int(parts[0]), int(parts[1]))
        pixel_count = bucket_reso[0] * bucket_reso[1]

        for i in range(0, len(sample_keys), batch_size):
            batch_sample_keys = sample_keys[i : i + batch_size]
            if drop_last and len(batch_sample_keys) < batch_size:
                continue

            # Extract image IDs and repeat indices for BatchInfo
            batch_ids: list[str] = []
            batch_repeat_indices: list[int] = []
            for sk in batch_sample_keys:
                if "#" in sk:
                    img_id, repeat_str = sk.rsplit("#", 1)
                    batch_ids.append(img_id)
                    batch_repeat_indices.append(int(repeat_str))
                else:
                    batch_ids.append(sk)
                    batch_repeat_indices.append(0)

            # Process captions if config provided
            processed_captions = []
            if caption_config is not None:
                for sample_key in batch_sample_keys:
                    img_id = sample_key.rsplit("#", 1)[0] if "#" in sample_key else sample_key
                    entry = manifest.get_entry(img_id)
                    if entry is not None:
                        # Use stable 64-bit hash for reproducibility across Python runs
                        # Each sample_key (including repeat index) gets unique randomness
                        caption_rng = random.Random(seed ^ stable_string_hash(sample_key) ^ epoch)
                        processed = process_caption(
                            entry.caption,
                            caption_config,
                            caption_rng,
                            current_step=current_step,
                            current_epoch=epoch,
                            max_train_steps=max_train_steps,
                        )
                        processed_captions.append(processed)
                    else:
                        processed_captions.append("")

            batch_info = BatchInfo(
                image_ids=batch_ids,
                bucket_reso=bucket_reso,
                repeat_indices=batch_repeat_indices,
                processed_captions=processed_captions,
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


def tokenize_epoch_manifest(
    epoch_manifest: EpochManifest,
    tokenize_fn,
    output_path: str | Path,
    encoder_names: list[str] | None = None,
    max_token_length: int = 77,
) -> Path:
    """
    Tokenize all processed_captions in an EpochManifest and save to safetensors.

    Creates a binary file with token tensors for efficient loading during training.
    Does NOT modify the EpochManifest - tokens are stored separately.

    Args:
        epoch_manifest: The epoch manifest to tokenize.
        tokenize_fn: Function that takes list[str] and returns list of token tensors.
            Should match TokenizeStrategy.tokenize() signature.
        output_path: Path to save the .safetensors file.
        encoder_names: Names for each encoder output. If None, auto-detects
            based on number of outputs (1=["clip"], 2=["clip_l", "clip_g"]).
        max_token_length: Token length for metadata recording.

    Returns:
        Path to the saved safetensors file.

    Example:
        >>> strategy = SdxlTokenizeStrategy(...)
        >>> path = tokenize_epoch_manifest(
        ...     epoch_manifest, strategy.tokenize, "epoch_1_tokens.safetensors"
        ... )
    """
    from pathlib import Path
    from safetensors.torch import save_file

    output_path = Path(output_path)

    # Collect all captions (flattened across batches)
    all_captions: list[str] = []
    batch_sizes: list[int] = []

    for batch in epoch_manifest.batches:
        captions = batch.processed_captions if batch.processed_captions else [""] * len(batch.image_ids)
        all_captions.extend(captions)
        batch_sizes.append(len(captions))

    if not all_captions:
        logger.warning("No captions to tokenize")
        return output_path

    # Tokenize all captions at once (batch tokenization is more efficient)
    token_tensors = tokenize_fn(all_captions)

    # Determine encoder names
    if encoder_names is None:
        num_encoders = len(token_tensors)
        if num_encoders == 1:
            encoder_names = ["clip"]
        elif num_encoders == 2:
            encoder_names = ["clip_l", "clip_g"]
        elif num_encoders == 3:
            encoder_names = ["clip_l", "clip_g", "t5"]
        else:
            encoder_names = [f"encoder_{i}" for i in range(num_encoders)]

    # Build tensors dict for safetensors
    tensors = {}
    for name, tensor in zip(encoder_names, token_tensors):
        # Ensure int64 for HF compatibility
        if hasattr(tensor, "to"):
            tensor = tensor.long()  # Convert to int64
        tensors[name] = tensor

    # Metadata
    metadata = {
        "epoch": str(epoch_manifest.epoch),
        "seed": str(epoch_manifest.seed),
        "num_samples": str(len(all_captions)),
        "num_batches": str(len(batch_sizes)),
        "max_token_length": str(max_token_length),
        "encoder_names": ",".join(encoder_names),
    }

    save_file(tensors, output_path, metadata=metadata)
    logger.info(f"Saved epoch tokens to {output_path}: {len(all_captions)} samples, {len(encoder_names)} encoders")

    return output_path


def load_epoch_tokens(
    tokens_path: str | Path,
) -> tuple[dict[str, "torch.Tensor"], dict[str, str]]:
    """
    Load tokenized captions from a safetensors file.

    Args:
        tokens_path: Path to the .safetensors file.

    Returns:
        Tuple of (tensors dict, metadata dict).

    Example:
        >>> tensors, metadata = load_epoch_tokens("epoch_1_tokens.safetensors")
        >>> input_ids_1 = tensors["clip_l"]  # [num_samples, max_length]
        >>> input_ids_2 = tensors["clip_g"]  # [num_samples, max_length]
    """
    from safetensors import safe_open

    tokens_path = Path(tokens_path)

    tensors = {}
    with safe_open(tokens_path, framework="pt") as f:
        metadata = f.metadata()
        for key in f.keys():  # noqa: SIM118 - safe_open requires .keys()
            tensors[key] = f.get_tensor(key)

    return tensors, metadata
