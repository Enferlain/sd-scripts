"""
Training DataLoader for fast epoch iteration.

Consumes pre-computed EpochManifest to serve batches with minimal overhead.
"""

import logging
from collections.abc import Iterator
from typing import Any

import torch
from torch.utils.data import IterableDataset, DataLoader

from library.data.pipeline.dataclasses import DatasetManifest, EpochManifest, CacheEntry, BatchInfo
from library.data.pipeline.caching_engine import CachingStrategy
from library.utils.common_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class TrainingDataset(IterableDataset):
    """
    Iterable dataset that reads from pre-computed epoch manifest.

    Unlike the current implementation that does bucketing/batching in __getitem__,
    this simply iterates through pre-organized batches, loading cached tensors.

    Supports two token loading modes:
    - Token file: Load from pre-tokenized safetensors file (for caption augmentation)
    - TE cache: Load cached text encoder outputs per-image (no augmentation)

    For distributed training, use rank/world_size to shard batches across GPUs.
    Tensors are yielded on CPU - the training loop handles GPU transfer.
    """

    def __init__(
        self,
        dataset_manifest: DatasetManifest,
        epoch_manifest: EpochManifest,
        latent_strategy: CachingStrategy,
        te_strategy: CachingStrategy | None = None,
        tokens_path: str | None = None,
        streaming_tokens: bool = False,
        flip_aug: bool = False,
        prior_loss_weight: float = 1.0,
        rank: int = 0,
        world_size: int = 1,
    ):
        """
        Initialize training dataset.

        Args:
            dataset_manifest: Full dataset manifest with entry metadata.
            epoch_manifest: Pre-computed batch order for this epoch.
            latent_strategy: Strategy for loading latent caches.
            te_strategy: Optional strategy for loading text encoder caches.
            tokens_path: Path to epoch token file (safetensors). If provided,
                tokens are loaded from file using offset-based slicing.
            streaming_tokens: If True, use memory-efficient streaming for tokens.
                If False, load all tokens into memory at start.
            flip_aug: If True, randomly use flipped latents (50% probability).
            prior_loss_weight: Loss weight for regularization images (is_reg=True).
            rank: Process rank for distributed training (0-indexed).
            world_size: Total number of processes for distributed training.
        """
        self.dataset_manifest = dataset_manifest
        self.epoch_manifest = epoch_manifest
        self.latent_strategy = latent_strategy
        self.te_strategy = te_strategy
        self.tokens_path = tokens_path
        self.streaming_tokens = streaming_tokens
        self.flip_aug = flip_aug
        self.prior_loss_weight = prior_loss_weight
        self.rank = rank
        self.world_size = world_size

        # Token data (loaded if not streaming)
        self._tokens: dict[str, torch.Tensor] | None = None
        self._token_file = None  # For streaming mode

        if tokens_path and not streaming_tokens:
            self._load_all_tokens()

    def _load_all_tokens(self) -> None:
        """Load all tokens into memory (non-streaming mode)."""
        from safetensors import safe_open

        self._tokens = {}
        with safe_open(self.tokens_path, framework="pt") as f:
            metadata = f.metadata()
            # Validate manifest hash
            self._validate_token_metadata(metadata)

            for key in f.keys():  # noqa: SIM118 - safe_open requires .keys()
                self._tokens[key] = f.get_tensor(key)

        logger.info(f"Loaded epoch tokens: {sum(t.shape[0] for t in self._tokens.values()) // len(self._tokens)} samples")

    def _load_tokens_streaming(self, token_offset: int, batch_size: int) -> dict[str, torch.Tensor]:
        """Load tokens for a single batch using memory-efficient streaming.

        Uses safetensors get_slice to load only the rows needed for this batch,
        avoiding loading the entire token file into memory.

        Args:
            token_offset: Starting row index in the token file.
            batch_size: Number of rows to load.

        Returns:
            Dict mapping encoder names to token tensors [batch_size, seq_len].
        """
        from safetensors import safe_open

        result = {}
        with safe_open(self.tokens_path, framework="pt") as f:
            for key in f.keys():  # noqa: SIM118 - safe_open requires .keys()
                # get_slice returns a lazy slice object that supports numpy-style indexing
                tensor_slice = f.get_slice(key)
                # Load only the rows for this batch
                result[key] = tensor_slice[token_offset : token_offset + batch_size]

        return result

    def _validate_token_metadata(self, metadata: dict[str, str]) -> None:
        """Validate that token file matches the epoch manifest."""
        from library.utils.hash_utils import stable_string_hash

        expected_id = f"epoch_{self.epoch_manifest.epoch}_seed_{self.epoch_manifest.seed}_batches_{len(self.epoch_manifest.batches)}"
        expected_hash = str(stable_string_hash(expected_id))

        if metadata.get("manifest_hash") != expected_hash:
            logger.warning(
                f"Token file manifest hash mismatch. "
                f"Expected {expected_hash}, got {metadata.get('manifest_hash')}. "
                f"Token file may not match epoch manifest."
            )

        expected_samples = sum(len(b.processed_captions or b.image_ids) for b in self.epoch_manifest.batches)
        if metadata.get("num_samples") != str(expected_samples):
            raise ValueError(f"Token file sample count mismatch. Expected {expected_samples}, got {metadata.get('num_samples')}")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """
        Iterate through batches in epoch manifest order.

        Handles two levels of sharding:
        1. Distributed training: rank/world_size (set at init)
        2. DataLoader workers: worker_id/num_workers (detected at runtime)

        Yields:
            Dict containing batch data (on CPU):
            - "latents": Batched latent tensors [B, C, H, W]
            - "captions": List of caption strings (or processed_captions if available)
            - "input_ids": Tokenized input (if using token file or BatchInfo.input_ids)
            - "text_encoder_outputs": Cached TE outputs (if using TE cache)
            - Other metadata as needed
        """
        # Get worker info for DataLoader multi-processing
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        # Token offset for sequential batch slicing
        # Must track all batches (not just this rank's/worker's) for correct offset
        token_offset = 0

        for batch_idx, batch_info in enumerate(self.epoch_manifest.batches):
            batch_size = len(batch_info.processed_captions or batch_info.image_ids)

            # Two-level sharding:
            # 1. Distributed: batch_idx % world_size == rank
            # 2. Workers: (batch_idx // world_size) % num_workers == worker_id
            is_this_rank = (batch_idx % self.world_size) == self.rank
            rank_local_idx = batch_idx // self.world_size
            is_this_worker = (rank_local_idx % num_workers) == worker_id

            if is_this_rank and is_this_worker:
                batch_data = self._load_batch(batch_info, token_offset, batch_size)
                yield batch_data

            # Always advance token offset (to stay aligned with token file)
            token_offset += batch_size

    def __len__(self) -> int:
        """Number of batches this rank will yield per epoch.

        Returns the total batches for this rank across all workers.
        This is the correct value for training loop progress bars and step counting,
        since all workers together yield exactly this many batches for the rank.

        Note: Individual workers each yield `len(self) // num_workers` batches.
        """
        total = self.epoch_manifest.num_batches
        # Divide by world_size, rounding up for last rank
        return (total + self.world_size - 1) // self.world_size

    def _load_batch(
        self,
        batch_info: BatchInfo,
        token_offset: int = 0,
        batch_size: int = 0,
    ) -> dict[str, Any]:
        """
        Load a single batch of cached data.

        Args:
            batch_info: BatchInfo with image IDs and metadata.
            token_offset: Starting index for token slicing.
            batch_size: Number of samples in this batch (for token slicing).

        Returns:
            Dict with batched tensors and metadata.
        """
        from pathlib import Path

        entries = [self.dataset_manifest.get_entry(img_id) for img_id in batch_info.image_ids]
        entries = [e for e in entries if e is not None]  # Filter missing

        if not entries:
            raise ValueError(f"No valid entries found for batch IDs: {batch_info.image_ids}")

        # Load latents and conditioning from cache
        latents_list = []
        alpha_masks_list = []
        conditionings = []
        loss_weights = []
        flippeds = []

        for entry in entries:
            if entry.latent_cache_path:
                cache_data = self.latent_strategy.load_cache(Path(entry.latent_cache_path))
                if cache_data.latents is None:
                    raise ValueError(f"Cache missing latents for {entry.id}")

                # Flip augmentation: 50% chance to use flipped latents if available
                use_flipped = False
                if self.flip_aug and cache_data.latents_flipped is not None:
                    import random

                    use_flipped = random.random() < 0.5

                if use_flipped:
                    latents_list.append(cache_data.latents_flipped)
                else:
                    latents_list.append(cache_data.latents)
                flippeds.append(use_flipped)

                # Alpha mask (optional)
                if cache_data.alpha_mask is not None:
                    alpha_masks_list.append(cache_data.alpha_mask)

                conditionings.append(cache_data.conditioning)

                # Loss weight: prior_loss_weight for reg images, 1.0 for training
                loss_weights.append(self.prior_loss_weight if entry.is_reg else 1.0)
            else:
                raise ValueError(f"Missing latent cache for {entry.id}")

        # Stack into batch (kept on CPU - training loop handles GPU transfer)
        latents = torch.stack(latents_list, dim=0)

        # Use processed_captions if available, otherwise fall back to raw captions
        captions = batch_info.processed_captions if batch_info.processed_captions else [e.caption for e in entries]

        # Build batch dict with all required fields
        batch = {
            "latents": latents,
            "captions": captions,
            "image_ids": batch_info.image_ids,
            "bucket_reso": batch_info.bucket_reso,
            # Model-specific conditioning (opaque to dataloader, training loop casts to concrete type)
            "conditionings": conditionings,
            # Training metadata
            "loss_weights": torch.tensor(loss_weights, dtype=latents.dtype),
            "flippeds": flippeds,
            # Alpha masks (only present if images have alpha channels)
            "alpha_masks": torch.stack(alpha_masks_list, dim=0) if alpha_masks_list else None,
        }

        # Add tokens from epoch token file (offset-based slicing)
        if self._tokens is not None and batch_size > 0:
            # Non-streaming mode: slice from pre-loaded tensors
            batch["input_ids"] = {name: tensor[token_offset : token_offset + batch_size] for name, tensor in self._tokens.items()}
        elif self.streaming_tokens and self.tokens_path and batch_size > 0:
            # Streaming mode: load only this batch's tokens from disk
            batch["input_ids"] = self._load_tokens_streaming(token_offset, batch_size)
        # Fall back to BatchInfo.input_ids if available (legacy/inline mode)
        elif batch_info.input_ids:
            batch["input_ids"] = {encoder_name: torch.tensor(tokens) for encoder_name, tokens in batch_info.input_ids.items()}

        # Load text encoder outputs if available
        if self.te_strategy and entries[0].te_cache_path:
            te_outputs = self._load_te_outputs(entries)
            batch["text_encoder_outputs"] = te_outputs

        return batch

    def _load_te_outputs(self, entries: list[CacheEntry]) -> dict[str, torch.Tensor]:
        """Load and batch text encoder outputs."""
        from pathlib import Path

        assert self.te_strategy is not None

        outputs = []
        for entry in entries:
            if entry.te_cache_path:
                cache_data = self.te_strategy.load_cache(Path(entry.te_cache_path))
                outputs.append(cache_data.aux)

        # Stack each output type
        result = {}
        if outputs:
            for key in outputs[0]:
                tensors_list = [o[key] for o in outputs]
                result[key] = torch.stack(tensors_list, dim=0)

        return result


def create_training_dataloader(
    dataset_manifest: DatasetManifest,
    epoch_manifest: EpochManifest,
    latent_strategy: CachingStrategy,
    te_strategy: CachingStrategy | None = None,
    tokens_path: str | None = None,
    streaming_tokens: bool = False,
    flip_aug: bool = False,
    prior_loss_weight: float = 1.0,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for training iteration.

    Args:
        dataset_manifest: Full dataset manifest.
        epoch_manifest: Pre-computed epoch batch order.
        latent_strategy: Strategy for loading latents.
        te_strategy: Optional strategy for TE outputs (cached per-image).
        tokens_path: Path to epoch token file (safetensors). For caption augmentation.
        streaming_tokens: If True, use memory-efficient streaming for tokens.
        flip_aug: If True, randomly use flipped latents (50% probability).
        prior_loss_weight: Loss weight for regularization images (default 1.0).
        rank: Process rank for distributed training (0-indexed).
        world_size: Total number of processes for distributed training.
        num_workers: Number of data loading workers (default 4 to avoid blocking I/O).
        prefetch_factor: Batches to prefetch per worker.
        pin_memory: Whether to use pinned memory for faster GPU transfer.

    Returns:
        DataLoader that yields batch dicts (on CPU, pin_memory enabled).
    """
    dataset = TrainingDataset(
        dataset_manifest=dataset_manifest,
        epoch_manifest=epoch_manifest,
        latent_strategy=latent_strategy,
        te_strategy=te_strategy,
        tokens_path=tokens_path,
        streaming_tokens=streaming_tokens,
        flip_aug=flip_aug,
        prior_loss_weight=prior_loss_weight,
        rank=rank,
        world_size=world_size,
    )

    # For IterableDataset, batch_size=None since batching is pre-done
    return DataLoader(
        dataset,
        batch_size=None,  # Disable automatic batching
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory,
    )
