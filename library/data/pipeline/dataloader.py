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
    """

    def __init__(
        self,
        dataset_manifest: DatasetManifest,
        epoch_manifest: EpochManifest,
        latent_strategy: CachingStrategy,
        te_strategy: CachingStrategy | None = None,
        device: torch.device | None = None,
    ):
        """
        Initialize training dataset.

        Args:
            dataset_manifest: Full dataset manifest with entry metadata.
            epoch_manifest: Pre-computed batch order for this epoch.
            latent_strategy: Strategy for loading latent caches.
            te_strategy: Optional strategy for loading text encoder caches.
            device: Device to load tensors to.
        """
        self.dataset_manifest = dataset_manifest
        self.epoch_manifest = epoch_manifest
        self.latent_strategy = latent_strategy
        self.te_strategy = te_strategy
        self.device = device

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """
        Iterate through batches in epoch manifest order.

        Yields:
            Dict containing batch data:
            - "latents": Batched latent tensors [B, C, H, W]
            - "captions": List of caption strings (or processed_captions if available)
            - "input_ids": Tokenized input (if not using TE cache)
            - "text_encoder_outputs": Cached TE outputs (if using TE cache)
            - Other metadata as needed
        """
        for batch_info in self.epoch_manifest.batches:
            yield self._load_batch(batch_info)

    def __len__(self) -> int:
        """Number of batches in this epoch."""
        return self.epoch_manifest.num_batches

    def _load_batch(self, batch_info: BatchInfo) -> dict[str, Any]:
        """
        Load a single batch of cached data.

        Args:
            batch_info: BatchInfo with image IDs and metadata.

        Returns:
            Dict with batched tensors and metadata.
        """
        entries = [self.dataset_manifest.get_entry(id) for id in batch_info.image_ids]
        entries = [e for e in entries if e is not None]  # Filter missing

        if not entries:
            raise ValueError(f"No valid entries found for batch IDs: {batch_info.image_ids}")

        # Load latents
        latents_list = []
        for entry in entries:
            if entry.latent_cache_path:
                cache_data = self.latent_strategy.load_cache(entry.latent_cache_path)
                latents_list.append(cache_data.get("latents"))
            else:
                raise ValueError(f"Missing latent cache for {entry.id}")

        # Stack into batch
        latents = torch.stack(latents_list, dim=0)
        if self.device:
            latents = latents.to(self.device)

        # Use processed_captions if available, otherwise fall back to raw captions
        captions = batch_info.processed_captions if batch_info.processed_captions else [e.caption for e in entries]

        # Build batch dict
        batch = {
            "latents": latents,
            "captions": captions,
            "image_ids": batch_info.image_ids,
            "bucket_reso": batch_info.bucket_reso,
            # TODO: Add more fields as needed (loss_weights, alpha_masks, etc.)
        }

        # Add tokenized input_ids if available (dict keyed by encoder name)
        if batch_info.input_ids:
            batch["input_ids"] = {encoder_name: torch.tensor(tokens) for encoder_name, tokens in batch_info.input_ids.items()}

        # Load text encoder outputs if available
        if self.te_strategy and entries[0].te_cache_path:
            te_outputs = self._load_te_outputs(entries)
            batch["text_encoder_outputs"] = te_outputs

        return batch

    def _load_te_outputs(self, entries: list[CacheEntry]) -> dict[str, torch.Tensor]:
        """Load and batch text encoder outputs."""
        assert self.te_strategy is not None

        outputs = []
        for entry in entries:
            if entry.te_cache_path:
                cache_data = self.te_strategy.load_cache(entry.te_cache_path)
                outputs.append(cache_data)

        # Stack each output type
        result = {}
        if outputs:
            for key in outputs[0]:
                tensors = [o[key] for o in outputs]
                result[key] = torch.stack(tensors, dim=0)
                if self.device:
                    result[key] = result[key].to(self.device)

        return result


def create_training_dataloader(
    dataset_manifest: DatasetManifest,
    epoch_manifest: EpochManifest,
    latent_strategy: CachingStrategy,
    te_strategy: CachingStrategy | None = None,
    device: torch.device | None = None,
    num_workers: int = 0,
    prefetch_factor: int = 2,
) -> DataLoader:
    """
    Create a DataLoader for training iteration.

    Args:
        dataset_manifest: Full dataset manifest.
        epoch_manifest: Pre-computed epoch batch order.
        latent_strategy: Strategy for loading latents.
        te_strategy: Optional strategy for TE outputs.
        device: Device to load to.
        num_workers: Number of data loading workers.
        prefetch_factor: Batches to prefetch per worker.

    Returns:
        DataLoader that yields batch dicts.
    """
    dataset = TrainingDataset(
        dataset_manifest=dataset_manifest,
        epoch_manifest=epoch_manifest,
        latent_strategy=latent_strategy,
        te_strategy=te_strategy,
        device=device,
    )

    # For IterableDataset, batch_size=1 since batching is pre-done
    return DataLoader(
        dataset,
        batch_size=None,  # Disable automatic batching
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=device is not None and device.type == "cuda",
    )
