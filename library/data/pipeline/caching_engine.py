"""
Caching engine for fast latent and text encoder output caching.

This module provides the generic caching infrastructure that delegates
model-specific encoding to strategy objects from library/strategies/.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch

from library.data.pipeline.dataclasses import DatasetManifest, CacheEntry
from library.utils.common_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class CachingStrategy(ABC):
    """
    Abstract interface for model-specific caching behavior.

    Implementations live in library/strategies/ (e.g., SdSdxlLatentsCachingStrategy).
    The CachingEngine calls these methods to delegate model-specific work.
    """

    @abstractmethod
    def get_cache_path(self, entry: CacheEntry) -> str:
        """
        Get the cache file path for an entry.

        Args:
            entry: The cache entry.

        Returns:
            Absolute path where cache should be saved.
        """
        raise NotImplementedError

    @abstractmethod
    def encode_batch(
        self,
        images: torch.Tensor,
        model: Any,
        entries: list[CacheEntry],
    ) -> list[dict[str, torch.Tensor]]:
        """
        Encode a batch of images to cacheable tensors.

        Args:
            images: Batch of image tensors [B, C, H, W].
            model: The model to use for encoding (VAE or text encoder).
            entries: Corresponding CacheEntry objects for metadata.

        Returns:
            List of dicts, each containing tensors to save (e.g., {"latents": tensor}).
        """
        raise NotImplementedError

    @abstractmethod
    def save_cache(self, data: dict[str, torch.Tensor], path: str) -> None:
        """
        Save encoded data to cache file.

        Args:
            data: Dict of tensor name -> tensor to save.
            path: Output file path.
        """
        raise NotImplementedError

    @abstractmethod
    def load_cache(self, path: str) -> dict[str, torch.Tensor]:
        """
        Load cached data from file.

        Args:
            path: Cache file path.

        Returns:
            Dict of tensor name -> tensor.
        """
        raise NotImplementedError


class CachingEngine:
    """
    High-performance caching engine with multi-GPU coordination.

    Handles:
    - Batch organization and parallel I/O
    - Multi-GPU workload distribution
    - Progress tracking
    - Error handling and recovery

    Delegates model-specific encoding to a CachingStrategy.
    """

    def __init__(
        self,
        strategy: CachingStrategy,
        batch_size: int = 4,
        num_workers: int = 4,
    ):
        """
        Initialize the caching engine.

        Args:
            strategy: Model-specific caching strategy.
            batch_size: Number of images to process per batch.
            num_workers: Number of parallel I/O workers.
        """
        self.strategy = strategy
        self.batch_size = batch_size
        self.num_workers = num_workers

    def cache_dataset(
        self,
        manifest: DatasetManifest,
        model: Any,
        accelerator: Any,
        cache_dir: str | Path,
        skip_existing: bool = True,
    ) -> DatasetManifest:
        """
        Cache all entries in a dataset manifest.

        Args:
            manifest: Dataset manifest with entries to cache.
            model: Model to use for encoding (VAE or text encoder).
            accelerator: HuggingFace Accelerator for multi-GPU.
            cache_dir: Directory to save cache files.
            skip_existing: Whether to skip already-cached entries.

        Returns:
            Updated manifest with cache paths populated.
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Get entries that need caching
        entries_to_cache = self._get_entries_to_cache(manifest, skip_existing)

        if not entries_to_cache:
            logger.info("All entries already cached, nothing to do")
            return manifest

        # Split work across GPUs
        my_entries = self._split_for_gpu(entries_to_cache, accelerator)

        logger.info(f"Caching {len(my_entries)}/{len(entries_to_cache)} entries on GPU {accelerator.process_index}")

        # TODO: Implement actual caching loop with:
        # - Batched image loading
        # - Parallel VAE encoding
        # - Async file saving
        # - Progress bar

        # For now, just a placeholder
        for entry in my_entries:
            cache_path = self.strategy.get_cache_path(entry)
            entry.latent_cache_path = cache_path
            # Actual encoding would happen here

        # Sync across GPUs
        accelerator.wait_for_everyone()

        return manifest

    def _get_entries_to_cache(
        self,
        manifest: DatasetManifest,
        skip_existing: bool,
    ) -> list[CacheEntry]:
        """Get list of entries that need caching."""
        entries = []
        for entry in manifest.entries.values():
            if skip_existing and entry.latent_cache_path:
                cache_path = Path(entry.latent_cache_path)
                if cache_path.exists():
                    continue
            entries.append(entry)
        return entries

    def _split_for_gpu(
        self,
        entries: list[CacheEntry],
        accelerator: Any,
    ) -> list[CacheEntry]:
        """Split entries across GPUs using modulo assignment."""
        return [entry for i, entry in enumerate(entries) if i % accelerator.num_processes == accelerator.process_index]
