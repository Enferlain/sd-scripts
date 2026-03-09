"""
Caching engine for fast latent and text encoder output caching.

This module provides the generic caching infrastructure that delegates
model-specific encoding to cache handlers from library/strategies/.
"""

import os
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from library.data.structures import CacheData, CacheEntry, DatasetManifest


logger = logging.getLogger(__name__)


class CacheHandler(ABC):
    """
    Abstract interface for model-specific caching behavior.

    Implementations live in library/strategies/ (e.g., SdSdxlLatentsCachingStrategy).
    The CachingEngine calls these methods to delegate model-specific work.

    Cache paths are set on entries at manifest creation time. Strategies just need
    to know which field to read (latent_cache_path vs te_cache_path).
    """

    def get_entry_cache_path(self, entry: CacheEntry) -> str | None:
        """
        Get the cache path from an entry.

        Override in subclasses to return the appropriate field.
        Default returns latent_cache_path for VAE caching strategies.

        Args:
            entry: The cache entry.

        Returns:
            Path string or None if not set.
        """
        return entry.latent_cache_path

    @abstractmethod
    def encode_batch(
        self,
        images: torch.Tensor,
        model: Any,
        entries: list[CacheEntry],
    ) -> list[dict[str, Any]]:
        """
        Encode a batch of images to cacheable tensors.

        Args:
            images: Batch of image tensors [B, C, H, W].
            model: The model to use for encoding (VAE or text encoder).
            entries: Corresponding CacheEntry objects for metadata.

        Returns:
            List of dicts, each containing data to save (e.g., {"latents": tensor}).
        """
        raise NotImplementedError

    @abstractmethod
    def save_cache(self, data: dict[str, Any], path: Path) -> None:
        """
        Save encoded data to cache file.

        Args:
            data: Dict of tensor name -> tensor to save.
            path: Output file path.
        """
        raise NotImplementedError

    @abstractmethod
    def load_cache(self, path: Path) -> "CacheData":
        """
        Load cached data from file.

        Args:
            path: Cache file path.

        Returns:
            CacheData with latents/embeddings and optional model-specific conditioning.
            The conditioning field contains model-specific data (e.g., SdxlConditioning).
        """
        raise NotImplementedError

    @abstractmethod
    def is_cache_valid(
        self,
        path: Path,
        entry: CacheEntry,
        flip_aug: bool = False,
        alpha_mask: bool = False,
    ) -> bool:
        """
        Check if a cache file is valid for the given entry and config.

        Validates:
        - Required keys exist (latents, hidden_states, etc.)
        - Tensor shapes match expected bucket resolution
        - Optional data present if needed (flipped latents, alpha mask)
        - Stored metadata matches entry metadata

        Args:
            path: Cache file path.
            entry: The CacheEntry to validate against.
            flip_aug: Whether flipped latents are required.
            alpha_mask: Whether alpha mask is required.

        Returns:
            True if cache is valid, False if it needs re-caching.
        """
        raise NotImplementedError

    def preprocess_image(
        self,
        image: Image.Image,
        target_size: tuple[int, int],
        resized_size: tuple[int, int] | None = None,
        random_crop: bool = False,
        random_crop_padding_percent: float = 0.05,
        resize_interpolation: str | None = None,
    ) -> torch.Tensor:
        """
        Preprocess an image for encoding.

        Resizes maintaining aspect ratio to resized_size, then crops to target_size.
        Subclasses should override with model-specific implementations.

        Args:
            image: PIL Image.
            target_size: Final bucket resolution (width, height) after cropping.
            resized_size: Intermediate size before crop (width, height). If None,
                uses target_size directly (may cause distortion).
            random_crop: If True, use random crop offset. If False, center crop.
            random_crop_padding_percent: Extra padding when random crop enabled.
            resize_interpolation: Interpolation method ('lanczos', 'hamming', 'area', etc.).
                If None, auto-selects based on scale direction.

        Returns:
            Tensor [C, H, W] ready for batching.
        """
        # Default implementation - subclasses override with full resize+crop logic
        if image.size != target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)

        if image.mode != "RGB":
            image = image.convert("RGB")

        arr = np.array(image).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor


class CachingEngine:
    """
    High-performance caching engine with multi-GPU coordination.

    Handles:
    - Batch organization by bucket resolution
    - Parallel image loading (ThreadPoolExecutor)
    - Progress tracking (tqdm)
    - Multi-GPU workload distribution

    Delegates model-specific encoding to a CacheHandler.
    """

    def __init__(
        self,
        handler: CacheHandler,
        batch_size: int = 4,
        num_workers: int = 4,
        random_crop: bool = False,
        random_crop_padding_percent: float = 0.05,
        resize_interpolation: str | None = None,
    ):
        """
        Initialize the caching engine.

        Args:
            handler: Model-specific cache handler.
            batch_size: Number of images to process per batch.
            num_workers: Number of parallel I/O workers.
            random_crop: If True, use random crop during preprocessing.
            random_crop_padding_percent: Extra padding for random crop.
            resize_interpolation: Interpolation method ('lanczos', 'hamming', 'area', etc.).
                If None, auto-selects based on scale direction.
        """
        self.handler = handler
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_crop = random_crop
        self.random_crop_padding_percent = random_crop_padding_percent
        self.resize_interpolation = resize_interpolation

    def cache_dataset(
        self,
        manifest: DatasetManifest,
        model: Any,
        accelerator: Any,
        cache_dir: str | Path,
        skip_existing: bool = True,
        skip_validity_check: bool = False,
        flip_aug: bool = False,
        alpha_mask: bool = False,
        show_progress: bool = True,
        cache_type: str = "Caching",  # Display name for progress bar (e.g., "Latent Caching", "TE Caching")
    ) -> DatasetManifest:
        """
        Cache all entries in a dataset manifest.

        Args:
            manifest: Dataset manifest with entries to cache.
            model: Model to use for encoding (VAE or text encoder).
            accelerator: HuggingFace Accelerator for multi-GPU.
            cache_dir: Directory to save cache files.
            skip_existing: Whether to skip already-cached entries.
            skip_validity_check: If True, only check file exists (faster).
                If False, validate cache contents match expected config.
            flip_aug: Whether flipped latents are required.
            alpha_mask: Whether alpha mask is required.
            show_progress: Whether to show tqdm progress bar.

        Returns:
            Updated manifest with cache paths populated.
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Get entries that need caching
        entries_to_cache = self._get_entries_to_cache(manifest, cache_dir, skip_existing, skip_validity_check, flip_aug, alpha_mask)

        if not entries_to_cache:
            logger.info("All entries already cached, nothing to do")
            return manifest

        # Split work across GPUs
        my_entries = self._split_for_gpu(entries_to_cache, accelerator)

        logger.info(f"Caching {len(my_entries)}/{len(entries_to_cache)} entries on GPU {accelerator.process_index}")

        # Group entries by bucket for efficient batching
        batches = self._batch_entries_by_bucket(my_entries)

        # Process batches with progress bar
        pbar = tqdm(
            total=len(my_entries),
            desc=f"{cache_type} (GPU {accelerator.process_index})",
            disable=not show_progress or accelerator.process_index != 0,
        )

        debug_memory = os.environ.get("DEBUG_CACHING_MEMORY", "").lower() in ("1", "true", "yes")
        batch_count = 0
        prev_bucket_key = None

        for _bucket_key, bucket_entries in batches.items():
            # Clear CUDA cache when switching bucket sizes to prevent memory fragmentation
            if prev_bucket_key is not None and _bucket_key != prev_bucket_key and torch.cuda.is_available():
                torch.cuda.empty_cache()
                if debug_memory:
                    logger.info(f"[MEM] Cleared cache between buckets: {prev_bucket_key} → {_bucket_key}")
            prev_bucket_key = _bucket_key

            for batch_entries in bucket_entries:
                batch_count += 1

                if debug_memory and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    mem_before = torch.cuda.memory_allocated() / 1024**3
                    reserved_before = torch.cuda.memory_reserved() / 1024**3

                self._cache_batch(batch_entries, model, cache_dir)

                if debug_memory and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    mem_after = torch.cuda.memory_allocated() / 1024**3
                    reserved_after = torch.cuda.memory_reserved() / 1024**3
                    bucket_reso = batch_entries[0].bucket_reso
                    logger.info(
                        f"[MEM] Batch {batch_count}: bucket={bucket_reso}, "
                        f"alloc={mem_before:.2f}→{mem_after:.2f}GB, "
                        f"reserved={reserved_before:.2f}→{reserved_after:.2f}GB"
                    )

                pbar.update(len(batch_entries))

        pbar.close()

        # Sync across GPUs
        accelerator.wait_for_everyone()

        return manifest

    def _get_entries_to_cache(
        self,
        manifest: DatasetManifest,
        cache_dir: Path,
        skip_existing: bool,
        skip_validity_check: bool,
        flip_aug: bool,
        alpha_mask: bool,
    ) -> list[CacheEntry]:
        """Get list of entries that need caching."""
        entries = []
        for entry in manifest.entries.values():
            # Get pre-set cache path from entry
            cache_path_str = self.handler.get_entry_cache_path(entry)
            if not cache_path_str:
                # No cache path set - this entry can't be cached
                logger.warning(f"No cache path set for {entry.id}, skipping")
                continue

            cache_path = Path(cache_path_str)

            if skip_existing and cache_path.exists():
                # Fast path: only check existence
                if skip_validity_check:
                    continue
                # Full validation: check cache contents
                try:
                    if self.handler.is_cache_valid(cache_path, entry, flip_aug, alpha_mask):
                        continue
                    else:
                        logger.debug(f"Cache invalid for {entry.id}, will re-cache")
                except Exception as e:
                    logger.warning(f"Cache validation failed for {entry.id}: {e}, will re-cache")
            entries.append(entry)
        return entries

    def _split_for_gpu(
        self,
        entries: list[CacheEntry],
        accelerator: Any,
    ) -> list[CacheEntry]:
        """Split entries across GPUs using modulo assignment."""
        return [entry for i, entry in enumerate(entries) if i % accelerator.num_processes == accelerator.process_index]

    def _batch_entries_by_bucket(
        self,
        entries: list[CacheEntry],
    ) -> dict[str, list[list[CacheEntry]]]:
        """
        Group entries by bucket resolution and split into batches.

        Returns:
            Dict mapping bucket key -> list of batches (each batch is a list of entries).
        """
        # Group by bucket
        by_bucket: dict[str, list[CacheEntry]] = defaultdict(list)
        for entry in entries:
            bucket_key = f"{entry.bucket_reso[0]}x{entry.bucket_reso[1]}"
            by_bucket[bucket_key].append(entry)

        # Split each bucket into batches
        batches: dict[str, list[list[CacheEntry]]] = {}
        for bucket_key, bucket_entries in by_bucket.items():
            batches[bucket_key] = [bucket_entries[i : i + self.batch_size] for i in range(0, len(bucket_entries), self.batch_size)]

        return batches

    def _load_images(
        self,
        entries: list[CacheEntry],
    ) -> list[Image.Image]:
        """Load images in parallel using ThreadPoolExecutor."""

        def load_one(entry: CacheEntry) -> Image.Image:
            return Image.open(entry.image_path)

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(load_one, e): i for i, e in enumerate(entries)}
            results: list[Image.Image | None] = [None] * len(entries)
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        # Filter out None (shouldn't happen, but satisfies type checker)
        return [img for img in results if img is not None]

    def _cache_batch(
        self,
        entries: list[CacheEntry],
        model: Any,
        cache_dir: Path,
    ) -> None:
        """
        Load, encode, and save a batch of images.

        Args:
            entries: List of CacheEntry objects in this batch.
            model: The encoding model (VAE or text encoder).
            cache_dir: Directory to save cache files.
        """
        # Load images in parallel
        images = self._load_images(entries)

        # Preprocess and stack into batch tensor
        target_size = entries[0].bucket_reso  # All entries in batch have same bucket
        tensors = []
        for img, entry in zip(images, entries):
            tensor = self.handler.preprocess_image(
                img,
                target_size=target_size,
                resized_size=entry.resized_size,
                random_crop=self.random_crop,
                random_crop_padding_percent=self.random_crop_padding_percent,
                resize_interpolation=self.resize_interpolation,
            )
            tensors.append(tensor)
            img.close()

        batch_tensor = torch.stack(tensors)  # [B, C, H, W]

        # Encode via strategy
        encoded_list = self.handler.encode_batch(batch_tensor, model, entries)

        # Save each result
        for entry, encoded in zip(entries, encoded_list):
            cache_path_str = self.handler.get_entry_cache_path(entry)
            if not cache_path_str:
                logger.warning(f"No cache path for {entry.id}, skipping save")
                continue
            cache_path = Path(cache_path_str)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.handler.save_cache(encoded, cache_path)
