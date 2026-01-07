
import time
import logging
import torch
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

from library.data.pipeline.dataclasses import DatasetManifest, CacheEntry, EpochManifest, BatchInfo
from library.data.pipeline.epoch_preparation import prepare_epoch
from library.data.pipeline.dataloader import create_training_dataloader, TrainingDataset
from library.data.pipeline.caching_engine import CachingStrategy

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark")

class MockCachingStrategy(CachingStrategy):
    """Mock strategy that returns dummy tensors without disk I/O."""

    def __init__(self, latent_dim=(4, 64, 64)):
        self.latent_dim = latent_dim

    def load_cache(self, path):
        # Return object with .latents, .conditioning, etc.
        # mimicking the structure returned by actual strategies
        mock_data = MagicMock()
        mock_data.latents = torch.randn(self.latent_dim)
        mock_data.latents_flipped = torch.randn(self.latent_dim)
        mock_data.alpha_mask = None
        mock_data.conditioning = MagicMock()
        mock_data.conditioning.original_size_hw = (1024, 1024)
        mock_data.conditioning.crop_top_left = (0, 0)
        mock_data.conditioning.target_size_hw = (1024, 1024)

        # Mock aux data for TE cache
        mock_data.aux = {"hidden_state1": torch.randn(1, 77, 768)}

        return mock_data

    # Implement abstract methods to satisfy interface
    def get_entry_cache_path(self, entry): return Path("/tmp/mock")
    def is_cache_valid(self, *args, **kwargs): return True
    def encode_batch(self, *args, **kwargs): return []
    def save_cache(self, *args, **kwargs): pass

def generate_manifest(num_images: int) -> DatasetManifest:
    """Generate a synthetic dataset manifest."""
    entries = {}

    # 3 buckets: 1024x1024, 768x1024, 1024x768
    buckets = [(1024, 1024), (768, 1024), (1024, 768)]

    for i in range(num_images):
        img_id = f"img_{i:06d}"
        bucket = buckets[i % len(buckets)]

        # Note: Using CacheEntry instead of DatasetEntry (renamed in recent refactor)
        entry = CacheEntry(
            id=img_id,
            image_path=f"/tmp/dataset/{img_id}.png",
            original_size=bucket,
            bucket_reso=bucket,
            resized_size=bucket,
            caption=f"a photo of image {i}, highly detailed, 8k",
            num_repeats=1,
            is_reg=False,
            split="train",
            latent_cache_path=f"/tmp/cache/{img_id}.safetensors",
            te_cache_path=f"/tmp/cache/{img_id}_te.safetensors",
            has_flipped=True,
            has_alpha_mask=False
        )
        entries[img_id] = entry

    return DatasetManifest(entries=entries)

def benchmark_prepare_epoch(manifest: DatasetManifest, batch_size: int = 4):
    """Measure prepare_epoch performance."""
    logger.info(f"Benchmarking prepare_epoch with {len(manifest.entries)} images...")

    start_time = time.perf_counter()
    epoch_manifest = prepare_epoch(
        manifest=manifest,
        epoch=1,
        seed=42,
        batch_size=batch_size,
        shuffle=True,
        warmup_largest_first=True,
        caption_config=None # Skip caption processing for basic overhead check
    )
    end_time = time.perf_counter()

    duration = end_time - start_time
    logger.info(f"prepare_epoch finished in {duration:.4f}s")
    return duration, epoch_manifest

def benchmark_dataloader(dataset_manifest: DatasetManifest, epoch_manifest: EpochManifest):
    """Measure DataLoader creation and iteration overhead."""
    logger.info("Benchmarking DataLoader...")

    # Mock strategies
    latent_strategy = MockCachingStrategy()
    te_strategy = MockCachingStrategy()

    # Measure creation time
    start_time = time.perf_counter()
    # Use 0 workers for benchmark to avoid multiprocessing overhead masking the results
    # and to prevent timeouts in restricted environments
    dataloader = create_training_dataloader(
        dataset_manifest=dataset_manifest,
        epoch_manifest=epoch_manifest,
        latent_strategy=latent_strategy,
        te_strategy=te_strategy,
        num_workers=0,
        prefetch_factor=None # prefetch_factor requires num_workers > 0
    )
    creation_time = time.perf_counter() - start_time
    logger.info(f"DataLoader creation: {creation_time:.4f}s")

    # Measure time to first batch
    start_time = time.perf_counter()
    iterator = iter(dataloader)
    try:
        first_batch = next(iterator)
        first_batch_time = time.perf_counter() - start_time
        logger.info(f"Time to first batch: {first_batch_time:.4f}s")
    except StopIteration:
        logger.error("DataLoader yielded no batches!")
        return creation_time, 0, 0

    # Measure throughput (next 100 batches)
    # We use a limited number of batches to avoid running forever on large datasets
    num_batches_to_measure = 100
    start_time = time.perf_counter()
    count = 0
    try:
        for _ in range(num_batches_to_measure):
            next(iterator)
            count += 1
    except StopIteration:
        pass

    duration = time.perf_counter() - start_time
    throughput = count / duration if duration > 0 else 0
    logger.info(f"Throughput ({count} batches): {throughput:.2f} batches/s")

    return creation_time, first_batch_time, throughput

def run_benchmarks():
    # Reduced sizes to prevent timeouts in the sandbox
    sizes = [1_000, 10_000, 50_000]
    results = []

    print("| Dataset Size | prepare_epoch (s) | DL Creation (s) | First Batch (s) | Throughput (batch/s) |")
    print("|--------------|-------------------|-----------------|-----------------|----------------------|")

    for size in sizes:
        manifest = generate_manifest(size)

        # Benchmark prepare_epoch
        prep_time, epoch_manifest = benchmark_prepare_epoch(manifest)

        # Benchmark DataLoader
        dl_create_time, first_batch_time, throughput = benchmark_dataloader(manifest, epoch_manifest)

        print(f"| {size:<12} | {prep_time:<17.4f} | {dl_create_time:<15.4f} | {first_batch_time:<15.4f} | {throughput:<20.2f} |")

        results.append({
            "size": size,
            "prep_time": prep_time,
            "dl_create_time": dl_create_time,
            "first_batch_time": first_batch_time,
            "throughput": throughput
        })

if __name__ == "__main__":
    run_benchmarks()
