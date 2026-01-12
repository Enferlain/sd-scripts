"""
GPU Memory Profiling Script for Caching

Usage in PyCharm:
1. Right-click this file → Run 'profile_caching'
2. Or: Run → Profile 'profile_caching'

This will show exactly where VRAM is allocated during caching.
"""

import torch
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from library.data.caching_engine import CachingEngine
from library.data.manifest import create_manifest_from_config
from library.strategies.sdxl.caching import SdxlLatentsPipelineStrategy
from library.config.dataclasses.data import DataConfig, SourceConfig, PreprocessingConfig, CachingConfig, BucketingConfig, CaptionConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_memory(label: str):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[MEM] {label}: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB, max={max_allocated:.2f}GB")


def profile_caching():
    """Profile VAE caching with detailed memory tracking."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Reset memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    print_memory("Initial")

    # Load VAE
    from diffusers import AutoencoderKL

    print("\n=== Loading VAE ===")
    vae = AutoencoderKL.from_single_file(
        "D:/stable-diffusion-webui-reForge/models/Stable-diffusion/21862-seele_pop3_v-pred.safetensors",
        subfolder="vae",
    )
    vae.to(device, dtype=torch.float32)  # no_half_vae=True
    vae.requires_grad_(False)
    vae.eval()
    print_memory("After VAE load")

    # Create manifest (small subset for profiling)
    print("\n=== Creating Manifest ===")
    data_config = DataConfig(
        source=SourceConfig(
            train_data_dir="D:/Projects/sd-scripts/tests/assets/training_images",
            dataset_repeats=1,
        ),
        preprocessing=PreprocessingConfig(resolution="1024,1024"),
        bucketing=BucketingConfig(
            enable_bucket=True,
            min_bucket_reso=256,
            max_bucket_reso=1024,
            bucket_reso_steps=64,
        ),
        caching=CachingConfig(
            cache_dir="D:/Projects/sd-scripts/profile_cache",
            cache_latents=True,
            cache_latents_to_disk=True,
        ),
        caption=CaptionConfig(caption_extension=".txt"),
    )

    manifest = create_manifest_from_config(data_config, cache_dir=data_config.caching.cache_dir)
    print(f"Manifest: {manifest.image_count} images, {len(manifest.buckets)} buckets")
    print_memory("After manifest")

    # Create mock accelerator
    class MockAccelerator:
        process_index = 0
        num_processes = 1

        def wait_for_everyone(self):
            pass

    accelerator = MockAccelerator()
    accelerator.device = device

    # Profile caching with memory snapshots
    print("\n=== Profiling Caching ===")

    strategy = SdxlLatentsPipelineStrategy(flip_aug=False, dtype="fp32")
    engine = CachingEngine(
        strategy=strategy,
        batch_size=2,  # Match your config
        num_workers=4,
    )

    print_memory("Before caching")

    # Enable memory history for detailed analysis
    if torch.cuda.is_available():
        torch.cuda.memory._record_memory_history(max_entries=100000)

    try:
        # Cache a subset (first 20 images)
        subset_entries = dict(list(manifest.entries.items())[:20])
        from library.data.structures import DatasetManifest

        subset_manifest = DatasetManifest(
            version=manifest.version,
            created_at=manifest.created_at,
            base_resolution=manifest.base_resolution,
            bucket_reso_steps=manifest.bucket_reso_steps,
            min_bucket_reso=manifest.min_bucket_reso,
            max_bucket_reso=manifest.max_bucket_reso,
            latent_channels=manifest.latent_channels,
            latent_scale_factor=manifest.latent_scale_factor,
            latent_dtype=manifest.latent_dtype,
            entries=subset_entries,
            buckets=manifest.buckets,
        )

        engine.cache_dataset(
            manifest=subset_manifest,
            model=vae,
            accelerator=accelerator,
            cache_dir=data_config.caching.cache_dir,
            skip_existing=False,  # Force re-cache
        )

    finally:
        if torch.cuda.is_available():
            torch.cuda.memory._record_memory_history(enabled=None)

    print_memory("After caching")

    # Dump memory snapshot for analysis
    if torch.cuda.is_available():
        snapshot_path = Path("memory_snapshot.pickle")
        torch.cuda.memory._dump_snapshot(str(snapshot_path))
        print(f"\nMemory snapshot saved to: {snapshot_path}")
        print("View with: python -m torch.cuda.memory._viz memory_snapshot.pickle")


if __name__ == "__main__":
    profile_caching()
