import logging
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Any, List
from torch.utils.data import DataLoader

# Mocks and Imports
from library.data.pipeline.dataclasses import DatasetManifest, EpochManifest, BatchInfo, CacheEntry
from library.data.pipeline.dataloader import TrainingDataset, create_training_dataloader
from library.data.pipeline.caching_engine import CachingStrategy

# Setup simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("repro_script")


# Mock Caching Strategy
class MockCachingStrategy(CachingStrategy):
    def get_cache_path(self, entry: CacheEntry, cache_dir: Path) -> Path:
        return Path(f"/tmp/mock_cache/{entry.id}.safetensors")

    def encode_batch(self, images: torch.Tensor, model: Any, entries: List[CacheEntry]) -> List[dict]:
        return [{"latents": torch.zeros((4, 64, 64))} for _ in entries]

    def save_cache(self, data: dict, path: Path) -> None:
        pass

    def load_cache(self, path: Path) -> "CacheData":
        # Return fake CacheData
        from library.data.pipeline.dataclasses import CacheData

        return CacheData(latents=torch.zeros((4, 64, 64)))

    def is_cache_valid(self, path: Path, entry: CacheEntry, flip_aug: bool = False, alpha_mask: bool = False) -> bool:
        return True


def run_reproduction():
    logger.info("Starting reproduction of num_workers duplication bug...")

    # 1. Create Mock Manifests
    num_batches = 10
    batch_size = 2

    entries = {}
    batches = []

    for i in range(num_batches):
        batch_ids = []
        for j in range(batch_size):
            img_id = f"img_{i}_{j}"
            entries[img_id] = CacheEntry(
                id=img_id,
                image_path=f"/tmp/{img_id}.png",
                original_size=(512, 512),
                bucket_reso=(512, 512),
                resized_size=(512, 512),
                caption="test caption",
                latent_cache_path=f"/tmp/cache/{img_id}.safetensors",
            )
            batch_ids.append(img_id)

        batches.append(BatchInfo(image_ids=batch_ids, bucket_reso=(512, 512)))

    dataset_manifest = DatasetManifest(entries=entries)
    epoch_manifest = EpochManifest(epoch=0, seed=42, batches=batches)

    # 2. Initialize Dataset & Loader
    strategy = MockCachingStrategy()

    # Test 1: num_workers=0 (Should be correct)
    logger.info("Testing with num_workers=0...")
    loader_0 = create_training_dataloader(dataset_manifest, epoch_manifest, strategy, num_workers=0)
    count_0 = sum(1 for _ in loader_0)
    logger.info(f"num_workers=0 yielded {count_0} batches. Expected: {num_batches}")
    assert count_0 == num_batches, f"Failed num_workers=0: Got {count_0}, expected {num_batches}"

    # Test 2: num_workers=2 (Should be duplicated if bug exists)
    logger.info("Testing with num_workers=2...")
    loader_2 = create_training_dataloader(dataset_manifest, epoch_manifest, strategy, num_workers=2)
    count_2 = sum(1 for _ in loader_2)
    logger.info(f"num_workers=2 yielded {count_2} batches. Expected: {num_batches}")

    if count_2 == num_batches * 2:
        logger.error("BUG REPRODUCED: Data was duplicated by factor of num_workers!")
        print("BUG_CONFIRMED")
    elif count_2 == num_batches:
        logger.info("Test Passed: Data was correctly sharded.")
        print("BUG_NOT_FOUND")
    else:
        logger.warning(f"Unexpected count: {count_2}. Expected {num_batches} (pass) or {num_batches * 2} (bug).")
        print("BUG_UNCLEAR")


if __name__ == "__main__":
    run_reproduction()
