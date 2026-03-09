import logging
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from library.data.structures import DatasetManifest, CacheEntry, Bucket
from library.data.epoch_preparation import prepare_epoch
from library.data.dataloader import TrainingDataset
from library.data.caching_engine import CachingEngine, CacheHandler

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_pipeline_multigpu")


class TestPipelineMultiGPU(unittest.TestCase):
    def setUp(self):
        # Create a synthetic dataset manifest
        self.entries = {}
        self.buckets = {}

        # Create 2 buckets: 512x512 (10 images) and 768x768 (10 images)
        # 512x512
        bucket_512 = Bucket(resolution=(512, 512), image_ids=[])
        for i in range(10):
            img_id = f"img_512_{i}"
            bucket_512.image_ids.append(img_id)
            self.entries[img_id] = CacheEntry(
                id=img_id,
                image_path=f"/tmp/{img_id}.png",
                original_size=(512, 512),
                bucket_reso=(512, 512),
                resized_size=(512, 512),
                caption=f"caption for {img_id}",
                latent_cache_path=f"/tmp/{img_id}.safetensors",
            )
        self.buckets["512x512"] = bucket_512

        # 768x768
        bucket_768 = Bucket(resolution=(768, 768), image_ids=[])
        for i in range(10):
            img_id = f"img_768_{i}"
            bucket_768.image_ids.append(img_id)
            self.entries[img_id] = CacheEntry(
                id=img_id,
                image_path=f"/tmp/{img_id}.png",
                original_size=(768, 768),
                bucket_reso=(768, 768),
                resized_size=(768, 768),
                caption=f"caption for {img_id}",
                latent_cache_path=f"/tmp/{img_id}.safetensors",
            )
        self.buckets["768x768"] = bucket_768

        self.manifest = DatasetManifest(entries=self.entries, buckets=self.buckets)

    def test_prepare_epoch_determinism(self):
        """Verify prepare_epoch produces identical results for same seed/epoch."""
        logger.info("Test 1: Checking prepare_epoch determinism...")

        # Run 1
        epoch_manifest_1 = prepare_epoch(self.manifest, epoch=1, seed=42, batch_size=2, shuffle=True)

        # Run 2
        epoch_manifest_2 = prepare_epoch(self.manifest, epoch=1, seed=42, batch_size=2, shuffle=True)

        # Assertions
        self.assertEqual(epoch_manifest_1.epoch, epoch_manifest_2.epoch)
        self.assertEqual(epoch_manifest_1.seed, epoch_manifest_2.seed)
        self.assertEqual(len(epoch_manifest_1.batches), len(epoch_manifest_2.batches))

        # Check exact batch order
        for i, (b1, b2) in enumerate(zip(epoch_manifest_1.batches, epoch_manifest_2.batches)):
            self.assertEqual(b1.image_ids, b2.image_ids, f"Batch {i} content mismatch")
            self.assertEqual(b1.bucket_reso, b2.bucket_reso, f"Batch {i} resolution mismatch")

        logger.info("✅ prepare_epoch is deterministic.")

    def test_shuffling(self):
        """Verify different seeds produce different orders."""
        logger.info("Test 2: Checking shuffling...")

        # Run 1
        manifest_seed_42 = prepare_epoch(self.manifest, epoch=1, seed=42, batch_size=2, shuffle=True)

        # Run 2 (different seed)
        manifest_seed_43 = prepare_epoch(self.manifest, epoch=1, seed=43, batch_size=2, shuffle=True)

        # Check that batches are NOT identical (highly unlikely to be identical by chance with 20 images)
        # We check the sequence of image IDs across all batches
        ids_42 = [img_id for b in manifest_seed_42.batches for img_id in b.image_ids]
        ids_43 = [img_id for b in manifest_seed_43.batches for img_id in b.image_ids]

        self.assertNotEqual(ids_42, ids_43, "Different seeds produced same order!")

        # Also check same seed but different epoch (should also shuffle differently)
        manifest_epoch_2 = prepare_epoch(self.manifest, epoch=2, seed=42, batch_size=2, shuffle=True)
        ids_epoch_2 = [img_id for b in manifest_epoch_2.batches for img_id in b.image_ids]
        self.assertNotEqual(ids_42, ids_epoch_2, "Different epochs produced same order!")

        logger.info("✅ Shuffling works as expected.")

    def test_sharding(self):
        """Verify TrainingDataset shards batches correctly."""
        logger.info("Test 3: Checking sharding logic...")

        # Prepare a manifest
        epoch_manifest = prepare_epoch(self.manifest, epoch=1, seed=42, batch_size=2, shuffle=True)
        total_batches = len(epoch_manifest.batches)
        logger.info(f"Total batches in epoch: {total_batches}")

        # Mock strategy (we don't need real loading for this test)
        mock_strategy = MagicMock(spec=CacheHandler)
        # Mock load_cache to return a dummy tensor
        import torch

        mock_strategy.load_cache.return_value = {"latents": torch.zeros((4, 64, 64))}

        # Create datasets for Rank 0 and Rank 1 (World Size 2)
        ds_rank0 = TrainingDataset(self.manifest, epoch_manifest, mock_strategy, rank=0, world_size=2)
        ds_rank1 = TrainingDataset(self.manifest, epoch_manifest, mock_strategy, rank=1, world_size=2)

        # Now verify what the dataset actually yields
        # We need to mock _load_batch to avoid actual file system access
        ds_rank0._load_batch = MagicMock(return_value={"id": "mock_batch"})
        ds_rank1._load_batch = MagicMock(return_value={"id": "mock_batch"})

        iter_0 = list(ds_rank0)
        iter_1 = list(ds_rank1)

        # Check counts
        logger.info(f"Rank 0 got {len(iter_0)} batches")
        logger.info(f"Rank 1 got {len(iter_1)} batches")

        self.assertEqual(len(iter_0) + len(iter_1), total_batches)

        # Verify call args to ensure correct batches were requested
        # rank 0 should have called _load_batch for indices 0, 2, 4...
        # rank 1 should have called _load_batch for indices 1, 3, 5...

        # Check Rank 0
        calls_0 = ds_rank0._load_batch.call_args_list
        for call_idx, call in enumerate(calls_0):
            batch_info = call[0][0]
            # Find index of this batch in original manifest
            original_idx = epoch_manifest.batches.index(batch_info)
            self.assertEqual(original_idx % 2, 0, f"Rank 0 processed batch {original_idx} which should be for Rank 0")

        # Check Rank 1
        calls_1 = ds_rank1._load_batch.call_args_list
        for call_idx, call in enumerate(calls_1):
            batch_info = call[0][0]
            # Find index of this batch in original manifest
            original_idx = epoch_manifest.batches.index(batch_info)
            self.assertEqual(original_idx % 2, 1, f"Rank 1 processed batch {original_idx} which should be for Rank 1")

        logger.info("✅ Sharding logic is correct (Round Robin).")

    def test_caching_distribution(self):
        """Verify CachingEngine distribution logic."""
        logger.info("Test 4: Checking CachingEngine distribution...")

        mock_strategy = MagicMock(spec=CacheHandler)
        mock_strategy.get_entry_cache_path.return_value = Path("/tmp/mock.safetensors")
        mock_strategy.is_cache_valid.return_value = False  # Force caching

        engine = CachingEngine(mock_strategy, batch_size=1, num_workers=1)

        # Mock accelerator
        class MockAccelerator:
            def __init__(self, rank, num_processes):
                self.process_index = rank
                self.num_processes = num_processes

        # Create 10 dummy entries
        entries = [
            CacheEntry(id=f"{i}", image_path="", original_size=(0, 0), bucket_reso=(0, 0), resized_size=(0, 0), caption="")
            for i in range(10)
        ]

        # Test Rank 0 of 2
        acc_0 = MockAccelerator(0, 2)
        my_entries_0 = engine._split_for_gpu(entries, acc_0)
        # Should get 0, 2, 4, 6, 8
        ids_0 = [e.id for e in my_entries_0]
        self.assertEqual(ids_0, ["0", "2", "4", "6", "8"])

        # Test Rank 1 of 2
        acc_1 = MockAccelerator(1, 2)
        my_entries_1 = engine._split_for_gpu(entries, acc_1)
        # Should get 1, 3, 5, 7, 9
        ids_1 = [e.id for e in my_entries_1]
        self.assertEqual(ids_1, ["1", "3", "5", "7", "9"])

        logger.info("✅ Caching distribution logic is correct (Modulo).")
