# Data Pipeline Audit

## Summary

This audit verifies the correctness of the new data pipeline implementation in `library/data/pipeline/`, focusing on multi-GPU support, determinism, and sharding logic.

**Date:** 2026-01-04
**Auditor:** Jules

## Findings

### 1. Multi-GPU Consistency
**Question:** Does `prepare_epoch()` produce identical batches across all ranks?
**Answer:** **Yes.**

The `prepare_epoch()` function is fully deterministic given the same `epoch` and `seed` arguments. It uses `random.Random(seed + epoch)` for shuffling and `stable_string_hash` (blake2b) for caption processing. This ensures that if every GPU rank runs `prepare_epoch` with the same arguments (standard DDP practice), they will all generate identical `EpochManifest` objects.

**Verification:**
- Validated via `Test 1` in `audit_data_pipeline.py`.
- Two calls with identical parameters produced bit-identical `EpochManifest` objects.

### 2. Shuffling Determinism
**Question:** Is shuffling deterministic across processes with same seed?
**Answer:** **Yes.**

Shuffling behavior is controlled entirely by the seed passed to `prepare_epoch`.
- Same seed + same epoch = Identical order.
- Different seed = Different order.
- Different epoch = Different order (even with same base seed, as effective seed is `seed + epoch`).

**Verification:**
- Validated via `Test 2` in `audit_data_pipeline.py`.
- Observed identical orders for same seed, and divergent orders for different seeds/epochs.

### 3. Sharding vs Replication
**Question:** Do we need to shard batches (each GPU gets different batches) or replicate?
**Answer:** **Shard.** (And the current implementation correctly does so).

The implementation in `TrainingDataset` correctly shards the epoch so that each GPU processes a disjoint subset of batches. The logic used is:
```python
is_this_rank = (batch_idx % world_size) == rank
```
This ensures that:
- Every batch is processed exactly once across the cluster.
- Workload is evenly distributed (round-robin).
- No duplication of training steps occurs.

**Verification:**
- Validated via `Test 3` in `audit_data_pipeline.py`.
- Simulating `world_size=2` showed Rank 0 processing even indices and Rank 1 processing odd indices.
- The union of batches covered the entire epoch with no overlap.

### 4. Caching Workload Distribution
**Additional Finding:** The `CachingEngine` also correctly shards workload for cache generation.

It uses a modulo strategy (`idx % num_processes == process_index`) to ensure each image is cached by exactly one GPU.

**Verification:**
- Validated via `Test 4` in `audit_data_pipeline.py`.

## Conclusion

The new data pipeline's core logic for shuffling, determinism, and multi-GPU sharding is **correct**. It correctly produces a consistent global view of the epoch (via `prepare_epoch`) and then correctly shards that view for distributed training (via `TrainingDataset`).
