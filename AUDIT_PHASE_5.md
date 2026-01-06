# Audit Phase 5: Memory & Performance

## Overview

This audit investigates the performance characteristics of the new data pipeline, specifically focusing on per-epoch overhead, worker strategies, and scaling with dataset size.

**Reference:**
- Legacy Codebase: `library/data/_deprecated/dataset.py`
- New Pipeline: `library/data/pipeline/`

## 1. Per-Epoch Overhead & Scaling

Benchmarks were run on synthetic datasets of varying sizes to measure the overhead of `prepare_epoch()` and the `DataLoader` initialization.

| Dataset Size | `prepare_epoch` (s) | DL Creation (s) | First Batch Latency (s) | Throughput (batch/s)* |
|--------------|---------------------|-----------------|-------------------------|-----------------------|
| 1,000        | 0.0025              | 0.0003          | 0.0156                  | 89.62                 |
| 10,000       | 0.0264              | 0.0001          | 0.0099                  | 88.67                 |
| 50,000       | 0.3143              | 0.0001          | 0.0101                  | 102.59                |

*\*Throughput measured with `num_workers=0` (main process only) and mocked I/O to isolate Python logic overhead.*

### Findings:
- **`prepare_epoch()` is fast:** Even for 50k images, it takes ~0.3s. This is negligible compared to a typical epoch duration. It scales linearly with dataset size.
- **DataLoader creation is instant:** Since `EpochManifest` is pre-computed, the DataLoader just wraps it. No heavy indexing occurs at init time.
- **Low Startup Latency:** The time to yield the first batch is consistent (~10-15ms) regardless of dataset size.
- **Conclusion:** The designed approach of regenerating the manifest per epoch is **valid and performant**.

## 2. Worker Strategy: ThreadPool vs Persistent Workers

### Legacy Approach (`library/data/_deprecated/dataset.py`)
- **Architecture:** Uses a custom `BaseDataset` that mimics a batch sampler. `__getitem__` returns a full batch.
- **Workers:** Relies on standard PyTorch `num_workers`.
- **Bottleneck:** `__getitem__` performs heavy operations (image loading, resizing, augmentation, tokenization) for *every image in the batch* sequentially (unless a ThreadPool is used internally, which legacy has commented out code for).
- **Persistent Workers:** The legacy code supports `persistent_workers=True` but requires `set_current_epoch(epoch)` to trigger re-shuffling inside the worker or dataset state updates.

### New Pipeline Approach
- **Architecture:** `TrainingDataset` is an `IterableDataset` that reads from a pre-computed `EpochManifest`.
- **Heavy Lifting:**
    - **Image Processing:** Removed from the loop (cached).
    - **Shuffling/Bucketing:** Moved to `prepare_epoch` (main process, once per epoch).
    - **Tokenization:** Moved to `prepare_epoch` (cached/processed upfront) or efficient on-the-fly.
- **Worker Duty:** The worker primarily performs file I/O (loading `.safetensors` caches) and tensor stacking.

### Recommendation: Stick with Standard Workers (for now)
The "ThreadPool per-batch" idea (launching threads inside `__iter__` to load files) aims to parallelize I/O within a single worker. However:
1.  **PyTorch `num_workers`** already provides process-based parallelism.
2.  **GIL Release:** File I/O releases the GIL, but tensor allocation does not.
3.  **Complexity:** Managing a custom thread pool inside an `IterableDataset` that is already wrapped in `DataLoader` processes adds significant complexity (signal handling, deadlock risks).

**Current Strategy:**
- Use standard `DataLoader(num_workers=N)`.
- Since `EpochManifest` partitions batches deterministically (`is_this_worker`), each worker operates independently without lock contention.
- **Performance bottleneck** is likely to be disk random read IOPS (loading many small `.safetensors` files), not CPU.

## 3. Potential Optimizations (Flagged for Later)

1.  **Massive Datasets (>1M images):** `prepare_epoch` might take ~6s for 1M images. If this becomes a blocker, we could move bucket shuffling to C++ or optimize the Python loop. currently it is acceptable.
2.  **Streaming Tokens:** The `_load_tokens_streaming` method (using `safe_open` slice) is implemented but should be benchmarked against "load all to RAM" for memory usage vs I/O latency.
3.  **Pin Memory:** The benchmark warning shows `pin_memory` requires an accelerator (GPU) to be effective. Ensure this is tested in the real training loop.

## 4. Benchmark Script
A reusable benchmark script has been added at `tests/performance/benchmark_pipeline.py`.
