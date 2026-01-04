# Data Pipeline Audit: Phase 4 (DataLoader)

## Overview
This audit evaluates the state of the Phase 4 DataLoader implementation in `library/data/pipeline/dataloader.py` against the requirements for high-performance, multi-GPU training.

## Findings

### 1. Multi-GPU / Distributed Training Strategy
**Status:** ❌ **Incomplete**

The current `TrainingDataset` implementation iterates linearly through the provided `EpochManifest` without any sharding logic. In a distributed setting (e.g., Accelerate with multiple GPUs), every process receives the same manifest and will attempt to train on the exact same batches.

**Analysis:**
- `IterableDataset` is not automatically sharded by PyTorch's `DataLoader`.
- The `EpochManifest` generated in Phase 3 is a global manifest for the epoch.
- Runtime sharding is required to ensure each GPU processes a unique subset of the data.

**Recommended Action:**
Implement runtime sharding in `TrainingDataset.__iter__` using process rank and world size.

```python
def __iter__(self):
    # Pseudo-code for required logic
    if self.distributed:
        rank = get_rank()
        world_size = get_world_size()
    else:
        rank = 0
        world_size = 1

    for batch_idx, batch_info in enumerate(self.epoch_manifest.batches):
        if batch_idx % world_size != rank:
            continue
        yield self._load_batch(batch_info)
```

### 2. Pin Memory & Device Placement
**Status:** ❌ **Incorrect**

The current implementation creates a conflict between `DataLoader` pinning and manual device movement.
- `create_training_dataloader` enables `pin_memory=True` when a CUDA device is detected.
- `TrainingDataset._load_batch` explicitly moves tensors to the GPU (`.to(self.device)`).

**Analysis:**
- `pin_memory` is an optimization for CPU tensors to speed up transfer to GPU.
- If the Dataset yields GPU tensors, `pin_memory` causes overhead or errors.
- The standard practice is for the Dataset to yield CPU tensors, `DataLoader` to pin them, and the training loop (or Accelerator) to move them to the GPU.

**Recommended Action:**
- Remove the `device` argument from `TrainingDataset`.
- Remove all `.to(self.device)` calls in `_load_batch`.
- Keep `pin_memory=True` in `create_training_dataloader`.

### 3. Missing Batch Fields
**Status:** ⚠️ **Critical Data Missing**

The new `_load_batch` method returns a simplified dictionary that lacks fields required by the `train_network.py` and SDXL training loops.

| Field | Status | Source in New Pipeline | Issue |
| :--- | :--- | :--- | :--- |
| `original_sizes_hw` | ❌ Missing | Cache Metadata / `CacheEntry` | `load_cache` strategy returns tensors only, dropping metadata. |
| `crop_top_lefts` | ❌ Missing | Cache Metadata | `load_cache` strategy returns tensors only, dropping metadata. |
| `loss_weights` | ❌ Missing | `CacheEntry.is_reg` | Logic to convert `is_reg` to weight is missing. |
| `alpha_masks` | ❌ Missing | Cache File | Logic to check/load `alpha_mask` tensor is missing. |
| `flippeds` | ❌ Missing | Runtime Logic | No logic to decide/record flip status. |
| `adapter_multipliers` | ❌ Missing | Config | Missing entirely. |

**Critical Issue - Metadata Access:**
The `SdxlLatentsPipelineStrategy.load_cache` method uses `safetensors.torch.load_file`, which **does not return metadata**. The `crop_ltrb` (needed for `crop_top_lefts`) and `original_size` are stored in the safetensors metadata but are inaccessible via the current API.

**Recommended Actions:**
1.  **Update Strategy Interface:** Modify `load_cache` to return a tuple `(tensors, metadata)` or use `safe_open` to retrieve metadata.
2.  **Populate Fields:** Update `_load_batch` to construct these fields.
    - `original_sizes_hw`: From cache metadata (preferred) or `CacheEntry`.
    - `crop_top_lefts`: Must come from cache metadata (`crop_ltrb`).
    - `loss_weights`: Derive from `entry.is_reg`.
    - `flippeds`: Implement runtime flip probability (e.g., 50% if `flip_aug` enabled) and load `latents_flipped` accordingly.

### 4. Input IDs Format
**Status:** ℹ️ **Changed (Intentional)**

- **Legacy:** List of tensors (position-dependent).
- **New:** Dictionary `{'clip_l': ..., 'clip_g': ...}`.

**Analysis:**
This is a design change. The training scripts (`train_network.py`, etc.) must be updated to handle this dictionary format instead of expecting a list. This is better for supporting Flux/SD3 which have 3+ encoders.

## Summary of Tasks for Refactoring

1.  **Fix Sharding**: Add rank/world_size logic to `TrainingDataset.__iter__`.
2.  **Fix Device**: Ensure `TrainingDataset` yields CPU tensors.
3.  **Expose Metadata**: Update `CachingStrategy.load_cache` to return metadata (vital for SDXL crops).
4.  **Complete Batch Construction**: Add logic for `loss_weights`, `alpha_masks`, and `flipped` handling in `_load_batch`.
