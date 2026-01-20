# Audit 1: Training Loop Integration

This document details the findings of the training loop integration audit for the refactored data pipeline.

## Summary

| Question | Status | Notes |
| :--- | :--- | :--- |
| How to handle `current_epoch` / `current_step` References? | ✅ Resolved | Handled during Phase 3 (Epoch Preparation) rather than runtime. |
| How to integrate with `accelerator.prepare()`? | ⚠️ Warning | `IterableDataset` requires manual sharding; `accelerator.prepare()` should arguably **not** be used on the DataLoader, or configured to not shard. |
| Does `skip_first_batches` work with `IterableDataset`? | ⚠️ Performance | Works functionally but inefficiently (linear scan). Recommended to implement manifest slicing for resume. |
| How to handle `train_dataset_group.set_max_train_steps()`? | ✅ Resolved | Moved to Phase 3 (`prepare_epoch`) where token warmup is calculated. |

## Detailed Findings

### 1. `current_epoch` / `current_step` References
**Legacy:** Used `multiprocessing.Value` objects (`current_epoch`, `current_step`) shared between the training loop and the `Dataset` (via `collator`). Logic like token warmup and bucket shuffling happened inside `__getitem__` or `set_current_epoch` based on these values.

**New Pipeline:**
- **Epoch:** `prepare_epoch` (Phase 3) accepts `epoch` as an argument. It generates a static `EpochManifest` for that specific epoch. All epoch-dependent logic (shuffling, seed generation) is baked into this manifest before training starts.
- **Step:** `prepare_epoch` also accepts `current_step`. Token warmup logic (limiting tags based on progress) is executed during manifest generation by `process_caption`.
- **Conclusion:** The `TrainingDataset` (Phase 4) does not need to know the current step or epoch. It simply serves the pre-computed batches from the manifest.

### 2. Integration with `accelerator.prepare()`
**Legacy:** `train_dataloader` is passed to `accelerator.prepare()`, which handles sharding for distributed training (splitting batches across GPUs).

**New Pipeline:**
- The `TrainingDataset` in `library/data/pipeline/dataloader.py` implements manual sharding in `__iter__`:
  ```python
  is_this_rank = (batch_idx % self.world_size) == self.rank
  ```
- **Risk:** If `accelerator.prepare()` is called on a DataLoader wrapping this `TrainingDataset`, it might attempt to shard the data *again* or fail because it's an `IterableDataset`.
- **Recommendation:** Do not pass the new DataLoader to `accelerator.prepare()`, OR ensure `accelerator` is configured to not shard this specific dataloader. The manual sharding in `TrainingDataset` is already correct for distributed training.

### 3. `skip_first_batches` with `IterableDataset`
**Legacy:** `accelerator.skip_first_batches(train_dataloader, steps)` is used to resume training. On a standard map-style `Dataset`, this is efficient.

**New Pipeline:**
- `TrainingDataset` is an `IterableDataset`.
- `accelerator.skip_first_batches` works by calling `next(iterator)` N times and discarding the result.
- **Performance Impact:** For a large `initial_step`, this will trigger N iterations of logic (though mostly just indexing if cached). While safer than loading images, it still incurs overhead.
- **Optimization:** Since `EpochManifest` contains a list of batches, we can implement "skipping" natively in `TrainingDataset` by slicing `self.epoch_manifest.batches` in `__init__` or `__iter__`, passing `initial_step` to the constructor. This converts O(N) skipping to O(1).

### 4. `train_dataset_group.set_max_train_steps()`
**Legacy:** The training script calculates `max_train_steps` and calls `set_max_train_steps()` on the dataset group. This is used internally for token warmup calculations (e.g., `current_step / max_train_steps`).

**New Pipeline:**
- Token warmup is now a "build-time" operation for the epoch, not a "runtime" operation.
- `prepare_epoch` function signature in `library/data/pipeline/epoch_preparation.py`:
  ```python
  def prepare_epoch(..., current_step=0, max_train_steps=0):
  ```
- **Flow:** The training loop (or a supervisor script) calculates `max_train_steps` once, then passes it to `prepare_epoch` at the start of every epoch. The resulting manifest contains captions with the correct tokens already selected.
- **Conclusion:** The method `set_max_train_steps()` is obsolete for the runtime dataset.
