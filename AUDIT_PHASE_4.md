# Audit Phase 4: Resume & Checkpointing

## 1. Executive Summary

This audit assesses the readiness of the data pipeline for **Resume & Checkpointing** functionality. The primary requirement is **Functional Continuation**, meaning training can resume from a checkpoint with equivalent model state and training progression, without requiring bit-exact reproducibility of data ordering or RNG states.

### Key Findings

| Component | Status | Recommendation |
| :--- | :--- | :--- |
| **Epoch Manifest** | ✅ Ready | Rely on deterministic regeneration (`seed + epoch`). No need to save to checkpoint. |
| **Mid-Epoch Resume** | ⚠️ Optimization Needed | Current `IterableDataset` incurs high I/O cost when skipping batches. **Fast Skip** logic is required. |
| **Token Files** | ⚠️ Logic Gap | Token files are currently blindly regenerated. Logic should be added to **reuse existing files** if valid. |
| **Storage** | ✅ Ready | Adopt **Ephemeral + Cleanup** strategy. Delete intermediate files after epoch completion. |

---

### Definition: Functional Continuation

**Functional Continuation** means:
- ✅ **Preserved:** Model checkpoint state (weights), Optimizer and Scheduler state, Training epoch and batch counters.
- ❌ **Not Preserved:** Data ordering within the resumed epoch (may differ from original run), RNG sequences (random states for augmentation/dropout).

---

## 2. Detailed Findings

### 2.1 Epoch Manifest Persistence

**Question:** Does epoch manifest need to be saved in checkpoint?
**Answer:** **No.**

**Reasoning:**
The `prepare_epoch()` function is fully deterministic given the `seed` and `epoch` number.
- **RNG:** `random.seed(seed + epoch)` ensures consistent shuffling per epoch.
- **Processing:** Caption processing (dropout, shuffling) uses the same seeded RNG.
- **Warmup:** Batch ordering (largest first) is deterministic.

**Action:**
- Do not add logic to save/load `epoch_manifest.json` from checkpoints.
- Ensure `seed` and `epoch` are correctly restored from the checkpoint state (handled by `sdxl_peft.py` logic).

### 2.2 Mid-Epoch Resume Efficiency

**Question:** How to resume mid-epoch with new dataloader?
**Answer:** **Regenerate manifest + Fast Skip.**

**Problem:**
The current `IterableDataset` implementation in `dataloader.py` loads data (latents, tokens) inside `__iter__` via `_load_batch()`.
When `accelerator.skip_first_batches()` is called during resume:
1. It creates an iterator from the dataset.
2. It calls `next()` `N` times to discard batches.
3. `TrainingDataset` **loads files from disk** for every discarded batch.
4. This causes significant delay (minutes) when resuming late in an epoch.

**Recommendation (Fast Skip):**
Modify `TrainingDataset` to accept a `start_batch_index` (or implement a specific skip method).
- **Logic:** `__iter__` should slice `self.epoch_manifest.batches` starting from `start_batch_index`.
- **Benefit:** Zero I/O for skipped batches. Instant resume.
- **Implementation:**
  ```python
  # dataloader.py
  def __iter__(self):
      # ...
      # Skip logic
      skipped_batches = self.epoch_manifest.batches[self.start_index:]
      for batch_info in skipped_batches:
          # load and yield
  ```

### 2.3 Token File Handling on Resume

**Question:** Token file handling on resume?
**Answer:** **Reuse if valid, Regenerate if missing.**

**Problem:**
Currently, `tokenize_epoch_manifest` blindly generates `tokens.safetensors`, overwriting any existing file.
If a crash occurs mid-epoch, `tokens_epoch_N.safetensors` likely exists and is valid. Regenerating it takes time (dataset size dependent).

**Recommendation:**
Update `epoch_preparation.py` to check for file existence and validate:
1. **Check:** Does file exist?
2. **Validate:** load metadata, check `manifest_hash`.
3. **Action:** If valid, return path. If invalid/missing, regenerate.

### 2.4 Storage Strategy

**Strategy:** **Ephemeral + Cleanup**

- **Creation:** Generate `epoch_manifest.json` and `tokens.safetensors` at start of epoch.
- **Usage:** Used by `DataLoader` during epoch.
- **Cleanup:** Delete files immediately after epoch loop finishes.
- **Resume:** If files are missing (cleanup happened or new machine), they are automatically regenerated (see 2.1 & 2.3).

---

## 3. Implementation Plan

### Step 1: Optimize Dataloader for Skipping
Modify `TrainingDataset` in `library/data/pipeline/dataloader.py`:
- Add `start_epoch_step` argument to `__init__`.
- Update `__iter__` to skip `batch_info` entries before the main loop.
- **Logic:** This `start_epoch_step` replaces the *data loading* part of `accelerator.skip_first_batches`. The accelerator's method may still be called to fast-forward the scheduler/optimizer, but the dataset will yield instantly for skipped steps.
- **State:** `token_offset` must be correctly advanced by `sum(batch_size)` of all skipped batches to ensure proper alignment when sequential access resumes.

### Step 2: Smart Token File Generation
Modify `tokenize_epoch_manifest` in `library/data/pipeline/epoch_preparation.py`:
- Use `load_epoch_tokens` (existing function) or `safetensors.safe_open` to verify `manifest_hash`.
- Skip tokenization if hash matches.
- **Resilience:** If file exists but is corrupt (e.g., `safe_open` fails), catch exception and regenerate.

### Step 3: Cleanup Logic
Update `scripts/sdxl_peft.py` (and similar scripts):
- Create a helper `cleanup_epoch_files(epoch_manifest_path, token_path)`.
- Call this at the **end of the epoch loop** (after validation and saving).
- **Multi-machine:** If resume happens on a new machine where files don't exist, the standard generation logic (Step 2) will naturally regenerate them.

### Step 4: Verify `initial_step` Logic
The `calculate_initial_step` in `trainer_utils.py` logic handles the calculation.
- Ensure `sdxl_peft.py` passes the calculated `initial_step` (converted to batches) to the `TrainingDataset` or `DataLoader`.
- **Clarification:** `accelerator.skip_first_batches` is mainly for aligning the **Learning Rate Scheduler** and **Optimizer** states. The `TrainingDataset` fast-skip is for **Data I/O Efficiency**. Both are needed: accelerator ensures mathematical correctness, dataset ensures performance.

### Open Questions & Edge Cases

1.  **Multi-machine Resume:**
    - *Scenario:* Machine A crashes after cleaning up epoch N. Machine B resumes from start of epoch N+1.
    - *Handling:* `prepare_epoch` generates manifest for N+1. `tokenize_epoch_manifest` generates tokens for N+1. Everything works as a fresh epoch.
    - *Scenario:* Machine A crashes mid-epoch N. Machine B resumes.
    - *Handling:* Machine B does not have `tokens_epoch_N.safetensors`. `tokenize_epoch_manifest` sees missing file and regenerates it. Resume proceeds.

2.  **Partial Token File Corruption:**
    - *Scenario:* Process killed while writing `tokens.safetensors`.
    - *Handling:* `tokenize_epoch_manifest` validation (opening file, checking metadata) will fail. Exception handling should catch this and trigger regeneration.
