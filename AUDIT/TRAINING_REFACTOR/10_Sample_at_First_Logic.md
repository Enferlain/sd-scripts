# Audit Report: Sample-at-First Logic

## 1. Summary

| Aspect | Status | Notes |
| :--- | :--- | :--- |
| **Global Step Resume** | **CRITICAL FAIL** | `global_step` is not synchronized with resumed step count. |
| **Sampling Execution** | **PASS** | `sample_images_common` logic and RNG restoration are correct. |
| **Validation Logic** | **PASS** | Helper functions `calculate_val_loss_check` are correctly implemented. |

---

## 2. Critical Findings

### 2.1 Missing `global_step` Synchronization on Resume

**Severity:** Critical
**Impact:** Broken logging, checkpoint naming, step-based triggers, and potentially incorrect LR scheduling visualization.

**Description:**
In the legacy script (`sdxl_peft.py`), `global_step` is updated to match `initial_step` (the step count loaded from the checkpoint) before the training loop starts. In the new `PeftTrainer` architecture, `global_step` is initialized to `0` and **never updated** to reflect the resumed state before the loop begins.

The training loop increments `global_step` starting from `0` (or `0 + 1`), meaning a resumed training session (e.g., from step 1000) will log as "Step 1", "Step 2", etc., instead of "Step 1001".

**Code Comparison:**

**Legacy (`scripts/sdxl_peft copy.py`):**
```python
# ... (after sampling checks) ...
if initial_step > 0:
    for skip_epoch in range(epoch_to_start):
        logger.info(f"skipping epoch {skip_epoch + 1} ...")
        initial_step -= num_batches_per_epoch
    global_step = initial_step  # <--- CRITICAL UPDATE HERE
```

**New (`library/training/phases/optimizer.py`):**
```python
# ...
if steps_from_state is not None:
    trainer._initial_step = steps_from_state
    # ...

trainer.global_step = 0  # <--- Initialized to 0, never updated to _initial_step
```

**New (`library/training/phases/training_loop.py`):**
```python
# ...
dataloader_iter = iter(train_dataloader)
if trainer._initial_step > 0:
    # Skips data, but does NOT update global_step
    dataloader_iter = itertools.islice(dataloader_iter, trainer._initial_step - 1, None)
    trainer._initial_step = 1

for step, batch in enumerate(dataloader_iter):
    trainer.global_step += 1  # Counts up from 0
```

---

## 3. Detailed Audit Responses

### Question 1: Imports `sample_images_check` and `calculate_val_loss_check` - are these correctly implemented?

**Status:** **YES**

*   **Implementation:** `sample_images_check` correctly checks `sampling_config.sample_at_first` for step 0. `calculate_val_loss_check` correctly handles step/epoch intervals.
*   **Imports:** In `PeftTrainer._maybe_sample_at_start`, these are imported locally:
    ```python
    from library.training.sample_generation import sample_images_check
    from library.training.trainer_utils import calculate_val_loss_check
    ```
    This is valid and works as expected.

### Question 2: Mode switching (eval → train) - is it complete?

**Status:** **YES**

The `_maybe_sample_at_start` method in `PeftTrainer` correctly handles the mode switch:

1.  **Switch to Eval:** `self.accelerator.unwrap_model(self.adapter).eval()`
2.  **Switch Optimizer:** `self.optimizer_eval_fn()`
3.  **Perform Sampling/Validation**
4.  **Switch Back to Train:** `self.optimizer_train_fn()`
5.  **Switch Back to Train Mode:** `self.accelerator.unwrap_model(self.adapter).train()`

This matches the legacy logic perfectly.

### Question 3: Does this affect training state?

**Status:** **PARTIAL** (RNG is safe, but `global_step` state is broken)

*   **RNG State:** **SAFE**. `sample_images_common` in `library/training/sample_generation.py` correctly saves and restores the RNG state for both CPU and CUDA:
    ```python
    rng_state = torch.get_rng_state()
    # ... sampling ...
    torch.set_rng_state(rng_state)
    ```
    This ensures that the "Sample at First" operation does not disturb the determinism of the training loop that follows.

*   **Global Step State:** **BROKEN**. As detailed in the Critical Findings, the `global_step` variable is not correctly synchronized when resuming training, leading to incorrect state tracking.

## 4. Recommendations

1.  **Fix `global_step` Initialization:** In `library/training/phases/optimizer.py`, immediately after calculating `trainer._initial_step` from the resumed state, set `trainer.global_step = trainer._initial_step`.
2.  **Verify Loop Logic:** Ensure `run_training_loop` respects the updated `global_step` and doesn't reset it or count incorrectly.
