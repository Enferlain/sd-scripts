# Audit Report: Resume from Checkpoint Behavior

**Date:** 2026-01-20
**Topic:** Resume from Checkpoint in `PeftTrainer` Architecture
**Status:** 🔴 **Issues Found**

## Executive Summary

The audit confirms that the current implementation of "Resume from Checkpoint" is **incomplete and incorrect** in the new `PeftTrainer` architecture. While the mechanism to load the step count from `train_state.json` exists, it fails to properly propagate this state to the `trainer.global_step` and the `SimpleNamespace` state containers.

As a result, resuming a training run will cause the global step counter to **reset to 0**, leading to incorrect logging, incorrect future checkpoint names, and potentially incorrect training duration (depending on how `max_train_steps` is interpreted relative to the reset counter).

## Findings

### 1. `register_adapter_state_hooks` Implementation
**Question:** Are the `SimpleNamespace` objects (`trainer._current_epoch_state`, `trainer._current_step_state`) correctly updated by the hooks?
**Answer:** **No.**

The `load_model_hook` defined in `library/training/checkpointing.py` reads the `train_state.json` file but **does not update** the `current_epoch` or `current_step` arguments passed to `register_adapter_state_hooks`. It only updates a local `state_container` dictionary.

```python
# library/training/checkpointing.py

def load_model_hook(models, input_dir):
    # ...
    if os.path.exists(train_state_file):
        with open(train_state_file, encoding="utf-8") as f:
            data = json.load(f)
        state_container["steps_from_state"] = data["current_step"]
        # MISSING: current_epoch.value = data["current_epoch"]
        # MISSING: current_step.value = data["current_step"]
```

### 2. `PeftTrainer` Attribute Restoration
**Question:** Does `accelerator.load_state()` correctly restore all trainer attributes?
**Answer:** **Partially (but critically failed on `global_step`).**

*   **`trainer._initial_step`**: ✅ Correctly set to the resumed step count (e.g., 11).
*   **`trainer.epoch_to_start`**: ✅ Correctly derived from `_initial_step`.
*   **`trainer.global_step`**: ❌ **Reset to 0.**

In `library/training/phases/optimizer.py`, `trainer.global_step` is explicitly reset to 0 *after* the resume logic:

```python
# library/training/phases/optimizer.py

    # ... resume logic ...
    steps_from_state = get_steps_from_state()

    # Calculate initial step for resuming
    trainer._initial_step = 0
    trainer.epoch_to_start = 0
    if steps_from_state is not None:
        trainer._initial_step = steps_from_state
        trainer.epoch_to_start = trainer._initial_step // trainer.num_batches_per_epoch

    trainer.global_step = 0  # <--- BUG: This ignores steps_from_state
```

### 3. Training Loop Propagation
**Question:** Are `initial_step`, `epoch_to_start` correctly propagated to the training loop?
**Answer:** **Yes, but with side effects due to `global_step` reset.**

The training loop (`library/training/phases/training_loop.py`) correctly uses `trainer.epoch_to_start` and `trainer._initial_step` to skip data.
*   It skips `epoch_to_start` epochs.
*   It skips `_initial_step` batches in the current epoch.

**However**, because `trainer.global_step` starts at 0:
1.  The first processed batch after resume will be logged as **Step 1**, not **Step N+1**.
2.  `trainer._current_step_state.value` is updated to `trainer.global_step` (0), essentially de-synchronizing the state.
3.  The check `trainer.global_step >= cfg.training.max_train_steps` will be evaluated against a reset counter, effectively extending training beyond the intended total steps.

## Verification

A reproduction script `tests/audit/test_resume_logic.py` was created to mock the `Accelerator` and `PeftTrainer` resume flow.

**Test Scenario:**
1.  Train for 10 steps. Checkpoint at step 10 (saved state says `current_step: 11`).
2.  Resume from this checkpoint.

**Test Results:**
*   `trainer._initial_step`: **11** (Correct)
*   `trainer.epoch_to_start`: **1** (Correct)
*   `trainer.global_step`: **0** (Incorrect, expected 11)

## Recommendations

1.  **Update `trainer.global_step` on Resume:**
    Modify `library/training/phases/optimizer.py` to initialize `trainer.global_step` from `steps_from_state` if it exists.

    ```python
    if steps_from_state is not None:
        trainer.global_step = steps_from_state
    else:
        trainer.global_step = 0
    ```

2.  **Update `load_model_hook` (Optional but Recommended):**
    While the current `optimizer.py` logic manually sets `_initial_step`, it is safer if the hook itself updates the `SimpleNamespace` objects passed to it. This ensures that `trainer._current_step_state` is correct immediately after load, preventing any window where the state is desynchronized.

3.  **Refactor `register_adapter_state_hooks`:**
    Consider making `register_adapter_state_hooks` return a dictionary or object containing all restored state (epoch and step), rather than relying on side-effects or a closure-bound container.
