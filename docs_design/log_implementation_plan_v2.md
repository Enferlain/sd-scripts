# Training Logging Plan v2 (2026-02-23)

## Objective

Improve logging architecture and correctness without reducing currently visible training signals.

## Non-Downgrade Constraint

This plan must preserve all current categories of visibility:

1. Console lifecycle logs (model load, caching, optimizer/scheduler creation, training progress).
2. Startup diagnostics block (components, trainable counts, optimizer groups).
3. Progress bar and per-step status postfix.
4. Tracker metrics (TensorBoard/W&B/other Accelerate trackers).

## Verified Findings (Current Code)

### 1. Logging structure is mostly good

1. `Trainer` startup diagnostics are centralized and mode-agnostic:
   - `library/training/runners/trainer.py` (`_log_training_info`)
   - `library/training/trainer_utils.py` (`log_training_diagnostics`)
2. Per-step tracker emission is centralized:
   - `library/logging/step_logging.py`
3. Training loop focuses on orchestration and delegates metric generation:
   - `library/training/phases/training_loop.py`

### 2. Correctness issues that should be fixed first

1. **Duplicate/overlapping LR metric emission** in `generate_step_logs`:
   - `library/logging/step_logging.py`
   - A `for ... else` block causes the fallback LR logging path to run every time, not only when needed.
2. **`wandb_run_name` can be overwritten** by `log_tracker_config`:
   - `library/logging/step_logging.py`
   - `init_kwargs["wandb"]` is set, then replaced when `log_tracker_config` exists.

### 3. Architecture issues (maintainability and consistency)

1. `setup_logging()` is called at import time in multiple modules, then reconfigured later with `reset=True`.
2. Mixed use of `logger.info` and `accelerator.print` produces inconsistent multi-process behavior (some messages main-process only, some all-rank).
3. No built-in `log_every_n_steps` control for tracker writes (can be noisy for long runs).

## Recommended Target Architecture

### A. Single logging bootstrap

1. Configure logging once per process from the script entrypoint (or one explicit initialization location).
2. Remove import-time `setup_logging()` side effects from library modules.
3. Keep runtime config-driven behavior (`console_log_level`, file/simple mode) unchanged.

### B. Rank-aware console contract

1. Introduce thin helpers for:
   - main-process messages
   - all-process debug messages (rare)
2. Use one style consistently in trainer/training loop startup and lifecycle logs.

### C. Tracker emission contract

1. Keep metric key schema stable where possible.
2. Fix LR emission path so each key is written once per step.
3. Merge tracker init kwargs deterministically (do not drop `wandb_run_name`).
4. Add optional `log_every_n_steps` in logging config (default `1` to preserve behavior).

## Implementation Plan

### Phase 0: Correctness patch (low risk, immediate value)

1. Refactor LR logging in `generate_step_logs` to remove `for ... else` duplication.
2. Fix `init_trackers` kwargs merge:
   - start with `wandb_run_name` if set
   - deep-merge/overlay `log_tracker_config` on top
3. Add focused unit tests for:
   - LR key set uniqueness
   - W&B run name retention with/without extra tracker config

### Phase 1: Logging initialization cleanup

1. Remove import-time `setup_logging()` calls from:
   - `library/training/runners/trainer.py`
   - `library/training/phases/training_loop.py`
   - `library/training/trainer_utils.py`
2. Ensure one explicit initialization point still configures root handlers before meaningful logs are emitted.
3. Verify console/file modes remain equivalent to current behavior.

### Phase 2: Output consistency (rank-aware)

1. Add helper(s) in shared training/logging utility for rank-aware human logs.
2. Convert key trainer/training-loop startup messages to helpers for consistency.
3. Keep third-party/module-local lifecycle logs unchanged unless they cause duplication.

### Phase 3: Tracker volume control

1. Add `log_every_n_steps: int = 1` to `LoggingConfig`.
2. Gate step tracker emission by this interval.
3. Keep progress bar updates every optimization step (no downgrade in local feedback).

## Acceptance Criteria

1. Console output still includes all current lifecycle categories.
2. Startup diagnostics block still appears in PEFT and fine-tune runs.
3. Tracker metrics do not contain duplicated LR series caused by control-flow bug.
4. `wandb_run_name` works when `log_tracker_config` is also provided.
5. Multi-GPU logs avoid accidental all-rank duplication for startup/status messages.

## Verification Checklist

1. Run one PEFT config and one fine-tune config.
2. Compare startup output before/after for category parity.
3. Confirm tracker keys for LR are stable and singular.
4. Confirm W&B run name is applied when configured.
5. Run targeted unit tests for step logging and tracker initialization behavior.

## Out of Scope

1. Replacing `accelerate` tracker integration.
2. Redesigning model/caching lifecycle log wording.
3. Introducing a mandatory JSON logging pipeline.
