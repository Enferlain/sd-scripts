# Audit Report: Phase Ordering & Dependencies

> **Status: ✅ ADDRESSED** (2026-01-20)

## 1. Phase Execution Order

The current phase execution order in `PeftTrainer.train()` is confirmed as:

1.  **`setup()`**: Initialize accelerator, logging, and dataset manifests.
2.  **`run_caching()`**: Cache latents and text encoder outputs (if enabled).
3.  **`prepare_models()`**: Load adapter and configure model precision.
4.  **`prepare_optimizer()`**: Create optimizer/scheduler, wrap with accelerator, and setup gradient checkpointing.
5.  **`run_training_loop()`**: Execute the main training loop.

This high-level ordering is logical, but specific dependencies within phases have issues detailed below.

## 2. Redundant `configure_precision()` Calls

**Finding:**
The `configure_precision()` function is currently called in two places:

1.  **`prepare_models()`** (Phase 3): This is the semantic home for model configuration.
2.  **`prepare_optimizer()`** (Phase 4): A duplicate call exists here.

**Analysis (Comparison with Legacy):**

- In the legacy script (`scripts/sdxl_peft copy.py`), precision configuration (like casting the adapter to fp16) happened _after_ optimizer creation (lines 381+), but _before_ `accelerator.prepare` was called on the adapter (line 420).
- The refactored code correctly places a call in `prepare_models()` (Phase 3).
- However, it _also_ retains a call in `prepare_optimizer()` (Phase 4).
- The call in `prepare_optimizer()` happens _before_ `accelerator.prepare` (via `_prepare_with_accelerator`), essentially matching the legacy timing, but rendering the Phase 3 call redundant.

**Recommendation:**

- **Remove** the duplicate call from `prepare_optimizer()`.
- **Keep** the call in `prepare_models()`.
- Rationale: Configuring model precision belongs in the model preparation phase. Doing it before optimizer creation is safe and semantically cleaner.

> **✅ RESOLVED:** Removed redundant call from `optimizer.py`. The call in `prepare_models()` is now the single location.

## 3. Gradient Checkpointing Timing Risk

**Finding:**
Gradient checkpointing is enabled in `prepare_optimizer()` via `_setup_gradient_checkpointing()`. Crucially, this happens **after** `accelerator.prepare()` has already wrapped the models.

```python
# In prepare_optimizer()
_prepare_with_accelerator(trainer)    # Models wrapped here
_setup_gradient_checkpointing(trainer) # Gradient checkpointing enabled here
```

**Analysis (Legacy Behavior Preserved):**

- This exact ordering exists in the legacy script (`scripts/sdxl_peft copy.py`):
  - `accelerator.prepare()` is called at line 420.
  - `unet.enable_gradient_checkpointing()` is called at line 434.
- While this proves the refactor didn't introduce a _new_ regression, the underlying risk remains.

**Risk Assessment:**

- **Standard Training:** Usually fine.
- **Distributed Data Parallel (DDP):** Enabling gradient checkpointing _after_ wrapping with DDP is risky. When `cpu_offload=True` is used with gradient checkpointing, DDP wrappers may not correctly handle the offloaded activation checkpoints if the hooks are not registered in the expected order.

**Recommendation:**

- **Flag for Manual Verification:** This specific timing (Checkpointing AFTER Accelerator Prepare) requires manual testing in a distributed environment (multi-GPU) with `cpu_offload=True`.
- **Potential Fix:** If tests fail, move `_setup_gradient_checkpointing()` to before `_prepare_with_accelerator()`, or move it entirely to `prepare_models()` (Phase 3).

> **✅ DOCUMENTED:** Added inline comment in `optimizer.py` referencing this audit. Matches legacy behavior; flagged for future multi-GPU testing with `cpu_offload=True`.

## 4. Training Flags State Consistency

**Finding:**
The flags `_train_unet` and `_train_text_encoder` are set in `prepare_optimizer()` (Phase 4). However, earlier phases (like `prepare_models` -> `create_adapter`) need this information. Currently, they re-calculate it on the fly:

```python
# In prepare_models / create_adapter:
train_unet = trainer.strategies.is_train_unet(cfg) # Calculated locally

# In prepare_optimizer:
trainer._train_unet = trainer.strategies.is_train_unet(cfg) # Stored in state
```

**Analysis:**
This is not a functional bug, as the calculation is deterministic based on the config. However, it represents a minor state management inconsistency where "truth" is calculated in multiple places rather than being established once in `setup()`.

**Recommendation:**

- Low priority. Consider moving the flag initialization to `setup()` or `prepare_models()` so `trainer._train_unet` is the single source of truth for all phases.

> **✅ RESOLVED:** Moved flag initialization to `create_adapter()` in `model_prep.py`. Added comment in `optimizer.py` noting the single source of truth location.
