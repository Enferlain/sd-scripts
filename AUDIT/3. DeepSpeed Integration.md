# DeepSpeed Integration Audit

**Status**: ✅ Fully Supported (Parity with Legacy)

## Executive Summary

The DeepSpeed integration has been successfully ported to the new `PeftTrainer` architecture. The logic for model wrapping, optimizer preparation, and training loop execution matches the legacy implementation in `scripts/sdxl_peft copy.py`.

## Detailed Analysis of Concerns

### 1. `_prepare_with_accelerator()` DeepSpeed Branch
**Concern**: Is the DeepSpeed branch in `_prepare_with_accelerator` well-tested/correct?

**Finding**:
The logic is correctly implemented in `library/training/phases/optimizer.py`.
- It checks `if cfg.performance.deepspeed:`.
- It calls `deepspeed_utils.prepare_deepspeed_model` with the correct components (`unet`, `text_encoders`, `adapter`).
- It prepares the `ds_model`, `optimizer`, and `lr_scheduler` using `accelerator.prepare`.
- It correctly assigns `trainer._training_model = ds_model` which is used by the gradient synchronization context in the training loop.

### 2. `ds_model` Wrapping
**Concern**: Does `ds_model` correctly wrap unet + text_encoders + adapter?

**Finding**:
Yes. `library/performance/deepspeed_utils.py` defines `prepare_deepspeed_model` which:
- Accepts `unet`, `text_encoder1/2`, and `adapter` as keyword arguments.
- Wraps them in a `DeepSpeedWrapper` (which is a `torch.nn.ModuleDict`).
- Patches the `forward` method of each component to use `torch.autocast` (if mixed precision is enabled).
- This ensures that when `strategies.process_batch` calls `trainer.unet(...)` or `trainer.adapter(...)`, it executes the patched forward method, and the parameters are part of the `ds_model` ModuleDict, ensuring DeepSpeed manages them.

### 3. Optimizer and Scheduler Preparation
**Concern**: Are the optimizer and scheduler correctly prepared with DeepSpeed?

**Finding**:
Yes. In `library/training/phases/optimizer.py`:
```python
ds_model, trainer.optimizer, trainer.lr_scheduler = trainer.accelerator.prepare(
    ds_model, trainer.optimizer, trainer.lr_scheduler
)
```
This ensures `accelerate` (and underlying DeepSpeed) correctly wraps the optimizer and scheduler.

## Comparison with Reference (`scripts/sdxl_peft copy.py`)

| Feature | Legacy Implementation | New Implementation | Status |
| :--- | :--- | :--- | :--- |
| **Model Wrapping** | `deepspeed_utils.prepare_deepspeed_model(...)` | `deepspeed_utils.prepare_deepspeed_model(...)` in `optimizer.py` | ✅ Match |
| **Accelerator Prepare** | `accelerator.prepare(ds_model, optimizer, lr_scheduler)` | `accelerator.prepare(ds_model, optimizer, lr_scheduler)` in `optimizer.py` | ✅ Match |
| **Training Model** | `training_model = ds_model` | `trainer._training_model = ds_model` | ✅ Match |
| **Loop Context** | `determine_grad_sync_context(..., training_model)` | `determine_grad_sync_context(..., trainer._training_model)` | ✅ Match |

## Potential Issues / Regressions

### 1. `sys.path.append` Location
In `library/training/phases/model_prep.py`:
```python
sys.path.append(os.path.dirname(__file__))
```
In the legacy script, `__file__` referred to `scripts/sdxl_peft.py`, so it added the `scripts/` directory to `sys.path`.
In the new implementation, `__file__` refers to `library/training/phases/model_prep.py`, so it adds `library/training/phases/` to `sys.path`.

**Impact**: If a user has a custom adapter module located in the `scripts/` directory and relies on it being importable without `scripts.` prefix, this might fail.
**Mitigation**: Users should place custom modules in `library/` or ensure their python path is correct. This is likely a minor edge case but worth noting.

### 2. ZeRO-3 Saving (Existing Legacy Behavior)
The saving logic uses `accelerator.unwrap_model(trainer.adapter).save_weights(...)`.
In the DeepSpeed case, `trainer.adapter` is the *raw* `PeftModel` (not the prepared `ds_model`). `accelerator.unwrap_model` will return it as-is.
If using ZeRO-3, the weights in the raw model might be partitioned. `save_weights` usually relies on `state_dict()`. Unless `save_weights` internally handles `zero.GatheredParameters`, saving might produce empty weights on rank 0.
*Note: This behavior is identical to the legacy script, so it is not a regression introduced by refactoring.*
