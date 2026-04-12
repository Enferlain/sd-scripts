## Why

The optimization layer now models scheduler ownership, execution groups, and optimizer train/eval runtime behavior on `OptimizationPlan`, but the base fine-tune path still threads a legacy `optimizer_train_fn` / `optimizer_eval_fn` pair through mode return values and trainer state. That extra plumbing no longer drives orchestration, so keeping it in the primary contract makes the plan-aware path look more transitional than it really is.

## What Changes

- Remove the train/eval callback pair from the plan-aware `OptimizerBuildResult` contract.
- Update the fine-tune optimizer build path to stop constructing legacy optimizer runtime callbacks.
- Keep the legacy tuple-based adapter path working, but treat callback-pair handling as compatibility behavior owned by the optimizer phase instead of the base-path mode contract.
- Tighten trainer/runtime tests so plan-aware orchestration proves it does not depend on callback-pair state.

## Capabilities

### New Capabilities
- `optimization-runtime-transitions`: The optimization layer owns plan-aware optimizer runtime transitions without requiring mode implementations to return legacy train/eval callback pairs.

### Modified Capabilities

## Impact

- Affected code: `library/optimization/types.py`, `library/optimization/optimizer_utils.py`, `library/training/modes/base.py`, `library/training/modes/finetune_mode.py`, `library/training/phases/optimizer.py`, `library/training/runners/trainer.py`
- Affected tests: `tests/unit/training/test_training_optimizer.py`, `tests/unit/training/modes/test_finetune_mode.py`, `tests/unit/training/phases/test_optimizer.py`, plus any trainer/orchestration coverage that still assumes callback-pair state
- No external dependencies or user-facing config changes
