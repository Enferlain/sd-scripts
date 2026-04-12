## Why

The optimization plan now has explicit logical and execution groups, but scheduler ownership is still inferred ad hoc from optimizer registration capabilities, wrapper detection, and runtime inspection inside `get_scheduler_fix(...)`. That keeps an important execution concern outside the plan model even though the plan is becoming the orchestration-facing source of truth.

## What Changes

- Add explicit scheduler/runtime metadata to the optimization plan.
- Populate that metadata in the optimizer phase once the actual optimizer runtime object exists.
- Update the scheduler entrypoint to consume plan-provided scheduler/runtime metadata when available, while preserving the current fallback behavior for legacy callers.
- Keep wrapper/offload behavior unchanged; this slice only makes scheduler ownership and target selection explicit.
- Keep adapters, fused execution, and broader runtime remapping out of scope for this change.

## Capabilities

### New Capabilities
- `optimization-scheduler-runtime-metadata`: Represent scheduler ownership and scheduler target selection as explicit optimization-plan metadata for orchestration.

### Modified Capabilities

## Impact

- Affected code: `library/optimization/types.py`, `library/optimization/scheduler.py`, `library/training/phases/optimizer.py`, and focused scheduler/optimizer-phase tests.
- Affected systems: scheduler construction, wrapper/base-optimizer routing, and optimization-plan runtime metadata.
- Dependencies: no new runtime dependency; preserves the existing scheduler registry and optimizer factory behavior.
