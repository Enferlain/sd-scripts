## Why

The optimization plan now has stable logical groups and a shared grouping module, but the execution side is still just an unnamed `parameter_groups` payload. That keeps the architecture half-finished because logical groups are first-class while execution groups are still represented as compatibility-shaped data without explicit ownership in the plan model.

## What Changes

- Add explicit execution-group modeling to the optimization plan.
- Rename the plan’s execution payload from generic `parameter_groups` to execution-oriented naming while preserving compatibility accessors.
- Update the shared fine-tune grouping helper and fine-tune mode to build the plan through explicit execution groups.
- Keep optimizer factory behavior unchanged by continuing to materialize legacy optimizer dicts at the construction boundary.
- Keep wrappers, scheduler-routing redesign, adapters, and fused/multi-optimizer execution out of scope for this slice.

## Capabilities

### New Capabilities
- `optimization-execution-group-foundation`: Represent execution optimizer groups as explicit plan-owned objects instead of an unnamed compatibility payload.

### Modified Capabilities

## Impact

- Affected code: `library/optimization/types.py`, `library/optimization/grouping.py`, `library/training/modes/finetune_mode.py`, and focused unit tests.
- Affected systems: optimization-plan data modeling, base-path grouping output, and optimizer-construction boundaries.
- Dependencies: no new runtime dependency; preserves existing optimizer factory and scheduler behavior.
