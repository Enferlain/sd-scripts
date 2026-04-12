## Why

The optimization layer now has a solid repo-owned optimizer and scheduler surface, but the trainer-facing model still treats runtime `optimizer.param_groups` as the source of truth. That makes grouping, logging, scheduler ownership, and future wrapper/offload execution behavior harder to reason about because logical training intent is not represented as a stable shared object.

## What Changes

- Introduce a shared optimization-plan foundation for the active base-model training path.
- Add stable logical parameter-group objects that represent trainer-facing grouping intent separately from backend execution groups.
- Move training diagnostics and LR reporting toward logical-group metadata instead of reconstructing meaning from runtime optimizer groups.
- Thread the shared plan through the optimizer phase so optimizer construction, scheduler construction, and runtime preparation can consume one normalized input.
- Keep the initial change scoped to the active base fine-tune path; current adapter integration remains compatibility-oriented and is not used as the architecture target.

## Capabilities

### New Capabilities
- `optimization-logical-grouping`: Represent optimization intent as stable logical groups and shared plan metadata that survive wrapper/runtime execution rewrites.

### Modified Capabilities

## Impact

- Affected code: `library/optimization/types.py`, new shared optimization/grouping modules, `library/training/phases/optimizer.py`, `library/training/modes/base.py`, `library/training/modes/finetune_mode.py`, trainer/logging helpers, and focused unit tests.
- Affected systems: optimizer construction, scheduler attachment/routing, training diagnostics, LR logging, and future optimization-layer follow-up work.
- Dependencies: no new external runtime dependency; uses the existing `openspec` workflow for design/task artifacts.
