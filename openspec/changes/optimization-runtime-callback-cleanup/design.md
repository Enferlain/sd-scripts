## Context

The optimization-layer refactor has already moved scheduler routing, logical groups, execution groups, and optimizer runtime mode behavior onto `OptimizationPlan`. The base fine-tune path now consumes plan-owned runtime metadata through shared helpers, but the plan-aware build contract still carries `optimizer_train_fn` / `optimizer_eval_fn` fields that were originally introduced for schedule-free optimizers before the plan model existed.

Those callback fields no longer drive orchestration. They remain mostly as pass-through data from mode code into trainer state, which keeps the base-path API looking more legacy-shaped than it needs to. The adapter path still returns the older tuple and can remain compatibility-oriented until the later adapter redesign.

## Goals / Non-Goals

**Goals:**
- Remove the callback pair from the plan-aware optimizer build contract.
- Keep the fine-tune/base path fully functional through `OptimizationPlan` runtime metadata and shared helpers.
- Move any remaining callback-pair compatibility handling into the optimizer phase boundary instead of mode implementations.
- Tighten tests so plan-aware code proves it does not require callback-pair state.

**Non-Goals:**
- Redesign the adapter optimizer-preparation path.
- Remove the legacy tuple return contract from `TrainingMode`.
- Rework wrapper internals or change optimizer runtime behavior semantics.

## Decisions

### `OptimizerBuildResult` stops carrying train/eval callback fields

The plan-aware result should contain optimizer identity, the optimizer instance, and the optional `OptimizationPlan`. Runtime transitions are already modeled explicitly on the plan, so keeping callback fields in the first-class result only duplicates meaning and encourages new code to keep depending on them.

Alternative considered:
- Keep the fields but mark them deprecated.
  - Rejected because the base-path contract would still suggest they are part of the intended architecture.

### The optimizer phase owns callback compatibility fallback

The optimizer phase remains the normalization boundary between plan-aware and legacy mode outputs. When it receives a legacy tuple, it can still store the returned callbacks. When it receives `OptimizerBuildResult`, it should stop expecting mode-provided callbacks and can explicitly clear or populate trainer compatibility state as needed.

Alternative considered:
- Push compatibility callback creation back into every mode implementation.
  - Rejected because compatibility handling belongs at the shared boundary, not repeated across plan-aware modes.

### Trainer/orchestration tests should assert plan-owned runtime behavior directly

The tests should prove that runtime train/eval transitions happen through plan metadata and `apply_optimizer_runtime_mode(...)`, not through callback-pair side state on the trainer.

Alternative considered:
- Leave the older callback-oriented test setup in place.
  - Rejected because it hides whether orchestration is truly decoupled from legacy fields.

## Risks / Trade-offs

- [Legacy tests or tooling still inspect callback fields on trainer] -> Keep the legacy tuple path working and only simplify the plan-aware/base path in this slice.
- [The adapter path still emits the old tuple] -> Leave that contract intact and treat this change as base-path cleanup ahead of the adapter redesign.
- [Removing fields from `OptimizerBuildResult` touches several call sites at once] -> Keep the change narrow and add focused unit coverage for the updated normalization seam.
