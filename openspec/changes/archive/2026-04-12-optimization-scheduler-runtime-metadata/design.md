## Context

The optimization-layer architecture now has:

- trainer-facing logical groups
- explicit execution groups
- shared base-path grouping policy

The next remaining orchestration leak is scheduler ownership. Today `get_scheduler_fix(...)` still decides whether to return a dummy scheduler or whether to schedule `optimizer` vs `base_optimizer` by re-deriving that behavior from registry capabilities, wrapper-name heuristics, and compatibility flags. That logic is valid, but it is not yet represented as plan-owned runtime metadata.

The optimizer phase is the clean place to improve this because it already owns both the built optimizer instance and scheduler creation. That means it can populate plan runtime metadata after optimizer construction and before scheduler creation without pushing runtime concerns back into mode code.

## Goals / Non-Goals

**Goals:**
- Add explicit scheduler/runtime metadata to `OptimizationPlan`.
- Populate that metadata centrally in the optimizer phase.
- Make `get_scheduler_fix(...)` consume plan metadata when available.
- Preserve existing fallback behavior for legacy callers and compatibility paths that do not provide a plan.

**Non-Goals:**
- Reworking wrapper implementations or changing their runtime behavior.
- Redesigning scheduler registration or adding new scheduler kinds beyond explicit metadata wiring.
- Moving scheduler/runtime ownership into training modes.
- Reworking adapters/PEFT, fused execution, or multi-optimizer orchestration.

## Decisions

### 1. Put scheduler/runtime metadata on `OptimizationPlan`

Decision:
- Add a dedicated scheduler/runtime metadata object to the plan model, with fields for scheduler mode and scheduler target.

Rationale:
- Scheduler ownership is orchestration-facing runtime metadata, so it belongs with the plan rather than staying implicit in the scheduler builder.
- This keeps the plan moving toward “one shared source of orchestration truth.”

Alternatives considered:
- Keep all scheduler routing logic only inside `get_scheduler_fix(...)`: rejected because that leaves the plan incomplete as a runtime orchestration model.
- Put metadata on optimizer registrations only: rejected because runtime wrapper shape and compatibility flags are still resolved after optimizer construction.

### 2. Populate scheduler/runtime metadata in the optimizer phase, not the mode

Decision:
- Compute scheduler/runtime metadata in `library/training/phases/optimizer.py` after the optimizer is built and stored on the trainer.

Rationale:
- The optimizer phase already owns the runtime optimizer object and scheduler creation.
- Modes should keep owning training-mode behavior, not scheduler-target inference for wrapped runtime objects.

Alternatives considered:
- Set scheduler metadata in `FineTuneMode`: rejected because the final optimizer runtime shape is better handled centrally after construction.

### 3. Preserve fallback behavior in `get_scheduler_fix(...)`

Decision:
- `get_scheduler_fix(...)` will accept optional plan metadata and use it when present.
- If no plan metadata is provided, it will continue using the current capability/heuristic resolution path.

Rationale:
- This keeps the change safe for legacy callers and compatibility paths.
- It also gives us a narrow migration step instead of forcing every scheduler call site to change at once.

Alternatives considered:
- Require plan metadata for all scheduler construction immediately: rejected because it would create unnecessary churn across legacy paths.

## Risks / Trade-offs

- [Risk] The metadata could duplicate logic already present in the scheduler builder. → Mitigation: keep one shared resolver function and have both the optimizer phase and the scheduler entrypoint use it as the fallback source.
- [Risk] This slice adds more plan metadata without changing visible behavior much. → Mitigation: keep it narrow and tie it directly to the next runtime-oriented slices where this metadata will matter more.
- [Risk] Wrapper/base-optimizer scheduling tests could become brittle if they over-specify internal routing. → Mitigation: add focused tests for the plan metadata seam and preserve existing behavior-driven wrapper tests.

## Migration Plan

1. Extend `OptimizationPlan` with scheduler/runtime metadata.
2. Add a shared scheduler-runtime resolver in `library/optimization/scheduler.py`.
3. Populate plan scheduler metadata in the optimizer phase before scheduler construction.
4. Update `get_scheduler_fix(...)` to use plan metadata when present and fall back otherwise.
5. Add focused unit coverage and record the change in `CHANGELOG.md`.

## Open Questions

- Should future runtime metadata also carry train/eval ownership hints for schedule-free wrappers, or should that remain a separate concern from scheduler routing?
- When adapter work resumes later, should it also populate scheduler/runtime metadata in the optimizer phase the same way as the base path, or will it need adapter-owned hints?
