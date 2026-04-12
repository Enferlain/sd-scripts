## Context

The optimization-layer plan now carries logical groups, execution groups, and scheduler/runtime metadata. The remaining runtime ownership leak is optimizer train/eval switching: schedule-free-like optimizers are currently handled through `optimizer_train_fn` / `optimizer_eval_fn` callback pairs that are derived from runtime detection and then called directly in training orchestration.

That works operationally, but it leaves the plan model incomplete. The orchestration layer still does not have an explicit optimizer-runtime concept for “this optimizer participates in train/eval mode switching.”

## Goals / Non-Goals

**Goals:**
- Add explicit optimizer runtime metadata for train/eval switching to `OptimizationPlan`.
- Populate that metadata for plan-aware optimizer paths.
- Add shared helper functions for switching optimizer runtime state between train and eval.
- Update orchestration call sites to prefer the shared helper path over raw callback invocation.

**Non-Goals:**
- Reworking the callback return shape from every legacy optimizer path.
- Changing actual schedule-free optimizer behavior.
- Redesigning wrappers or fused execution.
- Reworking adapters/PEFT.

## Decisions

### 1. Model optimizer train/eval ownership explicitly on the plan

Decision:
- Add a dedicated optimizer-runtime metadata object to the plan with a boolean for train/eval participation.

Rationale:
- The plan is becoming the shared orchestration model, so optimizer runtime mode ownership belongs there too.
- This is the minimum useful metadata for the current runtime seam.

Alternatives considered:
- Keep only callback pairs forever: rejected because that leaves runtime behavior implicit and harder to reason about.

### 2. Keep the metadata narrow for now

Decision:
- Start with a simple `supports_train_eval_toggle` flag instead of a richer runtime-state model.

Rationale:
- Current behavior only needs to know whether orchestration should call `optimizer.train()` / `optimizer.eval()`.
- A richer model can come later if fused or multi-optimizer runtimes need more structure.

Alternatives considered:
- Add a larger runtime ownership model now: rejected because it would over-design the current seam.

### 3. Use shared helper functions at orchestration points

Decision:
- Add shared helper functions that take the trainer/plan and apply optimizer runtime mode transitions.
- Update orchestration points to use those helpers.

Rationale:
- This removes duplicated “call the callback pair here” behavior from multiple training paths.
- It makes the plan-aware runtime route visible at the actual transition sites.

Alternatives considered:
- Leave call sites unchanged and only add metadata: rejected because the orchestration path would still be centered on raw callback pairs.

## Risks / Trade-offs

- [Risk] Keeping callback fields during migration could leave two sources of truth temporarily. → Mitigation: populate plan metadata from the same runtime detection logic and move orchestration call sites to the helper immediately.
- [Risk] Some wrappers may expose `train`/`eval` differently than expected. → Mitigation: keep helper behavior narrow and no-op safely when metadata says train/eval switching is not supported.
- [Risk] The slice may look small compared to the amount of metadata already added. → Mitigation: this is the remaining execution-runtime seam needed before larger runtime abstractions.

## Migration Plan

1. Extend `OptimizationPlan` with optimizer-runtime metadata for train/eval participation.
2. Add a shared resolver/helper in `library/optimization/optimizer_utils.py`.
3. Populate the metadata on plan-aware paths.
4. Update orchestration helpers and trainer/finalization call sites to use the shared runtime-mode helper.
5. Add focused tests and record the change in `CHANGELOG.md`.

## Open Questions

- Should future runtime metadata distinguish between “optimizer owns mode switching” and “wrapper proxies mode switching” or is that distinction unnecessary as long as orchestration has one correct call target?
- When callback compatibility is eventually removed, should plan-aware build results stop carrying `optimizer_train_fn` / `optimizer_eval_fn` entirely?
