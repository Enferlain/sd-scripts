## Context

The previous optimization-plan slice established `LogicalParameterGroup`, `OptimizationPlan`, and a plan-aware trainer seam for the active fine-tune path. That removed the biggest trainer-facing dependency on raw runtime `optimizer.param_groups`, but the actual logical-group assembly still lives inline in `FineTuneMode.build_optimizer_params(...)`.

This leaves grouping policy split across layers:

- `FineTuneMode` chooses what is trainable
- `FineTuneMode` also chooses how those trainables become logical groups
- `library/optimization/types.py` holds shared group objects but not shared grouping policy

That is better than the old tuple-only seam, but it still means the optimization layer has no single shared home for base-path grouping behavior. Since adapters are intentionally out of scope for now, this is a good moment to move base-model grouping into shared optimization-layer code without dragging the unstable PEFT seam into the design.

## Goals / Non-Goals

**Goals:**
- Move base-path logical-group assembly out of `FineTuneMode` into shared optimization-layer code.
- Keep group ordering, LR assignment, and metric naming behavior unchanged for the current fine-tune path.
- Preserve the separation between trainable selection and group construction.
- Leave the optimizer factory and scheduler behavior unchanged.
- Make the shared grouping code reusable for later base-path extensions without forcing a second architecture reset.

**Non-Goals:**
- Reworking adapters/PEFT or defining their long-term grouping contract.
- Introducing fused groups, multi-optimizer orchestration, or execution/runtime remapping.
- Moving optimizer construction out of the current optimizer phase/factory path.
- Reworking wrapper or offload scheduler-routing behavior in this slice.

## Decisions

### 1. Add a small shared grouping module under `library/optimization/`

Decision:
- Introduce a new shared module, `library/optimization/grouping.py`.
- This module will own base-path grouping helpers that convert already-selected trainables into `ParameterGroup` and `LogicalParameterGroup` objects.

Rationale:
- Grouping policy belongs closer to the optimization layer than to training modes.
- A dedicated module is clearer than continuing to grow `types.py` with both data models and assembly logic.

Alternatives considered:
- Keep the helper logic in `FineTuneMode`: rejected because it leaves grouping policy mode-local.
- Add builder methods onto `OptimizationPlan`: rejected because it mixes data containers with policy assembly.

### 2. Keep mode code responsible for trainable selection, not group construction

Decision:
- `FineTuneMode` will still decide whether the denoiser and which text encoders are trainable.
- Shared grouping helpers will receive explicit inputs such as modules, text-encoder flags, and learning-rate config, then return the grouped plan payload.

Rationale:
- Modes still own training-mode-specific selection and lifecycle behavior.
- This keeps the optimization layer responsible for “how selected things become groups” without forcing it to decide “what trains.”

Alternatives considered:
- Move both selection and grouping into the optimization layer: rejected because it would overreach into mode ownership and complicate future adapter rework.

### 3. Return a shared grouping result that plugs directly into the existing optimizer-plan seam

Decision:
- The grouping helper will return the execution `ParameterGroup` list and matching logical groups in deterministic order.
- `FineTuneMode` will continue to assemble the final `OptimizationPlan` and optimizer build result, but it will do so from the shared grouping helper output rather than building groups inline.

Rationale:
- This keeps the refactor narrow and non-breaking.
- The optimization plan seam already exists and does not need another shape change for this slice.

Alternatives considered:
- Introduce a second plan or selection object now: rejected because it adds more new surface than this slice needs.

### 4. Lock behavior with focused grouping tests instead of broader runtime changes

Decision:
- Add focused unit coverage for the new shared grouping helper and update existing fine-tune tests to assert that the helper-backed path preserves order and labels.

Rationale:
- This slice is about ownership and architecture, not runtime semantics.
- Tight helper tests are enough to keep the refactor honest without forcing larger optimizer/runtime changes.

Alternatives considered:
- Expand runtime integration tests in the same slice: rejected because it would mix architecture cleanup with slower, noisier coverage work.

## Risks / Trade-offs

- [Risk] The new grouping module could become a thin pass-through with no real architectural value. → Mitigation: move actual denoiser/text-encoder grouping logic there, not just wrappers around existing mode-local code.
- [Risk] The first shared helper could accidentally bake in fine-tune-specific assumptions that later make adapter work awkward. → Mitigation: keep the helper narrowly base-path oriented and explicit about inputs instead of pretending it is already universal.
- [Risk] This slice may feel modest because optimizer construction still happens in the mode. → Mitigation: treat it as a deliberate layering step; grouping ownership is being moved first, execution/runtime work comes later.

## Migration Plan

1. Add `library/optimization/grouping.py` with shared base-path grouping helpers.
2. Update `FineTuneMode.build_optimizer_params(...)` to use the shared helper instead of assembling groups inline.
3. Add/update focused unit tests for shared grouping behavior and fine-tune integration.
4. Update `CHANGELOG.md` to record the shared grouping-layer move.

## Open Questions

- Do we want the next slice after this to introduce a shared trainable-selection object, or is keeping selection mode-owned enough until adapters are redesigned?
- Should future scheduler-routing metadata live on `OptimizationPlan` directly, or remain derived from optimizer/wrapper capabilities until execution mapping exists?
