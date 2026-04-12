## Context

The optimization layer now has two important pieces in place:

- trainer-facing logical groups via `OptimizationPlan`
- shared base-path grouping policy via `library/optimization/grouping.py`

What is still missing is an explicit execution-side model. The plan still stores execution payload as `parameter_groups`, which is effectively a compatibility-shaped list rather than a clearly named execution concept. That leaves the logical/execution split incomplete even though the architecture is already moving in that direction.

This slice is intentionally narrow. It is not about changing optimizer construction or wrapper behavior yet. It is about making execution groups explicit in the plan model so later runtime work has a cleaner foundation.

## Goals / Non-Goals

**Goals:**
- Make execution groups an explicit concept in `OptimizationPlan`.
- Preserve the existing optimizer-construction boundary by continuing to materialize legacy optimizer dicts where needed.
- Keep existing code working through compatibility accessors while moving new code to execution-oriented names.
- Update the shared fine-tune grouping helper and fine-tune mode to use the explicit execution-group terminology.

**Non-Goals:**
- Redesigning wrappers, offload behavior, or scheduler routing.
- Introducing fused execution groups or multi-optimizer orchestration.
- Reworking adapters/PEFT.
- Removing compatibility helpers like `ParameterGroup` in this slice if they still reduce churn.

## Decisions

### 1. Introduce explicit execution-group naming without forcing a hard rename everywhere

Decision:
- Add explicit execution-group-oriented types and fields to the optimization plan model.
- Preserve compatibility properties/accessors so existing callers and tests do not all need to change in one pass.

Rationale:
- The architectural value comes from making the execution concept explicit, not from maximizing churn.
- Compatibility accessors let this slice stay narrow and safe.

Alternatives considered:
- Hard-rename every `ParameterGroup`/`parameter_groups` use immediately: rejected because it adds broad churn without additional architectural value.
- Leave execution payload unnamed until a later runtime rewrite: rejected because that would keep the logical/execution split half-implemented.

### 2. Keep execution groups as plan-owned data objects, not just raw optimizer dicts

Decision:
- The execution side of the plan will remain typed and materializable, rather than being collapsed into raw dicts at grouping time.

Rationale:
- This preserves the same data-model discipline already applied to logical groups.
- Future runtime mapping work will need an explicit execution object to attach metadata to.

Alternatives considered:
- Materialize execution groups directly in the grouping helper: rejected because it would throw away the typed seam we are trying to strengthen.

### 3. Update the shared grouping helper before touching broader optimizer code

Decision:
- `library/optimization/grouping.py` and the fine-tune mode will be the first consumers of the explicit execution-group naming.
- The optimizer factory will continue accepting the materialized payload with no semantic change.

Rationale:
- This follows the same base-path-first sequencing as the earlier slices.
- It proves the execution model in the most stable path before broader runtime work.

Alternatives considered:
- Start in wrappers or scheduler code first: rejected because those layers should consume the cleaner model later, not define it now.

## Risks / Trade-offs

- [Risk] Compatibility accessors could blur the migration and leave mixed terminology around. → Mitigation: use explicit execution-group names in all new/updated code and limit compatibility shims to narrow bridging points.
- [Risk] The slice may feel abstract because runtime behavior does not change yet. → Mitigation: keep the change small and directly tied to the next runtime-oriented architectural work.
- [Risk] Renaming execution payload concepts could ripple into tests unexpectedly. → Mitigation: preserve compatibility properties and update only focused tests for the new naming in this slice.

## Migration Plan

1. Extend `library/optimization/types.py` with explicit execution-group terminology and compatibility accessors.
2. Update the shared grouping helper to return execution-group-oriented data.
3. Update `FineTuneMode` to build the plan through the execution-group field.
4. Add focused tests for the new execution-group naming and base-path integration.
5. Record the change in `CHANGELOG.md`.

## Open Questions

- When the runtime/wrapper slice arrives, should execution groups gain their own capability metadata directly, or should that remain separate from the group objects?
- Do we eventually want to retire the `ParameterGroup` name entirely, or keep it as a compatibility alias even after the broader execution model lands?
