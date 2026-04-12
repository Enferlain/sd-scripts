## Context

The repo now has a largely repo-owned optimization surface: optimizer registrations, wrappers, scheduler registrations, and shared construction entrypoints all exist under `library/optimization/`. The remaining architectural problem is that trainer-facing code still infers optimization meaning from runtime `optimizer.param_groups`, which couples diagnostics, LR logging, and future grouping work to backend execution details.

The active fine-tune path already builds typed parameter groups, but those groups are immediately materialized back into legacy optimizer dicts before optimizer construction. The PEFT path still uses a compatibility-oriented adapter optimizer seam and is expected to be reworked later, so it should not define the architecture target for this change.

This change is cross-cutting because it touches optimization types, mode output contracts, trainer logging/diagnostics, and scheduler/runtime metadata without yet introducing a full execution-runtime rewrite.

## Goals / Non-Goals

**Goals:**
- Introduce a stable trainer-facing optimization-plan object for the active base fine-tune path.
- Represent logical parameter groups separately from execution/runtime optimizer groups.
- Make diagnostics and LR reporting consume logical-group metadata instead of reconstructing meaning from runtime optimizer groups.
- Preserve current optimizer and scheduler behavior while improving the seam between grouping, construction, and trainer consumption.
- Leave room for wrapper/offload/fused follow-up work without forcing a second architectural reset.

**Non-Goals:**
- Reworking the adapter/PEFT optimization seam in this change.
- Implementing fused optimizer groups, multi-optimizer orchestration, or a full execution runtime layer.
- Replacing optimizer-local runtime behavior with shared abstractions where the behavior is still optimizer-specific.
- Requiring repo-owned optimizers to stop using runtime `param_groups` internally.

## Decisions

### 1. Add a stable `OptimizationPlan` layer instead of expanding `lr_descriptions`

The trainer currently receives a 6-tuple from `build_optimizer_params(...)`, and the only stable trainer-facing group metadata is a list of strings in `lr_descriptions`. That is too weak for future grouping, diagnostics, or logical-to-runtime mapping.

Decision:
- Introduce a typed `OptimizationPlan` in `library/optimization/types.py`.
- The plan will carry:
  - logical groups
  - materialized execution groups for optimizer construction
  - trainer-facing LR/logging metadata
  - scheduler/runtime routing metadata where needed

Rationale:
- This upgrades an existing shared seam instead of creating a second orchestration path.
- The optimizer phase already centralizes optimizer creation, scheduler creation, and `accelerator.prepare()`, so it is the natural place to consume a plan object.

Alternatives considered:
- Keep the 6-tuple and append more parallel lists/dicts: rejected because it keeps group semantics fragmented.
- Make the trainer consume optimizer runtime groups directly: rejected because wrappers/offload/fused behavior will keep making runtime groups unstable as a user-facing model.

### 2. Treat logical groups as trainer-facing truth and execution groups as backend-private

Decision:
- Add a richer logical-group type that carries stable identifiers, labels, declared LR, parameter-count metadata, and optional group options.
- Preserve a separate materialized execution-group payload for optimizer construction.
- Keep execution-group rewrites backend-private unless a future runtime layer explicitly exposes a mapping.

Rationale:
- The trainer needs a stable object for diagnostics, logging, and future validation.
- Repo-owned optimizers and wrappers can continue to use runtime `param_groups` internally without leaking those details back to higher-level code.

Alternatives considered:
- Make logical groups and execution groups the same object: rejected because wrappers and future fused/offload layouts break that assumption.
- Normalize all wrappers to preserve 1:1 group identity: rejected because it would force backend constraints onto the architecture.

### 3. Make the base fine-tune path the architecture target for this phase

Decision:
- Implement the first pass against `FineTuneMode`.
- Keep the current PEFT path on a compatibility bridge until the adapter system is redesigned.

Rationale:
- The fine-tune path already emits repo-owned typed parameter groups and is the most stable place to define the intended architecture.
- Adapters are explicitly lower priority and expected to be reworked, so designing around them now would add the wrong constraints.

Alternatives considered:
- Force PEFT and fine-tune onto the same fully shared grouping seam immediately: rejected because the current adapter seam is not stable enough to be the architecture target.

### 4. Move trainer diagnostics and LR reporting onto plan metadata first

Decision:
- First migration step after adding the plan is to change startup diagnostics and step logging to use logical-group metadata instead of raw `optimizer.param_groups` and plain string `lr_descriptions`.

Rationale:
- This gives immediate architectural value without needing a larger runtime rewrite.
- It also creates tests that lock in the intended trainer-facing behavior before deeper runtime work begins.

Alternatives considered:
- Start with scheduler/runtime routing instead: rejected because the strongest current leak is in diagnostics/logging, and it is easier to harden without affecting optimizer execution semantics.

## Risks / Trade-offs

- [Risk] The trainer/mode contract changes from a simple tuple to a typed plan and may ripple through tests. → Mitigation: add the plan in a compatibility-friendly way, update focused orchestration tests first, and keep optimizer runtime tests unchanged.
- [Risk] Scheduler logging still depends on positional LR outputs from schedulers. → Mitigation: preserve a deterministic logical-group ordering and store per-group reporting metadata on the plan.
- [Risk] The first pass could accidentally start modeling wrapper execution details too early. → Mitigation: keep this phase limited to base-path logical grouping and trainer-facing metadata.
- [Risk] Leaving PEFT on a temporary bridge may create short-term asymmetry. → Mitigation: document that asymmetry as intentional; adapter redesign is a later integration point, not a blocker for the base optimization architecture.

## Migration Plan

1. Extend `library/optimization/types.py` with logical-group and optimization-plan types while preserving existing parameter-group materialization.
2. Update the base mode contract so fine-tune can return a plan object instead of fragmented tuple metadata.
3. Update the optimizer phase to consume the plan, build the optimizer/scheduler, and store plan metadata on the trainer.
4. Update diagnostics/LR logging helpers to prefer logical-group metadata from the plan.
5. Add focused unit coverage around plan construction, trainer diagnostics/logging metadata, and base-path scheduler behavior.
6. Leave PEFT on compatibility behavior for now; future adapter redesign can target the new plan seam directly.

## Open Questions

- How much scheduler-routing metadata belongs directly on `OptimizationPlan` versus remaining derived from optimizer registration capabilities?
- Should the first pass keep `trainer.lr_descriptions` as a compatibility alias over plan metadata, or remove it immediately and update call sites in one step?
- Do we want a separate `grouping.py` module in the first pass, or should the initial implementation stay inside `types.py` plus the fine-tune mode until the shared grouping layer grows beyond one path?
