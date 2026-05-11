## Context

The active strategy/trainer path still treats a loaded training model as a fixed tuple:

- `model_version`
- `text_encoder` or `text_encoders`
- `vae`
- `denoiser`

That tuple is defined at the strategy loading contract layer and then unpacked directly into trainer state. From there, multiple subsystems rebuild their own view of "model components" from the same assumption:

- startup diagnostics and resource accounting
- adapter runtime context and target resolution
- optimizer grouping and target refs
- component-qualified selector generation
- model inspection / dump tooling

Recent work moved shared public component naming out of the parameter-dump module and into `library.models`, which improved ownership, but it also made the deeper issue easier to see: the repo still assumes the SD/SDXL/SD3-shaped tuple is the universal top-level model contract.

The next broader topic in this area is metadata. If metadata evolves on top of the current tuple, the repo will deepen the same assumption again. This change is therefore meant to define the actual top-level contract before more systems build on it.

## Goals / Non-Goals

**Goals:**

- Replace the fixed `text_encoders/vae/denoiser` loader tuple with a family-declared loaded-component contract.
- Make top-level components the primary generic control surface for trainer/runtime code.
- Ensure generic code consumes component semantics through declared roles/capabilities rather than through hardcoded family names or SD-shaped slot assumptions.
- Allow multiple components to share the same role without collapsing them into synthetic universal slots.
- Move diagnostics, resource accounting, targeting, grouping, selectors, and tooling onto the same family-declared component surface.
- Make the transition breaking and explicit instead of carrying dual contracts forward.

**Non-Goals:**

- Implementing the full metadata redesign itself.
- Defining optimizer policy below the component boundary; module- and parameter-level expansion remain lower-level optimization concerns.
- Designing a universal inference-pipeline API for every future architecture.
- Solving every future model-family difference up front; the goal is to remove the current narrowing assumption, not to freeze the final shape of every future family.

## Decisions

### Decision: Introduce a family-declared loaded-component contract

The repo will define a new top-level loaded-component surface owned by model families rather than by the trainer or by one specific training mode.

Each loaded component will be family-declared and ordered by the family implementation. At minimum, a declared component needs:

- a stable component key
- a public display / selector label
- the live loaded module object
- explicit roles and/or capabilities for generic consumers

This is preferred over the current fixed tuple because it lets families describe their real top-level structure without having to pretend they fit `text_encoders + vae + denoiser`.

Alternative considered:

- Keep the tuple and attach richer metadata beside it.
  Rejected because it preserves the wrong primary contract and would keep forcing new consumers to decide whether the tuple or the metadata is authoritative.

### Decision: Make the loaded-component collection the primary trainer/runtime representation

Trainer/runtime code will move toward one primary loaded-component collection instead of treating `text_encoders`, `vae`, and `denoiser` as the root representation.

Generic code should consume the loaded-component collection directly or through helper queries over that collection. The repo should not preserve the old trio as the long-term source of truth solely because current diffusion families already fit it.

Alternative considered:

- Keep the trio as compatibility convenience attributes while adding a generic collection underneath.
  Rejected because the user explicitly wants a proper transition rather than reinforcing the old assumption through a dual contract.

### Decision: Generic behavior uses declared semantics, not component-name branches

Generic repo code must not regain family lock-in by switching from `text_encoder1` to `clip_l` branches. When generic behavior needs semantics, it should consume declared component roles/capabilities rather than hardcoded component-name checks.

This also allows multiple components to share the same role. For example, a family may expose several conditioning towers without forcing the generic repo contract to rename them into synthetic slot names.

Alternative considered:

- Treat family-defined component keys alone as enough semantic information.
  Rejected because generic code would quickly drift back into string comparisons and family-specific branching.

### Decision: Component level remains the highest generic control surface

The new contract is about top-level model components, not arbitrary internal submodules and not direct optimizer-parameter policy.

Families declare top-level components.
Generic trainer/runtime code reasons at the component level.
Optimization and adapter layers may still expand those components into module targets or parameter refs, but that lower-level expansion remains a separate concern.

Alternative considered:

- Jump directly to module-level generic ownership.
  Rejected because it would move the primary abstraction below the level the user considers useful, and it would blur the boundary between component architecture and optimizer policy.

### Decision: Family declarations own ordering and top-level participation

Families will own the order of their loaded components. That order becomes the default for:

- human-facing startup diagnostics
- resource-accounting tables
- inspection tool dumps
- any other generic presentation that needs a stable top-level component order

Families will also own which top-level components participate in generic behaviors such as reporting, metadata, or targeting, whether through explicit capabilities or equivalent family-declared metadata.

Alternative considered:

- Let each consumer impose its own component order and participation rules.
  Rejected because it would recreate multiple inconsistent model-component views across logging, tooling, optimization, and metadata.

### Decision: Transition all affected consumers instead of isolating the change to metadata

This change must cover the real shared boundary, not just metadata plumbing. The change therefore includes the loader/trainer contract plus the immediate consumers that currently depend on the fixed tuple:

- training observability
- resource monitoring
- selector generation
- adapter targeting
- optimization target refs/grouping
- model inspection tooling

Alternative considered:

- Limit the change to metadata/reporting only and leave the loader/trainer tuple alone.
  Rejected because that would postpone the real boundary problem and force metadata to depend on a contract already known to be too narrow.

## Risks / Trade-offs

- [Broad migration surface] -> Mitigation: Land the contract change as one explicit architecture transition with staged task slices, rather than letting multiple partial compatibility layers coexist.
- [Under-specified component semantics] -> Mitigation: Keep the initial role/capability vocabulary intentionally small and extend it only when a concrete family or consumer requires more.
- [Families may diverge in declaration quality] -> Mitigation: Require focused family-level tests for declared component order, selector prefixes, and role/capability participation.
- [Generic code may drift back into component-name checks] -> Mitigation: Make the design and spec text explicit that semantic behavior must come from declared roles/capabilities, not from hardcoded family-native names.
- [Metadata follow-up could try to redefine the same contract again] -> Mitigation: Make the loaded-component declaration the shared source of top-level component identity so metadata work builds on it rather than replacing it.

## Migration Plan

1. Define the new loaded-component contract and family declaration requirements.
2. Update the model-loading strategy contract to return the loaded-component surface instead of the old tuple.
3. Update trainer setup/state to treat loaded components as the primary model representation.
4. Migrate diagnostics and resource-monitor inputs to consume loaded components.
5. Migrate selector generation, target refs, optimizer grouping, and adapter target resolution to consume declared components.
6. Migrate model-inspection tooling to the new surface.
7. Remove the old tuple-oriented assumptions and related tests once all affected consumers have moved.

Rollback is repo-internal and code-level: if the migration proves too disruptive, the branch can be reverted before archive/merge. No separate production rollout or user data migration is needed.

## Open Questions

- What is the smallest initial role/capability vocabulary that still keeps generic code out of name-based branching?
- Should the loaded-component contract distinguish between public display labels and public selector prefixes, or should those remain one field unless a concrete family proves they need to differ?
- Which current consumer should become the first implementation slice after the contract lands: trainer state, observability, or optimization targeting?
