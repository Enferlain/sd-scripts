## Context

The repo's current adapter path still carries compatibility-shaped assumptions
from older PEFT/Kohya-era surfaces. That has been acceptable while other
systems were being cleaned up, but it is now blocking follow-up work because
the adapter layer is still missing a clear repo-owned architecture.

Recent work in the repo has already moved optimization ownership toward a more
explicit model. The adapter system needs to fit that direction instead of
continuing to own targeting or optimizer-group behavior indirectly.

The current design direction settled in discussion is:

- strategy exposes targetable model structure
- optimization owns targeting policy for original-model effect surfaces
- the adapter system realizes adapter types against those resolved targets
- optimization groups and schedules whatever training/runtime information comes
  back
- `PeftMode` remains the training-side owner of the adapter path

The main design constraint is breadth. The architecture should not be locked to
current LyCORIS or built-in diffusion adapter shapes. Future adapter types may
have very different runtime behavior, persistence needs, or target surfaces.

This change is also not a compatibility-preserving cleanup. The current PEFT
adapter surface is not something the repo needs to preserve as-is if it blocks
the new architecture.

## Goals / Non-Goals

**Goals:**

- Define a repo-owned adapter architecture that fits the current optimization
  and orchestration direction
- Keep optimization as the owner of model-side targeting and grouping policy
- Keep `PeftMode` as the training-side owner for adapter training
- Define the adapter system as the realization layer between optimization and
  training
- Allow adapter-type-specific configuration through Hydra without forcing fake
  shared fields
- Preserve enough flexibility for future adapter types that are unlike current
  LoRA/LyCORIS examples
- Clarify save/load ownership for the adapter-training path
- Allow the new adapter architecture to replace compatibility-era PEFT surfaces
  rather than preserving them

**Non-Goals:**

- Finalize exact runtime type names or class names
- Finalize exact config field names for every adapter type
- Finalize exact save/load method names and signatures
- Define a full support matrix for every future adapter type up front
- Collapse adapter persistence into the fine-tune persistence model
- Preserve the current PEFT adapter runtime surface for compatibility's sake

## Decisions

### 1. Optimization owns model-side targeting and grouping policy

Optimization remains responsible for deciding what original-model parts are in
scope and how returned adapter state is grouped for training.

Rationale:

- This keeps model-side selection policy in one layer
- It prevents adapter types from re-owning grouping behavior through
  compatibility-era hooks like `prepare_optimizer_params(...)`
- It matches the direction already established in the repo's optimization work

Alternatives considered:

- Let each adapter type define its own final optimizer grouping behavior
  - Rejected because it mixes adapter-local behavior with optimizer policy
- Move targeting policy into the adapter layer
  - Rejected because it would make the adapter system own selector semantics
    and model-side policy

### 2. The adapter system is the realization layer

The adapter system takes resolved original-model targets and instantiates the
selected adapter type in that context.

Rationale:

- It gives the adapter system a clear job without making it the owner of the
  whole training flow
- It lets adapter types stay focused on their own behavior instead of model
  discovery or optimizer semantics
- It leaves room for future adapter types that are not shaped like current
  diffusion adapters

Alternatives considered:

- Treat the adapter layer as the owner of targeting, realization, and grouping
  as one large system
  - Rejected because the responsibilities become too blurred

### 3. `PeftMode` remains the training-side owner

`PeftMode` remains responsible for adapter-training orchestration.

Rationale:

- The repo already has a training-side owner for the adapter path
- Save/load and training lifecycle concerns belong to the adapter-training path,
  not to optimization
- This avoids inventing a second orchestration owner while the architecture is
  being clarified

Alternatives considered:

- Move training-side ownership out of `PeftMode`
  - Rejected for this design pass because the central issue is adapter-system
    architecture, not mode replacement

### 4. Adapter config should be explicit and adapter-type-specific when needed

The config direction should use Hydra to model real differences instead of
forcing every adapter type into one broad shared config surface.

Rationale:

- Shared fields should exist only where they are genuinely shared
- Hydra makes explicit per-type config practical
- This avoids encoding compatibility-era assumptions into the final config
  surface

Alternatives considered:

- One universal adapter config schema with many optional fields
  - Rejected because it encourages smuggling unrelated behavior through
    compatibility-shaped settings

### 5. Adapter persistence belongs to the adapter-training path

Adapter save/load behavior is an adapter-training concern and remains owned on
the training side by `PeftMode`, even if adapter-specific runtime behavior
participates in the persistence flow.

Rationale:

- Adapter training is not the same subject as full fine-tuning
- In adapter training, the model is a reference context and the learned object
  is the augmentation state
- The current code already follows this split in practice: `PeftMode` owns the
  training-side flow, while adapter objects provide adapter-specific
  `load_weights(...)` / `save_weights(...)` behavior
- Treating persistence this way avoids forcing adapter behavior into the
  fine-tune model-saving shape

Alternatives considered:

- Unify adapter and fine-tune persistence under one shared save model
  - Rejected because the learned subject is fundamentally different

### 6. The architecture should be broad by default

The design should not assume that current LyCORIS or built-in diffusion
adapters define what an adapter is.

Rationale:

- Future adapters may affect models through very different learned
  modifications
- Some future adapters may not resemble current module-local low-rank patterns
- The system should stay open to unusual cross-system or cross-model
  relationships as long as they fit the general adapter framework

Alternatives considered:

- Treat current diffusion adapters as the architectural template
  - Rejected because it would narrow the system too early

### 7. Resolved-target ownership should follow the fine-tune split

The adapter path should keep the same base ownership split as fine-tune:
strategy exposes targetable structure and optimization decides what is in
scope. The adapter path then adds a realization step in the middle.

Rationale:

- This avoids inventing a second ownership model just for adapters
- It keeps strategy as the source of model-family structure
- It keeps optimization as the owner of trainability policy
- It makes the adapter path a structured extension of the fine-tune path
  instead of a separate architecture

Alternatives considered:

- Move more resolved-target construction into the adapter system
  - Rejected because it would pull model-side policy away from optimization
- Treat adapters as needing a unique ownership split unrelated to fine-tune
  - Rejected because no current requirement justifies the extra complexity

### 8. Compatibility-era adapter hooks should be treated as replaceable

The existing PEFT adapter surface should be treated as transitional and
replaceable, not as a contract the new system must preserve.

Rationale:

- The adapter layer is currently the blocker rather than a stable foundation
- The repo does not need to preserve old PEFT behavior just because it exists
- Keeping compatibility as a requirement would constrain the architecture
  around a path that is already known to be the problem

Alternatives considered:

- Preserve the existing PEFT adapter hooks as long-term compatibility shims
  - Rejected because this change exists specifically to get out of that shape
- Incrementally wrap the old surface without allowing architectural breakage
  - Rejected because the old surface is not important enough to justify
    designing around it

### 9. The adapter package should start from a coherent runtime-first layout

The initial package layout should reflect the adapter architecture directly:
top-level registry and shared types, a runtime-oriented subpackage, per-adapter
type folders, and a shared area for code reused by more than one adapter type.

### 10. Optimization-to-adapter boundaries must stay contract-based

The optimization layer should depend on repo-owned adapter-facing contracts and
returned metadata, not on the internals of any concrete adapter method
implementation.

Rationale:

- Optimization owning policy does not mean optimization should know built-in
  adapter class names, ad hoc runtime attributes, or method-local naming quirks
- Hard-coding current LoRA/DyLoRA/OFT internals into optimization would turn
  policy code into a second adapter implementation layer
- A contract-based boundary keeps future adapter methods from forcing more
  special cases into optimization

Alternatives considered:

- Let optimization branch directly on current adapter implementation details
  - Rejected because it creates current-shape lock-in in exactly the layer that
    is supposed to stay broad

### 11. Grouping code should integrate as grouping, not as overflow

Grouping-related logic belongs in grouping code. The problem is not that the
grouping area may cover multiple grouping responsibilities. The problem is
adding behavior there that is not properly integrated as grouping logic, such
as config-loading concerns, compatibility shims that become de facto policy
surfaces, or direct dependence on adapter-method internals.

Rationale:

- Grouping is a real optimization concern, so related policy and construction
  logic should live where grouping is owned
- The failure mode is not "too much grouping in grouping"; it is unrelated or
  weakly related behavior getting parked there because it is nearby
- Fine-tune grouping and adapter grouping may both live under optimization
  ownership as long as their integration stays explicit and the boundaries stay
  clean

Alternatives considered:

- Treat any broad grouping module as suspicious by default
  - Rejected because module size is not the core issue
- Keep adding adjacent behavior to grouping even when it is really config
  parsing, compatibility translation, or adapter-specific integration glue
  - Rejected because it hides boundary problems inside an optimization-owned
    file

### 12. Production interfaces should stay typed and not bend around test fixtures

Typed production helpers should continue accepting the specific config/runtime
objects they are designed for. Tests should construct realistic typed inputs
instead of driving production code toward loose duck-typed access patterns.

Rationale:

- The repo's config direction is explicit dataclasses and typed helpers
- Loosening production interfaces to accommodate shortcut test fixtures hides
  real integration mistakes and weakens contracts
- The adapter rewrite needs stronger boundaries, not softer ones

Alternatives considered:

- Accept broad attribute-based objects in production helpers for convenience
  - Rejected because it trades away clarity and validation in a part of the
    repo that is still being actively shaped

### 13. `PeftMode` should remain orchestration-first

`PeftMode` owns the training-side adapter lifecycle, but its optimizer-build
path should remain explicit orchestration rather than hiding adapter-state
mutation inside optimizer preparation.

Rationale:

- Mode ownership is about coordinating the flow, not smuggling state changes
  into unrelated steps
- Hidden trainability mutation during optimizer construction makes lifecycle
  behavior harder to reason about and test
- The adapter path is already more complex than fine-tune; keeping the
  orchestration surface explicit matters

Alternatives considered:

- Allow optimizer-build helpers to silently mutate adapter trainability state
  as needed
  - Rejected because it blurs lifecycle boundaries inside the mode layer

### 14. Loaded-runtime merge should stay repo-owned and request-based

When adapter weights are loaded for rank discovery, merge, or inference-style
setup, the repo-owned boundary should stay centered on a loaded runtime plus a
repo-owned merge request rather than mirroring current built-in merge
signatures directly.

Rationale:

- `PeftMode` should only choose the flow and provide merge context
- Adapter methods should own how loaded adapter state merges into model
- The repo-owned seam should stay broad enough for future adapter methods that
  do not resemble today's built-in diffusion merge signatures

Alternatives considered:

- Let `PeftMode` call current built-in `merge_to(...)` signatures directly
  - Rejected because it leaks method-local merge semantics into training-side
    orchestration
- Define the repo-owned seam as a helper that forwards
  `text_encoder, denoiser, weights, dtype, device`
  - Rejected because it only re-expresses the current built-in LoRA merge
    shape under a new name instead of defining a broader runtime-owned
    contract

## Current Implementation Note

A recent implementation attempt explored some of this direction, but it was
reverted because the resulting shape did not match the intended architecture.

The design constraints above should therefore be read as active implementation
guardrails for the next pass, especially:

- optimization-owned grouping should not be pushed into compatibility-shaped
  config surfaces
- the adapter layer should not become the home for optimization logic
- grouping code may own grouping behavior, but not unrelated config parsing or
  adapter-specific internals that bypass repo-owned contracts
- typed production helpers should stay strict
- `PeftMode` lifecycle steps should remain explicit

Current working sketch:

```text
library/adapters/
  __init__.py

  types.py
  registry.py

  runtime/
    __init__.py
    build.py
    context.py
    targets.py
    persistence.py

  methods/
    __init__.py
    lora/
      __init__.py
      config.py
      runtime.py
      persistence.py
    oft/
      __init__.py
      config.py
      runtime.py
      persistence.py
    loha/
      ...
    lokr/
      ...

  shared/
    __init__.py
    base.py
    functional.py
    state_io.py
```

Rationale:

- It keeps each adapter type self-contained
- It gives the adapter system a real runtime-oriented home instead of letting
  everything collapse into top-level files
- It avoids splitting one adapter type across a separate top-level config
  folder
- It leaves a practical `shared/` area for code reused by more than one
  adapter type without falling back to a vague `utils/` dumping ground
- It lines up well with the useful lessons from vendored LyCORIS without
  copying LyCORIS's targeting/config entanglement

Alternatives considered:

- Put adapter-type-specific internal config files in a separate top-level
  `configs/` folder
  - Rejected because it weakens locality by splitting one adapter type across
    multiple top-level areas
- Use `utils/` instead of `shared/`
  - Rejected because `utils/` is too likely to become a junk drawer
- Keep all adapter types as flat top-level files
  - Rejected because the architecture is likely to outgrow that shape quickly

## Risks / Trade-offs

- [The architecture stays too abstract] -> Capture explicit requirements in the
  spec and follow up with concrete runtime concept decisions before
  implementation begins
- [Too much flexibility leads to a vague framework] -> Keep the ownership split
  firm even while exact runtime types stay open
- [Config fragmentation across adapter types] -> Use explicit per-type config
  only where it reflects real differences; share config only where it is truly
  common
- [Save/load remains inconsistent across adapter types] -> Keep `PeftMode` as
  the training-side owner and define the adapter-framework persistence surface
  before implementation
- [Future breadth weakens current practicality] -> Design for broad adapter
  shapes while still validating the architecture against current diffusion
  adapter needs
- [The rewrite removes old PEFT behavior too aggressively] -> Treat that as an
  acceptable trade-off unless a specific behavior is proven necessary for the
  new architecture
- [The package sketch is over-structured too early] -> Treat it as an initial
  working shape and collapse or move files if early implementation shows some
  parts are unnecessary

## Migration Plan

1. Use this change to lock the architecture and requirements before coding.
2. Identify the runtime concepts, package shape, and
   adapter-to-optimization handoff needed to support the new design.
3. Create the initial adapter package scaffolding around the agreed runtime
   layout before moving substantial behavior into it.
4. Rework the adapter path behind the current training entrypoint so `PeftMode`
   remains the training-side owner while the adapter system is rebuilt.
5. Replace compatibility-era PEFT adapter surfaces where they conflict with the
   new architecture instead of preserving them by default.
6. Treat the current `PeftConfig` and compatibility-shaped adapter interfaces as
   disposable transitional surfaces rather than the target architecture.

Rollback is straightforward at the proposal stage because this change defines
architecture and does not yet alter runtime behavior.

## Open Questions

- What are the final runtime concepts and names the framework should use?
- What exact runtime information must the adapter system expose so optimization
  can build the desired trainables, groups, and scheduling without pushing that
  work back into the adapter layer?
- What common persistence surface should `PeftMode` use to drive
  adapter-specific save/load behavior?
- Which old PEFT-facing hooks can be deleted immediately once the new adapter
  runtime path exists?
