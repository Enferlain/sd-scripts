## Context

Fine-tune and adapter training now both route optimizer construction through
`library/optimization/grouping.py`, but they reached that point through
different migration paths.

The fine-tune path already works from live base-model parameters:

- `FineTuneMode.prepare_trainables()` calls `resolve_finetune_selection(...)`
  to decide which existing base parameters are trainable.
- `_collect_component_named_parameters(...)` builds component-qualified
  parameter selector names such as `unet.foo.weight` and `clip_l.foo.weight`.
- `build_finetune_grouping(...)` turns selected base parameters into optimizer
  groups.

The adapter path now has a repo-owned runtime handoff, but its target identity
is still adapter-local under `library/adapters/runtime/targets.py`:

- `PeftMode.prepare_trainables()` calls `resolve_adapter_target_selection(...)`.
- Optimization expands selected components into module targets through
  `build_component_module_targets(...)`.
- Adapter runtimes realize adapter-owned trainable parameters from those module
  targets.
- `build_adapter_grouping(...)` groups the returned adapter trainable refs by
  component learning rate.

That split works mechanically, but it leaves no shared way to say that both
paths are selecting from the same model structure. It also means module type is
available only by inspecting live module objects during adapter target
construction, while fine-tune parameter grouping has no owner-module type
provenance at all.

## Goals / Non-Goals

**Goals:**

- Add a shared optimization-owned target reference model that can represent
  component, module, and parameter targets.
- Preserve existing fine-tune and adapter training behavior while enriching the
  provenance carried through each path.
- Expose module type for module targets and owner-module path/type for parameter
  targets.
- Let adapter module targets consume or wrap the shared target model instead of
  defining a separate long-term adapter-only target vocabulary.
- Give future module-type selection and LyCORIS-style adapter method work a
  stable foundation.

**Non-Goals:**

- Do not add a new user-facing selector DSL in this change.
- Do not make adapter grouping support arbitrary `optimizer.learning_rates.groups`
  yet.
- Do not merge all fine-tune and adapter grouping functions into one function in
  this pass.
- Do not move training-mode lifecycle sequencing into optimization.
- Do not make every future target kind or mixed-method overlap behavior part of
  this first target-reference foundation.

## Decisions

### 1. Add an optimization-owned target model alongside adapter runtime targets

Introduce a small optimization-owned target module, tentatively
`library/optimization/targets.py`, rather than expanding `grouping.py`.

This does not replace `library/adapters/runtime/targets.py` outright. The
adapter runtime target module can remain the adapter-facing compatibility and
runtime construction layer, but it should consume or wrap the shared
optimization target model instead of defining the long-term source of target
identity by itself.

The module should define a shared target reference shape similar to:

```python
OptimizationTargetKind = Literal["component", "module", "parameter"]

@dataclass(slots=True)
class OptimizationTargetRef:
    kind: OptimizationTargetKind
    component: str
    component_key: str
    path: str
    selector: str
    obj: Any
    module_type: str | None = None
    parent_module_path: str | None = None
    parent_module_type: str | None = None
    tags: frozenset[str] = field(default_factory=frozenset)
    metadata: dict[str, Any] = field(default_factory=dict)
```

Names can change during implementation if the final code reads better, but the
concepts should stay stable:

- `kind` says which level is being represented.
- `component` is the public component label.
- `component_key` is the internal stable component key used by training code.
- `path` is the component-local object path.
- `selector` is the public component-qualified selector.
- `obj` is the live object the mode or runtime needs.
- `module_type` applies to module targets.
- `parent_module_path` and `parent_module_type` apply to parameter targets.

Alternative considered: put the target reference dataclass in
`library/adapters/runtime/targets.py`. Rejected because fine-tune parameter
targets need the same provenance model, and the shared target vocabulary should
not be owned by the adapter runtime package.

Alternative considered: put the target reference dataclass in `grouping.py`.
Rejected because grouping already owns group construction and LR matching. The
target vocabulary will be used by selection, adapter target construction,
inspection-adjacent behavior, and grouping.

### 2. Keep mode sequencing, but unify target provenance

Fine-tune and adapter training have different timing:

- Fine-tune trainables already exist as base-model parameters.
- Adapter trainables are created only after adapter runtimes consume resolved
  module targets.

This change should not hide that lifecycle difference. `FineTuneMode` and
`PeftMode` should continue to orchestrate their own order of operations.

What should converge is the provenance model:

- fine-tune parameter refs should carry `OptimizationTargetRef(kind="parameter")`
- adapter module targets should carry `OptimizationTargetRef(kind="module")`
- adapter trainable refs should preserve source target provenance when they are
  emitted back to optimization

Alternative considered: make `PeftMode` own adapter module targeting because
adapter methods are module-built. Rejected because targeting policy still
belongs in optimization; the mode owns only when the selection/build/grouping
steps happen.

### 3. Preserve current behavior first

The first implementation should be intentionally boring from the user's point
of view:

- fine-tune `optimizer.learning_rates.groups[*].match` still matches existing
  component-qualified parameter selector strings
- adapter target selection still derives selected components from learning-rate
  policy
- adapter grouping still groups returned trainable refs by component LR
- existing tests for order, labels, and learning-rate behavior should continue
  to pass

The important change is that the internal objects now carry enough shared
provenance to support later module-type selection.

Alternative considered: add `match_module_type` or a broader target-selector
config immediately. Rejected because the first step should make module and
parameter targets explicit before standardizing new user-facing syntax.

### 4. Treat module type as a stable target fact, not an adapter-only detail

Adapter module targets already have live modules, and LyCORIS-style methods
commonly branch on module type. The shared target model should expose
`module_type` as a normalized class-name string such as `Linear`, `Conv2d`, or
`LayerNorm`.

Fine-tune parameter targets should expose their owner module path and type, so
later grouping policy can target parameters by the module that owns them
without reconstructing that relationship from names.

Alternative considered: keep module type only in `AdapterResolvedTarget.metadata`.
Rejected because fine-tune also needs owner-module provenance, and storing the
fact only in adapter metadata would keep the two paths conceptually split.

## Risks / Trade-offs

- [Risk] Target refs become a second naming system that drifts from inspection
  selectors. -> Mitigation: build selectors with the same
  `build_selector_name(...)` helper and test that parameter target selectors
  match the component-qualified selector surface.
- [Risk] Fine-tune parameter owner-module discovery may add overhead. ->
  Mitigation: compute it while collecting named parameters and keep the first
  pass simple; this happens during setup, not per training step.
- [Risk] Adapter targets and trainable refs may duplicate provenance fields
  during migration. -> Mitigation: allow compatibility properties or mirrored
  fields temporarily, but make the shared target ref the authoritative source
  for new behavior.
- [Risk] This may invite selector-DSL expansion before the target model settles.
  -> Mitigation: keep user-facing module-type matching out of scope for this
  change and require a follow-up spec for new config syntax.

## Migration Plan

1. Add `library/optimization/targets.py` with shared target-ref types and helper
   constructors.
2. Update adapter module target construction to build shared module target refs
   while preserving the existing adapter target fields used by runtimes.
3. Update fine-tune parameter collection to carry shared parameter target refs,
   including owner-module path/type.
4. Update adapter trainable refs and repo-owned adapter runtimes to preserve
   source target provenance.
5. Keep existing grouping behavior unchanged, then add focused assertions for
   module type and source target provenance.
6. Update docs and changelog once implementation lands.

Rollback is straightforward because this change should not alter user-facing
config semantics. Reverting the target-ref plumbing should restore the previous
component/name-based behavior.

## Open Questions

- Should live object storage be named `obj`, `module_or_param`, or split into
  separate optional `module` / `parameter` fields?
- Should `AdapterResolvedTarget` in `library/adapters/runtime/targets.py` wrap a
  shared target ref, subclass it, or keep compatibility fields plus a
  `target_ref` field for the first pass?
- Should parameter target owner-module discovery choose the deepest owning
  module or the nearest non-container module when containers are involved?
