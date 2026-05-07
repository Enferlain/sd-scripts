# Optimization Layer Plan

Date: 2026-04-01

## Purpose

This note tracks the intended future direction for the optimizer / scheduler /
parameter-grouping layer after the package rename toward `library/optimization/`.

The goal is **not** to implement every desired capability in one pass.

The goal is to establish a layer shape that can support future additions
without repeated architecture rework.

## Desired Future Capabilities

- fused optimizer groups / fused execution paths
- selective learning rates for major components
- arbitrary selections, exclusions, and groupings of modules / layers
- support for both base-model and adapter-owned parameters
- easy addition of new optimizers and schedulers
- support for numeric/runtime features such as stochastic rounding, Kahan
  summation, and similar future experiments
- intuitive end-to-end flow
- torchao code needs to be vendored so we can freely modify it

## Current Reading

The active code already has a decent high-level split:

- training phases own ordering
- modes own divergent trainable/grouping behavior
- shared factory code owns optimizer instantiation
- shared scheduler code owns LR scheduler creation
- the training loop owns the normal optimizer / scheduler step path
- objective runtime owns diffusion/noise scheduler behavior separately
- EDM2 already acts like a second sidecar optimization runtime

The newer grouping and adapter work also now has a shared target-provenance
foundation:

- `library/optimization/targets.py` owns shared component/module/parameter
  target refs
- fine-tune parameter selection now carries selector-stable parameter target
  refs plus owner-module provenance
- adapter module target construction now wraps shared module target refs
  instead of acting like a separate long-term target vocabulary

This matters because future module-type selectors, adapter-specific grouping
extensions, and inspection-adjacent tooling should build on that shared target
model instead of inventing a second selector surface.

This is a useful base, but the current layer still mixes:

- parameter grouping policy
- adapter-specific optimizer preparation
- optimizer/vendor integration
- scheduler creation
- special optimizer behavior and compatibility quirks

## Vendor Absorption Audit

The vendored `LoraEasyCustomOptimizer` tree is useful as an algorithm source,
but it is also a pressure test for the optimization layer because it contains
multiple integration shapes, not just plain optimizer classes.

### Observed Integration Categories

#### 1. Plain optimizer classes

These mostly look like normal `Optimizer` / `BaseOptimizer` subclasses with
extra constructor options.

Examples:

- `AdaBelief`
- `Adan`
- `LaProp`
- `RMSProp`
- `ScalableShampoo`
- `SOAP`
- `GaLore`

Likely absorption path:

- repo-local optimizer registration
- normalized config mapping
- optional capability metadata

#### 2. Wrapper optimizers

These wrap another optimizer class rather than acting as a plain leaf
optimizer.

Examples:

- `ScheduleFreeWrapper`
- `SNOO_ASGD`

Observed characteristics:

- require a `base_optimizer` / `base_optimizer_type`
- own extra train/eval or averaging behavior
- expose `base_optimizer` for scheduler interaction

Implication:

- wrappers must stay a first-class integration kind
- they should not be flattened into plain optimizer registration

#### 3. Schedule-free leaf optimizers

These are leaf optimizers with schedule-free behavior integrated directly into
the optimizer implementation.

Examples:

- `ADOPTScheduleFree`
- `FADOPTScheduleFree`
- `ADOPTEMAMixScheduleFree`
- `ADOPTNesterovScheduleFree`
- `ADOPTMARSScheduleFree`
- `ADOPTAOScheduleFree`
- `FADOPTEMAMixScheduleFree`
- `FADOPTNesterovScheduleFree`
- `FADOPTMARSScheduleFree`
- `ProdigyPlusScheduleFree`

Implication:

- the layer must continue to distinguish
  - schedule-free leaf optimizers
  - schedule-free wrappers
- train/eval capability metadata remains important

#### 4. Low-bit / quantized optimizer families

The vendor tree includes both wrapper-style and direct low-bit implementations.

Examples:

- `AdamW8bitAO`
- `AdamW4bitAO`
- `AdamWfp8AO`
- `low_bit_optim.*`

Observed characteristics:

- custom state tensor formats
- stochastic rounding options
- nonstandard block-size / quantization arguments

Implication:

- these should be modeled as optimizer-specific augmentations or absorbed
  low-bit families, not as generic numeric features by default

#### 5. Optimizer-specific numeric augmentations

Some variants modify optimizer internals directly rather than just wrapping
step orchestration.

Examples:

- `AdamW8bitKahan`
- multiple bf16 stochastic-rounding variants
- optimizer-local clipping / stabilization variants

Implication:

- this reinforces the current rule that numeric/runtime extras should begin as
  optimizer-specific integration, then only move into shared hooks if they
  prove genuinely generic

#### 6. Offload / runtime wrappers

Some vendor components act more like runtime/execution helpers than normal
optimizers.

Examples:

- `CPUOffloadOptimizer`

Observed characteristics:

- wraps one or more inner optimizers
- changes where optimizer state lives
- affects step timing and memory behavior

Implication:

- offload wrappers are closer to execution/runtime integration than plain
  instantiation
- the layer should keep room for `offload_wrapper` or similar integration
  kinds

#### 7. Optimizers with internal scheduler-like behavior

Some optimizers or wrappers carry their own warmup / cosine-decay helpers
internally.

Examples:

- `ScheduleFreeWrapper`
- `Compass*`
- `StableSPAM`
- `SCION`

Implication:

- not every learning-rate-like knob should become a top-level scheduler
- some “scheduler-ish” vendor settings are really optimizer-local behavior and
  should stay there during absorption

### Absorption Guidance

The vendor tree should not be treated as a single package to preserve.

Instead, absorption should happen by category:

- absorb plain optimizers into repo-owned registrations
- absorb wrappers as explicit wrapper/offload integration kinds
- absorb low-bit families only when we are ready to own their state/runtime
  implications
- absorb optimizer-specific numeric tricks as local optimizer variants first
- avoid promoting optimizer-internal warmup/decay behavior into the shared
  scheduler layer unless it is truly generic

### Planning Implication

This audit suggests a good next sequencing:

1. continue improving optimizer/scheduler integration modeling
2. add a small vendor absorption matrix
3. only then start the shared grouping layer

That order reduces the chance that grouping work is designed around an
incomplete understanding of the optimizer/adaptation surface we eventually want
to host.

## Initial Absorption Matrix

This matrix is a first sorting pass, not a commitment to absorb every item.

Statuses:

- `now`: good early absorption target
- `later`: valuable, but should wait until the layer grows a bit more
- `maybe never`: only worth it if a concrete need appears

| Candidate | Kind | Notable Pressure / Capability | Likely Dependencies | Suggested Wave |
| --------- | ---- | ----------------------------- | ------------------- | -------------- |
| `ScheduleFreeWrapper` | wrapper | base-optimizer wrapper, train/eval toggling, scheduler-on-base-optimizer behavior | vendor wrapper code | now |
| `SNOO_ASGD` | wrapper | alternate wrapper shape, optimizer state wrapping, scheduler/base-optimizer interaction | vendor wrapper code | now |
| `AdamW8bitKahan` | optimizer augmentation | optimizer-local Kahan + low-bit behavior, tests numeric-augmentation modeling | `bitsandbytes` | now |
| `Adan` | optimizer | plain absorbed optimizer with richer defaults than core torch path | `pytorch_optimizer`-style donor code | now |
| `AdaBelief` | optimizer | plain optimizer absorption case, lower architectural pressure than heavy families | `pytorch_optimizer`-style donor code | now |
| `LaProp` | optimizer | plain optimizer with centered/AMSBound state and stochastic packing behavior | `pytorch_optimizer`-style donor code | now |
| `ADOPT` | optimizer | plain optimizer with update-strategy policy and clipped normalized updates | `pytorch_optimizer`-style donor code | now |
| `ADOPTScheduleFree` | schedule-free optimizer | repo-owned leaf schedule-free optimizer with train/eval toggling and no external scheduler | repo-local schedule-free leaf | now |
| `LPFAdamW` | optimizer | plain optimizer with three-beta state and low-pass first-moment smoothing | vendor implementation | now |
| `SGDSaI` | optimizer | SGD-family optimizer with warmup-derived GNSR scaling and optional cautious updates | vendor implementation | now |
| `Adai` | optimizer | adaptive-momentum optimizer with dampening and optional gradient centralization | `pytorch_optimizer`-style donor code | now |
| `VSGD` | optimizer | variational SGD-style optimizer with probabilistic state updates and stochastic pack support | `pytorch_optimizer`-style donor code | now |
| `RACS` | optimizer | row/column-scaled SGD with Adam-style fallback for unsupported tensor ranks | vendor implementation | now |
| `CosineAnnealingWarmRestarts` / `RexAnnealingWarmRestarts` | scheduler | true standalone scheduler classes with param-group resume state and custom restart math | repo-local scheduler classes | now |
| `ProdigyPlusScheduleFree` | schedule-free optimizer | leaf schedule-free optimizer with external dependency surface | `prodigyplus` | later |
| `AdamW8bitAO` / `AdamW4bitAO` / `AdamWfp8AO` | optimizer / low-bit family | quantized state formats, nonstandard state/runtime implications | `torchao` + vendor low-bit code | later |
| `CPUOffloadOptimizer` | offload wrapper | runtime/memory/offload behavior, not just plain construction | `torchao` | later |
| `GaLore` | optimizer | likely interacts with grouping/selection because of projected low-rank behavior | donor utils + projector helpers | later |
| `ScalableShampoo` | optimizer | heavier state and helper coupling, more complex config surface | donor shampoo helpers | later |
| `SOAP` | optimizer | advanced optimizer with helper coupling and preconditioner assumptions | donor shampoo/soap helpers | later |
| `Ranger21` | optimizer | optimizer with internal scheduler-like behavior and extra policy assumptions | donor implementation | later |
| `Compass*` family | mixed family | many variants, clipping/stability/warmup behaviors, likely needs its own sub-plan | donor utils + optional bnb/ao hooks | later |
| `StableSPAM` | optimizer | optimizer-local warmup/decay and clipping behavior, more integration pressure | donor utils | later |
| `SCION` / `SCORN` / `SCORNMachina` | optimizer family | heavy internal norms/scheduling-like behavior, likely noisy first absorption target | donor norm helpers | later |
| `GOODDOG`, `TALON`, `WiwiOpt`, `Mythical`, `Alice`, `RACS`, `Glyph` | optimizer | probably plain-ish candidates, but lower priority until we need them | donor implementation | maybe never |
| `ProjectiveAdam` | optimizer | interesting but niche; better after the absorption path is proven | donor implementation | maybe never |

### First-Wave Recommendation

If we want the smallest useful first absorption set, it should probably be:

- `ScheduleFreeWrapper`
- `SNOO_ASGD`
- `AdamW8bitKahan`
- one plain optimizer: `Adan` or `AdaBelief`
- a second plain-optimizer pair: `LaProp` / `ADOPT`
- one repo-owned schedule-free leaf: `ADOPTScheduleFree`
- another plain-optimizer pair: `LPFAdamW` / `SGDSaI`
- another adaptive plain-optimizer pair: `Adai` / `VSGD`
- another row/column-scaled plain optimizer: `RACS`
- one true standalone scheduler pair:
  `CosineAnnealingWarmRestarts` / `RexAnnealingWarmRestarts`

That combination would pressure-test:

- wrapper integration
- scheduler-on-base-optimizer behavior
- optimizer-local numeric augmentation
- plain absorbed optimizer registration

without immediately dragging in the heaviest families.

### Current Progress

Wrapper absorption now has a first explicit implementation seam in the active
code:

- registered wrappers can declare a wrapper-construction style explicitly
- `ScheduleFreeWrapper` and `SNOO_ASGD` are now modeled as wrappers around a
  built base optimizer
- wrapper configs can pass namespaced base-optimizer options via
  `base_optimizer.*`
- scheduler routing onto the wrapped base optimizer is now covered by tests
- `AdaBelief` and `Adan` are now repo-owned absorbed plain optimizers, which
  means the registration path can host both simple and richer plain optimizers
  without routing them through arbitrary fallback imports
- `AdamW8bitKahan` is now the first optimizer-local augmentation on the shared
  registration path, which means repo-owned optimizer variants can still
  declare third-party backend dependencies without falling back to the old
  arbitrary optimizer path
- absorbed optimizer implementations now live under
  `library/optimization/optimizers/`, which keeps the package root focused on
  orchestration rather than implementation sprawl
- `CosineAnnealingWarmRestarts` and `RexAnnealingWarmRestarts` are now
  repo-owned absorbed schedulers under `library/optimization/schedulers/`,
  which means standalone scheduler classes no longer need to live only behind
  `lr_scheduler_type` custom imports
- the custom-optimizer smoke config now uses the repo-facing
  `RexAnnealingWarmRestarts` scheduler name directly instead of a vendor module
  path in `lr_scheduler_type`
- `LaProp` and `ADOPT` now live under `library/optimization/optimizers/` as
  additional repo-owned plain optimizers, which means the current registration
  path has now absorbed multiple optimizer styles without adding new factory
  special cases
- `ADOPTScheduleFree` is now the first repo-owned schedule-free leaf on the
  shared registration path, which means the layer has now pressure-tested a
  distinct integration shape beyond plain optimizers, wrappers, and
  optimizer-local augmentations
- `LPFAdamW` and `SGDSaI` are now also repo-owned plain optimizers under
  `library/optimization/optimizers/`, which broadens the absorbed plain
  optimizer surface without introducing new factory-specific construction code
- `Adai` and `VSGD` are now also repo-owned optimizers on the shared
  registration path, and the absorbed `VSGD` path now owns its explicit
  `stochastic_fp` default instead of depending on a donor-side implicit field
- `RACS` is now also a repo-owned optimizer on the shared registration path,
  and the small shared optimizer helpers now live under
  `library/optimization/optimizers/utils/` instead of staying inline or at the
  package root
- `Alice` and `Lamb` are now also repo-owned optimizers on the shared
  registration path, which means the absorbed plain-optimizer surface now spans
  both subspace-style and large-batch adaptive update variants without adding
  new factory-specific construction code
- the `adopt.py` family now lives together under the repo-owned path, with
  `ADOPT`, `ADOPTMARS`, and `FADOPTMARS` all absorbed and sharing the same
  repo-owned helper layer for clipping and adaptive-epsilon behavior
- absorbed optimizer coverage has started moving out of the large
  training-level optimizer test into `tests/unit/optimizers/`, which keeps the
  shared orchestration coverage separate from absorbed implementation smoke
  coverage as the surface grows

This does not solve the broader absorbed-wrapper design permanently, but it
does remove the old assumption that all wrappers can be treated like plain
optimizer constructors with one extra class argument.

## Target Responsibilities

The future layer should separate five concerns clearly.

### 1. Selection

Responsible for deciding which parameters are eligible for training at all.

Examples:

- denoiser-only
- specific text encoders
- adapter params only
- future include / exclude selections

### 2. Grouping

Responsible for partitioning selected parameters into named groups with
per-group overrides.

Examples:

- denoiser LR vs text encoder LR
- per-text-encoder learning rates
- block-level grouping
- regex / path / module-type grouping
- adapter-vs-base grouping
- exclusions / zero-LR groups / disabled groups

The grouping layer should own the stable, trainer-facing *logical* group
descriptions. Optimizers and wrappers may still rewrite those groups into
different *execution* groups internally (for example per-parameter, offloaded,
or fused layouts), but that internal execution shape should not leak back into
logging, LR description indexing, or scheduler/reporting assumptions.

### 3. Instantiation

Responsible for constructing optimizers and LR schedulers from config plus
grouped parameters.

Examples:

- built-in torch / transformers / diffusers choices
- custom repo-local optimizers or schedulers
- external optimizers / schedulers via qualified import path

### 4. Execution

Responsible for runtime stepping behavior.

Examples:

- `step()`
- `zero_grad()`
- schedule-free train/eval switching
- multi-optimizer orchestration
- fused optimizer groups

### 5. Numeric / Runtime Features

Responsible for optional enhancements that affect precision or stepping
behavior, but do not belong to grouping policy.

The default assumption should be that these begin life as optional optimizer
augmentations. If a capability requires changing optimizer internals or state
handling, it should be modeled as optimizer-specific integration first.

Only capabilities that remain genuinely optimizer-agnostic in both interface
and implementation should be promoted into shared runtime/feature hooks.

Examples:

- stochastic rounding
- Kahan summation
- fused Adafactor patching
- orthograd-like extras
- future numeric/runtime experiments

## Design Rules

- Modes and adapters should not own scheduler construction.
- Factories should not decide parameter grouping policy.
- Grouping should be able to describe both base-model and adapter-owned params.
- Advanced execution behavior should live in runtime/execution code, not in
  factories or grouping.
- Numeric tricks should be explicit capabilities, not hidden special cases.
- Optimizer-specific numeric tricks should stay close to optimizer integration
  unless they prove reusable across multiple optimizers without special-case
  logic.
- Config should express user intent; runtime objects should express execution
  state.
- Logging and diagnostics should receive stable group descriptions from the
  grouping layer instead of reconstructing them heuristically.
- Execution-time optimizer group rewrites should be treated as backend-private
  implementation details unless the grouping layer explicitly exposes a mapped
  logical-to-execution relationship.

## Proposed Package Direction

After the rename, `library/optimization/` is the intended long-term home for
this layer.

Likely module directions:

- `selection.py`
- `grouping.py`
- `optimizer_factory.py`
- `scheduler_factory.py`
- `runtime.py`
- `features.py`
- `types.py` or `models.py`

These names are directional, not final API commitments.

## Migration Plan

### Phase 1: Foundation

Goal: normalize the current layer without changing behavior.

Planned work:

- move current optimizer / scheduler code under the renamed package
- centralize duplicated optimizer-arg parsing
- centralize duplicated scheduler-arg parsing where practical
- introduce a shared parameter-group payload shape
- make fine-tune and PEFT both emit the same group payload shape
- keep the current phase flow and training-loop behavior intact

Success criteria:

- one shared group payload shape
- one shared optimizer construction path
- one shared scheduler construction path
- no intentional behavior changes yet

### Phase 2: Real Grouping Layer

Goal: make grouping a first-class shared concern instead of mode-local glue.

Planned work:

- add explicit selection and grouping helpers
- support named groups and group metadata
- support per-component and per-TE learning rates through the shared grouping
  layer
- support future arbitrary include / exclude / matching rules
- move fine-tune’s direct manual group building into shared grouping code
- add adapter-facing bridges so adapters can either:
  - emit ready-made groups
  - or emit trainables plus grouping hints

Success criteria:

- grouping policy has one clear home
- future block/pattern grouping has a natural extension point
- validation can eventually move back on for grouping-related config

### Phase 3: Registry / Integration Layer

Goal: make adding new optimizers and schedulers straightforward.

Planned work:

- reduce branch-heavy construction logic in favor of normalized registries or
  adapters
- support built-in, repo-local, and external optimizer / scheduler classes
- isolate wrapper/special-case behavior
- keep scheduler construction independent from grouping policy

Success criteria:

- adding a new optimizer or scheduler is mostly registration or adaptation
- vendor-specific quirks stop leaking into unrelated parts of the layer

### Phase 4: Execution Runtime

Goal: create a real home for advanced stepping behavior.

Planned work:

- introduce runtime/execution helpers that own:
  - optimizer stepping
  - scheduler stepping
  - zero-grad
  - optional train/eval hooks
  - future multi-optimizer or fused-group execution
- keep the normal loop readable while letting advanced behavior live behind a
  smaller seam

Success criteria:

- fused optimizer groups have a clear home
- execution behavior no longer needs to be embedded in factory or mode code

### Phase 5: Numeric / Precision Features

Goal: support optional numeric/runtime enhancements without architectural
sprawl.

Planned work:

- add explicit feature hooks/capabilities for:
  - stochastic rounding
  - Kahan summation
  - fused Adafactor patching
  - future similar experiments
- keep these features separate from parameter grouping and selection policy

Success criteria:

- numeric/runtime enhancements are opt-in capabilities
- they do not require unrelated factory or mode rewrites

## Capability Ownership Map

### Selective learning rates

Primary owner:

- grouping layer

### Arbitrary selections / exclusions / groupings

Primary owners:

- selection layer
- grouping layer

### Fused optimizer groups

Primary owner:

- execution/runtime layer

### Adding new optimizers and schedulers

Primary owners:

- optimizer factory / registry layer
- scheduler factory / registry layer

### Stochastic rounding / Kahan / similar numeric features

Primary owners:

- numeric/runtime feature layer
- execution/runtime integration

## First Concrete Milestone

The most useful low-risk first milestone is:

- complete the package rename
- unify shared arg parsing
- introduce one shared parameter-group payload shape
- make both fine-tune and PEFT emit that shape
- keep current optimizer/scheduler behavior unchanged

That sets the architecture direction correctly without forcing fused execution,
registry redesign, or numeric-feature work immediately.

## Resolved Direction

- Parameter selection and grouping logic belongs to the optimization layer.
- The adapter system is not the long-term owner of grouping policy.
- If the current adapter boundary blocks this refactor, adapter rework should
  be prioritized rather than pushing grouping back into adapters.
- Scheduler construction should remain one shared entrypoint for both standard
  and custom schedulers unless a concrete limitation forces a split.
- Numeric/runtime add-ons should be treated as optional optimizer
  augmentations by default; only clearly optimizer-agnostic behavior should
  become a shared feature hook.

## Remaining Open Questions

- When fused optimizer groups arrive, do they imply multiple real optimizer
  instances, grouped stepping within one optimizer, or both?
- Which numeric/runtime features belong as generic layer capabilities versus
  optimizer-specific adapters?

## Current Understanding: Fused Optimizer Groups

Based on the local reference implementation and the legacy SDXL script kept in
this repo, "fused optimizer groups" currently appears to mean:

- split trainable parameters into several groups
- create one real optimizer per group
- create one LR scheduler per optimizer
- register per-parameter post-accumulate-grad hooks
- step and zero each optimizer once all gradients for its group have arrived

This is conceptually a grouped form of "optimizer step in backward" rather than
a special fused optimizer class.

Expected benefit:

- lower peak VRAM usage by freeing gradients earlier instead of holding the
  full gradient set until a single optimizer step at the end of backward
- possible additional savings when optimizer intermediates are smaller because
  work is split across smaller optimizers/groups

Expected tradeoffs:

- more complex execution/runtime behavior
- scheduler coordination across multiple optimizers
- more complicated gradient clipping semantics
- more intrusive integration with accelerator/distributed behavior

This should therefore be treated as an execution/runtime concern first, even if
group construction influences how the execution path is configured.

## Future Exploration

These are intentionally out of immediate scope for the first optimization-layer
cleanup, but they are relevant follow-up areas and should stay visible:

- optimizer step-in-backward support beyond the current Adafactor-specific
  fused-backward path
- fused optimizer groups as a grouped multi-optimizer execution mode for VRAM
  reduction
  - worth validating explicitly under multi-GPU / distributed training, since
    there has already been anecdotal OOM pain around fused optimizer groups in
    trainer-style execution and the failure mode was not yet explained
  - related follow-up: wrappers such as CPU offload can explode a few logical
    groups into hundreds or thousands of execution groups, which currently
    breaks trainer/logging assumptions; the future grouping work should
    explicitly separate logical group ownership from optimizer-internal
    execution grouping
- optional optimizer augmentations such as stochastic rounding, Kahan-style
  summation, or similar precision/runtime experiments
  - reported future lead: a fused stochastic-rounding path was said to improve
    end-to-end training throughput by roughly 4%, which makes this category
    worth keeping visible as a real performance exploration rather than only a
    precision experiment
- kernel-fused optimizer implementations and related lower-level performance
  work
- clearer capability modeling for which optimizers support which execution or
  augmentation paths

## Vendor Absorption Notes

The vendored `LoraEasyCustomOptimizer` tree is a useful pressure test for the
future optimization layer. Its contents suggest several concrete requirements
for absorption:

### 1. "Optimizer" does not always mean a plain optimizer class

The vendor tree includes:

- ordinary optimizer classes with larger constructor surfaces
- wrapper optimizers such as schedule-free wrappers that take a base optimizer
  class as an argument
- low-bit / offload wrappers that behave like optimizers but impose extra
  runtime constraints
- standalone LR scheduler classes

The future layer therefore needs to distinguish:

- plain optimizer classes
- wrapper optimizers
- optimizer augmentations
- standalone schedulers

### 2. Constructor surfaces are much richer than current built-ins

Many vendor optimizers expose options such as:

- `use_orthograd`
- adaptive clipping settings
- stable spam clipping settings
- stochastic rounding flags
- state precision / low-bit state settings
- compile flags such as `torch_compile` / `compile_step`
- internal schedule or warmup-like knobs

This reinforces the need to treat many "features" as optimizer-specific
constructor/runtime capabilities rather than assuming they belong in a generic
shared feature layer.

### 3. Wrapper optimizers need class-valued arguments

Some wrappers, especially schedule-free wrappers, expect a base optimizer class
or fully qualified import path rather than only literal scalar arguments.

That means the registry/integration layer should support:

- importing optimizer classes from qualified names
- distinguishing constructor args from imported class references
- filtering forwarded kwargs against the wrapped/base optimizer signature when
  needed

### 4. Some schedulers are true standalone scheduler classes

The vendor tree includes explicit scheduler classes such as
`CosineAnnealingWarmRestarts` and `RexAnnealingWarmRestarts`.

These schedulers:

- patch or track optimizer step behavior
- store scheduler state inside optimizer param groups
- require resume-time validation of param-group metadata

This supports keeping one scheduler entrypoint, but it also means scheduler
integration should be able to handle custom resume/state expectations cleanly.

### 5. Some optimizers own schedule-like behavior internally

A number of optimizers appear to include their own schedule-like logic (for
example warmup/death-rate/decay behavior) instead of relying only on an
external LR scheduler.

The layer should therefore avoid assuming:

- every optimizer wants an external scheduler
- scheduler policy is always separate from optimizer behavior

### 6. Runtime constraints vary by optimizer family

Some vendor optimizers/wrappers impose constraints such as:

- needing explicit `train()` / `eval()` transitions
- reduced compatibility with standard LR scheduler behavior
- restrictions around gradient accumulation
- restrictions around gradient clipping
- dependence on `torch.compile()` support
- CPU-offload or mixed device/state behavior

This strengthens the case for a future capability matrix so the runtime can
decide which execution paths and warnings apply to which optimizer families.

### Working absorption takeaway

When absorbing from this vendor tree, prefer:

- normalizing import/registration and config ownership first
- preserving optimizer-specific options rather than flattening them too early
- treating wrappers/offload/schedule-free paths as distinct integration shapes
- adding capability metadata as these optimizers come in, instead of assuming
  uniform behavior across the whole layer

## Proposed Absorption Model

The long-term goal should be to absorb optimizers and schedulers into the repo's
own optimization layer without inheriting vendor package shape, naming, or API
quirks wholesale.

### 1. Registration should be normalized

The layer should register optimizers and schedulers through explicit metadata
entries rather than giant branch trees.

Conceptually, each optimizer entry should describe:

- canonical repo-facing name
- implementation target (class or factory)
- integration shape
- capability flags
- config adapter / argument-normalization hook

Conceptually, each scheduler entry should describe:

- canonical repo-facing name
- implementation target (class or factory)
- scheduler style / required runtime inputs
- capability flags
- config adapter / argument-normalization hook

Example optimizer registration shape:

```python
OptimizerRegistration(
    name="compass",
    target="library.optimization.vendors.compass.Compass",
    kind="optimizer",  # optimizer | wrapper | offload_wrapper
    capabilities={
        "train_eval_toggle": False,
        "supports_external_scheduler": True,
        "supports_compile_flag": True,
        "supports_stochastic_rounding": False,
    },
    adapt_options=adapt_compass_options,
)
```

Example scheduler registration shape:

```python
SchedulerRegistration(
    name="rex_warm_restarts",
    target="library.optimization.schedulers.rex.RexAnnealingWarmRestarts",
    kind="scheduler",  # scheduler | optimizer_embedded | no_op
    required_inputs={"optimizer", "num_training_steps"},
    adapt_options=adapt_rex_scheduler_options,
)
```

The key point is not the exact class names above; it is that absorbed
implementations should plug into repo-owned metadata rather than forcing the
factory layer to infer behavior from ad hoc string checks.

### 2. Integration shape should be first-class

Not every absorbed optimizer has the same role. The layer should explicitly
distinguish at least these categories:

- `optimizer`
  - normal optimizer class
- `wrapper`
  - wraps another optimizer and may need a base optimizer class or factory
- `offload_wrapper`
  - optimizer-like wrapper with runtime/device/scheduler constraints
- `optimizer_embedded_schedule`
  - optimizer whose own behavior includes schedule-like logic
- `scheduler`
  - ordinary external LR scheduler
- `no_op_scheduler`
  - explicit no-op scheduler path for schedule-free / optimizer-owned schedule

This lets the runtime and scheduler factory make decisions based on declared
integration shape instead of scattered name checks.

### 3. Config should have a shared core plus optimizer-specific options

The absorbed layer should have one repo-facing config surface with:

- shared core fields used by most optimizers/schedulers
- a normalized options bag for implementation-specific knobs
- compatibility adapters for older `optimizer_type` / `optimizer_args` style
  configs during migration

Desired direction:

- shared core:
  - optimizer name
  - scheduler name
  - learning rates
  - max grad norm
  - high-level runtime toggles such as fused backward/groups
- implementation-specific:
  - optimizer options
  - scheduler options

Conceptually:

```yaml
optimizer:
  name: "compass"
  options:
    betas: [0.95, 0.9999]
    weight_decay: 0.0
    adaptive_clip: 1.0
    use_orthograd: true

  scheduler:
    name: "rex_warm_restarts"
    options:
      gamma: 0.9
      first_cycle_max_steps: 1000
      warmup_steps: 100
```

The important boundary is:

- shared config should express intent the repo understands generically
- optimizer-specific options should stay namespaced under the absorbed
  optimizer/scheduler

That avoids polluting the top-level schema with every vendor-specific knob while
still letting absorbed implementations keep their real behavior.

### 4. Capability metadata should drive orchestration

As optimizers and schedulers are absorbed, each registration should declare
capabilities or constraints such as:

- requires `train()` / `eval()` transitions
- supports external LR scheduler
- prefers no-op scheduler
- supports fused backward adaptation
- supports grouped fused execution
- supports stochastic rounding
- supports compile flag
- requires CPU offload handling
- disallows gradient accumulation
- disallows generic gradient clipping

The runtime/orchestration layer should read this metadata instead of inferring
behavior from string names.

This is especially important for:

- schedule-free wrappers
- CPU-offload optimizers
- low-bit optimizers
- optimizer-specific augmentations

### 5. Wrappers should be expressed explicitly, not hidden in args

Vendor code often uses patterns like:

- wrapper optimizer class
- `base_optimizer_type=<qualified path>`
- wrapper-specific options mixed with base optimizer options

The absorbed design should represent this explicitly.

Conceptually:

```python
OptimizerSpec(
    name="schedulefree_wrapper",
    options={"sf_momentum": 0.9},
    wrapped=OptimizerSpec(
        name="compass",
        options={"weight_decay": 0.0, "use_orthograd": True},
    ),
)
```

Even if the first implementation still uses compatibility parsing internally,
the architecture should move toward explicit wrapped specs rather than
continuing to hide class references inside free-form arg strings forever.

### 6. Scheduler handling should stay one entrypoint

The user-facing layer should still have one scheduler entrypoint, but internally
it should support multiple scheduler shapes:

- normal external schedulers
- custom absorbed schedulers
- no-op schedulers for schedule-free optimizers
- optimizer-owned schedule behavior where an external scheduler should be
  suppressed or warned on

So:

- one public scheduler selection path
- multiple internal scheduler integration kinds

### 7. Absorption should happen through repo-owned adapters

When we absorb from vendor code, the preferred sequence should be:

1. copy or port the implementation into a repo-owned module layout
2. add a registration entry
3. add an option adapter that normalizes config
4. declare capability metadata
5. only then wire it into shared orchestration/runtime behavior

This keeps the absorbed implementation decoupled from the vendor package's
assumptions and makes later cleanup possible.

### 8. Immediate practical direction

The next useful design steps should likely be:

- keep Phase 1 compatibility with current `optimizer_type` and `optimizer_args`
- introduce repo-owned registration objects for built-ins first
- add scheduler registrations alongside optimizer registrations
- represent wrapper/offload/schedule-free integration kinds explicitly
- postpone schema expansion until the registration/capability model is stable

That sequence should let the repo absorb vendor optimizers gradually without
locking itself into vendor API shape.

### Current absorbed set

The optimization layer now has repo-owned examples for each of the main
integration shapes we expected to pressure-test first:

- wrappers: `ScheduleFreeWrapper`, `SNOO_ASGD`
- plain optimizers: `AdaBelief`, `Adan`, `LaProp`, `ADOPT`, `ADOPTMARS`,
  `FADOPTMARS`, `LPFAdamW`, `SGDSaI`, `Adai`, `VSGD`, `RACS`, `Alice`, `Lamb`
- schedule-free leaves: `ADOPTScheduleFree`, `ADOPTEMAMixScheduleFree`,
  `ADOPTNesterovScheduleFree`, `ADOPTMARSScheduleFree`,
  `ADOPTAOScheduleFree`,
  `FADOPTScheduleFree`, `FADOPTEMAMixScheduleFree`,
  `FADOPTNesterovScheduleFree`, `FADOPTMARSScheduleFree`
- optimizer-local augmentations: `AdamW8bitKahan`
- low-bit family absorptions: `AdamW8bitAO`, `AdamW4bitAO`, `AdamWfp8AO`
- additional family absorption proving the donor-family workflow: `RMSProp`,
  `RMSPropADOPT`, `RMSPropADOPTMARS`
- additional helper-heavy family absorption: `AdEMAMix`,
  `SimplifiedAdEMAMix`, `SimplifiedAdEMAMixExM`
- additional Fisher/Compass-family absorption: `FCompass`, `FCompassADOPT`,
  `FCompassADOPTMARS`, `FCompassPlus`
- schedulers: `CosineAnnealingWarmRestarts`, `RexAnnealingWarmRestarts`

## Working Recommendation

Treat this layer as a long-term subsystem, but avoid introducing one oversized
umbrella abstraction too early.

The first stable seams should likely be:

- selection
- grouping
- instantiation
- execution
- feature hooks

If a central object naturally emerges later, name it based on the role it
actually plays rather than forcing a premature umbrella type now.
