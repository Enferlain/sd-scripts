# Adapter Layer Direction

Date: 2026-04-04

## Purpose

This note is about the shape of the adapter layer once LyCORIS is absorbed.

The main question is not:

- whether LyCORIS should stay vendored
- whether the long-term answer is a wrapper around vendor code

The main question is:

- what kind of system the repo wants for training adapters at large
- where absorbed LyCORIS algorithms fit inside that system
- how that shape stays useful for future adapter families without making the
  current diffusion path worse

Important scope assumption for this pass:

- component/pattern/override selection and much of the grouping policy are
  planned as optimization-layer ownership for now

So this note treats the adapter layer primarily as a trainable augmentation
system, not as the owner of selector grammar or optimizer policy.

## Current Reading

The active code already has a meaningful split:

- training phases own ordering
- training modes own divergent trainable behavior
- strategies own model-family preparation
- adapter implementations own most adapter-local math and weight I/O

That base is good.

The main current issue is that the adapter system is still implicit and
Kohya-shaped.

### Current Runtime Shape

Today `library/training/modes/peft_mode.py` is effectively the adapter runtime.
It dynamically imports an adapter module and expects:

- `create_adapter(...)`
- `create_adapter_from_weights(...)`

It also expects the returned trainable object to support a loose surface such
as:

- `apply_to(...)`
- `load_weights(...)`
- `merge_to(...)`
- `prepare_optimizer_params(...)`
- `prepare_grad_etc(...)`
- `enable_gradient_checkpointing()`
- `on_epoch_start(...)`
- `get_trainable_params()`
- `save_weights(...)`
- optional `apply_max_norm_regularization(...)`
- optional `get_diagnostics_components()`

That is a real contract, but it is still only a duck-typed convention.

### Current Pressure Points

The biggest architectural pressure points are:

- the adapter layer is implicitly defining itself through `PeftMode`
- built-in adapters and LyCORIS-shaped adapters are all still organized around
  the older diffusion/Kohya mental model
- `PeftConfig` mixes generic adapter concerns with LoRA/LyCORIS-specific knobs
- optimizer preparation is still partly adapter-owned in a way that leaks
  grouping policy into adapter code

The result is a system that works, but does not yet have a clean answer to
"what is an adapter in this repo?"

## Main Conclusion

The adapter layer should be treated as a system for trainable augmentations.

That system should have three main parts:

1. algorithm layer
2. attachment/runtime layer
3. handoff boundary to strategy/mode and optimization code

Under that shape:

- LyCORIS is not a special outer boundary
- LyCORIS algorithms become absorbed algorithm implementations inside the
  adapter system
- the adapter layer owns augmentation behavior and state
- the optimization layer owns selection/grouping policy for now

That is the framing that fits the current direction best.

## Layer Responsibilities

### Adapter Layer Owns

The adapter layer should own:

- algorithm implementations
- attachment/injection into already-resolved target modules
- trainable augmentation lifecycle
- state dict save/load behavior
- merge/diff behavior where relevant
- adapter-local diagnostics
- adapter-local metadata

In short: it owns what the augmentation is and how it behaves once attached.

### Optimization Layer Owns For Now

The optimization layer should own, for now:

- component selection
- pattern selection
- overrides by component or pattern
- final parameter grouping
- LR policy and grouping semantics

This means the adapter layer should not become the home for a giant selector
DSL or for long-lived grouping logic unless that responsibility clearly moves
later.

### Strategy / Mode Layer Owns

Strategies and modes should own:

- model-family target discovery
- construction of a resolved target bundle
- deciding when adapter training is active
- bridging the adapter runtime into the shared trainer flow

This keeps family-specific structure such as:

- SD / SDXL text encoders
- denoiser naming
- future transformer or LLM component layouts

out of generic adapter algorithm code.

## Proposed Shape

### 1. Algorithm Layer

The first-class concept should be the adapter algorithm.

Examples:

- `lora`
- `locon`
- `loha`
- `lokr`
- `ia3`
- `dylora`
- `oft`
- `diag-oft`
- `boft`
- `full`

These should live as repo-owned algorithm entries, regardless of whether some
of their implementation still starts from donor code.

This is the key point for LyCORIS absorption:

- LyCORIS should fit in as a source of algorithms
- not as a parallel architecture that the repo keeps routing around forever

That suggests a shape like:

```text
library/adapters/
  api.py
  runtime.py
  registry.py
  targets.py
  algorithms/
    lora.py
    loha.py
    lokr.py
    ia3.py
    dylora.py
    oft.py
    diag_oft.py
    boft.py
    full.py
  internals/
    low_rank.py
    state_io.py
    metadata.py
```

The exact file layout is flexible. The important thing is the concept:

- algorithm names are first-class
- absorbed LyCORIS content lands in the same algorithm space as built-in
  adapters

### 2. Attachment / Runtime Layer

The second concept should be the attached trainable adapter instance.

This is the object the trainer actually works with.

It should represent:

- one training-time augmentation instance
- attached onto a resolved set of target modules
- with save/load/merge/lifecycle behavior

Conceptually:

```python
class AdapterInstance(Protocol):
    def attach(self, targets: ResolvedAdapterTargets) -> None: ...
    def prepare_for_training(self) -> None: ...
    def iter_trainable_parameters(self) -> Iterator[AdapterParameterRef]: ...
    def on_epoch_start(self) -> None: ...
    def save_weights(self, output_path: str, dtype: torch.dtype | None, metadata: dict[str, str] | None) -> None: ...

    # Optional capabilities
    def load_weights(self, input_path: str) -> object: ...
    def merge_into_base(self, merge_context: AdapterMergeContext) -> None: ...
    def apply_max_norm_regularization(self, max_norm: float, device: torch.device) -> tuple[int, float, float]: ...
    def get_diagnostics_components(self) -> tuple[list[tuple[str, nn.Module]], list[tuple[str, str]] | None]: ...
```

The exact method names can differ.

The important change is that the repo should define this contract explicitly
instead of letting `PeftMode` define it accidentally.

### 3. Resolved Target Bundle

The adapter layer should consume a resolved target bundle, not raw selector
rules.

For this direction, the target object should stay intentionally narrow.

It should answer:

- which components are being adapted
- what modules inside those components are selected
- what logical names/tags those modules have
- what already-resolved per-target options should be applied

It should not try to become the place where selection grammar lives.

Conceptually:

```python
class ResolvedAdapterTarget(TypedDict):
    component: str
    path: str
    module: nn.Module
    tags: set[str]
    options: dict[str, object]


class ResolvedAdapterTargets(TypedDict):
    family: str
    targets: list[ResolvedAdapterTarget]
```

The real types can be stronger than this. The point is the boundary:

- strategy/mode + optimization code resolve what is being trained
- adapter algorithms consume that resolved set

That is a much better fit for the current ownership plan.

### 4. Optimization Handoff

The adapter layer should stop owning the final optimizer grouping interface in
its current form.

Today adapters expose `prepare_optimizer_params(...)`, which mixes together:

- adapter-local knowledge
- parameter discovery
- grouping decisions
- LR-policy behavior

That is convenient, but it blurs the boundary.

The better long-term shape is:

- adapter instance exposes trainable parameter refs plus adapter-local tags
- optimization layer decides how those refs are grouped and scheduled

For example, the adapter layer can expose metadata such as:

- component name
- algorithm name
- matrix role
- parameter kind
- rank or structural hints

without owning the final grouping logic itself.

That lines up better with the current plan to keep component/pattern/override
logic in the optimization layer.

## What LyCORIS Becomes After Absorption

LyCORIS should not remain a distinct outer integration shape.

After absorption, it should mostly disappear as an architecture boundary and
remain visible primarily as:

- algorithm implementations
- some shared internal helpers
- some compatibility/import logic during migration

In other words:

- `loha` should just be an adapter algorithm
- `lokr` should just be an adapter algorithm
- `ia3` should just be an adapter algorithm
- `diag-oft` / `boft` / `full` should just be adapter algorithms

That is the cleanest fit into the adapter layer at large.

If temporary adaptation or translation scaffolding exists during the migration,
that is fine, but it should be treated as migration scaffolding, not as the
target architecture.

## How Built-ins And Absorbed LyCORIS Should Meet

The current built-ins (`lora.py`, `dylora.py`, `oft.py`) and absorbed LyCORIS
content should converge at the runtime/algorithm level.

They do not need one giant inheritance tree.

They do need to converge on:

- one explicit adapter-instance contract
- one resolved-target input shape
- one state I/O and metadata story
- one diagnostics story
- one optimization handoff story

That is the point where the adapter layer becomes a system instead of a pile
of parallel implementations.

## Config Direction

If the adapter layer is about trainable augmentations, then the config should
describe:

- which algorithm is being trained
- the core algorithm hyperparameters
- resume/load/base-weight behavior
- algorithm-specific extras

It should not be the sole home for target-selection grammar if that logic
belongs to optimization for now.

So the likely direction is:

- adapter config declares the algorithm and augmentation-local settings
- optimization config declares selection/grouping policy

Conceptually:

```text
adapter:
  algorithm: loha
  rank: 16
  alpha: 16
  dropout: 0.0
  weights: ...
  base_weights: ...
  algorithm_args: ...

optimization:
  adapter_targeting: ...
  adapter_grouping: ...
  adapter_overrides: ...
```

This does not require an immediate config rewrite.

It does suggest that the current `PeftConfig` should be seen as a compatibility
container rather than the final architectural shape.

## Why This Shape Still Supports Future Adapters

This direction still leaves room for future non-diffusion adapters because the
important extensibility seam is not "provider".

The important seam is:

- strategy/mode code can produce different resolved targets for different model
  families
- adapter algorithms operate on those resolved targets
- optimization code can still own grouping/selection policy across families

That means future support for things like LLM adapters is mostly a question of:

- target resolution
- compatibility rules
- algorithm applicability

not a question of rewriting the adapter runtime again.

## Suggested Near-Term Direction

If we want the next steps to line up with this model, the most useful work is:

1. Define the explicit adapter-instance contract currently implied by
   `PeftMode`.
2. Introduce a resolved target bundle so the adapter runtime stops taking raw
   diffusion-era positional assumptions as its conceptual input.
3. Move the optimizer handoff away from `prepare_optimizer_params(...)` toward
   exposing tagged trainable parameter refs.
4. Start absorbing LyCORIS algorithms into the common adapter algorithm space
   rather than preserving LyCORIS as a separate public runtime boundary.
5. Revisit config splitting once the runtime and optimization boundary are
   explicit.

That sequence keeps the design centered on "system for training adapters"
instead of "how do we keep routing around a donor package cleanly".

## Non-Goals

This note does not recommend:

- keeping LyCORIS as the public architecture boundary
- pushing the optimization selector/grouping DSL into the adapter layer right
  now
- forcing every adapter algorithm into one heavy base class
- doing a giant one-pass config migration before the runtime shape is clear

The target is a repo-owned adapter training system where absorbed LyCORIS
algorithms fit naturally inside the same layer as the rest of the adapter
family.
