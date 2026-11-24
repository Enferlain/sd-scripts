# Model, Strategy, And Trainer Production Inventory

## Status

This is a current-state research record, not an implemented contract, accepted
design, or migration plan.

It inventories the active `train.py` path as it exists on 2026-07-28 and tests
that path against the vocabulary in
[`strategy_system_direction.md`](strategy_system_direction.md). The purpose is
to make the next design discussion concrete:

- what `library/models/` owns today
- what `library/strategies/` owns today
- what the trainer and training modes own today
- which current strategy methods are actually trainer-facing
- which methods are only family-internal collaboration seams
- where loaded state moves back and forth
- which placements and boundaries are already under pressure

The inventory deliberately excludes `tools/`, deprecated entrypoints, and the
still-separate textual-inversion runtime. Those paths are not part of the active
model/strategy/trainer redesign.

The findings were produced from the current source plus the repository code
graph at generation `2026-07-28T23:55:46Z`. The relevant model, strategy,
training, and strategy-test paths had no recorded source coverage gaps. The
only bounded-scope exclusions were generated `__pycache__` directories. Graph
coverage remains a best-effort aid, so material claims below were also checked
against source.

## Executive Findings

The present architecture has a coherent foundation:

```text
train.py
  → constructs one model-family TrainingStrategy
  → constructs one TrainingMode
  → gives both to one Trainer
```

It also has five structural inconsistencies that the next contract cannot
preserve accidentally.

1. **`TrainingStrategy` is both a trainer façade and an internal mixin bus.**
   The trainer calls only part of its 25-operation surface. Family
   implementations call the rest through `self` because the aggregate inherits
   every facet.
2. **Loaded components have one useful generic representation, but no single
   lifecycle owner.** Model packages declare the component surface, strategies
   load and bind it, the trainer stores it, modes replace prepared modules
   inside it, and the trainer passes role projections back to strategies.
3. **Training modes are collaborators with unrestricted trainer access.**
   Their protocol intentionally takes the whole `Trainer`, and both active
   modes mutate trainer state to communicate their results.
4. **The batch boundary mixes several concerns.** Family `process_batch()`
   methods combine representation preparation, conditioning, objective work,
   prediction, loss construction, and some training policy. Validation then
   implements a parallel family flow instead of using that advertised
   training-or-validation boundary.
5. **Folder placement mostly expresses the intended split, but several files
   contain mixed responsibilities.** The clearest examples are cache codec plus
   cache persistence in family strategy files, accelerator patching in model
   utilities, runtime SDXL conditioning math in a conversion module, and a
   logging process stored on each family strategy.

The production system therefore supports the principle:

```text
contract → strategy fulfillment → trainer execution
```

but its current ABC inheritance does not yet constitute strong fulfillment.
It proves that methods exist. It does not prove that one selected arrangement
is complete, compatible, or valid before training starts.

## Active Construction And Ownership Path

The active construction path is:

```text
train.py::train
├── prepare and validate RunConfig
├── build_training_strategy(cfg)
│    └── static model_type registry
│         ├── sd1 / sd15 / sd2 → SdTrainingStrategy
│         ├── sdxl             → SdxlTrainingStrategy
│         └── sd3              → Sd3TrainingStrategy
├── build_training_mode(cfg)
└── Trainer(cfg, strategy, mode)
     └── trainer.train()
```

`build_training_strategy()` returns one aggregate object. The current problem
is therefore not a missing top-level aggregate. It is what that aggregate
means and owns.

Each family aggregate is assembled through multiple inheritance:

```text
<family> model loading
+ tokenization
+ text encoding
+ conditioning
+ caching
+ sampling
+ checkpointing
+ validation
+ diffusion training
+ denoiser calling
+ model preparation
+ selected feature ABCs
+ TrainingStrategy
```

This gives the trainer one object, but it gives that object three roles:

1. trainer-facing model-family façade
2. namespace through which family facets find one another
3. holder of some per-run mutable state

Those roles are not distinguished in the type surface.

## Loaded Component Surface

The strongest model-side contract already present is
`LoadedModelComponentSpec` / `LoadedModelComponent` in
`library/models/components.py`.

Model packages declare stable top-level keys, public names, roles, and
capabilities. Loaders bind live modules to those declarations:

| Family | Declared loaded components |
| --- | --- |
| SD1/SD2 | `clip_l`, VAE, UNet |
| SDXL | `clip_l`, `clip_g`, VAE, UNet |
| SD3 | `clip_l`, `clip_g`, T5-XXL, VAE, MMDiT |

Generic callers can filter components by role or capability without a central
family-name branch. Unloaded declarations remain in the tuple with
`module=None`, which preserves absent or deferred slots. Current tests
explicitly cover an SD3 surface with absent T5 and denoiser modules.

The trainer stores `loaded_components` as the primary surface and derives
compatibility projections:

```text
loaded_components
├── role=text_encoder → trainer.text_encoders
├── role=vae          → trainer.vae
└── role=denoiser     → trainer.denoiser
```

The projection setters rewrite modules back into `loaded_components` after
Accelerate wrapping. This is materially better than independent untracked
`text_encoder`, `vae`, and `unet` slots, but it remains shaped around the
current three generic roles. Capabilities exist in the dataclasses, yet the
current family declarations do not use them.

### Current component lifecycle

```text
model package
  declares component specs
        │
strategy loader
  calls model-family loader
  applies run/performance choices
  binds modules to specs
        │
Trainer.setup
  stores loaded_components
  files model realization metadata
        │
TrainingMode
  selects trainables
  freezes/unfreezes modules
  wraps selected modules with Accelerator
  replaces modules through trainer projections
        │
training phases
  query projections
  pass VAE/encoders/denoiser back to strategy methods
```

The surface is coherent; ownership is not. The model packages own declarations,
the family strategy creates the bound collection, the trainer owns the live
collection, and modes mutate it through the trainer.

## Complete Current Strategy Contract Inventory

`TrainingStrategy` inherits eleven ABC facets containing 25 operations. The
table classifies actual production use, not just the class in which each
method was declared.

Classification vocabulary:

- **Trainer lifecycle**: directly called by the trainer, a shared phase, or a
  training mode.
- **Conditional concern**: directly used only when a selected run feature
  requires it. It still needs fulfillment for that selected arrangement.
- **Family collaboration**: used by one family facet to call another through
  the aggregate `self`.
- **Mixed boundary**: one method currently spans more than one ownership area.

| Facet | Operation | Production caller | Current classification |
| --- | --- | --- | --- |
| Loading | `load_target_model` | `Trainer.setup` | Trainer lifecycle; mixed loading/result boundary |
| Loading | `load_denoiser_lazily` | model-preparation phase when the denoiser role is unloaded | Conditional concern with no active family override |
| Tokenization | `tokenizers` | `Trainer.setup` | Trainer lifecycle state exposure |
| Tokenization | `tokenize` | SD/SDXL conditioning facets | Family collaboration |
| Tokenization | `tokenize_captions` | per-epoch token-cache preparation | Conditional concern |
| Text encoding | `encode_tokens` | SD/SDXL conditioning facets | Family collaboration |
| Text encoding | `encode_te_outputs_in_memory` | shared caching phase | Conditional concern |
| Text encoding | `get_models_for_text_encoding` | family conditioning facets | Family collaboration |
| Conditioning | `resolve_conditioning` | family diffusion and validation facets | Family collaboration; candidate feature boundary |
| Caching | `create_latent_caching_strategy` | shared caching phase | Conditional concern; mixed codec/storage boundary |
| Caching | `create_te_caching_strategy` | shared caching phase | Conditional concern; mixed codec/storage boundary |
| Caching | `get_token_cache_encoder_names` | per-epoch token-cache preparation | Conditional concern |
| Caching | `build_te_cache_model_bundle` | shared caching phase | Conditional concern |
| Sampling | `sample_images` | training-loop and orchestration helpers | Trainer lifecycle; broad family operation |
| Checkpointing | `resolve_model_artifact_facts` | trainer metadata/checkpoint preparation | Trainer lifecycle; narrow typed request/result |
| Checkpointing | `save_model_checkpoint` | `FineTuneMode` | Conditional concern; broad whole-`Trainer` input |
| Validation | `calculate_val_loss` | shared sampling/validation orchestration | Trainer lifecycle; very broad state input |
| Diffusion | `process_batch` | training loop | Trainer lifecycle; mixed representation/objective/loss boundary |
| Denoiser | `call_denoiser` | family diffusion facets | Family collaboration; scoped forward seam |
| Preparation | `cast_text_encoder` | setup/model preparation | Conditional concern expressed as a family hook |
| Preparation | `cast_vae` | `Trainer.setup` | Conditional concern expressed as a family hook |
| Preparation | `cast_denoiser` | model preparation and adapter mode | Conditional concern expressed as a family hook |
| Preparation | `prepare_text_encoder_grad_ckpt_workaround` | optimizer phase | Conditional concern; family/component quirk |
| Preparation | `prepare_text_encoder_fp8` | model-preparation phase | Conditional concern; family/component quirk |
| Preparation | `post_process_trainable` | adapter and fine-tune modes | Mixed family/mode preparation boundary |

Two important conclusions follow from this inventory.

First, moving some methods to a list called "optional capabilities" would not
solve the contract. Lazy loading, caching, full-model saving, FP8 handling, and
gradient-checkpoint preparation are required when the selected run requests
them. The missing concept is conditional fulfillment, not optionality.

Second, `tokenize()`, `encode_tokens()`,
`get_models_for_text_encoding()`, `resolve_conditioning()`, and
`call_denoiser()` are not presently trainer-facing operations. They are public
ABCs because family facets collaborate through multiple inheritance. They may
be valid feature protocols later, but their presence in `TrainingStrategy`
does not prove the trainer needs them on its top-level contract.

## Contract Fulfillment Gaps

Python ABC construction verifies that abstract methods are present. It does
not verify that the selected run can use them coherently.

### Default methods that fail late

`load_denoiser_lazily()` and several caching/preparation methods are concrete
defaults that raise `NotImplementedError`. The component surface and model
preparation phase explicitly permit a deferred denoiser, and the tests preserve
that state, but none of the active family strategies implements the lazy-load
operation.

`save_model_checkpoint()` also defaults to `NotImplementedError`. SDXL and SD3
implement it; the SD checkpointing facet currently implements artifact fact
resolution but not full-model serialization. The current type can therefore
be constructed even when a selected full-model save route is not fulfilled.

This is the nominal-placeholder problem described in the direction document:
method presence is not equivalent to supported behavior.

### No arrangement-level declaration

The current strategy has no single declaration equivalent to:

```text
this run selected:
  latent representation
  live text conditioning
  adapter training
  DDPM objective
  adapter safetensors persistence

and every required provider is compatible
```

Configuration validation handles many individual combinations elsewhere, but
the completed strategy object does not publish one typed fulfillment result.

### Weak semantic types

Material boundaries use `Any`, positional component lists, broad `cfg`, or the
entire `Trainer`. Family conditioning payloads are a positive exception:
`ModelConditioning` and typed family dataclasses let the data path transport
different internals. That pattern has not yet been extended to loading,
preparation, execution, validation, or saving.

### Tests verify implementations, not one shared contract

Current strategy tests cover:

- base helper behavior and inheritance placement
- family assembly and method origin
- family tokenization, encoding, conditioning, denoiser calls, and caching
- batch output behavior
- loading surfaces, including absent/deferred components
- checkpoint and strategy-context behavior

There is not yet one shared conformance suite that applies the same contract
scenarios to every active family according to its declared selections. A
method-origin assertion can prove that SDXL received its diffusion facet; it
cannot prove that a configured SDXL run has a complete compatible filing.

## Family Strategy Assembly And Internal Collaboration

The active family assemblies are broadly uniform:

| Concern | SD | SDXL | SD3 |
| --- | --- | --- | --- |
| Aggregate strategy | Multiple-inheritance façade | Multiple-inheritance façade | Multiple-inheritance façade |
| Representation | VAE latent | VAE latent | SD3 VAE latent transform |
| Conditioning | CLIP-L | CLIP-L + CLIP-G + size conditioning | CLIP-L + CLIP-G + T5 |
| Predictor | UNet | UNet | MMDiT |
| Objective support in batch path | DDPM | DDPM and rectified flow | Rectified flow |
| Weighted prompts feature ABC | Yes | Yes | No |
| Family metadata feature ABC | No | No | Yes |
| Full-model save override | No | Yes | Yes, safetensors only |

The uniform assembly shape is valuable: the trainer has no family branch for
normal execution. The weakness is how the facets are wired.

For SD and SDXL, conditioning calls methods supplied by sibling mixins:

```text
conditioning
├── self.get_models_for_text_encoding(...)
├── self.tokenize(...)
└── self.encode_tokens(...)
```

All three family diffusion facets call:

```text
self.resolve_conditioning(...)
self.call_denoiser(...)
```

This is why code navigation makes operations such as `process_batch()` appear
to have no concrete caller: the trainer calls the aggregate contract
dynamically, and the concrete implementation is selected through factory
construction plus method resolution order.

Explicit composition is not automatically superior, but the current wiring
has no declaration of which collaborator supplies which concern or whether
their inputs and outputs are compatible. Multiple inheritance is performing
both implementation reuse and dependency resolution.

## State Ownership And Back-And-Forth

### Trainer-owned state

The trainer currently owns:

- the live loaded-component tuple and role projections
- model version and realization metadata state
- VAE, denoiser, text-encoder, and save dtypes
- adapter and primary trainable
- cache backends
- optimizer, scheduler, and optimization plan
- objective runtime and loss modifier
- trainability flags
- dataloaders and loop coordinates
- accelerator/distributed state
- validation, logging, resource, and runtime-trace state

### Strategy-owned state

The aggregate strategies currently own:

- tokenizer instances and maximum token length
- SD `clip_skip`
- SD3 mask/dropout encoding settings
- SDXL source/checkpoint facts: `load_stable_diffusion_format`,
  `logit_scale`, and `ckpt_info`
- SD3 `_model_version`
- the live-plotter subprocess handle

Some of this is legitimate bound family state. Some is evidence of accidental
ownership:

- SDXL loading writes checkpoint facts onto the strategy so checkpointing can
  read them later.
- SD3 loading writes `_model_version` while also returning the same model
  version for the trainer to store.
- logging code stores and updates a plotter subprocess on every family
  strategy even though it is not model-family behavior.

### Mode-owned state communicated by mutation

`TrainingMode` deliberately receives the whole trainer as its only context.
The current protocol says this keeps signatures stable as the trainer grows.
In practice, mode results are communicated by mutating trainer fields.

For example, fine-tune preparation:

```text
read trainer.loaded_components
→ resolve selected denoiser/text-encoder parameters
→ mutate train flags
→ freeze/unfreeze modules
→ call strategy.post_process_trainable(...)
→ set trainer._primary_trainable
```

Accelerator preparation then replaces trainer component projections and
assigns `_grad_sync_handle`, optimizer, scheduler, and primary trainable.
Adapter mode follows the same general pattern while also constructing and
attaching the adapter.

The resulting runtime conversation is:

```text
strategy loads components
    ↓ returns them
trainer stores components
    ↓ exposes role projections
mode reads and mutates trainer
    ↓ replaces prepared modules
trainer updates component bindings
    ↓ passes projections back
strategy executes batch / validation / sampling / saving
```

That is the concrete "back and forth." Introducing a value merely named
`ModelRuntime` would not fix it. A future design must decide which participant
owns the bound run state and give the other participants narrower operations
or typed results.

## Active Flow Inventory

### Loading

All three strategy loading facets call a lower-level family loader under
`library/models/<family>/loader.py`. The strategy then applies training-run
choices such as RamTorch replacement and attention backend selection before
binding the result to family component specs.

The model loaders themselves also accept accelerator, memory, caching, and
precision configuration. Therefore current loading is already split across:

```text
models loader
  source resolution + module construction + some device/memory policy

strategy loader
  family training integration + additional performance transformations

trainer
  stores and later prepares the returned modules
```

The question is not simply "move loading to models" or "move loading to
strategies." Artifact interpretation and component construction are model
mechanics; selected run policy and contract fulfillment are strategy/training
concerns. The current functions contain both and need responsibility-level
separation.

### Caching

The shared caching phase owns when caching happens, device movement around the
phase, and construction of the data caching engine. Family strategies create
the backend and family-specific model bundle.

Each family `caching.py` then contains both:

- family representation semantics: image-to-latent or text-to-conditioning
  encoding, tensor names, shapes, and cache signatures
- data/storage mechanics: safetensors reads and writes, path suffixes,
  metadata validation, and cache entry behavior

The three files total more than 1,500 lines. This is a genuine split boundary,
not proof that all caching should move out of strategies. Representation
features need to own what is encoded and how it is interpreted. The data
system should probably own reusable persistence and cache lifecycle. The exact
request/result boundary remains to be designed.

### Model preparation

The shared phase:

1. detects an unloaded denoiser role and calls the strategy lazy-load hook
2. asks the mode to prepare trainables
3. calculates shared precision state
4. asks family strategy hooks for casting and component-specific workarounds
5. asks the mode to finalize trainable precision

This flow correctly separates temporal order from many family/mode details,
but it has no typed preparation result. The trainer is the shared mutable
workspace.

### Training step

The trainer owns accumulation timing and calls:

```text
mode.on_step_start(trainer)
strategy.process_batch(
  batch,
  text_encoders,
  denoiser,
  trainable_model,
  vae,
  objective_runtime,
  dtypes,
  accelerator,
  cfg,
  train flags,
  global step,
)
```

Each family `process_batch()` then owns:

- cached-latent selection or VAE encoding
- conditioning resolution
- objective input construction
- family denoiser invocation
- target construction
- differential-output preservation
- base loss selection
- masking and per-sample weighting
- objective-specific loss post-processing
- loss multiplier

It returns `BatchLossOutput`. The trainer then owns:

- objective adaptive-state update
- loss modifiers
- backward
- distributed gradient reduction and clipping
- optimizer and scheduler advancement
- mode post-step behavior
- logging and trigger execution

There is a defensible current boundary—family strategy returns the base
per-sample loss, trainer performs optimization—but the implementation places
generic objective and loss work inside each family. The same DDPM/RF and loss
algorithms are consequently repeated around family-specific representation,
conditioning, and prediction seams.

### Validation

The `process_batch()` contract claims to process training or validation.
Production validation does not call it.

Each family validation facet separately:

1. prepares latents
2. resolves conditioning
3. iterates fixed validation timesteps
4. calls the family diffusion helper to obtain prediction and target
5. calculates L2 loss
6. aggregates and records validation state

This duplicates the broad flow across SD, SDXL, and SD3 and reaches into
family diffusion helpers that are not part of the base contract. The mismatch
is strong evidence that `process_batch()` is not yet the stable shared
execution abstraction.

### Sampling

Shared training orchestration decides when sampling occurs and temporarily
changes train/eval state. It passes the trainer's VAE, tokenizers, text
encoders, denoiser, device, config, epoch, and step to the family strategy.

Sampling is appropriately family-sensitive, but the current signature again
exposes the present latent-diffusion anatomy rather than a stable sampling
request over bound strategy state.

### Checkpointing

Checkpoint timing, retention, naming, metadata assembly, and uploads are
distributed across generic checkpoint helpers, the mode, and family
checkpointing.

There are two noticeably different boundaries:

- `resolve_model_artifact_facts()` takes a narrow typed
  `ModelArtifactResolutionContext` and returns typed facts.
- `save_model_checkpoint()` takes the whole `Trainer` plus save coordinates
  and metadata.

The first is a useful example of a discoverable contract request/result. The
second relies on unrestricted ambient state because family saving needs
loaded modules, source/checkpoint facts, configuration, accelerator unwrap
behavior, and upload settings.

## Placement Inventory

The following classifications are observations and design pressures, not
approved moves.

### Clearly model-owned and already in `models/`

- neural component definitions and forward internals
- family checkpoint conversion and low-level serialization
- artifact/source loading mechanics
- family-declared component identities
- component-local transformations such as VAE padding behavior
- low-level UNet attention implementation replacement

The strategy may select when or which model mechanic to apply. That does not
make the mechanic itself strategy code.

### Clearly strategy-owned and already in `strategies/`

- family assembly for the active training path
- prompt tokenization and text-encoding policy
- construction of family conditioning payloads
- mapping contract prediction inputs to a family forward signature
- family representation/cache tensor semantics
- family validation and sampling behavior
- coordination of family persistence semantics

### Strong placement pressures

| Current location | Item | Pressure |
| --- | --- | --- |
| `models/runtime_utils.py` | `patch_accelerator_for_fp16_training` | Mutates Accelerate training infrastructure; not a model mechanic |
| `models/sdxl/conversion.py` | `get_size_embeddings` | Runtime SDXL micro-conditioning math; not checkpoint conversion |
| family `training.py` | `live_plotter_process` | Logging subprocess state; not family behavior |
| family `caching.py` | safetensors IO/path/validation | Reusable data/cache persistence mixed with family encoding semantics |
| family `diffusion.py` | SD3 VAE encode and shift/scale callbacks | Real representation seam; final ownership between model semantics and a latent-representation feature needs definition |
| `strategies/base/contracts.py` | family-internal tokenization/conditioning/denoiser helpers | Internal feature protocols are conflated with trainer-facing contract |
| `training/modes/base.py` | whole-`Trainer` hook context | Stable signatures are achieved by unrestricted coupling and mutation |
| family checkpointing | whole-`Trainer` save input | Persistence coordination is strategy-owned, but the state boundary is untyped and overly broad |

### Items that should not be moved mechanically

- `replace_unet_modules()` is currently called by strategy loading, but the
  operation changes model internals. The selection policy may move while the
  mechanic remains model-owned.
- CLIP code must be split by meaning. A CLIP module or low-level adapter is a
  model component; tokenization, weighted prompt handling, hidden-state
  selection, and conditioning construction are strategy features.
- Cache code must be split by meaning. Encoding semantics belong with the
  representation/conditioning provider; persistence orchestration belongs
  with the data/cache system.
- SD3 latent normalization may be a property of the model component interface,
  a latent representation feature, or both through a typed model-provided
  codec. Its current duplication between training and validation shows the
  need, but does not decide the final folder.

## Existing Feature And Context Seams

`library/strategies/base/features.py` currently contains two different ideas:

- `WeightedPromptStrategy`: reusable executable behavior implemented by SD and
  SDXL
- `ModelFamilyMetadataStrategy`: a capability seam detected by the trainer and
  currently implemented only by SD3 checkpointing

This validates the intuition that the aggregate contract needs vocabulary
outside its universal ABC list. It also shows why one `features.py` file is not
yet an architecture: it mixes behavior providers and incidental capability
detection without fulfillment rules.

`library/strategies/shared/clip/` is another early reusable feature area. Its
code is shared by current users, but the folder name reports reuse rather than
the concern being fulfilled.

`library/strategies/base/context.py` publishes scoped phase, step, timestep,
batch-size, and sample-index facts around denoiser calls. Its current production
consumer is adapter behavior that needs lower-level forward context without
threading those values through injected module calls.

That is a legitimate scoped publication mechanism. It is not the general
answer to trainer/strategy state ownership: ordinary component state and
lifecycle results should not become ambient context merely to shorten method
signatures.

## Current Compatibility Facts We Can State

The existing families already imply contract constraints that a future
fulfillment model must express.

### Representation and predictor

- SD/SDXL predictors consume their established VAE latent convention.
- SD3 uses its VAE's `process_in` transform before MMDiT prediction.
- Cached latents must carry the same representation semantics expected by live
  encoding.
- Pixel input and latent input are alternative filings of a required
  representation concern, not silently interchangeable tensor sources.

### Conditioning

- SD requires one CLIP-derived conditioning route.
- SDXL requires two encoder routes plus size/crop/target micro-conditioning.
- SD3 can declare absent encoder components, but the chosen conditioning route
  must still produce the payload required by the denoiser.
- Token cache order and names must agree with the selected encoder providers.
- A cache hit must be semantically equivalent to the live conditioning route.

### Objective and prediction

- SD currently fulfills DDPM training.
- SDXL currently branches between DDPM and rectified-flow runtime behavior.
- SD3 currently expects rectified-flow behavior.
- Objective input/target semantics must agree with the predictor's output
  interpretation and the selected representation.

### Training subject

- Fine-tune mode requires selected loaded components with actual `nn.Module`
  instances and parameter selections.
- Adapter mode requires compatible attachment targets and an adapter runtime
  that can participate in family forwards.
- The semantic trainable and the object used for distributed accumulation may
  differ; current code represents them as `_primary_trainable` and
  `_grad_sync_handle`.

### Persistence

- Adapter persistence is mode-owned through the adapter export boundary.
- Full-model persistence requires a family serializer for the selected output
  format.
- Metadata/artifact semantics and physical serialization are related but
  distinct concerns.

These are useful first conformance inputs. They are not yet a complete
compatibility system.

## Smallest Stable Trainer-Facing Shapes Suggested By Current Code

The inventory does not select final APIs, but it shows where typed boundaries
would remove current accidental coupling.

### Loading

Current:

```python
model_version, loaded_components = strategy.load_target_model(
    cfg,
    weight_dtype,
    accelerator,
)
```

Pressure:

```text
typed load request
→ typed bound loading result
  - model/version/source facts
  - declared component bindings
  - deferred-state declarations
  - family persistence facts
```

This would remove the need to hide SDXL checkpoint information and duplicate
SD3 version state on the strategy.

### Preparation

Current:

```text
mode mutates Trainer
strategy preparation hooks mutate modules
Accelerate wrapping replaces Trainer projections
```

Pressure:

```text
typed trainable selection
→ typed preparation plan/result
→ explicit replacement of bound components
```

The exact owner remains open, but preparation needs observable inputs, outputs,
and lifecycle transitions.

### Batch execution

Current:

```python
BatchLossOutput = strategy.process_batch(
    batch,
    text_encoders,
    denoiser,
    trainable_model,
    vae,
    objective_runtime,
    dtypes,
    accelerator,
    cfg,
    flags,
    step,
)
```

Pressure:

```text
bound strategy state + typed step request
→ prediction/objective result with stable semantics
```

The design still needs to decide whether the strategy returns a prediction and
target for shared loss policy, returns a base loss as today, or exposes a
smaller set of staged concern results. The current signature is evidence of
the state problem, not a candidate API to rename.

### Validation, sampling, and saving

These are valid broad trainer-triggered operations, but their requests should
carry coordinates and policy, not repeat the whole component anatomy or grant
unrestricted access to `Trainer`.

## Questions This Inventory Unlocks

The following questions are now sufficiently concrete for design discussion.

1. **What is the trainer-facing contract?** Which broad lifecycle operations
   must the trainer be able to invoke on every fulfilled training strategy?
2. **What are internal feature protocols?** Which current ABCs describe
   collaboration among representation, conditioning, prediction, caching, and
   persistence providers rather than trainer operations?
3. **Who owns bound loaded state?** Does the fulfilled training strategy own
   component bindings, or does the trainer own a typed loaded-run value that
   strategies operate over?
4. **How does mode participate in fulfillment?** Should mode return typed
   trainable selections/preparation results, become a collaborator contained
   by the fulfilled strategy, or remain separately supplied to the trainer?
5. **Where does generic objective/loss work move?** Family strategies must own
   the predictor call and family semantics, but DDPM/RF input construction,
   target construction, and generic loss policy should not be copied without a
   proven family reason.
6. **What is the shared train/validation execution seam?** The current
   `process_batch()` claim is contradicted by production validation.
7. **How are conditional requirements declared?** Lazy loading, caching,
   prompt weighting, FP8 workarounds, and persistence must be required when
   selected and absent otherwise, without placeholder methods that fail late.
8. **How are feature inputs and outputs described?** A useful first version
   needs enough compatibility facts for current families without becoming a
   general dependency graph.
9. **How should caching split?** Representation/conditioning providers own
   codecs; data infrastructure owns persistence. What is the narrow boundary
   between them?
10. **Which generic component roles survive?** `text_encoder`, `vae`, and
    `denoiser` are useful current projections, but the contract must not assume
    every future trainable arrangement has exactly those slots.

## Recommended Next Discussion Order

The inventory suggests the following order before an OpenSpec:

1. define the trainer-facing lifecycle vocabulary
2. define loaded-state ownership and replacement semantics
3. define mode/trainable participation in fulfillment
4. split the current contract methods into public lifecycle operations and
   internal feature protocols
5. define the training/validation prediction-objective-loss seam
6. define conditional fulfillment and first conformance scenarios
7. only then choose package/class/decorator/registration mechanics

This order keeps the feature catalog, Hydra targets, decorators, and explicit
composition as implementation options. It prevents those mechanisms from
deciding the contract before the responsibilities are understood.
