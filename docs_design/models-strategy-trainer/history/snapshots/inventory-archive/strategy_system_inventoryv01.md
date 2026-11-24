# Strategy System Inventory

## Status And Purpose

This is a code-grounded inventory for the exploratory direction recorded in
`strategy_system_direction.md`. It describes the active model/strategy/trainer
system as it exists now. It is not an implemented redesign, a migration plan,
or a replacement for a future OpenSpec.

The inventory answers four immediate questions:

1. What does the current strategy contract actually require?
2. Who calls each part, and which parts are only family-internal collaboration?
3. Where does loaded state live and move during a run?
4. Which current placements and interfaces reinforce or violate the intended
   model → strategy → trainer boundary?

The findings should be used to decide the next contract vocabulary and the
first bounded migration. They should not be copied into module READMEs as
implemented facts until the corresponding design is accepted and implemented.

## Evidence Scope And Method

The bounded production scope is:

```text
train.py
library/models/
library/strategies/
library/training/runners/
library/training/phases/
library/training/modes/
library/objectives/
directly involved library/data/ cache seams
directly involved config validation
```

The inventory excludes deprecated launchers, textual inversion, standalone
tools, and arbitrary future model families except where an active contract
leaks into those paths.

Codebase Memory generation `2026-07-26T00:54:44Z` was ready in full-index
mode. The relevant source scopes had no recorded source gaps; the reported
excluded subtrees were generated `__pycache__` directories.

Graph evidence was interpreted conservatively:

- low-confidence `suffix_match` `CALLS` edges with multiple candidates were
  treated as provisional;
- `OVERRIDE` relationships were expanded before assessing polymorphic calls;
- exact source call sites were checked when the graph could not resolve the
  active strategy implementation;
- graph absence was not treated as proof that a caller or implementation does
  not exist.

This matters here because the graph currently resolves several
`strategies.<method>(...)` calls to the SD3 implementation alone. Source shows
that those calls dispatch through the active aggregate strategy and may reach
SD, SDXL, or SD3. The graph also does not represent the tokenization facet
methods as overrides of `TokenizationStrategy`, because those implementations
arrive through sibling mixins in the final aggregate class.

An import-time MRO inspection was also attempted. It could not run in the
current no-GPU execution environment because importing the optional RamTorch
integration eagerly called `torch.cuda.current_device()`. The source-level MRO
and ABC relationships remain available, but that import behavior is itself a
small example of runtime integration leaking into module import.

## Active Construction And Execution

The active launcher constructs three independently selected objects:

```text
cfg
├── build_training_strategy(cfg) -> SD / SDXL / SD3 aggregate strategy
├── build_training_mode(cfg)     -> adapter / fine-tune mode
└── Trainer(cfg, strategy, mode)
      └── build_objective(cfg)   -> DDPM / rectified-flow definition
```

The trainer then owns the run sequence:

```text
setup
  → caching
  → model/trainable preparation
  → optimizer and accelerator preparation
  → runtime/metadata initialization
  → startup evaluation
  → epoch/step loop
  → final checkpoint and cleanup
```

The current aggregate strategy is assembled with multiple inheritance:

```text
SdTrainingStrategy
├── SdModelLoadingStrategy
├── SdTokenizeStrategy
├── SdTextEncodingStrategy
├── SdConditioningStrategy
├── SdCachingStrategy
├── SdSampleGenerationStrategy
├── SdCheckpointingStrategy
├── SdValidationStrategy
├── SdDiffusionTrainingStrategy
├── SdDenoiserCallingStrategy
├── SdModelPreparationStrategy
├── WeightedPromptStrategy
└── TrainingStrategy
```

SDXL follows the same shape. SD3 omits `WeightedPromptStrategy` and inherits a
family-metadata feature through `Sd3CheckpointingStrategy`.

The final object therefore serves four roles at once:

1. trainer-facing family facade;
2. namespace for sibling facets to call each other;
3. owner of tokenizer and family-local mutable state;
4. factory for cache backends and sampling/persistence adapters.

## Base Contract Inventory

`TrainingStrategy` combines eleven ABC facets containing 25 methods/properties.
The table classifies what each surface actually does today. “Current
classification” describes behavior, not the final desired owner.

| Contract surface | Verified production caller | Current classification | Main pressure |
| --- | --- | --- | --- |
| `load_target_model` | `Trainer.setup()` | Trainer-facing loading concern | Returns loaded state to the trainer while retaining family save facts on the strategy |
| `load_denoiser_lazily` | `model_prep.prepare_models()` when no denoiser is loaded | Conditional loading concern | No active family overrides it; the default raises |
| `tokenizers` | `Trainer.setup()` | Strategy-owned state projection | Trainer copies the list and later passes it back |
| `tokenize` | Family conditioning; deprecated dataset path | Family-internal/reusable conditioning feature | Required base method despite not being a trainer operation |
| `tokenize_captions` | Per-epoch token-cache preparation | Conditional trainer-facing representation concern | Takes tokenizers already owned by the strategy |
| `encode_tokens` | Family conditioning | Family-internal/reusable conditioning feature | Hidden sibling-facet dependency |
| `encode_te_outputs_in_memory` | Training caching phase | Conditional trainer-facing cache encoding concern | Topology-shaped inputs and cache-specific output dictionary |
| `get_models_for_text_encoding` | Family conditioning | Family-internal preparation helper | Exposes accelerator wrapping details through the base contract |
| `resolve_conditioning` | Family diffusion and validation facets | Family-internal composition seam | Stable responsibility, but payload and dependencies are untyped in the contract |
| `create_latent_caching_strategy` | Training caching phase | Conditional cache backend factory | Returns `Any`; representation semantics and data IO are combined |
| `create_te_caching_strategy` | Training caching phase | Conditional cache backend factory | Returns `Any`/`None`; support is not declared during fulfillment |
| `get_token_cache_encoder_names` | Per-epoch token-cache preparation | Conditional cache schema concern | Non-abstract default raises despite active caller dependence |
| `build_te_cache_model_bundle` | Disk TE caching phase | Family-local cache adapter | Packs arbitrary positional component topology into `Any` |
| `sample_images` | Startup, step, and epoch evaluation paths | Trainer-facing sampling concern | Signature hardcodes VAE/tokenizers/text encoders/one denoiser and repeats trigger policy |
| `resolve_model_artifact_facts` | Trainer checkpoint metadata assembly | Family artifact-semantics concern | Metadata dependency is mandatory even when persistence support differs |
| `save_model_checkpoint` | Fine-tune mode checkpoint routing | Conditional persistence concern | Receives the whole trainer; non-abstract default raises |
| `calculate_val_loss` | Shared sampling/validation orchestration | Trainer-facing validation operation | Owns generic RNG, dataloader iteration, recording, and family execution together |
| `process_batch` | Main training loop | Trainer-facing step operation | Mixes representation, conditioning, objective, denoiser, loss, and mode-specific behavior |
| `call_denoiser` | Family diffusion helpers | Family-internal prediction feature | Stable family boundary, but incorrectly exposed as a universal trainer contract |
| `cast_text_encoder` | Shared precision phase | Preparation policy query | Hardcodes a text-encoder role and reduces policy to a boolean |
| `cast_vae` | Trainer setup | Preparation policy query | Hardcodes a VAE role and controls trainer dtype inference indirectly |
| `cast_denoiser` | Shared precision phase and adapter mode | Preparation policy query | Hardcodes one denoiser role and is queried from multiple owners |
| `prepare_text_encoder_grad_ckpt_workaround` | Optimizer preparation | Component/feature preparation hook | CLIP/T5 component mechanics are elevated to the aggregate contract |
| `prepare_text_encoder_fp8` | Shared precision phase | Component/feature preparation hook | Support can fail only after a particular encoder/dtype is selected |
| `post_process_trainable` | Adapter and fine-tune modes | Family/mode collaboration seam | Meaning changes by family and mode; accepts an arbitrary trainable plus fixed topology |

`ModelConditioning` is also defined in `contracts.py` as a marker base class,
but the contract does not require `resolve_conditioning()` to return it.

### Abstractness does not equal support

Several actively callable methods are intentionally non-abstract even though
their defaults only raise:

```text
load_denoiser_lazily
get_token_cache_encoder_names
build_te_cache_model_bundle
save_model_checkpoint
prepare_text_encoder_grad_ckpt_workaround
prepare_text_encoder_fp8
```

This lets a concrete aggregate strategy be instantiable without proving those
operations work for the selected run. Some active families override all of a
subset, while others do not:

- no active family overrides `load_denoiser_lazily`;
- SD does not override `save_model_checkpoint`;
- all active families currently override token-cache names/bundles and text
  encoder preparation hooks.

The ABC therefore verifies method availability more strongly than run
fulfillment.

## Feature Surface Inventory

`base/features.py` contains two feature ABCs:

| Feature | Implemented by | Current consumer | Observation |
| --- | --- | --- | --- |
| `WeightedPromptStrategy` | SD and SDXL aggregates through sibling tokenization/encoding facets | Family conditioning when weighted captions are configured | A selected feature with a real conditional requirement; SD3 rejects it at runtime |
| `ModelFamilyMetadataStrategy` | SD3 through its checkpointing facet | Trainer model-realization filing via `isinstance()` | A capability probe that changes behavior without being part of the base strategy contract |

These are the clearest current examples of the proposed feature catalog, but
they still depend on aggregate MRO rather than explicit wiring.

`base/context.py` is not general loaded strategy state. It publishes a scoped
denoiser-forward context—phase, family, global step, timesteps, and selected
sample indices—for lower layers such as TLoRA. It should be evaluated as an
execution-context feature, not as the answer to trainer/strategy state
ownership.

## Family Fulfillment Matrix

This matrix records implemented behavior, including late runtime restrictions.

| Concern | SD 1.x/2.x | SDXL | SD3 |
| --- | --- | --- | --- |
| Declared top-level components | 1 text encoder, VAE, UNet | 2 text encoders, VAE, UNet | 3 text encoders, VAE, MMDiT |
| Objective compatibility | DDPM only | DDPM or rectified flow | Rectified flow only |
| Representation | VAE latent | VAE latent | SD3 VAE latent transform |
| Weighted prompts | Implemented | Implemented | Runtime `NotImplementedError` |
| Latent cache backend | Implemented | Implemented | Implemented |
| TE cache backend | Implemented | Implemented | Implemented |
| Per-epoch token cache | Implemented | Implemented | Implemented |
| Validation | Fixed-timestep L2 loop | Fixed-timestep L2 loop | Fixed-timestep L2 loop |
| Sampling | SD pipeline/DDPM scheduler | SDXL DDPM or RF backend | Custom SD3 RF backend |
| Artifact fact roles | Adapter, full model, textual inversion | Adapter, full model, textual inversion | Full model only |
| Full-model save | Base default raises | Stable-format and Diffusers paths | Safetensors only |
| Lazy denoiser load | Not implemented | Not implemented | Not implemented |
| Family-local realization facts | None | None | Attention-mask settings |

The objective matrix is enforced in central config validation through explicit
family-name branches:

```text
SD1/SD1.5/SD2 → DDPM
SD3            → rectified flow
SDXL           → either current objective path
```

It is not declared by the family strategy or checked as part of a unified
fulfillment result.

## Current Flow Inventories

### Model loading and realization filing

```text
strategy constructor
  └── loads/owns tokenizers and family configuration state

Trainer.setup
  ├── copies strategy.tokenizers
  ├── asks strategy.cast_vae()
  ├── strategy.load_target_model(...)
  │    ├── model loader constructs components
  │    ├── strategy applies runtime attention/RamTorch changes
  │    ├── strategy may retain checkpoint-format facts
  │    └── returns model version + LoadedModelComponent tuple
  ├── trainer owns loaded_components
  ├── trainer derives compatibility projections
  └── trainer files the initial model realization once
```

The model layer’s declared component surface is a strong existing building
block. Each family publishes ordered `LoadedModelComponentSpec` entries and
generic callers can filter the bound runtime components by role/capability.

The lifecycle around that surface is incomplete:

- a future lazy loader would replace `loaded_components` after initial
  realization filing;
- `prepare_models()` intentionally does not refile/finalize model realization
  metadata after lazy loading;
- accelerator preparation can replace modules again;
- no explicit current lifecycle distinguishes initially observed, fully
  loaded, accelerator-wrapped, or final component state.

### Training step

All three family paths follow approximately:

```text
trainer loop
  ├── mode.on_step_start(trainer)
  └── strategy.process_batch(all model projections + objective + cfg + flags)
       ├── obtain cached latents or encode images with the VAE
       ├── resolve cached/live conditioning
       ├── objective-specific corruption/timestep state
       ├── family denoiser call
       ├── objective target construction
       ├── optional adapter differential-output preservation
       ├── generic loss selection/masking/sample weighting
       ├── objective-specific loss post-processing
       └── BatchLossOutput

trainer loop
  ├── updates objective adaptive state
  ├── applies post-loss modifier
  ├── backward/clip/optimizer/scheduler
  └── mode.on_step_end(trainer)
```

The boundary has one useful invariant: the strategy returns a base
`BatchLossOutput` before trainer-owned post-loss modifiers and optimization.
The internals do not follow the ownership split implied by that result:

- representation preparation is shared but located in `training/diffusion.py`;
- DDPM/RF state and targets partly live in `objectives/` and partly in each
  family diffusion facet;
- generic conditional loss, masking, sample weights, and loss multipliers are
  repeated in all three family implementations;
- adapter-specific differential-output preservation is embedded in family
  diffusion implementations;
- family prediction invocation is a distinct `call_denoiser()` seam, but it is
  exposed in the base trainer contract.

### Validation

`calculate_val_loss()` owns:

```text
RNG switching
validation dataloader iteration
fixed timestep selection
family batch execution
L2 loss
recorder update
RNG restoration
```

Each family then has a non-contract `process_val_batch()` that repeats latent
preparation and conditioning, and invokes another non-contract
`get_noise_pred_and_target()` method supplied by the family diffusion sibling
mixin.

This creates a hidden dependency:

```text
<Family>ValidationStrategy
  requires <Family>DiffusionTrainingStrategy.get_noise_pred_and_target
  but neither its inheritance nor the base contract declares that requirement
```

It also disproves the `process_batch(is_train=False)` description as the active
shared validation execution path.

The public validation signature has already drifted from its meaning:
`train_dataloader` is typed/documented as a dataloader, but the shared caller
passes `trainer.num_batches_per_epoch`. Current family implementations do not
use that argument.

### Caching

```text
training phase
  ├── decides whether/when to cache
  ├── moves VAE/text encoders
  ├── asks strategy to construct family backend/bundle
  └── invokes data-owned CachingEngine

family CacheBackend
  ├── preprocesses images
  ├── encodes through model components
  ├── defines cache tensor/schema semantics
  ├── validates cache compatibility
  └── reads/writes safetensors
```

This is a legitimate cross-boundary concern, but the current files combine
three ownership categories:

1. representation/conditioning semantics;
2. data preprocessing and persistence;
3. trainer-facing backend construction.

The future contract needs to distinguish those without forcing all of them
into one owner. The inventory does not yet decide the final cache split.

### Sampling

The trainer computes sampling triggers and switches the run to evaluation
mode. Family sampling strategies then:

- check trigger policy again;
- unwrap and move components;
- construct a family inference pipeline/backend;
- run shared prompt/output orchestration;
- restore devices/dtypes and clean memory.

The contract hardcodes one VAE, tokenizer list, text-encoder list, and denoiser.
The active call sites also violate that contract for single-encoder families:
they pass `trainer._text_encoder`, which is a single module for SD and a list
for multi-encoder families. `SdSampleGenerationStrategy` indexes
`text_encoders[0]` as though it always received a list.

`epoch` is declared as `int` in the strategy contract while step-triggered
sampling legitimately passes `None`.

These are concrete examples of compatibility projections making the boundary
less reliable than the declared loaded-component surface.

### Checkpointing

```text
trainer
  ├── owns timing, naming, retention, and metadata assembly
  └── delegates to mode.save_checkpoint(...)

mode
  ├── routes adapter/sidecar/full-model intent
  └── for full models calls strategy.save_model_checkpoint(trainer=...)

strategy
  ├── reads components/config/accelerator from the whole trainer
  ├── invokes model conversion/serialization
  └── performs Hugging Face upload
```

Family artifact facts and physical persistence support are separate today:

- SD declares full-model artifact semantics but does not implement full-model
  saving through the active strategy contract;
- SD3 implements full-model safetensors facts/saving but rejects adapter
  artifact facts;
- central config validation allows model/mode combinations more broadly than
  those artifact resolvers and savers.

For example, an SD3 adapter run can be configured and prepared, but checkpoint
metadata assembly asks the SD3 resolver for an `adapter` artifact and receives
a late `ValueError`. An SD fine-tune save reaches the non-abstract base
`save_model_checkpoint()` and receives a late `NotImplementedError`.

These combinations demonstrate that the current system constructs a strategy
object before proving the selected run is fulfilled.

## Current State Ownership And Back-And-Forth

### Strategy-owned state

Current aggregate strategies retain:

- tokenizer objects and maximum token lengths;
- SD/SDXL prompt configuration such as `clip_skip`;
- SDXL source-format, logit-scale, and checkpoint-info facts used at save time;
- SD3 attention-mask/dropout settings and model version;
- some trainability observations such as SD3 CLIP/T5 flags;
- miscellaneous family runtime fields such as the live plotter process slot.

### Trainer-owned state

The trainer retains:

- the canonical live `loaded_components` tuple;
- compatibility projections for text encoders, VAE, and denoiser;
- tokenizer copy;
- VAE/base/denoiser/text-encoder dtypes;
- cache backend instances;
- mode-selected trainability flags and the primary trainable;
- objective runtime;
- optimizer, scheduler, and distributed wrappers;
- accepted model-realization identity and metadata runtime access.

### Mode-owned state

Modes retain:

- selected component keys and parameter references;
- text-encoder train flags;
- adapter continuation/runtime selection;
- the rules for accelerator wrapping, train/eval transitions, and checkpoint
  routing.

### Objective-owned state

Objective runtimes retain:

- scheduler/timestep sampling state;
- DDPM alpha/SNR state or RF weighting configuration;
- adaptive timestep updates;
- post-loss modifier runtime.

### Repeated hand-back pattern

The common sequence is:

```text
strategy creates/loads object
  → trainer stores object
  → trainer or mode mutates/wraps object
  → trainer passes object back to strategy method
```

Examples:

- strategy owns tokenizers → trainer copies them → trainer passes them to
  tokenization, cache, and sampling operations;
- strategy loads components → trainer stores/projections them → trainer passes
  VAE/text encoders/denoiser back to process, validation, sampling, and save;
- strategy constructs cache backends → trainer stores them → data engine
  executes them;
- strategy retains SDXL checkpoint facts while trainer owns the modules needed
  to consume those facts.

### Projection mutation defects

`Trainer.text_encoders` returns a newly constructed list projected from
`loaded_components`. Its whole-list setter correctly updates the component
tuple. Mutating one returned list element does not.

Fine-tune accelerator preparation currently performs:

```python
trainer.text_encoders[i] = trainer.accelerator.prepare(t_enc)
```

That assignment mutates a temporary list and does not replace the module in
`trainer.loaded_components`. Tests use a mock trainer with a mutable list, so
they do not exercise the real property semantics.

The separate `_text_encoder` compatibility projection creates the sampling
contract mismatch described above. These are not merely naming issues; they
show that the system has more than one apparent live model surface.

## Placement Inventory

### Code that fits `library/models/`

The following current responsibilities match the model-layer rule:

- UNet, MMDiT, VAE, ControlNet, and related module definitions;
- forward internals and low-level component behavior;
- checkpoint namespace conversion and low-level state-dict serialization;
- family top-level component declarations;
- generic loaded-component binding/filtering by declared roles/capabilities;
- low-level construction of components from model artifacts.

### Model-layer pressure points

Some current model files contain more than component mechanics:

- family `loader.py` entry points coordinate accelerator process ordering,
  low-RAM device placement, config policy, and memory cleanup in addition to
  constructing components;
- `models/runtime_utils.py` is named generic but imports the SD UNet type and
  contains runtime attention selection plus an accelerator-scaler patch;
- `models/sd/tokenizer.py` is a generic Hugging Face cache/bootstrap helper
  whose ownership is not inherently SD component mechanics;
- `models/parameter_dump.py` is repository model-inspection rendering rather
  than a neural component definition.

These do not all require the same destination. They are candidates for
responsibility-level review:

```text
artifact/component construction
runtime/performance preparation
tokenizer/bootstrap infrastructure
model inspection/presentation
```

The inventory supports keeping low-level loading and conversion in models
while questioning orchestration and performance policy mixed into those
functions.

### Code that fits `library/strategies/`

The following responsibilities align with strategy ownership:

- mapping loaded family components onto repository conditioning;
- prompt tokenization/encoding behavior;
- family-specific prediction invocation;
- representation transforms specific to how the repository trains a family;
- family sampling/persistence adapters;
- family artifact semantics;
- known family assembly.

The shared CLIP tokenization and preparation helpers are credible early
feature-catalog examples because they describe reusable repository behavior,
not a new CLIP neural implementation.

### Strategy-layer pressure points

Current family strategy files also own:

- generic loss selection, masking, weighting, and multiplication;
- objective construction steps already partly owned by `objectives/`;
- generic validation dataloader/RNG/recorder orchestration;
- data preprocessing and safetensors cache IO;
- sampling trigger checks and memory cleanup;
- Hugging Face upload policy;
- component preparation details elevated into universal methods;
- metadata-specific optional behavior mixed into checkpointing.

The issue is not that these files are large. The issue is that their reasons to
change cross objective, data, output, performance, metadata, and family
boundaries.

### Training/mode/trainer pressure points

The trainer and modes successfully own temporal and trainable-form policy, but
their current interfaces are broad:

- every mode method receives the whole trainer;
- modes both select trainables and directly replace trainer model projections;
- the trainer knows VAE/text-encoder/denoiser anatomy throughout phases;
- training phases ask multiple fine-grained family policy questions;
- `training/diffusion.py` implements latent representation behavior despite
  living under generic training orchestration;
- validation and sampling operations receive most trainer state positionally
  instead of consuming a stable fulfilled-strategy boundary.

The modes show a useful path forward: adapter and fine-tune selection already
consume `loaded_components` by roles and stable component keys in several
places. The remaining fixed anatomy and whole-trainer mutation should be
evaluated against that stronger model surface.

## Fulfillment Audit

The current run is validated by several independent mechanisms:

```text
Hydra/dataclass schema
  + central config validation
  + strategy ABC instantiability
  + mode runtime checks
  + family runtime checks
  + persistence/metadata checks at save time
```

There is no single accepted fulfillment result.

### What is enforced early

- known family name in the strategy factory;
- known mode and objective names;
- SD/SD3 objective family restrictions;
- objective prediction type;
- several cache/offload/text-encoder-training conflicts;
- adapter/fine-tune config shape.

### What remains implicit or late

- family support for weighted prompts;
- family/mode/artifact-role compatibility;
- full-model serialization support and format;
- component-specific FP8 support;
- lazy loading support;
- conditioning payload compatibility with the predictor;
- representation compatibility with objective/predictor;
- feature dependencies hidden through sibling mixins;
- whether all declared components are present and usable for a selected
  operation.

### Hidden sibling dependencies

Current family implementations rely on methods that are available only because
the final aggregate includes several sibling classes:

```text
conditioning
  → tokenization
  → text encoding
  → weighted prompt feature (SD/SDXL)

diffusion
  → conditioning
  → denoiser calling

validation
  → conditioning
  → family-local diffusion helper

sampling
  → tokenization/encoding helpers and family configuration
```

Those dependencies are not constructor fields, typed collaborator protocols,
or fulfillment declarations. Python MRO makes the methods available, but the
contract cannot explain why the combination is valid.

## Findings Strong Enough For The Next Design Discussion

The inventory supports these conclusions without yet choosing final classes or
packages:

1. **Keep the contract → fulfilled strategy → trainer principle.**
   The current problems are caused by weak fulfillment and state boundaries,
   not by lack of family strategies.
2. **Separate the trainer-facing contract from family-internal feature
   collaboration.**
   Only a minority of the 25 base methods are called by the trainer/phases.
3. **Describe selections and compatibility before runtime execution.**
   An instantiable aggregate is not proof that the selected mode, objective,
   representation, conditioning, precision, cache, and persistence paths work
   together.
4. **Use the declared loaded-component surface as the source model boundary.**
   Compatibility projections currently create real replacement and call-shape
   defects.
5. **Resolve loaded-state lifecycle and mutation explicitly.**
   Initial loading, deferred loading, accelerator replacement, metadata
   observation, and saving currently observe different moments without a
   declared state model.
6. **Do not preserve `process_batch()` merely because it is the current
   caller.**
   Its return boundary may be useful, but its implementation responsibilities
   need to be decomposed against representation, objective, prediction, loss,
   and mode ownership.
7. **Treat validation as shared orchestration plus a family/strategy execution
   concern.**
   The current duplicated validation path and hidden MRO helper dependency are
   not a stable contract.
8. **Treat caching and persistence as cross-boundary designs.**
   Their semantic and IO responsibilities should be separated deliberately,
   not moved wholesale to whichever folder currently has fewer files.

## Questions Now Ready For Focused Discussion

The next discussion can be narrower than the original architecture debate:

1. Which trainer-facing operations are the minimum lifecycle vocabulary?
2. What typed loaded-component/query surface may modes and the trainer use?
3. Which current operations become selected features, and what do they declare
   they require/provide?
4. Where is fulfillment validated and represented before `Trainer.train()`?
5. Does the fulfilled strategy own loaded components, or does it expose
   controlled mutation over a separately owned loaded-component state?
6. What is the smallest typed training-step input/result boundary after
   representation, objective, prediction, and loss ownership are separated?
7. What shared validation execution can use that same boundary?
8. How are component wrapping/replacement and model-realization finalization
   ordered?
9. Which first migration removes a real defect and proves the direction
   without attempting the end-game component catalog?

Those questions are now backed by the active code rather than hypothetical
future models.
