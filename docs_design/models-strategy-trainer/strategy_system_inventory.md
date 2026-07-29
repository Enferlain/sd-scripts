# Model–Strategy–Trainer Production Inventory

## Status And Purpose

This is an evidence-backed inventory of the active production architecture. It
is not a replacement contract, migration plan, or claim that the current
placement is correct.

The inventory supports the exploratory direction in
[`strategy_system_direction.md`](strategy_system_direction.md). Findings are
recorded after each exploration milestone so the evidence and intermediate
conclusions survive context compaction and multi-session work.

The durable tracker for this investigation is bead `sd-scripts-syv`.

## Evidence Standard

The inventory uses the codebase-memory graph at Verify tier and then reads the
material source directly. The graph generation used for the first milestone
was `2026-07-29T00:14:05Z` on branch `adapter-layer`. Coverage checks reported
no recorded indexing gaps in the cited active launcher, strategy factory,
trainer, phase, or mode files. That is a best-effort signal rather than proof
of completeness.

The final placement/contract pass was checked against graph generation
`2026-07-29T00:21:41Z`. Exact-path coverage reported no recorded issues for
the cited active model, strategy, training, config-validation, and strategy
test files. Scope checks found only excluded `__pycache__` directories, not
unindexed source files.

Archived code, deprecated paths, standalone `tools/`, and textual inversion are
outside the active-path inventory unless a current production path still calls
them.

## Milestone Record

| Milestone | State | Recorded result |
| --- | --- | --- |
| 1. Construction and lifecycle | Complete | The launcher constructs one family strategy and one mode, but the trainer remains the runtime integration hub and mutable state owner. |
| 2. Base contract and callers | Complete | The 25-method aggregate mixes 20 pipeline-facing operations with 5 family-internal collaboration seams; conditional support is represented by late `NotImplementedError` defaults. |
| 3. Family flows | Complete | All three families implement the same broad latent-training skeleton, with meaningful differences in payloads, objective compatibility, model invocation, representation transforms, sampling, and persistence. |
| 4. State and placement | Complete | State is split across trainer, strategy, mode, objective, cache backends, and scoped context; several files mix component mechanics, family behavior, and run orchestration. |
| 5. Contract seams | Complete | The current paths expose stable concern-level inputs/results, a partial but scattered compatibility matrix, and concrete conformance scenarios for the next design discussion. |

## Milestone 1: Active Construction And Lifecycle

### Construction

The active launcher performs three independent constructions:

```text
RunConfig
├── build_training_strategy(cfg) ──> one family-selected TrainingStrategy
├── build_training_mode(cfg) ──────> one TrainingMode
└── Trainer(cfg, strategy, mode)
      └── build_objective(cfg) ────> one ObjectiveDefinition
```

Evidence:

- `train.py:18-30` validates the composed config, builds the strategy and mode,
  constructs the trainer, and enters the guarded training call.
- `library/strategies/factory.py:13-34` maps `sd1`, `sd15`, and `sd2` to
  `SdTrainingStrategy`, `sdxl` to `SdxlTrainingStrategy`, and `sd3` to
  `Sd3TrainingStrategy`.
- `library/training/runners/trainer.py:130-150` stores the strategy and mode but
  independently constructs the objective.

The strategy factory therefore already returns one aggregate object. The
architectural question is not whether an aggregate exists; it is what that
aggregate owns and what the trainer may assume about it.

### Top-Level Lifecycle

`Trainer.train()` owns the active temporal sequence:

```text
setup
  → cache
  → prepare models/trainables
  → prepare optimizer/distributed wrappers
  → initialize run state
  → startup sampling/validation
  → training loop
  → final checkpoint/finalization
  → unconditional cleanup and reports
```

Evidence: `library/training/runners/trainer.py:256-301`.

The phase methods do not receive a narrow runtime value. They receive or close
over the entire mutable `Trainer`:

- caching: `library/training/phases/caching.py:27-42`
- model preparation: `library/training/phases/model_prep.py:30-62`
- optimizer/distributed preparation:
  `library/training/phases/optimizer.py:61-185`
- training loop: `library/training/phases/training_loop.py:632-693`

This makes the trainer both the lifecycle coordinator and the shared mutable
state carrier between phases.

### State Created Or Retained By The Trainer

The trainer currently retains all of the following categories
(`library/training/runners/trainer.py:130-254`):

| Category | Representative state |
| --- | --- |
| integration objects | strategy, mode, objective definition/runtime |
| execution infrastructure | accelerator, device, distributed handles |
| loaded model | family-declared `loaded_components`, realization state, VAE/text-encoder/denoiser projections |
| numeric policy | weight/save/VAE/denoiser/text-encoder dtypes |
| data | train/validation manifests, dataloaders, cache backends |
| training subject | adapter, primary trainable, train-denoiser/text-encoder flags |
| optimization | optimizer, optimization plan, scheduler, train/eval callbacks |
| lifecycle | epoch, step, accumulation, resume, progress and validation state |
| observation | metadata, resource monitor, console, observer, loss recorders |

The `loaded_components` tuple is the primary family-declared representation,
but the trainer exposes and mutates role-based projections through
`text_encoders`, `vae`, and `denoiser` properties
(`library/training/runners/trainer.py:1017-1042`).

### Strategy State Used By The Trainer

The trainer also depends on state retained by the strategy:

- `strategy.tokenizers` is copied into `trainer.tokenizers` during setup.
- family loading and checkpoint facts may be retained on the strategy after
  `load_target_model()`.
- `strategy.live_plotter_process` is read and mutated by the training loop.
- family strategy methods collaborate through the aggregate strategy object's
  multiple-inheritance namespace.

The contract and family-state milestones will enumerate these precisely.

### The Current Back-And-Forth

Setup asks the strategy to load the model:

```text
strategy.load_target_model(cfg, weight_dtype, accelerator)
    → (model_version, loaded_components)
trainer stores loaded_components
trainer derives role projections
```

If the denoiser is deferred, model preparation passes trainer-owned state back:

```text
strategy.load_denoiser_lazily(
    cfg,
    trainer.weight_dtype,
    trainer.accelerator,
    trainer.loaded_components,
)
    → replacement loaded_components
trainer stores and re-projects them
```

The training loop then supplies the strategy with the objects it previously
helped load:

```text
strategy.process_batch(
    batch,
    trainer.text_encoders,
    trainer.denoiser,
    trainer.trainable_model,
    trainer.vae,
    trainer.objective_runtime,
    trainer.vae_dtype,
    trainer.weight_dtype,
    trainer.accelerator,
    cfg,
    is_train=True,
    train_text_encoder=...,
    train_denoiser=...,
    global_step=...,
)
```

Evidence:

- load and projection: `library/training/runners/trainer.py:412-423`
- deferred load: `library/training/phases/model_prep.py:37-44`
- batch call: `library/training/phases/training_loop.py:498-516`

This is the concrete meaning of the trainer/strategy state split. It does not
yet prove that a bound strategy should own all loaded state, but it does prove
that the current aggregate strategy is not the sole per-run model integration
object after construction.

### What The Trainer Currently Knows About Model Topology

The generic path directly names and operates on:

- one VAE-like projection;
- an ordered list of text encoders;
- one denoiser-like projection;
- one primary trainable model;
- optional deferred denoiser loading;
- separate train-denoiser and train-text-encoder flags.

This topology appears in setup, caching, precision configuration, training
mode APIs, batch execution, sampling, validation, and checkpointing. The
contract inventory must distinguish between:

1. topology the active pipeline genuinely requires;
2. topology that is only a convenience projection for the current families;
3. topology that belongs behind a family or feature boundary.

### Milestone 1 Conclusion

The active architecture is not simply:

```text
trainer executes one complete strategy
```

It is more accurately:

```text
family strategy ── loads/provides behavior ──┐
training mode ─── selects/wraps trainables ──┤
objective ─────── supplies learning policy ──┼──> mutable Trainer hub
data/optimizer/logging infrastructure ───────┤          │
                                             └──────────┘
                                                  │
                              trainer passes selected state back into each
                              collaborator at lifecycle boundaries
```

That hybrid is the baseline the rest of the inventory must explain before a
replacement contract or ownership model is chosen.

## Milestone 2: Base Contract, Implementations, And Callers

### Aggregate Shape

`TrainingStrategy` inherits eleven concern ABCs
(`library/strategies/base/contracts.py:546-567`):

```text
ModelLoadingStrategy
TokenizationStrategy
TextEncodingStrategy
ConditioningStrategy
CachingStrategy
SampleGenerationStrategy
CheckpointingStrategy
ValidationStrategy
DiffusionTrainingStrategy
DenoiserCallingStrategy
ModelPreparationStrategy
```

Those ABCs declare 25 methods/properties. Nineteen are abstract. Six are
ordinary methods with defaults that only raise `NotImplementedError`:

- `load_denoiser_lazily`
- `get_token_cache_encoder_names`
- `build_te_cache_model_bundle`
- `save_model_checkpoint`
- `prepare_text_encoder_grad_ckpt_workaround`
- `prepare_text_encoder_fp8`

The distinction is important: those six are not necessarily optional
behaviors. They become required under selected runtime conditions, but Python
ABC construction does not prove that the active strategy supports them.

### Complete Method And Caller Classification

“Pipeline-facing” means a production caller outside the family strategy
implementation invokes the method. “Family-internal” means the method is on the
base aggregate only so family facets can collaborate through `self`.

| Contract | Method | Production caller | Present role |
| --- | --- | --- | --- |
| loading | `load_target_model` | `Trainer.setup` | pipeline-facing lifecycle operation |
| loading | `load_denoiser_lazily` | model-preparation phase when `trainer.denoiser is None` | conditionally pipeline-facing lifecycle operation |
| tokenization | `tokenizers` | `Trainer.setup` copies it into trainer state | pipeline-facing state exposure |
| tokenization | `tokenize` | SD/SDXL conditioning facets; no generic trainer call | family-internal feature collaboration |
| tokenization | `tokenize_captions` | epoch token-cache preparation | pipeline-facing cache codec behavior |
| text encoding | `encode_tokens` | SD/SDXL conditioning facets; SD3 has a typed variant used by its encoding logic | family-internal feature collaboration |
| text encoding | `encode_te_outputs_in_memory` | text-encoder caching phase | pipeline-facing cache codec behavior |
| text encoding | `get_models_for_text_encoding` | each family's conditioning facet | family-internal component-selection collaboration |
| conditioning | `resolve_conditioning` | each family's diffusion and validation facets | family-internal execution collaboration |
| caching | `create_latent_caching_strategy` | latent-caching phase | pipeline-facing backend construction |
| caching | `create_te_caching_strategy` | text-encoder caching phase | pipeline-facing backend construction |
| caching | `get_token_cache_encoder_names` | epoch token-cache preparation | pipeline-facing cache schema behavior |
| caching | `build_te_cache_model_bundle` | text-encoder caching phase | pipeline-facing family bundle construction |
| sampling | `sample_images` | sampling/validation orchestration and epoch finalization | pipeline-facing lifecycle operation |
| checkpointing | `resolve_model_artifact_facts` | trainer checkpoint-metadata builder | pipeline-facing artifact semantics |
| checkpointing | `save_model_checkpoint` | fine-tune mode | conditionally pipeline-facing persistence operation |
| validation | `calculate_val_loss` | sampling/validation orchestration | pipeline-facing lifecycle operation |
| diffusion | `process_batch` | training loop | pipeline-facing step operation |
| denoiser | `call_denoiser` | each family's diffusion facet | family-internal model invocation seam |
| preparation | `cast_text_encoder` | shared precision preparation | pipeline-facing policy query |
| preparation | `cast_vae` | trainer setup | pipeline-facing policy query |
| preparation | `cast_denoiser` | shared precision preparation and adapter mode | pipeline-facing policy query |
| preparation | `prepare_text_encoder_grad_ckpt_workaround` | optimizer/preparation phase | conditionally pipeline-facing component workaround |
| preparation | `prepare_text_encoder_fp8` | shared precision preparation | conditionally pipeline-facing component workaround |
| preparation | `post_process_trainable` | adapter and fine-tune modes | pipeline-facing preparation hook |

Caller evidence:

- trainer setup and artifact resolution:
  `library/training/runners/trainer.py:323,419,424,565`
- caching: `library/training/phases/caching.py:59,115-121,154`
- model preparation:
  `library/training/phases/model_prep.py:37-44,99-109`
- optimizer preparation: `library/training/phases/optimizer.py:208`
- epoch token caching, sampling, and training:
  `library/training/phases/training_loop.py:327-334,381,506`
- sampling/validation orchestration:
  `library/training/phases/orchestration_helpers.py:76-110`
- mode calls:
  `library/training/modes/adapter_mode.py:167,273` and
  `library/training/modes/finetune_mode.py:155,395`
- internal conditioning/encoding:
  `library/strategies/sd/conditioning.py:23-68`,
  `library/strategies/sdxl/conditioning.py:49-123`, and
  `library/strategies/sd3/conditioning.py:30-99`
- internal denoiser calls:
  `library/strategies/sd/diffusion.py:24-106`,
  `library/strategies/sdxl/diffusion.py:33-140`, and
  `library/strategies/sd3/diffusion.py:39-119`

### Family Implementation Matrix

All three aggregate family classes are assembled through broad multiple
inheritance:

- SD: 11 family concern facets + `WeightedPromptStrategy` +
  `TrainingStrategy` (`library/strategies/sd/training.py:19-50`)
- SDXL: the same shape
  (`library/strategies/sdxl/training.py:20-58`)
- SD3: 11 family concern facets + `TrainingStrategy`, without the weighted
  prompt feature (`library/strategies/sd3/training.py:17-63`)

For the 25 base methods:

| Family | Concrete methods | Inherited raising defaults |
| --- | --- | --- |
| SD | 23 | lazy denoiser load; full-model save |
| SDXL | 24 | lazy denoiser load |
| SD3 | 24 | lazy denoiser load |

No active family overrides `load_denoiser_lazily`, even though generic model
preparation invokes it whenever initial loading leaves the denoiser projection
empty. SD does not override `save_model_checkpoint`, while fine-tune mode calls
that method for full-model persistence.

These are not automatically active bugs: configuration or current loaders may
make the unsupported branches unreachable for a particular family. They are
contract gaps because support is established by runtime path assumptions
rather than by fulfillment-time validation.

### Feature Surface Outside `TrainingStrategy`

`library/strategies/base/features.py` contains two additional ABCs:

- `WeightedPromptStrategy`, implemented by SD and SDXL and consumed internally
  by their conditioning paths;
- `ModelFamilyMetadataStrategy`, currently implemented by the SD3
  checkpointing facet and detected by metadata integration.

This is already a second contract vocabulary with different discovery rules:
`TrainingStrategy` uses aggregate inheritance, while metadata checks a
feature/capability type. The future feature catalog does not need to be
invented from nothing, but the existing split has not yet been given one
consistent fulfillment model.

### Structural Problems Demonstrated By The Inventory

1. **Public and private contracts are collapsed.** Five base methods are not
   trainer-facing at all. They exist so independent mixins can call one another
   through the final aggregate namespace.
2. **Conditional requirements are late failures.** A strategy can construct
   successfully while selected lazy-load, save, cache, gradient-checkpoint, or
   FP8 behavior still resolves to a raising default.
3. **Method presence says little about meaning.** Most signatures accept
   `Any`, broad `cfg`, positional component lists, or the whole `Trainer`.
4. **Family implementations satisfy names more strongly than semantics.**
   Conformance is mainly tested family by family; the contract itself does not
   express compatible representations, component roles, objective
   expectations, or persistence choices.
5. **The aggregate object is an internal service locator.** Conditioning finds
   tokenization and encoding through `self`; diffusion finds conditioning and
   denoiser invocation through `self`. Multiple inheritance supplies those
   collaborators implicitly.

### Milestone 2 Conclusion

The immediate contract problem is not solved by moving arbitrary methods into
an “optional” list. The current surface needs at least three explicit concepts:

```text
trainer-facing lifecycle/step contract
selected conditional requirements validated before execution
family-internal feature composition that is not exposed as trainer contract
```

The family-flow inventory must now determine which of those methods share real
semantics and which only share historical names.

## Milestone 3: SD, SDXL, And SD3 Production Flows

### Family Component Surfaces

All active families file a `tuple[LoadedModelComponent, ...]`, then the trainer
projects family-declared roles:

| Family | Loaded top-level surface |
| --- | --- |
| SD1/SD2 | CLIP text encoder, VAE, UNet denoiser |
| SDXL | CLIP-L, CLIP-G, VAE, UNet denoiser |
| SD3 | CLIP-L, CLIP-G, T5-XXL, VAE, MMDiT denoiser |

Evidence:

- SD loader: `library/strategies/sd/loading.py:25-68`
- SDXL loader: `library/strategies/sdxl/loading.py:24-92`
- SD3 loader: `library/strategies/sd3/loading.py:22-72`

The shared role vocabulary is useful, but current trainer projections still
assume one VAE, one denoiser, and a flat ordered text-encoder list.

### Common Training-Step Skeleton

The three `process_batch()` implementations follow the same responsibility
sequence:

```text
batch
  → cached latent or live VAE encode
  → cached conditioning, cached tokens, or live text encoding
  → objective-specific noisy input / timestep / target state
  → family-specific denoiser call
  → optional differential-output preservation pass
  → configured pointwise loss
  → optional objective weighting and masks
  → per-sample reduction and dataset weights
  → objective-specific post-processing
  → BatchLossOutput
```

Evidence:

- SD: `library/strategies/sd/diffusion.py:109-198`
- SDXL: `library/strategies/sdxl/diffusion.py:143-242`
- SD3: `library/strategies/sd3/diffusion.py:122-213`

The family differences inside that shared skeleton are real:

| Concern | SD | SDXL | SD3 |
| --- | --- | --- | --- |
| representation | AutoencoderKL latent, SD scale | AutoencoderKL latent, SDXL scale | SDVAE 16-channel latent with `process_in` transform |
| conditioning | one CLIP hidden state | CLIP-L/G hidden states + pooled output + size/crop conditioning | typed CLIP-L/G/T5 conditioning + pooled output |
| objective runtime | DDPM | DDPM or rectified flow | rectified flow |
| predictor | conditional UNet | UNet with text and vector/micro-conditioning | MMDiT with context and pooled vector |
| objective loss shaping | DDPM post-processing | DDPM post-processing only on DDPM path | RF loss weighting |
| weighted prompts | supported | supported | rejected |

The correct future seam is therefore not “all families execute identical
code.” It is a shared training concern whose selected representation,
conditioning, objective, predictor invocation, and loss semantics must be
compatible and produce one stable step result.

### Conditioning Flow

All three conditioning facets implement the same decision:

```text
accepted cached encoder outputs available and no encoder is trained?
├── yes → use cached conditioning
└── no  → use cached token IDs or tokenize captions
          → run selected text encoders
          → merge live values over any retained cached values
```

Evidence:

- SD: `library/strategies/sd/conditioning.py:11-91`
- SDXL: `library/strategies/sdxl/conditioning.py:33-129`
- SD3: `library/strategies/sd3/conditioning.py:12-107`

The output type is where the implementations diverge:

- SD returns a one-element tensor list.
- SDXL returns a positional three-tensor tuple.
- SD3 uses `Sd3TextConditioning`, a typed payload with conversion, selection,
  merge, device, and cache projection behavior
  (`library/strategies/sd3/encoding.py:92-213`).

SD3 demonstrates that the contract does not need to force every family into
the same positional tensor container. It does need a stable concern-level
meaning and explicit compatibility with the selected predictor.

### Denoiser Invocation

All three denoiser facets:

- publish scoped phase/step/timestep context;
- support indexed no-adapter passes for differential-output preservation;
- control gradient enablement;
- enter accelerator autocast;
- adapt the contract-level prediction inputs to the model's forward signature.

The final calls differ:

```text
SD:   denoiser(latents, timesteps, clip_hidden).sample
SDXL: denoiser(latents, timesteps, joined_hidden, pooled_plus_size)
SD3:  denoiser(latents, timesteps, context=clip_t5, y=pooled)
```

Evidence:

- `library/strategies/sd/denoiser.py:10-67`
- `library/strategies/sdxl/denoiser.py:12-105`
- `library/strategies/sd3/denoiser.py:11-68`

This is a strong strategy-owned boundary: arbitrary model forward signatures
are normalized into the predictor behavior the training concern expects.

### Validation Flow

Every family has a separate `process_val_batch()` that repeats:

```text
prepare latents
→ resolve conditioning
→ iterate configured fixed timesteps
→ reuse family get_noise_pred_and_target()
→ force L2 loss
→ average timestep losses
```

Each `calculate_val_loss()` then repeats seeded RNG switching, cyclic
dataloader advancement, progress reporting, accumulation, recorder updates,
and RNG restoration.

Evidence:

- SD: `library/strategies/sd/validation.py:18-145`
- SDXL: `library/strategies/sdxl/validation.py:18-176`
- SD3: `library/strategies/sd3/validation.py:19-155`

The base `process_batch()` advertises training or validation through
`is_train`, but production validation does not call it. This confirms that the
current batch contract and the actual reusable execution seam are misaligned.
The truly shared family-local seam is currently the undeclared
`get_noise_pred_and_target()` method mixed into each validation facet through
the aggregate class.

### Cache Flow

The shared caching phase owns lifecycle and dataset traversal. Each family
facet constructs:

- a latent `CacheBackend`;
- a text-encoder-output `CacheBackend`;
- token-cache encoder names;
- a positional model bundle for text encoding.

The three strategy modules also contain the full cache backend
implementations: preprocessing, VAE/text encoding, safetensors schemas,
validity checks, load/save behavior, and family conditioning reconstruction.

Evidence:

- SD: `library/strategies/sd/caching.py`
- SDXL: `library/strategies/sdxl/caching.py`
- SD3: `library/strategies/sd3/caching.py`

There is broad duplication in image preprocessing, safetensors IO, flip
handling, metadata fields, validity checks, and text-cache mechanics. The
family-specific invariants are narrower:

- latent channel count and VAE encoding/scale transform;
- cache suffix and VAE signature;
- SDXL crop/size conditioning reconstruction;
- text encoder/token schema and conditioning payload.

This is evidence for separating representation/cache codecs from data-owned
cache orchestration. It is not yet evidence for putting all cache code in one
generic backend.

### Sampling Flow

Sampling cadence and prompt-file orchestration are shared through
`sample_images_common()`. Family facets still own:

- model unwrapping and temporary device moves;
- inference pipeline/backend construction;
- objective-compatible sampler behavior;
- conditioning/model invocation;
- VAE decoding and state restoration.

Evidence:

- SD pipeline adapter: `library/strategies/sd/sampling.py:19-87`
- SDXL DDPM/RF selection:
  `library/strategies/sdxl/sampling.py:64-143`
- SD3 RF backend and orchestration:
  `library/strategies/sd3/sampling.py:23-172`

Sampling is a valid conditional training-contract concern, but its current
signature exposes the same fixed component topology and repeats device
lifecycle code across families.

### Persistence Flow

Artifact semantic resolution is implemented by all three families.
Full-model serialization differs:

- SD inherits the raising default.
- SDXL supports stable-diffusion and Diffusers output and retains
  `ckpt_info`/`logit_scale` on the strategy after loading.
- SD3 supports safetensors output through its family conversion helper.

Evidence:

- SD semantics only: `library/strategies/sd/checkpointing.py:15-57`
- SDXL semantics and saving:
  `library/strategies/sdxl/checkpointing.py:17-124`
- SD3 semantics, family metadata, and saving:
  `library/strategies/sd3/checkpointing.py:24-128`

Both concrete save implementations accept the entire trainer and then unpack
family components by role and positional index. Persistence is therefore
family behavior, but its current boundary is coupled to trainer state rather
than a typed artifact request plus a bound family/model state.

### Model Preparation Flow

All families say yes to casting the VAE, text encoders, and denoiser. SD and
SDXL share CLIP FP8 and gradient-checkpoint workarounds. SD3 applies those to
its first two encoders and separately handles T5, including an explicit FP8
unsupported path. Family post-processing ranges from no-op (SD), to freezing
the unstable tail of SDXL TE1, to recording SD3 train flags and rejecting an
incompatible T5/cache selection.

Evidence:

- `library/strategies/sd/model_preparation.py`
- `library/strategies/sdxl/model_preparation.py`
- `library/strategies/sd3/model_preparation.py`
- shared CLIP behavior:
  `library/strategies/shared/clip/model_preparation.py`

This area already contains both reusable component behavior and
family/run-compatibility checks. Those are distinct roles even though they
currently share one facet.

### Milestone 3 Conclusion

The inventory supports a contract vocabulary based on concerns and compatible
filings:

```text
representation
conditioning
objective/corruption and target semantics
predictor invocation
loss construction
validation evaluation policy
cache codecs
sampling
persistence
preparation/runtime compatibility
```

It does not support treating SD, SDXL, and SD3 as wholly independent training
pipelines. Nor does it support erasing their typed differences behind one
hard-coded tensor tuple. Milestone 4 determines which existing code owns each
part of those concerns today and where that ownership is misplaced or
ambiguous.

## Milestone 4: State Ownership And Placement

### Existing Boundary Decisions

The implemented model-layer README already establishes the baseline this
inventory must respect:

```text
models     own component definitions, forward internals, conversion,
           component loading/saving, and low-level component quirks

strategies own how the repository uses a model family: tokenization,
           encoding assembly, conditioning, caching semantics, and
           predictor-call shaping

training   owns temporal lifecycle and infrastructure policy
```

It also establishes `LoadedModelComponent` as the model-owned top-level
component surface and tells generic callers to use declared roles and
capabilities instead of family-native names
(`library/models/README.md:6-37,39-96`).

The inventory therefore does not treat every family-specific function as
model code or every training-time function as strategy code. The deciding
question is whether the code defines a component mechanic, fulfills
family/feature behavior, or coordinates a run.

### Current State Ownership

State is not owned by one fulfilled per-run integration today:

| Owner | State retained today | Consequence |
| --- | --- | --- |
| `Trainer` | loaded components and role projections, model version, dtypes, manifests/caches, trainable subject, objective runtime, optimizer, lifecycle, observation | it is both temporal coordinator and general service locator |
| aggregate family strategy | tokenizers and maximum lengths, family encoding options, loading/checkpoint facts, SD3 preparation flags, live-plotter process | behavior and run-derived state are mixed; later facets depend on earlier calls mutating `self` |
| training mode | selected component keys, text-encoder train flags, selected parameter maps | trainable selection is mode-owned, while selected modules remain trainer-owned |
| objective runtime | corruption/timestep runtime, target-related state, loss modifier | built separately by the trainer and passed into family step/validation calls |
| cache backend instances | cache schema, dtype, VAE signatures, family encoding options | appropriately instance-local, but storage mechanics and family codecs share the same family files |
| scoped strategy context | phase, family, step, timesteps, sample indices, batch size during a predictor call | a legitimate ambient adapter seam, not a replacement for ordinary run-state transfer |

Evidence:

- trainer state and projection replacement:
  `library/training/runners/trainer.py:130-254,967-1042`
- mode-owned fine-tune selection:
  `library/training/modes/finetune_mode.py:83-116`
- strategy construction/loading state:
  `library/strategies/sd/training.py:34-50`,
  `library/strategies/sdxl/training.py:38-58`,
  `library/strategies/sdxl/loading.py:24-95`, and
  `library/strategies/sd3/training.py:36-63`
- SD3 preparation writes strategy state after trainable selection:
  `library/strategies/sd3/model_preparation.py:14-71`
- scoped forward context:
  `library/strategies/base/context.py`

`live_plotter_process` is especially revealing. It is not part of the base
contract, but the generic training loop reads and mutates it directly
(`library/training/phases/training_loop.py:195-201`). The aggregate therefore
has an undocumented observation responsibility in addition to its declared
family responsibilities.

### Model Loading Is Split Across Two Policy Layers

The family strategy loaders call model-layer loaders, then apply further
run-specific choices:

```text
strategy loading
  → call models/<family>/loader.py
  → optionally replace modules with RamTorch
  → select attention implementations / VAE attention behavior
  → retain family loading facts
  → bind modules to LoadedModelComponent declarations
```

The model-layer loaders do not only materialize components. They also:

- sequence loading across accelerator processes;
- inspect memory, caching, and precision configuration;
- choose load devices and move modules;
- clean accelerator-device memory;
- apply SD3 block swapping and FP8 policy.

Evidence:

- model loaders:
  `library/models/sd/loader.py:24-146`,
  `library/models/sdxl/loader.py:21-179`, and
  `library/models/sd3/loader.py:340-481`
- strategy wrappers:
  `library/strategies/sd/loading.py:25-61`,
  `library/strategies/sdxl/loading.py:24-95`, and
  `library/strategies/sd3/loading.py:22-69`

The correct conclusion is not “move all loaders.” Their component
materialization, checkpoint conversion, source fallback, and component-local
quirks belong in the model layer. Distributed sequencing, selected precision
policy, performance substitutions, and fulfillment of the run's declared
component surface are separate concerns. The current files interleave those
roles and return family-specific positional tuples before the strategy
normalizes them.

### Placement Audit

The following table records responsibility pressure, not a migration plan:

| Current location | Current responsibility | Inventory classification |
| --- | --- | --- |
| `library/models/components.py` | family declarations, live component binding, role/capability queries and replacement | correctly model-owned repo component contract |
| family model modules and conversion files | module definitions, forward internals, checkpoint conversion | correctly model-owned component mechanics |
| family model loaders | materialization plus distributed/memory/precision orchestration | mixed: retain materialization mechanics; separate run policy and coordination |
| `library/models/runtime_utils.py` | UNet attention mutation, VAE padding mutation, and an Accelerate FP16 optimizer patch | mixed: component mutation belongs with component adapters; the Accelerate patch belongs to training/performance infrastructure |
| `library/models/sdxl/conversion.py:get_size_embeddings` | builds runtime SDXL micro-conditioning used by strategy and inference pipeline | wrong subdomain: it is not checkpoint conversion; exact home is the SDXL conditioning/predictor adapter boundary |
| `library/models/sd/tokenizer.py` | loads/caches an upstream tokenizer object | component acquisition helper, not family tokenization behavior; current narrow content can remain model/component-adapter owned |
| `library/models/parameter_dump.py` | renders component/module/parameter inspection text | outside the active training path and only called by standalone tooling; do not let it define the model or metadata contracts |
| `library/training/diffusion.py:prepare_latents` | chooses cached/live representation, runs VAE encoding, applies family transform callbacks | representation-feature behavior currently placed under generic training |
| family diffusion facets | representation, conditioning, objective, predictor, target, loss, and masks in one `process_batch` | valid strategy territory, but too many contract concerns collapsed into one operation |
| family validation facets | family prediction plus RNG, dataloader, progress, recorder, and aggregation orchestration | mixed: prediction/evaluation adapter is strategy behavior; run traversal and reporting are trainer responsibilities |
| family caching facets/files | family cache schemas/codecs plus preprocessing, safetensors IO, and backend mechanics | mixed: representation/conditioning codecs are strategy behavior; traversal and general persistence are data infrastructure |
| family sampling facets | predictor/pipeline construction plus device movement and shared run restoration | mixed: family generation behavior is strategy-owned; cadence and generic execution lifecycle are trainer/inference infrastructure |
| family checkpoint facets | family artifact semantics and serialization, but save accepts the entire trainer and may coordinate upload | mixed: serialization is family behavior; save coordinates and publishing infrastructure need narrow external inputs |
| `TrainingMode` and implementations | trainable selection and mode behavior through whole-`Trainer` arguments | correct axis, over-coupled state access |
| training phases | temporal policy, but with VAE/text-encoder/denoiser projections and family method calls embedded throughout | correct lifecycle owner, over-aware of the current latent-family topology |
| `config_validation.py` | some model/objective/cache/mode compatibility rules through family-name branches | necessary checks exist, but contract fulfillment knowledge is scattered outside the objects that claim support |

Two precise examples establish the classification:

1. `patch_accelerator_for_fp16_training()` is defined in
   `library/models/runtime_utils.py` but its active production caller is the
   optimizer-preparation phase. It mutates Accelerate optimizer behavior, not
   a model component.
2. `prepare_latents()` is defined under training but every active production
   caller is an SD/SDXL/SD3 diffusion or validation facet. Its callback seams
   select SD/SDXL scaling or SD3 VAE transforms, which is exactly
   representation behavior.

### What Is Missing From `models/`

There is no large body of neural architecture code stranded in strategies.
The main model-side deficit is a clean, typed component-adapter boundary:

- model loaders return family-specific tuples rather than a typed
  materialization result;
- low-level component workarounds are selected through strategy hooks whose
  arguments are untyped modules and positional indices;
- forward adapters rely on functions whose placement does not advertise
  whether they are component mechanics or strategy behavior;
- third-party components such as CLIP are loaded as external objects without
  one explicit repo-level component adapter vocabulary.

That does **not** mean CLIP or every third-party component should now be
vendored. It means future contract work must distinguish:

```text
component implementation or adapter
    what the component is, how it loads, and low-level quirks

strategy feature
    why this run uses it, how outputs are assembled, and what concern it fulfills
```

### What Is Missing From `strategies/`

Several strategy-owned meanings exist only as callbacks, casts, or conventions:

- representation choice and its tensor semantics;
- the conditioning payload accepted by a predictor;
- objective support claimed by a family execution path;
- cache codec/schema support;
- persistence formats;
- preparation and precision constraints;
- component-adapter requirements.

The implementations contain those facts, but the aggregate does not publish
them as fulfillment information. As a result, config validation, phases, and
individual methods each rediscover part of the compatibility matrix.

### Root Cause Of The Trainer/Strategy Back-And-Forth

The back-and-forth is not caused merely by long method signatures. There is no
single typed value that represents:

```text
the selected and validated training concerns
+ the bound loaded component surface
+ the prepared/replaced live modules
+ the family facts needed by later execution and persistence
```

The trainer therefore acts as that value implicitly, while the strategy keeps
the pieces that its mixins need internally. Whole-`Trainer` mode/save APIs and
large positional strategy APIs are two manifestations of the same missing
boundary.

### Milestone 4 Conclusion

The existing three-way principle remains sound:

```text
models define component mechanics
strategies fulfill model/use behavior
trainer executes temporal and infrastructure policy
```

The problem is that today's files do not consistently stop at those
boundaries, and no explicit per-run state/fulfillment surface joins them.
Placement changes should follow that surface design; they should not precede
it as a file-shuffling exercise.

## Milestone 5: Contract Seams, Compatibility, And Conformance Gaps

### Stable Concern-Level Seams Visible Today

The current system already exposes several meanings stable enough to preserve,
even if their signatures and owners change:

| Concern boundary | Input meaning | Result meaning today | Pressure |
| --- | --- | --- | --- |
| family selection | run config | family aggregate strategy | factory is family-name based and proves no selected compatibility |
| load/bind | source, dtype/device/performance choices | model version plus ordered loaded components | result omits load/provenance facts and later strategy state |
| component access | role/capability/key query | ordered live component handles | strongest existing topology-neutral surface |
| prepare/select | mode, precision and loaded components | mutated modules plus selected trainable state | spread across trainer, mode, strategy, and accelerator wrapping |
| training step | batch, bound execution state, step coordinates | `BatchLossOutput` with per-sample loss/timesteps | current best trainer-facing result, but input is fourteen positional concerns |
| validation | evaluation request and bound state | current/average loss tuple plus recorder side effects | family prediction and trainer traversal are collapsed |
| sample | generation request and bound state | files/logging side effects, no typed result | cadence and family generation share one call |
| persist | artifact request, coordinates, bound state | checkpoint side effects, no typed artifact result | family serialization receives whole trainer |

This is an inventory of semantic seams, not a proposed final method list.

### Compatibility Rules That Already Exist

Compatibility is not wholly absent. It is currently distributed:

| Selection | Current rule | Where enforced |
| --- | --- | --- |
| SD1/SD2 objective | DDPM only | config validation |
| SDXL objective | DDPM or rectified flow | SDXL diffusion/sampling branches plus generic prediction validation |
| SD3 objective | rectified flow only | config validation and SD3 runtime casts |
| prediction | epsilon/v-prediction for DDPM; flow for RF | config validation |
| TE-output cache with TE training | incompatible | config validation |
| TE offload with TE training/cache | incompatible | config validation |
| SD3 T5 training with TE-output cache | incompatible | SD3 post-preparation check, overlapping the generic rule |
| weighted captions | SD/SDXL implement; SD3 rejects | capability inheritance for SD/SDXL, late runtime rejection for SD3 |
| full-model saving | SDXL and SD3 implement; SD does not | late `NotImplementedError` if the fine-tune save path selects it |
| deferred denoiser | component surface can represent absence | generic phase calls a method that no active family implements |
| TE FP8/gradient-checkpoint workarounds | family/component dependent | preparation hooks, some only failing when selected |
| sampler/objective pairing | family dependent | family sampling branches |

Evidence for the model/objective and training/cache rules:
`library/config/config_validation.py:574-643`. The remaining rules are in the
family files inventoried above.

The current matrix proves two things:

1. compatibility is a real part of the contract rather than speculative future
   machinery;
2. it is not yet one fulfillment result that can answer “is this concrete run
   executable?” before lifecycle work begins.

### Fulfillment Gaps

A constructed `TrainingStrategy` currently proves method presence, not that the
selected run is fulfillable. The main gaps are:

1. **No selected-concern declaration.** Callers infer support from family name,
   method inheritance, and config branches.
2. **No compatibility result.** Representation, conditioning, predictor,
   objective, cache, precision, mode, and persistence compatibility are not
   validated together.
3. **No lifecycle state contract.** It is not explicit when components are
   loaded, prepared, wrapped, replaced, finalized, or safe to save/query.
4. **No typed failure boundary.** Unsupported selections can survive until a
   batch, preparation hook, lazy load, sample, or save.
5. **No shared conformance suite.** Current tests strongly cover family
   behavior and current facet composition, but some tests assert the split
   module/inheritance arrangement itself. They do not make every family run the
   same applicable fulfillment scenarios.
6. **No explicit contract evolution rule.** ABC changes can silently turn into
   more late defaults or require every family at once; external integrations
   would need a declared compatibility/version policy.

Representative current tests:

- base/facet inheritance:
  `tests/unit/strategies/test_strategies_base.py:262-319`
- SD and SDXL module-placement composition:
  `tests/unit/strategies/test_strategies_sd.py:258-263` and
  `tests/unit/strategies/test_strategies_sdxl.py:607-612`
- typed component loading surface:
  `tests/unit/strategies/test_model_loading_baseline.py`

These tests remain useful baselines. The future conformance layer should test
semantic filings rather than require one permanent inheritance/file layout.

### Conformance Scenarios The Next Design Must Support

The inventory suggests writing scenarios around selected concerns:

1. **Known family baseline**
   Each current family loads, files its component surface, fulfills its
   configured representation/conditioning/objective/predictor combination,
   prepares, executes one step, validates, and performs every selected cache,
   sample, and persistence action.
2. **Unsupported selection fails before execution**
   Examples include SD full-model save, SD3 weighted captions, incompatible
   objective/prediction, and unavailable precision preparation.
3. **Component replacement remains coherent**
   Accelerator wrapping, adapter attachment, RamTorch substitution, lazy
   loading, or another prepared replacement updates the authoritative bound
   surface seen by steps, metadata, diagnostics, and persistence.
4. **Representation substitution**
   A pixel-space filing does not require a fake VAE; a latent filing requires
   and validates its encoder/decoder, transform, and cache semantics.
5. **Additional trainable component**
   A declared component can be selected and optimized without pretending to be
   the one denoiser or a text encoder.
6. **Adapter plus base model**
   Mode-owned trainable selection can access justified components without
   receiving the whole trainer.
7. **Distillation**
   Teacher and student participants can fulfill distinct predictor/component
   roles within one training strategy instead of being forced into one family
   identity.
8. **Non-image tensor domains**
   Video or audio representations can declare shape/encoding semantics without
   inheriting an image-specific VAE batch assumption.
9. **LLM-containing model**
   An LLM component can be declared, prepared, queried, and used by a
   conditioning/predictor filing without being mislabeled as one entry in the
   current flat text-encoder list.

The current loaded-component contract helps with scenarios 3, 5, and 9.
Current trainer projections and batch APIs still constrain all nine to the
SD-shaped VAE/text-encoder/denoiser interpretation.

### What The Inventory Supports Deciding

The evidence is strong enough to carry these conclusions forward:

- keep the model → strategy → trainer route;
- define contract concerns by stable pipeline meaning, not by predicting every
  future model anatomy;
- require a concrete run to select and fulfill enough concerns to train;
- separate trainer-facing contract operations from family-internal feature
  collaboration;
- validate selected compatibility before expensive lifecycle work;
- give bound loaded/prepared state one authoritative owner or value;
- preserve model-owned component roles/capabilities as the generic topology
  surface;
- move code only after its component/feature/orchestration responsibility is
  named;
- add shared semantic conformance tests without freezing inheritance or file
  placement.

### What Remains Genuinely Open

The inventory does not settle:

- whether a bound fulfilled strategy directly owns live loaded state or holds
  a dedicated typed value owned at the strategy boundary;
- whether mode and objective are contained by the final aggregate or remain
  separate collaborators that jointly fulfill the contract;
- the exact public operation split that replaces or narrows `process_batch`;
- whether feature wiring uses composition, inheritance, decorators, Hydra-like
  import/config selection, or a limited combination;
- the exact cache-codec versus data-persistence package boundary;
- when a third-party component adapter is enough and when repo-hosted
  definitions are warranted;
- the external integration and contract-versioning policy.

Those are now bounded design questions rather than reasons to keep exploring
the same call paths.

### Recommended Discussion Order

Before an OpenSpec or production edits, discuss these in order:

1. name the minimum contract concerns and their trainer-facing result
   semantics;
2. choose the authoritative bound-state/lifecycle ownership model;
3. define how a family integration files selected features and returns one
   compatibility/fulfillment result;
4. decide how mode and objective participate in that fulfillment;
5. write the conformance scenarios in executable terms;
6. only then choose package/class/decorator/config mechanisms and a migration
   sequence.

### Milestone 5 Conclusion

The system does not need a graph or a wholly new abstraction to make progress.
It needs the existing principle to become enforceable:

```text
CONTRACT
  defines accepted concerns, compatibility, lifecycle, and result meanings

STRATEGY FULFILLMENT
  binds family/component behavior to the selected run and rejects gaps

TRAINER
  executes the fulfilled behavior without reconstructing family topology
```

This inventory is now complete enough to start that design discussion. It
does not authorize production moves by itself.
