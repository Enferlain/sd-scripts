# Model–Strategy–Trainer Production Inventory

## Status And Purpose

This is an evidence-backed inventory of the active production architecture. It
is not a replacement contract, migration plan, or claim that the current
placement is correct.

The inventory supports the exploratory direction in
[`strategy_system_direction.md`](strategy_system_direction.md). Findings are
recorded after each exploration milestone so the evidence and intermediate
conclusions survive context compaction and multi-session work.

This file is evidentiary rather than normative. When an intermediate
conclusion here conflicts with the later direction document, the direction
document records the current decision while this inventory preserves how the
evidence was obtained.

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
| 6. Mode/objective classification | Complete | `TrainingMode` should dissolve; strategy declares intent while Trainer owns generic mechanics and selected capabilities provide specialized behavior through typed exchanges. |
| 7. Logical identity and routes | Complete | Stable strategy-scoped participant identity is distinct from Python objects; materially different prepared representations use revisioned named routes with explicit freshness. |
| 8. Arrangement transitions | Complete | Declaration, materialization, replacement, route rebinding, relationship transitions, arrangement amendment, and merge/fold have distinct semantics. |
| 9. Artifact persistence | Complete | Persistence selects coherent semantic products through declaration/request/plan/result rather than serializing an incidental live object. |
| 10. Binding authority | Complete | One dedicated contract-owned per-run authority lives behind the complete Trainer-facing strategy boundary; remaining work is concrete exchange design. |

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
| `TrainingMode` and implementations | trainable selection and mode behavior through whole-`Trainer` arguments | historically useful separation, but not a target authority; Milestone 6 classifies its responsibilities for dissolution |
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
   The strategy can declare combined trainable subjects and required
   attachment behavior while Trainer/optimization realizes them without a
   parallel mode object.
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

- whether the active strategy directly owns live loaded state or holds a
  dedicated typed value at the strategy boundary;
- how specialized capabilities divide behavior between Trainer-owned services,
  domain implementations, and strategy-internal features;
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
3. define how a strategy files selected components, features, capabilities,
   and compatibility constraints;
4. classify current mode and objective behavior by semantic owner;
5. write the conformance scenarios in executable terms;
6. only then choose package/class/decorator/config mechanisms and a migration
   sequence.

### Milestone 5 Conclusion

The system does not need a graph or a wholly new abstraction to make progress.
It needs the existing principle to become enforceable:

```text
CONTRACT
  defines accepted concerns, compatibility, lifecycle, and result meanings

STRATEGY
  explicitly defines what is trained and how, then rejects contract gaps

TRAINER
  executes the strategy without reconstructing model topology
```

This inventory is now complete enough to start that design discussion. It
does not authorize production moves by itself.

## Milestone 6: Training Mode And Objective Responsibility Classification

This milestone follows the later direction decision that strategy means the
authored definition of what is being trained and how. It supersedes the
Milestone 4 description of `TrainingMode` as the "correct axis." Separating
adapter and fine-tune behavior from model-family classes was an improvement
over the older combinatorial strategy design, but the current mode object is
not a target parallel authority.

The classification uses five semantic owners:

```text
strategy declaration
  the deliberate choice of training intent, participants, features, and
  capabilities

Trainer mechanism
  generic lifecycle, preparation, optimization, and infrastructure execution

Trainer-recognized capability/domain behavior
  specialized behavior invoked by the training mechanism through an explicit
  contract; its code need not live in the central Trainer class

strategy-internal feature
  reusable behavior used by the strategy to compute its training semantics

model/component behavior
  mechanics inherent to a component rather than to the run lifecycle
```

This distinction avoids two equal mistakes:

- moving every mode method into the strategy and preserving two-way
  orchestration behind another name;
- copying every adapter, distillation, or future technique branch directly
  into one growing `Trainer` class.

The strategy selects what is used. The Trainer executes the mechanism.
Specialized code may remain delegated while still being Trainer-coordinated.

### `TrainingMode` caller surface

The active `TrainingMode` protocol declares fifteen methods plus
`checkpoint_artifact_role` (`library/training/modes/base.py:26-203`). Fourteen
methods are called from production Trainer/phases. Diagnostics is called
indirectly from logging:

| Current method | Active caller |
| --- | --- |
| `prepare_trainables` | `training/phases/model_prep.py:30-62` |
| `configure_trainable_precision` | `training/phases/model_prep.py:30-62` |
| `build_optimizer_params` | `training/phases/optimizer.py:61-185` |
| `prepare_with_accelerator` | `training/phases/optimizer.py:61-185` |
| `setup_gradient_training` | `training/phases/optimizer.py:188-221` |
| `register_state_hooks` | `training/phases/optimizer.py:61-185` |
| `on_epoch_start` | `training/phases/training_loop.py:632-693` |
| `on_step_start`, `on_step_end`, `get_trainable_params` | `training/phases/training_loop.py:461-618` |
| `set_eval`, `set_train` | `training/phases/training_loop.py:341-394` |
| `checkpoint_artifact_role`, `resolve_checkpoint_artifact_format` | `training/runners/trainer.py:529-576` |
| `save_checkpoint` | `training/runners/trainer.py:471-507` |
| `get_diagnostics_components` | `logging/summaries.py:300-313` |

The whole-`Trainer` parameter did not keep these hooks stable. It hid the
absence of defined inputs and let every method read or replace unrelated
state.

### Mode responsibility table

| Current responsibility | Classification | Direction |
| --- | --- | --- |
| Select adapter targets or fine-tuned component/parameter subjects | strategy declaration, expressed as a training-subject filing | The strategy deliberately says what is intended to train. It does not independently materialize optimizer groups. |
| Construct and attach an adapter, apply continuation policy, and load initialization weights | Trainer-recognized capability plus adapter-domain behavior | Trainer owns when attachment/preparation happens. Adapter construction and artifact loading remain delegated domain behavior with typed inputs/results. |
| Apply family-specific post-processing such as SDXL text-encoder tail freezing | strategy-internal feature or component compatibility rule | The standard strategy explicitly wires the rule. It must not be rediscovered from a mode or family-name branch. |
| Toggle `requires_grad`, choose train/eval state, and designate synchronization/clipping participants | Trainer/optimization mechanism over the declared training subjects | These are realized execution facts. They should be derived from one optimization/preparation result rather than stored as `_train_*` flags in several owners. |
| Cast trainables and frozen execution participants for full FP16/BF16 | Trainer precision/preparation mechanism | The strategy supplies participants and constraints; generic precision infrastructure performs casts. Component-specific limitations remain explicit component/features constraints. |
| Resolve logical parameter groups | optimization system acting on the strategy's declared subjects and constraints | Evolve `OptimizationPlan`; do not make the strategy or a mode build raw optimizer dictionaries by convention. |
| Instantiate the optimizer | Trainer-owned optimization mechanism | `get_optimizer()` and scheduler creation belong to one infrastructure-owned realization path. |
| Call `accelerator.prepare()`, build a DeepSpeed composite, replace module handles, and select the accumulation handle | Trainer/distributed preparation mechanism | The strategy publishes participants and joint-preparation constraints; infrastructure returns authoritative prepared bindings. |
| Enable generic gradient checkpointing | Trainer/performance mechanism | Operate over declared trainable/execution participants, not hard-coded denoiser/text-encoder projections. |
| Run adapter-specific `enable_gradient_checkpointing()` or `prepare_grad_etc()` | explicit component/capability behavior coordinated during Trainer preparation | The component may implement the specialized operation. A fine-tune no-op should not be required to pretend support. |
| Register Accelerate save/load hooks and restore epoch/step coordinates | Trainer runtime-checkpoint mechanism with explicit state contributors | Adapter or other features may contribute state serialization, but the mode should not own checkpoint registration or the run coordinates. |
| Put trainable participants into train/eval state at epoch/sample boundaries | Trainer lifecycle mechanism | Derive participants from prepared/optimization state. `set_train` and `set_eval` do not need mode-specific duplicates for ordinary modules. |
| Invoke adapter `on_epoch_start` or `on_step_start` behavior | specialized selected capability/component behavior with Trainer-owned timing | Preserve only behavior that has real semantics. Do not create a universal raising/no-op hook bag merely because one adapter method exists. |
| Apply adapter maximum-norm regularization and return its metrics | optimization-adjacent capability/domain behavior with Trainer-owned post-step timing | Give it typed inputs/results and metric output. It is not a generic meaning of training mode. |
| Return parameters for clipping | redundant projection of the optimization plan | The realized optimization state should name clipping/synchronization participants once. Remove the second mode query. |
| Select artifact role and serialization format | trained-artifact persistence capability declaration | Metadata and save orchestration consume typed artifact semantics supplied by the strategy/capability. |
| Serialize adapter or full-model artifacts | artifact persistence capability plus adapter/model-domain serialization | Trainer owns timing, retention, destination, and publishing coordination. The capability extracts/serializes the selected artifact and returns a typed result. |
| Save EDM2 loss-modifier side artifacts through the mode save method | misplaced owning-feature persistence contribution | A trainable loss modifier or objective-adjacent feature declares its own artifact contribution; adapter versus fine-tune mode is irrelevant. |
| Choose diagnostic components and aliases | observability over authoritative bindings and optimization subjects, with optional component descriptions | Logging should query stable component/optimization facts. It should not ask a mode to reconstruct the active topology. |

`prepare_trainables()` is the clearest example of why one-to-one method
migration would fail. In adapter mode it currently performs adapter
construction, continuation, target selection, family post-processing,
attachment, `requires_grad` mutation, and Trainer-state publication
(`library/training/modes/adapter_mode.py:115-179`). In fine-tune mode it
performs parameter selection, `requires_grad` mutation, train/eval transition,
family post-processing, and primary-trainable selection
(`library/training/modes/finetune_mode.py:84-158`). Those responsibilities
belong to different exchanges.

Likewise, `build_optimizer_params()` both constructs an `OptimizationPlan` and
materializes an optimizer, while `prepare_with_accelerator()` replaces loaded
module references, optimizer, scheduler, synchronization handle, and primary
trainable (`adapter_mode.py:209-290`; `finetune_mode.py:190-273`). The target
boundary is therefore:

```text
strategy declares training intent and constraints
                    ↓
Trainer/optimization resolves one preparation and optimization plan
                    ↓
Trainer infrastructure materializes and prepares execution state
                    ↓
authoritative prepared bindings and optimization runtime
```

### Objective responsibility table

The objective layer is smaller but similarly split. `Trainer.__init__`
currently constructs an objective independently from the strategy
(`training/runners/trainer.py:130-254`), later asks it to build runtime state,
and then passes that runtime back into family `process_batch()` calls. DDPM
helpers are called directly by SD/SDXL strategy facets, while rectified-flow
batch-state construction lives on its runtime
(`objectives/base.py:14-43`; `objectives/ddpm.py:27-259`;
`objectives/rectified_flow.py:18-149`).

| Current responsibility | Classification | Direction |
| --- | --- | --- |
| Choose DDPM versus rectified flow from global config | strategy declaration and contract validation | Objective choice is part of how the authored strategy trains. Trainer should not independently choose a second behavioral axis. Configuration may parameterize a choice explicitly exposed by that strategy. |
| Construct the noise scheduler, timestep sampler/runtime, prediction convention, and objective-specific state | strategy-internal objective feature | The strategy deliberately wires the objective implementation. Trainer may coordinate lifecycle creation without knowing DDPM/RF classes. |
| Produce corruption/noisy inputs, timesteps/noise levels, targets, and objective weighting | strategy-internal objective feature used by step execution | These are mathematical training semantics. They belong behind the strategy boundary and should not become universal Trainer fields. |
| Advance adaptive timestep scheduling at a global step | objective-feature state transition invoked at Trainer-owned timing | Trainer supplies the step coordinate through the step exchange; the selected feature owns its state transition. |
| Feed timestep/loss observations back into an adaptive sampler | objective-feature state update consuming a typed step observation | The step result declares the relevant observations. Trainer may route them at the defined boundary without understanding timesteps. |
| Expose `num_train_timesteps`, `timestep_runtime`, or `alphas_cumprod` for plotting, Huber thresholds, and logging | objective-specific capability/observation, not core strategy vocabulary | Replace direct runtime-field inspection with typed objective observations or logging/metadata contributions where requested. |
| Build and store `loss_modifier` inside `ObjectiveRuntime` | optimization capability currently misplaced inside the objective bundle | Trainer owns applying the modifier and backpropagating its result. The selected strategy may provide the modifier capability, but DDPM/RF identity should not automatically own all post-loss optimization behavior. |
| Apply DDPM-only post-loss weighting inside family `process_batch()` | objective feature | Keep the mathematical rule with the DDPM objective implementation; avoid duplicating its invocation in every family strategy. |
| Provide objective name/prediction facts for metadata and artifact semantics | strategy/objective metadata contribution | File typed facts from the selected strategy/objective. Avoid Trainer branches such as `objective.name == "rectified_flow"`. |

The objective implementation therefore remains useful reusable vocabulary, but
not as a separate top-level authority beside the strategy. Its mathematical
behavior belongs inside the authored strategy. Trainer owns the timing and
optimization mechanics that consume the strategy result.

### Resulting boundary

The classification reduces the active runtime relationship to:

```text
STRATEGY
  declares:
    what is trained
    objective/training semantics
    selected capabilities and constraints
    model/component arrangement

TRAINER
  executes:
    lifecycle
    preparation and authoritative rebinding
    optimization realization
    backward and stepping
    runtime checkpoint coordination
    triggers and observation

CAPABILITY / DOMAIN IMPLEMENTATIONS
  provide specialized behavior when selected
  without receiving or mutating the whole Trainer
```

This does not require an adapter strategy, a fine-tune strategy, or a renamed
mode object. The standard SD/SDXL/SD3 strategy definitions may deliberately
support and select those training arrangements through the contract.

### Decisions and remaining questions after classification

The classification settles:

- `TrainingMode` should dissolve rather than be renamed;
- objective selection belongs to strategy authoring rather than independent
  Trainer construction;
- optimization and distributed preparation are standard Trainer mechanisms;
- ordinary train/eval, clipping-participant, and diagnostics projections
  should derive from authoritative preparation/optimization state;
- runtime checkpoint coordination and trained-artifact persistence remain
  separate;
- specialized capability code does not need to live directly in the central
  Trainer class to be Trainer-coordinated.

It does not yet settle:

- the exact typed training-subject declaration;
- the exact preparation participants/result and authoritative binding holder;
- whether specialized Trainer-recognized capabilities are represented by
  behavior objects, typed data handled by training services, or a narrow
  combination;
- the generic step result and observation-routing shape;
- which existing adapter lifecycle methods are genuine required semantics
  versus compatibility conveniences that can disappear.

The next design milestone can now define binding/preparation and optimization
exchanges without treating mode or objective as peer runtime owners.

## Milestone 7: Logical Identity And Execution-Route Semantics

The first two authoritative-binding questions are settled at the semantic
level.

- One stable strategy-scoped logical identity denotes one semantically distinct
  training participant.
- Concrete Python objects, implementation types, source/catalog identities,
  wrappers, and parameter scopes do not define that identity.
- The standard contract provides one normal authoritative prepared execution
  binding per logical component.
- A selected capability may add a named route only when a materially different
  prepared/callable representation is required. Training phase names or caller
  intentions alone do not justify separate routes.
- Exactly one current binding is authoritative for each logical-component/route
  pair.
- Shared live state is one component. Derived state may remain another route
  only with explicit refresh or synchronization semantics. Independently
  evolving state is a different logical component.
- Original, unwrapped, inspection, metadata, and artifact handles are typed
  access views rather than competing execution routes.
- A route whose freshness guarantee no longer holds cannot remain silently
  authoritative.

This resolves the semantic cardinality question without yet choosing a
`ComponentKey` representation, route-name vocabulary, refresh mechanism, or
authoritative storage owner. Those belong to the binding/preparation exchange.

## Milestone 8: Adapter Pressure Test And Settled Arrangement Semantics

Question 3 asks how additions, replacements, and delayed materialization alter
the authoritative arrangement. The current adapter path is useful evidence but
does not define the full adapter design.

### Current repository evidence

- the built-in registry currently exposes the PEFT family of adapter methods;
- `AdapterMode.prepare_trainables()` resolves model targets, builds one adapter
  runtime, applies it to existing modules, publishes the runtime separately,
  and then hands its parameters to optimization;
- current LoRA attachment replaces selected target-module `forward` methods in
  place, so adapter-owned state is added while the top-level target object is
  not replaced;
- VeRA demonstrates that one independently managed adapter runtime/artifact may
  contain shared state plus many target-local method modules; and
- `LoadedAdapterRuntime.merge_into()` already distinguishes destructive merge
  from ordinary attachment and export loading.

The accepted adapter requirements also state that current diffusion/LyCORIS
shapes must not define the universal adapter contract, permit later
parameter-granular targets, and leave mixed-method overlap semantics to later
design.

Relevant source locations:

- `library/training/modes/adapter_mode.py:115-179`
- `library/adapters/methods/peft/lora/module.py:229-231`
- `library/adapters/methods/peft/vera/runtime.py:65-136`
- `library/adapters/runtime/context.py:81-93`
- `library/adapters/shared/base.py:7-16`
- `openspec/specs/adapter-system/spec.md`
- `openspec/specs/adapter-module-targeting/spec.md`

### Broader adapter shapes

The term adapter covers several topologies that the binding contract must not
collapse into the current LoRA shape:

| Shape | State and interaction pressure |
| --- | --- |
| injected delta/reparameterization | owns separate learned state but changes selected host operations without necessarily exposing a top-level execution route |
| prefix/prompt/state augmentation | may own trainable tensors that participate through conditioning or input assembly without an independent callable module |
| sidecar network | adds an independently executable component whose outputs feed another component |
| wrapper/replacement adapter | owns adaptation state and may replace the host's prepared execution route with a composite callable |
| merged/folded adapter | transfers its effect into host state and may cease to exist as an active independent participant |
| multiple/routed adapters | require explicit identities, target relationships, ordering/overlap semantics, and activation state |

Representative primary sources are the
[LoRA paper](https://arxiv.org/abs/2106.09685),
[Prefix-Tuning paper](https://arxiv.org/abs/2101.00190),
[ControlNet paper](https://arxiv.org/abs/2302.05543), and
[T2I-Adapter paper](https://arxiv.org/abs/2302.08453).

### Settled operation taxonomy

The existing three-way distinction—execution rebinding, logical-slot
replacement, and participant addition—is necessary but incomplete. The
settled taxonomy is:

```text
declare a participant
materialize authoritative bound state
replace authoritative bound state
rebind an execution route
transition an operational relationship
amend the arrangement by adding or retiring a participant
merge/fold state and record the resulting lineage
```

These operations must not be inferred from incidental Python assignment or
loading order. In particular, an authored external autoencoder that is selected
before materialization is an initial binding, not a semantic replacement merely
because an implementation happens to load an integrated VAE first.

Materialization moves an absent or deferred declared participant to bound
state. Replacement supersedes an existing authoritative binding and therefore
has distinct retirement/supersession and fact-invalidation consequences.
Merge/fold transforms host state using adaptation state, may retire the adapter
from the current arrangement, and records transition and artifact provenance;
preserving a separately usable adapter remains valid behavior.

For ordinary authored adapter training, the leading direction is:

```text
strategy declares adapter participant and intended relationship
  -> structural validation
  -> model/adapter materialization and target resolution
  -> attachment establishes the concrete effect
  -> affected constraints are validated
  -> runtime preparation establishes authoritative bindings
```

Ordinary adapter attachment therefore should not require an arbitrary
post-validation arrangement mutation. Truly dynamic additions remain possible
through an explicit arrangement-amendment extension that declares additions,
relationships, capability changes, invalidated prepared routes, optimization
plans, and caches before the amendment is accepted.

### Logical identity and relationship findings

A logical component represents one independently addressable semantic
participant for which authoritative bound state is maintained, whether or not
it is independently executable. For adapters, logical identity follows the
unit that can be independently addressed and managed under the strategy
contract, rather than a Python runtime object or every injected submodule.
Shared banks, tensors, and target-local modules may remain qualified internal
state when they share one lifecycle and persistence unit. One runtime may also
expose multiple logical adapters when they are independently selectable,
trainable, attachable, persistable, mergeable, or retireable.

Participant identity is separate from relationship/effect identity. One
adapter can have multiple operational relationships, and attachment,
activation, or detachment transitions those relationships rather than creating
or retiring the participant. Structural relationships describe the current
arrangement; lineage relationships record history. A wrapper can alter a
host's prepared route without acquiring the host component's semantic
execution ownership merely by becoming the outermost callable.

The independent state axes are:

```text
participant lifecycle
  declared, bound, retired

optimization status
  selected/unselected, trainable/frozen

operational relationship lifecycle
  declared, resolved, active, inactive, detached
```

### Important qualification to question 2

Adapter breadth qualifies, rather than discards, the execution-binding
cardinality decision:

```text
execution-capable logical component
  -> one normal authoritative prepared execution binding

state-bearing component without independent execution
  -> authoritative state/optimization/artifact bindings
     without a fabricated forward route
```

### Persistence views and conclusion

Current authority and historical evidence must not collapse into one mutable
container. Later persistence design must distinguish:

```text
current arrangement
run transition history
durable artifact provenance
```

Question 3 is settled at the semantic level. In particular, component identity
follows semantic participant identity; componenthood does not imply
executability; ordinary authored adapters exist before materialization;
materialization is not addition; attachment is not component creation;
execution preparation is not arrangement mutation; and historical provenance
is not current bound state. Concrete representation, storage ownership,
invalidation machinery, and APIs remain downstream design work.

## Milestone 9: Artifact-Persistence Product Pressure Test

Question 4 asks what persistence saves once logical identity, current bound
state, and arrangement changes no longer follow incidental Python objects. The
active repository paths show that there is already no single honest answer such
as "the model object" or "the current strategy state dict." Different products
select and assemble semantically different state.

### Current Trainer boundary

`Trainer.save_checkpoint()` is the common trained-artifact trigger, but it
currently delegates the write to `TrainingMode.save_checkpoint()` and passes an
overloaded `target_model` value. `_build_checkpoint_metadata()` also derives the
artifact role from the mode before the physical write. The method returns no
artifact result describing the resources that were actually emitted.

The ordinary step and epoch paths establish a useful consistency boundary:
they wait for distributed participants before the main process writes. The
finalization path coordinates training shutdown and then performs state and
artifact saves, although the same consistency intent is less explicit in its
public exchange.

Relevant source locations:

- `library/training/runners/trainer.py:471-576`
- `library/training/phases/training_loop.py:133-183`
- `library/training/phases/training_loop.py:341-394`
- `library/training/runners/trainer.py:956-968`

This supports keeping lifecycle timing and the persistence boundary under
Trainer control. It does not support mode ownership of artifact meaning or the
assumption that one target Python object defines what must be saved.

### Full-model products

The current SDXL strategy unwraps the denoiser and text encoders, obtains the
autoencoder, and passes those components into family conversion code. The
external checkpoint is therefore assembled from several logical components and
access views rather than serialized from the outermost prepared object.

The SD3 strategy similarly unwraps several components and asks the model-domain
serializer to produce a main MMDiT-plus-VAE checkpoint and optional CLIP-L,
CLIP-G, and T5 sidecars. The serializer returns physical paths, not a semantic
artifact result, and only the main resource currently receives the supplied
metadata. That is evidence for a multi-resource product, but it does not by
itself establish whether a particular SD3 product is complete, partial, or
referential. Optional members and external dependencies must answer that.

Relevant source locations:

- `library/strategies/sdxl/checkpointing.py:49-121`
- `library/strategies/sd3/checkpointing.py:70-120`
- `library/models/sd3/conversion.py:32-78`

### Adapter products and resume state

Adapter export already makes the strongest semantic-ownership case.
`AdapterExportIO` is distinct from the Accelerator resume hooks. LoRA export
iterates adapter-owned method modules to assemble one adapter weight mapping.
VeRA export combines shared projection-bank state with target-local module
state. Neither product can be defined correctly as the state dict of one
arbitrary wrapper or host model.

The resume path has a different purpose. `accelerator.save_state()` and
`accelerator.load_state()` preserve and restore live execution state, while
adapter hooks narrow or supplement that state for continuation. This is a
separate persistence contract from publishing trained adapter weights even when
both are triggered at Trainer lifecycle boundaries.

Relevant source locations:

- `library/adapters/shared/state_io.py:13-113`
- `library/adapters/methods/peft/lora/runtime.py:121-122`
- `library/adapters/methods/peft/lora/state_dict.py:23-43`
- `library/adapters/methods/peft/vera/runtime.py:65-136`
- `library/adapters/methods/peft/vera/state_dict.py:78-109`
- `library/training/checkpointing.py:45-103`
- `library/training/checkpointing.py:324-355`

### EDM2 correction

`FineTuneMode` still contains a filename-based `_edm2_loss_weights` branch.
That is residual compatibility evidence, not the active ownership model. The
active step, epoch, and final paths call `loss_modifier.save_sidecar()`
directly, and `EDM2LossModifier` owns that behavior. No active production caller
was found that routes the EDM2 suffix through `Trainer.save_checkpoint()`.

The active path is closer to the desired capability ownership, but it still
names products through a suffix, returns no typed artifact result, and uses the
generic mode-derived metadata builder. The correct conclusion is therefore not
that EDM2 persistence is mode-owned; it is that capability-owned semantic state
is already escaping a persistence exchange that cannot describe it properly.

Relevant source locations:

- `library/training/modes/finetune_mode.py:359-403`
- `library/training/phases/training_loop.py:83-130`
- `library/training/phases/training_loop.py:268-310`
- `library/training/runners/trainer.py:926-954`
- `library/losses/edm2/edm2_modifier.py:82-86`

### Product pressure summary

| Current product | Semantic state selected | Current packaging | Important pressure |
| --- | --- | --- | --- |
| SDXL checkpoint | denoiser, text encoders, autoencoder, configuration/provenance | one converted checkpoint in the common path | extraction crosses several logical components and unwrapped access views |
| SD3 checkpoint | MMDiT, VAE, and optional text-encoder state | main resource plus optional sidecars | product/member/resource and dependency semantics cannot be inferred from file count |
| LoRA/VeRA export | adapter-owned state, including shared and target-local substructure | adapter weight resource | semantic ownership does not follow host or wrapper object identity |
| EDM2 sidecar | loss-modifier-owned learned state | separately named resource | capability-owned product is not described by the current mode-derived checkpoint exchange |
| runtime resume snapshot | execution, optimizer, scheduler, progress, backend, and continuation state | Accelerator-managed state directory/resources | restoration semantics are different from trained-artifact semantics |

### Settled persistence model

Persistence should be described through four stages:

```text
capability declaration
  -> persistence request
  -> resolved artifact plan
  -> artifact result
```

The resolved plan selects a purpose-specific **artifact state projection** by
logical participant and relationship identity. It records semantic coverage,
dependencies, declared semantic transformations, expected members,
representation, packaging expectations, and consistency constraints. The
result records the members and physical resources actually produced, including
formats, sizes, checksums, references, and partial/failure status.

The containment levels are:

```text
artifact product
  one semantic result

artifact member
  one semantically meaningful constituent

physical resource
  file, directory, shard, blob, or remote object
```

These are not one-to-one. One member may be sharded across resources and one
resource may encode several participants. "Bundle" describes packaging or
cardinality; it does not prove that a product is complete or self-contained.

The consistency invariant is that every product member corresponds to one
accepted arrangement and one Trainer-established persistence boundary while
satisfying each contributor's declared freshness relationship. This need not
mean one universal revision number. Frozen components and external references
can participate coherently when their dependency and freshness semantics are
explicit.

Responsibility divides as follows:

```text
strategy/product capability
  declares supported semantic products and transformations

Trainer/runtime infrastructure
  chooses lifecycle timing, establishes the persistence boundary,
  coordinates ranks, and obtains stable current state

resolved artifact plan
  identifies semantic coverage, dependencies, expected members,
  representation, and constraints

domain serializer
  performs mechanical representation conversion and writes resources

artifact result
  reports what was actually emitted
```

Transformations that change coverage, dependencies, lineage, or realization
identity—such as merge/fold, pruning, semantic quantization, or producing a
delta—must be declared at the product/capability level. Mechanical operations
such as key renaming, tensor-layout conversion, resolved dtype encoding,
sharding, and embedded-metadata writing belong to the domain serializer. A
serializer may implement the mechanics of a semantic transformation but must
not silently choose one.

Runtime resume remains a separate contract concerned with restoring execution.
It may share Trainer-controlled timing, stable-state infrastructure, and storage
services with artifact persistence, but the two should not be collapsed into a
single request differentiated only by a kind enum.

Question 4 is settled at the semantic level. The current API shapes,
serialization returns, member metadata policy, failure representation, and
asynchronous publication mechanics remain downstream implementation design.
Question 5 still needs to determine which object owns the authoritative binding
map from which persistence and other consumers obtain their narrow current-state
views.

## Milestone 10: Target-First Binding-Authority Scenarios

Question 5 must not be answered by wrapping the current Trainer fields in a new
class. No production implementation has yet moved toward the direction in this
document. The current code is therefore used only as a **pressure oracle**:

```text
target-first scenario
  establishes the semantic transitions and views the contract needs

current code evidence
  reveals lifecycle pressure, information that must survive, and failure modes

not assumed
  that current object placement, call direction, names, or cardinality survive
```

The scenarios below use **binding authority** as a deliberately abstract term.
It means the one contract-governed authority for accepted current participant,
relationship, access-view, and execution-route bindings. It does not yet choose
a class name, module, container shape, or whether that authority is physically
embedded in the strategy object or paired with it at filing time.

### Scenario A: ordinary multi-component fine-tuning

An authored SDXL strategy may deliberately declare participants resembling:

```text
model.denoiser
representation.autoencoder
conditioning.primary
conditioning.secondary
```

These example identities are not a final universal vocabulary. The strategy
also declares which participants are training subjects, the relationships and
features it uses, and the artifact products it supports.

The target lifecycle is:

```text
author strategy
  -> declare participants and relationships
  -> materialize original/current state by stable participant identity
  -> validate the filing needed for this training arrangement
  -> expose a generic preparation projection
  -> Trainer/runtime infrastructure prepares the requested participants
  -> accept prepared execution bindings by the same stable identities
  -> expose an optimization projection
  -> execute strategy behavior against narrow authoritative binding views
```

Trainer may see a preparation participant's opaque identity, concrete object,
requirements, and joint-preparation group. It must not learn that SDXL has one
autoencoder, two text encoders, or one denoiser. Strategy and selected feature
implementations interpret their own identities.

Current evidence: `LoadedModelComponent` already couples a stable key and
generic roles/capabilities to a live module. `Trainer.setup()` stores the
resulting tuple and then exposes family-shaped compatibility properties. Those
properties write replacement objects back into the tuple by role. This proves
that stable keyed replacement is useful; it does not justify Trainer ownership,
role-unique setters, or one module field as the final binding record.

Pressure result: the authority must preserve stable logical identity across
loading and preparation, while Trainer consumes generic projections rather than
the complete family arrangement.

Relevant current source:

- `library/models/components.py:25-69`
- `library/training/runners/trainer.py:307-431`
- `library/training/runners/trainer.py:970-1045`

### Scenario B: declared but deferred SD3 participant

A strategy can declare a participant before its concrete state is available:

```text
arrangement revision R
  model.denoiser: declared, unbound, deferred by an authored loading policy
  representation.autoencoder: bound
  conditioning.*: bound or explicitly absent under the filing
```

A later loader result performs **materialization**, not arrangement addition:

```text
model.denoiser
  declared/unbound -> bound

arrangement identity
  unchanged

binding revision
  advanced
```

Any preparation, optimization, cache, or artifact plan that depended on the
unbound state becomes invalid or must be recomputed. A lifecycle phase that does
not require the denoiser may proceed only when the authored contract explicitly
allows that partial fulfillment; a phase requiring it receives a named
unfulfilled requirement rather than `None` failing somewhere downstream.

Current evidence: the SD3 loader constructs the complete declared component
surface even when optional text encoders or a lazily loaded denoiser are `None`,
and Trainer explicitly comments that the denoiser may be absent during initial
loading. The current tuple does not distinguish unbound, explicitly absent,
deferred, retired, or failed states, nor does later materialization establish an
explicit revision/invalidation boundary.

Pressure result: the authority must outlive individual loaders, represent
declared-but-unbound state, and distinguish binding revision from arrangement
amendment.

Relevant current source:

- `library/strategies/sd3/loading.py:22-69`
- `library/models/components.py:43-69`
- `library/training/runners/trainer.py:419-431`

### Scenario C: authored adapter materialization and attachment

An ordinary adapter is declared before runtime construction:

```text
adaptation.main
relationship: adaptation.main adapts model.denoiser
```

The target sequence is:

```text
resolve declared host relationship
  -> materialize adapter-owned state
  -> activate attachment relationship
  -> update affected execution semantics
  -> validate/recompute preparation and optimization projections
```

Two attachment shapes must both work:

```text
in-place injection
  host Python object identity may stay unchanged
  host execution semantics and relationship revision still change

wrapper/composite attachment
  adapter remains a distinct participant
  host execution route may rebind to a composite callable
```

The in-place case is particularly important for Q5: an object-keyed dictionary
cannot detect that authoritative execution semantics changed when the object is
the same. The transition must advance an accepted binding/relationship revision
and invalidate any compiled route or other dependent projection whose freshness
guarantee no longer holds.

VeRA-like shared banks and target-local modules may remain substructure of one
adapter participant when they share one lifecycle and artifact unit. They do
not require Trainer to enumerate every injected Python module.

Current evidence: `AdapterMode.prepare_trainables()` reads Trainer-owned loaded
components, resolves concrete targets, builds and applies the adapter, then
publishes adapter and target state back onto Trainer. Its Accelerator path later
mutates denoiser, text-encoder, adapter, optimizer, and synchronization fields
through the whole Trainer. The target resolver's stable component/path
provenance is valuable; the whole-Trainer mutation and separate shadow fields
are not the desired contract.

Pressure result: one authority must cover participants and relationships while
allowing capability code to propose typed transitions. Attachment must not be
hidden as arbitrary module mutation.

Relevant current source:

- `library/training/modes/adapter_mode.py:115-179`
- `library/training/modes/adapter_mode.py:243-290`
- `library/adapters/runtime/targets.py:184-248`
- `library/optimization/grouping.py:197-308`

### Scenario D: distributed preparation, execution, and persistence

After distributed preparation, one logical participant may expose several
different kinds of current access without confusing their purposes:

```text
model.denoiser
  current execution route -> prepared/distributed callable
  original access view     -> underlying component when valid
  artifact state view      -> product-specific semantic projection

backend coordination
  grad-sync/composite handle -> infrastructure state, not automatically a
                                logical model participant
```

Trainer obtains a generic preparation projection, prepares the concrete inputs,
and submits a keyed preparation result. Accepting that result atomically rebinds
the affected execution routes. Joint DeepSpeed preparation may return one
backend coordination object for several participants; that object must not
erase their distinct logical identities or automatically become another model
component.

At a persistence boundary, an artifact capability resolves a coherent snapshot
from the same accepted authority. It may obtain state through an unwrapped
access view, adapter-owned state, several participant views, or external source
references. It does not infer the product from the current outer wrapper.

Current evidence: mode preparation presently writes prepared modules and a
separate `_grad_sync_handle` directly onto Trainer. SDXL and SD3 checkpoint
paths explicitly unwrap and assemble several components, while adapter export
assembles capability-owned state. The diversity is useful evidence that
execution, backend coordination, and artifact access are distinct views.

Pressure result: the authority must support atomic keyed rebinding and coherent
read snapshots without forcing backend coordination handles or artifact
projections into one generic `module` slot.

Relevant current source:

- `library/training/modes/finetune_mode.py:234-273`
- `library/training/modes/adapter_mode.py:243-290`
- `library/strategies/sdxl/checkpointing.py:49-121`
- `library/strategies/sd3/checkpointing.py:70-120`
- `library/adapters/shared/state_io.py:13-113`

### Scenario E: replacement and dependency-aware invalidation

Replacing authoritative component state is an explicit transition:

```text
replace model autoencoder A with B
  -> supersede A's current state binding
  -> advance the participant binding revision
  -> re-evaluate relationships involving that participant
  -> invalidate only dependent preparation/optimization/cache/artifact views
  -> publish accepted transition facts
```

Invalidation cannot be a hard-coded family cascade. A latent cache may depend on
the autoencoder realization; a conditioning cache may instead depend on one or
more encoders; an adapter target plan may depend on the host's structural
realization. Features and resolved plans therefore need to declare their
dependencies using stable participant/relationship identities. The authority
can then report what became stale without knowing diffusion-specific anatomy.

Current evidence: Trainer compatibility setters replace modules in
`loaded_components`, but they do not represent supersession or perform
dependency-aware invalidation. Existing optimization grouping and adapter target
selection derive new plans from the current tuple, which demonstrates the need
to know their inputs but not a safe way to retain their freshness.

Pressure result: every derived projection that may outlive a transition needs a
declared dependency set and source revision. Replacement must return an
invalidation outcome rather than relying on callers to remember which fields to
refresh.

Relevant current source:

- `library/training/runners/trainer.py:1010-1034`
- `library/optimization/grouping.py:197-228`
- `library/optimization/grouping.py:457-550`

### Scenario F: compound research strategy beyond current family shapes

The authority must also work for a manually authored strategy that the current
repository does not implement:

```text
student.denoiser
teacher.denoiser
adaptation.student

relationships
  teacher supervises student
  adaptation.student adapts student.denoiser

optimization selection
  student.denoiser and/or adaptation.student

artifact products
  student realization
  adapter delta over student/base dependency
  optional compound research snapshot
```

Teacher and student remain different logical participants even if they started
from the same source. A pixel-space strategy may omit an autoencoder entirely.
A strategy may introduce an LLM, discriminator, reward model, or another
trainable side network without expanding Trainer with new family fields.

Trainer still needs only generic preparation, optimization, step-result, and
persistence exchanges. Strategy behavior and selected features request their
own narrow binding views by declared identity.

Pressure result: any proposed owner that assumes one denoiser, one primary
trainable, a VAE/autoencoder, a text-encoder list, or a single model family fails
even if it can migrate today's SDXL flow.

### Requirements derived from all scenarios

The scenario model supports these requirements without yet selecting the final
Python shape:

1. **One strategy-scoped current authority.** There cannot be independent
   authoritative copies on Trainer, loaders, modes, strategy facets, and
   metadata.
2. **Contract-governed identity and transitions.** Declarations,
   materialization, replacement, route rebinding, relationship transitions,
   and arrangement amendments pass through named operations.
3. **One writer protocol, multiple transition producers.** Loading features,
   adapter features, and Trainer infrastructure may propose results, but one
   authority validates and accepts them atomically.
4. **Revisioned participant, relationship, and route state.** Object identity is
   insufficient, especially for in-place adapter attachment.
5. **Narrow typed projections.** Trainer receives preparation, optimization,
   synchronization, and persistence exchanges; capabilities receive only their
   declared participant/relationship views.
6. **Atomic snapshots and result application.** Multi-participant preparation
   and persistence cannot observe half-applied replacements.
7. **Dependency-aware freshness.** Derived plans and routes name the identities
   and revisions they depend on and cannot remain silently authoritative after
   those guarantees fail.
8. **Backend handles stay distinct.** A synchronization/composite handle is not
   automatically a logical component or its normal execution route.
9. **Runtime authority is not durable history.** Metadata observes accepted
   facts and transition history; it does not become the live binding owner.
10. **No family-shaped core cardinality.** The authority supports absent,
    deferred, state-only, executable, multiply routed, and compound
    participants without Trainer learning their anatomy.

### Ownership conclusion after scenario and outside pressure review

The scenarios make the ownership boundary substantially narrower:

```text
authored strategy and selected features
  declare semantics and propose typed transitions

contract-governed binding authority
  accepts and maintains canonical current state

Trainer/runtime infrastructure
  consumes generic projections and returns preparation/optimization results

metadata and persistence
  consume accepted snapshots/projections
```

This rules out Trainer family fields, metadata storage, family-mixin attributes,
or an unrestricted shared dictionary as the canonical owner. The scenarios
point toward a binding authority belonging to the **bound strategy contract
scope**, but they cannot choose a physical layout: an embedded authority, a
paired object, and a separate object can all be constructed to satisfy the same
normal-path scenarios.

The follow-up pressure review therefore used discriminating responsibility
criteria instead of adding more ordinary topology scenarios:

```text
single-writer enforceability
independent conformance testing
state/behavior separation
one-per-run construction and lifetime
scoped access discipline
incremental migration from LoadedModelComponent
```

The resulting direction is a dedicated contract-owned, per-run binding
authority inside the complete Trainer-facing strategy boundary. Trainer still
receives one strategy; physical separation of the authority prevents mutable
binding state from becoming arbitrary strategy-facet fields rather than
creating a second public object for Trainer to coordinate.

Additional scenarios remain useful as conformance tests rather than ownership
blockers:

- rejected and partially failed multi-participant transitions must not expose
  half-applied state;
- distributed ranks and backend replicas must remain views of one logical run
  authority rather than independent writers;
- overlapping adapters must preserve participant and relationship identity;
- compiled or separately materialized routes must declare freshness and refresh
  semantics;
- independently evolving EMA state remains a separate participant;
- resume/restore must rebuild authoritative current state without confusing
  durable history with live ownership; and
- mid-run trainability or parameter surgery must explicitly invalidate affected
  optimization and execution projections.

The reviews also reinforced that migration sequencing and architectural
semantics are separate decisions. The project does not adopt a deliberately
reduced first-version contract such as universal invalidate-all behavior, an
intrinsically single-route representation, or a persistence shortcut that
discards the settled plan/result/product/member/resource meanings. Work may be
implemented incrementally, but the interfaces introduced by each slice should
be shaped for the intended contract.

Graph evidence for the cited source paths used generation
`2026-08-21T02:44:51Z` on branch `model-strategy-trainer`. Exact cited paths had
no recorded coverage issue and matched index metadata. This remains a
best-effort signal rather than proof of complete source coverage; excluded
`__pycache__` trees are irrelevant to the source-level pressure test.
