# Strategy System Direction

## Status

This is an exploratory design record, not an implemented contract or an
implementation plan.

It records why the current model/strategy/trainer architecture is being
revisited, the routes considered during discussion, the reactions that ruled
some routes out, and the direction that currently feels most promising. It is
intentionally more historical and argumentative than a module README. Once the
direction is settled, an OpenSpec change should turn the relevant conclusions
into requirements, design decisions, migration tasks, and acceptance tests.

The existing strategy system remains the active production architecture while
this direction is explored.

## Why This Discussion Exists

The strategy system was itself an organizational response to the original
`sd-scripts` architecture. Model-family behavior that had been intertwined
with large training scripts was separated into:

```text
library/models/       model/component definitions and mechanics
library/strategies/   model-family behavior used by this repository
library/training/     shared training lifecycle and orchestration
```

That was a substantial improvement. The current problem is not that the
strategy idea failed. The problem is that the first organization has matured
enough to expose its own limits:

- `TrainingStrategy` is one aggregate object, but also a union of eleven ABCs.
- Trainer-facing operations and family-internal collaboration methods share the
  same public contract surface.
- The trainer owns some loaded-model state while strategies retain other
  loading facts on themselves.
- The trainer passes VAE, text encoders, denoiser, trainable model, dtypes,
  configuration, objective runtime, and execution flags back into strategy
  methods after the strategy originally loaded many of those objects.
- Generic training phases still understand the topology of the three current
  latent diffusion families.
- SD, SDXL, and SD3 repeat broad training and validation flows even when the
  objective and loss semantics are shared.
- The current contracts describe the historical SD/SDXL extraction more
  strongly than they describe the full set of arrangements the training
  pipeline could intentionally accept.

Metadata work made this pressure more visible. Durable model identity,
component provenance, and model realization records need a model system whose
loaded component boundaries and lifecycle are intentional. Finishing richer
model metadata on top of an unclear model integration architecture would turn
the current uncertainty into metadata schema debt.

## Current Architecture: What Is Real Today

The active path is approximately:

```text
train.py
├── build_training_strategy(cfg)
├── build_training_mode(cfg)
└── Trainer(cfg, strategy, mode)
     ├── setup
     │    ├── obtains strategy tokenizers
     │    ├── asks strategy for casting policy
     │    └── asks strategy to load family components
     ├── caching
     │    ├── asks strategy for cache backends/bundles
     │    └── operates directly on trainer VAE/text-encoder projections
     ├── model preparation
     │    ├── asks mode to select/create trainables
     │    ├── asks strategy for family preparation hooks
     │    └── operates directly on trainer component projections
     └── training loop
          ├── asks mode for step hooks
          ├── asks strategy to process the batch
          ├── applies trainer-owned loss modifiers
          ├── performs backward/optimizer/scheduler work
          └── triggers validation, sampling, and checkpointing
```

The intended ownership principle underneath that shape remains valuable:

```text
trainer       owns when the run does something
training mode owns the selected form of training
strategy      owns how this repository uses a model family
models        own model components and their mechanics
```

The weakness is not the principle. The weakness is that the current public
interfaces and state ownership do not consistently enforce it.

## The Central Principle

The direction emerging from this discussion is:

```text
CONTRACT
   ↓ is fulfilled by
STRATEGY
   ↓ is executed by
TRAINER
```

The contract is the training pipeline's acceptance definition. It describes
the vocabulary, responsibilities, choices, constraints, and observable results
that make a model integration usable by the repository.

A model itself may have any internal definition. It can be a UNet, DiT,
autoencoder, pixel model, multimodal system, model containing an LLM, or an
arrangement not anticipated when this document was written. The strategy is
the adapter between those arbitrary internals and the pipeline contract.

The contract does not need to predict every future model. It needs to express
what the current pipeline is deliberately receptive to. When the pipeline
learns a new arrangement, the contract vocabulary and its conformance tests
expand deliberately.

## Contract As Vocabulary, Not A Universal Checklist

An early framing divided strategy behavior into required methods and optional
capabilities. That framing was rejected because almost any current contract
method could be moved into an "optional" list, producing a loose grab bag
instead of a meaningful contract.

The stronger framing is:

```text
Contract = vocabulary + required selections + compatibility rules
Strategy = a completed filing of that contract
```

A concern may admit several valid filings. For example:

```text
input representation
├── pixels
├── autoencoder latents
└── an already-computed representation
```

One representation is still required for training. Pixel and latent behavior
are alternatives within a required concern, not optional methods that may be
missing silently.

Other concerns may allow multiple selections:

```text
training subjects
├── base model parameters
├── adapter parameters
├── a newly attached component
├── base model + adapter
└── another explicitly supported combination
```

Some requirements are conditional on the selected arrangement:

```text
latent caching selected
  → the chosen representation must provide compatible cache behavior

distillation selected
  → a teacher/source behavior and compatible objective must be present

full-model output selected
  → family serialization must be available

adapter training selected
  → attachment, target selection, execution, and persistence requirements apply
```

Nothing is optional once the run has selected it. The completed strategy either
fulfills the resulting contract or the configuration must be rejected before
training.

## Working Vocabulary

The following terms currently seem useful. The names are not yet locked, but
the distinctions matter.

### Contract concern

A responsibility the training pipeline knows how to ask about, constrain, or
consume. Examples include input representation, conditioning, prediction,
trainable selection, validation, and persistence.

### Feature

A concrete or reusable behavior that can fill, modify, or help implement a
contract concern. Examples include CLIP tokenization/encoding behavior, latent
diffusion preparation, pixel diffusion preparation, weighted prompts, or an
adapter execution behavior.

### Capability

A declaration that an integration can provide, accept, or compose a particular
feature or semantic behavior. A capability describes availability; it is not a
substitute for fulfilling the selected run's requirements.

### Fulfillment

The validated result of filing selected features and family behavior into the
contract. Fulfillment proves that required concerns have providers and that the
providers are mutually compatible.

### Family integration / family strategy

The known, tested behavior through which one model family and its components
participate in contract fulfillment. Current classes such as
`SdxlTrainingStrategy` combine this role with parts of the complete training
strategy, so the eventual code-level terminology remains open.

### Fulfilled training strategy

The complete validated strategy for one concrete run. A normal run may be
dominated by one family integration, while a compound run may involve a teacher
family, a student family, adapters or newly attached components, and a selected
objective. "Fulfilled family strategy" is therefore too narrow as the general
name for the trainer-accepted result.

### Model component

A neural or structural part of the loaded model, such as an autoencoder, text
encoder, LLM, vision encoder, UNet, diffusion transformer, connector, or
adapter. Components belong to the model system; strategy features describe how
the repository uses them.

## Why A Feature Or "Grab Box" Layer Emerged

The current family folders repeatedly contain behaviors that are neither
universal base contracts nor naturally owned by only one family. The existing
code already contains two early expressions of a feature catalog:

- `library/strategies/base/features.py` separates weighted-prompt behavior from
  the large required ABC surface.
- `library/strategies/shared/clip/` holds reusable CLIP-family strategy
  behavior used by more than one family.

Those are useful instincts but incomplete vocabulary:

- "optional feature" understates that a selected run may require the feature.
- "shared" only reports that multiple current users happen to reuse the code;
  it does not describe what the code is.
- keeping features as one file will eventually recreate the same pressure as
  the current aggregate contract.

The current leading organizational direction is therefore a feature catalog:

```text
library/strategies/
  base/
    contracts.py
    context.py

  features/
    conditioning/
      clip.py
      weighted_prompt.py

    diffusion/
      pixel.py
      latent.py

  sd/
  sdxl/
  sd3/
```

This shape is illustrative rather than final. It establishes three different
homes:

```text
base/contracts.py  defines the contract vocabulary and rules
features/          contains reusable ways to file parts of the contract
<family>/          assembles known family behavior into a fulfilled strategy
```

`features` is currently the strongest folder name:

- `tools` is overly generic and conflicts conceptually with repository tools.
- `capabilities` sounds like availability declarations rather than the
  implementations themselves.
- `shared` describes present reuse rather than architectural meaning.
- `features` continues the existing idea while allowing it to become a real
  organized catalog.

### Naming and discovery mechanisms to try later

The feature folder may provide enough context to avoid suffix-heavy names such
as `LatentDiffusionFeature`:

```python
from library.strategies.features.diffusion import LatentDiffusion
```

If features later become config-selectable, a Hydra-style qualified target may
carry the same context without encoding the category into every class name:

```yaml
representation:
  _target_: library.strategies.features.diffusion.LatentDiffusion
```

A decorator is another option worth testing if feature descriptors or
registration become useful:

```python
@feature(
    concern=DiffusionRepresentation,
    requires=(AutoencoderEncoding,),
    provides=(LatentDiffusionState,),
)
class LatentDiffusion:
    ...
```

A decorator should carry real contract information rather than serve only as a
cosmetic marker, and it should not replace executable protocols or conformance
tests. Import-time global registration would also need scrutiny because it can
make availability depend on import order. Explicit imports/catalogs, Hydra
targets, decorators that attach descriptors, and ordinary class metadata
should all remain candidates until the feature contract is concrete enough to
compare them.

## Pixel And Latent Diffusion As Features

Pixel and latent diffusion initially appeared as possible strategy subclasses:

```text
DiffusionTrainingStrategy
├── PixelDiffusionStrategy
└── LatentDiffusionStrategy
```

That route was rejected as too rigid. Pixel versus latent is a training
representation choice, not necessarily the permanent identity of a model
family. A run may experiment with a different representation, attach a bridge
or adapter, or compose components in a way the family's default assembly does
not use.

The stronger direction is a representation feature selected inside the family
strategy's contract filing:

```python
class PixelDiffusion:
    def prepare_clean_state(self, batch, context):
        ...


class LatentDiffusion:
    def __init__(self, autoencoder):
        self.autoencoder = autoencoder

    def prepare_clean_state(self, batch, context):
        ...
```

This does not imply arbitrary compatibility. A pixel feature that provides a
pixel-space state cannot be paired with a predictor that requires an
incompatible latent state unless another selected feature explicitly bridges
the two. Contract fulfillment must detect that mismatch.

## CLIP: Component Versus Feature

`clip.py` is potentially ambiguous because CLIP can refer to several different
things:

- the neural component implementation
- tokenizer/bootstrap behavior
- prompt tokenization
- weighted prompts
- text encoding
- hidden-state selection and pooling
- construction of a conditioning payload

The current direction is to distinguish them by ownership:

```text
models/       owns a CLIP component definition or low-level component adapter
strategies/   owns how CLIP is tokenized, encoded, weighted, and composed here
```

Therefore a path such as `features/conditioning/clip.py` or a larger
`features/conditioning/clip/` package is clearer than a top-level
`features/clip.py`.

Whether to keep using the installed `transformers` implementation or host a
CLIP definition in the repository is a separate decision. Vendoring may offer
stable forward semantics, modification control, tensor inspection, and exact
checkpoint behavior, but it also adds maintenance, upstream compatibility,
licensing, and security-update responsibility.

A plausible intermediate route is a repository-owned component interface or
adapter over the third-party implementation. Repository-owned strategy
features would depend on that interface. A hosted implementation could replace
the third-party component later without rewriting every family strategy.

## The Long-Term Component Grab Box

The likely end-game model system is broader than a strategy feature catalog:

```text
model component catalog
├── autoencoders
├── text encoders
├── vision encoders
├── language models
├── UNets
├── diffusion transformers
├── connector/projection modules
└── adapters

strategy feature catalog
├── representations
├── conditioning
├── prediction behavior
├── caching behavior
├── preparation behavior
└── persistence behavior
```

In that eventual system, strategies could be assembled from different
autoencoders, encoders, language models, predictors, and connectors. That is an
end-game direction rather than immediate scope because constructing the Python
objects is the easy part. Compatibility must account for:

- tensor shapes and channel counts
- latent scale and normalization conventions
- tokenizer/encoder pairing
- conditioning dimensions and sequence conventions
- prediction parameterization
- checkpoint namespaces and conversion
- supported precision and execution modes
- provenance and durable component identity
- trainable combinations
- artifact serialization

Current SD, SDXL, and SD3 family assemblies should remain known, tested
combinations. Near-term feature APIs should nevertheless accept components
through meaningful interfaces rather than hard-coding one global component
implementation. That preserves an evolution path without pretending arbitrary
component assembly is safe today.

## Model, Strategy, Mode, Objective, And Trainer Boundaries

The feature direction does not erase the existing major axes.

### Model system

Owns what components are and how they work in general:

- neural module definitions and component wrappers
- component construction and artifact loading
- checkpoint conversion and low-level serialization
- family-declared loaded component identities
- low-level component quirks independent of a particular workflow
- eventually, component compatibility facts and replaceable implementations

### Strategy system

Owns how the repository uses components to fulfill the training contract:

- family assembly
- tokenization and encoding behavior
- representation and conditioning construction
- mapping contract-level prediction inputs onto the family forward signature
- family cache representation and encoding behavior
- family validation and sampling behavior
- family-facing preparation and persistence coordination

### Training mode

Owns the selected form of training:

- base fine-tuning
- adapter attachment and training
- combined base/adapter training where supported
- newly attached trainable components
- future training-subject arrangements

### Objective

Owns the learning problem where it is genuinely independent of family:

- corruption/noise-level construction
- target construction
- DDPM, rectified-flow, distillation, or other objective semantics
- objective-specific observations and adaptive state

### Trainer

Owns temporal and infrastructure policy:

- phase ordering
- epochs, steps, accumulation, backward, and optimizer advancement
- distributed coordination and generic device lifecycle
- trigger scheduling
- logging, observation, interruption, and cleanup

The contract must state how these participants meet. It remains open whether
the trainer will eventually receive one fulfilled training strategy that
contains family integrations plus mode/objective collaborators or continue
receiving those axes as separate objects that jointly fulfill one contract.

## Trainer And Strategy State

An intermediate proposal introduced a trainer-owned `ModelRuntime` object.
That name and ownership were not convincing because it appeared to add another
abstraction while preserving the same back-and-forth.

The underlying problem is real:

```text
trainer currently owns
  loaded components, VAE/TE/denoiser projections, dtypes, objective, flags

strategy currently owns
  tokenizers and some family-specific loading/checkpoint facts

then trainer passes many of those values back into strategy calls
```

The current leading direction is an explicitly bound, per-run strategy:

```python
strategy.load(load_context)
strategy.prepare(preparation_context)
result = strategy.training_step(step_context)
```

Loaded model state may still have a typed internal value such as
`LoadedModel`, but it would be deliberately owned by the bound strategy rather
than returned to the trainer and repeatedly passed back. The trainer and
training mode would access only the component/query surface justified by the
contract.

This direction is not settled enough for implementation. The following must be
resolved first:

- how modes select and manipulate trainable components without receiving the
  whole trainer
- how distributed preparation wraps or replaces loaded modules
- who owns component replacement after accelerator preparation
- how metadata observes loaded state without becoming its owner
- how checkpointing receives both family state and trainer-owned coordinates

## What A Strong Contract Must Do

The current ABCs mostly prove that methods with particular names exist. Heavy
use of `Any`, positional lists/tuples, broad `cfg`, and whole-`Trainer`
arguments prevents them from enforcing architectural meaning.

A stronger contract needs:

1. **Typed concerns and results**
   Family-internal payloads may differ, but trainer-facing results must have
   stable meaning.
2. **Explicit fulfillment rules**
   Required selections and conditional requirements must be known before the
   run starts.
3. **Compatibility validation**
   Selected features must agree on representation, tensor semantics, component
   roles, objective expectations, and persistence support.
4. **Lifecycle/state rules**
   Loading, preparation, execution, replacement, and saving must have explicit
   valid states.
5. **Conformance tests**
   Every active family must pass the same applicable contract tests, plus
   feature-specific and family-specific suites.
6. **No nominal placeholders**
   An unsupported filing must be absent or rejected, not represented by a
   method that exists only to raise later or silently do unrelated work.
7. **Discoverable internal composition**
   Family features should be wired explicitly rather than found accidentally
   through a large multiple-inheritance namespace.

Different model implementations should still do different things internally.
The contract constrains their observable interaction with the pipeline, not
their neural architecture.

## Routes Considered And Why The Discussion Turned

### Route: preserve the current strategy internals because they already work

Initial discussion treated operations such as `process_batch()` as stable
because the trainer calls them today.

Reaction: this felt circular and unprincipled. Reading the actual code showed
that `process_batch()` mixes representation preparation, conditioning,
objective work, denoiser invocation, target construction, and loss policy.
Validation separately repeats much of the same flow despite `process_batch()`
claiming to cover training and validation. Existing call sites alone do not
prove the responsibility boundary is correct.

Conclusion: preserve the successful strategy idea, not every current method or
facet boundary.

### Route: replace the current system with a wholly new abstraction

Several alternatives initially sounded like replacements for the strategy
system even though the current contract → strategy fulfillment → trainer
principle had not been disproven.

Reaction: these felt like architectural downgrades because they discarded
weeks of useful separation without identifying a stronger principle.

Conclusion: improve the existing model/strategy/trainer route first. A new
shape must explain what it preserves, what it fixes, and why.

### Route: rename the aggregate as a prepared program

A `PreparedTrainingProgram` façade was proposed with entry points such as
`train_step()`, `validate()`, and `save()`.

Reaction: this was only another synonym for strategy. It did not introduce new
semantics or solve contract/state ownership.

Conclusion: rejected. Renaming an imperative service object is not an
architecture.

### Route: use a typed strategy graph

A real graph would be distinct if operations, inputs, outputs, and dependencies
were declarative and generically executable.

Reaction: this was finally different rather than a synonym, and it aligns with
long-term node-oriented UI interests. It also introduces a much larger design
space before the existing architecture is understood well enough.

Conclusion: deferred. The present work deliberately focuses on the
model/strategy/trainer route. Nothing here requires or forbids a future graph.

### Route: split one small core contract from optional capabilities

This attempted to avoid forcing future models to implement irrelevant methods.

Reaction: almost any current contract item could be called optional. The
result risked becoming a weak bag of methods rather than a definition of what
the training pipeline accepts.

Conclusion: replace required/optional vocabulary with contract concerns,
available features, selected filings, conditional requirements, compatibility,
and fulfillment.

### Route: define latent and pixel diffusion as strategy subclasses

This made category differences explicit.

Reaction: it still fixed an experimental training choice into the permanent
identity of the family strategy. A model may be trained through an adapter,
distillation, an alternate representation, or some combination that does not
fit one inheritance label.

Conclusion: treat pixel/latent behavior as selectable feature implementations
inside a contract filing, not necessarily as the family integration's class.

### Route: organize reusable behavior as a feature grab box

This began from the observation that difficult systems often become tractable
once their distinct concepts have honest, discoverable homes. The strategy
system itself succeeded by applying that principle to the original scripts.

Reaction: this direction felt additive rather than destructive. It explains
the existing `features.py` and `shared/clip/` experiments, preserves family
strategies as tested assemblies, supports current variation, and opens a route
toward later component composition.

Conclusion: this is the current leading direction, subject to contract and
compatibility research before implementation.

## Decisions, Leanings, And Open Questions

### Decisions strong enough to carry into an OpenSpec

- Preserve the model → strategy → trainer architectural route.
- Treat the contract as the pipeline acceptance definition.
- Require strategies to fulfill the applicable selected contract rather than
  nominally inherit one universal SD-shaped surface.
- Keep model internals free to differ behind the strategy boundary.
- Separate contract vocabulary from reusable feature implementations and
  family assemblies.
- Treat a selected feature as required for that run; do not use "optional" as
  an escape from fulfillment.
- Keep trainer temporal/infrastructure policy separate from family behavior.
- Keep model component mechanics separate from strategy-owned model behavior.
- Do not pursue a strategy graph during this design pass.

### Current leanings that still need pressure testing

- Promote `base/features.py` into an organized `features/` catalog.
- Reclassify `shared/clip/` by semantic feature rather than by current reuse.
- Represent pixel and latent diffusion as feature implementations rather than
  family strategy subclasses.
- Make the active per-run strategy explicitly bound to its loaded model state.
- Replace multiple-inheritance discovery with more explicit feature wiring
  inside family strategies.
- Preserve SD/SDXL/SD3 assemblies as known tested defaults while designing
  injection seams for future component substitution.

### Open questions

- What is the minimum contract vocabulary that accurately describes the
  current training pipeline without encoding SD-specific anatomy?
- Which concerns belong to a family integration, training mode, objective, or
  the fulfilled training strategy?
- Does one aggregate strategy ultimately contain mode and objective
  collaborators, or does the trainer receive separate objects that jointly
  satisfy one contract?
- What exact loaded-state surface replaces the current trainer/strategy split?
- Which current methods are trainer-facing contract operations, reusable
  features, family-local helpers, or misplaced model/data/performance logic?
- How should features declare what they provide, require, and accept without
  building a premature general-purpose dependency system?
- What compatibility facts are structural and testable now?
- Where should cache codecs and cache IO live when representation features own
  encoding semantics but the data system owns persistence orchestration?
- When is a third-party component adapter sufficient, and when should a model
  component be hosted in the repository?
- How should contract evolution be recorded if external strategy plugins are
  supported later?

## Recommended Research Before A Formal Change

Before implementation, the current code should be inventoried against the new
vocabulary:

1. List every method in `base/contracts.py` and every caller.
2. Classify each method as:
   - trainer-facing concern
   - conditional contract concern
   - reusable feature implementation
   - family-local collaboration
   - model/component mechanic
   - objective/loss behavior
   - data/cache infrastructure
   - trainer/mode orchestration
3. Map the complete SD, SDXL, and SD3 training, validation, caching, sampling,
   preparation, and checkpoint flows.
4. Identify the smallest stable trainer-facing inputs and results.
5. Define known feature compatibility constraints from current families.
6. Pressure-test the vocabulary against:
   - current latent SD/SDXL/SD3 paths
   - a pixel diffusion path
   - adapter plus base-model training
   - an added trainable component
   - distillation
   - video/audio tensor shapes
   - a model with an integrated or external LLM component
7. Write conformance scenarios before choosing the final class/package shape.

Only after that inventory should an OpenSpec lock down migrations. The first
implementation should improve the current three families and trainer boundary;
it should not attempt the end-game arbitrary component catalog at the same
time.

## Current Direction In One View

```text
                           TRAINING CONTRACT
                  vocabulary + rules + expected results
                                   │
                  ┌────────────────┼────────────────┐
                  │                │                │
             model parts      strategy features  run selections
             and mechanics    and family behavior mode/objective/config
                  │                │                │
                  └────────────────┼────────────────┘
                                   ▼
                      FULFILLED TRAINING STRATEGY
                  family integrations, selected features,
                  validated choices, and bound loaded state
                                   │
                                   ▼
                                TRAINER
                   lifecycle, optimization, distributed
                      execution, triggers, observation
```

The feature grab box is not the entire design. It is the organizational layer
that may let the contract remain expressive without turning either
`contracts.py` or every family folder into another architectural thicket.
