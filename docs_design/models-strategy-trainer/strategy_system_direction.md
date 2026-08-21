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

Within this pre-OpenSpec work, this document is the normative source for the
current architectural direction. `strategy_system_inventory.md` retains code
evidence and pressure tests; `notes.md` retains chronological checkpoints and
may therefore include positions that were later superseded here.

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

The ownership principle that produced the current shape was:

```text
trainer       owns when the run does something
training mode owns the selected form of training
strategy      owns how this repository uses a model family
models        own model components and their mechanics
```

That separation was a useful improvement over older adapter-specific and
fine-tune-specific strategy classes, but it is no longer the target. It leaves
multiple runtime authorities coordinating one training definition. The
direction now being explored collapses that interaction to:

```text
trainer   owns the training mechanism and defines its acceptance contract
strategy  defines what is being trained and how it uses that mechanism
models    own model components and their mechanics
```

Training approach, objective behavior, model-family behavior, and reusable
features may all participate in authoring one strategy. They do not
automatically require parallel top-level runtime objects.

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
that make a training strategy usable by the repository.

A model itself may have any internal definition. It can be a UNet, DiT,
autoencoder, pixel model, multimodal system, model containing an LLM, or an
arrangement not anticipated when this document was written. A strategy is the
repository-authored definition of what is being trained and how. It uses
whatever model components, objectives, features, and Trainer-recognized
capabilities are needed to fulfill the pipeline contract.

The contract does not need to predict every future model. It needs to express
what the current pipeline is deliberately receptive to. When the pipeline
learns a new arrangement, the contract vocabulary and its conformance tests
expand deliberately.

## Authored Fulfillment, Not Automatic Assembly

The overall construction goal has not changed from the current strategy
system: repository developers write the strategies that the trainer receives.
The new contract mechanics are meant to make those strategies intentional,
constrained, testable, and discoverable, not to replace their authors with a
resolver.

```text
handwritten strategy definition
  ├── chooses the concrete training intent and model components
  ├── wires the features and capabilities needed by that strategy
  ├── defines any bounded choices exposed through configuration
  └── states which core contract and capabilities it fulfills
                    │
                    ▼
          contract validation
                    │
                    ▼
        validated strategy
                    │
                    ▼
                 trainer
```

The run does not inspect a model and infer a strategy. Configuration does not
silently assemble features or drag undeclared dependencies into the strategy.
Execution configuration may request use of a Trainer-recognized capability
already provided by the authored strategy—for example, sampling with a
particular cadence and request settings. Validation then confirms that the
authored strategy provides a compatible implementation. Configuration may
choose between internal arrangements only when the strategy author explicitly
wrote, exposed, and constrained that choice. The initial construction
mechanism remains ordinary, explicit Python in the repository's strategy
builders/factories.

After fulfillment, the trainer interacts with the complete strategy through
the trainer-facing contract. It should not coordinate the features or model
components used to build that strategy. Composition remains behind the
strategy boundary even when the implementation uses reusable catalog entries.

Validation has the narrower job of proving that the authored arrangement is
complete and mutually compatible. If a latent representation requires an
autoencoder behavior, the strategy author wires both; validation may reject a
missing or incompatible pairing, but it must not invent the pairing.

This makes the maintained SD, SDXL, and SD3 strategies repository-authored,
tested definitions of what and how the repository trains in those standard
cases. Model family is an ingredient of those strategies, not the definition
of the strategy abstraction. The feature and eventual component catalogs are
tools available to strategy authors, not an automatic dependency resolution
system. A future declarative or configuration-driven authoring layer might
target the same contract, but it is outside the current direction.

## Conformance Without A "Noob Mode"

The contract exists to make a strategy interoperable with the training
pipeline, not to restrict the experiments the repository is allowed to
perform. It should distinguish these conceptual levels without prematurely
requiring them to become separate Python classes:

```text
active trainer contract
  the trainer-facing interaction and results a compatible strategy must supply

standard repository strategy vocabulary
  the concerns, features, lifecycle rules, and constraints used by maintained
  family strategies

research/custom strategy surface
  deliberate replacements, additions, or direct implementations that either
  satisfy the active trainer contract or explicitly target an extension of it
```

A custom strategy may replace a standard feature, introduce a teacher and
student, attach a new trainable component, use an unusual representation, or
implement a trainer-facing operation directly when the standard decomposition
does not fit. It must not claim conformance to a trainer contract whose
observable requirements it does not satisfy. If the experiment changes what
the trainer itself must do, that is an explicit contract extension or version,
not a nominal implementation of the unchanged contract.

This is not the previously rejected unstructured “small required core plus an
optional method bag.” The core is dictated by the intended Trainer.
Trainer-recognized capabilities have named request, result, lifecycle, and
compatibility semantics. Within any contract surface a strategy claims—and
whenever execution requests a provided capability—applicable concerns remain
real requirements. The distinction is between the maintained standard way of
fulfilling those requirements and a deliberate research implementation that
reaches the same trainer boundary—or openly extends it.

## Contract As Vocabulary, Not A Universal Checklist

An early framing divided strategy behavior into required methods and optional
capabilities. That framing was rejected because almost any current contract
method could be moved into an "optional" list, producing a loose grab bag
instead of a meaningful contract.

The stronger framing is:

```text
Contract = vocabulary + required selections + compatibility rules
Strategy = an authored and validated filing of that contract
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

Some requirements are conditional on the authored arrangement or on a bounded
choice that its strategy explicitly exposes:

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

A feature is not "optional" merely because every strategy does not use it. Once
the strategy author wires it into an arrangement, or makes its use part of a
bounded request exposed to execution configuration, the completed strategy
must fulfill the resulting requirements or validation must reject the
configuration before training.

## Three Contract Surfaces

The contract system is broader than one base class. It needs to define three
different surfaces whose consumers and obligations are different:

```text
training contract system
│
├── core strategy contract
│     the minimum behavior the intended Trainer requires to train
│
├── Trainer-recognized capabilities
│     additional operations the training pipeline knows how to coordinate
│
└── strategy feature contracts
      author-facing building blocks used to implement the core or capabilities
```

The dividing rule is based on who consumes the behavior:

```text
Trainer or delegated pipeline orchestration consumes capabilities.
Strategy implementations consume features.
```

“Trainer-recognized” does not mean “implemented inside the `Trainer` class.”
The Trainer may delegate traversal, storage, inference, or publishing
orchestration to the data, checkpointing, or pipeline systems. It means the
active training pipeline understands the capability's request, lifecycle, and
result semantics.

### Core strategy contract

The core is dictated by the Trainer being designed, not mechanically extracted
from today's Trainer implementation. It is the minimum every compatible
strategy must provide for that Trainer to perform training at all.

The current starting hypothesis is that the core must establish exchanges for:

- the concrete model/training arrangement used by the run;
- runtime preparation required before execution;
- training subjects and optimization inputs;
- execution of a training batch and production of the result the Trainer needs.

This is a semantic hypothesis, not an accepted method list. “Establish the
arrangement” may involve declaration, loading, binding, preparation, and
rebinding rather than one large `bind()` method. The concrete exchanges must be
derived before naming methods or classes.

### Trainer-recognized capabilities

A capability is a named interaction the training pipeline knows how to
coordinate. Capability does not mean unimportant or freely optional. The core
may require a compatible capability from a category without requiring every
strategy to use the same implementation. A strategy may also provide
capabilities that its concrete training intent does not require merely to
compute an update.

Current examples include:

- validation or evaluation;
- sampling or generation;
- representation and conditioning caching;
- resumable runtime-state contribution;
- trained-artifact persistence.

Training arrangements such as base fine-tuning, adapter attachment and
training, combined base/adapter training, distillation, or newly attached
trainable components may also be expressed through capabilities when that
makes the Trainer interaction explicit. This is intentionally not a reason to
create adapter-specific or fine-tune-specific strategy classes.

The authored strategy either provides a named compatible filing or it does
not. The filing may select an implementation, supply typed information for
Trainer-owned mechanics, or both; that ownership boundary remains to be
derived from the concrete code. There should be no raising default pretending
to provide it, no `hasattr()` discovery, and no family-name branch acting as
the real support declaration.

Execution configuration may request that a provided capability be used. It may
set cadence, destination, prompts, formats, or other request parameters.
Configuration does not thereby construct the strategy or choose an undeclared
internal implementation. If the request cannot be satisfied by the authored
strategy, validation rejects it before execution.

### Strategy feature contracts

Features are reusable strategy-authoring vocabulary. Examples include:

- pixel or latent representation behavior;
- CLIP tokenization, encoding, and conditioning behavior;
- weighted-prompt behavior;
- predictor invocation adapters;
- representation or conditioning cache codecs;
- adapter execution behavior;
- objective-integration behavior.

The Trainer does not discover, enumerate, or coordinate these features.
Repository authors explicitly use them when writing a family or research
strategy. A capability may itself be composed from several features:

```text
sampling capability
  ├── conditioning feature
  ├── predictor adapter
  ├── representation decoder
  └── scheduler integration

caching capability
  ├── representation cache codec
  ├── conditioning cache codec
  └── component signatures
```

Features may help fulfill the core, a capability, or both. Their existence does
not automatically enlarge the Trainer-facing API.

### Contract evolution

The three surfaces give contract growth an intentional meaning:

```text
the Trainer gains a new universal requirement
  → evolve or version the core strategy contract

the training pipeline learns an additional operation
  → add a named Trainer-recognized capability

strategy authors gain a reusable implementation building block
  → add a feature contract and implementation
```

This permits comprehensive vocabulary without requiring every strategy to
implement every operation. It also avoids treating any new feature as a reason
to change the Trainer.

## Lifecycle Is An Orthogonal Axis

Core, capability, and feature describe what kind of contract something is.
They do not describe runtime readiness. The strategy arrangement and its
provided capabilities move through lifecycle states such as:

```text
authored definition
  → validated arrangement
  → loaded/bound runtime
  → prepared execution runtime
```

These states do not require four public Python classes. They require explicit
validity rules:

- what facts and behavior are available in each state;
- which transitions the pipeline coordinates;
- which failures must occur before expensive work;
- which logical identities survive runtime replacement;
- which prepared execution bindings must receive forward-like calls.

The lifecycle applies across all three surfaces. A sampling capability may be
declared and validated before its components are loaded, then become executable
only after the strategy has accepted prepared bindings. An internal
conditioning feature follows the same runtime state even though the Trainer
never calls it directly.

This keeps two established distinctions visible:

```text
logical model/component binding
  identity, source, relationships, metadata, and artifact semantics

prepared execution binding
  the live wrapper/module that must execute in the current runtime
```

The exact state container and transition API remain design work. The semantic
distinction does not.

## Runtime State And Artifact Persistence

Runtime checkpointing and model artifact persistence must not become one
generic `Checkpointing` capability.

```text
runtime state
  resume this execution
  optimizer, scheduler, RNG, scaler, loop and distributed state

artifact persistence
  save or publish this logical trained result
  adapter weights, family checkpoints, Diffusers layouts, metadata, provenance
```

Runtime-state orchestration primarily belongs to Trainer and distributed
infrastructure. A strategy may need to contribute state participants,
extraction behavior, and restoration semantics; that does not make the
strategy the owner of the overall runtime checkpoint.

Artifact persistence is more directly a strategy/model-family capability.
Trainer-owned orchestration still supplies timing, coordinates, destinations,
and publishing policy. The capability owns family/model-specific extraction,
conversion, layout, and artifact results.

The precise runtime-state contribution contract remains to be derived. The two
meanings are separate from the beginning.

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
adapter execution behavior. Features are consumed by authored strategy
implementations rather than coordinated directly by the Trainer.

### Capability

A named operation the training pipeline understands how to coordinate, together
with its request, result, compatibility, and lifecycle semantics. A strategy
may provide the capability through one or more explicitly wired features.
Availability is not a substitute for the author choosing and fulfilling an
arrangement, and a configured request does not choose the implementation.

### Fulfillment

The validated state of a strategy whose required concerns have providers and
whose providers are mutually compatible. Fulfillment is a status, not another
runtime noun: the Trainer still receives a strategy.

### Family integration / family strategy

The known, tested behavior through which one model family and its components
participate in strategy authoring. Current classes such as
`SdxlTrainingStrategy` combine this role with most of the strategy itself.
Family integration may remain useful internal vocabulary, but strategy no
longer means model family.

### Training strategy

The explicit definition of what is being trained and how it interacts with the
Trainer. A normal strategy may use one familiar model family, while a compound
strategy may involve a teacher, student, adapters, newly attached components,
and a distillation objective. Fine-tuning and adapter training are possible
capabilities or ingredients of a strategy; neither should create a parallel
strategy hierarchy.

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

- "optional feature" understates that an authored strategy arrangement may
  require the feature.
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
<family>/          explicitly assembles known family behavior into a strategy
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

If a future authoring layer makes features config-selectable, a Hydra-style
qualified target could carry the same context without encoding the category
into every class name:

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
compare them. None of these mechanisms implies that the current work should
build an automatic strategy assembler.

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

The stronger direction is a representation feature deliberately wired inside
the strategy's contract filing, or exposed by that strategy as a bounded
supported choice:

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
the third-party component later without rewriting every affected strategy.

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

## Model, Strategy, And Trainer Boundaries

The present `TrainingMode` and separate objective runtime are source evidence,
not assumed permanent axes.

### Model system

Owns what components are and how they work in general:

- neural module definitions and component wrappers
- component construction and artifact loading
- checkpoint conversion and low-level serialization
- family-declared loaded component identities
- low-level component quirks independent of a particular workflow
- eventually, component compatibility facts and replaceable implementations

### Strategy system

Owns the authored definition of what is being trained and how:

- selected model/component arrangement
- selected training-subject treatment
- selected objective behavior
- tokenization and encoding behavior
- representation and conditioning construction
- mapping contract-level prediction inputs onto the family forward signature
- capabilities and features intentionally used by the strategy
- typed information required by Trainer-owned mechanics
- strategy-specific execution and artifact behavior

### Trainer

Defines and executes the training mechanism:

- the core strategy contract and recognized capability contracts
- phase ordering
- epochs, steps, accumulation, backward, and optimizer advancement
- distributed coordination and generic device lifecycle
- trigger scheduling
- logging, observation, interruption, and cleanup

`TrainingMode` should not remain a second top-level authority beside the
strategy. Its current behavior must be classified by responsibility:

```text
strategy
  deliberately selects and describes the training treatment

Trainer / optimization system
  performs generic trainable realization, preparation, and optimization

capability or domain implementation
  performs specialized behavior where putting it directly in Trainer would
  create technique-specific branches
```

Whether a particular behavior such as adapter attachment belongs directly in
Trainer, in a Trainer-owned capability handler, or in a reusable feature
cannot be decided for `TrainingMode` as one block. Its current methods combine
several responsibilities. Objective behavior is likewise something the
strategy deliberately uses to fulfill the contract, not automatically another
orchestration authority passed the whole Trainer.

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

The current leading direction is one per-run strategy boundary. The following
calls are illustrative lifecycle meanings, not an
accepted method list or a plan to reproduce another framework's hook API:

```python
strategy.load(load_context)
strategy.prepare(preparation_context)
result = strategy.training_step(step_context)
```

Loaded model state may still have a typed internal value such as
`LoadedModel`. The exact holder remains undecided. The requirement is one
authoritative answer for each stable logical binding and current prepared
execution binding, rather than contradictory copies distributed across
Trainer, strategy, and mode. Trainer-owned mechanics may receive the narrow
concrete participants required by the contract without learning
family-specific anatomy.

The trainer-facing contract is conceptually dictated by the intended Trainer:
the consumer defines what it needs, and strategies implement it. This does not
mean copying every dependency of the current `Trainer` into the contract.
Current VAE/text-encoder/denoiser knowledge may itself be architecture leakage
that the intended Trainer should no longer have.

This direction is not settled enough for implementation. The following must
be resolved first:

- how a strategy declares its intended training subjects while
  Trainer/optimization mechanics realize trainable parameters
- how distributed preparation wraps or replaces loaded modules
- who owns component replacement after accelerator preparation
- how metadata observes loaded state without becoming its owner
- how checkpointing receives both family state and trainer-owned coordinates

## Code Pressure-Test Conclusions

A source-level pass over the active binding, preparation, optimization, step,
caching, validation, and persistence paths confirms the direction above. It
also narrows the remaining problem: the repository does not lack all useful
contract values. It has several typed islands that are still connected through
whole-`Trainer` mutation and diffusion-shaped projections.

Useful foundations already exist:

- `LoadedModelComponent` gives each family a common declared top-level
  component surface while allowing different component counts and names.
- `OptimizationPlan` distinguishes logical parameter groups from executable
  optimizer groups.
- `OptimizerBuildResult` and `ObjectiveRuntime` are early typed exchanges
  rather than anonymous tuples.
- the training loop already gives Trainer temporal and infrastructure
  ownership over accumulation, backward, clipping, optimizer advancement,
  triggers, reporting, and cleanup.

The intended migration should evolve these seams rather than introduce an
unrelated all-purpose runtime container.

### Binding state needs two module meanings

`Trainer.setup()` currently asks the family strategy to load components, stores
the returned `LoadedModelComponent` tuple, and becomes the authoritative owner
of it. Trainer properties then project that tuple back into
`vae`/`text_encoders`/`denoiser` and later pass those modules into strategy
calls.

The component declaration is a useful logical foundation, but its single
`module` field currently represents both:

```text
logical/original component binding
prepared execution binding
```

Accelerator preparation replaces that field with the prepared object. The
family-local component key survives, but the contract cannot independently
describe the logical component and the object whose forward path must execute.
The bound-state design therefore needs stable component identity plus distinct
logical and prepared bindings. This can be an evolution of the existing
component collection; it does not justify adding a generic `ModelRuntime`
holder that merely relocates the same ambiguity.

### Preparation must stop mutating the whole Trainer

Both active modes receive the complete Trainer, choose concrete modules, call
`accelerator.prepare()`, replace modules/optimizer/scheduler, and set the
gradient-synchronization handle and primary trainable. This proves that
preparation is a genuine exchange rather than a family-local hook:

```text
training integration
  publishes preparation participants and constraints
                 ↓
Trainer-owned infrastructure
  prepares concrete execution objects
                 ↓
prepared-binding result
                 ↓
training integration accepts authoritative execution bindings
```

Trainer-owned infrastructure may need concrete objects without learning
family-specific roles. The strategy's declaration—and, today, mode
behavior—may decide which participants matter without owning the distributed
preparation operation.

### Optimization already has the beginning of its contract value

Fine-tune and adapter modes currently select trainables, create logical and
execution groups, construct an `OptimizationPlan`, and also materialize the
optimizer. Trainer then creates the scheduler, coordinates preparation, and
performs every ordinary optimizer step.

This supports a leading split:

```text
authored training integration
  selects training subjects
  publishes the optimization plan and constraints

Trainer
  applies optimizer/scheduler configuration
  materializes and prepares the optimization runtime
  owns the standard backward/step/zero-grad lifecycle
```

The exact ownership profile remains a contract decision, especially for
research strategies that deliberately take over optimization. The existing
`OptimizationPlan` should nevertheless be treated as a likely evolutionary
base rather than discarded.

### The current batch result is not the final optimization loss

The standard loop does not backpropagate through `BatchLossOutput.loss`.
It passes `per_sample_loss` through the Trainer-owned loss-modifier runtime and
backpropagates the modifier result. The `loss` field is subsequently used for
accounting. A future step result must therefore distinguish:

```text
base differentiable loss state
objective-specific observations
metrics
strategy-owned state changes
Trainer-derived final optimization loss
```

Current `timesteps` are useful objective observations, not a universal fact
about training a neural network. The step exchange must not make diffusion
timesteps part of the minimum contract.

### `TrainingMode` should not remain a parallel authority

`TrainingMode` currently receives the whole Trainer for trainable selection,
adapter attachment, parameter grouping, distributed preparation, train/eval
transitions, runtime-state hooks, and artifact routing. `ObjectiveRuntime` is
Trainer-owned, but family strategies also inspect and invoke its
objective-specific behavior.

The intended replacement is not an adapter-specific strategy or another
aggregate object around the existing three participants. Strategy already
means the authored definition of what is trained and how. Fine-tuning, adapter
training, objective behavior, and model-family behavior become deliberately
selected capabilities/features or contract filings within that definition.
The Trainer receives the strategy and executes its contract.

The current mode methods still need individual placement. Generic
trainable/optimizer realization, distributed preparation, and temporal
lifecycle behavior lean toward Trainer or its delegated optimization systems.
Specialized attachment, state extraction, and persistence behavior may need
capability or domain implementations. Capability recognition does not by
itself settle implementation placement.

The responsibility classification in
[`strategy_system_inventory.md`](strategy_system_inventory.md) Milestone 6
resolves the former mode/objective participation blocker. Two semantic
blockers remain before concrete API design:

1. define authoritative bound state and its logical/prepared identities;
2. define the standard optimization-ownership profile and how an explicit
   research profile may extend it.

The classification does not prescribe one code location for every specialized
capability. It establishes that `TrainingMode` dissolves, objective behavior
is selected within strategy authoring, and generic preparation/optimization
mechanics belong to Trainer or its delegated infrastructure.

## What A Strong Contract Must Do

The current ABCs mostly prove that methods with particular names exist. Heavy
use of `Any`, positional lists/tuples, broad `cfg`, and whole-`Trainer`
arguments prevents them from enforcing architectural meaning.

A stronger contract needs:

1. **Typed concerns and results**
   Family-internal payloads may differ, but trainer-facing results must have
   stable meaning.
2. **Explicit fulfillment rules**
   Authored selections, deliberately exposed runtime choices, and their
   conditional requirements must be known before the run starts.
3. **Compatibility validation**
   Authored features and any explicitly supported runtime choice must agree on
   representation, tensor semantics, component roles, objective expectations,
   and persistence support.
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
   through a large multiple-inheritance namespace or inferred by an automatic
   dependency resolver.

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
available features, authored filings, deliberately exposed runtime choices,
conditional requirements, compatibility, and fulfillment.

Later discussion recovered a narrower and more principled meaning for
capability. The three-surface contract model does not revive the rejected
optional-method bag:

```text
core
  obligations genuinely required by the intended Trainer

capability
  a named operation the pipeline knows how to request and validate

feature
  an internal building block consumed by strategy authors
```

A capability is not an excuse for a missing core obligation. When execution
requests a capability, its compatible implementation becomes a real validated
requirement. This consumer-based distinction is the additional principle the
earlier proposal lacked.

### Route: define latent and pixel diffusion as strategy subclasses

This made category differences explicit.

Reaction: it still fixed an experimental training choice into the permanent
identity of the strategy. A model may be trained through an adapter,
distillation, an alternate representation, or some combination that does not
fit one inheritance label.

Conclusion: treat pixel and latent diffusion as reusable representation
implementations. A strategy author explicitly chooses and connects one when
implementing the training contract; the choice does not need to define the
strategy's inheritance hierarchy.

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

### External analogy: authored recipe systems

The comparison to [`ljleb/sd-mecha`](https://github.com/ljleb/sd-mecha) is
useful at a base level even though that project executes model-merging recipes
rather than training strategies. The attractive part is not a graph or
automatic resolver. It is the separation between reusable operations,
explicitly authored composition, validation of that composition, and a shared
executor.

The corresponding direction here is:

```text
reusable model/strategy features
        + handwritten standard or research strategy
        + contract validation
        → shared trainer execution
```

This is an analogy, not an implementation template.

### Route: adopt Lightning or Fabric instead of refining this system

The proposed Trainer → strategy relationship was recognized as
structurally similar to Lightning Trainer → LightningModule. That raised a
necessary challenge: a custom `training_step()` plus lifecycle hooks,
optimizer configuration, distributed preparation, validation, callbacks, and
checkpoint machinery could become Lightning under different names.

This comparison is subordinate to the direction already established in this
document. It tests whether an external implementation or pattern fits the
repository's contract → strategy → trainer design. It does not reopen that
direction, replace its vocabulary, or grant another framework authority over
the problem definition.

Reaction: the similarity is real and should constrain our design. Lightning is
valuable prior art for the authored-recipe/executor boundary. Fabric is a
different layer: it leaves the loop in application code and supplies explicit
distributed/runtime operations, which is closer to the role Accelerate already
plays here.

The implementation comparison makes the analogy narrower. A LightningModule is
not consumed through one isolated contract: it has a reference back to Trainer,
participates in a dynamic hook system, obtains optimizers/loggers/runtime state
through Trainer, and is paired with separate original and prepared module
identities inside Lightning's distributed Strategy. Manual optimization
explicitly transfers backward/step ownership into the module. Fabric and
Accelerate also return prepared execution wrappers that callers must rebind.

The strongest reusable result is therefore not `training_step()` as an API. It
is the requirement to distinguish stable logical model identity from prepared
execution bindings, and to define a generic preparation plan/result exchange
between strategy and trainer-owned infrastructure.

Adoption is not automatically preferable to custom code. If an external system
constrains the repository's contract, phase model, state ownership, component
arrangements, or research escape routes, it does not serve the design. Vendored
or forked code would also make this repository responsible for maintaining a
larger implementation and incorporating upstream changes.

Conclusion: none of these implementations is a drop-in answer to the
repository's model-family training contract. Use them as source-level
references for concrete boundary mechanics. Do not adopt, vendor, or reproduce
their complete object models during this design pass. Reconsider a dependency
or separable subsystem only if it preserves the repository's design authority
and passes an explicit control-versus-maintenance test. The detailed comparison
is recorded in
[`framework_pattern_comparison.md`](framework_pattern_comparison.md).

## Decisions, Leanings, And Open Questions

### Decisions strong enough to carry into an OpenSpec

- Preserve the model → strategy → trainer architectural route.
- Treat the contract as the pipeline acceptance definition.
- Define strategy as what is being trained and how, not as a synonym for model
  family.
- Treat SD, SDXL, and SD3 strategies as maintained default training
  definitions that use those model families, not as proof that strategy
  identity must be family identity.
- Do not retain `TrainingMode` as a parallel top-level runtime authority.
- Do not create adapter-specific, fine-tune-specific, or combinatorial
  strategy hierarchies. Training treatments may be capabilities/features
  deliberately selected by a strategy.
- Treat the contract system as three distinct surfaces: the minimum core
  dictated by the intended Trainer, named capabilities understood by pipeline
  orchestration, and author-facing features consumed inside strategies.
- Define capabilities by pipeline consumption and features by strategy
  consumption; Trainer recognition does not imply implementation inside the
  `Trainer` class.
- Keep strategy construction explicitly authored; validation constrains and
  verifies authored choices but does not select features or add dependencies.
- Let execution configuration request use of an authored capability without
  treating that request as strategy assembly or implementation selection.
- Require strategies to fulfill the contract surface they claim rather than
  nominally inherit one universal SD-shaped surface.
- Keep model internals free to differ behind the strategy boundary.
- Separate contract vocabulary from reusable feature implementations and
  family assemblies.
- Treat a feature wired into a strategy, or deliberately exposed as a bounded
  runtime choice, as required when applicable; do not use "optional" as an
  escape from fulfillment.
- Preserve an explicit research path: custom strategies may replace the
  standard internal decomposition while satisfying the active trainer
  contract, or target an explicit extension when the trainer boundary changes.
- Keep Trainer execution mechanics separate from strategy declarations and
  specialized model/component behavior.
- Keep one complete training strategy as the Trainer-facing integration. A
  dedicated per-run binding authority is an internal contract responsibility,
  not a second family-shaped object that Trainer coordinates beside strategy.
- Keep authoritative participant, relationship, access-view, and execution
  route state behind one contract-governed writer protocol. Loaders,
  capabilities, and Trainer infrastructure may propose typed transitions but
  must not maintain competing authoritative copies.
- Keep model component mechanics separate from strategy-owned model behavior.
- Distinguish stable logical model/component identity from the prepared
  execution handles required by distributed and precision infrastructure.
- Treat lifecycle readiness as separate from core/capability/feature
  classification and define valid authored, validated, bound, and prepared
  states without assuming one class per state.
- Require infrastructure preparation and rebinding to use generic contract
  meanings rather than Trainer knowledge of family-specific component roles.
- Keep resumable runtime-state orchestration separate from trained-artifact
  persistence; strategies may contribute runtime state without owning the
  overall runtime checkpoint.
- Do not pursue a strategy graph during this design pass.
- Do not build an automatic dependency resolver or configuration-driven
  strategy assembler during this design pass.
- Treat Lightning/Fabric/Accelerate as prior art rather than an assumed
  architecture or adoption target.
- Keep framework comparison subordinate to the repository's established
  contract → strategy → trainer direction.
- Do not grow a general hook framework or duplicate generic runtime machinery
  while defining the repository-specific contract.
- Require any future dependency, vendoring, or subsystem-adoption proposal to
  preserve repository design control and justify its ongoing maintenance cost.
- Allow implementation and migration to proceed in bounded slices without
  deliberately weakening the settled semantics. A staged migration must not
  introduce disposable one-route, invalidate-everything, or underspecified
  persistence contracts merely because they are called a first version.

### Current leanings that still need pressure testing

- Derive the minimum core from binding, runtime-preparation, optimization, and
  step exchanges before choosing method names.
- Represent sampling, validation, caching, runtime-state contribution, and
  artifact persistence through named capability contracts rather than raising
  defaults or incidental method discovery.
- Promote `base/features.py` into an organized `features/` catalog.
- Reclassify `shared/clip/` by semantic feature rather than by current reuse.
- Represent pixel and latent diffusion as feature implementations rather than
  strategy subclasses.
- Consider `participant` as the contract-level vocabulary and `component` as a
  model/system-level entity that may serve as a participant. Do not rename
  concrete APIs until the exchange design tests this distinction.
- Replace multiple-inheritance discovery with more explicit feature wiring
  inside strategies.
- Preserve SD/SDXL/SD3 assemblies as known tested defaults while designing
  injection seams for future component substitution.

### Open questions

- What is the minimum contract vocabulary that accurately describes the
  current training pipeline without encoding SD-specific anatomy?
- Which current pipeline operations belong to the universal core, which are
  named capabilities, and which are only internal features?
- Which training-subject treatments should be explicit capabilities, and what
  typed information does Trainer require to execute them?
- Which capability implementations belong directly to Trainer, to delegated
  Trainer-owned handlers, or to domain/feature implementations?
- What concrete types and scoped views implement the settled binding authority
  without exposing unrestricted lookup or family anatomy?
- How should features declare what they provide, require, and accept without
  implying that declarations automatically select or wire dependencies?
- What is the precise trainer-facing conformance boundary that both standard
  and direct/custom strategies must satisfy?
- Which trainer-facing exchanges should be behavioral calls, and which require
  typed semantic plans/results so trainer-owned infrastructure can act without
  unpacking family anatomy?
- How should a strategy expose provided capabilities in code without
  `hasattr()`, stringly typed registries, automatic assembly, or one permanently
  growing optional attribute bag?
- How should lifecycle validity be represented and tested without requiring a
  separate public class for every state?
- What runtime-state contribution must a strategy provide while Trainer and
  distributed infrastructure remain owners of the resumable checkpoint?
- How should explicit trainer-contract extensions be versioned and tested
  without turning experimentation into nominal non-conformance?
- What compatibility facts are structural and testable now?
- Where should cache codecs and cache IO live when representation features own
  encoding semantics but the data system owns persistence orchestration?
- When is a third-party component adapter sufficient, and when should a model
  component be hosted in the repository?
- How should contract evolution be recorded if external strategy plugins are
  supported later?

### Resolution status and next semantic block

The five authoritative-binding questions are now settled at the architectural
level:

```text
Q1 logical participant identity       settled
Q2 execution-binding cardinality      settled
Q3 arrangement transitions           settled
Q4 artifact-state projection         settled
Q5 authoritative binding ownership   settled
```

The remaining work under those questions is concrete representation and
exchange design, not reopening their semantics. The next separate semantic
block is optimization ownership: define the standard Trainer-owned
optimization profile, the explicit research/extension route for unusual
ownership, and the typed information exchanged between strategy and Trainer.

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

The current production-code inventory is recorded in
[`strategy_system_inventory.md`](strategy_system_inventory.md). It completes
the method, caller, family, and active-flow audit; tests the candidate
classifications and compatibility constraints in items 2, 4, and 5; and
identifies the contract and state seams that the conformance scenarios in
items 6–7 need to exercise. It does not itself select the replacement contract
shape or count as an implemented architecture.

The external-framework pattern audit is recorded in
[`framework_pattern_comparison.md`](framework_pattern_comparison.md). It
separates the Lightning recipe boundary from the Lightning Trainer framework,
distinguishes Fabric/Accelerate infrastructure from the strategy contract, and
records the control and maintenance gate for any future adoption proposal.

The inventory and framework comparison are now complete enough to stop
revisiting their call paths. The remaining pre-OpenSpec work is semantic
contract design, not another architecture comparison.

## Settled Binding Semantics

The first binding questions now have settled architectural answers strong
enough to constrain the concrete exchange design.

A **logical component identity** is the stable strategy-scoped identity of one
semantically distinct training participant. It is not Python object identity,
implementation type, source/catalog identity, prepared-wrapper identity, or a
parameter scope. The identity survives loading, source replacement, device and
precision changes, distributed wrapping, and compilation. Concurrently
distinct participants require distinct logical identities even when they share
an origin or initially share state.

A logical component represents one independently addressable semantic
participant for which authoritative bound state is maintained, whether or not
it is independently executable. Componenthood therefore does not imply a
callable forward route. The standard execution-capable case remains
deliberately simple:

```text
execution-capable logical component
  -> one normal authoritative prepared execution binding

state-bearing component without independent execution
  -> authoritative state/optimization/artifact bindings
     without a fabricated execution route
```

Selected capabilities may declare additional **named execution routes**, but
only when materially different runtime preparation or callable
representations require them. Different caller intentions do not create routes
by themselves. Training and validation using the same prepared wrapper share
one route; a separately compiled sampling representation may justify another.
For every logical-component/route pair, exactly one current binding is
authoritative.

Every route belonging to one logical component must have a defined relationship
to that component's authoritative current state:

```text
shared live state
  -> the same logical component

derived state with explicit refresh/synchronization semantics
  -> may remain a route of the same logical component

independently evolving state
  -> a distinct logical component
```

EMA, teacher/student participants, and independently trained adapters therefore
receive distinct logical identities rather than becoming execution-route names.
An original or unwrapped module retained for inspection, metadata, or artifact
extraction is a typed access/view binding, not a competing forward route.
Backend-managed replicas likewise do not create new strategy-level logical
identities.

An additional route must not remain silently authoritative after its declared
freshness guarantee stops holding. It must become invalid, be explicitly
allowed as stale, or be refreshed according to declared semantics. The exact
refresh mechanism and route representation remain open for the concrete
exchange design; their authoritative state belongs to the binding authority
settled below.

## Settled Arrangement-Change Semantics

Additions, replacements, materialization, execution preparation, and adapter
attachment are distinct semantic operations even when one current code path
performs several of them together. The settled taxonomy is:

```text
declare a participant
materialize authoritative bound state
replace authoritative bound state
rebind an execution route
transition an operational relationship
amend the arrangement by adding or retiring participants
merge/fold state and record the resulting lineage
```

Materialization moves a declared participant from absent or deferred state to
bound state. Replacement supersedes an existing authoritative binding and must
make the old realization, affected facts, and invalidation consequences
explicit. Runtime preparation may rebind a route without changing the logical
arrangement. Merge/fold transforms host state using adaptation state, may
retire the adapter from the current arrangement, and records transition and
artifact provenance.

Ordinary adapters authored into a strategy exist as logical participants
before runtime materialization or attachment. Materializing their state is not
adding a participant, and establishing their effect on a host is not creating
one. Truly dynamic additions require an explicit arrangement amendment rather
than appearing as an incidental result of loading or Python assignment.

A logical adapter identity denotes one independently addressable adaptation
participant whose state and relationships are managed as one unit under the
strategy contract. It does not follow Python runtime-container boundaries or
every injected target-local module. Internal tensors, shared banks, and
target-local modules may remain qualified substructure when they share one
lifecycle and persistence unit; conversely, one runtime container may expose
multiple logical participants when they are independently addressable.

Participant identity and relationship identity are separate. One adapter may
participate in multiple independently addressable effects, and activating or
detaching an effect changes an operational relationship rather than creating
or retiring the participant. A wrapper may change the host's execution-route
composition, but being the outermost callable does not transfer semantic
execution ownership from the host component to the wrapper or adapter.

The state axes must remain separate:

```text
participant lifecycle
  declared, bound, retired

optimization status
  selected/unselected, trainable/frozen

operational relationship lifecycle
  declared, resolved, active, inactive, detached
```

Historical lineage is not current bound state. Persistence must be able to
distinguish three views without turning the live arrangement into an
ever-growing history container:

```text
current arrangement
run transition history
durable artifact provenance
```

The concrete representation, invalidation machinery, and APIs remain
downstream design questions. Their canonical current-state owner is the
binding authority settled below; those details do not reopen these question 3
semantics.

## Settled Artifact-Persistence Semantics

Artifact persistence does not mean calling `state_dict()` on whichever Python
object happens to be available. It begins with a **semantic product
declaration** that selects the participant-owned state, relationship state,
dependencies, and declared transformations that one trained artifact is meant
to represent. Python object identity, current wrapper nesting, and filesystem
layout do not determine that selection.

An **artifact state projection** is the purpose-specific semantic view of
authoritative current state selected for that product. It is indexed by logical
participant and relationship identity. Obtaining the projection may require
prepared execution bindings, original/unwrapped access views, distributed
gathering, capability-owned state, or source references; the projection itself
is not required to be a new container, a copied `state_dict`, or one live
module.

The consistency rule is:

> Every product member corresponds to one accepted current arrangement and one
> Trainer-established persistence boundary, while satisfying the declared
> freshness and consistency relationship of every contributor.

This does not require every contributor to share one literal revision counter.
A frozen autoencoder, an external base-model reference, and a freshly updated
adapter can participate in one coherent product when their relationships and
freshness guarantees are explicit. Trainer and runtime infrastructure establish
the boundary and coordinate ranks; a selected capability declares any stronger
cross-participant consistency requirement. Serialization or publication may
continue asynchronously after a stable projection has been obtained.

Persistence has four distinct stages:

```text
capability declaration
  which artifact products and representations the strategy supports

persistence request
  which declared product Trainer asks to produce at this lifecycle point

resolved artifact plan
  participant/relationship coverage, dependencies, transformations,
  expected members, representation, and consistency constraints

artifact result
  actual members/resources, formats, sizes, checksums, references,
  and any partial or failed outcome
```

The plan is authoritative about intent; the result is authoritative about what
was actually emitted. Pre-write embedded metadata can derive from the plan.
Post-write facts such as actual resources, sizes, checksums, and final member
status must derive from the result rather than being guessed before writing.

Three containment levels must remain separate:

```text
artifact product
  one semantic persistence result

artifact member
  one semantically meaningful constituent of that product

physical resource
  a file, directory, shard, blob, or remote object backing members
```

One member may span several shards, while one physical file may encode several
participants. A bundle is therefore a packaging/cardinality property, not a
claim that the product is complete or self-contained.

Artifact descriptions also keep these dimensions orthogonal:

```text
semantic coverage
  participants, state scopes, and relationships represented

dependency semantics
  standalone state, delta-over-base state, and/or external references

semantic transformations
  merge/fold, pruning, quantization-as-product, derived realization, etc.

external representation
  checkpoint, Diffusers layout, adapter weights, and other formats

physical packaging
  single resource, directory, shards, or multi-resource bundle

consistency
  accepted arrangement, boundary, and contributor freshness guarantees
```

Dependency forms should be represented as typed relationships rather than
forced into a single mutually exclusive completeness label. For example, one
product may contain adapter deltas, embed one auxiliary component, and refer to
an external base model at the same time.

Transformation ownership splits at the semantic boundary:

```text
strategy/product capability
  declares transformations that change artifact meaning, dependencies,
  coverage, lineage, or realization identity

domain serializer
  performs mechanical representation conversion such as key renaming,
  tensor-layout conversion, resolved dtype encoding, sharding, and
  embedded-metadata writing
```

A serializer may implement the mechanics of a semantic transformation, but it
must not invent that transformation from incidental runtime state.

Runtime resume snapshots remain a separate contract. Their purpose is to
restore execution state—optimizers, schedulers, scalers, progress, backend
state, and any capability continuation state—not to describe or publish the
trained semantic product. Resume snapshots and trained artifacts may share
Trainer-controlled timing, consistency infrastructure, and storage services,
but should not be collapsed into one request type distinguished only by a
`kind` field.

The settled question 4 statement is:

> Artifact persistence produces a declared artifact product from a coherent
> semantic projection of authoritative current state. The product identifies
> its participant and relationship coverage, dependencies, semantic
> transformations, external representation, and consistency requirements.
> Trainer and runtime infrastructure establish the persistence boundary and
> obtain stable current state; domain serializers render that state into
> semantic artifact members backed by one or more physical resources.
> Successful persistence returns the post-write facts describing what was
> actually emitted. Persistence never selects state through incidental Python
> object identity.

## Settled Binding-Authority Ownership

Question 5 is settled at the architectural level. The canonical current state
belongs to one dedicated, contract-governed, per-run **binding authority**
inside the complete Trainer-facing strategy boundary.

```text
Trainer
  <-> complete TrainingStrategy
         |- authored behavior and selected capabilities
         `- dedicated binding authority
              |- participant and relationship state
              |- authoritative current bindings and access views
              |- named execution routes
              |- revisions, dependencies, and freshness
              `- validated atomic transitions
```

The separation is one of responsibility, not another public orchestration
axis. Trainer still receives and interacts with the complete strategy. It does
not receive a parallel family/runtime object and does not learn the strategy's
participant anatomy. Strategy behavior may use scoped authority interfaces,
but arbitrary fields on strategy facets are not authoritative state.

The target-first pressure models are recorded in
[`strategy_system_inventory.md`](strategy_system_inventory.md), Milestone 10.
They deliberately do not treat the current `Trainer.loaded_components`,
family-shaped compatibility properties, or mode-owned mutation as the target
shape. They test a contract-governed binding authority against ordinary
fine-tuning, deferred loading, adapter attachment, distributed preparation,
persistence, replacement/invalidation, and a compound teacher/student research
strategy. Those scenarios established the required semantics but could not, by
construction, distinguish physical layouts that all honored the same
semantics. The ownership decision was therefore made using responsibility
criteria instead:

1. one-writer transition enforcement;
2. independent conformance testing without constructing a family strategy;
3. separation of mutable binding state from authored behavior;
4. an obvious one-per-run lifetime;
5. scoped capability access rather than unrestricted lookup; and
6. an incremental path from `LoadedModelComponent` without preserving the
   current Trainer-owned family projections.

A dedicated internal authority satisfies those criteria while preserving one
public Trainer-to-strategy relationship. Physically placing the authority in a
separate class does not mean Trainer receives it separately.

The assumed initial execution model is one active strategy and one logical
binding authority per Trainer run. Strategy-state transitions are accepted
through the run's coordination boundary; distributed ranks and backend
replicas do not become independent authorities. Failure, rollback, rank
consistency, multi-adapter overlap, compiled routes, EMA, resume, and mid-run
trainability changes remain important conformance pressures, but none reopens
the ownership answer. Independently evolving EMA or teacher/student state
continues to require distinct participant identities under questions 1–2.

Concrete construction, access, transition, snapshot, and exchange APIs remain
to be designed. In particular, the design must establish who creates the
authority, how strategy filing seeds it, how producers submit typed transition
results, how atomic rejection/rollback works, and which scoped projections are
available to each consumer.

## Subsequent Design Work: Concrete Exchanges And Optimization Ownership

The minimum core should emerge from concrete exchanges rather than from a list
of attractive method names. With binding ownership settled, the next design
milestone should define four tables.

### Binding exchange

```text
inputs
result
logical state established
allowed side effects
failure conditions
```

This must distinguish declaring required components, materializing them,
binding their relationships, and establishing authoritative logical identity.
It must not assume those meanings collapse into one `bind()` call.
`LoadedModelComponent` is the current evolutionary starting point, but the
result must preserve logical/original and prepared execution bindings
separately. The exchange must also define when strategy-scoped participant
identities are assigned and how duplicate or conflicting declarations are
rejected.

### Runtime-preparation exchange

```text
participants exposed to infrastructure
stable binding identities
joint-preparation constraints
prepared execution bindings returned
rebinding guarantees
```

This exchange applies the wrapper/replacement lessons from Fabric and
Accelerate within the repository's own design. Trainer-owned infrastructure
may require concrete modules and optimizers without learning family-specific
anatomy. The current mode-owned `prepare_with_accelerator(trainer)` mutation
should be treated as source evidence for the exchange, not retained as its
contract.

### Optimization exchange

```text
training subjects and parameter groups
trainable logical identities
clipping participants
synchronization participants
optimizer/backend constraints
```

This is where the strategy's declared training intent meets Trainer-owned
optimizer and distributed policy. The current mode behavior is evidence for
the necessary information, not a participant to preserve. The existing
`OptimizationPlan` already separates logical and execution parameter groups
and should be pressure-tested as the starting payload.

### Step exchange

```text
batch request
execution coordinates
differentiable optimization result
metrics and objective observations
allowed state updates
forbidden infrastructure actions
```

The existing `BatchLossOutput` is evidence for this result boundary, not a
permanent universal type. Its `per_sample_loss`, rather than its `loss` field,
feeds the actual Trainer-owned loss-modifier/backward path. The exchange must
name that distinction honestly, preserve current diffusion needs without
exposing family component topology, and avoid assuming every future contract
uses diffusion timesteps.

Binding and runtime preparation should be discussed together first because
logical identity and prepared execution identity must remain coherent across
their boundary. Optimization ownership is then the next semantic decision, not
an implementation detail silently answered by the current Trainer. The
standard profile is expected to keep optimizer realization, backward,
stepping, and distributed mechanics Trainer-owned; an explicit contract
extension must describe experiments that deliberately take over any of those
mechanics.

Once the four exchanges and optimization ownership are defined, an OpenSpec
can lock down the migration for SD, SDXL, SD3, and the shared Trainer. Work may
be divided into reviewable slices, but each introduced contract should carry
the intended semantics rather than a knowingly weaker temporary design. This
does not require implementing the end-game arbitrary component catalog at the
same time.

## Current Direction In One View

```text
                       TRAINING CONTRACT SYSTEM
              vocabulary + rules + lifecycle + results
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
     CORE CONTRACT        PIPELINE CAPABILITIES    FEATURE CONTRACTS
  minimum Trainer need    sampling, validation,   representation,
                          caching, persistence     conditioning, etc.
          │                       │                       │
          │              pipeline coordinates      strategy consumes
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  ▼
                   AUTHORED TRAINING STRATEGY
             explicit training intent/components/features/capabilities
             internal per-run binding authority for accepted current state
                                  │
          authored → validated → bound → prepared
                                  │
                                  ▼
                               TRAINER
                 executes core and requested capabilities
                 owns time, optimization, infrastructure,
                         triggers, and observation
```

The feature grab box is not the entire design. It is the organizational layer
that may let the contract remain expressive without turning either
`contracts.py` or every family folder into another architectural thicket.
