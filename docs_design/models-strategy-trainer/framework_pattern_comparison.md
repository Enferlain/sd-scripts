# Training Framework Pattern Comparison

## Status And Purpose

This is a bounded source-level comparison for the model–strategy–trainer
design. It is not an adoption proposal, dependency recommendation,
implementation plan, or claim that the repository should reproduce another
framework's API.

> **Later topology correction (2026-08-26):** the source audit and boundary
> lessons in this document remain useful evidence, but its recurring
> `Trainer -> fulfilled strategy` framing is superseded. The current direction
> accepts an authored strategy into an executable run arrangement. The Trainer
> executes that arrangement; the authored strategy need not remain an ordinary
> runtime collaborator. An authorized imperative escape region is a distinct
> execution role, not proof that the strategy object must stay active.

The first pass over this topic relied too heavily on public documentation.
Documentation can establish an intended public API, but it cannot show the
real ownership graph, wrapper behavior, hidden collaboration surfaces, or cost
of adapting only part of a framework. The conclusions in this revision are
therefore based on selected implementation paths as well as documentation.

The comparison was started because the then-emerging relationship

```text
Trainer → fulfilled TrainingStrategy
```

resembles the established

```text
Lightning Trainer → LightningModule
```

relationship closely enough that continuing without comparison risks
reimplementing a mature framework by accident. The hypothesis being tested is
that Lightning, Fabric, and Accelerate may be useful references for specific
boundary mechanics while this repository keeps its own design authority. That
hypothesis is not treated as established until after the source comparison
below.

The durable codebase inventory remains
[`strategy_system_inventory.md`](strategy_system_inventory.md). The broader
design argument and accepted direction remain
[`strategy_system_direction.md`](strategy_system_direction.md).

## Scope And Evidence

The comparison asks:

- who owns the training loop;
- who owns model/component state;
- how authored training behavior meets the executor;
- how optimizers and distributed wrappers cross that boundary;
- how much lifecycle surface the framework exposes;
- how unusual training arrangements escape the default path;
- which patterns are useful here without importing the surrounding framework.

It does not include:

- a Lightning migration prototype;
- a Fabric-versus-Accelerate performance comparison;
- a licensing or redistribution review;
- a complete framework feature matrix;
- an assumption that similar architecture requires shared implementation.

External behavior was checked against the official documentation for
[LightningModule](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html),
[Lightning Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html),
[Lightning manual optimization](https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html),
[Lightning Fabric](https://lightning.ai/docs/fabric/stable/),
[Accelerate gradient accumulation](https://huggingface.co/docs/accelerate/main/usage_guides/gradient_accumulation),
and
[Accelerate checkpointing](https://huggingface.co/docs/accelerate/main/usage_guides/checkpoint).

Implementation evidence was checked against:

- Lightning `2.6.5`, commit
  [`be98784a`](https://github.com/Lightning-AI/pytorch-lightning/tree/be98784a1a03581b7051a355ae1084fd352d7cea);
- the repository's installed Accelerate `1.11.0`, checked locally and against
  the
  [`v1.11.0` source](https://github.com/huggingface/accelerate/tree/v1.11.0);
- this repository's `model-strategy-trainer` branch, with codebase-memory graph
  evidence refreshed and the material source files checked directly.

The audit follows representative control and state paths. It is not a
line-by-line audit of either external project.

## What The Public Analogy Conceals

Under the superseded topology that motivated this audit, the public API analogy
looked simple:

```text
Lightning Trainer → LightningModule
repository Trainer → fulfilled TrainingStrategy
```

The source does not implement the Lightning side as one narrow interface call.
It is a mutually connected runtime composed of:

```text
Trainer
  ├── loop objects
  ├── checkpoint/data/logger connectors
  ├── callbacks
  ├── distributed Strategy
  └── original LightningModule
          └── direct reference back to Trainer

distributed Strategy
  ├── original LightningModule
  ├── potentially wrapped execution model
  ├── optimizers and scheduler configurations
  └── precision plugin
```

Lightning's `Strategy` is also not equivalent to this repository's
`TrainingStrategy`. It represents distributed/device execution behavior. The
closest analogue to the then-proposed fulfilled training strategy is the
`LightningModule`, but even that object is simultaneously a PyTorch module,
training recipe, lifecycle-hook provider, logger client, optimizer client, and
child of a Trainer-owned runtime.

That difference matters. Adopting the surface name `training_step()` would not
adopt a small stable contract; adopting its behavior would pull on the rest of
that object graph.

## Current Repository Baseline

The current launcher explicitly constructs one family strategy and one training
mode:

```text
build_training_strategy(cfg)
build_training_mode(cfg)
Trainer(cfg, strategy, mode)
```

The trainer constructs the objective separately. The strategy factory remains
ordinary repository-authored Python: it selects a known family class and
instantiates it with the run configuration
(`library/strategies/factory.py:22-34`).

`TrainingStrategy` is already one aggregate type, but it is formed by inheriting
eleven strategy ABCs (`library/strategies/base/contracts.py:546-567`). It mixes
trainer-facing lifecycle operations with family-internal collaboration
surfaces.

The trainer is the active state and orchestration hub
(`library/training/runners/trainer.py:130-254`). It retains:

- strategy, mode, and objective;
- Accelerate infrastructure;
- loaded components and SD-shaped role projections;
- dtypes, manifests, caches, trainable state, optimizer state, and scheduler;
- lifecycle, validation, logging, metadata, and resource-observation state.

The step boundary is partly clean and partly inverted. The family strategy
produces a `BatchLossOutput`, while the trainer owns loss modifiers, backward,
gradient synchronization, optimizer/scheduler advancement, and step side
effects. However, the trainer calls `process_batch()` by passing the strategy
the VAE, text encoders, denoiser, primary trainable, objective runtime, dtypes,
Accelerator, configuration, flags, and coordinates that the trainer itself
holds (`library/training/phases/training_loop.py:461-618`).

This is not merely a long signature. It means the aggregate strategy is not
the authoritative bound model integration after construction.

Accelerate is already the repository's infrastructure substrate
(`pyproject.toml:10`). The active loop uses Accelerate for preparation,
accumulation/synchronization, backward, clipping, process coordination, and
related runtime behavior. Lightning is not an active project dependency; the
repository only contains a narrow dummy compatibility module under
`library/vendor/pytorch_lightning/`.

## The Three Different Layers

Lightning, Fabric, and Accelerate do not answer the same architectural
question.

```text
authored training behavior
        │
        │ LightningModule is prior art here
        ▼
training-loop orchestration
        │
        │ Lightning Trainer is a full implementation here
        ▼
distributed/device/precision infrastructure
        │
        │ Fabric and Accelerate operate mainly here
        ▼
PyTorch runtime
```

The repository currently owns the first two layers and uses Accelerate for
substantial parts of the third.

## LightningModule And Lightning Trainer

Lightning organizes model state, forward/loss behavior, validation behavior,
optimizer construction, and lifecycle hooks on `LightningModule`. Its Trainer
owns the temporal loop and normally owns device placement, distributed
execution, backward, and optimizer timing.

That makes Lightning's public shape highly relevant to this design:

```text
Lightning Trainer
  calls stable LightningModule operations

LightningModule
  owns arbitrary internal torch modules
  implements authored batch behavior
  exposes optimizer and lifecycle integration
```

The useful lesson is not the names `training_step()` or
`configure_optimizers()`. It is that the executor consumes an authored object
through generic behavior instead of unpacking that object's encoder, decoder,
predictor, or other component anatomy.

Lightning also provides manual optimization. A module can opt out of automatic
optimizer handling and perform backward/step behavior itself. That demonstrates
an escape route for unusual algorithms, but it also moves trainer behavior into
the authored module. It does not by itself answer where this repository should
place the boundary.

### Binding and state ownership in the implementation

`Trainer._run()` does not merely retain a module and call it. It connects the
module to the distributed Strategy, invokes callbacks and model hooks, sets up
the execution environment, restores state on a strategy-dependent side of
setup, asks the Strategy to convert/wrap the module and initialize optimizers,
then enters the selected loop. See
[`trainer.py:997-1091`](https://github.com/Lightning-AI/pytorch-lightning/blob/be98784a1a03581b7051a355ae1084fd352d7cea/src/lightning/pytorch/trainer/trainer.py#L997-L1091).

The relevant ownership exchange is approximately:

```python
# Trainer._run()
trainer.strategy.connect(lightning_module)
trainer.strategy.setup_environment()
call_setup_and_configure_model_hooks()
trainer.strategy.setup(trainer)
restore_training_state()
trainer._run_stage()
```

`Strategy.connect()` stores both `_lightning_module` and `model`. During setup,
`model` may become a precision-converted or distributed wrapper while
`_lightning_module` remains the original authored object. Optimizers and
scheduler configurations also live on the Strategy. See
[`strategies/strategy.py:111-180`](https://github.com/Lightning-AI/pytorch-lightning/blob/be98784a1a03581b7051a355ae1084fd352d7cea/src/lightning/pytorch/strategies/strategy.py#L111-L180).

The connection is bidirectional. `LightningModule` stores `_trainer`, and
properties such as `optimizers()`, `lr_schedulers()`, `current_epoch`, logging,
manual backward, and device information reach back into Trainer or its
Strategy. See
[`core/module.py:133-320`](https://github.com/Lightning-AI/pytorch-lightning/blob/be98784a1a03581b7051a355ae1084fd352d7cea/src/lightning/pytorch/core/module.py#L133-L320).

This is the first major difference from the intended repository direction:

```text
desired boundary
  Trainer knows the strategy contract
  strategy does not require unrestricted access back into Trainer

Lightning implementation
  Trainer owns and calls LightningModule
  LightningModule can reach broadly back into Trainer state and services
```

The Lightning arrangement is coherent, but it is not evidence that passing the
whole Trainer into a strategy is harmless. It shows the amount of runtime
machinery needed to make that bidirectional design consistent.

### The actual training-step call path

The training epoch loop performs batch transfer and lifecycle hooks, then
branches on `lightning_module.automatic_optimization`. See
[`training_epoch_loop.py:309-365`](https://github.com/Lightning-AI/pytorch-lightning/blob/be98784a1a03581b7051a355ae1084fd352d7cea/src/lightning/pytorch/loops/training_epoch_loop.py#L309-L365).

Automatic optimization follows this path:

```text
TrainingEpochLoop
  → AutomaticOptimization.run()
      → build optimizer closure
          → Strategy.training_step()
              → wrapped execution model or original LightningModule
          → normalize/validate returned loss
          → Strategy.backward()
      → LightningModule.optimizer_step()
          → wrapped optimizer / precision plugin
```

The closure implementation and strategy-dispatched `training_step()` are in
[`automatic.py:153-316`](https://github.com/Lightning-AI/pytorch-lightning/blob/be98784a1a03581b7051a355ae1084fd352d7cea/src/lightning/pytorch/loops/optimization/automatic.py#L153-L316)
and
[`strategies/strategy.py:380-394`](https://github.com/Lightning-AI/pytorch-lightning/blob/be98784a1a03581b7051a355ae1084fd352d7cea/src/lightning/pytorch/strategies/strategy.py#L380-L394).

This does establish one valuable precedent: the loop can own optimization
timing while the authored object owns arbitrary model-specific forward and loss
work. But the boundary is intentionally small and weakly typed: the step
returns a Tensor, a mapping containing `loss`, or `None`, while surrounding
semantics are supplied through hooks and Trainer access.

That is not the contract quality currently being sought here. This repository
wants conformance rules rich enough to say what was fulfilled, validate
composed features, preserve research variants, and keep model topology out of
the loop. Lightning's step result demonstrates an executor seam, not that
contract.

### Manual optimization is a separate ownership mode

When `automatic_optimization` is false, Lightning's manual loop mostly calls
`training_step()` and records its returned values. The module retrieves wrapped
optimizers through its Trainer reference and calls `manual_backward()`, which
routes back through the Trainer Strategy. See
[`manual.py:70-135`](https://github.com/Lightning-AI/pytorch-lightning/blob/be98784a1a03581b7051a355ae1084fd352d7cea/src/lightning/pytorch/loops/optimization/manual.py#L70-L135)
and
[`core/module.py:1092-1125`](https://github.com/Lightning-AI/pytorch-lightning/blob/be98784a1a03581b7051a355ae1084fd352d7cea/src/lightning/pytorch/core/module.py#L1092-L1125).

```text
automatic optimization
  Trainer loop owns backward/step cadence

manual optimization
  LightningModule owns backward/step decisions
  but uses Trainer/Strategy services to perform them correctly
```

Therefore manual optimization is not a free-form extension of the same narrow
contract. It is an explicit transfer of optimization ownership. If this
repository needs such an escape route, it should represent that distinction in
the contract rather than hide it behind a boolean whose consequences are known
only by convention.

### Checkpointing is a composed run snapshot

`Trainer.save_checkpoint()` asks its checkpoint connector to assemble a
dictionary, then asks the distributed Strategy to store it. The dictionary
combines:

- the original LightningModule state dict;
- loop progress;
- callback state;
- optimizers and schedulers;
- precision-plugin state;
- hyperparameters and datamodule state;
- module/callback hook additions.

See
[`trainer.py:1439-1467`](https://github.com/Lightning-AI/pytorch-lightning/blob/be98784a1a03581b7051a355ae1084fd352d7cea/src/lightning/pytorch/trainer/trainer.py#L1439-L1467)
and
[`checkpoint_connector.py:409-510`](https://github.com/Lightning-AI/pytorch-lightning/blob/be98784a1a03581b7051a355ae1084fd352d7cea/src/lightning/pytorch/trainer/connectors/checkpoint_connector.py#L409-L510).

This is a useful implementation of resumable training state, but it does not
replace this repository's family artifact persistence. A single
`LightningModule.state_dict()` can contain arbitrarily many child modules, yet
the framework does not know whether those children form SDXL, a teacher/student
pair, an attached adapter, or an ecosystem-specific checkpoint layout.

### Patterns worth borrowing

- The training executor is the conceptual consumer and therefore defines the
  interface it needs.
- One authored aggregate may contain several neural modules without exposing
  their identities to the loop.
- The loop invokes stable semantic operations and receives stable semantic
  results.
- Standard automation can coexist with an explicit escape route.
- Lifecycle ordering belongs to the executor even when authored behavior is
  invoked at lifecycle boundaries.
- The original logical module and prepared execution wrapper are different
  identities and both may be required.
- Automatic and manual optimization are different ownership contracts, not
  merely different implementations of one method.

### Patterns not accepted by default

- Making every authored strategy or accepted arrangement an `nn.Module` merely
  to match `LightningModule`.
- Copying a large hook catalog before the repository has identified the
  trainer-facing semantics it actually needs.
- Moving generic backward, optimizer, logging, checkpoint, or device policy
  into family strategies just because manual optimization permits it.
- Treating the Lightning Trainer lifecycle as the definition of what this
  repository may train.
- Adopting framework-owned logging, checkpoint, callback, or state behavior
  when the repository already has deliberate first-party systems.

## Lightning Fabric

Fabric deliberately leaves the training loop in application code. The caller
constructs models, optimizers, and dataloaders, asks Fabric to prepare them,
then explicitly performs the forward pass, loss computation, backward, and
optimizer step.

Fabric therefore does not supply the missing model–strategy–trainer contract.
It supplies infrastructure services beneath a user-owned contract and loop.
Its most relevant pattern is:

```text
application owns orchestration
infrastructure object provides explicit distributed/runtime operations
```

That is much closer to the repository's current relationship with Accelerate
than to the proposed fulfilled-strategy relationship.

### What `Fabric.setup()` actually returns

Fabric does not prepare a module in place and leave its identity unchanged. Its
setup path:

1. retains the original module;
2. applies precision conversion and device movement;
3. lets a Fabric Strategy wrap the module and optimizer;
4. wraps the resulting execution module again in `_FabricModule`;
5. wraps optimizers in `_FabricOptimizer`;
6. returns those replacement objects to the caller.

See
[`fabric.py:229-376`](https://github.com/Lightning-AI/pytorch-lightning/blob/be98784a1a03581b7051a355ae1084fd352d7cea/src/lightning/fabric/fabric.py#L229-L376).

```python
original_module = module
module = precision.convert_module(module)
module = strategy.setup_module(module)
module = _FabricModule(
    forward_module=module,
    original_module=original_module,
)
return module
```

The caller must rebind the returned object. This is the same class of concern
that the current repository handles by assigning prepared objects back to
`trainer.denoiser`, `trainer.text_encoders`, or `trainer.adapter`.

### The wrapper is behavior, not just storage

`_FabricModule` retains both the original module and prepared forward module.
Its `forward()` converts inputs, enters the precision context, calls the
prepared module, converts outputs, and installs backward-related behavior.
`state_dict()` delegates to the original module.

Custom methods are harder. Calling a method directly on an original module can
bypass DDP/FSDP forwarding. Fabric therefore maintains a set of forward-like
method names and can temporarily redirect a custom method through the wrapped
module's `forward()`. Unmarked methods are monitored and may raise if they call
submodules outside the distributed wrapper. See
[`wrappers.py:101-267`](https://github.com/Lightning-AI/pytorch-lightning/blob/be98784a1a03581b7051a355ae1084fd352d7cea/src/lightning/fabric/wrappers.py#L101-L267).

This is directly relevant to an accepted arrangement that may hide several
model components:

```text
logical component identity
  is not necessarily the object that must execute

prepared execution identity
  may be a wrapper whose call path must not be bypassed
```

The repository cannot solve this by placing loaded modules on a strategy and
forgetting about them. It needs an explicit preparation/rebinding rule, or a
generic prepared-binding object, so strategy execution always uses the correct
handle while metadata and family persistence can still reach the logical
component.

### Fabric's Strategy is an infrastructure plugin

Fabric Strategy owns setup, distributed collectives, backward routing,
optimizer stepping, and generic checkpoint conversion. Its base
`save_checkpoint()` receives a dictionary of modules, optimizers, and arbitrary
state, converts stateful objects, and delegates to checkpoint IO. See
[`fabric/strategies/strategy.py:150-205`](https://github.com/Lightning-AI/pytorch-lightning/blob/be98784a1a03581b7051a355ae1084fd352d7cea/src/lightning/fabric/strategies/strategy.py#L150-L205)
and
[`fabric/strategies/strategy.py:260-344`](https://github.com/Lightning-AI/pytorch-lightning/blob/be98784a1a03581b7051a355ae1084fd352d7cea/src/lightning/fabric/strategies/strategy.py#L260-L344).

It is therefore an implementation of the infrastructure layer, despite sharing
the word “strategy.” It does not express model-family behavior, representation
features, conditioning, objectives, caching, or family artifact formats.

### Patterns worth borrowing

- Infrastructure capabilities can remain explicit without owning the training
  architecture.
- A custom trainer can use a focused runtime service rather than inherit a
  complete trainer framework.
- Opt-in infrastructure makes unusual loop behavior easier to reason about than
  a large default hook lifecycle.
- Preparation is a transformation with explicit returned identities, not a
  side effect that callers may ignore.
- A model integration may need both logical/original and prepared/execution
  handles.

### Patterns not currently additive

- Replacing Accelerate with Fabric solely because Fabric demonstrates the same
  user-owned-loop pattern.
- Treating an infrastructure toolkit as the source of the strategy contract.
- Maintaining two overlapping distributed/precision abstraction layers.

Fabric should be reconsidered only if a separate capability or maintenance
comparison shows that it materially improves the runtime substrate.

## Hugging Face Accelerate

Accelerate also leaves the loop in repository code. Its documented pattern is
to prepare models, optimizers, schedulers, and dataloaders, then use explicit
operations such as `accumulate()` and `backward()` inside the caller-owned
loop.

That pattern already appears in the active trainer. It has one important
architectural consequence for the accepted run boundary: preparation can
return wrapped or replacement runtime objects. The run binding authority must
be able to accept those prepared routes without leaving Trainer infrastructure,
runtime operations, metadata, diagnostics, and saving paths with inconsistent
references.

Accelerate checkpoint state is also different from family artifact
persistence:

- runtime checkpointing owns resumable model/optimizer/RNG/scaler state;
- family persistence owns repository and ecosystem model artifacts;
- trainer coordinates when each is requested.

Those meanings should not be collapsed into one generic `save()` hook merely
because another framework exposes one lifecycle.

### Preparation and internal registration

`Accelerator.prepare()` accepts modules, optimizers, schedulers, and
dataloaders, prepares them in two passes when necessary, marks the returned
objects, and returns replacements in the same positional order. Backend
branches impose different joint-preparation constraints: for example,
DeepSpeed and FSDP2 may require model and optimizer preparation together. See
[`accelerator.py:1413-1590`](https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L1413-L1590).

`prepare_model()` also registers the model internally, may replace its forward
method for mixed precision, and may return DDP, FSDP, compiled, or backend
specific wrappers. See
[`accelerator.py:1707-1935`](https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L1707-L1935).

Accelerate is therefore not stateless glue:

```text
caller
  owns semantic meaning and loop

Accelerator
  tracks prepared models/optimizers/schedulers/dataloaders
  owns distributed and precision state
  may replace execution identities
```

The current repository spreads the necessary rebinding across training modes:

- fine-tuning assigns prepared denoiser and text encoders back onto Trainer;
- adapter training assigns prepared denoiser, text encoders, and adapter back
  onto Trainer;
- both modes separately choose a gradient-synchronization handle and primary
  trainable;
- DeepSpeed uses a different aggregate preparation path.

See
`library/training/modes/finetune_mode.py:234-273`,
`library/training/modes/adapter_mode.py:243-290`, and
`library/training/phases/optimizer.py:62-166`.

This is evidence for a real repository contract concern: infrastructure
preparation needs a generic way to obtain preparation participants and return
their execution identities. It is not evidence that the Trainer must know
those participants as “denoiser,” “text encoder,” or “adapter.”

### Accumulation and backward

`Accelerator.accumulate(*models)` changes gradient synchronization based on its
gradient state and enters `no_sync()` for the supplied prepared models when
appropriate. `Accelerator.backward(loss)` scales the loss for accumulation,
then selects DeepSpeed, Megatron, scaler, LOMO, or ordinary backward behavior.
See
[`accelerator.py:1254-1300`](https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L1254-L1300)
and
[`accelerator.py:2708-2745`](https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L2708-L2745).

That confirms the current Trainer should continue to own the temporal decision
to accumulate and backpropagate. The accepted arrangement must expose what the
infrastructure needs for correct synchronization, while maintained or custom
operations should not need the entire `Accelerator` merely to express
model-family behavior.

### Runtime state and family artifacts

`Accelerator.save_state()` walks its registered models, optimizers, schedulers,
dataloaders, scaler, RNG state, and custom objects. Backend-specific paths
handle FSDP, DeepSpeed, and Megatron. Pre-hooks allow applications to replace
model saving or add files. See
[`accelerator.py:3435-3631`](https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L3435-L3631).

The repository uses those hooks differently by training mode. That is suitable
for resumable runtime state, but the hooks are not a family artifact contract:
they receive registered modules and generic state dictionaries, not repository
model realizations, component lineage, compatibility projections, or
ecosystem-specific layouts.

### Patterns worth preserving

- The repository retains explicit control over loop and phase ordering.
- Distributed/runtime behavior is requested through a focused infrastructure
  API.
- Preparation and backward remain generic infrastructure operations rather
  than family behavior.
- Runtime checkpoint state and model artifact serialization can remain
  distinct.
- Accelerator's backend constraints are inputs to preparation validation, not
  model-family definitions.

### Pressure exposed by current use

- Wrapped/replaced modules require an authoritative rebinding model.
- Passing the `Accelerator` through most family operations expands
  infrastructure awareness beyond the places that need it.
- Passing wrapped component projections back into the strategy preserves the
  current trainer/strategy state inversion.

These are contract and state-ownership problems above Accelerate, not evidence
that Accelerate itself should become the contract.

## Findings That Require Source Evidence

The source comparison supports narrower and more useful conclusions than the
public API comparison did.

### 1. “One object” does not eliminate runtime collaboration

Lightning presents one `LightningModule` to the user, but the implementation
still requires Trainer, loop, distributed Strategy, precision plugin,
connectors, callbacks, original module, prepared module, and optimizer wrappers
to collaborate. Its success comes from making those relationships consistent,
not from literally reducing the runtime to two objects.

The repository goal should therefore be interpreted as:

```text
contract establishment produces one accepted run arrangement
Trainer executes that arrangement through one contract-governed boundary
Trainer does not reconstruct family topology
internal runtime collaborators may still exist behind typed boundaries
```

It should not be interpreted as “all state and behavior must reside in one
Python instance.”

### 2. Prepared execution state must be modeled explicitly

Lightning, Fabric, and Accelerate all distinguish an authored/original module
from a prepared execution object in at least some backends. This is not an
optional design flourish. DDP, FSDP, DeepSpeed, precision conversion, and
compilation can change which object must receive the forward call.

A future accepted run arrangement therefore needs some equivalent of:

```text
stable logical component identity
  source, metadata, family persistence, relationships

current execution binding
  prepared module/wrapper used for calls in this runtime
```

The names and container are undecided. The semantic distinction is no longer
optional.

### 3. The Trainer needs a generic preparation exchange

The Trainer owns infrastructure timing, but infrastructure must see concrete
modules and optimizers. Complete component opacity is therefore not practical.
The useful boundary is generic exposure rather than family-specific exposure:

```text
accepted arrangement → preparation requirements with stable identities
Trainer/runtime service → prepares the concrete participants
run binding authority ← accepts returned routes under the same identities
```

This is a plan/result exchange, not the Trainer learning that participant 1 is
an SDXL VAE and participant 2 is CLIP-G. It also provides a place to validate
backend restrictions such as joint model/optimizer preparation.

### 4. A step method alone is not the contract

Lightning's `training_step()` works because a large surrounding framework
defines batch transfer, hook order, logging context, optimization ownership,
wrapper redirection, checkpoint semantics, and Trainer access.

Replacing the current `process_batch()` with a method named `training_step()`
would leave the architectural problem intact. The repository must define:

- what runtime-varying inputs accepted operations or regions may rely on;
- what outputs, observations, and declared state effects the Trainer consumes;
- whether loss construction is complete or still subject to Trainer policy;
- which authority and side effects each operation or region is allowed; and
- which execution/ownership profile applies.

### 5. Research escape routes require explicit authority

Lightning manual optimization is useful evidence because it is not presented
as invisible magic. It transfers backward and optimizer-step decisions from
the automatic loop to the authored module while retaining infrastructure
services.

That is Lightning's ownership contract, not the required topology here. This
repository can support equally strong experimentation by accepting imperative
regions with explicit authority under a named contract extension or profile.
The imperative runtime role is distinct from the authored strategy and
receives only the responsibility the contract grants. “Custom strategy” cannot
mean that the standard contract silently stops constraining Trainer and
runtime responsibilities.

### 6. Runtime snapshots and model artifacts remain separate

Both Lightning and Accelerate can serialize complete resumable training state.
Neither supplies the repository's model-family artifact semantics. Replacing
family checkpointing with their generic state mechanism would lose the
distinction between:

```text
resume this exact execution
publish/save this logical model realization
```

The two operations may coordinate and share state-dict extraction, but they
should not be collapsed.

## Direct Pressure Against The Current Repository

| Concern | Current implementation | External source lesson | Contract implication |
| --- | --- | --- | --- |
| loaded state | Trainer owns SD-shaped projections while strategy retains some family facts | original and prepared execution identities must both be tracked | define authoritative bound state and generic execution bindings |
| distributed preparation | training modes mutate specific Trainer fields after `Accelerator.prepare()` | replacement identities and backend-specific joint setup are normal | the accepted arrangement exposes preparation requirements; infrastructure returns results and the run authority accepts rebound routes |
| batch execution | Trainer passes model projections, objective, dtypes, Accelerator, flags, and coordinates into `process_batch()` | authored behavior needs an accepted execution form with a defined surrounding runtime contract | derive the accepted execution meanings and effects rather than rename the method or preserve a live strategy callback |
| optimization | Trainer, mode, objective, loss modifier, strategy, and Accelerator divide responsibility | automatic and manual paths are explicit ownership modes | define the standard ownership profile and explicit extensions |
| wrapper-safe calls | strategies call prepared components supplied by Trainer | bypassing a prepared wrapper can be incorrect | execution bindings must be used for forward-like behavior |
| persistence | family saving and Accelerate state hooks are separate but coordinated through Trainer | runtime and logical artifact state have different consumers | keep separate contracts with shared extraction where justified |
| observation | metadata reads Trainer component projections | logical identity should survive runtime wrapper replacement | observation should target bound logical state, not incidental Trainer fields |

This table is the useful output of the comparison. It changes the next design
step from “invent cleaner strategy methods” to “derive the concrete accepted
execution and infrastructure exchanges needed by the run arrangement and
Trainer.”

## Comparison Matrix

| Question | Lightning Trainer + LightningModule | Fabric | Accelerate | Direction for this repository |
| --- | --- | --- | --- | --- |
| Who owns the loop? | Lightning Trainer | application | application | repository Trainer |
| Who authors training behavior? | LightningModule | application | application | strategy author, using the contract vocabulary and available capabilities |
| What reaches runtime? | LightningModule | application objects | application objects | accepted run arrangement; exact structured representation remains open |
| Who owns model internals? | usually LightningModule | application | application | accepted behavior and binding authority hide family anatomy from Trainer |
| Who owns backward/step timing? | Trainer by default; module in manual optimization | application | application | repository Trainer unless an explicit contract extension changes it |
| Who owns distributed preparation? | Lightning Trainer/strategies | Fabric service | Accelerator service | repository Trainer using focused infrastructure |
| Does it define a model-family contract? | no; it defines a framework module contract | no | no | repository must define it |
| Does it solve component features? | application concern | no | no | repository model/strategy system |
| Does it solve cache semantics? | application concern | no | no | repository strategy/data boundary |
| Does it solve family artifact formats? | application concern | no family artifact model | runtime state support | repository model/strategy persistence |
| Primary value here | recipe/executor prior art | infrastructure prior art | active infrastructure substrate | custom, bounded architecture |

## Adoption And Maintenance Gate

Similarity is not sufficient reason to adopt an implementation. An external
system is useful only if it can support the repository's design without making
the design conform to its hidden or non-replaceable assumptions.

The relevant routes have different ownership costs:

| Route | Design control | Maintenance owner |
| --- | --- | --- |
| external dependency through public APIs | constrained by supported seams | mainly upstream |
| wrapping or subclassing | moderate; vulnerable to internal assumptions | shared with upstream |
| vendoring or maintaining a fork | high | this repository, plus upstream merge cost |
| adapting a small separable subsystem | high and bounded | this repository |
| using patterns as prior art | complete | this repository's deliberately chosen surface |

Before any adoption or vendoring proposal, the candidate must demonstrate:

1. The repository can define its own trainer–strategy contract.
2. Phase ordering, state ownership, and research escape routes remain
   replaceable.
3. Multi-component families, teacher/student arrangements, attached trainables,
   representation changes, and family persistence do not require misleading
   framework abstractions.
4. Existing metadata, observability, interruption, caching, and artifact
   systems remain first-class.
5. The retained implementation is smaller and easier to maintain than the
   custom behavior it replaces.
6. The repository can realistically maintain vendored code and selectively
   incorporate upstream fixes.

The current comparison does not establish those conditions for wholesale
Lightning or Fabric adoption. It therefore treats them as references, while
leaving open the possibility that a small separable mechanism could later pass
the gate.

## Patterns To Carry Into Contract Design

### Borrow

1. **Consumer-owned contract**
   The intended Trainer defines the behavior and result meanings it requires.
   Authored strategies must be accepted under that contract before execution.
2. **Authored aggregate**
   The strategy author explicitly composes model integrations and reusable
   features. The Trainer does not infer or assemble them.
3. **Encapsulated anatomy**
   The Trainer operates on training meanings, not VAE/text-encoder/denoiser
   identities.
4. **Semantic requests and results**
   Typed boundary values may bundle loss, observations, trainable groups,
   preparation targets, or artifact results without becoming raw component
   bags.
5. **Explicit infrastructure**
   Distributed preparation and backward remain visible trainer-owned actions
   backed by a focused runtime service.
6. **Declared escape route**
   An accepted arrangement may compose maintained and custom structured
   operations with explicitly authorized imperative regions. The authoring API
   and exact runtime representation remain open.

### Avoid

1. A general framework hook for every lifecycle event.
2. An `nn.Module` inheritance requirement for the completed strategy.
3. Automatic strategy assembly or dependency resolution.
4. A trainer that unpacks family component topology.
5. A strategy that silently takes over generic optimization/infrastructure
   policy.
6. Duplicating Fabric/Accelerate functionality inside the strategy system.
7. Vendoring a full framework merely to acquire an interface pattern.

### Not decided by prior art

- how mode and objective concepts contribute to the accepted arrangement;
- the exact physical layout of authoritative loaded state inside the accepted
  run scope;
- the exact form of the generic preparation and rebinding exchange;
- the exact training-step result and loss-policy boundary;
- how validation, sampling, and family persistence divide behavior from
  orchestration;
- where the trainer-facing contract module should live.

## Candidate Boundary Questions From The Current Inventory

Prior art narrows the questions but does not answer them:

| Trainer responsibility | Current exchange | Contract question |
| --- | --- | --- |
| construct integration | factory returns family aggregate; mode and objective are built separately | what authored inputs establish one accepted run arrangement? |
| load/bind model | strategy returns components; Trainer stores and projects them | how does the accepted run authority own bound state while exposing generic preparation needs? |
| distributed preparation | Trainer/phase wraps or replaces component references | how are generic preparation targets declared and rebound without exposing family anatomy? |
| select trainables | mode receives the whole Trainer and returns parameters | how does authored intent become accepted semantic trainable groups while preserving experimentation? |
| execute batch | Trainer passes fourteen concerns into `process_batch()` | what accepted execution form lets the Trainer run authored semantics without a live strategy callback? |
| apply objective/loss policy | split across objective runtime, family strategy, loss modifier, and Trainer | which semantics belong in the accepted arrangement and which mechanics remain Trainer-owned? |
| validate/sample | family behavior is mixed with traversal, cadence, device, and reporting | what family behavior can be called without copying Lightning's lifecycle-hook catalog? |
| persist | family serialization may receive the whole Trainer | what typed artifact request/result separates coordinates and orchestration from serialization? |
| observe model | metadata reads Trainer-owned component projections | how does observation query authoritative bound state without owning or mutating it? |

The next design milestone should answer these from the repository's needs, using
the external systems only to challenge avoidable complexity.

## Current Conclusion

The comparison does not change the overall goal:

```text
repository Trainer
  dictates a bounded training-strategy contract

repository-authored strategy
  explicitly composes training intent using that contract

contract establishment
  validates the authored recipe and produces one accepted run arrangement

repository Trainer
  executes the accepted arrangement without reconstructing model-family topology
  or requiring the authored strategy to remain active
```

The source supports a bounded conclusion:

- Lightning provides the closest precedent for a recipe/executor boundary, but
  its implementation is a larger bidirectional object model rather than the
  contract-established accepted execution boundary being designed here;
- Fabric is a runtime substrate and wrapper system, not an implementation of
  the missing model-family training contract;
- Accelerate already supplies that runtime role and imposes real preparation
  and rebinding constraints that the new contract must represent;
- adopting Lightning wholesale now would replace the contract question with
  Lightning's answer rather than solve it under the repository's requirements;
- replacing Accelerate with Fabric has no demonstrated architectural benefit
  from this audit.

This does not prove that no external subsystem should ever be adopted, nor that
every current Trainer behavior deserves custom maintenance. A concrete,
separable candidate could still be prototyped against the adoption gate.

The current route is therefore to derive the repository contract using these
source findings, not reproduce a framework. If that contract begins
accumulating a Lightning-like dynamic hook surface or duplicating generic
runtime machinery, this comparison must be revisited before implementation
continues.
