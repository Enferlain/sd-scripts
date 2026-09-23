# Provisional tree layout

The names are not final.

Things that will need to be looked at for names:

- strategies/contract.py, requirements.py, checking.py
- strategies/strategy/training.py
- training/run_state (pobably just state)
- training/run_state/creation.py, definition.py, authority.py, relationships.py, bindings.py

---

Here is the provisional strategy and training layout as one system. Names remain adjustable, but every placement has a distinct job.

```text
library/
|
+-- strategies/
|   |
|   +-- contract/
|   |   +-- __init__.py
|   |   |     Public strategy-author API.
|   |   |     Re-exports the stable contract vocabulary and main contract.
|   |   |
|   |   +-- contract.py
|   |   |     Central readable/queryable TrainingContract.
|   |   |     Assembles its core, capabilities, features, and profiles.
|   |   |
|   |   +-- core.py
|   |   |     Meanings required by every compatible training run.
|   |   |     Defines common strategy-facing behavior seams.
|   |   |
|   |   +-- capabilities.py
|   |   |     Contracts for caching, validation, sampling,
|   |   |     persistence, restoration, and future pipeline operations.
|   |   |
|   |   +-- features.py
|   |   |     Contracts for reusable authoring features:
|   |   |     representation, conditioning, prediction, PEFT, etc.
|   |   |
|   |   +-- profiles.py
|   |   |     Standard Trainer ownership and explicit research extensions.
|   |   |
|   |   +-- requirements.py
|   |   |     Run-specific requirements derived from the general contract.
|   |   |     Defines evidence and readiness meanings used after authoring.
|   |   |
|   |   +-- checking.py
|   |         Checks one completed strategy using the contract.
|   |         Produces detailed problems or derived run requirements.
|   |
|   +-- features/
|   |   |
|   |   +-- representation/
|   |   |     +-- latent.py
|   |   |     +-- pixel.py
|   |   |
|   |   +-- conditioning/
|   |   |     +-- clip.py
|   |   |     +-- weighted_prompts.py
|   |   |
|   |   +-- prediction/
|   |   |     Reusable predictor invocation/integration behavior.
|   |   |
|   |   +-- objectives/
|   |   |     Reusable strategy integration for objective implementations.
|   |   |
|   |   +-- peft/
|   |         Shared authored PEFT integration and method adoption.
|   |
|   +-- sd/
|   |   +-- training.py
|   |   |     Readable maintained SD recipe.
|   |   |
|   |   +-- <SD-specific implementations large enough to stand alone>
|   |
|   +-- sdxl/
|   |   +-- training.py
|   |   |     Readable maintained SDXL recipe.
|   |   |
|   |   +-- <SDXL-specific implementations large enough to stand alone>
|   |
|   +-- sd3/
|   |   +-- training.py
|   |   |     Readable maintained SD3 recipe.
|   |   |
|   |   +-- <SD3-specific implementations large enough to stand alone>
|   |
|   +-- <future maintained or compound strategy>/
|       +-- training.py
|       +-- <strategy-specific implementations>
|
+-- training/
    |
    +-- run_state/
    |   +-- __init__.py
    |   |
    |   +-- creation.py
    |   |     create_run(...)
    |   |     Checks the completed strategy and creates the initial
    |   |     governed run, or returns detailed failure information.
    |   |
    |   +-- definition.py
    |   |     Accepted contract/version/profile, selected behavior,
    |   |     capabilities, dynamic policies, and run requirements.
    |   |
    |   +-- authority.py
    |   |     The one writer for canonical current run state.
    |   |
    |   +-- participants.py
    |   |     Run participant identities, declarations, and lifecycles.
    |   |
    |   +-- relationships.py
    |   |     Relationship identities, endpoints, states, and revisions.
    |   |
    |   +-- bindings.py
    |   |     Current bound state, execution routes, and typed access views.
    |   |
    |   +-- snapshot.py
    |   |     Coherent current state and revision/freshness information.
    |   |
    |   +-- transitions.py
    |         Materialization, replacement, amendment, rebinding,
    |         relationship changes, retirement, and restoration transitions.
    |
    +-- capabilities/
    |   +-- caching.py
    |   |     Accepted caching requests, readiness, and result coordination.
    |   |
    |   +-- validation.py
    |   |     Scheduling inputs, traversal coordination, aggregation,
    |   |     and result routing.
    |   |
    |   +-- sampling.py
    |   |     Trigger/request/destination and result coordination.
    |   |
    |   +-- persistence.py
    |   |     Trained-product request and result coordination.
    |   |
    |   +-- restoration.py
    |         Exact-run restoration coordination and contributor handling.
    |
    +-- phases/
    |   +-- materialization.py
    |   |     Requests domain-produced participants and submits evidence.
    |   |
    |   +-- caching.py
    |   |     Runs requested pre-training cache work through capabilities.
    |   |
    |   +-- preparation.py
    |   |     Coordinates precision, device, distributed, compilation,
    |   |     routes, and backend state as one governed attempt.
    |   |
    |   +-- optimization.py
    |   |     Coordinates semantic-plan realization through
    |   |     library/optimization.
    |   |
    |   +-- loop.py
    |   |     Drives batches, accepted operations, Trainer mechanics,
    |   |     triggers, observations, and transitions.
    |   |
    |   +-- finalization.py
    |         Cleanup and requested end-of-run capabilities.
    |
    +-- trainer.py
          Owns generic execution mechanics:
          time, accumulation, synchronization, backward, clipping,
          optimizer/scheduler advancement, zeroing, interruption,
          observation, and cleanup.
```

## How the similarly named areas differ

```text
strategies/contract/capabilities.py
    defines what a valid capability means

training/capabilities/
    coordinates accepted capability requests during a run
```

Likewise:

```text
strategies/contract/features.py
    defines what feature implementations must provide

strategies/features/
    contains reusable implementations of those contracts
```

And:

```text
strategies/<strategy>/training.py
    selects and connects behavior

training/trainer.py
    executes generic mechanics over the resulting prepared run
```

## Overall flow

```text
strategies/contract
    provides the authoring vocabulary and guidance
            |
            v
strategies/sdxl/training.py
    composes one complete strategy
            |
            v
training/run_state/creation.py
    applies contract checking
    derives run requirements
    establishes identities and authority
            |
            v
training/phases/
    materializes, caches, prepares, and realizes optimization
            |
            v
training/trainer.py
    executes the prepared run
            |
            +--> training/capabilities/
            |        coordinates requested operations
            |
            +--> training/run_state/
                     governs transitions and current state
```

A few boundaries remain intentionally flexible:

- Family packages do not receive a fixed template beyond `training.py`; supporting files appear according to actual behavior.
- Feature subfolders appear only when implementations exist.
- The exact accepted-operation representation still depends on the execution design work. Its definitions will likely be referenced by `run_state/definition.py` and consumed by `trainer.py`, but we should not invent an `execution/` package before that representation is chosen.
- The outer caller that sequences run creation, realization phases, and Trainer construction still needs a clear API; it should not force another `runners/` abstraction merely to hold that sequence.

---

The sweep supports the target direction. The existing `training/` package should be reorganized around `Trainer`, run state, phases, and capabilities—but the code cannot be moved wholesale. Current ownership crosses those boundaries in both directions.

One important distinction:

> “Move to Trainer” usually means move under training-owned coordination, not put it directly into the `Trainer` class.

The final `Trainer` should be much smaller than today.

## What the current package is doing

The current [`Trainer`](/home/imi/Projects/sd-scripts/library/training/runners/trainer.py:115) simultaneously holds:

1. Run-authority state:
   loaded components, model realization, current model references.

2. Trainer execution state:
   progress, dataloaders, optimizer, scheduler, backend, synchronization, logging.

3. Behavior-owned state:
   objective runtime, cache backends, adapter state, validation state, loss-modifier state.

That is why it has accumulated so many fields and why phases accept the entire `Trainer`.

The current launcher also exposes the old topology directly:

```text
build aggregate TrainingStrategy
build TrainingMode
give both to Trainer
```

That is visible in [`train.py`](/home/imi/Projects/sd-scripts/train.py:18). The current aggregate [`TrainingStrategy`](/home/imi/Projects/sd-scripts/library/strategies/base/contracts.py:546) combines loading, caching, sampling, validation, persistence, preparation, and per-batch computation into one runtime object.

The graph found calls in both directions:

```text
training phases → strategy callbacks
concrete strategies → training helpers and Trainer fields
```

So this is a real circular ownership problem, not merely an untidy folder layout.

## Training-folder disposition

| Current area | Target disposition |
|---|---|
| `runners/trainer.py` | Becomes the smaller `training/trainer.py`. Keeps overall execution, time, phase ordering, failure handling, and cleanup. Stops owning arbitrary component slots and stops receiving raw strategy/mode objects. |
| `phases/training_loop.py` | Remains the loop phase. Keeps accumulation, synchronization, backward, clipping, optimizer/scheduler advancement, step accounting, and triggers. Calls approved runtime behavior instead of `strategy.process_batch()` and `TrainingMode` hooks. |
| `phases/triggers.py` | Remains training-owned lifecycle scheduling. It may be shared by capability coordinators. |
| `phases/caching.py` | Splits into caching phase timing plus a caching capability coordinator. Traversal, storage, device staging, and publication stay in training; semantic encoders/codecs remain selected strategy behavior. |
| `phases/model_prep.py` | Dissolves into materialization, preparation, and optimization concerns. It should not continue as a collection of strategy hooks. |
| `phases/optimizer.py` | Becomes training-owned optimization realization and backend preparation. Validation-dataloader setup and restoration work move to their respective concerns. |
| `phases/validation.py` | Grows into the training-owned validation coordinator, probably under capabilities. Current scheduler logic stays training-owned. |
| `phases/orchestration_helpers.py` | Dissolves. Monitoring, evaluation projections, optimizer mode changes, sampling, and validation belong to different owners. |
| `modes/` | Removed as an architectural concept. Adapter and fine-tune modes are decomposed rather than renamed. |
| `checkpointing.py` | Splits into trained-artifact persistence and exact runtime restoration. Naming, retention, writing, upload, and result reporting remain training infrastructure. |
| `sample_generation.py` | Splits into a sampling coordinator and selected generation behavior. Request traversal, destination, triggering, and result handling stay in training. Model-family pipeline construction does not. |
| `trainer_utils.py` | Dissolves into backend preparation, optimization mechanics, validation RNG handling, restoration, and observability. |
| `diffusion.py` | Moves out of generic training. Latent representation and VAE encoding are selected model/representation behavior. |
| `noise_utils.py` | Moves to the objective area, not strategy or Trainer. Its production consumer is the DDPM objective in [`ddpm.py`](/home/imi/Projects/sd-scripts/library/objectives/ddpm.py:17). |
| `interrupts.py` | Stays training-owned, associated with Trainer lifecycle and cleanup. |
| `metadata.py` | Compatibility reexports should disappear. Metadata consumes typed run projections rather than owning training state. |
| `_deprecated/` | Removed after replacement paths exist; none of it defines the target architecture. |

The core loop demonstrates the intended division surprisingly well. In [`training_loop.py`](/home/imi/Projects/sd-scripts/library/training/phases/training_loop.py:461), the generic mechanics beginning with backward, clipping, stepping, zeroing, and progress are Trainer work. The broad call to `strategies.process_batch()` and the `mode` callbacks are the parts that must be replaced.

## What actually moves from training to strategy or another domain

Very little should move specifically to strategy:

- [`training/diffusion.py`](/home/imi/Projects/sd-scripts/library/training/diffusion.py:34) prepares model-family latent representations and is called from family diffusion and validation code. It belongs with representation behavior, potentially shared by several family features.
- Parts of `sample_generation.py` that construct or operate a particular diffusion pipeline belong with selected generation implementations or `library/pipelines/`.
- `noise_utils.py` belongs with DDPM/objective regularization rather than strategy.

Most of the rework is therefore not “push training code into strategies.” It is splitting training’s existing grab-bags and pulling generic workflow ownership out of concrete strategies.

## What moves from concrete strategies into training ownership

These are the important transfers.

### Validation coordination

Concrete family strategies currently own the entire validation loop. For example, SDXL’s [`calculate_val_loss()`](/home/imi/Projects/sd-scripts/library/strategies/sdxl/validation.py:106) owns RNG isolation, dataloader traversal, step limits, aggregation, and recorder updates.

Those are training-capability responsibilities.

The family-specific `process_val_batch()` computation remains selected model/objective behavior.

### Sampling coordination

SDXL’s [`sample_images()`](/home/imi/Projects/sd-scripts/library/strategies/sdxl/sampling.py:67) currently owns:

- trigger checking;
- unwrapping;
- device movement;
- evaluation preparation;
- cleanup and restoration;
- generic sample traversal.

Training’s sampling capability should own those. SDXL pipeline construction and image-generation semantics stay with the selected SDXL implementation.

### Persistence coordination

Family checkpoint code currently receives the entire Trainer. SDXL’s implementation directly reads config, accelerator, denoiser, encoder cardinality, VAE, output paths, and upload settings in [`checkpointing.py`](/home/imi/Projects/sd-scripts/library/strategies/sdxl/checkpointing.py:49).

Training persistence should own:

- the persistence request;
- coherent state capture;
- selected product members;
- output planning;
- writing and upload;
- retention;
- reporting what was actually produced.

Family conversion and serialization remain family/domain code, but receive only the selected product projection—not the entire Trainer.

### Generic preparation mechanics

Concrete `ModelPreparationStrategy` implementations currently answer casting questions and perform family workarounds. For example, [`SdxlModelPreparationStrategy`](/home/imi/Projects/sd-scripts/library/strategies/sdxl/model_preparation.py:12) combines:

- generic casting choices;
- CLIP-specific FP8 support;
- gradient-checkpointing support;
- a semantic rule freezing the unstable TE1 tail.

These should not all move wholesale into Trainer.

Instead:

- generic casting, wrapping, gradient setup, and backend preparation become training infrastructure;
- the accepted strategy supplies requirements and any selected family-specific preparation operation;
- constraints such as the TE1 tail rule are checked and enforced through the accepted training-subject/preparation definition.

### Materialization publication

Family loaders should remain family implementations. Trainer should not learn how to load SDXL or SD3.

What moves to training is the surrounding job:

```text
request materialization
invoke selected loader
collect typed facts
check accepted obligations
publish bindings atomically through run authority
```

### Runtime execution context

The current strategy itself publishes a global [`StrategyContext`](/home/imi/Projects/sd-scripts/library/strategies/base/context.py:24), which adapters such as T-LoRA read.

In the target, common facts such as phase, step, selected routes, batch, and timesteps should be supplied by the training execution boundary. The model-specific denoiser call remains selected behavior, while consumers such as T-LoRA still own how they use those facts.

## `TrainingMode` is almost entirely displaced training work

The [`TrainingMode` protocol](/home/imi/Projects/sd-scripts/library/training/modes/base.py:26) currently controls trainable creation, precision, optimizer construction, backend preparation, restoration hooks, clipping parameters, train/eval state, step hooks, persistence, and diagnostics.

It should dissolve as follows:

- PEFT selection and supported methods: authored strategy feature.
- Direct/PEFT training-subject declaration: strategy authoring.
- Concrete target and parameter resolution: training optimization realization.
- Trainability enforcement and non-overlap: training optimization.
- Optimizer/scheduler creation and backend preparation: training.
- Train/eval projections: training capability/runtime projections.
- Adapter-specific effects such as max-norm regularization: approved adapter runtime behavior.
- Artifact serialization: selected persistence implementation.
- Exact state restoration: restoration contributor coordinated by training.

`FineTuneMode` contains no durable “mode” behavior. It is mostly standard training plumbing.

## New training-side code genuinely required

The existing code cannot represent several parts of the design:

- One run authority for participants, relationships, bindings, routes, revisions, and freshness.
- A run-creation boundary that receives the completed authored recipe, invokes contract-guided checking, and seeds accepted state.
- Materialization requests, candidate results, evidence, and atomic publication.
- One coherent preparation job spanning participants, backend representations, optimizer runtime, and publication.
- A semantic optimization definition separate from concrete parameters and optimizer objects.
- A prepared optimization runtime used by the loop.
- A runtime execution seam carrying batches, coordinates, approved state access, results, observations, and requested effects.
- Capability coordinators for caching, validation, sampling, persistence, and restoration.
- Contributor-based exact restoration.
- Explicit enforcement of any research/imperative authority.
- Typed projections and events for metadata and observability.

The existing optimization package is useful lower-level machinery, but it is not the new semantic layer. Its current [`OptimizationPlan`](/home/imi/Projects/sd-scripts/library/optimization/types.py:48) already contains live parameters, while [`OptimizationTargetRef`](/home/imi/Projects/sd-scripts/library/optimization/targets.py:12) contains the live object itself. These fit candidate/runtime realization, not the authored semantic optimization definition.

## Two concerns missing from the provisional training sketch

The proposed `run_state / capabilities / phases / trainer.py` structure still needs explicit homes for:

- Data/input lifecycle: manifest construction, train/validation dataloaders, epoch input preparation, and resumption position.
- Lifecycle observability: phase monitoring, runtime traces, observer events, run reports, failure reporting, and cleanup.

These do not necessarily require two more top-level folders. But they must be assigned deliberately rather than remaining hidden in `Trainer` and `orchestration_helpers.py`.

## Resulting dependency order

This is a construction dependency order, not a runnable-slice plan:

1. Define exactly what the future Trainer consumes.
2. Define run authority and its scoped projections.
3. Define materialization, preparation, and optimization realization around that authority.
4. Rebuild Trainer and the loop against a hand-built accepted run fixture, without depending on the new strategy implementation yet.
5. Build the recognized capability coordinators, plus data lifecycle and observability.
6. Make the strategy contract and checking process produce the structures the training side now demonstrably requires.
7. Rebuild SDXL, SD, and SD3 strategy definitions and selected implementations against that contract.
8. Replace the launcher boundary and remove `TrainingMode`, the aggregate strategy contract, compatibility projections, and deprecated paths.

That matches the training-first approach: we first make the intended Trainer concrete enough to define its contract. Then strategy authoring and checking catch up to it. It avoids deriving the architecture from today’s `TrainingStrategy` while still preserving all current model-specific behavior as evidence.

No files were changed during this sweep. The existing untracked `provisional_tree.md` was left untouched.
