# Strategy System Implementation Mapping

## Status And Purpose

This is the source-backed bridge from the model–strategy–Trainer design to the
production-code shape and governing OpenSpec. It replaces the earlier
placeholder that deferred the current-to-target mapping and now serves as a
supporting input to the governing change.

The design sequence is:

```text
settled architecture + exchange work to date + current-code evidence
  -> create the governing OpenSpec
    -> complete its intended-Trainer design and normative requirements
      -> derive target code, migration dependencies, milestones, and tests
```

This document is not a source of architectural authority. Use:

- [`strategy_system_direction.md`](strategy_system_direction.md) for the
  normative architecture and settled semantics;
- [`strategy_contract_exchange_design.md`](strategy_contract_exchange_design.md)
  for the evolving exchange design;
- [`strategy_system_inventory.md`](strategy_system_inventory.md) for the
  broader evidence inventory; and
- [`notes.md`](notes.md) for chronological discussion history.

The current-code mapping below begins with the shared Trainer and ordinary SDXL
fine-tune and adapter paths because together they form the first detailed
implementation case being inspected. SDXL is evidence about one maintained
filing of the contract; it is not the source or shape of the contract itself.
The target architecture comes from the intended Trainer and the
contract/capability system, and it must be tested against multiple model
families and training plans.

SD and SD3 therefore remain in scope, as do cases that depart further from the
ordinary maintained path. Before the execution exchange or governing OpenSpec
design is finalized, their materially different requirements and at least one
strong imperative/research case must be compared with the SDXL evidence. The
isolated executable spike likewise remains evidence about binding and
preparation semantics; it is not a prototype of the production object model.

## Current Production Path

The current path is not one strategy recipe being accepted and executed. SDXL
is shown below only as the first detailed case; the diagram is not a target
family shape. The launcher independently constructs three major inputs whose
responsibilities are then recombined through mutable Trainer state:

```text
RunConfig
   ├─> family TrainingStrategy
   ├─> TrainingMode
   └─> Trainer constructs ObjectiveDefinition
                 │
                 ▼
          Trainer.setup()
     loads components through strategy
     and stores family-shaped projections
                 │
                 ▼
          prepare_models()
     mode selects/builds trainables;
     mode and strategy mutate Trainer state
                 │
                 ▼
         prepare_optimizer()
     mode groups parameters, builds optimizer,
     prepares models/optimizer/scheduler, and rebinds Trainer fields
                 │
                 ▼
         shared training loop
     strategy.process_batch() supplies SDXL batch semantics;
     Trainer/objective/loss modifier/mode perform the rest of the step
```

The construction split is visible in [`train.py`](../../train.py) lines 18–29.
The resulting aggregation is visible in
[`trainer.py`](../../library/training/runners/trainer.py) lines 130–301 and
797–810.

This shape works, but the recipe is spread across:

- the family aggregate in
  [`sdxl/training.py`](../../library/strategies/sdxl/training.py);
- fine-tune or adapter behavior in
  [`finetune_mode.py`](../../library/training/modes/finetune_mode.py) and
  [`adapter_mode.py`](../../library/training/modes/adapter_mode.py);
- the separately selected objective under
  [`library/objectives/`](../../library/objectives/); and
- Trainer fields and phases that act as the shared mutation surface.

That fragmentation—not the mere existence of a shared loop—is the main thing
the new architecture must remove.

## Target Production Path

The target path follows the normative direction:

```text
       INTENDED TRAINER REQUIREMENTS + MAINTAINED CAPABILITY/FEATURE SURFACES
                                      │
                                      ▼
                         TRAINING CONTRACT SYSTEM
                 vocabulary + rules + profiles + lifecycle
                                      │
                   guides and constrains explicit authoring
                                      │
                                      ▼
                        AUTHORED TRAINING STRATEGY
          model integration + treatment + objective + selected behavior
                                      │
               validate and establish under the active contract
                                      │
                                      ▼
                         ACCEPTED RUN ARRANGEMENT
            chosen behavior + obligations + run binding authority
                                      │
                                      ▼
          materialization + coordinated preparation/optimization realization
                                      │
                                      ▼
                    EXECUTABLE RUN STATE + ONE TRAINER ENGINE
```

The contract exists first. Its core is dictated by the intended Trainer, while
its recognized capabilities and feature contracts describe the maintained
library behavior and explicit extension points available to authors. A strategy
is written using that vocabulary and guidance for a selected contract version
and execution/ownership profile.

Contract establishment is not the act of choosing a contract after someone has
written an arbitrary strategy. It is the admission step under the already-active
contract: it checks the complete authored filing, derives its concrete
obligations, rejects unsupported or contradictory choices, and produces the
accepted arrangement. A strategy that needs a changed Trainer boundary must
explicitly target a contract extension or version during authoring.

In other words, “contract establishment” is shorthand for establishing an
authored filing and run arrangement *under* the contract. It does not mean
establishing the contract itself.

The accepted arrangement retains the selected implementations, their
configuration and dependencies, the run authority, and the execution structure
that Trainer needs. The authoring object therefore does not have to remain alive
or participate in ordinary steps.

“Executable run state” is a role label here, not a proposed
`PreparedTrainingProgram` class. The rejected façade was a large lifecycle
method bag that merely renamed strategy. The required target is different: an
accepted arrangement whose operations, state, authority, and allowed effects
have already been established and prepared.

## Current-To-Target Responsibility Mapping

The fate terms mean:

- **retain**: the responsibility and much of its current implementation remain;
- **evolve**: the responsibility remains but its inputs, state, or result become
  explicit;
- **combine**: choices currently assembled as independent top-level axes become
  one coherent authored filing;
- **move**: the implementation belongs under a different established owner;
- **split**: current code combines responsibilities that need distinct owners;
- **replace**: a current state or communication mechanism gives way to the
  already settled authority or exchange;
- **restructure**: useful behavior remains but is recomposed around explicit
  operations and dependencies; and
- **dissolve**: the current top-level abstraction has no target counterpart.

| Current responsibility | Target responsibility | Fate | Code evidence and consequence |
| --- | --- | --- | --- |
| Launcher separately builds a family strategy and mode; Trainer separately builds the objective | The normal construction path first resolves the active contract/version/profile, composes or loads one complete strategy authored for that contract, and invokes establishment before Trainer runtime begins | **evolve and combine** | [`train.py`](../../train.py) lines 18–29 shows the three current construction axes. Objective selection becomes part of strategy authoring rather than an independent Trainer choice. Establishment validates the authored filing against the pre-existing contract; it does not select a contract after the fact. |
| `SdxlTrainingStrategy` combines many mixins and constructs some runtime helpers | Keep an understandable maintained SDXL authoring/composition surface; transfer its accepted implementations and dependencies into the arrangement | **split** | [`sdxl/training.py`](../../library/strategies/sdxl/training.py) and [`base/contracts.py`](../../library/strategies/base/contracts.py) show the current family aggregate and universal contract surface. The aggregate need not remain the runtime object. |
| `TrainingMode` selects subjects, builds adapters, mutates trainability, builds optimization, prepares backend state, and supplies step/state hooks | Authored treatment declares intent; specialized adapter/domain behavior becomes accepted capability behavior; generic trainability, optimization, preparation, and lifecycle timing belong to the pipeline/Trainer | **dissolve and split** | [`finetune_mode.py`](../../library/training/modes/finetune_mode.py) lines 84–273 and [`adapter_mode.py`](../../library/training/modes/adapter_mode.py) lines 115–351 show several unrelated owners hidden behind one mode object. `TrainingMode` must not merely be renamed. |
| `ObjectiveDefinition` is selected beside strategy and Trainer later creates `ObjectiveRuntime` | Objective semantics are authored into the strategy; accepted objective operations and any adaptive run state enter the arrangement and runtime-state contract | **move and split** | [`library/objectives/`](../../library/objectives/) contains a useful semantic implementation, but its current independent selection axis contradicts the target authoring boundary. Trainer may still own step timing and route observations to accepted adaptive state. |
| Trainer owns `loaded_components` plus `denoiser`, `text_encoders`, `vae`, adapter, primary-trainable, and sync-handle projections | The accepted arrangement’s binding authority owns participant identity, current bindings, routes, revisions, and access views; Trainer owns only its backend/runtime state | **replace** | [`trainer.py`](../../library/training/runners/trainer.py) lines 970–1045 shows compatibility projections and setters over mutable component storage. These cannot remain competing authorities. |
| `Trainer.setup()` asks the strategy to load model-family components and then populates Trainer fields | Pipeline orchestration requests materialization; model/domain loaders produce candidate values; the run authority validates and publishes authoritative bindings | **split** | [`trainer.py`](../../library/training/runners/trainer.py) lines 307–431 mixes generic setup, family loading, and state publication. The Trainer must not reconstruct family anatomy. |
| `prepare_models()` performs deferred loading, asks the mode to create/select trainables, applies shared precision, and asks the mode to finish precision | Accepted participant materialization, specialized capability work, semantic trainability realization, and generic precision/preparation become explicit stages over authority-qualified inputs | **split** | [`model_prep.py`](../../library/training/phases/model_prep.py) lines 30–112 currently passes the whole Trainer and mutates its fields. The phase timing can remain while its exchanges become narrow and typed. |
| Adapter mode resolves targets, constructs/loads adapter state, attaches effects, selects continuation behavior, and writes the adapter back to Trainer | Accepted adapter capability behavior resolves targets and proposes participant/relationship transitions; Trainer coordinates lifecycle and the authority publishes accepted state | **move and evolve** | [`adapter_mode.py`](../../library/training/modes/adapter_mode.py) lines 115–179 demonstrates that adapter materialization, attachment, continuation, and trainability are separate meanings even though one method currently performs them. |
| Fine-tune and adapter modes realize `requires_grad`, `train()`/`eval()`, selected parameters, and a primary trainable | Strategy declares semantic training subjects and constraints; optimization/pipeline realization derives parameter membership, trainability, synchronization, clipping, and ordinary runtime modes | **split** | [`finetune_mode.py`](../../library/training/modes/finetune_mode.py) lines 84–158 and [`adapter_mode.py`](../../library/training/modes/adapter_mode.py) lines 115–290 currently mix semantic selection with mechanical realization. “Primary trainable” and separate boolean flags cannot remain independent authority. |
| Modes build logical/execution parameter groups and immediately construct a concrete optimizer | Accepted semantic optimization plan is resolved against current bindings; a Trainer-owned realization attempt constructs candidates and later publishes the complete prepared optimization runtime | **evolve** | [`optimization/types.py`](../../library/optimization/types.py) lines 38–98 already separates logical and execution groups usefully. [`optimizer.py`](../../library/training/phases/optimizer.py) lines 61–150 shows why concrete optimizer/scheduler state crosses runtime preparation. Raw `nn.Parameter` lists and mutable runtime metadata are not durable semantic identity. |
| Modes call Accelerate/DeepSpeed and directly replace Trainer model, optimizer, scheduler, sync, and primary-trainable fields | A preparation request/result exchange jointly prepares the required participants and optimization candidates; authority routes and Trainer backend-group state publish together after all fallible work succeeds | **replace** | [`finetune_mode.py`](../../library/training/modes/finetune_mode.py) lines 234–273 and [`adapter_mode.py`](../../library/training/modes/adapter_mode.py) lines 243–290 show backend-specific replacement and composite handles. Prepared wrappers do not replace participant identity, and a DeepSpeed composite is Trainer infrastructure rather than a participant. |
| SDXL `process_batch()` performs latent preparation, conditioning, objective input construction, denoiser invocation, target construction, and loss policy | Establish and prepare a structured set of accepted training operations with explicit dependencies, inputs, results, observations, and permitted effects | **restructure** | [`sdxl/diffusion.py`](../../library/strategies/sdxl/diffusion.py) lines 33–242 shows several recognizable operations behind one broad call. This supports reusable operation/region structure but does not by itself require a literal graph. |
| The shared loop calls `strategy.process_batch()` and then performs objective updates, loss modification, backward, clipping, optimizer/scheduler advancement, triggers, and observation | Trainer keeps generic time, accumulation, synchronization, backward, clipping, advancement, triggers, and observation according to the selected ownership profile; it executes or invokes only accepted prepared behavior for model/objective-specific semantics | **retain and narrow** | [`training_loop.py`](../../library/training/phases/training_loop.py) lines 461–618 contains a viable Trainer engine skeleton. Its problem is the broad strategy/mode/Trainer-state dependency, not that generic loop mechanics exist. |
| SDXL validation owns both validation traversal/RNG handling and model/objective-specific evaluation semantics | Pipeline owns validation scheduling, traversal, ordinary mode transitions, and result routing; the accepted validation capability supplies prepared evaluation semantics | **split** | [`sdxl/validation.py`](../../library/strategies/sdxl/validation.py) lines 18–196 repeats much of the training semantic path while also owning orchestration. Training and validation should reuse accepted operations without making strategy an active callback object. |
| Mode hooks register checkpoint state and adapter step effects through the whole Trainer | Runtime checkpointing coordinates restoration; accepted capabilities contribute typed state and effects; identity restoration follows the settled authority/ref rules | **split** | Adapter state hooks and maximum-norm behavior in [`adapter_mode.py`](../../library/training/modes/adapter_mode.py) are real specialized contributions, but they do not justify a parallel mode authority or arbitrary whole-Trainer hooks. |

## What The First SDXL Evidence Pass Tells Us

The current SDXL `process_batch()` is one maintained implementation case, not a
template for the universal contract. Its body is useful evidence because it
already contains operations with recognizable meanings:

```text
batch
  -> obtain or encode latent representation
  -> obtain or encode conditioning
  -> sample/construct objective inputs
  -> invoke the prepared denoiser route
  -> construct the objective target
  -> compute loss and declared observations
```

Validation reuses most of those meanings with different traversal, gradient,
timestep, and reduction policies. For this case, that suggests a structured
accepted execution form made from reusable operations or regions. Another model
or training plan may omit, replace, reorder, combine, or add operations. The
SDXL decomposition therefore tells us what the system must be able to express
for maintained SDXL; it does not establish the minimum contract or settle
whether the production representation is a graph, tree, schedule, lowered
Python composition, or a hybrid.

The important code boundary is simpler:

- static choices, configuration, selected implementations, dependencies, and
  participant routes are established or prepared before the loop;
- the ordinary step supplies only genuinely changing inputs such as the batch
  and run coordinates;
- accepted operations perform the model- and objective-specific semantics; and
- Trainer retains the generic mechanics promised by the selected
  execution/ownership profile.

This is how the authored strategy can disappear without moving every model
capability into the `Trainer` class. The accepted arrangement keeps the chosen
capability implementations and prepared operations; Trainer only coordinates
and executes what the contract already accepted.

## Code That Is Likely To Survive

The redesign is not a rewrite from nothing. The following are useful foundations:

- the shared phase and loop skeleton in
  [`library/training/phases/`](../../library/training/phases/) and
  [`trainer.py`](../../library/training/runners/trainer.py);
- SDXL’s domain implementations for loading, representation, conditioning,
  diffusion objectives, validation, sampling, and persistence;
- the corresponding SD and SD3 domain implementations after their differing
  paths are mapped;
- adapter construction, target resolution, attachment, continuation, state,
  and serialization implementations;
- Accelerate/DeepSpeed as Trainer-owned preparation services; and
- the logical-versus-execution distinction already begun by
  `OptimizationPlan`.

They survive by receiving narrower accepted inputs and returning explicit
results. Their current placement behind one family strategy, one mode object, or
mutable Trainer fields is not part of what must survive.

## Code That Should Not Define The Target

The target must not preserve these current conveniences as architecture:

- `TrainingStrategy`, `TrainingMode`, and objective as three independent runtime
  peers;
- `process_batch()` as one permanent universal strategy callback;
- whole-Trainer mutation as the exchange between phases and specialized code;
- family-shaped Trainer fields as authoritative component state;
- adapter/fine-tune branching as the top-level organization of training;
- one “primary trainable” or one backend composite as participant identity; or
- a renamed all-purpose prepared-program façade.

## OpenSpec Design Dependency Order

The remaining design should continue inside the governing OpenSpec, backward
from what the intended Trainer must execute. This is the order for deriving the
concrete code, not the order in which the system gains authority at runtime.
The contract system remains prior to strategy authoring even though the exact
Python authoring and establishment APIs are chosen after their required output
is understood.

1. **Seed one governing OpenSpec from the completed exploration.** Carry forward
   the normative direction, settled semantics, exchange decisions, current-code
   inventory, implementation mapping, and chronological corrections as cited
   support. Do not copy superseded notes into normative requirements.
2. **Start from the intended-Trainer consumption frame.** Use the existing core
   and capability meanings to state what training execution and validation must
   let the Trainer/pipeline coordinate, what accepted state they require, and
   which authority the selected execution/ownership profile leaves with
   Trainer. Do not name SDXL participants or copy `process_batch()` into the
   contract.
3. **Map several implementation cases against that frame.** Begin with ordinary
   SDXL fine-tune and adapter paths as the first detailed source pass. Then
   inspect materially different SD and SD3 paths and at least one strong
   imperative/research case. These cases test what the contract must be able to
   express; no one case defines the core.
4. **Derive the minimum accepted execution meanings from both sides.** Reconcile
   the Trainer-led contract requirements with the behavior the representative
   filings need: operations or regions, dependencies, changing inputs, results,
   observations, effects, and authorized imperative scope. This is not yet the
   final step exchange.
5. **Finish the optimization realization and advancement exchange.** Use the
   accepted training subjects and prepared execution needs to define candidates,
   preparation membership, published runtime, normal advancement, and the
   explicit extension boundary.
6. **Complete the step exchange.** Now connect accepted execution behavior to
   the known optimization boundary without treating today’s `process_batch()` or
   one loss-result shape as universal.
7. **Choose concrete binding and preparation request/result types.** Their
   semantics are settled; their Python shape should be derived from the actual
   execution and optimization consumers rather than copied from the isolated
   spike.
8. **Choose the accepted-arrangement and executable-run code shape.** This is
   where selected operations, capability state, the run authority, prepared
   routes, optimization runtime, and authorized imperative regions meet.
9. **Choose the concrete authored-strategy and establishment APIs.** They must
   expose the pre-existing contract’s vocabulary and guidance, collect the
   explicit filing establishment needs, and produce only a contract-accepted
   arrangement. Designing these Python APIs here does not move the contract
   after the strategy in the actual architecture.
10. **Finalize traced implementation milestones and tests.** Each task should
    reference the governing requirement and scenarios it implements, remove or
    replace the corresponding old authority, and avoid a knowingly false
    temporary contract.

The trace for every material responsibility should be:

```text
intended Trainer responsibility
  -> contract requirement
    -> accepted-arrangement exchange
      -> representative scenario
        -> implementation milestone
          -> acceptance test
```

The OpenSpec proposal defines scope; its design owns the intended Trainer and
architecture; its focused specs own normative requirements and scenarios; its
tasks own the reviewed migration. This mapping and the other `docs_design/`
records remain supporting evidence rather than a parallel source of new
authority.

## Remaining Code-Shape Decisions

These are real remaining decisions, but most are downstream Python design rather
than reopened architecture questions:

### Construction and contract establishment

- Choose the readable Python construction API that accepts one authored
  strategy definition under the already-selected contract/version/profile and
  returns one accepted run arrangement.
- Make bypassing establishment or mixing contract versions impossible through
  the normal construction path.
- Decide whether the authoring surface uses ordinary composition, a factory,
  builder, decorators for custom operations, or a narrow combination.

### Structured and custom execution

- Choose the concrete operation/region representation and its dependency
  wiring.
- Define how maintained and custom structured operations file conformance
  without automatic method discovery or import-order-dependent registration.
- Define the strong imperative-region API and the authority it may request under
  an explicit execution/ownership profile.

### Obligation and runtime-evidence types

- Define the typed obligations derived during establishment.
- Separate facts known at establishment from evidence that loading,
  materialization, optimization realization, or preparation can supply only
  later.
- Avoid both one universal model-kind union and an unstructured fact dictionary.

### Binding, preparation, and optimization types

- Define the production participant, relationship, bound-state, route,
  snapshot, scoped-view, attempt, and publication types using the settled
  semantics.
- Define semantic optimization plan, concrete candidate, prepared runtime, and
  advancement result types without conflating optimizer objects with unit
  identity.
- Preserve atomic multi-participant and participant-plus-relationship changes,
  revisions, invalidation, structured failure, and history without copying the
  spike’s provisional names.

### Migration slicing

- Choose milestones that never leave both Trainer fields and the new authority
  acting as sources of truth.
- Avoid a temporary renamed `TrainingMode`, active-strategy callback contract,
  or façade that would need to be removed immediately afterward.
- Preserve current SD, SDXL, and SD3 behavior while each responsibility moves to
  its intended owner, using SDXL only as the first detailed migration slice.

This mapping should inform the governing OpenSpec design and remaining exchange
work. Once those semantics are complete, the mapping—not preliminary vocabulary
from the discussion or isolated spike—should drive final module placement,
class names, traced tasks, and tests inside that change.
