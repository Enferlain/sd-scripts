# Strategy Contract Exchange Design

## Status And Authority

This is the evolving pre-OpenSpec design record for the concrete exchanges
between the authored training strategy, its internal binding authority, and
Trainer-owned infrastructure.

It is deliberately downstream from
[`strategy_system_direction.md`](strategy_system_direction.md): the direction
document remains normative for the model–strategy–trainer architecture and the
settled Q1–Q5 semantics. This document must not reopen those decisions. It
turns them into concrete exchange meanings, invariants, inputs, results,
allowed side effects, failure behavior, and eventually candidate Python
shapes.

The other records retain their existing roles:

- [`strategy_system_inventory.md`](strategy_system_inventory.md) is evidence
  about current production code and pressure scenarios;
- [`strategy_system_implementation_mapping.md`](strategy_system_implementation_mapping.md)
  holds deferred Python/code-shape decisions and will later map current code
  into the target design and migration milestones;
- [`notes.md`](notes.md) is chronological discussion history and may contain
  superseded positions; and
- [`framework_pattern_comparison.md`](framework_pattern_comparison.md) is
  subordinate prior-art research rather than a source of requirements.

This document is not an implemented contract or an implementation plan. Names
such as `ParticipantKey`, `ParticipantRef`, `PreparationPlan`, and
`PreparationResult` are working vocabulary until a decision explicitly marks
them as accepted. The eventual OpenSpec should be derived from the completed
exchange semantics rather than treating every working name below as final.

The training contract system remains the authoritative acceptance definition
for the Trainer. This record must not move ordinary compatibility knowledge or
validation composition onto strategy authors. An authored strategy explicitly
selects and wires its intent, participants, features, capabilities, and bounded
choices; the contract system judges that complete authored definition, derives
the obligations implied by its known contracts and implementations, and only
then establishes the strategy runtime that may reach Trainer. The binding
authority is the runtime enforcement responsibility of that same accepted
contract, not a second source of compatibility meaning.

Decision markers have these meanings:

```text
SETTLED INPUT
  inherited from the normative Q1-Q5 direction; not reopened here

WORKING DECISION
  current recommended answer for a downstream concrete question; may be
  refined before OpenSpec

SUPERSEDED
  an earlier working answer retained for chronological clarity after a later
  correction replaced its authority or semantics

OPEN
  a concrete question that still needs an answer
```

## Scope And Completion Gate

This record must define four exchanges:

1. binding and arrangement-state establishment;
2. runtime preparation and authoritative execution rebinding;
3. optimization intent, realization, and ownership; and
4. per-step execution and result handling.

Binding and runtime preparation are designed together first. Optimization
ownership and the optimization exchange follow. The step exchange follows
once it can refer honestly to the accepted optimization boundary.

The pre-OpenSpec design is complete enough to propose a governing change when:

- each exchange has defined inputs, results, state effects, allowed external
  side effects, and failure conditions;
- the standard Trainer-owned optimization profile and explicit research
  extension are defined;
- the exchanges pass the current SD, SDXL, SD3, adapter, deferred-loading,
  distributed-preparation, persistence, replacement, and compound-strategy
  pressure scenarios; and
- the migration can be divided into reviewed milestones without introducing a
  temporary contract that discards settled identity, revision, route,
  freshness, or persistence meaning.

The likely result is one governing OpenSpec change containing several
capability specifications and numbered implementation milestones. This does
not imply one big-bang implementation.

## Dependency Path For Deriving The Code

### WORKING METHOD: design backward from the intended Trainer, not from the
current strategy classes

The runtime direction and the design-dependency direction are intentionally
opposite:

```text
runtime use
  authored definition
    -> contract establishment
      -> accepted TrainingStrategy
        -> Trainer execution

design derivation
  intended Trainer responsibilities
    -> Trainer-facing exchanges and observable results
      -> state and guarantees an accepted TrainingStrategy must provide
        -> contract establishment and conformance enforcement
          -> authored-definition inputs and authoring API
```

This follows from the contract being the acceptance definition **for the
Trainer**. The authoring input cannot be designed coherently until the accepted
result is known, and that result cannot be known until the intended Trainer's
interactions and ownership boundaries are known.

"Start from Trainer" does not mean mechanically preserving today's `Trainer`
fields, phases, or method calls. The current code predates this direction and
is migration evidence only. Every current dependency must first be classified
as one of:

```text
intended Trainer mechanism
  timing, infrastructure, distributed coordination, optimization, observation

Trainer-recognized capability
  a named request/result/lifecycle operation the pipeline deliberately knows

strategy-internal feature
  authored behavior used behind the complete strategy boundary

model/component mechanic
  behavior inherent to construction, loading, execution, or serialization

current topology or ownership leakage
  a convenience projection or mutable back-channel that need not survive
```

Only the first two categories directly determine the Trainer-facing contract.
The other categories constrain implementations or migration without enlarging
that surface automatically.

### Two evidence paths must remain separate

The target semantic dependency path is:

```text
intended Trainer mechanism
  -> core and recognized-capability exchanges
    -> required accepted state, readiness, and results
      -> accepted TrainingStrategy boundary and internal authority
        -> establishment-time obligations and conformance checks
          -> explicit authored definition
```

The current-code migration path is:

```text
launcher/config validation/factories
  -> current family strategy + TrainingMode + objective
    -> mutable Trainer and phase consumers
      -> useful typed islands and current pressure cases
        -> mapping into the target exchanges
```

The migration path may reveal a missing requirement or a difficult
transition, but it does not reverse authority and make a current class,
family-shaped projection, or configuration branch normative.

### Current-code evidence for this order

The active path demonstrates why authoring or validation should not be the
first API designed:

- `train.py` runs config-only preparation and validation before constructing
  any strategy, so current validation cannot judge the complete authored
  arrangement;
- `build_training_strategy()` selects one family aggregate from
  `cfg.model.model_type`, while the aggregate constructor initializes some
  runtime behavior such as tokenizers without filing an accepted contract;
- `Trainer` separately receives `TrainingStrategy` and `TrainingMode`, then
  constructs its own objective, so the value reaching Trainer is not yet the
  one complete authored strategy required by the direction;
- `Trainer.setup()` installs strategy-produced components as Trainer-owned
  state, and later phases pass those projections back into strategy and mode
  calls; and
- the current `TrainingStrategy` ABC combines core-looking operations,
  pipeline capabilities, family-internal collaboration, and conditional
  raising defaults, so its method list is not a semantic definition of the
  future core.

These facts are evidence for responsibilities that must be represented, not a
template for the target call graph.

### Derivation sequence

Use this order to determine the eventual code:

1. **Intended Trainer skeleton.** State what the target Trainer owns and what
   it must accomplish without naming SD-shaped participants or copying the
   current phase APIs.
2. **Trainer consumption table.** For each core exchange and recognized
   capability, define the request meaning, result meaning, lifecycle point,
   required accepted-state guarantees, and owner of side effects.
3. **Accepted strategy boundary.** Derive the minimum static declarations,
   behaviors, scoped current-state views, and internal authority that Trainer
   must be able to rely on. This is the output side of contract establishment.
4. **Runtime exchange dependencies.** Complete binding before preparation,
   preparation before optimization realization, and optimization before the
   step exchange. Attach caching, validation, sampling, persistence, resume,
   and other capabilities at the earliest state boundary their requests
   actually require.
5. **Contract establishment.** Define how the contract system derives and
   enforces the obligations necessary to create that accepted boundary.
6. **Known and custom conformance.** Define how maintained library selections
   prove the required exchanges and where explicit direct/custom conformance
   or contract extension begins.
7. **Authored-definition API.** Only then choose the Python authoring shape
   that supplies the explicit selections and wiring establishment needs.
8. **Migration and placement.** Map current strategy facets, mode behavior,
   objective behavior, phases, and typed islands into reviewed milestones;
   choose final modules and class names after recurring stages and dependency
   directions are concrete.

This sequence does not require finalizing all of Trainer before making
progress. The first bounded artifact is a semantic Trainer-consumption table
for the four core exchanges and the already recognized capabilities. Binding
and preparation remain the first exchanges completed because optimization and
step semantics depend on their accepted current state.

### First intended-Trainer consumption frame

This is a derivation frame, not a final method list. “Trainer consumes” includes
delegated pipeline orchestration whose lifecycle and result semantics Trainer
deliberately owns. It does not imply that every implementation lives on the
central `Trainer` class.

| Concern | Why the intended Trainer/pipeline consumes it | What must already be guaranteed by the accepted strategy | Result and side-effect ownership | Surface |
| --- | --- | --- | --- | --- |
| admission | Trainer must never receive an unjudged authored definition | active contract/version, complete explicit authored selections, successful establishment, one seeded internal authority | contract establishment either produces one accepted `TrainingStrategy` or rejects before Trainer construction; it does not materialize runtime objects | pre-Trainer contract boundary |
| arrangement materialization and binding | execution needs authoritative current participants and relationships without reconstructing family topology | accepted declarations, derived lifecycle/readiness/relationship obligations, permitted unbound initial states, and transition protocols | model/domain producers may construct or load values; only the internal authority accepts bindings, advances revisions, and publishes current state | core |
| runtime preparation | generic device/distributed infrastructure must prepare concrete execution participants without learning their family anatomy | authoritative source snapshot, required preparation participants, constraints, access meanings, joint-preparation rules, and route obligations | Trainer-owned infrastructure performs generic preparation; specialized component/capability behavior may participate; the authority accepts returned routes/views and invalidates stale dependents | core |
| optimization realization | Trainer must turn authored training intent into executable trainability, parameter groups, clipping/synchronization participants, optimizer, and scheduler state | accepted training-subject declarations and constraints resolved against current participant bindings and preparation state | optimization/Trainer infrastructure realizes generic trainability and optimizer state; specialized selected behavior returns typed results rather than mutating Trainer | core |
| training step | Trainer owns batch/step timing, accumulation, final optimization-loss handling, backward, advancement, triggers, and observation | current prepared routes, accepted optimization runtime, strategy-internal objective behavior, and a valid step request context | strategy computes the authored training semantics and returns a differentiable result plus declared observations/state effects; Trainer performs infrastructure actions | core |
| representation/conditioning caching | the pipeline may schedule and persist reusable data products requested for the run | a selected compatible caching capability, applicable representation/conditioning behavior, readable current bindings/views, and consistency/freshness requirements | data/cache infrastructure coordinates storage; strategy features supply semantic encoding/decoding behavior; typed results update cache/data state without becoming binding authority | recognized capability |
| validation/evaluation | the pipeline owns when evaluation occurs and how its results enter run control and observation | a selected compatible validation capability, valid evaluation routes/views, and any evaluation-state constraints | capability computes evaluation semantics and results; Trainer owns trigger timing and ordinary runtime mode transitions | recognized capability |
| sampling/generation | the pipeline owns sampling triggers, requests, output coordination, and observation | a selected compatible sampling capability plus its conditioning, predictor, representation, and persistence requirements | capability performs authored generation behavior; Trainer/pipeline owns scheduling and destination coordination | recognized capability |
| trained-artifact persistence | the pipeline owns checkpoint timing, retention, destination, publishing, and run coordinates | a selected product capability, coherent authoritative snapshot, declared product/member/representation/consistency semantics, and applicable serializers | product capability resolves semantic plan/contributions; domain serializers convert; Trainer/persistence infrastructure writes and reports the typed artifact result | recognized capability |
| runtime resume/state restoration | the pipeline owns restoration timing, run coordinates, and infrastructure state | an explicitly selected restoration contract and compatible state contributors, distinct from trained-product persistence | contributors restore their owned state through a coordinated restoration result; Trainer restores its coordinates/infrastructure and re-establishes coherent current runtime state | recognized capability |

Cross-cutting logging, metadata, resource observation, interruption, and cleanup
remain Trainer/pipeline responsibilities. They may consume accepted transition,
capability, optimization, step, and artifact results through typed observation
views, but observation does not become a second owner of live strategy state.

The table deliberately does **not** make current `tokenizers`, `vae`,
`text_encoders`, `denoiser`, `trainable_model`, `TrainingMode`, or
`ObjectiveRuntime` universal Trainer-facing inputs. Their underlying meanings
must reappear only where the relevant core exchange, recognized capability, or
strategy-internal feature actually requires them.

The first row to refine into a complete request/result/state/failure table is
arrangement materialization and binding. Runtime preparation is refined beside
it because its output must rebind the same accepted identities. The remaining
rows consume those results and therefore cannot determine the binding model
independently.

## Settled Inputs From Q1-Q5

The concrete design inherits these constraints:

1. One stable strategy-scoped logical identity denotes one independently
   addressable semantic participant, not one Python object.
2. Componenthood does not imply executability. An execution-capable
   participant normally has one authoritative `normal` execution route;
   selected capabilities may add materially distinct named routes.
3. Each participant/route pair has exactly one authoritative current binding.
   Original, unwrapped, inspection, metadata, and artifact handles are typed
   access views rather than competing execution routes.
4. Declaration, materialization, state replacement, execution-route
   rebinding, relationship transition, arrangement amendment, and merge/fold
   are distinct operations.
5. Participant lifecycle, optimization status, and operational-relationship
   lifecycle are independent axes.
6. Artifact persistence selects a coherent semantic product from accepted
   participant and relationship state; it does not serialize an incidental
   Python object.
7. One dedicated per-run binding authority owns canonical participant,
   relationship, access-view, execution-route, revision, dependency, and
   freshness state inside the complete Trainer-facing strategy boundary.
8. Transition producers submit typed proposals through one writer protocol.
   Loaders, capabilities, and Trainer infrastructure do not retain competing
   authoritative copies.
9. Backend coordination handles and replicas do not automatically become
   logical participants or execution routes.
10. Conservative behavior is valid when explicit and correct. The first
    implementation may invalidate more derived state than necessary, use only
    the normal route, and migrate current artifact products first, while still
    preserving the full contract meanings.

## Working Vocabulary: Authored Address And Runtime Identity

### Evidence checkpoint: current declaration and identity surfaces

The first concrete inspection compared the repository's existing
declaration- and identity-like values. They are useful inputs, but they do not
describe one interchangeable concept.

| Current value | Concern it serves today | Meaning to preserve | Meaning it must not acquire |
| --- | --- | --- | --- |
| [`LoadedModelComponentSpec`](../../library/models/components.py) | family-declared top-level model-component surface | stable family-local key plus declared descriptive roles/capabilities | complete training-participant declaration or universal family-shaped anatomy |
| `LoadedModelComponent` | associates one declared component surface with a live object, currently typed as `Any` | a producer must be able to expose the concrete live candidate it built | proof that the object fulfills participant semantics, an accepted authority binding, or a prepared route |
| `ModelLoadingStrategy.load_target_model()` | explicitly produces the current family component collection, including `None` for deferred modules | strategy-owned production of deliberately declared candidates | an untyped tuple becoming the atomic materialization/replacement protocol |
| SD/SDXL/SD3 `LOADED_MODEL_COMPONENT_SPECS` | known family component order and public labels | authored, deterministic declarations rather than inferred runtime topology | proof that every strategy has text encoders, one VAE, and one denoiser |
| `ModelRealizationFacts` / `RealizedModelComponentFacts` | durable metadata observation of one run realization and its declared components | qualified durable identities and presence/declaration facts for observation and provenance | live binding authority or source of runtime participant identity |
| `OptimizationTargetRef` | optimization-owned reference to one selected component/root/module/parameter-like target | component-qualified substructure, selector provenance, and live-object access for a bounded consumer | logical participant identity, canonical binding, or durable metadata identity |
| `AdapterResolvedTarget` | adapter-facing projection over optimization target resolution | adapter relationship targets and component-local paths remain distinct from adapter identity | evidence that every target-local module is an independent participant |
| `AdapterTargetSelection` | resolved training-target choice and train flags | training intent can be projected separately from component declarations | participant lifecycle, identity, or binding state |

The current family declarations contain only:

```text
key
public display name
generic roles
capabilities
```

That is enough to prove that authored stable keys are already useful, but not
enough to validate identity-preserving replacement or lifecycle fulfillment.
It also mixes presentation (`public_name`) with contract claims
(`roles`/`capabilities`). The family examples are deliberately evidence for
known default strategies rather than the universal participant vocabulary.

The metadata layer already derives a run-qualified component identifier from
one realization identifier and family-local component key. That identifier is
a durable observation/provenance coordinate. It cannot become the live
participant reference because metadata is not the binding owner, and one
training strategy may include adapters, teachers, students, reward models, or
other participants that are not identical to one source-model realization's
component inventory.

Optimization and adapter target references operate below or across participant
boundaries. They demonstrate the need for qualified substructure and typed
relationship endpoints, but their embedded live objects and selectors are
consumer-specific projections rather than declaration identity.

The first design task is therefore to separate, not merge, these concerns:

```text
authored participant declaration
presentation/diagnostic description
model source and realization observation
participant-qualified substructure/target reference
training-subject and optimization selection
```

`ParticipantDeclaration` should contain only the authored identity/addressing
meaning needed to establish the participant. Its semantic obligations are
derived by the training contract system from the participant's complete use in
the authored arrangement. Lifecycle/readiness constraints and the other
concerns must refer to the accepted participant rather than be copied wholesale
into its identity record or manually restated by the strategy author.

### Scenario pressure test: what a declaration must actually distinguish

The minimum declaration was pressure-tested against four deliberately
different participants:

| Scenario | Authored identity | Materialization pressure | Execution pressure | Identity result |
| --- | --- | --- | --- | --- |
| ordinary SDXL denoiser | `model.denoiser` | concrete predictor state is normally available before preparation | declares a normal execution route | source/module/precision/wrapper changes may preserve the participant when its authored semantics remain fulfilled |
| deferred SD3 denoiser | `model.denoiser` | starts declared and unbound, then materializes later | eventually requires the same normal route | deferred loading changes lifecycle/binding state, not declaration identity |
| injected LoRA-like adapter | `adaptation.main` | materializes independently managed adapter state | may require no independent execution route because its effect is expressed through a host relationship | adapter state remains a participant while target-local injected modules remain substructure unless independently managed |
| teacher and student from one source | `teacher.denoiser` and `student.denoiser` | both may materialize from the same checkpoint or even initially share state | each independently evolving predictor requires its own route and reference | common source and compatible predictor semantics do not collapse distinct participants |

This comparison rules out several tempting shortcuts:

- a model role such as `denoiser` cannot be the participant identity, because
  teacher and student may share that role;
- executability cannot define componenthood, because injected adapter state may
  be a participant without its own callable route;
- initial presence cannot define the declaration, because ordinary and
  deferred denoisers may be the same authored participant shape;
- source/implementation type cannot define identity, because replacements may
  change either while preserving declared meaning; and
- a semantic contract/facet cannot itself be identity, because several
  distinct participants may fulfill the same contract.

### WORKING DECISION: participant declarations identify; the contract derives
obligations

The minimum identity-bearing declaration meaning is an authored key/address:

```text
ParticipantDeclaration
  authored key/address
```

The **authored key** addresses the participant within the authored strategy.
The authority-established reference identifies its accepted run incarnation.
A key is an address, not a role classification. Human-readable segments may
contain role-like words, but those words have no identity authority:
`student.denoiser` and `teacher.denoiser` remain distinct addresses even when
both participate in the same predictor contract. The current
`LoadedModelComponentSpec` convention in which `key="denoiser"` and
`roles=("denoiser",)` coincide is useful migration evidence, not target
semantics.

Key alone does not explain the participant's complete semantic meaning. That
meaning comes from the participant's explicit uses and relationships in the
complete authored strategy: for example, which predictor behavior consumes it,
which objective and conditioning contracts interact with it, whether it is a
training subject or adapter host, and which artifact products cover it. The
training contract system evaluates those explicit choices and derives the
obligations that later materialization, preparation, execution, and persistence
must satisfy. An ordinary strategy author does not copy those obligations into
the participant declaration or supply a participant-specific validator.

The exact additional non-identity data colocated with a declaration remains
open. Presentation, source selection, implementation selection, lifecycle
state, route requirements, trainability, and artifact coverage remain distinct
meanings even if a future Python authoring API groups some of them for
convenience.

### WORKING DECISION: contract establishment derives accepted obligations

The training contract system is authoritative because it describes the
Trainer's accepted core, pipeline capabilities, feature contracts, lifecycle,
and results. It also defines how known library implementations demonstrate
conformance. Strategy authoring uses that vocabulary to make explicit choices;
contract establishment judges the complete result.

```text
active training contract system
  + explicitly authored strategy definition
  + known selected library implementations
  + bounded choices exposed by that strategy
                    |
                    v
       contract fulfillment/enforcement
                    |
          reject or establish
                    |
                    v
        complete TrainingStrategy
          |- accepted arrangement and capabilities
          |- contract-derived obligations
          `- internal contract-governed binding authority
```

Automatic enforcement does not imply automatic assembly. The contract system
may report missing or incompatible selections, but it does not choose a feature,
implementation, relationship, or dependency for the author. For maintained
library strategies, compatibility knowledge is already part of the core,
capability, feature, and implementation-conformance contracts. Additional
author-supplied validity behavior is required only when a custom implementation
or explicit contract extension deliberately leaves that known surface.

Derived obligations may concern one participant, several participants, a
relationship, a route, a capability, or the complete arrangement. They are not
forced into a participant-local taxonomy merely because one binding transition
triggers their re-evaluation.

### WORKING DECISION: one accepted contract version governs one authority

The contract/version accepted during establishment remains fixed for the
lifetime of that run authority. An explicit arrangement amendment may change
participants, relationships, routes, or other authored selections where the
accepted contract permits that evolution. The authority then derives and
re-evaluates every affected obligation under the same accepted contract before
publishing the amended state.

An arrangement amendment cannot silently replace the Trainer contract or make
an undeclared contract extension active. A different core contract, contract
version, or explicit Trainer-contract extension requires a new establishment
and a newly accepted strategy authority. Any future facility for migrating a
live run between contract versions would be a separate, explicit contract and
recovery feature rather than an ordinary binding transition.

### WORKING DECISION: runtime producers supply contract-defined evidence

Current loading proves that a producer must return a live object, but
`LoadedModelComponent.module: Any`, copied role/capability strings, and an
optional `None` do not prove that the accepted strategy obligations are met.
Metadata realization facts are durable observations after loading; optimization
and adapter target refs are consumer-local projections. None is independently
authoritative for contract acceptance.

The runtime binding exchange instead has this semantic shape:

```text
BindingTransitionProposal[Evidence]
  target ParticipantRef
  transition kind: materialize | replace
  authority-recognized proposal identity
  expected participant/dependency revisions
  concrete live candidate
  typed evidence required by the accepted contract obligations

Contract-governed transition acceptance
  accepted strategy definition and derived obligations
  exact proposal and prospective complete state
  prior accepted state when continuity matters
  current participant/relationship/route dependency revisions

BindingTransitionResult
  accepted/rejected state
  satisfied and unfulfilled obligations or structured failures
  accepted revisions and invalidations when committed
```

Evidence types may differ between known predictor, adapter, autoencoder, or
research-extension contracts, but the applicable contract or implementation
conformance boundary defines them—not each ordinary strategy author. They must
not become a universal union of model kinds or an unrestricted
`dict[str, Any]` whose undocumented keys recreate the present contract problem.
The concrete live candidate is not itself sufficient evidence, and the proposer
cannot substitute a bare Boolean compatibility claim.

For materialization, prior bound state is absent because the participant is
declared and unbound. A deferred component therefore supplies no fake `None`
candidate; it remains unbound until a concrete proposal exists. For replacement,
the proposal identifies the prior accepted binding and exact revisions so the
contract-governed authority can evaluate both current fulfillment and any
continuity requirements derived for that participant's complete use.

### WORKING DECISION: the contract-governed authority evaluates and commits

The binding authority applies the obligations established by the training
contract system to the prospective complete state. The proposer supplies only
the candidate and contract-defined evidence; it neither selects validation
behavior nor authors the ordinary acceptance rules.

```text
producer proposes candidate and required evidence
  -> authority constructs the prospective next state
  -> contract enforcement evaluates every affected accepted obligation
  -> authority verifies expected revisions
  -> authority atomically rejects or publishes the transition
```

There is no current need for a separately transferable
`CompatibilityAssessment`. Validation is part of the authority's transition
acceptance. Structured obligation outcomes may appear in the transition result
and observation history, but they do not become reusable certificates. Only the
authority may install a candidate, advance revisions, invalidate dependents,
and return the accepted result. Rejection leaves canonical state unchanged,
subject to the separately settled rule for externally permitted in-place
mutation.

The four declaration scenarios exercise the corrected boundary as follows:

| Scenario | Contract-derived obligation pressure | Runtime evidence pressure |
| --- | --- | --- |
| ordinary SDXL denoiser | selected predictor, objective, conditioning, preparation, optimization, and persistence contracts determine what the participant must support | the known SDXL integration reports the concrete realization facts required by those contracts |
| deferred SD3 denoiser | the accepted lifecycle permits declared/unbound state until the relevant readiness checkpoint | no fake candidate is filed; later materialization reports the required SD3 realization facts |
| injected adapter | the selected adapter capability derives independently managed state, target/effect, optimization, and persistence obligations without inventing an execution route | adapter construction and target resolution report contract-defined state and relationship evidence |
| teacher and student | the authored uses derive distinct participant and relationship obligations even if predictor contracts and sources match | separate proposals and revisions preserve their independent runtime trajectories |

The exact Python representation of contract establishment, derived obligations,
and typed evidence remains open. Their authority relationship is not: ordinary
validity is communicated by the training contract system, not manually composed
inside the authored strategy.

### WORKING DECISION: execution requirements are contract-derived associated
obligations

Execution requirements remain separate from participant identity, but for the
standard path the contract system derives them from explicitly selected core,
capability, and feature uses:

```text
ParticipantDeclaration
  this meaningful participant exists

AcceptedExecutionRouteRequirement
  the selected contract use requires this participant to support this
  execution meaning
```

Zero derived route requirements means the participant has no independent
execution route. The standard core may derive the normal route for an
execution-capable participant, while a selected capability may derive an
additional materially different route. A route requirement defines its
accepted callable meaning, preparation constraints, and freshness obligations;
it does not contain the prepared callable. A direct/custom strategy may fulfill
the same established route contract through an explicit conformance path; it
does not silently redefine the standard requirement.

Separating the route requirement also allows arrangement/capability evolution
to add or retire a route requirement without silently redefining participant
identity. Route identity remains the participant reference plus the route key
defined by the applicable contract use unless later evidence requires another
incarnation layer; each such pair still has one authoritative current binding
and its own route revision.

### WORKING DECISION: readiness is a separate contract-derived constraint over
declarations

Required, deferred, and phase/capability-specific readiness are not identity
properties of `ParticipantDeclaration`.

The authority initially establishes an accepted participant in the declared,
unbound state only when the contract-derived lifecycle permits it. Separate
fulfillment/readiness constraints derived during contract establishment say
when a participant must be bound, when a relationship must be resolved or
active, and when a route must be prepared for the core or a requested
capability.

```text
participant declaration
  model.denoiser exists in the authored arrangement

binding state
  declared/unbound -> bound

fulfillment constraint
  normal execution route must be fresh before training-step execution
```

This lets a deferred SD3 denoiser and an eagerly loaded denoiser share the same
participant declaration while following different valid materialization
timelines. It also prevents Trainer phase names from becoming participant
identity. The eventual readiness vocabulary should name contract checkpoints
or requested operations rather than mirror every current phase function.

Relationships remain separate declarations because relationship identity and
lifecycle are independent from both endpoint identities. Presentation labels,
source/realization facts, current binding state, trainability, optimizer
selection, and artifact products likewise remain separate projections or
accepted contract meanings referring to the participant.

### WORKING DECISION: authored relationship addresses establish run identities

The settled direction already requires participant identity and relationship
identity to remain separate. An adapter may have several independently
addressable effects, and each relationship may transition through
declared/resolved/active/inactive/detached state without creating, replacing,
or retiring either endpoint.

The concrete working split is therefore:

```text
RelationshipKey
  authored address of one semantic relationship in the strategy definition

RelationshipRef
  identity of that accepted relationship incarnation in one run authority

relationship revision
  changing endpoint-resolution and operational state of that reference
```

`RelationshipDeclaration` refers to authored participant keys and explicitly
selects the relationship meaning provided by the applicable contract
vocabulary. Contract establishment validates that use and gives it run-scoped
endpoint references. The reference is justified by independently evolving
relationship state and history, not merely by symmetry with `ParticipantRef`.
It also avoids treating endpoint pair plus relationship kind as identity when
parallel independently addressable effects are valid.

The exact Python representation need not match `ParticipantRef`. Retirement
closes a relationship reference to current transitions while preserving its
run-history and artifact-provenance meaning.

### WORKING DECISION: authored keys and authority-established references

The current recommended split is:

```text
ParticipantKey
  authored semantic declaration address
  stable across equivalent authored strategy definitions and runs

ParticipantRef
  identity of one declaration incarnation accepted by one run authority
  stable across materialization, identity-preserving replacement,
  preparation, and route rebinding

binding / relationship / route revisions
  changing state associated with that reference
```

The strategy authors the arrangement. Establishment does not allow the
authority to invent participants or relationships. The contract system
validates the authored definition, and the authority establishes its run-scoped
identities and returns typed references that later exchanges use instead of
bare strings.

A retired reference is closed to further current-state transitions and live
access. Retirement does not erase the reference's meaning from run transition
history or durable artifact provenance.

The first implementation may reject reuse of a retired key within one run
authority. If later contract evolution permits redeclaration with the same
authored key, the new declaration must receive a different reference; a
retired reference can never revive.

### WORKING DECISION: participant incarnation is authority-scoped and durably projectable

Within one logical run authority, each authority-established `ParticipantRef`
permanently identifies exactly one participant incarnation. The authority
never reassigns or revives that reference. Its durable projection must
therefore qualify the authored `ParticipantKey` by both the logical run
authority and an incarnation discriminator corresponding to that reference.
Binding, relationship, route, and other accepted state revisions describe
change to the incarnation; they are not its identity.

Materialization, wrapping or casting during preparation, compatible
identity-preserving replacement, and route rebinding preserve the reference.
Retirement closes it to further current-state transitions and live access, but
the same reference continues to identify the same historical incarnation for
history and provenance. A new run authority, redeclaration after retirement,
or an incompatible replacement establishes a new reference and therefore a
new incarnation. When the new incarnation descends from earlier state, that
connection is expressed separately as an explicit typed lineage or succession
relationship. Loading the same source does not by itself make two participants
identical; a teacher and student initialized from one checkpoint are distinct
participants.

Artifacts are also not participant incarnations. An artifact captures accepted
run state and receives its own artifact and model-revision identity, with
lineage to the participant/run state from which it was produced. Portable
catalog identity, source representation, immutable model revision, run
realization, and participant incarnation are related qualifications, not one
interchangeable identifier.

Exact runtime resume preserves the logical training-run authority, its
participant references, the accepted arrangement, and the relevant revisions
even though Python objects, processes, and wrappers are recreated. Each process
execution may have a separate execution-session identity for observation. If a
snapshot does not persist and restore the authority and participant identity
state, it cannot claim identity-preserving exact resume; using it to begin a
new run establishes new authority-scoped incarnations connected by explicit
resume/derivation lineage where applicable.

The substantial metadata work, followed by the attempt to carry it into richer
model metadata, is the immediate architectural predecessor to this direction.
Model metadata did not first become a mature layer that was later found to be
poorly integrated. While trying to proceed into it properly, its need for
durable model identity, component provenance, run realization, artifact
identity, and lineage exposed how unclear the existing model/strategy/Trainer
lifecycle was. That discovery caused the pivot into this redesign before the
model-metadata work could be completed on an unsuitable foundation.

The fact that current Trainer checkpointing does not yet restore those
identities is consequently an unfinished capability that this redesign must
enable, not a surprising disagreement with an already-integrated metadata
architecture and not a reason to treat the metadata semantics as incidental
evidence.

That causal role does not make today's metadata record shape the owner of
future runtime participant identity. In the target architecture, the binding
authority establishes participant identity and makes its durable facts
available to metadata, persistence, and restoration. Metadata records and
preserves that projection; it must not infer identity from whatever live
objects happen to be available. The current realization/component identifier
format is therefore valuable evidence of required qualification without being
the normative representation of `ParticipantRef`.

### DEFERRED CODE DESIGN: exact identity representation

The semantic split does not yet decide whether a reference is an opaque value,
an authority identifier plus declaration ordinal, a typed handle, or another
representation. The chosen form must:

- reject cross-authority use;
- remain stable across binding revisions;
- remain meaningful for history after retirement;
- avoid using Python object identity; and
- have an intentional durable projection for metadata and provenance without
  making an ephemeral Python handle itself the durable schema.

## Revision And Freshness Model

### SETTLED INPUT: fine-grained authoritative revisions

Participant binding state, relationship state, and execution routes require
their own revisions. One global counter cannot replace those facts because a
coherent view may combine contributors with different relevant revisions.

### WORKING DECISION: coarse snapshot as the conservative default

The authority may issue an atomic snapshot identifier/revision describing one
accepted current view. A derived route, access view, preparation plan, cache
plan, optimization plan, or artifact plan that declares no narrower
dependency policy defaults to the exact authority snapshot from which it was
resolved.

```text
derived projection P resolved from authority snapshot R17
  -> any accepted transition changing snapshot-visible authority state
     produces R18
  -> P is stale unless it declared a more precise valid dependency set
```

This is a conservative fallback, not the fundamental revision model. A more
precise projection may depend on exact participant-binding, relationship, and
route revisions and survive unrelated changes.

No route or derived view remains silently fresh by caller convention. The
authority can answer whether its recorded dependency guarantee still holds.

### WORKING DECISION: reads are consumer-specific projections

The authority does not expose one general component lookup or unrestricted
object snapshot. A consumer receives a projection whose meanings are defined
by the accepted exchange or capability that requested it:

- Trainer-owned infrastructure receives only the participants, routes, access
  meanings, constraints, and revisions required by the preparation or later
  optimization exchange;
- strategy-internal features and selected capabilities receive only the
  participant and relationship views authorized by their accepted contracts;
- observability receives accepted transition facts, results, and history, not
  arbitrary live object access; and
- persistence receives a coherent product-specific artifact-state projection,
  not the current outer execution objects by default.

Every projection identifies the authority state on which it depends. A
multi-participant consumer therefore either receives one coherent authority
view or an explicitly valid fine-grained dependency set; it cannot assemble a
supposedly current view from unrelated reads by convention.

### WORKING DECISION: preparation is one attempt-scoped coherent job

One preparation attempt is governed by one complete preparation job derived
atomically from coherent accepted authority state. The job identifies the
exact strategy-owned inputs, permitted access meanings, target routes,
constraints, joint groups, mutation permissions, and source dependencies for
that attempt. Preparation membership is broader than trainability: a frozen
participant may still require movement, casting, wrapping, compilation, or
another execution preparation operation.

“One job” is a semantic coordination boundary, not a requirement for one
backend call, one literal Python object, or an object discarded after one
method invocation. A job may coordinate several ordered or joint backend
operations. Every result remains tied to that attempt and its source
dependencies; it cannot be accepted as the result of another attempt or
silently reused after those dependencies become stale.

For replacement-only work, the job retains coherence optimistically: all of
its strategy-owned inputs came from one accepted state, and the authority
rechecks their dependencies before final publication. For in-place-capable
work, the authority-recognized attempt transition withdraws affected
guarantees before mutation and becomes the basis for finalizing success or
failure. The later optimization and capability exchanges will define their
own narrow inputs without changing this preparation-state rule. Immutable
value, scoped-handle, copy, and similar Python representation choices are
downstream code design.

## Binding Exchange

Binding is a family of typed transitions rather than one `bind()` call.

### Exchange overview

| Operation | Principal inputs | Accepted result | Canonical state effect |
| --- | --- | --- | --- |
| establish validated strategy | authored strategy definition, active contract/version, selected known implementations, bounded choices, and explicit extensions | complete accepted `TrainingStrategy`, derived obligations, participant/relationship references, and initial revisions | validates the authored arrangement and creates its internal run authority before Trainer receives it |
| materialize participant state | participant reference, expected revision, bound-state proposal, contract-defined realization evidence, and access-view proposals | accepted binding revision, current lifecycle/readiness state, and invalidation outcome | moves a declared participant from unbound to bound when the prospective state satisfies accepted obligations |
| replace participant state | participant reference, expected binding revision, replacement realization, supersession/lineage facts, and contract-defined evidence | same reference with a new binding revision, or rejection requiring amendment | supersedes authoritative state without silently changing contract-derived participant meaning |
| transition relationship | relationship reference, expected revision, requested operational state, resolved endpoints/effect facts | new relationship revision and invalidation outcome | changes declared/resolved/active/inactive/detached state independently from participant lifecycle |
| amend arrangement | expected arrangement revision, explicit additions/retirements, relationship and capability consequences | new references where applicable, retirement results, new arrangement revision and invalidations | changes which semantic participants belong to the current arrangement |
| rebind execution route | participant reference, route key, expected source revisions, prepared callable and freshness guarantee | new route revision and current authoritative route binding | publishes the callable representation that execution must use |
| merge/fold | source/host references, expected revisions, declared transformation and retirement policy | transformed host binding, lineage, relationship changes, optional retirement, invalidations | records semantic state transfer rather than hiding it in serialization or module mutation |

### Establish validated strategy

#### Inputs

The establishment input must include at least:

- the complete explicitly authored strategy definition;
- the active Trainer core contract and contract-version context;
- the selected pipeline capabilities, feature contracts, and known library
  implementations used by the strategy;
- participant addresses plus the explicit uses and relationships that give
  them meaning in the arrangement;
- any bounded choices that the strategy deliberately exposes to execution
  configuration; and
- any explicit custom implementation, direct-conformance, or contract-extension
  declaration outside the maintained library surface.

The author chooses and wires these meanings. The contract system derives
participant, relationship, execution-route, lifecycle/readiness, result, and
cross-concern compatibility obligations from them. The input does not include
ordinary author-composed validators or duplicated low-level requirements that
the selected contracts already define.

#### Result

Successful establishment produces the complete `TrainingStrategy` that may be
given to Trainer, including behind its boundary:

- the accepted contract/version and authored arrangement;
- the provided and requested capability surface;
- typed participant and relationship references;
- the obligations derived by the core, selected capabilities, features,
  implementation-conformance contracts, and explicit extensions;
- the accepted initial arrangement revision and first atomic snapshot;
- any currently unfulfilled readiness requirements permitted at the declared
  lifecycle stage; and
- the one contract-governed binding authority seeded with that accepted state.

Failure produces structured author-facing rejection information and no
Trainer-acceptable strategy.

#### Allowed side effects

Establishment may validate the complete authored definition, derive its
obligations, and create the one per-run authority and declared current state.
It may use known library conformance information and report what is missing or
incompatible. It must not silently select features or implementations, infer
the author's intended arrangement, materialize modules, create an optimizer,
attach adapters, or perform distributed preparation.

#### Failure conditions

At minimum:

- a missing provider for a required core concern;
- a selected capability, feature, or bounded request whose contract cannot be
  fulfilled by the authored arrangement;
- incompatible core, capability, feature, implementation, relationship, or
  result semantics;
- an unknown or non-conforming implementation presented as part of the known
  library surface;
- a custom implementation or Trainer-contract extension that is not declared
  through its explicit conformance/extension path;
- duplicate or conflicting participant keys;
- relationships referencing undeclared endpoints;
- an invalid derived lifecycle, readiness, or route obligation; and
- an attempt to seed an already established authority inconsistently.

The Trainer-facing construction path must not accept a raw authored definition
that has bypassed this establishment boundary.

### Materialization

Materialization fulfills an existing declaration. It is not arrangement
addition.

The proposal must identify its source declaration/reference and expected
authority/binding revision. It carries the concrete bound-state result and the
typed evidence required by the already accepted obligations. Evidence may
include source/realization facts, structural facts, supported access meanings,
or relationship-resolution results when their contracts require them. A
state-only participant may materialize without any execution route.

The authority evaluates the prospective complete state rather than treating
producer evidence as a portable compatibility claim. Acceptance advances the
participant binding revision and reports every derived projection invalidated
by the transition. Rejection leaves canonical authority state unchanged,
subject to the separate rule for an external operation that was explicitly
permitted to mutate an authoritative live object in place.

### Replacement and identity preservation

### WORKING DECISION: replacement preserves declared meaning, not object shape

Replacement may preserve the participant reference only when the new
realization still fulfills the same meaning and obligations derived from that
participant's complete use in the accepted strategy.

These changes do not inherently require a new participant identity:

- source checkpoint or selected realization;
- implementation class;
- live Python object;
- precision or device;
- unprepared versus wrapped execution representation; and
- another source/implementation substitution explicitly compatible with the
  declaration.

These cannot be smuggled through replacement merely because a Python value can
be assigned:

- teacher becoming student;
- encoder becoming an unrelated reward model;
- host component becoming its adapter participant; or
- another change that invalidates the participant's accepted contract
  obligations and relationship meaning.

The authority does not infer a new participant in those cases. It rejects the
replacement and requires an explicit retirement/declaration amendment.

### DEFERRED CODE DESIGN: derived-obligation and evidence representation

The concrete design must represent obligations derived from the participant's
complete uses without encoding one SD-shaped role taxonomy. Its types must
distinguish facts knowable during establishment from evidence required after
materialization or preparation, and support known library conformance,
direct/custom implementations of an existing contract, and explicit contract
extensions without making ordinary authors assemble validation behavior. The
authority, timing, and enforcement semantics are settled; exact types and code
placement follow when the accepted exchanges are mapped into code.

### Binding result shape

Every accepted binding-affecting transition must return, directly or through a
shared result envelope:

```text
operation identity
accepted/rejected status
expected and observed base revisions
new arrangement/snapshot revision when accepted
changed participant binding revisions
changed relationship revisions
changed route revisions
invalidated derived projection identities or conservative invalidation scope
accepted transition facts for observation/history
structured failure details
```

The result must distinguish canonical current-state effects from historical
transition facts. Metadata may observe the latter but does not become the live
authority.

### DEFERRED CODE DESIGN: transaction and rejection representation

The atomic acceptance and rejection behavior above is settled. Concrete code
design still must decide:

- how producers obtain permission or an expected-revision token;
- whether all transitions use one proposal/result envelope or several narrow
  operation types;
- how atomic multi-participant and participant-plus-relationship transitions
  are represented;
- how partial external failures are reported without partially committing
  canonical state; and
- whether transition facts are emitted synchronously from acceptance or
  observed through a separate typed result.

## Runtime-Preparation Exchange

Runtime preparation is Trainer-owned infrastructure acting on a narrow,
revision-pinned projection of strategy state. It may wrap, replace, cast, move,
compile, shard, or jointly prepare execution objects without learning the
strategy's family anatomy.

```text
strategy/internal authority
  exposes preparation projection at snapshot R
                 |
                 v
Trainer-owned preparation infrastructure
  prepares declared concrete items under explicit constraints
                 |
                 v
typed preparation result
  |- execution-route rebinding proposals -> binding authority
  |- optimizer/scheduler runtime          -> Trainer optimization runtime
  `- backend coordination handles         -> Trainer/runtime infrastructure
```

### Preparation job projection and plan

The preparation input must identify:

- the exact authority snapshot or fine-grained source revisions;
- each participant reference and target execution route being prepared;
- the current concrete input binding or permitted access view;
- preparation requirements and constraints, not family roles;
- joint-preparation groups and ordering/atomicity requirements;
- whether the operation returns a replacement or may mutate an authoritative
  realization in place;
- any dependencies/freshness guarantee expected of the returned route; and
- typed infrastructure items required for joint preparation without declaring
  those items to be logical participants.

The validated strategy exposes the participant routes and preparation
constraints derived from its accepted contract uses. Trainer infrastructure
chooses and executes the backend mechanics allowed by the active
performance/distributed policy. Configuration does not author new participants
or strategy features through this exchange.

### Preparation result

The result must separate:

1. prepared execution-route bindings keyed by participant reference and route;
2. new or retained typed access views, including original/unwrapped access
   when their validity can be guaranteed;
3. prepared optimizer/scheduler objects destined for Trainer-owned
   optimization runtime;
4. backend coordination or synchronization handles that remain infrastructure
   state;
5. the exact source revisions and preparation constraints satisfied;
6. freshness/dependency guarantees for each returned route/view;
7. runtime facts and limitations useful to observation; and
8. structured total or partial external failure information.

A backend composite spanning several participants does not erase their
logical identities and does not automatically become another model
participant or the normal route of any one participant.

Optimizer and scheduler entries remain backend-pressure placeholders here.
This section settles only that jointly prepared optimization objects do not
belong to the binding authority and may need to participate in one backend
operation. Their exact request/result types, construction point, and
publication owner remain subordinate to the later optimization-ownership and
optimization-exchange design.

### Rebinding guarantee

Prepared execution objects are not authoritative merely because a backend
returned them. Backend work, component-specific preparation needed to make the
candidate ready, result assembly, rank agreement, and contract/freshness
validation all occur before final publication. A stale result resolved from an
authority state that has since changed is rejected or explicitly re-resolved;
it is never silently installed against a newer arrangement.

After successful validation, one publication coordinator logically commits
all affected authority-owned route/view changes and the already-built
Trainer-owned runtime/backend state without exposing a half-published state.
The final publication boundary performs only internal installation of prepared
state. It performs no backend work, conversion, contract validation, user or
observation callback, or other operation expected to fail. The exact code
mechanism may use guarded state replacement, a generation switch, or another
equivalent representation without merging the separate ownership of routes,
optimization runtime, and backend coordination state.

### WORKING DECISION: in-place mutation freshness invariant

Some backends or component preparation operations may mutate the concrete
object they receive before returning or failing. Atomic authority updates
cannot by themselves undo arbitrary external mutation.

The required semantic invariant is:

> An operation permitted to mutate an authoritative realization in place must
> withdraw every affected freshness guarantee before mutation begins. If the
> operation fails, the authority must not silently restore those guarantees
> without establishing that the realization remains valid.

This does not yet choose a mechanism. A future API may represent the rule with
a preparation lease, a transition token, an invalid/preparing route state, or
another design. The invariant applies to affected authoritative bindings and
derived views, not only to a callable route.

For replacement-only preparation that cannot mutate the authoritative input,
failure may leave the old route valid because the uncommitted prepared result
is merely discarded.

### WORKING DECISION: mutating preparation needs a coordination identity

The mutation invariant exposes a self-invalidating sequence:

```text
preparation is resolved from snapshot R17
  -> authority withdraws affected guarantees before mutation
  -> snapshot-visible state becomes R18
  -> backend result cannot be accepted merely by asserting R17 is still current
```

Therefore an in-place-capable preparation attempt needs an authority-recognized
coordination identity distinct from the source snapshot alone. That identity
must retain:

- the exact source snapshot or fine-grained dependency revisions used to
  resolve the plan;
- the accepted authority transition that withdrew affected guarantees;
- the participant/routes/views the attempt is authorized to replace or
  re-establish; and
- the current state against which success or failure is finalized.

This is a semantic requirement, not yet a choice of mechanism or name. A
lease, attempt reference, staged transition, or another representation may
fulfill it. Replacement-only preparation can remain a simpler optimistic
result against unchanged source dependencies because it does not invalidate
its own basis before external work.

### Preparation failure classes

The exchange must distinguish at least:

| Failure class | Required authority behavior |
| --- | --- |
| plan invalid before external work | reject with no current-state change |
| source revisions stale before work | reject/re-resolve; do not call backend |
| replacement-only backend failure | preserve old authoritative bindings/routes when their guarantees still hold |
| permitted in-place mutation failure | keep affected guarantees withdrawn until validity is re-established or state is replaced |
| incomplete keyed result | reject the whole atomic rebinding set |
| backend result violates route or joint-preparation constraints | reject publication and report the incompatible result |
| authority changes concurrently before commit | reject stale result; do not partially publish Trainer runtime |
| process or rank terminates during final publication | the run cannot continue; the separate restoration contract re-establishes runtime state rather than attempting in-process partial recovery |
| post-publication observation or history routing fails | accepted current state remains committed; reporting does not become a rollback authority |

### WORKING DECISION: expected fallibility precedes final publication

The normal replacement-only path does not require a general staged-commit or
rollback protocol:

```text
coherent attempt-scoped preparation job
  -> fallible backend and component work
  -> complete unpublished candidate
  -> rank agreement and authority validation
  -> non-failing logical state installation
  -> observation and history routing
```

If replacement-only work fails before publication, its candidate is discarded
and the old authoritative routes remain current when their guarantees still
hold. Publication installs already-created, already-validated state through an
unobservable coordination boundary. Failures outside the architecture's
ordinary control, such as process or rank termination, abort the run and belong
to restoration rather than an elaborate in-process distributed transaction.

The in-place-capable path has one necessary earlier state publication: before
external mutation, the authority accepts the attempt and withdraws the
affected guarantees. That preliminary safety transition is deliberately not
the final prepared-state publication. If fallible work then fails, the affected
state remains invalid until explicitly re-established or replaced; the system
does not pretend to roll back an arbitrary external mutation. On success, the
same non-failing final installation rule applies.

All ranks participate in one logical accepted transition; rank-local replicas
and backend handles do not become independent authorities. Device/dtype
movement, wrapping, compilation, and repeated preparation use this exchange
whenever they change authority-visible routes, views, or guarantees. Jointly
prepared optimizer/scheduler values remain opaque Trainer-owned candidate
state here; their exact request, construction, and lifecycle remain for the
later optimization exchange.

### Executable pressure test of the exchange

An isolated executable spike under
`docs_design/models-strategy-trainer/executable_spike/` now exercises the
binding and preparation semantics without importing or changing production
code. Its Python names and storage shapes are provisional; the architectural
result is the behavior the scenarios forced.

The pressure test confirms the established flow:

```text
complete authored definition
  -> contract establishment and derived obligations
  -> authority-scoped participant and relationship incarnations
  -> validated materialization and identity-preserving replacement
  -> one attempt-scoped preparation job
  -> unpublished candidate work
  -> authority + Trainer publication
  -> current execution, backend, artifact, and resume projections
```

It also makes four cross-owner constraints explicit:

1. A Trainer-owned backend group is not independently current. A consumer may
   use it only while the binding authority still reports every covered
   participant as prepared. Authority route state and Trainer backend state
   are separate ownership surfaces whose agreement creates the usable view.
2. Inseparable backend coverage constrains future preparation before any
   destructive work begins. A later job may replace, expand, or merge complete
   existing groups, but it may not omit a member of any group it overlaps.
   Retiring one member evicts the whole backend group and withdraws every
   surviving member's prepared guarantee until a coherent group is rebuilt.
3. A relationship revision invalidates prepared views derived from both
   endpoints regardless of whether the revision came from an explicit
   relationship transition, endpoint replacement, or endpoint retirement.
   Relationship state cannot change while either endpoint is under destructive
   preparation.
4. Destructive-attempt ownership dominates all older optimistic work. Once an
   in-place attempt withdraws guarantees, an older replacement-only result
   cannot restore them during the attempt or after failure. Only successful
   completion of the owning attempt, destructive re-establishment, or accepted
   binding replacement can restore a usable state.

The spike further supports the existing identity decisions: preparation
wrappers and backend composites do not replace participant references; exact
resume restores logical authority and accepted revisions into new Python
objects; loading an artifact into another run creates new participant
incarnations with lineage; and a successor that participates in relationships
requires a complete arrangement amendment so obligations and relationship
incarnations are re-established together.

The spike does not establish final Python APIs, concurrency primitives,
serialization formats, or Accelerate/DeepSpeed integration. Its fake
components and conformance catalog only show that the exchange can be made
enforceable and readable. Real-backend behavior, locking, physical publication,
and the optimization and step exchanges remain later work.

## Current-Code Evolution Map

This is a pressure map, not an instruction to preserve current ownership.

| Current seam | Useful meaning to evolve | Meaning that must not survive as authority |
| --- | --- | --- |
| `LoadedModelComponentSpec` | authored family-local component declaration evidence | assumption that generic roles define universal strategy anatomy |
| `LoadedModelComponent.key` | candidate authored declaration address | tuple position or key alone as complete run incarnation identity |
| `LoadedModelComponent.module` | concrete materialization evidence | one field conflating logical binding, prepared route, and artifact access |
| `load_target_model()` result | typed materialization results for declared participants | evidence-free tuple installed directly on Trainer |
| `Trainer.loaded_components` | temporary migration projection if needed | canonical binding owner |
| Trainer `vae`/`text_encoders`/`denoiser` setters | proof that keyed rebinding is necessary | family-shaped role setters, synthesis, and caller-owned replacement |
| adapter target resolution | stable participant/path relationship evidence | whole-Trainer reads and shadow state |
| `prepare_with_accelerator(trainer)` | evidence for preparation participants, joint constraints, returned routes, optimization runtime, and backend handles | whole-Trainer mutation and mode ownership |
| DeepSpeed composite handle | backend coordination spanning prepared items | logical participant or replacement for distinct participant identities |
| `OptimizationPlan` | likely base for the later optimization exchange | strategy/mode materializing the optimizer and retaining infrastructure ownership |

## Optimization Exchange Placeholder

This section is intentionally incomplete until binding and preparation are
stable enough to constrain it.

It must eventually define:

```text
authored training subjects and constraints
participant-qualified logical parameter groups
execution parameter groups
trainability realization
clipping participants
synchronization/accumulation participants
optimizer and scheduler materialization
backend preparation constraints
standard Trainer-owned backward/step lifecycle
explicit research extension when ownership differs
```

It must not recreate `TrainingMode`, make model family a training-treatment
axis, or treat an optimizer/backend object as a strategy participant merely
because it is prepared jointly.

## Step Exchange Placeholder

This section is intentionally incomplete until optimization ownership is
settled.

It must eventually distinguish:

```text
batch and execution coordinates
strategy-produced differentiable base loss state
Trainer-derived final optimization loss
metrics
objective-specific observations
strategy/capability-owned state changes
forbidden infrastructure actions
```

Diffusion timesteps remain objective observations rather than universal core
fields. The current `BatchLossOutput.loss` and `per_sample_loss` distinction is
evidence that accounting loss and the value consumed by Trainer-owned loss
modification/backward are not automatically the same result field.

## Working Decision Register

| ID | Status | Decision |
| --- | --- | --- |
| EX-001 | working | authored `ParticipantKey` addresses are established as run-scoped `ParticipantRef` identities by the one binding authority |
| EX-002 | working | retirement closes a reference to current operations without erasing its history/provenance meaning |
| EX-003 | working | replacement preserves identity only while the new realization fulfills the same accepted contract-derived participant meaning |
| EX-004 | working | an authority snapshot is the conservative default freshness dependency; fine-grained revisions remain fundamental |
| EX-005 | working | in-place mutation requires affected freshness guarantees to be withdrawn before mutation and not silently restored after failure |
| EX-006 | working | prepared routes, optimization runtime, and backend coordination handles have separate owners even when produced by one backend call |
| EX-007 | superseded | participant declarations do not carry ordinary author-supplied semantic compatibility filings; the contract system derives obligations from the participant's complete authored uses |
| EX-008 | working | materialization/readiness requirements are separate contract-derived constraints over participant, relationship, and route state rather than participant identity fields |
| EX-009 | working | execution-route requirements are contract-derived obligations associated with participants; zero derived requirements means no independent execution route |
| EX-010 | working | authored relationship addresses establish run-scoped relationship references because operational relationship state and history evolve independently from endpoint identities |
| EX-011 | working | in-place-capable preparation requires an authority-recognized coordination identity distinct from its source snapshot so pre-mutation invalidation does not invalidate its own completion basis |
| EX-012 | working | role labels and role-like key segments never establish participant identity; identity comes from the complete authored address and its authority-established run incarnation |
| EX-013 | working | runtime transition proposals use a small common authority envelope plus typed evidence defined by the accepted contracts and implementation-conformance boundaries, not a universal participant-kind taxonomy, untyped fact dictionary, or per-strategy validator protocol |
| EX-014 | working | the binding authority evaluates prospective state against obligations established by the training contract system; a proposer cannot choose validation behavior or submit a bare compatibility claim |
| EX-015 | superseded | a separately transferable single-use `CompatibilityAssessment` is not currently required because contract validation occurs inside authority transition acceptance; structured outcomes may remain part of the transition result/history |
| EX-016 | working | only the authority installs accepted state, advances revisions, and invalidates dependents; the exact internal separation between obligation evaluation and the public transition result remains an implementation question rather than a second exchange authority |
| EX-017 | superseded | ordinary validation clauses are not composed by strategy authors; the contract system derives them from explicit known selections, while custom implementation conformance or contract extension must be deliberate |
| EX-018 | working | the binding exchange begins by contract-validating the complete authored definition, deriving obligations, and establishing the strategy's internal authority before the resulting `TrainingStrategy` may reach Trainer |
| EX-019 | working | automatic contract enforcement does not imply automatic assembly: authors explicitly choose and wire the arrangement, while the contract system supplies ordinary validity knowledge and reports incompatible or missing choices |
| EX-020 | working | one accepted contract/version governs one run authority; permitted arrangement amendments re-evaluate obligations under it, while another contract/version or Trainer-contract extension requires new establishment |
| EX-021 | working | one preparation attempt is governed by one coherent authority-derived job that may span several backend operations; membership is not limited to trainables and every result remains bound to the attempt's source dependencies |
| EX-022 | working | all ordinarily fallible preparation and validation precedes an unobservable final installation of already-built authority-owned and Trainer-owned state; only destructive in-place work publishes its required invalid/preparing transition earlier |
| EX-023 | working | within one logical run authority, each authority-established participant reference permanently identifies exactly one incarnation; accepted state revisions preserve it, retirement only closes current use, and a new reference establishes a different incarnation connected by explicit lineage when applicable |
| EX-024 | working | exact runtime resume preserves authority and participant identity only when their accepted identity and revision state is persisted and restored; otherwise continuation begins new authority-scoped identities with explicit lineage |
| EX-025 | working | Trainer-owned backend coordination state is usable only in conjunction with matching authority-owned prepared guarantees for every participant it covers; neither ownership surface alone defines current prepared state |
| EX-026 | working | inseparable backend coverage constrains preparation before destructive work: later groups may replace, expand, or merge complete existing groups but may not omit an overlapped member, and retirement of one member evicts the group and withdraws surviving members until coherent rebuild |
| EX-027 | working | every relationship revision withdraws dependent prepared endpoint views regardless of transition path, and relationship state cannot change while either endpoint is under destructive preparation |
| EX-028 | working | destructive-attempt ownership and withdrawn-guarantee state reject every older optimistic preparation result during mutation and after failure; only an accepted re-establishment or replacement restores current use |

No working entry becomes an OpenSpec requirement merely because it appears in
this table. Discussion should either accept it, refine it, or mark it
superseded while preserving the reason.

## Open Question Register

The earlier register has now been audited. Items whose architecture was already
settled were removed; concrete Python mechanisms are not retained here as open
architecture questions. Former item 5 is settled by the authority-scoped
participant-incarnation and resume decisions above. Former items 9 and 11 are
settled by the attempt-scoped preparation-job and non-failing-final-publication
decisions. No binding/preparation architecture question remains active in this
register. This does not settle the deliberately later optimization and step
exchanges; in particular, the exact optimization preparation request/result
from former item 13 remains later work under the Optimization Exchange
Placeholder.

## Current Design Frontier

Continue concrete design through the nearest dependencies only:

1. Settle optimization ownership and the optimization exchange without letting
   joint backend preparation dictate that contract.
2. Settle the step exchange from the Trainer's execution needs and the accepted
   strategy projections.
3. Derive the concrete current-to-target code mapping, module placement, and
   migration milestones from the completed architecture rather than from the
   provisional executable-spike types.

Concrete Python representation, current-to-target code mapping, module
placement, and migration milestones follow in
`strategy_system_implementation_mapping.md` after the architecture and exchange
semantics are coherent. The governing OpenSpec follows that mapping.

## Compaction Handoff

At any context reset, use `strategy_system_direction.md` as normative,
`strategy_system_inventory.md` as current-code evidence, and the latest
decision register plus current design frontier above as the continuation point.
Use `strategy_system_implementation_mapping.md` only for deliberately deferred
code-shape and migration work.

Do not restart the architecture comparison, reopen Q1-Q5, or infer final
Python class/package names from the working vocabulary.
