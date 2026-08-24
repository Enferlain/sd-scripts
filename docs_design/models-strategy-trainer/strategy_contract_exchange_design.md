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

### OPEN: exact identity representation

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

### OPEN: snapshot contents and read isolation

The concrete snapshot design must decide:

- whether snapshots are immutable values, scoped read handles, or both;
- how live Python objects are exposed without making the snapshot an
  unrestricted object dictionary;
- how an atomic multi-participant read is retained through preparation or
  persistence; and
- when a snapshot may be held versus when consumers must copy or resolve a
  narrower projection.

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

### OPEN: derived-obligation and evidence representation

The design still must determine how the contract system represents obligations
derived from the participant's complete uses without encoding one SD-shaped
role taxonomy. It must distinguish facts knowable during establishment from
evidence required after materialization or preparation, and must support known
library conformance, direct/custom implementations of an existing contract,
and explicit contract extensions without making ordinary authors assemble
validation behavior.

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

### OPEN: transaction and rejection mechanics

The design still must decide:

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

### Preparation projection and plan

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
returned them. The authority validates the prospective result against its
source snapshot and the contract-derived preparation, route, relationship, and
freshness obligations, then accepts all affected route rebindings atomically.
Trainer publishes the prepared optimization runtime and backend handles only
after the binding portion is accepted, or through a broader coordination
protocol that provides the same no-half-published guarantee.

A stale result resolved from an authority state that has since changed is
rejected or explicitly re-resolved. It is never silently installed against a
newer arrangement.

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
| Trainer publication fails after authority acceptance | requires an explicit coordination/recovery rule; still OPEN |

### OPEN: preparation coordination protocol

The design still must settle:

- whether the authority issues a preparation attempt/lease before external
  work;
- how joint model/optimizer/scheduler preparation is keyed without collapsing
  their ownership;
- whether route acceptance and Trainer runtime publication need a staged
  commit protocol;
- how rank agreement is established while retaining one logical run authority;
- how ordinary device/dtype movement differs from execution-route preparation;
- which original/unwrapped access views can be produced generically versus by
  backend-specific adapters; and
- how repeated preparation, compilation, or re-preparation is represented.

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

No working entry becomes an OpenSpec requirement merely because it appears in
this table. Discussion should either accept it, refine it, or mark it
superseded while preserving the reason.

## Open Question Register

1. What exact Python authoring and establishment boundary lets the contract
   system judge the complete authored definition and return the same public
   `TrainingStrategy` abstraction in an accepted state rather than exposing a
   second Trainer-facing wrapper?
2. How does the contract system identify known library implementations and
   their conformance without automatic assembly, `hasattr()` discovery,
   family-name branches, or a stringly typed global registry?
3. What typed representation lets the contract derive participant,
   relationship, route, readiness, and cross-concern obligations, and what
   evidence boundaries distinguish establishment-time facts from facts only a
   materializer or preparation backend can report?
4. Who constructs the authority at contract establishment, and how is it
   prevented from accepting state for a raw or differently versioned authored
   definition?
5. What is the durable projection of `ParticipantRef`, distinct from its live
   typed runtime use?
6. Should retired authored keys be permanently reserved within one authority,
   or may a later amendment reuse a key while necessarily receiving a new
   reference?
7. What exact bound-state and access-view types replace the single
   `LoadedModelComponent.module` field?
8. What proposal/result envelope supports atomic multi-participant and
   participant-plus-relationship transitions?
9. What scoped read projections may Trainer, strategy features, capabilities,
   observability, and persistence request?
10. What exact attempt/transition protocol coordinates in-place preparation
   after affected guarantees are withdrawn, while replacement-only preparation
   retains a simpler optimistic path?
11. What publication/recovery rule prevents authority routes and Trainer-owned
   runtime from diverging after preparation?
12. How do ranks agree on one accepted transition while replicas remain
    backend views rather than independent authorities?
13. What exact optimization request/result joins backend preparation without
    letting this exchange decide optimizer construction or ownership early?
14. How may accepted contract-derived obligations evolve during one run: only
    through arrangement amendment/new incarnation under the same contract, or
    through an explicitly versioned contract-extension/re-establishment
    transition?

## Immediate Discussion Order

Continue concrete design in this order:

1. define the authored-strategy input and contract-establishment result that
   allow only an accepted `TrainingStrategy` to reach Trainer;
2. define how known selections produce derived obligations, how standard
   library conformance is known, and where explicit custom conformance or
   contract extension begins;
3. complete materialization/replacement proposal and authority-result types
   around contract-defined evidence and prospective-state enforcement;
4. define scoped snapshots/access views and default freshness application;
5. refine the preparation projection/plan so its requirements come from the
   accepted contract obligations;
6. define backend result acceptance, mutation failure, and publication
   coordination;
7. pressure-test the complete binding/preparation exchange against the six
   target-first scenarios plus current fine-tune and adapter paths;
8. proceed to optimization ownership and the optimization exchange; and
9. define the step exchange and only then draft the governing OpenSpec.

## Compaction Handoff

At any context reset, resume from:

- the settled inputs in `strategy_system_direction.md`;
- the contract-authority correction recorded by superseded EX-007, EX-015, and
  EX-017 plus replacement decisions EX-018 and EX-019;
- the latest working/accepted entries in the decision register above;
- the unresolved questions in the open-question register; and
- the immediate discussion order.

Do not restart the architecture comparison, reopen Q1-Q5, or infer final
Python class/package names from the working vocabulary.
