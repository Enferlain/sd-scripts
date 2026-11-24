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

Decision markers have these meanings:

```text
SETTLED INPUT
  inherited from the normative Q1-Q5 direction; not reopened here

WORKING DECISION
  current recommended answer for a downstream concrete question; may be
  refined before OpenSpec

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

`ParticipantDeclaration` should contain only the authored semantic obligations
needed to establish the participant and govern later identity-preserving
transitions. Lifecycle/readiness constraints and the other concerns must refer
to the accepted participant rather than be copied wholesale into its identity
record.

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

### WORKING DECISION: minimum participant declaration meanings

The current minimum declaration has two semantic parts:

```text
ParticipantDeclaration
  authored key/address
  explicitly wired semantic compatibility filing
```

The **authored key** addresses the participant within the strategy filing. The
authority-established reference identifies its accepted run incarnation.

A key is an address, not a role classification. Human-readable segments may
contain role-like words, but those words have no identity authority:
`student.denoiser` and `teacher.denoiser` are distinct addresses even when both
use the same denoiser/predictor role and semantic compatibility filing. The
current `LoadedModelComponentSpec` convention in which `key="denoiser"` and
`roles=("denoiser",)` coincide is useful migration evidence, not target
semantics.

The **semantic compatibility filing** identifies the authored, versioned rules
that candidate materializations and identity-preserving replacements must
satisfy. It is explicitly wired by the strategy or its selected features; it
is not selected through a central role enum, inferred from a Python module, or
assembled by an automatic resolver. Several participants may use the same
filing without sharing identity.

Here, **filing** uses the direction document's contract vocabulary. It does not
mean a file on disk, metadata record, global registry entry, configuration
fragment, or separately selected runtime plugin. It means the authored
strategy has explicitly supplied its answer for one contract concern:

> What must proposed bound state satisfy to count as a valid realization of
> this already-declared participant, and what must remain true for a
> replacement to preserve that participant's identity?

Conceptually, the filing combines two things whose final Python representation
may remain separate:

```text
typed semantic requirements/claims
  stable, inspectable meaning and compatibility constraints

explicitly wired validation behavior
  checks a concrete materialization or replacement and returns a typed
  fulfillment/compatibility result
```

For example, the filing for `model.denoiser` in one SDXL strategy might state
that proposed state must fulfill the predictor/input/output behavior used by
that strategy, support the access meanings needed by its selected training and
persistence behavior, and satisfy applicable structural or precision
constraints. It does not require the candidate to have one universal
`DENOISER` enum value or one Python class. A different implementation may be
accepted if the explicitly authored strategy behavior can validate and use it.

An injected adapter participant may instead file requirements for independently
managed adaptation state, target/effect compatibility, and the access meanings
needed for optimization and artifact persistence, while declaring no
independent execution route. Teacher and student participants may reuse the
same predictor compatibility filing but retain different keys and references;
the filing describes compatible meaning, not identity.

Not every semantic fact can or should be proven by generic runtime reflection.
The filing may combine centrally testable structural facts, capability- or
feature-specific checks, and authored assertions covered by strategy
conformance tests. The binding authority enforces that the explicitly wired
filing produced an applicable typed acceptance result. It does not itself learn
what a denoiser, teacher, adapter, VAE, or future research component means.

This boundary is needed because neither a label such as `denoiser`, Python
protocol/method presence alone, nor class identity can answer whether a new
realization preserves the participant semantics of this particular authored
strategy.

The filing must provide or authorize enough validation to answer:

1. whether proposed bound state fulfills this participant's declared meaning;
2. whether replacement state preserves that meaning;
3. which bound-state and access-view meanings are permitted;
4. which participant/relationship constraints require revalidation; and
5. which limitations or unfulfilled obligations must be reported rather than
   guessed by the authority.

This does not require one serializable universal schema to understand every
model. A standard repository strategy may wire reusable contract/feature
filings, while a research strategy may wire a custom filing that reaches the
same Trainer-facing boundary. The authority enforces that the applicable
filing validated a transition; it does not rediscover model semantics from
roles or object types.

The exact Python representation remains open. In particular, a semantic
compatibility filing might ultimately be a typed contract value paired with
explicitly wired validator behavior rather than one callback stored in an
otherwise passive dataclass. The semantics above should be settled before
choosing that mechanism.

### WORKING DECISION: compatibility evidence is proposal-scoped

Current loading proves that the strategy must be able to return a live object,
but `LoadedModelComponent.module: Any`, copied role/capability strings, and an
optional `None` do not prove semantic compatibility. Metadata realization facts
are durable observations after loading; optimization and adapter target refs
are consumer-local projections. None of those types supplies the missing
acceptance boundary.

The compatibility exchange should instead have this semantic shape:

```text
SemanticCompatibilityFiling[CandidateEvidence]
  authored filing identity and version
  inspectable semantic requirements/claims
  explicitly assembled validation behavior

CandidateBindingProposal[CandidateEvidence]
  target ParticipantRef
  transition kind: materialize | replace
  authority-recognized proposal identity
  expected participant/dependency revisions
  concrete live candidate
  filing-specific typed candidate evidence

CompatibilityEvaluationContext
  accepted participant declaration and current filing version
  exact proposal identity
  prior accepted binding view when replacing
  scoped current participant/relationship dependencies

CompatibilityAssessment
  filing identity and version actually applied
  exact proposal, participant, candidate, and transition assessed
  dependency revisions observed
  named requirement/clause outcomes
  fulfilled semantic and access meanings
  limitations, failures, and required revalidation
  accept | reject
```

These are semantic names, not final class names. The small common proposal
envelope carries authority coordination. `CandidateEvidence` belongs to the
explicitly wired filing and may differ between predictor, adapter, VAE, or
research participants. It must not become a universal union of model kinds or
an unrestricted `dict[str, Any]` whose undocumented keys recreate the present
contract problem.

The concrete live candidate is not itself the evidence. Producer-supplied facts
may cover meanings that are unsafe or impossible to rediscover generically;
validator observations may verify structural or behavioral facts; the scoped
authority context supplies current cross-participant facts. Source/provenance
may be included when one filing genuinely constrains it, but source identity
does not define participant identity.

For materialization, the prior binding view is absent because the participant
is declared and unbound. A deferred component therefore supplies no fake
`None` realization for validation; it remains unbound until a concrete
proposal exists. For replacement, the request includes the prior accepted
binding and exact revision so the filing can check both fulfillment of the
declaration and any continuity requirement that depends on the former
realization.

### WORKING DECISION: the authority invokes validation and owns acceptance

The proposer supplies the candidate and the filing-specific evidence, but it
does not choose the applicable validator or submit a reusable bare Boolean
compatibility claim. The binding authority resolves the already accepted
filing from the target participant declaration and invokes its explicitly
wired behavior.

```text
producer proposes candidate
  -> authority resolves the participant's accepted filing
  -> filing evaluates the exact proposal against a scoped current context
  -> authority verifies that the assessment still matches current revisions
  -> authority atomically accepts or rejects the binding transition
```

An accepting assessment is bound to the exact proposal identity, participant
reference, transition kind, filing version, candidate, and observed dependency
revisions. It is single-use evidence inside that authority transition, not a
portable certificate that another candidate or later snapshot may reuse.

The assessment is also not the materialization/replacement result. It answers
the strategy-owned semantic question. Only the authority may install the
candidate, advance binding revisions, invalidate dependents, and return the
accepted transition result. A rejection or validator failure makes no
authority state change. Candidate construction may already have external cost,
and validation may perform explicitly declared bounded probes on the
unaccepted candidate, but validation must not mutate accepted authority state
or current accepted realizations.

### WORKING DECISION: validation composition is authored, not discovered

A maintained strategy may reuse core, feature, and family-specific validation
clauses, while a research strategy may supply a custom clause or filing. The
composition point remains strategy authoring:

```text
strategy definition
  explicitly selects and composes named, versioned clauses
  files their combined requirements and validation behavior

binding authority
  invokes exactly that accepted composition
  does not search for validators or infer clauses from roles/types
```

Each filed clause has an inspectable identity, applicability/requirement
meaning, and typed outcome. Any condition is authored in the filing rather
than inferred by the authority. Every applicable required clause must be
satisfied; absence, rejection, insufficient evidence, and validator failure
remain distinguishable outcomes. The final Python representation may use a
generic protocol, typed callable/value pairing, or another explicit mechanism,
but it must preserve this authored composition and its per-clause evidence.

The four declaration scenarios now exercise the exchange as follows:

| Scenario | Candidate evidence pressure | Assessment identity pressure |
| --- | --- | --- |
| ordinary SDXL denoiser | a typed SDXL predictor realization exposes the live candidate plus the predictor/input/output, access, structure, and precision evidence required by that filing | acceptance applies only to this participant and proposal, not every object with a denoiser role |
| deferred SD3 denoiser | declaration remains unbound until a concrete SD3 candidate and its evidence are proposed | delayed timing changes neither the filing nor participant identity |
| injected adapter | adapter-managed state and target/effect evidence may be validated without inventing an independent callable route | an assessment for adapter state cannot be reused for each injected target-local module |
| teacher and student | both proposals may use the same predictor evidence type and filing | distinct participant/proposal identities produce distinct assessments even when source and semantics match |

This closes the conceptual compatibility-filing boundary while deliberately
leaving the exact Python generic/protocol shape for the binding-exchange API
design.

### WORKING DECISION: execution requirements are associated declarations

Execution requirements are separately authored declarations referring to a
participant key rather than fields that make the participant declaration grow
with every capability:

```text
ParticipantDeclaration
  this meaningful participant exists

ExecutionRouteDeclaration
  this participant must support this named execution meaning
```

Zero route declarations means the participant has no independent execution
route. The standard core may file the normal route for an execution-capable
participant, while a selected capability may file an additional materially
different route. A route declaration defines its accepted callable meaning,
preparation constraints, and freshness obligations; it does not contain the
prepared callable.

Separating the route declaration also allows arrangement/capability evolution
to add or retire a route requirement without silently redefining participant
identity. Route identity remains the participant reference plus authored route
key unless later evidence requires another incarnation layer; each such pair
still has one authoritative current binding and its own route revision.

### WORKING DECISION: readiness is a separate constraint over declarations

Required, deferred, and phase/capability-specific readiness are not identity
properties of `ParticipantDeclaration`.

The authority initially establishes the authored participant in the declared,
unbound state. Separate fulfillment/readiness constraints say when a
participant must be bound, when a relationship must be resolved or active, and
when a route must be prepared for the core or a requested capability.

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
filings referring to the accepted participant.

### WORKING DECISION: authored relationship addresses establish run identities

The settled direction already requires participant identity and relationship
identity to remain separate. An adapter may have several independently
addressable effects, and each relationship may transition through
declared/resolved/active/inactive/detached state without creating, replacing,
or retiring either endpoint.

The concrete working split is therefore:

```text
RelationshipKey
  authored address of one semantic relationship in the strategy filing

RelationshipRef
  identity of that accepted relationship incarnation in one run authority

relationship revision
  changing endpoint-resolution and operational state of that reference
```

`RelationshipDeclaration` refers to authored participant keys, states the
semantic relationship filing, and receives its run-scoped endpoint references
during establishment. The reference is justified by independently evolving
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
  stable across equivalent strategy filings and runs

ParticipantRef
  identity of one declaration incarnation accepted by one run authority
  stable across materialization, identity-preserving replacement,
  preparation, and route rebinding

binding / relationship / route revisions
  changing state associated with that reference
```

The strategy authors the arrangement. Establishment does not allow the
authority to invent participants or relationships. The authority validates
the authored filing, establishes its run-scoped identity, and returns typed
references that later exchanges use instead of bare strings.

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
| establish authored arrangement | strategy filing, participant declarations, relationship declarations, contract/version context | participant/relationship references, initial arrangement and snapshot revisions | creates the run authority's initial declared current arrangement |
| materialize participant state | participant reference, expected revision, bound-state proposal, source/realization facts, access-view proposals | accepted binding revision, current lifecycle state, invalidation outcome | moves a declared participant from unbound to bound |
| replace participant state | participant reference, expected binding revision, replacement realization, supersession/lineage facts, compatibility evidence | same reference with a new binding revision, or rejection requiring amendment | supersedes authoritative state without silently changing participant meaning |
| transition relationship | relationship reference, expected revision, requested operational state, resolved endpoints/effect facts | new relationship revision and invalidation outcome | changes declared/resolved/active/inactive/detached state independently from participant lifecycle |
| amend arrangement | expected arrangement revision, explicit additions/retirements, relationship and capability consequences | new references where applicable, retirement results, new arrangement revision and invalidations | changes which semantic participants belong to the current arrangement |
| rebind execution route | participant reference, route key, expected source revisions, prepared callable and freshness guarantee | new route revision and current authoritative route binding | publishes the callable representation that execution must use |
| merge/fold | source/host references, expected revisions, declared transformation and retirement policy | transformed host binding, lineage, relationship changes, optional retirement, invalidations | records semantic state transfer rather than hiding it in serialization or module mutation |

### Establish authored arrangement

#### Inputs

The establishment input must include at least:

- one authored strategy filing identity and contract/version context;
- participant declarations keyed by authored semantic address;
- explicitly wired semantic compatibility filings relevant to materialization
  and identity-preserving replacement;
- execution-route declarations associated with participant keys, which may be
  absent for state-only participants;
- relationship declarations with typed endpoints and intended lifecycle; and
- fulfillment/readiness constraints imposed by the core and selected
  capabilities without making those constraints participant identity.

#### Result

The authority returns:

- typed participant and relationship references;
- the accepted initial arrangement revision;
- the first atomic snapshot identity;
- any unfulfilled requirements that are valid at the declared lifecycle stage;
  and
- structured rejection information when establishment fails.

#### Allowed side effects

Establishment may create the one per-run authority and its declared current
state. It must not materialize modules, select undeclared features, infer
family topology, create an optimizer, attach adapters, or perform distributed
preparation.

#### Failure conditions

At minimum:

- duplicate or conflicting participant keys;
- relationships referencing undeclared endpoints;
- incompatible declaration obligations;
- undeclared requirements introduced by a selected capability;
- invalid lifecycle or route declarations; and
- an attempt to seed an already established authority inconsistently.

### Materialization

Materialization fulfills an existing declaration. It is not arrangement
addition.

The proposal must identify its source declaration/reference and expected
authority/binding revision. It carries the concrete bound-state result and the
facts needed to understand where that state came from. A state-only
participant may materialize without any execution route.

Acceptance advances the participant binding revision and reports every
derived projection invalidated by the transition. Rejection leaves canonical
authority state unchanged, subject to the separate rule for an external
operation that was explicitly permitted to mutate an authoritative live object
in place.

### Replacement and identity preservation

### WORKING DECISION: replacement preserves declared meaning, not object shape

Replacement may preserve the participant reference only when the new
realization still fulfills the same declared participant meaning and
obligations.

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
- another change that invalidates the participant's authored semantic
  obligations and relationship meaning.

The authority does not infer a new participant in those cases. It rejects the
replacement and requires an explicit retirement/declaration amendment.

### OPEN: participant identity obligations

The declaration needs enough structured meaning to validate
identity-preserving replacement without encoding one SD-shaped role taxonomy.
The design must determine which obligations are structural contract facts,
which are capability-specific compatibility checks, and which are assertions
made by an authored strategy implementation and verified through conformance
tests.

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

The strategy declares which participant routes require preparation and their
constraints. Trainer infrastructure chooses and executes the backend mechanics
allowed by the active performance/distributed policy. Configuration does not
author new participants or strategy features through this exchange.

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
returned them. The authority validates the result against its source snapshot
and accepts all affected route rebindings atomically. Trainer publishes the
prepared optimization runtime and backend handles only after the binding
portion is accepted, or through a broader coordination protocol that provides
the same no-half-published guarantee.

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
| EX-003 | working | replacement preserves identity only while the new realization fulfills the same declared participant meaning |
| EX-004 | working | an authority snapshot is the conservative default freshness dependency; fine-grained revisions remain fundamental |
| EX-005 | working | in-place mutation requires affected freshness guarantees to be withdrawn before mutation and not silently restored after failure |
| EX-006 | working | prepared routes, optimization runtime, and backend coordination handles have separate owners even when produced by one backend call |
| EX-007 | working | a participant declaration minimally combines an authored key and explicitly wired semantic compatibility filing |
| EX-008 | working | materialization/readiness requirements are separate fulfillment constraints over participant, relationship, and route state rather than participant identity fields |
| EX-009 | working | execution-route requirements are separate authored declarations associated with participants; zero declarations means no independent execution route |
| EX-010 | working | authored relationship addresses establish run-scoped relationship references because operational relationship state and history evolve independently from endpoint identities |
| EX-011 | working | in-place-capable preparation requires an authority-recognized coordination identity distinct from its source snapshot so pre-mutation invalidation does not invalidate its own completion basis |
| EX-012 | working | role labels and role-like key segments never establish participant identity; identity comes from the complete authored address and its authority-established run incarnation |
| EX-013 | working | compatibility validation uses a small common transition envelope plus filing-specific typed candidate evidence rather than a universal participant-kind taxonomy or untyped fact dictionary |
| EX-014 | working | the binding authority invokes the semantic compatibility filing already accepted for the target participant; a proposer cannot choose the validator or submit a bare compatibility claim |
| EX-015 | working | a compatibility assessment is single-use evidence scoped to the exact proposal, participant reference, transition kind, candidate, filing version, and dependency revisions |
| EX-016 | working | compatibility assessment and authority transition result remain distinct; only the authority installs accepted state, advances revisions, and invalidates dependents |
| EX-017 | working | reusable and custom validation clauses are composed explicitly during strategy authoring and retain named typed outcomes; the authority performs no validator discovery or role/type inference |

No working entry becomes an OpenSpec requirement merely because it appears in
this table. Discussion should either accept it, refine it, or mark it
superseded while preserving the reason.

## Open Question Register

1. What exact Python generic/protocol representation preserves the recorded
   proposal-scoped candidate evidence, explicit authored clause composition,
   and non-transferable assessment semantics?
2. Who constructs the authority, and at which strategy filing/validation
   boundary is the authored arrangement established?
3. What is the durable projection of `ParticipantRef`, distinct from its live
   typed runtime use?
4. Should retired authored keys be permanently reserved within one authority,
   or may a later amendment reuse a key while necessarily receiving a new
   reference?
5. What exact bound-state and access-view types replace the single
   `LoadedModelComponent.module` field?
6. What proposal/result envelope supports atomic multi-participant and
   participant-plus-relationship transitions?
7. What scoped read projections may Trainer, strategy features, capabilities,
   observability, and persistence request?
8. What exact attempt/transition protocol coordinates in-place preparation
   after affected guarantees are withdrawn, while replacement-only preparation
   retains a simpler optimistic path?
9. What publication/recovery rule prevents authority routes and Trainer-owned
   runtime from diverging after preparation?
10. How do ranks agree on one accepted transition while replicas remain
    backend views rather than independent authorities?
11. What exact optimization request/result joins backend preparation without
    letting this exchange decide optimizer construction or ownership early?
12. How may a participant's filed semantic rules evolve during one run: only
    through arrangement amendment/new incarnation, or through an explicitly
    versioned revalidation transition?

## Immediate Discussion Order

Continue concrete design in this order:

1. confirm or refine EX-001 through EX-003 and EX-007 through EX-017; the
   conceptual semantic-compatibility boundary is now drafted;
2. define authority construction and initial arrangement establishment;
3. complete the materialization/replacement proposal and authority-result
   types around the proposal-scoped compatibility assessment;
4. define scoped snapshots/access views and default freshness application;
5. define the preparation projection/plan;
6. define backend result acceptance, mutation failure, and publication
   coordination;
7. pressure-test the complete binding/preparation exchange against the six
   target-first scenarios plus current fine-tune and adapter paths;
8. proceed to optimization ownership and the optimization exchange; and
9. define the step exchange and only then draft the governing OpenSpec.

## Compaction Handoff

At any context reset, resume from:

- the settled inputs in `strategy_system_direction.md`;
- the latest working/accepted entries in the decision register above;
- the unresolved questions in the open-question register; and
- the immediate discussion order.

Do not restart the architecture comparison, reopen Q1-Q5, or infer final
Python class/package names from the working vocabulary.
