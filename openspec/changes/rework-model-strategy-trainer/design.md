## Context

This is the governing design record for the model–strategy–Trainer rework. See
[proposal.md](proposal.md) for motivation and scope.

Production implementation is not ready merely because this OpenSpec has all
standard artifact files. The settled architecture is recorded here now; the
remaining Trainer-consumption, execution, optimization, capability, and
migration decisions are completed through the numbered design gates below.
Production tasks are added only after those gates have evidence, scenarios,
and acceptance tests.

### Source authority and carry-forward rule

The pre-OpenSpec records have different jobs. They are not interchangeable:

| Record | Authority in this change |
| --- | --- |
| [Strategy system direction](../../../docs_design/models-strategy-trainer/strategy_system_direction.md) | Normative input for the settled architecture and Q1–Q5 decisions. This design must not silently weaken or reopen it. |
| [Contract exchange design](../../../docs_design/models-strategy-trainer/strategy_contract_exchange_design.md) | Detailed derivation of binding, preparation, identity, optimization, and the execution frontier. `SETTLED INPUT` is inherited; working decisions are adopted here only where this design or a delta spec states them. |
| [Production inventory](../../../docs_design/models-strategy-trainer/strategy_system_inventory.md) | Evidence about active code and pressure scenarios, not authority over the target. |
| [Implementation mapping](../../../docs_design/models-strategy-trainer/strategy_system_implementation_mapping.md) | Source-backed current-to-target responsibility map and design dependency order. |
| [Chronological notes](../../../docs_design/models-strategy-trainer/notes.md) | History. Later dated corrections supersede earlier positions. In particular, the accepted-run-arrangement correction supersedes the active-strategy topology. |
| [Framework comparison](../../../docs_design/models-strategy-trainer/framework_pattern_comparison.md) | Subordinate prior art. It does not select this repository's ownership model. |
| [Executable spike](../../../docs_design/models-strategy-trainer/executable_spike/) | Behavioral evidence for binding/preparation failure cases. Its class names and storage shapes are not a production prototype. |
| [Original discussion](../../../docs_design/models-strategy-trainer/strategy_discussion.md) and [external discussion](../../../docs_design/models-strategy-trainer/strategy_discussion_external.md) | Raw historical context. External comments are opinions; neither transcript is a requirement source. |

After this change exists, new normative decisions belong in this design and
its delta specs. The supporting records remain available for reasoning and
provenance rather than becoming a second living specification.

### Current production baseline

Current code still follows the pre-redesign topology:

```text
RunConfig
  -> configuration-only validation
  -> family TrainingStrategy
  -> TrainingMode
  -> Trainer, which separately builds objective state
       -> strategy loads components into Trainer fields
       -> mode selects or constructs trainables
       -> mode builds optimizer and mutates prepared Trainer state
       -> loop calls strategy.process_batch(... Trainer projections ...)
       -> loop performs loss modification, backward, advancement and triggers
```

The current graph and exact source confirm this baseline in
[`train.py`](../../../train.py),
[`trainer.py`](../../../library/training/runners/trainer.py),
[`model_prep.py`](../../../library/training/phases/model_prep.py),
[`optimizer.py`](../../../library/training/phases/optimizer.py), and
[`training_loop.py`](../../../library/training/phases/training_loop.py).
The relevant generation is `2026-09-19T19:13:58Z`; all code paths cited in
this design had matching source metadata and no recorded indexing gap. That is
a best-effort coverage signal, not proof that the graph is complete.

Useful typed islands already exist:

- family-declared `LoadedModelComponent` values;
- logical and execution groups in `OptimizationPlan`;
- typed optimizer and objective runtime values; and
- a shared loop that already owns ordinary accumulation, backward, clipping,
  advancement, triggers, interruption, observation, and cleanup.

The migration evolves these meanings. It does not wrap the current mutable
Trainer hub in a new all-purpose runtime object.

### Previously tried draft

An earlier September draft is retained only as
`openspec/changes/mst_pre_astra.zip`. It correctly recognized the governing
change and implementation gate, but left the full carry-forward audit and
Trainer exchanges as later work while already defining broad requirements. It
also did not resolve main-spec requirements that still make `AdapterMode` the
permanent owner. This active change is rebuilt from the source hierarchy above;
the archive is comparison evidence, not an artifact dependency.

### Active metadata-change overlap and sequencing

This change owns the target live run authority and Trainer topology. The active
`expand-model-metadata-identity-and-structure` change owns the broader catalog,
source, provenance, realization-composition, and structural metadata concerns.
Runtime-facing metadata work is designed on top of this rework, or together
with the relevant rework milestone, rather than fixing an older Trainer/state
boundary that this architecture would immediately replace. The metadata
requirements remain important inputs while this design chooses those
boundaries.

The changes already overlap on `loaded-model-components`,
`model-family-metadata`, and `optimization-target-refs`. Those overlapping
deltas may not be implemented, synced, or archived independently as though the
other change did not exist.

Two concrete conflicts must be reconciled rather than inherited silently. The
metadata change currently requires Trainer to retain the typed
loading/provenance state as its primary model representation, while this change
places that evidence in the accepted arrangement or a scoped projection and
forbids a competing Trainer-owned binding collection. Its target-ref delta also
uses optimization as the universal target-building actor and preserves an
unqualified live-parameter field, while this change permits several accepted
consumers and treats live objects only as revision-pinned projections. The
metadata requirements for typed evidence, structural identity, and
observation-local deduplication survive; their runtime owner and access shape
must be revised at the shared boundary.

The retained boundary is:

- the metadata change's typed loading result and its source/materialization
  evidence survive;
- the family-declared loaded-component surface is authoritative evidence of
  what loading produced and remains the highest generic model-structure
  surface;
- the accepted run authority, not that loading result or Trainer fields, owns
  current participant bindings, routes, views, revisions, and freshness; and
- metadata consumes durable projections of accepted runtime state rather than
  establishing live identity or binding authority.

When work reaches an overlapping boundary, this rework SHALL choose the best
runtime ownership and exchange with the broader metadata system in mind. The
metadata change SHALL then consume that boundary or be revised alongside it,
without losing its catalog, provenance, structural, or durable-history
requirements. Non-overlapping metadata foundation work may continue when it
does not encode a conflicting runtime authority. There is no blanket rule that
one entire change must land first; concrete implementation, sync, and archive
ordering is decided from the actual milestone dependencies once the shared
boundary is known.

OpenSpec requires a MODIFIED requirement to retain the current main-spec
scenario identifiers. The two `optimization-target-refs` scenario headings
that still say “fine-tune mode” and “adapter mode” are retained only for that
delta identity; their replacement bodies assign no ownership to either mode.
They may receive a cosmetic rename after the governing semantic change is in
the main specification.

## Goals / Non-Goals

**Goals:**

- Produce one readable architecture from contract-guided authoring through
  accepted execution, without an ordinary per-step strategy/Trainer dialogue.
- Make the intended Trainer's needs and retained authority explicit before
  choosing authoring classes or an execution representation.
- Preserve model-, objective-, adapter-, and capability-specific behavior in
  accepted implementations instead of moving that knowledge into the generic
  Trainer.
- Give current run state one writer, stable semantic identity, explicit
  transitions, coherent preparation, and typed consumer projections.
- Preserve an easy-to-follow standard path while making the research path
  strong enough to replace structured computation or request greater execution
  authority through an explicit contract extension.
- Keep every implementation milestone traceable from Trainer responsibility to
  requirement, exchange, scenario, code change, and test.

**Non-Goals:**

- Building an automatic feature resolver, strategy synthesizer, or Hydra-based
  strategy authoring language.
- Choosing graph, tree, region, schedule, lowered-Python, or hybrid execution
  representation before the representative cases establish its required
  meanings.
- Preserving `TrainingMode`, `process_batch()`, the current aggregate strategy
  ABC, or family-shaped Trainer fields merely for migration convenience.
- Making arbitrary model/component assembly safe in the first migration.
- Adopting, vendoring, or recreating Lightning, Fabric, or another complete
  framework as part of this change.
- Making metadata, observability, a backend wrapper, or a checkpoint file the
  owner of live participant identity.
- Renaming provisional spike vocabulary into production without deriving the
  actual consumers and exchanges first.

## Decisions

### D1. Intended Trainer and pipeline capabilities shape the contract

The architectural order is:

```text
repository goals + current-code evidence
                    ↓ inform
 intended Trainer + supported pipeline capabilities
                    ↓ shape
          training contract system
                    ↓ guides
          explicitly authored strategy
                    ↓ strategy fulfillment
           accepted run arrangement
       meaning + run-specific obligations + authority
                    ↓ governed realization
       materializes, binds, prepares, and verifies
                    ↓ publishes
         executable run + Trainer engine
```

The training contract definition exists before a strategy is authored. It
describes the vocabulary, supported choices, constraints, lifecycle,
observable results, and execution/ownership profiles supported by the intended
Trainer and pipeline. The broader training contract system applies that
definition during strategy fulfillment, governed realization, and later run
transitions. That causal relationship does not make either the definition or
the system code-owned by a Trainer instance or require it to live in the
central Trainer class.

Strategy fulfillment evaluates one completed authored filing under that
existing definition. The settled boundary is pass or fail with useful details:
success establishes one accepted semantic arrangement, one authoritative set
of requirements for realizing that run, and its run authority; failure
produces no Trainer-acceptable arrangement. The exact rule representation and
division between contract data, shared conformance knowledge, fulfillment
logic, and other mechanisms remain design work. Fulfillment does not select or
invent a contract after seeing arbitrary strategy code.

The accepted arrangement is not a claim that every concrete object is already
loaded, prepared, or executable. Governed realization evaluates concrete
evidence against the same run-specific requirements and publishes executable
state only after the readiness requirements for that use are fulfilled.

Current behavior can expose a missing desired capability and cause a deliberate
Trainer/contract change. It does not become core merely because SDXL or another
current strategy performs it.

Alternatives rejected:

- extracting the future core from the current `TrainingStrategy` method list;
- treating current config validation as complete arrangement acceptance; and
- allowing a strategy to redefine ordinary validation rules for known library
  behavior.

### D2. Authoring is explicit; fulfillment is contract-guided; assembly is not automatic

A repository author selects and wires participants, relationships, objective
behavior, features, capabilities, bounded configuration choices, and any
explicit custom or extension behavior. That authored behavior may include
stateful algorithms, schedules, reactions to observations, bounded decisions
made from live inputs, declared effects, and permitted authority-governed
transitions; it is not limited to static component selection. Strategy
fulfillment checks the completed definition against the applicable contract
meanings, reports missing or incompatible choices, and specializes those
general meanings into the authoritative obligations for one run. It does not
discover features from runtime types, choose dependencies, or silently repair
a filing.

This decision fixes the responsibility and boundary, not the code shape. It
does not yet require executable validators to live inside the contract
definition, the strategy definition, or any particular module. It requires the
normal strategy path to pass fulfillment before governed realization and to
return detailed acceptance or rejection information. Fulfillment and later
runtime conformance are lifecycle responsibilities of one training contract
system, not permission for independent validators to reinterpret the strategy.

Every obligation is evaluated at the earliest point where its authoritative
evidence exists. A fact already available during fulfillment is not deferred
to loading or preparation. A fact that inherently depends on a concrete
artifact, live object, resolved relationship, optimizer realization, or
backend result remains an explicit unfulfilled obligation until that evidence
is available. Dependent use cannot become ready in the meantime.

Configuration may request use and settings of a capability that the authored
strategy already provides. It does not choose an undeclared implementation or
assemble the strategy.

The ordinary first implementation remains explicit Python composition. A
decorator may later be useful for a custom operation or descriptor only if it
carries real contract meaning and does not create import-order-dependent
discovery.

### D3. The contract has three consumer-defined surfaces

```text
core contract
  meanings every compatible Trainer run needs

Trainer-recognized capabilities
  named requests/results/lifecycle operations coordinated by the pipeline

strategy feature contracts
  reusable authoring behavior consumed while constructing an arrangement
```

The consumer distinguishes capability from feature. “Trainer-recognized” does
not require code to live in the central Trainer class; delegated pipeline
services may coordinate it. Features and capabilities may be optional
selections for a particular strategy even when they are available in the
repository. Once selected, however, their applicable requirements are not
optional. The core may also require one compatible selection from a category
without requiring every available implementation.

The first recognized capability set is caching, validation/evaluation,
sampling/generation, trained-artifact persistence, and runtime restoration.
Their implementations may use shared accepted operations, but their request,
readiness, results, side effects, and failure semantics remain explicit.

Repository availability, strategy provision, runtime request, and current
readiness are different facts. A strategy may provide a capability that a
particular run never requests. Provision alone does not schedule it. Once it is
requested, it becomes usable only when its accepted dependencies and current
state satisfy the applicable readiness checkpoint.

### D4. The authored strategy is a recipe, not the ordinary runtime peer

The authored strategy is the explicit definition of what is trained and how.
Successful strategy fulfillment preserves that meaning in the accepted
arrangement:

```text
accepted run arrangement
  accepted contract version and execution/ownership profile
  selected implementations, configuration, and dependencies
  participants, relationships, obligations, and capabilities
  structured maintained/custom execution and dynamic decision policies
  owned runtime-state definitions, accepted initialization sources or rules,
    transitions, and observation inputs
  permitted authority-governed transitions
  explicitly authorized imperative regions
  one run binding authority
```

The authoring role is then complete and its broad construction interface is no
longer an ordinary runtime dependency. Ordinary training, validation,
sampling, persistence, and restoration do not call back into it to recover
choices already established during fulfillment. The accepted arrangement does,
however, retain the selected executable implementations, dynamic policies,
runtime-state contributors, and transition permissions needed to carry out the
authored meaning. A concrete Python object may implement both an authoring seam
and an accepted runtime seam, but execution uses only the latter and its
accepted authority. This does not move capability implementations into
Trainer.

An imperative runtime role is distinct from the authored strategy role even if
a future implementation happens to use one Python object for both. Its access
and authority come from the accepted execution/ownership profile, not from
receiving the whole Trainer.

Alternatives rejected:

- Trainer ↔ active-strategy callbacks as the standard topology;
- one broad `PreparedTrainingProgram` lifecycle façade; and
- a passive recipe whose missing behavior is reconstructed with family
  branches inside Trainer.

### D5. Trainer consumption is designed before authoring APIs

The target code is derived backward from what one Trainer engine must consume.
The following frame is normative at the responsibility level and deliberately
does not prescribe methods or one fixed call sequence. Its rows are lifecycle
uses of one authoritative obligation model, not separate sources of validity:

| Responsibility | Accepted input/readiness | Result and effect owner |
| --- | --- | --- |
| Fulfill an authored strategy | Complete authored selections under the active contract/profile | Strategy fulfillment returns detailed rejection or one semantically accepted arrangement with its authoritative run-specific obligations and seeded authority. No raw authored definition reaches Trainer. |
| Materialize and bind | Accepted declarations, relationships, lifecycle permissions, run-specific obligations, expected revisions | Domain producers create candidates and evidence; the run authority evaluates them against the exact obligation revision and alone publishes current state. |
| Prepare runtime | Coherent revision-pinned participants, routes/views, obligations, constraints, joint groups, and mutation permissions | Trainer infrastructure performs generic backend work as part of governed realization; complete candidate results are verified before the authority and Trainer publish route/view, backend, and optimization state together. |
| Realize optimization | Accepted training subjects, grouping/policy meaning, current bindings and preparation constraints | Trainer optimization infrastructure realizes trainability, parameters, optimizer/scheduler state, clipping and synchronization views. |
| Execute training | Accepted execution structure, current prepared routes, current optimization runtime, changing batch/coordinates, and declared authority | Trainer coordinates retained mechanics; accepted operations or imperative regions perform only their accepted computation and effects. |
| Coordinate caching | Selected caching capability, compatible representation/conditioning behavior, coherent readable state | Data/cache infrastructure owns traversal/storage; selected codecs own semantic encoding and return typed results. |
| Coordinate validation | Selected validation capability, evaluation-ready routes/views and state constraints | Trainer owns trigger/traversal/mode timing; accepted behavior owns evaluation semantics and results. |
| Coordinate sampling | Selected sampling capability and its prepared conditioning/predictor/representation needs | Trainer/pipeline owns trigger and destination; accepted generation behavior produces typed results. |
| Persist trained artifacts | Declared product, coherent authority projection, consistency boundary, serializers | Product capability resolves semantic intent; serializers render it; persistence infrastructure writes and reports actual results. |
| Restore runtime | Restoration contract, identity/revision state, Trainer/backend/optimization plus accepted operation and capability contributors | Trainer coordinates restoration and re-establishes coherent current state; contributors restore only their owned state. |

Logging, metadata, resource observation, interruption, and cleanup are
cross-cutting Trainer/pipeline responsibilities. They consume typed facts and
results and never become a second live authority.

Semantic acceptance and executable readiness are deliberately different. The
former says that the authored meaning is complete and permitted and establishes
the exact remaining obligations. The latter says that the evidence required at
the relevant readiness checkpoint has been accepted and the resulting state
has been published. Known incompatibility is rejected at the former boundary;
contingent artifact, backend, process, or resource failure may still prevent
the latter without causing downstream code to reinterpret the strategy.

The next design pass expands each row into request, result, readiness, effects,
and failure behavior before production types are chosen.

### D6. One run authority owns participant and relationship state

The Q1–Q5 conclusions are carried forward as one coherent rule set.

#### Identity

- An authored participant key is an address within the recipe. It is not a
  role, runtime identity, or claim that two runs contain the same participant.
- Within one logical run authority, one authority-established participant
  reference permanently identifies exactly one participant incarnation.
- Materialization, compatible replacement, preparation, wrapping, casting,
  compilation, and route rebinding preserve that reference.
- Retirement closes current use but preserves the same historical meaning.
  A retired reference is never revived or reassigned.
- A new authority, redeclaration, or incompatible replacement creates a new
  reference. Descent is recorded separately as lineage or succession.
- Teacher and student remain distinct even when initialized from one source.
  Backend replicas and composite handles do not become participants.

#### Routes and views

- Componenthood does not imply executability.
- An execution-capable participant normally has one authoritative `normal`
  route. A selected capability may require another named route only for a
  materially different callable/preparation representation.
- Each participant/route pair has exactly one current binding.
- Original, unwrapped, inspection, metadata, and artifact access are typed
  views rather than competing execution routes.
- Independently evolving state requires another participant. A derived route
  may remain under one participant only with explicit refresh or
  synchronization semantics.

#### Transitions and state axes

These operations remain distinct: declaration, materialization, bound-state
replacement, route rebinding, relationship transition, arrangement amendment,
and merge/fold. Participant lifecycle, optimization selection/trainability,
and relationship lifecycle are separate axes.

Ordinary authored adapters are declared participants before materialization.
Constructing their state is not arrangement addition; attachment changes a
relationship; an injected submodule is not automatically another participant;
merge/fold is a semantic transition with lineage and optional retirement.

#### Authority and freshness

One contract-governed authority owns canonical participant, relationship,
binding, route, view, revision, dependency, and freshness state. Producers
submit typed proposals through one writer protocol. The authority evaluates the
prospective complete state and atomically publishes or rejects it.

Participant binding, relationship, and route state have independent revisions.
An atomic authority snapshot is the conservative default dependency for a
projection that declares nothing narrower. A later implementation may retain a
projection across unrelated transitions only through an explicit valid
dependency set. No stale route or view remains current by caller convention.

Consumers receive scoped, coherent projections. Trainer infrastructure sees
only preparation/optimization meanings; accepted behavior sees only its
authorized state; persistence sees product-specific state; metadata and
observability see facts/history rather than unrestricted live objects.

### D7. One revisioned obligation set governs initial realization and later transitions

The accepted arrangement records the participant, relationship, route,
readiness, result, and cross-concern obligations established when the complete
authored strategy passes fulfillment. This is the authoritative interpretation
of the general contract for that arrangement revision. Ordinary authors do not
restate the normal validity rules for known implementations, and downstream
stages do not independently derive compatibility rules from the authored
strategy. The exact representation and placement of conformance logic remain
open; initial realization and later transitions consume the same authoritative
obligations.

During initial realization or a later transition, a materialization or
replacement proposal supplies:

```text
target accepted identity
transition kind and proposal/attempt identity
expected participant/dependency revisions
concrete candidate
typed evidence required by the accepted contract
```

Evidence is defined by the applicable contract and known/custom conformance
boundary. It is not an unrestricted fact dictionary, a universal model-kind
union, a role lookup, or a Boolean supplied by the producer. The authority
constructs the prospective state, applies the accepted obligations, verifies
revisions, and publishes or rejects the whole transition.

The obligation set is authoritative and revisioned rather than independently
mutable or permanently frozen. A permitted arrangement amendment derives and
re-evaluates the affected obligations under the same contract version and
execution/ownership profile. Every proposal and derived projection identifies
the exact authority and obligation revisions on which it depends. Evidence is
checked at its earliest authoritative availability; initial realization and
continued enforcement differ in lifecycle and continuity constraints, not in
their source of contract meaning.

Deferred state is represented as declared-but-unbound under an accepted
readiness rule, not as a fake `None` candidate. Replacement preserves identity
only while the candidate continues to fulfill the same accepted participant
meaning. Otherwise the authority requires retirement and declaration of a new
incarnation.

Concrete envelopes, generics, and class names are production code design. The
authority and enforcement direction are settled.

### D8. Preparation is one coherent job with two failure paths

Preparation is part of governed realization, not work performed after a run has
already been declared concretely verified. One preparation attempt is derived
atomically from coherent accepted state and its exact obligation revision.
It contains every participant required by the requested preparation, including
frozen participants that still need movement, casting, wrapping, sharding, or
compilation. One job may perform multiple ordered or joint backend calls.

The result separates:

1. route rebinding proposals owned by the run authority;
2. typed access views owned by the run authority;
3. optimizer/scheduler runtime owned by Trainer optimization;
4. backend coordination state owned by Trainer/runtime infrastructure; and
5. source revisions, constraints, guarantees, facts, and structured failures.

One coordinator must derive the coherent preparation job and coordinate final
installation across the authority-owned and Trainer-owned surfaces. Its exact
owner and API are not selected here; G1 must settle them explicitly before
production types are chosen. Neither surface may publish its portion
independently and then ask the other to catch up.

All ordinarily fallible work—backend operations, component-specific
preparation, candidate assembly, rank agreement, evidence collection,
obligation evaluation, and freshness validation—happens before final
publication. The complete result is checked against the same run-specific
obligation revision from which the attempt was derived. Final publication is
an unobservable installation of already-built, already-validated state. It
does not call expected-fallible external work or observation callbacks.

Replacement-only work is optimistic: on failure, discard the unpublished
candidate and keep old guarantees that remain valid. Destructive in-place work
has a necessary earlier safety transition: before mutation, an
authority-recognized attempt withdraws every affected guarantee. Failure leaves
those guarantees withdrawn until accepted replacement or re-establishment;
arbitrary external mutation is not “rolled back” by restoring flags.

An inseparable backend group remains Trainer coordination state, not a
participant. A later job may replace, expand, or merge complete overlapping
groups but may not omit an existing overlapped member. Retiring one member
evicts the group and withdraws the other members' prepared guarantees until a
coherent rebuild.

Relationship-dependent prepared views are invalidated for every relevant
relationship revision path. Relationship state cannot change while an endpoint
is under destructive preparation. A destructive attempt also outranks older
optimistic candidates; those candidates cannot restore its withdrawn state.

### D9. Optimization has semantic identity before concrete runtime

The standard profile contains one or more Trainer-owned optimization units. A
unit is one independently advanced optimization responsibility, not a model,
participant, logical group, parameter object, optimizer object, or backend
wrapper.

The design keeps five meanings separate:

```text
unit address
unit incarnation within one run
accepted-definition revision
concrete runtime realization
mutable optimizer/scheduler state and progress
```

Semantic parameter membership refers to authority-qualified participant
substructure, not current `nn.Parameter` identity. Changes to accepted
membership, optimizer-significant grouping, optimizer/scheduler policy,
advancement policy, or semantic dependencies advance the unit revision. A new
wrapper, recreated parameters, optimizer objects, backend replacement, or
process does not. Split, merge, and retire/recreate make new unit identities.

Standard resolution assigns each selected semantic parameter to one unit and
one execution group, including tied aliases. Deliberate overlap requires a
recognized capability or explicit extension whose contract defines ordering,
cadence, clipping, gradient, zeroing, state restoration, and backend support.
A flag or ordinary strategy mutation is insufficient.

The accepted strategy declares training-subject intent and constraints.
Trainer optimization resolves live parameters, logical and execution groups,
trainability, clipping, synchronization/accumulation, optimizer/scheduler
policy, and runtime. Module train/eval timing remains a pipeline lifecycle
concern; optimizer-local transitions such as schedule-free modes are invoked at
Trainer-owned lifecycle points.

Optimization is semantically downstream of accepted bindings but may
physically cross preparation. Only these ordering points are fixed:

```text
accepted semantic plan
  -> one Trainer-owned realization/preparation job in backend-required order
  -> complete published bindings + optimization runtime
```

The current `OptimizationPlan` logical/execution split is an evolutionary
foundation. Raw parameter lists, train flags, a primary-trainable field, and
later mode queries cannot remain separate authorities.

Detailed standard advancement policies, the backend-flexible
candidate/request/result exchange, and its connection to accepted execution
are design gate G3. EDM2, schedule-free, fused optimizers, and auxiliary learned
state are pressure cases, not automatic core units or ownership decisions.

### D10. Accepted execution is structured and authority-bounded

The normal accepted arrangement must preserve:

```text
operations or regions and their dependencies
selected implementations and static configuration
required participants/routes and readiness
inputs and outputs
declared effects and state transitions
observations and failures
retained Trainer authority
any explicitly granted imperative authority
```

This requirement does not select the concrete representation. Graph, tree,
regions, schedule, lowered Python, and hybrid forms remain candidates to test
against maintained SD/SDXL/SD3 paths and strong research cases.

Static choices, implementation selection, wiring, and permission checks are
resolved before the hot path. Ordinary step execution receives only changing
inputs and coordinates plus current prepared and operation-owned state. This
rule removes repeated discovery and authorization; it does not require the
accepted behavior itself to be static. Accepted execution may branch on live
inputs, advance schedules, use randomness, update adaptive state from
observations, make choices within accepted bounds, and request declared
effects or transitions. Trainer must not reconstruct model-family anatomy, and
the accepted behavior must not repeatedly ask the authored strategy for
information it already supplied.

The design distinguishes four kinds of change:

1. ordinary dynamic execution changes results or operation-owned state within
   already accepted behavior;
2. a bounded runtime choice selects among alternatives whose meaning and
   authority were accepted during fulfillment;
3. an authority-governed transition changes bindings, relationships,
   preparation, optimization, or another part of current run state and must
   follow its accepted transition and republication rules; and
4. behavior that requires a different contract version or ownership profile
   requires new strategy fulfillment rather than being smuggled through a
   runtime branch.

No candidate execution representation may satisfy the design only by reducing
an authored plan to a frozen linear sequence. It must preserve the stateful,
adaptive, lifecycle-driven, and explicitly imperative cases admitted by the
selected contract/profile.

The current `BatchLossOutput` is evidence, not the universal result. In the
active loop, `per_sample_loss` feeds Trainer-owned loss modification/backward
while `loss` is used for accounting; diffusion timesteps are objective
observations. The future exchange must distinguish computation outputs,
optimization inputs, observations, declared effects, and accounting without
assuming one loss producer, one differentiable tensor, one advancement
sequence, or diffusion-specific fields.

The standard profile keeps time, accumulation, synchronization, backward,
clipping, advancement, zeroing, triggers, observation, interruption, and
cleanup under Trainer ownership. A custom structured operation can replace
standard decomposition while satisfying that boundary. A different owner for
one of those mechanics requires an explicit extension/profile; the extension
states its actual executor and exchanges rather than assuming “the strategy
takes over.”

The concrete representation and step exchange are design gate G2/G3, not an
implementation detail to invent while coding.

### D11. Artifact persistence is semantic; restoration is separate

Trained-artifact persistence uses four distinct stages:

```text
capability declaration
  -> persistence request
  -> resolved artifact plan
  -> artifact result
```

The plan selects a coherent product-specific projection by participant and
relationship identity. It describes coverage, dependencies, transformations,
expected members, representation, packaging, and consistency. The result
describes what was actually emitted, including resources, formats, sizes,
checksums, references, omissions, partial results, and failures.

Product, semantic member, and physical resource remain separate. One member
may be sharded and one resource may contain several participants. “Bundle” is
packaging, not proof of completeness.

Trainer/runtime infrastructure establishes the persistence boundary and
coordinates ranks. The product capability declares semantic coverage and
transformations. Domain serializers perform mechanical conversion and writing.
A serializer does not infer a merge, delta, pruning, or semantic quantization
from incidental runtime objects.

Runtime restoration has a different purpose. Exact same-run restoration
preserves the logical authority, participant and optimization identities,
accepted arrangement, relevant revisions, mutable state, and run coordinates
only when the snapshot actually carries and restores them. New Python objects,
processes, or wrappers do not break identity; a new execution session may have
its own observation identity. If those facts are not restored—or an artifact
starts another run—the new run establishes new identities and records lineage
instead of claiming continuity.

Trainer/pipeline infrastructure coordinates restoration of the coherent run.
Accepted operations, capabilities, optimization, and backend integrations
contribute and restore only the continuation state they own. A trained-product
request and a runtime-snapshot request have independent results: either may
succeed when the other fails, and neither result may imply the other's
completeness.

Adapter persistence follows the same separation. Publishing adapter weights is
an artifact-product operation. Loading a prior adapter artifact into a new run
is materialization or initialization of a new participant incarnation and
records lineage to the artifact and its source state. Exact same-run restoration
instead restores the accepted adapter participant and every required owned
continuation state from a runtime snapshot. A shared serializer or weight format
does not collapse these into one save/load lifecycle.

### D12. TrainingMode dissolves; current specs must be reconciled explicitly

`TrainingMode` is not renamed or wrapped. Its responsibilities split by
meaning:

| Current mode responsibility | Target owner |
| --- | --- |
| Select direct, PEFT, or combined training treatment and semantic subjects | Authored strategy under the training contract |
| Adopt maintained PEFT support and declare the currently supported method set | Reusable strategy-authoring feature under the applicable feature contracts |
| Select and configure one supported PEFT method for a run | Explicit strategy construction before fulfillment |
| Declare semantic adapter target intent | Authored strategy through the maintained PEFT integration |
| Resolve that intent against the current authority-qualified host structure | Governed PEFT realization using shared policy-neutral target references |
| Construct/load method state against resolved targets and attach it to hosts | Method-specific adapter realization plus authority-governed participant/relationship transitions |
| Toggle trainability and resolve parameters/groups | Trainer optimization realization |
| Precision and distributed preparation | Trainer/runtime infrastructure plus explicit component constraints |
| Clip/sync participants and coordinate train/eval timing | Trainer optimization and lifecycle coordination over published projections |
| Specialized method behavior | Accepted execution operation, or a narrow capability only when the pipeline requests a named operation |
| Runtime checkpoint contributions | Restoration contributor under Trainer coordination |
| Artifact semantics and serialization | Product capability plus domain serializer |
| Diagnostics | Observability over accepted state/results |

The maintained PEFT integration defines the behavior shared by repository
adapter methods: authored adapter participants and host relationships, target
resolution inputs, realization and attachment, training-subject contribution,
preparation constraints, shared lifecycle participation, artifact products,
and exact-restoration contributions. A particular method supplies only its
method-specific implementation, settings, state representation, constraints,
and genuinely unique operations.

Targeting therefore has two distinct meanings. The strategy authors semantic
intent: which declared hosts or substructures the selected PEFT treatment is
meant to affect. Governed PEFT realization resolves that intent against a
coherent, authority-qualified projection of current host structure and gives the
selected method the resulting scoped targets. After the method realizes its
state, Trainer optimization consumes returned trainable parameter references
and owns trainability, grouping, and advancement. The shared target-reference
model may support both exchanges, but its code location or reuse does not
transfer semantic targeting policy to optimization.

A strategy adopts that integration deliberately and declares the methods it
supports at that point in its evolution. Strategy construction may select one
of those methods for a run; fulfillment rejects methods or targets outside the
declared set. Adding a repository method later does not silently expand an
existing strategy. This reusable authoring feature must not become a renamed
`AdapterMode` object with unrestricted Trainer access.

`FineTuneMode` has no semantic replacement. Directly training selected
substructures of existing participants is authored training-subject intent
realized by ordinary Trainer optimization and preparation. Emitting a full
model is a separate product choice. The target architecture therefore does not
make direct training, PEFT training, or their contract-compatible combination
mutually exclusive. A temporary user-facing `finetune` or `adapter` preset may
choose a maintained strategy composition, but no mode value or mode object
crosses the accepted-arrangement boundary into Trainer.

This change therefore modifies, rather than merely warns about, existing
`adapter-system`, `adapter-module-targeting`, `loaded-model-components`,
`optimization-target-refs`, `model-family-metadata`,
`training-observability`, `repo-owned-lora-method`, and
`repo-owned-vera-method` requirements that encode or invoke the old ownership.

Objective choice and mathematics likewise move into authored strategy meaning.
Trainer may route step coordinates or accepted observations to adaptive
objective state, but DDPM/RF choice is not a separate peer runtime axis and
objective-specific timestep fields are not universal Trainer inputs.

### D13. Conservative implementation is allowed; weakened contracts are not

An early slice may use only the normal execution route, support current product
types first, or invalidate all derived projections after a binding-affecting
transition. These are explicit conservative policies over the final semantics.

An early slice may not discard participant identity, independent revisions,
freshness, named-route representability, semantic artifact plans/results, or
the accepted-arrangement boundary and ask callers to reconstruct them later.
No milestone may maintain two canonical binding stores, preserve a renamed
mode authority, or introduce an active-strategy callback contract that the next
milestone immediately removes.

## Representative Case Matrix

No single maintained family defines the core. The design gates use the
following cases together:

| Case | Required pressure |
| --- | --- |
| Ordinary SDXL fine-tune | Multi-component materialization, selected base parameters, DDPM/RF variants, prepared routes, full-model product |
| SDXL adapter training | Separate adapter participant/state, host relationship, target provenance, attachment, adapter artifact, specialized lifecycle result |
| Joint base and PEFT training | Existing-participant and adapter subjects coexist without a mode axis, with explicit non-overlap, preparation, lifecycle, restoration, and product meanings |
| SD | Different component count and unsupported full-model product must fail before execution when selected |
| SD3 | Three encoders, typed conditioning, rectified flow, component-specific preparation constraints, and declared/deferred materialization |
| Teacher/student distillation | Shared source does not collapse identities; asymmetric roles, relationships, routes, trajectories, optimization participation, and products remain explicit |
| Pixel or non-latent representation | No fake VAE or latent-shaped universal field; representation-specific behavior remains selected implementation |
| Video, audio, or other tensor shapes | No universal image-batch or fixed-axis assumption; temporal, channel, sequence, and representation meanings remain selected behavior |
| Added side network or LLM-containing model | Additional executable/trainable participant without new universal Trainer slots |
| Scheduled and observation-adaptive behavior | Runtime schedules, feedback-driven state, bounded decisions, observations, persistence, and restoration remain dynamic after authoring ends |
| Custom structured operation | Replaces maintained decomposition while keeping standard Trainer authority |
| Strong imperative research region | Requests actual backward, gradient, or advancement authority through an explicit supported profile; acceptance and rejection are testable |
| Replacement/preparation failure | Stale optimistic candidate, destructive failure, relationship invalidation, backend-group rebuild |
| Artifact and exact restoration | Product identity and lineage remain distinct from same-run authority/participant restoration |

## Design Gates Before Production Implementation

### G1. Complete Trainer consumption meanings

For each D5 row, establish the common Trainer-consumption frame: the categories
of request input, result, readiness evidence, canonical state effect, permitted
external effect, and failure that its eventual exchange must make explicit.
Trace each responsibility to one or more delta requirements without promoting
current family slots. Name the owner that derives each governed-realization job
and the coordinator of final publication across authority-owned and
Trainer-owned state; do not leave cross-owner atomicity in passive voice.

G1 locates ownership, evidence checkpoints, and the common exchange questions.
It does not complete the domain-specific caching, validation, sampling,
persistence, or restoration protocols. G4 fills in those capability-specific
requests, results, readiness rules, effects, and failures after execution and
optimization semantics are settled.

### G2. Derive accepted execution from representative cases

Inspect SDXL fine-tune and adapter first, then the materially different SD and
SD3 paths and the strong custom/imperative cases. Select the minimum operation
or region meanings, dependency/effect vocabulary, and execution/ownership
profile rules. Do not choose the representation by analogy alone.

### G3. Complete optimization and step exchanges

Define supported standard advancement policies; the explicit extension
boundary; backend-flexible optimization candidates/requests/results; and the
connection among accepted execution, final optimization input, observations,
state effects, and advancement. Preserve the settled unit identity,
non-overlap, preparation, and publication rules.

### G4. Complete capability exchanges and overlap reconciliation

Finish caching, validation, sampling, artifact persistence, and restoration
request/result/readiness/failure semantics. Reconcile every changed main spec
and the active metadata change, distinguishing current migration compatibility
from target ownership.

### G5. Choose production shapes and traced migration milestones

Only after G1–G4, choose concrete authority/binding/preparation types, accepted
arrangement/executable-run representation, authoring/strategy-check boundary,
modules,
and names. Every production milestone must identify the old authority it
removes, exact code paths, requirements/scenarios, tests, performance checks,
and review boundary.

The gate is satisfied only after strict OpenSpec validation, user review of the
architecture, required review tooling, and a complete trace:

```text
Trainer responsibility
  -> normative requirement
    -> accepted-arrangement exchange
      -> representative scenario
        -> migration milestone
          -> acceptance test
```

## Risks / Trade-offs

- **The OpenSpec looks implementation-ready before the design is complete** →
  Keep G1–G5 explicit; tasks distinguish design milestones from production
  milestones, and no apply work begins from artifact status alone.
- **General vocabulary makes code hard to read** → Derive types from recurring
  consumers, keep ordinary SD/SDXL examples beside abstractions, and refine
  names after the base shape is executable rather than enforcing naming rules
  prematurely.
- **SDXL anatomy becomes the universal contract** → Require SD, SD3, compound,
  non-latent, and imperative cases before selecting the execution shape.
- **Extensibility becomes unchecked whole-Trainer access** → Imperative regions
  declare requested authority and target an accepted ownership profile; other
  actions remain forbidden.
- **Extensibility becomes too weak for research** → The strong case must control
  a genuinely Trainer-owned mechanic under an explicit extension, not merely
  supply a custom forward function.
- **Hot-loop abstraction hurts performance** → Resolve static composition and
  validation before execution; benchmark representative standard and extension
  paths during the first production milestone.
- **Old and new authorities coexist** → Migrate coherent vertical run paths and
  remove displaced writes/projections in the same milestone.
- **Metadata identity and runtime identity collapse** → Runtime authority owns
  participant incarnations; metadata stores intentional durable projections;
  artifacts and catalog/model revisions retain their own identities.
- **Current main specs contradict the target** → Carry explicit delta specs for
  every known conflict rather than relying on prose precedence.
- **Backend behavior forces accidental semantic ownership** → Backend order and
  composites constrain realization but never define participant or unit
  identity.

## Migration Plan

This section defines migration rules, not final code milestones. G5 replaces
the outline with exact reviewed slices.

Old and new launch paths may coexist temporarily for different runs during a
bounded migration, but one run uses exactly one authority topology. A run
accepted into the new arrangement must not fall through to active-strategy
callbacks, `TrainingMode` ownership, or old Trainer binding state. Until a full
vertical cutover is ready, the old path remains independently selected rather
than serving as the hidden executor behind a partially installed new contract.

1. Complete and review G1–G4 inside this change. No production code changes.
2. Choose production types and an accepted-arrangement boundary from the
   completed consumers rather than copying the isolated spike.
3. Establish contract-guided authoring and strategy fulfillment for one maintained SDXL
   path while preserving existing execution until the first full vertical
   cutover is ready.
4. Cut over one coherent SDXL run path so the accepted arrangement and its
   authority become the only current-state source; remove the displaced mode,
   strategy callback, and Trainer projection writes for that path in the same
   milestone.
5. Add adapter behavior through the same accepted participant/relationship,
   optimization, persistence, and restoration boundaries; do not create a
   second adapter orchestration authority.
6. Migrate SD and SD3 with conformance scenarios proving that the core did not
   inherit SDXL cardinality or objective assumptions.
7. Complete capability and research-extension migration, remove compatibility
   paths whose consumers are gone, and archive the change only after all main
   specs and acceptance tests agree.

Rollback is performed at coherent milestone boundaries. Runtime recovery from
a failed preparation attempt follows D8; source-code rollback does not pretend
to restore an externally mutated live object.

## Deferred Production-Shape Choices

The following implementation-shape choices are intentionally deferred because
their answers do not change the settled architecture or current delta
requirements. They are answered during G5 after the exchanges expose recurring
shapes:

- final Python class, protocol, and module names;
- opaque/reference value representation and durable string encoding;
- whether final installation uses immutable state replacement, a generation
  switch, or another equivalent internal mechanism; and
- whether custom-operation authoring uses ordinary composition, a descriptor,
  a decorator, or a narrow combination.
