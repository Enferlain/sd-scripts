## Context

This is the governing design record for the model–strategy–Trainer rework. See
[proposal.md](proposal.md) for motivation and scope.

Production implementation is not ready merely because this OpenSpec has all
standard artifact files. The settled architecture is recorded here now; the
remaining Trainer-consumption, execution, optimization, capability, and
migration decisions are completed through the numbered design gates below.
Production tasks are added only after those gates have evidence, scenarios,
and acceptance tests.

A bounded, non-integrated executable experiment may inform G2/G3's choice of
execution representation. It is design evidence, not a production migration
milestone or an accepted public API: the active Trainer and launcher must not
consume it, and G5 decides which parts, if any, become production interfaces.
This exception does not relax the production-readiness gate.

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

#### Settled-source trace into this change

This crosswalk records where the settled direction and the exchange register
became governing decisions and requirements. The EX register is working
history: its `working` entries are carried only to the extent stated by the
linked decisions and delta requirements. Superseded entries are recorded to
prevent their older wording from returning as a requirement.

| Settled source or register disposition | Design decision | Delta requirement home |
| --- | --- | --- |
| Direction: central principle, three contract surfaces, lifecycle, and model/strategy/Trainer boundaries | D1–D5 | [`training-contract`](specs/training-contract/spec.md): active contract, explicit authoring, fulfillment, consumer-defined surfaces; [`training-capability-coordination`](specs/training-capability-coordination/spec.md): named capability exchanges; [`accepted-training-execution`](specs/accepted-training-execution/spec.md): standard Trainer mechanics |
| Q1 participant identity; EX-001, EX-002, EX-003, EX-012, EX-023 | D6 | [`run-participant-state`](specs/run-participant-state/spec.md): authored address versus incarnation, identity-preserving changes, retirement, lineage |
| Q2 execution-binding cardinality; EX-008, EX-009 | D6–D7 | [`run-participant-state`](specs/run-participant-state/spec.md): routes versus views and independent lifecycles; [`training-contract`](specs/training-contract/spec.md): derived readiness checkpoints; [`accepted-training-execution`](specs/accepted-training-execution/spec.md): acceptance versus readiness |
| Q3 arrangement transitions; EX-010, EX-013, EX-014, EX-016 | D6–D7 | [`run-participant-state`](specs/run-participant-state/spec.md): relationship identity, revision-checked atomic transitions, one authority; [`training-contract`](specs/training-contract/spec.md): earliest authoritative evidence and shared conformance. The precise proposal/evidence types remain G5 code shape. |
| Q4 artifact-state projection; direction's settled artifact-persistence stages | D11 | [`training-artifact-persistence`](specs/training-artifact-persistence/spec.md): declaration/request/plan/result, semantic projection, product/member/resource, consistency, actual output |
| Q5 binding ownership; EX-004 | D6 | [`run-participant-state`](specs/run-participant-state/spec.md): one authority, conservative snapshot freshness, scoped coherent projections |
| EX-005, EX-011, EX-028: destructive attempt and older optimistic results | D8 | [`training-runtime-preparation`](specs/training-runtime-preparation/spec.md): destructive withdrawal, attempt continuity, stale-result rejection |
| EX-006, EX-025, EX-026, EX-027: distinct owners, joint usability, backend groups, relationship dependencies | D8 | [`training-runtime-preparation`](specs/training-runtime-preparation/spec.md): result ownership, matching prepared guarantees, group coverage, relationship invalidation |
| EX-007 and EX-017 **superseded**: ordinary author-supplied compatibility clauses or validators | D2, D7 | [`training-contract`](specs/training-contract/spec.md): known behavior uses shared conformance knowledge; custom behavior uses an explicit conformance or extension path |
| EX-015 **superseded**: a separate transferable `CompatibilityAssessment` | D7 | [`run-participant-state`](specs/run-participant-state/spec.md): transition acceptance and structured rejection remain with the authority; no separate assessment authority is required |
| EX-018 **superseded** active-strategy topology; EX-029 corrected accepted-arrangement topology | D4, D7 | [`training-contract`](specs/training-contract/spec.md): fulfillment establishes the arrangement; [`accepted-training-execution`](specs/accepted-training-execution/spec.md): execution survives the authoring object |
| EX-019, EX-020: explicit assembly and one accepted contract/version per authority | D1–D2, D7 | [`training-contract`](specs/training-contract/spec.md): authors assemble explicitly, one accepted contract governs the run; [`run-participant-state`](specs/run-participant-state/spec.md): permitted amendments update obligations atomically |
| EX-021, EX-022: one attempt-scoped preparation job and non-failing final installation | D8 | [`training-runtime-preparation`](specs/training-runtime-preparation/spec.md): coherent job, fallible work before publication, replacement-only and destructive failure paths |
| EX-024: exact same-run restoration versus a new run | D6, D11 | [`run-participant-state`](specs/run-participant-state/spec.md): authority identity and lineage; [`training-artifact-persistence`](specs/training-artifact-persistence/spec.md): runtime restoration distinct from trained artifacts |
| EX-030, EX-031: structured and imperative execution; representation form remains open | D10 | [`accepted-training-execution`](specs/accepted-training-execution/spec.md): explicit structure and effects, custom structured operations, authority-bounded imperative regions |
| Direction: optimization ownership; exchange optimization-unit identity and standard non-overlap decisions | D9 | [`training-optimization`](specs/training-optimization/spec.md): semantic unit identity, authority-qualified membership, standard non-overlap, Trainer-owned realization, profile-defined authority |
| Direction: `TrainingMode` dissolution and recognized capabilities | D3, D12 | [`training-capability-coordination`](specs/training-capability-coordination/spec.md): coordination without central implementation; [`adapter-system`](specs/adapter-system/spec.md): pipeline coordination replaces mode ownership |

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
The training-first sweep was refreshed against graph generation
`2026-09-22T05:52:18Z`; all code paths cited in
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

#### Training-first source disposition

The current `library/training/` package mixes several different owners. The
target keeps training as the package for the training process, but it does not
move these files wholesale or make the central Trainer class the owner of
everything that moves under training coordination.

| Current source | Target responsibility |
| --- | --- |
| `training/runners/trainer.py` | A smaller Trainer engine retains lifecycle order, time, failure handling, and cleanup. Participant/binding state, optimization runtime, capability state, inputs, and observations move into their explicit run-state sections or owners. |
| `training/phases/training_loop.py` and `triggers.py` | Retain ordinary accumulation, synchronization, backward, clipping, advancement, step/epoch accounting, and lifecycle triggers. Broad strategy and mode callbacks are replaced by accepted execution and prepared optimization inputs. |
| `training/phases/caching.py`, `validation.py`, `model_prep.py`, and `optimizer.py` | Split into phase timing, capability coordination, governed materialization/preparation, and optimization realization. They stop receiving the whole mutable Trainer as their implicit contract. |
| `training/phases/orchestration_helpers.py` and `trainer_utils.py` | Dissolve into the concerns they currently mix: phase observation, evaluation projections, backend preparation, optimization mechanics, validation RNG state, restoration progress, and diagnostics. |
| `training/modes/` | Dissolves rather than becoming another runtime axis. Authored training-subject and PEFT choices belong to strategy authoring; generic mechanics belong to training; specialized behavior remains an accepted domain operation or capability contribution. |
| `training/checkpointing.py` | Splits trained-artifact persistence from exact runtime restoration. Training owns consistency, I/O coordination, retention, and reported outcomes; selected serializers and restoration contributors retain their domain-owned behavior. |
| `training/sample_generation.py` | Splits Trainer/pipeline-owned trigger, traversal, destination, state projection, and result handling from selected family generation and pipeline behavior. |
| `training/diffusion.py` and `noise_utils.py` | Leave generic training ownership. Latent preparation belongs to selected representation behavior; DDPM noise regularization belongs with the objective behavior that consumes it. |
| Concrete strategy validation, sampling, checkpointing, and preparation facets | Generic outer workflows move to training-owned coordinators. Model-, objective-, serializer-, and family-specific computation remains selected accepted behavior and is not copied into Trainer branches. |
| `strategies/base/context.py` | Common phase, coordinate, route, and changing-input facts become part of the accepted execution invocation rather than being published by an active authoring strategy. Domain consumers retain only their accepted use of those facts. |

The sweep also exposes two state concerns that the provisional training shape
must not leave hidden in Trainer fields:

- input state, including manifests, prepared input sources, dataloaders,
  deterministic cursor/RNG facts, and the current epoch input; and
- observation/lifecycle state, including metric accumulators, validation
  history, runtime traces, reporting-session status, and cleanup progress.

The code that creates or advances those values may remain in phases,
capability coordinators, and the existing data or logging packages. Their
current per-run values belong to the coherent run-state aggregate rather than
requiring new top-level orchestration authorities.

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

#### Main-spec reconciliation ledger

The following records each affected main spec's disposition. This change's
`MODIFIED` or `REMOVED` delta is the target rule; unchanged requirements remain
in force. Mode-named scenario headings retained for OpenSpec matching are not
target runtime actors.

| Main spec | Contradiction or inherited constraint | Delta and unchanged dependency |
| --- | --- | --- |
| `loaded-model-components` | Trainer is named as the primary loaded-component holder, and loading names the current strategy method. | `MODIFIED` loading and top-level-surface requirements keep family-owned order, keys, labels, and generic semantics, but make the typed loading result candidate/evidence and accepted authority the current binding owner. Non-slot-based expansion survives. |
| `optimization-target-refs` | Its namespace and mode scenarios imply optimization/modes build all targets, and live objects are embedded in unqualified refs. | `MODIFIED` policy-neutral refs and consumer scenarios separate semantic identity from revision-pinned live projections. Trainer optimization resolves trainable parameters; governed PEFT realization resolves authored host-target intent. Unchanged selector compatibility, component labels/keys, grouping behavior, and declared-component provenance survive. |
| `adapter-system` | Optimization-owned targeting, two `AdapterMode` ownership requirements, and adapter-path persistence assign durable authority to old actors. | Four requirements are `REMOVED`; realization, method configuration, and adapter-facing optimization boundaries are `MODIFIED`. Added requirements assign target intent to authoring, resolution to governed PEFT realization, timing to Trainer/pipeline, and product/loading/restoration to separate exchanges. Broad applicability, typed helpers, and parameter-native grouping survive. |
| `adapter-module-targeting` | Optimization owns host-module selection; `AdapterMode` appears in target passing, trainable handoff, and `loha` export/merge. Mixed-method overlap was deferred. | Ownership and deferral are `REMOVED`; governed resolution and unsupported-overlap rejection are `ADDED`. Provenance, parameter grouping, `loha`, method settings, continuation, migration fields, and declared-component scope are `MODIFIED`. Method behavior, selector compatibility, strict artifact-continuation default, and non-SD component support survive. |
| `repo-owned-lora-method` and `repo-owned-vera-method` | Build scenarios name `AdapterMode` and call resolved targets optimization-owned. | Each build/loading requirement is `MODIFIED` to consume governed PEFT results. Method-local algorithm, configuration, state, export, and trainable-provenance requirements survive; methods still do not own host traversal or semantic selection. VeRA's malformed delta-shaped main-spec headings are normalized before validation. |
| `training-observability` | Producer and startup/resource scenarios allow mode-owned filtered state and mode/strategy authority. | Three requirements are `MODIFIED`: accepted projections/results supply facts, Trainer or accepted behavior choose when to emit, and observability owns formatting/routing without becoming authority. Sinks, ordering, resource separation, trackers, and provenance diagnostics survive. |
| `model-family-metadata` | Existing realization identity can be misread as live participant identity. | An `ADDED` requirement distinguishes authority-issued incarnation and exact-run restoration from catalog, source, realization, artifact, and cross-run lineage identities. Existing typed catalog/build/emission/projection and external-parity requirements survive. |

Two migration-scoped rules need explicit limits. The existing
`optimization-target-refs` selector and grouping requirements continue through
policy-neutral refs; neither authorizes optimization to choose PEFT host
targets. The `adapter-module-targeting` compatibility rule allows legacy
`component`/`component_key`/`path`/`module` fields only as aliases of the same
accepted, revision-pinned projection during migration. They are not another
identity, binding store, or permanent old adapter API. The strict
`peft.continue_from` default survives as artifact continuation or
initialization intent, never exact same-run restoration.

The scenario-level audit is complete: `AdapterMode`-gated instantiation,
adapter orchestration, saving, and optimizer-input preparation in
`adapter-system`; target passing, grouping handoff, and `loha` export/merge in
`adapter-module-targeting`; and LoRA/VeRA build requests all have matching
modified or removed requirements above. The mode-named consumer scenarios in
`optimization-target-refs` have modified bodies. The generic `mode` producer,
startup-filtered-component, resource-component, and orchestration scenarios
in `training-observability` have modified bodies. Inherited optimization-owned
host targeting also occurs in the adapter-system ownership rule, adapter-module
target resolution/provenance/build/scope, LoRA/VeRA build-source prose, and
target-ref construction scenarios; each is replaced by authored intent plus
governed PEFT realization, with optimization retaining only trainability,
parameter grouping, and advancement. No unchanged method requirement is
permission to restore old targeting or mode ownership.

#### Metadata-change reconciliation ledger and milestone dependencies

| Metadata area | Unchanged requirement | Overlap disposition and dependent milestone |
| --- | --- | --- |
| Catalog identity and source provenance | Durable catalog assignment, portable evidence, source selections, ordered materialization evidence, and lineage remain metadata-owned. | Metadata foundation tasks 2–3 can proceed independently. Its loading task 4.1 must reconcile this change's participant authority and D5 materialization exchange before tasks 4.2–4.6 publish runtime-facing loading state; family task 5 and composition task 6 then consume accepted transitions, not Trainer-owned bindings. |
| `loaded-model-components` | One typed loading-evidence shape, family-declared components, successful source decisions, deferred updates, and limitations remain required. | The metadata delta's `Trainer stores the primary loaded-component state` scenario conflicts directly. This change's `MODIFIED` scenario governs current bindings; metadata task 4.1 must replace the old owner while retaining evidence in an accepted scoped projection. Loader candidates become current only through authority publication. |
| `optimization-target-refs` and structural metadata | Qualified structural paths, distinct live-object/storage observations, alias evidence, optional catalog links, and observation-local execution deduplication remain required. | The metadata delta's optimization-only builder and unqualified live-parameter field conflict. This change's policy-neutral refs and revision-pinned consumer views govern runtime use. Metadata structural tasks 9 onward may supply structural identities but cannot make live parameters durable identity, metadata the binding authority, or catalog resolution a prerequisite for an otherwise valid live view. This change's optimization task 3 consumes shared refs without taking PEFT target policy. |
| `model-family-metadata` and resource intelligence | Typed model/realization/artifact facts, append-only composition history, registry filing, qualified resource owners, and bounded structural queries remain required. | Authority-issued participant and artifact results supply durable projections; metadata filing does not establish live identity. This change's product/restoration tasks 4.3–4.5 and metadata composition/artifact tasks 6–7 must agree on revisions and actual product results. Resource intelligence remains an observer. |

These are boundary dependencies, not a whole-change landing order. Before
either change syncs an overlapping spec, both deltas must be reconciled with
the realized authority, projection, and evidence types. Metadata task 4.1 is
the explicit shared-boundary gate, including repair of its
`model-family-metadata` modified-requirement scenario omission; it must be
revisited when this change's participant and D5 types become concrete, and
both changes must pass strict validation then.
The rework's `loaded-model-components` delta currently uses the same new
scenario headings as the metadata delta (including different source layouts
and deferred loading). Metadata task 4.1 must compare those headings and
their bodies with the rework delta again before either change syncs; a repair
or rename on one side cannot silently diverge from the other.

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
- Making every tensor operation, worker task, or backend collective a public
  graph node, or fixing final Python representation classes before G5.
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

This conceptual lifecycle order does not require production code to be
implemented from the authoring side forward. Because the intended Trainer and
pipeline capabilities shape the contract, the training-side consumer, state,
realization, and execution meanings may be constructed and tested first
against explicit test-only accepted inputs. Contract-guided authoring and
fulfillment then produce those same inputs. A test builder is not a production
acceptance bypass and must not become a compatibility facade around the old
active-strategy topology.

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

### Current-code evidence for the intended Trainer boundary

The existing [implementation mapping](../../../docs_design/models-strategy-trainer/strategy_system_implementation_mapping.md)
is the detailed source-to-target disposition; this is its focused evidence
refresh, not a new current-code design. Exact source and one-hop call traces
were checked against graph generation `2026-09-22T05:52:18Z`. Every cited code
path below had matching filesystem metadata and no recorded indexing gap;
that is a best-effort coverage signal, not proof of graph completeness.

| Existing seam | Verified current-state fact that the target must replace or preserve |
| --- | --- |
| `train.py:18–29`, `trainer.py:256–301` | The launcher constructs strategy and mode separately and passes both to Trainer; `Trainer.train()` orders setup, caching, model preparation, optimizer preparation, startup evaluation, loop, and finalization. The graph traces the launcher into both factories and Trainer, and the Trainer into those phases. |
| `trainer.py:307–431`, `trainer.py:970–1045` | Setup obtains `load_target_model()` from the strategy, assigns `loaded_components`, synchronizes family-shaped Trainer views, and files model-realization metadata. Trainer's component projections are mutable current state today, not an accepted-authority seam. |
| `model_prep.py:30–112` | Deferred denoiser loading replaces Trainer's component collection, then mode hooks create trainables and finish precision while generic casting calls strategy facets. The graph confirms calls to lazy loading, mode hooks, and `sync_component_views()`. |
| `optimizer.py:61–185`, `optimization/types.py:38–98` | The phase delegates parameter/optimizer construction and backend preparation to the mode, builds a scheduler and validation loader, registers state hooks, and resumes through Accelerate. Existing logical/execution groups are useful lower-level machinery but contain concrete runtime members, not durable authored unit identity. |
| `training_loop.py:461–618` | The loop invokes `strategies.process_batch()` and mode hooks but itself performs objective observation updates, backward, clipping, optimizer/scheduler stepping, trigger dispatch, and tracking. The graph traces these calls; their source order, not an assumed one-loss target contract, is the current evidence. |
| `trainer.py:471–507`, `checkpointing.py:45–103` | Trainer delegates trained checkpoint saving to the mode and records a save event; exact-state resume is a separate Accelerate `load_state()` path invoked from optimizer preparation. This is the current conflation/split to replace with explicit product and restoration results. |
| `caching.py:45–171`, `validation.py:55–95`, `training_loop.py:133–183`, `sample_generation.py:435–549`, `sdxl/sampling.py:67–150`, `sdxl/validation.py:106–196` | Pipeline code already owns cache traversal, validation scheduling, step triggers, and shared sampling traversal, while family strategy callbacks also own device/eval handling or validation traversal and model-specific computation. Graph traces confirm the shared sampling helper's family callers and the step trigger's sampling/validation path. Dynamic strategy dispatch is not fully represented as a graph caller edge, so the exact family callback source was checked directly. |

This evidence supports moving generic coordination into training-owned services
while preserving selected model/algorithm behavior outside Trainer. It does
not imply that today's class and file boundaries should be copied into the
target, nor that SDXL is the universal exchange shape.

### D5. Trainer consumption is designed before authoring APIs

The target code is derived backward from what one Trainer engine must consume.
The following frame is normative at the responsibility level and deliberately
does not prescribe methods or one fixed call sequence. Its rows are lifecycle
uses of one authoritative obligation model, not separate sources of validity:

Every eventual exchange makes six categories explicit: request inputs,
readiness evidence, result, canonical state effect, permitted external effect,
and failure. A category may be empty for a particular exchange, but it may not
be hidden behind whole-Trainer mutation or an active-strategy callback.
The input-production row below extends the completed G1 common frame with a
boundary exposed by the later G2 experiments; it does not retroactively claim
that task 1.4 already specified the input-specific exchange.

| Responsibility | Request and readiness | Successful result and state effect | External effect and failure boundary |
| --- | --- | --- | --- |
| Fulfill an authored strategy | Complete authored selections under the active contract/profile and every fact knowable before realization | Detailed acceptance containing one semantic arrangement, one authoritative run-specific obligation revision, and seeded run authority | No ordinary external runtime effect. Rejection returns detailed diagnostics and produces no Trainer-acceptable arrangement. |
| Materialize and bind | Accepted declarations, relationships, lifecycle permissions, source intent, obligations, and expected revisions | Training lifecycle coordination derives the request from accepted obligations and current state; a selected domain producer returns a candidate plus typed evidence; the run authority alone accepts and publishes a new or revised binding | Loading or construction may perform declared I/O before publication. Failure or stale evidence publishes no candidate; destructive replacement follows the separately accepted transition rules. |
| Prepare runtime | One coherent revision-pinned projection of every required participant, route/view, constraint, joint group, mutation permission, and obligation | Training-side preparation coordination derives the complete job and coordinates verification and final installation; its candidate separates authority-owned route/view changes, Trainer-owned backend state, and optimization runtime | Backend work may allocate, wrap, shard, compile, or communicate before publication. Replacement-only and destructive failures follow D8 and never expose a mixed current state. |
| Realize optimization | Accepted training subjects and units, grouping/policy meaning, current bindings, and preparation constraints | Trainer optimization infrastructure resolves live membership, trainability, logical/execution groups, optimizer/scheduler state, clipping, synchronization, and lifecycle projections | Optimizer/backend construction may allocate or communicate. Failure installs no partial runtime; later relevant revisions invalidate or require re-realization. |
| Produce and hand off input | Accepted source, selection, transformation, and packing meaning; current input-policy state; producer dependencies; and the consumer's representation and readiness obligations | Selected input behavior produces model-ready values with the identity, provenance, and logical boundaries required by the consumer; training input coordination checks admission and records handoff without owning the provider's internal policy state | Production may progress independently and perform I/O before handoff. Not-ready, failed, stale, misidentified, or incompatible input cannot be silently substituted or delivered as accepted work. |
| Execute training | Accepted execution structure, current prepared projections, prepared optimization runtime, changing inputs/coordinates, and granted authority | Accepted behavior returns declared computation outputs, optimization inputs, observations, owned-state updates, effects, or transition requests; Trainer performs retained mechanics | Only accepted effects may cross the execution boundary. Failure follows the active ownership profile, stops unsafe advancement, and cannot mutate canonical state outside an accepted transition. |
| Coordinate caching | Selected capability, data scope, representation/conditioning behavior, writable destination, and coherent dependency projection | Pipeline traversal/storage and selected codecs produce cache records plus dependency/readiness facts recorded as capability state | Cache writes are external effects. Failure or partial output is reported explicitly and cannot advertise stale or incomplete cache readiness. |
| Coordinate validation | Selected capability, trigger, evaluation-ready projections, accepted input/RNG and measurement policy | Trainer/pipeline traversal and accepted evaluation behavior produce typed evaluation results, observations, and declared owned-state effects | Evaluation may consume resources but has no undeclared training-state effect. Failure restores temporary projections or gates unsafe use, and cannot fabricate a completed measurement. |
| Coordinate sampling | Selected capability, trigger, destination, request set, and prepared conditioning/predictor/representation projections | Trainer/pipeline orchestration and accepted generation behavior produce typed sample results and updated sampling state | Selected output writes/publication and observation are explicit effects. Failure restores temporary projections or gates unsafe use, and reports which requested outputs did or did not materialize. |
| Persist trained artifacts | Declared product, coherent product-specific projection, consistency boundary, transformations, serializers, and destination | Product resolution and serialization produce a result describing actual members and physical resources without changing participant identity | File or remote publication is explicit and may partially fail. The result reports actual output independently from runtime-snapshot success. |
| Restore runtime | Restoration contract, snapshot identity, authority revisions, progress/input state, Trainer/backend/optimization state, and registered accepted-operation/capability contributors | Trainer coordinates contributor-owned restoration and publishes one coherent same-run state with restored progress and freshness | Snapshot reads and backend reconstruction are explicit effects. Failure never resumes from a partially restored mixture and is distinct from loading a trained artifact into a new run. |

Every row consumes the same accepted, revisioned obligation set established by
strategy fulfillment. A job carries the relevant obligation and source-state
revisions; known authored facts are checked during fulfillment, realized facts
when a candidate is produced, and backend/readiness facts when they become
available. Publication checks freshness again. None of the coordinators below
may invent a second acceptance rule or reinterpret authored meaning. The
named owners and coordinators describe responsibility, not proposed classes
or folders.

| D5 row | Job derivation and final publication | Governing delta requirement(s) |
| --- | --- | --- |
| Fulfill an authored strategy | Contract-guided strategy-side fulfillment checks the completed recipe against the already active contract and establishes the obligation revision and accepted arrangement; no partial arrangement is published on rejection. | `training-contract`: “Strategy fulfillment produces acceptance or rejection”, “Obligations are evaluated at the earliest authoritative evidence point”, “One accepted contract governs one run authority”. |
| Materialize and bind | Training lifecycle coordination derives a candidate job from accepted obligations/current revisions; the selected domain loader or constructor supplies evidence; only the run authority publishes binding/relationship changes. | `run-participant-state`: “Initial realization and later transitions share one enforcement direction”, “Transitions are revision-checked and atomic”; `loaded-model-components`: “Model loading returns the loaded-component surface”. |
| Prepare runtime | Training-side preparation coordination derives one complete attempt-scoped job and coordinates final publication of authority-owned routes/views with Trainer-owned backend/optimization state; neither side exposes a half-installed result. | `training-runtime-preparation`: “Preparation derives one coherent attempt-scoped job”, “Preparation results separate ownership surfaces”, “Fallible work precedes final publication”, “Stale results never publish against newer state”. |
| Realize optimization | Trainer optimization derives candidates from accepted subjects/units and current projections; the joint preparation coordinator publishes the prepared optimization runtime with the corresponding authority routes and Trainer backend state. | `training-optimization`: “Trainer owns standard realization and mechanics”, “Realization and preparation form one publication attempt”, “Binding changes invalidate affected optimization runtime”. |
| Produce and hand off input | Training input coordination manages producer lifecycle, readiness, admission, delivery, and failure routing under accepted obligations; selected input behavior owns live-source traversal, selection/packing policy, and private continuation state. Cache-production traversal remains with caching coordination. An admitted handoff is not proof of action completion or optimization advancement. | `training-contract`: “Authors assemble strategies explicitly”, “Obligations are evaluated at the earliest authoritative evidence point”; `accepted-training-execution`: “Input handoff retains identity and admission meaning”, “Cross-owner action facts remain correlatable”; `training-capability-coordination`: “Caching separates semantics from storage orchestration”, “Cache publication and consumer admission are distinct”, “Independent cache production has bounded lifecycle and owned continuation”. G4.1 files the cache-production exchange below; task 4.4 completes coordinated exact-restoration protocols. |
| Execute training | Trainer's loop supplies time/input and current prepared projections; accepted behavior executes only its granted operations, while Trainer commits its retained mechanics and the authority handles any requested canonical transition. | `accepted-training-execution`: “Accepted execution survives the authoring object”, “Execution uses current prepared state”, “The standard profile retains generic Trainer mechanics”. |
| Coordinate caching | The caching capability coordinator derives work from an accepted request and current data/representation dependencies; it publishes readiness only for completed, fresh cache results. | `training-capability-coordination`: “Capability readiness derives from accepted state”, “Caching separates semantics from storage orchestration”, “Cache requests and results preserve the selected computation”, “Cache freshness follows actual computation dependencies”, “Cache publication and consumer admission are distinct”, “Cache storage lifecycle preserves outstanding use”, “Independent cache production has bounded lifecycle and owned continuation”. |
| Coordinate validation | Trainer/pipeline scheduling derives the request from accepted scope, measurement policy, and current protected projections; validation coordination publishes completed measurements only when their accepted coverage/reduction boundary succeeds. Partial outcomes and declared effects remain separately reportable through their owners. | `training-capability-coordination`: “Validation separates evaluation semantics from traversal”, “Validation results preserve measurement and effect meaning”, “Evaluation and generation use protected current state”, “Capability temporary projections have scoped cleanup”. G4.2 completes the exchange below. |
| Coordinate sampling | Trainer/pipeline scheduling derives a request from accepted generation choices, destination, and current protected projections; sampling coordination distinguishes generation, output publication, and observation outcomes for each required work/output association. | `training-capability-coordination`: “Sampling separates generation semantics from orchestration”, “Sampling results distinguish generation publication and reporting”, “Evaluation and generation use protected current state”, “Capability temporary projections have scoped cleanup”. G4.2 completes the exchange below. |
| Persist trained artifacts | Persistence coordination derives the accepted product plan from a coherent state projection; publication of the artifact result follows actual writes and does not publish a fictitious product on partial failure. | `training-artifact-persistence`: “Artifact persistence has declaration, request, plan, and result stages”, “Artifact results report actual output”, “Persistence uses one accepted boundary”. |
| Restore runtime | Trainer/pipeline restoration coordination derives the job from the accepted snapshot contract and registered state contributors; it coordinates one coherent final publication across authority, Trainer/backend, optimization, input, and contributor-owned state. | `training-capability-coordination`: “Persistence and restoration remain distinct capabilities”; `run-participant-state`: “Runtime identity and lineage remain separate”; `training-runtime-preparation`: “Process failure belongs to restoration”. |

The capability rows provide a common request/evidence/ownership frame;
completed G4 exchanges below refine their domain-specific protocols. This
table alone does not claim every capability delta is complete. Remaining G4
tasks must make their corresponding failure requirements explicit before
production implementation. None requires SDXL-shaped universal inputs, and no
coordinator may bypass the authority's binding and revision checks.

#### G4.1 Caching exchange

Caching reuses an accepted intermediate computation; it does not decide which
training inputs, captions, augmentations, or gradient paths the run should use.
The selected representation or conditioning implementation defines the cache
boundary: what computation is already done, what remains live, the complete
value/schema needed downstream, and the dependencies and reuse guarantees.
Shared contract conformance judges those meanings; ordinary strategy authors
do not recreate cache validators. The coordinator evaluates new evidence
against the existing run-specific obligations, not a second acceptance policy.

The cooperating owners remain distinct:

- Selected domain behavior owns encoding, decoding, representation/schema,
  and dependency meaning, including stochastic and gradient semantics.
- Data/cache infrastructure owns cache-production traversal, storage,
  production work, and readiness publication. Its internal workers need not
  become individual run-graph nodes.
- The selected input provider owns consumption traversal, sample/caption
  selection, transformations/packing, and its private continuation state.
- Run coordination owns the accepted activities' lifecycle and cross-owner
  handoff/failure relationships. The consuming coordination surface checks
  admission under that consumer's existing obligations: training input
  coordination for training inputs, or the relevant capability coordination
  for validation/generation consumption, directly or through shared input
  coordination. This does not transfer provider algorithms or create separate
  acceptance rules, and requires no repeated strategy queries.

These are responsibilities, not four new classes or a fixed execution chain.
Pre-training caching, lazy production on a miss, and independently progressing
production use the same boundary meanings where selected and supported.

| Exchange category | Required caching meaning |
| --- | --- |
| Request inputs | Selected accepted computation/behavior, capability/codec, and representation boundary; requested work or production-source scope; exact relevant input/variant and transformation meaning; required consumer schema; producer dependencies and consistency/reuse policy; destination/storage support; accepted miss, capacity, cancellation, and failure policies. Bounded requests may be derived during execution without changing the accepted behavior. |
| Readiness evidence | Required source bindings/views and protected producer-state access; current obligation/dependency evidence; permitted placement and resource capacity; usable storage. Reuse additionally needs evidence for the stored value's actual inputs, representation, and source state. Known unsupported combinations fail at fulfillment; facts only available during production are checked at their authoritative point and before publication/use. |
| Result | Correlation to the request and production attempt; completed publication units, their actual input/transformation and production-state provenance, schema/dependency evidence, storage access, and publication outcome. Pending, failed, cancelled, stale-for-the-request, and uncertain work remain distinguishable. Partial job coverage does not imply complete coverage. |
| Canonical state effect | Cache coordination records verified cache availability and production continuation. Input selection, packing, delivery, and exposure state remain with their own owners. Cache results do not publish participant bindings, amend optimization, or advance training progress. |
| External effect | Accepted source reads, encoding/transfer, payload/index writes, and storage lifecycle changes. Eviction, compaction, or replacement cannot invalidate an outstanding accepted read; access must remain protected through its required completion or fail before dependent use. |
| Failure | Report the reached boundary, actual completed/publicly usable output, known effects, and uncertainty. Incomplete encoding, writing, indexing, or dependency evidence cannot publish readiness. Accepted wait, retry/recompute, fallback, quarantine, or stop policies govern affected work; unsupported or semantics-changing substitutes are rejected. Cleanup failures remain distinct from the primary failure. |

**Identity and reuse.** A logical sample is not a cache record, request, caption
variant, or file path. One sample may use several cached representations or
variants; several samples may share one reusable value when its complete
accepted dependencies match. Storage access may address a file, shard member,
memory entry, or another selected backend without defining semantic identity.
The exchange preserves only the associations needed by accepted consumers,
coordination, observation, and restoration; it imposes no universal image,
latent, tokenizer, or tensor-axis fields.

The selected computation is not identified solely by a physical Python
callable, compiler artifact, or backend object. A different realization of the
same accepted behavior need not invalidate reuse if evidence establishes that
its relevant computation, numerical/stochastic, and dependency obligations
remain satisfied. Realization differences remain dependencies where they
affect those guarantees; neither a matching semantic label nor an assumed
eager/compiled equivalence is sufficient. This does not introduce a universal
computation-ID scheme or automatic implementation substitution.

Dependency evidence covers the computation upstream of the cache boundary,
including relevant source content, realized transformations/randomness,
encoder state, tokenizer/formatting, precision, output selection, and schema
where the selected semantics depend on them. These are examples, not a
mandatory universal key tuple. A path, caption hash, unchanged participant
reference, or unchanged binding revision alone cannot establish all of them.
Ordinary weight updates can invalidate derived values without replacing the
participant or rebuilding prepared execution. Conversely, unrelated
downstream updates need not invalidate a precisely evidenced upstream cache.
Persistent values from an earlier run use the same evidence and admission
rules against the new run's obligations. Prior availability or matching
authored addresses do not prove current compatibility or transfer participant
identity; reuse needs evidenced correspondence to the current producers and
representation requirements.

For example, cached frozen Qwen states and T5 token information can feed a live
trainable LLM adapter. Adapter updates leave that upstream cache valid; encoder
updates affect its producer-state guarantee. A detached value cannot replace
a required derivative path through a trainable encoder. Likewise, reusing one
realized crop or caption does not preserve a policy that requests different
ones, and an approximate embedding edit is not exact re-encoding. Such changed
algorithms require their own explicitly accepted behavior, not a cache miss
shortcut. If supported, a mixed hit/miss path must encode the actual requested
caption and preserve the full selected conditioning schema.

**Publication and admission.** Produced, stored, published-ready, admitted,
handed-off, consumed, and update-contributing are different facts. Readiness
is published only for a complete usable accepted publication unit with storage
and dependency evidence. That unit may be a value, bundle, or independently
consumable chunk where the selected representation permits it; this names a
consistency boundary, not a new execution-language construct. The boundary may
cover several resources and is not automatically the entire logical
representation, dataset, or production job. A complete chunk cannot advertise
the whole representation as ready, and a consumer requiring a full bundle
must wait for that bundle's obligations. Other complete records may remain
ready after an unrelated item fails if the accepted policy
permits partial coverage. A queue message or orphan payload is not readiness.
Payload and locator/index availability must agree before dependent reads;
interrupted writes require the storage implementation's declared recovery
behavior rather than an assumed crash-safe transaction.

The result describes the source state actually used, not merely the state
requested at launch. This may be one protected snapshot or structured
provenance over portions of the work where the accepted production policy
allows that. In-flight dependency changes cannot relabel an old result as a
new one or automatically admit mixed-state work. Publication checks the
request's current obligations, and the consuming coordinator checks exact
work association, representation, freshness/lag, and gradient requirements
again under its existing consumer obligations. One published representation
may be admissible for validation but not training, or vice versa; shared
storage does not imply shared admission or require a universal training-input
gateway.
The value's evidence and required source/storage access must remain valid
through the dependent use, following the shared current-use/completion rules.
Older-state values may remain stored or serve an explicitly accepted versioned
policy; they cannot remain current under a guarantee that has failed. A route
change alone need not invalidate a stored value if its accepted semantic and
numerical dependencies demonstrably remain satisfied.
Published availability records completed storage and production provenance,
not a perpetual current-use guarantee. A query claiming current usability must
evaluate current dependency evidence, including ordinary producer-state
changes that do not revise bindings. Implementations may proactively withdraw
affected guarantees or re-evaluate them at query/admission; neither may expose
a failed guarantee as current. This does not require one notification path or
invalidate unchanged physical availability merely to record incompatibility.

**Independent production and continuation.** The run structure may relate
cache production and consumption without one shared step/epoch clock or a
dataset-wide startup barrier. A selected implementation exposes bounded work,
resource placement, backpressure, readiness, cancellation, and failure to run
coordination while retaining its private queue/workers. Assignment follows
accepted data ownership and distributed topology, not a universal modulo over
all process ranks. Background work is not permission for unsafe concurrent
access to training weights or unlimited same-device allocation.

An exhausted cache or failed producer follows the accepted policy. Waiting,
recomputing, or an explicitly equivalent fallback may preserve requested work;
silently choosing another caption/sample or using incomplete conditioning does
not. A provider that intentionally selects a ready subset owns that exposure
policy and its continuation. Cache coordination cannot infer that policy from
arrival order or claim selection, consumption, or optimizer advancement from
a completed write. Lifecycle stop/cancellation must account for in-flight
work before releasing resources or establishing recoverable state.

For exact continuation, cache coordination contributes the production position,
pending/ready/publication state, dependencies, and stored-resource guarantees
needed by the selected recovery policy. The provider separately contributes
selection/packing/RNG and handoff state. Reconstructible cache payloads may be
reissued rather than embedded in every snapshot where the accepted recovery
claim permits it. Saving an index, pausing a producer, or restoring a queue
alone does not establish a coherent cut with model/optimizer/input state.
G4.4 defines that cross-owner snapshot protocol; G5 chooses concrete types and
backend publication/access mechanisms.

**Evidence and implementation limits.** The following checks complete the
caching meanings, not a claim that the current backend implements them:

| Pressure case / current evidence | Required consequence |
| --- | --- |
| `Trainer.train()` (`library/training/runners/trainer.py:256–301`) invokes caching before the loop; `CachingEngine.cache_dataset()` (`library/data/caching_engine.py:252–348`) processes a finite manifest and synchronizes ranks. | Preserve useful batch/encoding machinery without making full-manifest completion a universal execution prerequisite. Independently progressing production needs its own supported implementation and lifecycle tests. |
| `CacheBackend` (`caching_engine.py:71–208`) accepts image tensors and per-entry paths; `CacheEntry` (`library/data/structures.py:54–112`) stores one image, caption, and latent/TE paths. | Decouple cached representation and storage access from image/sample identity; allow shared/variant records and sharded or bounded storage without requiring those backends in the first migration. |
| SDXL TE encoding/validation (`library/strategies/sdxl/caching.py:472–630`) records a caption hash and checks it only when present; `_load_te_outputs()` (`library/data/dataloader.py:311–330`) loads assigned paths. | Complete producer/schema dependency evidence and per-request admission are required; a successful old validity check does not demonstrate them. |
| Current configuration (`library/config/config_validation.py:90–100,152–158,624–647`) rejects dynamic-caption caching and cache/offload combinations. | Those restrictions protect the current implementation; future supported exact variant/miss/offload arrangements must be judged by their actual obligations, not inherit the old global exclusions. Required live gradients remain a genuine compatibility concern. |
| [Anima conditioning boundary](research/anima-llm-adapter.md), [data pipeline research](research/training-data-pipeline.md), and [conditioning research](../../../docs_design/future_ideas/text_encoder_conditioning_research.md) distinguish frozen upstream work, live downstream computation, stochastic transformations, and exact caption variants. | Invalidate only genuinely affected dependencies and do not detach gradients or freeze dynamic semantics merely to obtain a cache hit. |
| [Async production](../../../docs_design/future_ideas/async_data.md), [TE prefetch](../../../docs_design/future_ideas/async_te_offload_otf.md), [shards](../../../docs_design/future_ideas/data_shards.md), [streaming](../../../docs_design/future_ideas/streamed_training.md), and [accounting](../../../docs_design/future_ideas/data_accounting.md). | Preserve bounded independent lifecycle, partial readiness, storage/reader safety, and separate readiness/exposure/advancement. Snapshot-ready epochs and live-ready selection remain selected alternatives, not a universal recommendation. |
| Existing [producer and composition experiments](research/training-mechanism-sketch.md) simulate out-of-order work, caption/state admission, backpressure, failure, and separate input/optimization outcomes. | They support the owner and handoff distinctions; they do not prove real producer-state protection, concurrent storage publication/eviction, distributed assignment, or exact recovery. G5 conformance must exercise the chosen implementation, including interrupted publication and an outstanding reader during storage changes. |

Graph discovery and exact-source checks support the bounded production claims;
coverage metadata at generation `2026-10-03T00:09:00Z` recorded no gaps on the
relied-on paths. Dynamic dispatch was checked in source rather than inferred
from absent call edges. Future-data notes are design evidence, not promises
that SQLite, a particular shard format, streaming ingestion, learned embedding
editing, or every asynchronous backend ships in the first migration. This
exchange adds no execution-language construct or universal training sequence.

#### G4.2 Validation and sampling exchanges

Validation measures accepted behavior; sampling runs selected generation
behavior. Neither requires another Trainer or an active strategy callback.
Trainer/pipeline coordination owns due requests, outer input/request traversal,
resource access, ordinary runtime projections, aggregation execution,
destinations, and observation routing. Selected domain behavior owns the
evaluation or generation algorithm and its declared state. Coordination uses
the existing obligation revision and readiness evidence, not new acceptance
rules invented at the trigger.

**Validation.** The selected evaluation defines what is measured, the
contributions needed to compute it, and its reduction/weighting meaning.
Pipeline aggregation executes that meaning; it does not guess a formula from
batch losses. A selected reducer can be ordinary Python. This does not require
a metric expression language or graph nodes for every contribution.

| Exchange category | Required validation meaning |
| --- | --- |
| Request inputs | Selected evaluation; requested input scope and stopping/coverage rule; input/provider policy; source-state consistency; prepared view, numerical and derivative requirements; measurement/reduction policy; declared effects and partial/failure policy. Triggers may derive bounded requests during execution. |
| Readiness evidence | Current accepted source/view access and input admission; supported runtime/gradient requirements; available reduction participants/resources. Known incompatible choices fail during fulfillment; realized/backend evidence is checked when available and before use. |
| Result | Request/attempt and evaluated-input associations; actual source-state provenance; typed contributions or completed measurements with coverage, weighting/denominator, and reduction meaning as needed by consumers; completion, partial, cancellation, failure, and declared-effect outcomes. No universal scalar or tensor axes are required. |
| State effect | Validation coordination owns outer traversal/aggregation coordination records; validation history remains observation state under its observation owner. Selected evaluation or another accepted algorithm owns its declared adaptive state/effects. Participant transitions and optimization mechanics retain their existing owners; computing derivatives does not implicitly grant advancement authority. |
| External effect | Accepted input reads, computation/communication, and routed observations. Any additional publication or service effect must be declared; evaluation does not silently become checkpointing or a training-state writer. |
| Failure | Preserve reached coverage, known measurements/effects, and uncertainty. An unfinished evaluation cannot claim the requested completed score. An explicitly accepted partial measurement remains labelled with its scope; undefined or empty reductions follow the selected policy, not a fabricated zero. Cleanup and observation failures are separate outcomes. |

For unequal batches, masks, token lengths, or distributed input partitions,
mean-of-batch-means is valid only if that is the selected measurement.
Coverage/reduction must account for relevant missing or repeated work and
rank contributions. Request identity is not an input identity, and a trigger
step is not proof of the weights actually evaluated. Stable comparison may
require a protected source state for the entire measurement; a supported
mixed-state measurement needs explicit consistency and provenance meaning.
No single version counter is prescribed.

Validation coordination owns outer traversal, not the input provider's
selection, packing, cursor, or RNG algorithm. Evaluation uses its own provider
continuation or an explicitly accepted sharing policy; it cannot accidentally
consume the training cursor. Bounded stream evaluation need not have a dataset
length or epoch. Cache-backed inputs obey G4.1 admission rules, including
exact caption/variant and encoder dependencies.

Evaluation is not universally passive or `no_grad`. The recorded BD3LM case
in [LLM research](research/llm-training.md) uses validation measurements to
choose a later training noise interval. Such a selected adaptive reaction
must survive fulfillment, with explicit state ownership, observation delivery,
allowed derivative work, and continuation contribution. It is not a generic
Trainer policy, accidental objective mutation, or permission for hidden
optimizer work. A reaction may be a separate accepted activity; neither its
completion nor its failure is automatically the measurement's outcome.

**Sampling.** Generation owns conditioning, prediction/objective conventions,
schedule/guidance, decoding, and output meaning. Pipeline coordination owns
request timing/traversal, resource access, destinations, writing/publication,
and result routing through selected implementations. A domain inference loop
is legitimate selected computation, not a reason to put family algorithms in
Trainer. Prompt text, image dimensions, PIL images, and PNG are not universal
exchange fields.

| Exchange category | Required sampling meaning |
| --- | --- |
| Request inputs | Selected generator and its supported domain parameters; work/output associations and relevant random-input policy; source/view consistency; destination/publication requirements; declared effects, cancellation, and partial/failure policy. Seeds do not alone promise exact replay. |
| Readiness evidence | Current protected participant/view and conditioning/representation dependencies; supported generator/objective pairing; required placement/capacity and usable destinations. Known unsupported combinations fail before runtime use without family discovery in Trainer. |
| Result | Request/attempt and output associations; actual source-state/input provenance; produced payloads and actual saved/published resources where required; separately known generation, publication, observation, cancellation, failure, and effect outcomes. A planned filename is not evidence of a written output. |
| State effect | Sampling coordination owns request/publication continuation. Selected generation owns its declared algorithmic state; it cannot silently alter authority state or optimization progress. |
| External effect | Accepted source/service reads, generation/transfer, output writes/publication, and routed observations. Text, video, audio, or related multi-output products use selected schemas and serializers, not an image-shaped core. |
| Failure | Preserve successful outputs and each failed, uncertain, or unattempted part at the accepted boundary. Generated-but-unsaved differs from saved-but-unreported. Reporting failure cannot erase saved output or authorize regeneration; retry/reissue requires its accepted policy and actual outcome evidence. No job-wide rollback or crash-safe publication is assumed. |

Sample output publication is not automatically trained-artifact persistence
or an exact runtime snapshot. G4.3 and G4.4 complete those distinct exchanges.

**Shared access and temporary state.** Trigger eligibility is not execution
readiness. Both capabilities use G3's protected-current-use relationships and
prepared projections. They cannot blindly unwrap, move, or recast a live
object or read a stale separate inference representation. A read conflicting
with a weight update, retained derivative use, or temporary SAM perturbation
must wait, fail before use, or use a supported isolated/synchronized view
under its accepted consistency policy. Protect access through actual backend
completion, not merely the Python return. Actual provenance may be one
protected state, a snapshot, or accepted structured state/segment evidence;
recording mixed provenance does not itself authorize mixed-state work.

Pipeline coordination establishes and restores ordinary model/optimizer
runtime projections, placement, dtype, and RNG according to their declared
owners and accepted numerical policy. It preserves the prior relevant state,
including mixed module modes, rather than unconditionally calling
`train(True)`. Cleanup covers partial setup, execution failure, cancellation,
and asynchronous completion. RNG isolation must cover the relevant declared
generators/devices/activities or reject an unsupported overlap; saving one
global generator is not proof of concurrent isolation or bitwise replay.

If restoration or handback is uncertain, dependent use remains unavailable
until accepted recovery establishes validity. Exclusion release alone is not
that evidence. Keep the primary failure and cleanup failures, and retain
known measurements, outputs, and effects even if final cleanup or a sink
fails. Stateful effects may have occurred before an invalid/missing result;
the boundary is not a sandbox for arbitrary Python.

Validation, sampling, input production, and optimization retain their own
accepted due/completion/failure relationships. A shared trigger does not impose
the current helper's sample-then-validate order, a universal optimizer clock,
or a global pause. Capability and algorithm owners contribute their own
continuation needs; G4.4 defines a coherent restoration cut, not this exchange.

**Evidence and implementation checks.** These complete the semantic exchanges,
not production capability implementations:

| Bounded evidence / pressure case | Consequence and remaining check |
| --- | --- |
| `ValidationScheduler` (`library/training/phases/validation.py:38–95`) and `_run_step_side_effects()` (`library/training/phases/training_loop.py:133–183`) provide useful due logic; `run_sampling_and_validation()` (`library/training/phases/orchestration_helpers.py:60–114`) hard-codes sequencing and broad strategy calls. | Preserve accepted scheduling without carrying that fixed sequence or active strategy boundary into the engine. Test independently due capabilities and their declared dependencies. |
| SDXL evaluation (`library/strategies/sdxl/validation.py:18–196`) combines selected timestep/L2 meaning with RNG switching, cyclic loader traversal, batch-mean aggregation, and history updates; SD3 (`library/strategies/sd3/validation.py:19–154`) repeats the workflow with different conditioning/representation. | Keep domain evaluation selected while moving outer coordination out. G5 must test unequal weights/masks, partial or empty coverage, distributed reduction when supported, and input-provider continuation without assuming the current loss formula is universal. |
| `temporarily_in_eval_mode()` (`orchestration_helpers.py:49–57`) changes mode before its `try` and restores training unconditionally; validation and `sample_images_common()` (`library/training/sample_generation.py:435–549`) restore global RNG only on their normal path. | G5 must inject partial setup, evaluation/generation, cancellation, and cleanup failures and verify prior projections or unsafe-use gating. Existing helpers are not proof of scoped restoration. |
| SD, SDXL, and SD3 sampling (`library/strategies/sd/sampling.py:19–83`, `library/strategies/sdxl/sampling.py:67–150`, `library/strategies/sd3/sampling.py:116–180`) unwrap/move family components and construct generation implementations. SDXL/SD3 flow backends contain selected inference algorithms. | Preserve family algorithm differences without universal component slots; test stale inference views and reads during conflicting temporary mutations using the chosen backend access protocol. |
| `SamplingRequest` (`sample_generation.py:45–59`) is image-shaped; `sample_image_inference()` (`sample_generation.py:552–625`) saves an image before direct tracker logging. | G5 must distinguish generated, saved, and reported outcomes, including save failure, later sink failure, and a partial multi-request result. Selected non-image/multi-output schemas must not require new Trainer family branches. |
| Recorded [BD3LM evaluation](research/llm-training.md), [video/audio cases](research/video-audio-training.md), and [train/inference-view coordination](research/preference-and-rl-post-training.md). | Preserve adaptive reactions and domain-specific measurement/generation semantics; these research cases establish requirements/neutrality pressure, not that every algorithm or separate inference backend ships in the first migration. |

Tier 2 graph discovery, traces, and exact-source checks support these bounded
current-code claims; coverage generation `2026-10-03T01:55:27Z` recorded no gaps
on relied paths. Dynamic strategy dispatch was verified in source, not inferred
from absent edges. Existing scheduling/parser tests and local G3 experiments
do not demonstrate real concurrent view protection, distributed reduction,
full RNG isolation, or exact recovery. Those implementation checks remain G5;
no new execution-language construct, universal result object, or fixed run
sequence is selected here.

#### Run state is composed, not universally owned

The training process needs one coherent per-run state aggregate, but that does
not make every value part of the participant authority or give one object
unrestricted mutation rights. The aggregate distinguishes at least:

- governed semantic state: accepted arrangement meaning, obligations,
  participants, relationships, bindings, routes, revisions, and freshness;
- Trainer execution state: step/epoch progress, accumulation position,
  synchronization and backend state, interruption, and cleanup progress;
- input state: accepted source/policy dependencies, prepared input sources,
  provider-owned cursors/RNG/buffers and other continuation state, plus
  training-coordinated readiness, admission, and handoff facts;
- optimization state: semantic unit identity plus candidate and prepared
  Trainer-owned runtime state;
- capability state: selected capability readiness, dependencies, requests,
  results, and any capability-owned runtime state;
- accepted-behavior state: state owned by objective, sampling, adaptive, or
  other accepted operations; and
- observation state: metric accumulators, validation history, runtime trace,
  and reporting-session status.

The aggregate may hold or index these sections and supply coherent snapshots,
but each section keeps its declared writer. Phases, capability coordinators,
accepted operations, and Trainer mechanics create or advance the state they
own. Observability formats, routes, buffers, and persists accepted facts; it
does not own the training state that produced them.

Logging, metadata, resource observation, interruption, and cleanup are
cross-cutting Trainer/pipeline responsibilities. They consume typed facts and
results and never become a second live authority.

At this common-frame level, the D5 result/state-effect homes and permitted
writers are as follows. A listed consumer receives a scoped projection or
result, not unrestricted access to the owning section. G2–G4 refine these to
concrete result types and permitted effects.

| D5 row | State section and writer | Principal projection/result consumers |
| --- | --- | --- |
| Fulfill | Governed semantic state; contract-guided fulfillment establishes obligations and seeds the authority. | Training lifecycle coordination, Trainer, selected operations, capability coordinators. |
| Materialize/bind | Governed semantic state; only the run authority publishes bindings, relationships, and revisions from accepted candidates. | Preparation, optimization, execution, capabilities, metadata/observation. |
| Prepare | Governed routes/views via run authority; Trainer execution/backend and optimization state via their owners in one coordinated publication. | Trainer loop, accepted operations, capabilities, observability. |
| Optimize | Optimization state via Trainer optimization; binding-dependent readiness coordinated with preparation publication. | Trainer loop, artifact/restoration coordination, observability. |
| Produce/handoff input | Input state; selected input behavior writes its policy and producer continuation state, while training input coordination records readiness/admission and handoff under accepted obligations. | Accepted computation, Trainer scheduling/accounting, caching, observability, restoration. |
| Execute | Accepted-behavior state via its accepted operation; Trainer execution/progress via Trainer; canonical transitions via authority only. | Optimization, capability triggers, observability, restoration. |
| Cache | Capability state and input/cache records via caching coordination; no direct authority binding write. | Input preparation, accepted execution/validation, observability. |
| Validate | Capability state and observation state via validation coordination and observation owner respectively. | Trainer scheduling, accepted adaptive consumers, observability/reporting. |
| Sample | Capability state and actual output results via sampling coordination; observation state via observation owner. | Trainer scheduling, artifact/reporting and observability consumers. |
| Persist product | Artifact publication result via persistence coordination; durable metadata receives the actual product fact, without changing participant identity. | Run reporting, metadata, observability, later artifact-based initialization. |
| Restore | Governed state via authority; Trainer/input/optimization/capability/accepted-behavior sections via their respective restoration contributors under Trainer/pipeline coordination. | Trainer and every resumed consumer of a coherent current projection. |

Semantic acceptance and executable readiness are deliberately different. The
former says that the authored meaning is complete and permitted and establishes
the exact remaining obligations. The latter says that the evidence required at
the relevant readiness checkpoint has been accepted and the resulting state
has been published. Known incompatibility is rejected at the former boundary;
contingent artifact, backend, process, or resource failure may still prevent
the latter without causing downstream code to reinterpret the strategy.

G2–G4 refine the domain-specific contents of these categories before
production types are chosen. The frame above fixes the ownership questions the
training-first sweep can already answer without pretending that the detailed
execution, optimization, or capability protocols are complete.

### D6. One run authority owns participant and relationship state

The Q1–Q5 conclusions are carried forward as one coherent rule set.

This authority owns the governed semantic section of the broader run-state
aggregate described in D5. It does not thereby own Trainer progress, input
resources, optimizer/backend objects, accepted-operation state, capability
state, or observation buffers. Cross-section publication uses an explicit
coordinator; it does not collapse those owners into the participant authority.

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
- Separately declared teacher and student participants remain distinct even
  when initialized from one source. The words "teacher" and "student" may
  instead name two execution roles of one participant, or an upstream source
  that is not live in the current run; role labels do not establish participant
  identity. Backend replicas and composite handles do not become participants.

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

Training-side preparation coordination derives the coherent job from the
accepted requirements and one current authority projection, directs the
permitted backend work, and coordinates final installation across the
authority-owned and Trainer-owned surfaces. The run authority evaluates and
publishes its routes and views; Trainer optimization and backend owners install
their runtime state as part of the same logical publication. The coordinator
has no separate binding map or compatibility rules. Its Python location and
API remain for G5. Neither surface may publish its portion independently and
then ask the other to catch up.

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

#### G3.1 Standard advancement policies

The standard profile accepts **Trainer-coordinated contributions and unit
advancement**, not one fixed `backward(); step()` sequence. The contract offers
these policies before strategy authoring; the authored strategy selects and
connects them, fulfillment accepts the resulting obligations, and preparation
must show that the selected backend can realize them. An accepted runtime due
policy may choose among accepted actions using current coordinates and owned
state. This preserves deterministic unusual cadence and bounded adaptive
choices without a per-step query to the strategy authoring interface.

Three supported shapes share the same Trainer-owned mechanics:

| Standard shape | Contribution and advancement meaning |
| --- | --- |
| One due unit | An accepted action offers a source to one unit. That unit may contribute over a declared accumulation window before it is eligible to advance; the action occurring does not itself prove an optimizer step. |
| Independently due units | A bounded policy selects accepted actions/units at different cadences, including alternating or nonconsecutive contribution windows. Only the selected unit's gradient window and advancement are touched. Other participants may still take part in its computation under their accepted gradient/view requirements. No G/D pair or one shared global-step counter is built into Trainer. |
| Jointly due units | One accepted action offers addressed sources to two or more disjoint units. A shared source may admit one backward serving all; distinct sources require accepted per-unit gradient routing rather than an unexamined sum of losses. Contribution, window completion, step order, and per-unit outcomes remain explicit. The action is not an atomic multi-optimizer transaction. |

For **each** shape, the accepted policy identifies the source and gradient
route for every contributing unit, the accumulation/completion rule, any
required synchronization participants, the clipping scope and timing, the
ordered units eligible to advance, the scheduler/update trigger, and the
zeroing/discard rule. Trainer owns backward or an observably equivalent
backend lowering, accumulation, synchronization, clipping, optimizer and
scheduler invocation, final zeroing, and per-unit progress/outcome reporting.
Selected operations supply the addressed computation values; they do not gain
those mechanics by returning an optimization input. One source shared by
several units is not a universal one-backward rule. When distinct sources can
influence several units, a naive summed backward is invalid if it introduces
cross-unit gradients the accepted routing did not authorize. Per-unit gradient
extraction, recomputation, or another lowering is allowed only when it
preserves the accepted gradients and the backend can support it. Otherwise
the arrangement is not executable under that backend; it is not silently
converted to an imperative strategy callback.

Clipping occurs at the accepted synchronized completion boundary, after any
required unscaling, over the declared unit scope or a declared joint scope
whose participating units complete together. A unit's pending gradients are
not cleared merely because a different unit is due. Trainer opens and closes
each gradient window according to the accepted zeroing rule; a skipped or
failed attempt cannot silently carry unsafe gradients into the next window.
Different unit clocks and partially overlapping windows require backend
support for the selected isolation and synchronization; an implementation
that cannot realize them must reject readiness before executing that policy.
This is a backend conformance obligation, not a promise that current
Accelerate/DeepSpeed preparation already supplies it.

Trainer/pipeline owns action scheduling, progress, training/evaluation
boundaries, and invocation of accepted lifecycle effects. A unit-local
optimizer mode or scheduler transition runs at its accepted lifecycle point;
selected post-advancement behavior (such as adapter regularization) remains
selected behavior invoked under its declared effect authority. The policy
must say whether a scheduler follows an action, an attempted optimizer call,
or a backend-confirmed non-skipped advancement. If it requires a distinction
the backend cannot report, realization rejects that policy. `step()`
returning, a backend reporting a skip, progress being recorded, and numerical
parameter change are different facts; the last is not inferred.

Before mutation, Trainer checks that current unit definitions, prepared
routes, due selection, offered sources, and gradient scopes match the
accepted policy. An operation/backward failure may already have changed RNG,
owned state, or gradients. An optimizer call that fails after entering its
backend is **uncertain** unless that backend establishes a narrower result.
In a joint action, previously returned unit calls and later unattempted units
are reported separately. Trainer stops dependent work after uncertain live
effects; it neither retries nor rolls back the action by default. No ordinary
standard policy promises all-or-nothing advancement across units.

Checkpoint ownership follows the same split for all three shapes: Trainer
contributes each unit's identity/revision, mutable optimizer and scheduler
state, unit-local contribution/advancement position, any still-pending
gradients needed to continue, and backend scaling/synchronization state;
the due-policy owner contributes its changing state; other owners contribute
input, operation, participant, and progress state. A snapshot may instead be
taken at a proven quiescent boundary with no pending gradient window. Only a
coordinated complete cut can claim exact same-run continuation; G4 defines
its protocol. G3.4–G3.6 still define the concrete prepared runtime and
attempt/result exchanges, including how those outcomes correlate with input
handoff and progress. G3.2 separately defines transferred authority.

The bounded [advancement-policy experiment](../../../tests/unit/training/test_advancement_policy_experiment.py)
checks nonconsecutive per-unit accumulation and distinct-source gradient
routing. The earlier [joint-unit experiment](research/training-mechanism-sketch.md#second-check-two-optimization-units-due-in-one-action)
checks shared-source backward and partial failure. These are evidence for the
semantic distinctions, not proof of distributed lowering, exact restoration,
or a production policy interface.

#### G3.2 Explicit changed-authority boundary

An extension is a **supported ownership profile offered by the active
contract**, not permission created by an authored operation calling Trainer
methods. The authored strategy selects a supported profile and supplies the
required region, scope, state/effect, and recovery declarations. Fulfillment
accepts or rejects that filing against the pre-existing contract. The
accepted runtime region—not the strategy's authoring interface—executes its
granted work. Unrelated actions in the same run keep their standard owners.

The deliberately narrow SAM-style test profile from G2.5 has one due
optimization unit and a quiescent gradient window. Its phase ownership is:

| Phase/action | Owner in this chosen split |
| --- | --- |
| Due selection, current-readiness check, input handoff, initial gradient clear, and exclusive-interval entry | Trainer/run coordination |
| First selected computation and backward; reading the scoped first gradient; temporary edit of the accepted parameter substructure | Accepted region executor |
| Intermediate gradient clear/replacement; second selected computation and backward; restoration of the unperturbed parameter view | Accepted region executor |
| Clean-handback check, final clipping, optimizer/scheduler advancement, final gradient clear, unit/action progress, triggers, and failure coordination | Trainer/optimization and run coordination |

This profile does **not** transfer the final optimizer step or scheduler to
the region. Initial/final gradient clears and the *intermediate* clear are
different scoped actions; the coarse experimental `ZERO_GRAD` mechanic cannot
describe their owners by itself. Each accepted action/phase has exactly one
owner. A region requesting `advance`, omitting intermediate gradient control,
or sharing an owned phase with Trainer is rejected at fulfillment. A selected
SAM wrapper whose second step restores parameters **and** advances its base
optimizer does not fit this split unchanged. It must be decomposed, target a
different explicitly offered profile that transfers advancement and its
state/failure duties, or be rejected; selecting the wrapper does not silently
grant ownership.

The region receives the current prepared route/view and only the accepted
parameter substructure, batch/coordinate values, required selected
computation, and its own state/RNG projections. It may read first-pass
gradients, change only the granted parameters temporarily, clear/replace only
the specified gradients between passes, and emit declared observations and
owned-state effects. It does not receive unrestricted run authority, a whole
Trainer, canonical binding setters, or optimizer/scheduler advancement
handles under this split. The accepted scope is expressed in semantic
participant/substructure and unit identities plus revisions; physical Python
parameters are a checked realization, not the grant's identity. Narrow
interfaces and declared effects make conformance inspectable, but arbitrary
Python closures can still hide mutation. The profile therefore requires
trusted selected implementations and runtime evidence where static checking
cannot prove effects or gradient correctness; it is not a sandbox claim.

Fulfillment decides everything knowable from the authored filing: the
offered profile, exact phase owners, scoped subject, declared temporary and
persistent effects, state contributors, allowed accumulation rule, required
clean handback, and failure/restoration claim. The chosen test profile rejects
entry with pending accumulated gradients rather than erasing them; a different
two-pass accumulation policy would need its own accepted synchronization and
gradient rules. Governed realization then checks current binding/route and
unit revisions, backend access to the scoped parameters and gradients,
exclusive-use and snapshot exclusion, synchronization/scaling behavior, and
whether temporary edits can be restored and evidenced. A backend conflict
fails readiness **before** the region executes, not as a late grant of broader
authority. Current Accelerate/DeepSpeed support is not presumed.

On success, the region must hand back evidence that the unperturbed current
view is restored, the second-pass gradient is ready under the accepted
synchronization/scaling policy, and all declared region-owned effects are
accounted for. Trainer checks that handback against the same attempt and
current revisions before final clipping or advancement. A mere success
Boolean from arbitrary Python is not proof of restored backend state; the
concrete evidence and exchange belong to G3.4–G3.6. The region may not publish
new participant bindings or optimization definitions as a side effect of its
temporary edit. A requested structural change still goes through the run
authority's accepted transition path.

This is an already-produced gradient handback, not a request for Trainer to
run standard backward again. Current-use protection must cover handback
verification and the retained mechanics that rely on those same views and
gradients. Releasing exclusion and then rechecking revisions does not close
the race with a conflicting edit or publication. An implementation may retain
the interval or transfer protected access without a gap; this does not require
one particular lock or result type.

Failure during the first pass may leave gradients, RNG, input, or region state
changed. Failure after perturbation may additionally leave parameters or
backend state uncertain. Even if restoration succeeds after a failed second
pass, the action is not complete and Trainer must not step; any narrower
in-process continuation needs an accepted recovery rule covering *all*
effects. If restoration or handback is uncertain, affected live use and exact
snapshots stop until a valid state is re-established. This split permits
exact snapshots only at coordinated quiescent, unperturbed boundaries and
requires every persistent region-state contributor to join the same-run
snapshot. It promises neither mid-region snapshots, automatic rollback, nor
blind replay. Failure during Trainer's retained final advancement follows
G3.1's per-unit uncertain-outcome rule, not a special SAM transaction.

The [bounded two-pass experiment](../../../tests/unit/training/test_imperative_authority_experiment.py)
checks phase ownership, final Trainer advancement after clean restoration,
pre-mutation rejection, and two distinct failure states on one-process
PyTorch parameters. It cannot establish distributed exclusive access,
backend-confirmed restoration, hidden-effect prevention, or exact recovery;
those remain realization and restoration obligations rather than reasons to
weaken this boundary.

#### G3.4 Backend-flexible optimization and preparation exchange

The accepted **optimization plan** names units by run-scoped identity and
definition revision; semantic participant/substructure membership; logical
groups; optimizer, scheduler, trainability, and advancement policies; and the
relationships and constraints on which those meanings depend. It contains no
current `nn.Parameter`, optimizer, wrapper, or backend composite as an identity.
This is accepted meaning supplied to Trainer optimization, not a second
strategy-authored recipe for physical construction.

Training-side preparation coordination derives one **request** from that plan,
the current authority/obligation revision, and the requested execution work.
It names every required participant and route/view (including frozen
participants), complete overlapping backend groups, permitted mutation mode,
and the accepted unit definitions and constraints to realize. It pins the
relevant participant, binding, relationship, route, and obligation sources;
conservative whole-snapshot dependency is valid until narrower dependencies
can be demonstrated. Optimization supplies its unit and grouping requirements
to this same attempt, not an independent optimizer-preparation publication.
Known incompatibility is rejected before expected-expensive backend work.

An **optimization candidate** is provisional physical resolution under that
request. For each semantic unit it accounts for selected members and tied
aliases, one standard-profile owner and execution group per resolved
parameter, logical-group correspondence, trainability, gradient/clipping and
synchronization membership, and an optimizer/scheduler construction policy.
Several accepted paths may alias one physical parameter within the same unit
and execution group; their provenance remains visible while that parameter
appears only once in the optimizer. The standard-profile conflict is a second
unit or execution group claiming the same parameter, not the mere existence
of two names for one owner.
The candidate may be a construction recipe before wrapping or concrete
objects already built; neither form is current run state. Each backend
transformation must retain enough provenance to associate its resulting
parameter and callable views with the accepted member/participant meanings.
If the backend replaces parameters, it must build or correctly rebind the
optimizer against the resulting parameters. If that correspondence cannot be
established, the attempt fails readiness; a plausible optimizer object is not
evidence that it will update the accepted subjects.

The complete **result** separates authority route/view proposals, a
Trainer-owned prepared runtime for each semantic unit, Trainer backend and
inseparable-group coordination state, source/dependency revisions, evidence
for constraints and member correspondence, and structured failures. Each
prepared unit retains its address, incarnation, definition revision, logical
groups, and accepted advancement/source policy independently of the physical
optimizer's identity or ordering in a returned tuple. Current execution
requires both a current authority route/view guarantee and matching Trainer
backend/optimization readiness. A composite backend handle may cover several
participants or units without merging their identities.

Two valid physical traces realize the same accepted meanings:

| Backend order | Provisional work within one attempt | Required completed evidence |
| --- | --- | --- |
| Optimizer before wrapper | Resolve accepted members against source bindings; build unit optimizers and schedulers; jointly or sequentially prepare required participants and optimization objects. | Returned optimizers still target the final prepared parameter views, or the backend provides an evidenced rebind. Every required route, including frozen execution participants, and every unit is accounted for. |
| Wrapper before optimizer | Prepare/transform required participants; use the transformation's provenance to resolve accepted members against prepared parameter views; build and, if needed, prepare unit optimizers and schedulers. | The resolved post-transform members still implement the same accepted subjects, groups, and policies; every unit and required route is accounted for. |

The ordering is a backend realization choice, not a change to a unit's
definition revision. Some backends or policies may support only one trace;
acceptance of semantic meaning does not promise executable readiness under
every backend. A backend may also require multiple ordered calls, including a
joint model/optimizer call, inside the same attempt. The result is checked for
member coverage/non-overlap, optimizer-to-current-member correspondence,
source freshness, accepted obligations, backend-group completeness, and rank
agreement before final installation. Only a complete, checked result becomes
current across the separately owned surfaces. The final visibility change
does not perform expected-fallible backend work or call observers. Failure of
replacement-only work leaves still-valid old state current; in-place mutation
first withdraws affected guarantees as D8 requires. Process/rank loss at the
publication boundary follows restoration, not an in-process rollback claim.

The [bounded preparation-order experiment](../../../tests/unit/training/test_preparation_exchange_experiment.py)
checks both traces with two disjoint units and a frozen required participant.
It also rejects an optimizer left on pre-wrapper parameters, tied or
backend-created overlap, an incomplete result, and a stale source revision
before publication. Its single-process replacement models the visibility
invariant only: it does not implement separate owner stores, rank agreement,
substantive obligation evaluation beyond revision pinning, a real backend
provenance or evidenced optimizer-rebind protocol, destructive preparation,
cross-attempt inseparable-group overlap, or distributed optimizer construction.
It models one execution group per unit, so cross-group alias checks remain a
production conformance obligation.
Those are production realization and conformance work, not reasons to make
one physical ordering part of authored meaning. G3.5 handles later
invalidation/replanning, and G3.6 connects prepared units to execution-time
contributions and outcomes.

#### G3.5 Invalidation and replanning after accepted change

Prepared usability is a claim about **current dependencies**, not a property
of a surviving Python object. Each prepared route/view, optimization unit
runtime, and backend group records the accepted obligation and exact
participant, binding, relationship, substructure, route, unit-definition, and
backend/preparation revisions that its guarantee needs. A projection without
a justified narrower set depends on the whole authority snapshot. Comparing
those dependencies happens before dependent work and again before an in-flight
candidate publishes. An unrelated change may leave an explicitly precise
projection usable; sharing an inseparable backend group is a relevant
dependency even when a participant's own binding did not change.
An obligation-revision change is not ignored merely because a physical
binding stayed put: reuse needs authority-backed evidence that the
projection's applicable obligations remain satisfied under the new revision.
Without that evidence it becomes stale.

The accepted transition first determines **what changed in meaning**, then
which current realizations and guarantees depend on it. It does not infer a
new optimization definition from `requires_grad`, parameter-object identity,
or a backend wrapper. The following distinctions apply under one accepted
contract and run authority:

| Accepted event | Semantic unit continuity | Current-use consequence |
| --- | --- | --- |
| Ordinary optimizer advancement, a schedule/due-policy decision within accepted bounds, or an accepted operation-state update | Same identity and definition revision; mutable state/progress advances. | No automatic preparation rebuild. A derived product that depends on the changed weights or state may still become stale under its own dependency policy. |
| Backend rewrap, compilation, route rebinding, or recreation of concrete optimizer objects without changing accepted subjects/policy | Same unit identity and definition revision. | Re-realize affected routes/views and unit runtime against current bindings and backend evidence. Retain only unrelated precise projections outside any affected inseparable group. |
| Compatible participant binding replacement or accepted parameter surgery that preserves its participant meaning and the unit's semantic selector, grouping, policy, and dependencies | Same participant and unit identities; same unit definition revision, even if new physical parameters resolve under the accepted selector. | Binding/substructure dependencies and affected prepared routes/units become non-current. Re-resolve members, aliases, trainability, and backend preparation. Mutable optimizer state follows an accepted preserve/migrate/reset rule, never object-order inference. |
| Relationship transition, such as activating an adapter effect | Participant and relationship incarnations remain; a unit's definition stays the same *only if* its accepted subject and semantic dependency definitions stay the same. | Invalidate every relationship-dependent route/view/unit and any inseparable backend group. If the transition changes accepted unit membership or meaning, revise the unit as in the next row. |
| Accepted training-subject, optimizer-significant grouping, optimizer/scheduler, advancement-policy, or semantic-dependency-definition change | Preserve an independently managed unit's identity but advance its accepted-definition revision. An ordinary bounded runtime policy choice does not count as a definition change. | Old unit runtime cannot advance under the new definition. Replan membership and state-continuity treatment, then prepare and publish the complete new runtime. A transient train/eval or backend `requires_grad` setting does not itself amend the accepted subjects. |
| Surgery or arrangement amendment changes semantic member selection, or creates/splits/merges/replaces an independently managed responsibility | Revise a continuing unit if its responsibility remains the same; split, merge, retirement/recreation, or replacement establishes new unit identities. An incompatible participant replacement likewise needs a new participant incarnation. | Withdraw affected current guarantees, resolve the new accepted obligations and unit meanings, and prepare from the resulting state. No old optimizer state is silently assigned to a new unit or incompatible member. |

For parameter surgery, the accepted selector's meaning matters. A selector
that deliberately covers a changing substructure may resolve new physical
parameters at the **same** unit definition revision after an allowed
structure-preserving transition. If the accepted selection or grouping itself
changes, the unit revision advances. Both cases require concrete member and
alias resolution again. Two paths tied to one parameter within one unit and
execution group remain one optimizer owner; a second unit or execution group
claiming it still fails standard-profile resolution.
When an accepted temporary trainability or lifecycle choice changes a
consumer-visible runtime projection, its owner updates that coherent
projection without revising the semantic unit definition; an arbitrary
`requires_grad` mutation cannot become a new accepted subject by convention.

Concrete traces check the distinctions without making their example names
universal:

- A compatible denoiser replacement changes its binding from revision 4 to
  5. A `main` unit at incarnation 10, definition revision 3, still selects the
  same accepted denoiser substructure. Its old parameter/optimizer runtime is
  unusable; a new complete preparation may realize **incarnation 10,
  revision 3** against binding 5 after the accepted optimizer-state
  continuation rule is satisfied. An optimistic result pinned to binding 4
  cannot publish afterward.
- An accepted change from denoiser-only to denoiser-plus-adapter subjects
  changes that continuing unit's membership: incarnation 10 remains but its
  definition becomes revision 4. If instead one responsibility is split into
  independently advanced base and adapter units, the two resulting units get
  new incarnations; neither inherits incarnation 10 merely by reusing its
  address or optimizer object.
- If an adapter relationship changes inside a joint backend group covering
  denoiser and adapter, both members' prepared guarantees are withdrawn.
  A separate frozen encoder's route may remain usable when its precise
  dependencies exclude that relationship and group; a whole-snapshot
  dependency would conservatively invalidate it too.
- Progressive distillation can replace the teacher's bound weights, reset
  accepted EMA and stage-local progress, and replace the student's optimizer
  at a stage boundary. Teacher binding, student unit definition, and EMA
  continuity are judged separately: a new optimizer object alone does not
  create a unit, while a changed accepted schedule or subject revises its
  definition. The next action waits for the coherent new stage; no partial
  old/new mixture becomes executable.

Before a structural or policy transition takes effect, training coordinates
a safe boundary for dependent activities and unit gradient windows. Pending
contributions are completed, preserved, or discarded only under an accepted
rule; neither switching routes nor changing a unit definition may silently
carry old gradients into the new runtime. A replacement-only candidate may be
built off to the side, but it cannot become current against the wrong
authority/obligation/unit revisions. If an accepted transition publishes
before replacement preparation is ready, it atomically withdraws affected
prepared guarantees and pauses dependent work until a complete new result
publishes. A staged candidate for prospective state must be checked against
the actually accepted prospective revisions before any combined publication;
it may not treat an anticipated amendment as already authoritative. D8's
destructive path withdraws guarantees **before** in-place mutation and makes
older optimistic candidates unable to restore them.

The same rule handles group fan-out. If a change affects one member of an
inseparable backend group, the whole group loses prepared usability; a later
job must include all overlapping members even when some retain their
participant and unit meanings. A precise dependency set may spare unrelated
groups, routes, and units. A conservative full-snapshot dependency instead
invalidates them explicitly; it cannot pretend they stayed fresh. New
preparation checks the current accepted obligations, source dependencies,
member ownership, state-continuity rule, and backend evidence before one
complete publication. Failure does not make any affected old guarantee valid
again merely because its Python object still exists.

Unit **identity/revision continuity** and mutable **optimizer-state
continuity** are separate decisions. Recreating the same unit revision with
new parameters does not prove that old momentum, scheduler counters,
accumulated gradients, or backend state can be reused. The accepted transition
must specify which state is preserved, migrated with evidence, intentionally
reset, or unavailable; failure to establish the required continuation blocks
readiness. This is especially important for compatible checkpoint replacement,
parameter surgery, and stage changes. The exact state migration formats and
coordinated snapshot mechanism remain G4/G5 work.

For exact same-run restoration, a coherent snapshot restores the authority's
accepted arrangement and obligation revisions, participant/relationship
incarnations, unit identities and definition revisions, mutable unit state and
advancement coordinates, and the other required owner contributions. New
process objects and backend wrappers are same-revision re-realizations, not
new units. Restored optimizer state must be matched to the restored semantic
members and verified before exact continuation becomes ready; an implicit
reset cannot masquerade as exact resume. A trained artifact or snapshot used
to start another run instead establishes new authority, participant, and unit
identities with provenance, not live identity continuity. G4 defines the
coordinated recoverable cut; this section fixes only the optimization and
freshness obligations it must preserve.
For example, restoring the same run's recorded `main` incarnation 10,
definition revision 3, and mutable state into a new process keeps those
identifiers after backend re-preparation; loading its weights to start a new
run does not.

These cases follow the settled D6–D9 and delta-spec rules. They do not
prescribe a production dependency index, migration API, or fixed
transition-to-preparation call order; G5 chooses those representations.

#### G3.6 Execution and optimization across run activities

The "step exchange" is the set of **owner-to-owner handoffs** needed to run
the accepted structure, not one mandatory step call or result packet. The
contract offers the supported ownership profiles; the authored strategy
selects behavior and policy; fulfillment fixes their accepted meanings and
obligations; preparation establishes current executable routes and unit
runtimes. During the run, coordination supplies changing inputs and checks
readiness, selected behavior executes within its grant, and each owner reports
the facts needed by its consumers. An independently progressing activity need
not wait for, or produce, a training action.

| Boundary | What must cross it | Owner and consequence |
| --- | --- | --- |
| Due work and readiness | The accepted activity/action and current coordinates, bounded policy decision and its owned state, required dependencies, and the current readiness result. | Run coordination invokes the accepted due policy and gates dependent work; the policy owner advances its state under its accepted rule. Different activities may have different clocks; a due choice is not proof that work ran. |
| Independent producer to consumer | Requested work identity, produced identity and actual source dependencies, selected payload and required logical-example/packing boundaries, readiness, admission, handoff, and producer failure. | Selected input behavior owns its queue/workers and continuation; training input coordination owns lifecycle and admission against the current consumer obligation. A ready value may be rejected or await a declared policy. Handoff is not action consumption. |
| Coordination to accepted operation or region | Admitted input values, changing coordinates, current prepared participant uses/routes/views, accepted owned-state projection, and only the authority granted to this work. | The selected implementation executes; Trainer does not reconstruct its model/objective algorithm or call the authored strategy for choices. Freshness and backend guarantees are checked where the relevant state may have changed, without rediscovering static wiring on every invocation. |
| Selected work to its consumers | Addressed computation outputs; optimization offers when its profile uses Trainer-owned optimization; observations and accounting values; operation-owned state changes; declared external effects; and proposed authority-governed transitions. These are distinct meanings even if one implementation produces several. | The operation/region owns its selected computation and permitted state/effects. An optimization offer identifies its accepted source, destination unit(s), and gradient requirements but neither steps an optimizer nor certifies advancement. An observation is routed to its declared consumer; a transition proposal cannot publish canonical state itself. |
| Optimization and region handback | For each unit or transferred phase: contribution/window status, gradient and synchronization readiness, actual attempted mechanics, backend-supported skip/return/uncertain outcome, scheduler/zeroing status, and any required clean-handback evidence. | Trainer owns standard mechanics and per-unit progress; an imperative region owns only its explicitly transferred phases. Trainer checks a region's handback before retained work. A returned optimizer call does not prove numerical parameter change. |
| Accepted transition and subsequent work | Transition request, authority decision and new revisions, affected dependency invalidation, identity/definition continuity, mutable-state continuation, and preparation readiness for each dependent activity. | The run authority alone publishes participant/relationship changes; the relevant owner updates its own state. G3.5 determines reuse or replanning. A proposal or accepted transition is not itself a ready prepared runtime. |

Only relationships needed by another owner, preparation, coordination,
observation, or restoration need to be visible at this level. A selected
operation may keep internal tensors and branching in Python. Likewise, a
producer may keep its worker schedule private. The accepted structure must
nevertheless show an externally relevant handoff, dependency, effect, or
failure instead of hiding a second training loop inside an opaque call.
The concrete representation may use scoped results, state projections, or
reporting events; it need not use one universal result class or event log.

**Preparation preserves required interactions, not internal call sequences.**
Preparation may change how accepted work executes, but must preserve every
fact and interaction required by its accepted consumers, owners and recovery
rules. This includes input identity/admission, computation and gradients under
the accepted numerical policy, owned-state and external effects, dynamic
decisions and observation order, authority, current-use dependencies,
separately reportable completion/failure outcomes, and restoration coverage.
Matching the final loss or weights alone is insufficient. A lowering cannot
silently change the algorithm, defer feedback past a decision that needs it,
or erase a required opportunity for admission, capability work, transition or
snapshot coordination.

Fusion, partitioning, loop chunking and different backend schedules are
permitted where those obligations remain satisfied. Private values, branches
and calls need not remain separate executable nodes. Required boundaries may
survive as scoped access, completion facts, guards or owner reports inside a
combined executable rather than as individual dispatches. Generating
Trainer-owned mechanics into that executable does not transfer their
authority to a selected operation. If a lowering cannot preserve required
partial outcomes or supported interactions, it is not ready for that accepted
profile; preparation must not substitute a weaker meaning. These are shared
conformance obligations, not a demand that every author implement validators
or that arbitrary Python can be proved conformant statically.

**Differentiated work connects gradient meaning to state lifetime.** Wherever
another owner relies on it, the accepted description preserves the relevant
outputs and requested derivatives, seed information where applicable, subject
or input-path routing, intentional gradient cuts, contribution destinations
and windows, applicable numerical policy, and the executor owning each phase.
Governed preparation establishes how the selected implementation/backend
satisfies those demands,
including any required saved values, numerical state or recomputation
conditions, supported requests, actual completion and safe release. An
unsupported derivative route or retention requirement fails readiness rather
than silently detaching a path, summing losses or changing ownership.

A ready gradient does not by itself authorize an update. If another required
derivative, recomputation or device use still needs the earlier parameter or
view state, conflicting mutation waits. A supported isolated representation
may permit earlier mutation only when it preserves the remaining work's
accepted meaning and all other window, synchronization, clipping and ordering
requirements are satisfied. Backend evidence may prove that fewer values
need retaining; without it, use a conservative safe lifetime. A revision
identifier alone does not retain numerical contents.

When derivative work crosses an independently coordinated boundary, preserve
its association with the originating invocation, required state and remaining
requests, and the completion/release or cancellation rule. This is a lifetime
and handoff obligation, not a mandatory continuation object for every forward
call or a project-owned autodiff engine. Existing framework differentiation
may realize it implicitly within closed work; supported imported or compiled
regions may realize the same meaning differently. Failure or cancellation
must not release still-used resources or certify uncertain gradients/state as
usable; replay or recomputation requires the accepted state/RNG/effect rule.

The hierarchy and its relationships must preserve the necessary language
information: selected work and dynamic policies; values/handoffs and gradient
routes; scoped state uses, effects and lifetimes;
ownership/grants and unit obligations; and completion, change and recovery
facts. This does not prescribe a node type for each category, a persistent
control IR, public tensor operators or one universal runtime packet. G5
chooses concrete constructs according to their consumers.

**Protection follows actual use.** Admission, execution and conflicting
publication must share a current-use protection protocol. A prepared revision
check alone does not protect mutable state after that check. Protection covers
the required use lifetime, including backward, recomputation, transfers or
outstanding device work where applicable; backend evidence may establish a
narrower safe interval. This applies across owners, including capability
readers and preparation publication, not only inside one action. The concrete
acquisition, handoff and completion mechanism remains G5 work.

**Correlation without a transaction.** The run must relate the accepted work
address, each invocation/attempt, admitted input work and its actual
dependencies, operation/region effects, addressed optimization unit
incarnations and definition revisions, unit-local outcomes, and any progress
record. These relationships may be distributed across owner-held records;
training run coordination is responsible for preserving their join even after
partial failure, using facts supplied by each owner rather than taking over
their state. G3.8 must make that obligation testable; G5 chooses its storage
and reporting form. An activity may produce work before an action exists,
an accepted action may consume several inputs,
and one action may offer to several units. No input, action, optimizer, or
global-step counter is the universal identity or completion boundary.
Completion is judged for the accepted activity and relationship in question:
produced, ready, admitted, handed off, consumed, contributed, advanced,
reported, and durably captured are different facts. G4 determines the
coordinated recoverable snapshot, not this exchange alone.

**Changing behavior is preserved.** Coordinates, batch contents, randomness,
current routes, observations, and accepted owner state can affect an
operation's outputs and next decision. A schedule or adaptive sampler may
change its state and select a bounded alternative without rebuilding the
accepted arrangement. Its owner records the state/effect even when no
optimization offer follows. A choice outside accepted bounds is rejected;
changing participant bindings, unit definitions, or the authority profile
instead follows G3.5 and the applicable transition/fulfillment path. Ordinary
optimizer advancement or an adaptive-state update does not automatically
invalidate prepared execution, though a derived input product can become
stale under its own dependency policy.

**Failure is local to the reached boundary, not a failed atomic step.** A
pre-invocation readiness or admission rejection cannot claim selected work
ran. Once selected Python or a backend call begins, failure may leave its
owned state, RNG, gradients, input position, external effects, or even live
parameters changed. A post-return result-shape failure is therefore a
post-invocation failure with possible effects. The responsible owner reports
what it can establish and marks the rest uncertain; coordination stops
dependent work until the accepted recovery rule re-establishes validity.
For jointly due units, returned, skipped, uncertain, and not-attempted calls
remain separate. A failed region handback cannot authorize Trainer's retained
advancement. Neither a zero progress increment nor a missing success result
proves the state unchanged. No ordinary exchange promises automatic rollback,
exact replay, or all-or-nothing optimizer advancement.

Partial acquisition, cancellation and cleanup failures obey the same rule:
releasing execution rights must not restore usability of uncertain state.
Before conflicting work can acquire access, affected dependencies must remain
withheld unless usable state has been established. Preserve known outcomes
and the primary failure alongside cleanup failures. An accepted recovery may
restore or replace affected state and re-establish readiness; withholding
ordinary use must not itself prevent that authorized recovery. These are
safety obligations, not a prescribed lock implementation or a claim that all
run activities stop together.

Concrete arrangements check that these meanings are not a disguised fixed
input-action-optimizer pipeline. Their names and payload details are examples,
not universal fields or built-in activity kinds:

- An SDXL-like operation consumes the current batch and coordinates, produces
  a differentiable optimization source and a separate accounting value, and
  emits timestep/loss observations to its selected adaptive owner. Later
  timestep selection changes with that owner's state; timesteps are not core
  fields.
- A joint Muon/AdamW-style partition can offer one shared computation to two
  disjoint units, each with its own contribution and advancement outcome.
  Alternating generator/discriminator work instead selects distinct actions
  and units at different due times. The same exchange permits both without
  making either optimizer ordering a universal sequence.
- An asynchronous text encoder can publish requested caption work out of
  order while no training action runs. Admission relates the result to its
  work/sample/caption and **actual** encoder dependency, not queue order or
  only the revision requested when work began. If the encoder changes while
  work is in flight, a produced value from the new state is not mislabeled as
  old; a genuinely old-state value follows the accepted freshness/lag policy.
  Producer failure and bounded backpressure belong to its run-level
  lifecycle, not a blocking read hidden in a model operation. Detached
  embeddings do not silently replace a required live gradient path through
  a trainable encoder. The producer's pending/ready/handoff state and the
  action's later unit outcomes remain separately owned but correlatable.
- A bounded two-pass imperative region can own its accepted intermediate
  backward, temporary edit, and gradient clear, then hand back an unperturbed
  view and final gradient for Trainer's retained clipping/advancement. A
  failed or uncertain handback stops that advancement; the region does not
  acquire the whole Trainer or an implicit final optimizer step.
- A progressive-distillation stage change can request a teacher binding,
  unit-definition, and owner-state transition. Dependent activities pause,
  the authority accepts or rejects the change, and affected routes/units are
  re-prepared before later work. Other independent capability or input work
  may have its own due and failure boundary; G4 specifies those exchanges.

The [G3.3 candidate](research/training-mechanism-sketch.md#g33-check-one-candidate-language-spans-run-level-input-and-repeated-actions)
demonstrates pre-resolved action wiring, independent
input publication, and a local attempt join, while G3.1/G3.2 and G3.4/G3.5
establish ownership, prepared-unit, and freshness evidence. It does **not**
demonstrate a general whole-run coordinator, autonomous producer scheduling,
multiple feeds, operation-owned state and transitions, capability work,
distributed backend outcomes, or exact restoration. Its `ActionResult`,
`AttemptReport`, one-feed action, and `completed_actions` counter therefore
remain experimental. In particular, the multiple-input relation above is an
exchange requirement, not a behavior proven by that candidate; G3.7 must
include it among the conformance cases. G3.7 also tests cost, G3.8 files the
resulting normative requirements, G4 completes capability and restoration
protocols, and G5 chooses concrete Python types and lowering.

#### G3.7 Conformance and execution-cost acceptance

These gates follow G3.1–G3.6 and D10's hierarchical run structure. They
evaluate observable meanings independently of experimental class names,
storage, and lowering. G5 must map each gate to the proposed production
representation and its implementation milestones. Defining a gate here does
not claim its full behavior has already been implemented or demonstrated.

Each conformance case needs a positive arrangement, a relevant rejected or
failed arrangement, and an oracle identifying the required values, identities,
owner state, gradients/outcomes, and ordering relationships. Deterministic
local fixtures may use an explicit expected trace or equivalent direct Python.
Alternative lowerings may reorder internal work only where accepted
dependencies, numerical/gradient requirements, effects and failure semantics
permit it; tests must not require identical internal callback or backend-call
sequences. Backend-specific numerical tolerances and nondeterminism follow
the accepted numerical policy rather than a universal bitwise-equality rule.

| Case and governing meaning | Required observable test | Existing evidence and remaining implementation check |
| --- | --- | --- |
| Ordinary computation and differing modalities; D10/G3.6 | A prepared diffusion-like arrangement and a materially different arrangement retain their selected computation, accounting, observations and representation boundaries. Releasing the authored strategy does not affect execution. Reject malformed outputs after invocation without claiming unchanged state. | `test_execution.py` tests ordinary/alternating work and output failure; G2 supplies SD/SD3 and non-image cases. This is not a production multi-family engine or an author-object-lifetime proof. |
| Equivalent lowering and combined work; G3.6 | Compare separate and combined execution under the same accepted profile. Preserve gradient routing, input association, adaptive feedback before dependent decisions, required capability/transition interactions and separately known partial outcomes. Reject a variant that matches final values but loses a required interaction or reports a returned unit as uncertain merely because later work failed. | Existing CPU comparisons check a narrow unary fixture, not general fusion or chunking. G5 must test the actual lowering against these owner-visible requirements; private call sequences need not match. |
| Multiple input correlation; D10/G3.6 | One accepted consumer uses work from two sources with distinct identities/dependencies and out-of-order readiness. Preserve both associations through operation effects and unit outcomes. Reject swapped, stale or missing members before joined computation; retain partial handoffs under the selected wait/reissue policy rather than silently duplicating or discarding them. | G3.3 has one feed per action. An action accepting several values is not proof of independently produced inputs or their continuation. This join remains a required implementation case. |
| Independent production; D10/G3.6 | Production progresses when no training action is due; ready work can remain inadmissible. Changing caption/actual encoder dependencies cannot pair results with the wrong request. Check lifecycle failure, bounded backpressure and shutdown independently of action completion. | Producer and run-language tests supply deterministic local evidence. True concurrent publication, cancellation races and distributed producer behavior need tests when an implementation claims them. |
| Joint and independently due units; G3.1 | Exercise shared-source and distinct-source gradients, alternating actions and nonconsecutive accumulation windows. Verify each unit's authorized gradients, clipping/synchronization scope, schedule trigger and zeroing rule; reject unsupported routing before mutation. | Execution/advancement tests demonstrate local shared and isolated gradients and accumulation. Backend scaling, skips and distributed synchronization need the applicable backend conformance test. |
| Delayed derivatives and numerical last use; G3.6 | Use a frozen differentiable conduit and two routed losses. In a supported recomputation variant, the first gradient is ready while the second derivative still requires earlier numerical state; prevent a conflicting update. Contrast a supported isolated-state variant that permits earlier update only when all unit obligations allow it. Check request/invocation association, gradient cuts, completion, cancellation and safe release; reject unsupported routing/retention before invocation. | G3.1 records local distinct-source routing evidence; the concrete proposal's B fixture supplies a retention trace, not an executed lifetime test. G5 must demonstrate the selected backend's live/isolated-state and release mechanisms without requiring a universal continuation object. |
| Membership and realization; G3.4 | Preserve semantic unit identities across both physical construction orders. Consolidate tied aliases in one unit/group; reject cross-owner overlap, stale physical optimizer membership and incomplete publication. | Preparation tests cover a limited local one-group realization. Real backend correspondence, multiple groups and complete cross-owner installation remain implementation checks. |
| Changing owner state and bounded choices; D10/G3.6 | Live coordinates/observations change selected decisions and owner state without author callbacks or recompilation. Include a state-only effect without an optimization offer. Reject choices outside accepted bounds; after a possible state update and failure, gate dependent work according to the accepted recovery rule. | Input-policy tests cover selected changing state; the G3.7 test confirms a changing selected operation still runs with compilation disabled. Owner continuity, observation delivery and restoration are broader requirements, not proven by a Python closure. |
| One participant, different uses; G2.6/D10 | Preserve one participant identity with distinct accepted view/gradient requirements; a frozen base may transmit gradients to a selected side network. Unsafe shared-view overlap is rejected or serialized, and uncertain restoration stops dependent use. | G2 worked cases establish required meanings; action wiring alone does not implement view exclusion or backend restoration. |
| Explicit alternative authority; G3.2 | Standard and granted work coexist with dynamic selected behavior after authoring ends. Each phase has one owner; unsupported/duplicate ownership fails before use. A two-pass region supplies the required restored view and final gradient before retained advancement, without another standard backward. Failed or uncertain handback cannot authorize advancement or an exact snapshot. | Imperative tests check the local chosen split. Backend evidence/exclusion, author-object lifetime and other offered profiles require their own conformance checks; arbitrary Python hidden mutation is not statically sandboxed. |
| Protected current use across owners; G3.2/G3.6 | Attempt conflicting edits/publication and capability reads during required numerical use, handback verification and retained mechanics; reject, wait or isolate according to accepted dependencies. Inject partial acquisition, cancellation and cleanup failure. Releasing exclusion must not admit dependent work on uncertain state; preserve known outcomes and primary/cleanup failures, and allow authorized recovery to re-establish readiness. | Existing local tests do not demonstrate this shared protocol. The concrete proposal's section 5a supplies a worked trace, not executable race, backend-completion or recovery evidence. G5 must test its actual access and completion mechanism; G4 supplies coordinated snapshot recovery. |
| Changed or stale preparation; G3.5 | Distinguish ordinary owner-state evolution, same-definition re-realization, revised definitions and new unit identities. Invalidate dependent routes/groups/units; reject an old in-flight result at publication. Unrelated precise projections may survive; pending gradient/state continuation follows its accepted rule. | G3.5 traces and preparation stale-source tests establish part of the evidence. A working transition-to-preparation path and group fan-out still need implementation tests. |
| Partial execution/advancement failure; G3.1/G3.6 | Relate admitted work and operation effects to returned, skipped, uncertain and unattempted unit/phase outcomes. Stop unsafe dependent work and preserve the primary failure if shutdown also fails. Neither missing completion nor a returned step proves numerical change, rollback or replay safety. | Execution/run-language/imperative tests demonstrate several local boundaries. Scheduler/zeroing failures and backend-specific knowledge must be exercised by their actual owners. |
| Work beyond training actions; D10/G3.6 | A triggered capability and a permitted stage transition have their own readiness, request, completion and failure relationships. Producer or state-only progress is not forced through an optimizer call or one global clock. | Governing whole-run requirements and G3.5/G3.6 traces require this. Capability protocols and a coordinated recoverable cut are completed in G4; G5 must demonstrate their composition. |

Conformance observations are test oracles, not a requirement for one runtime
event log. Run coordination owns correlation across owner-supplied facts;
selected providers, operations, optimization and authority keep their state.
For each applicable backend/profile, G5's migration gate must identify which
tests run locally, which require that backend, and which are explicitly
unsupported and rejected before execution. A local test cannot certify an
unexercised distributed guarantee. Exact restoration additionally requires
G4's coordinated snapshot protocol and contributors; these execution tests
cannot substitute for that gate.

Performance acceptance uses the same conformance arrangements. Every timing
comparison first demonstrates equivalent selected computation, required
runtime guards and owner-visible results. Compilation, fulfillment, expensive
preparation, model/backend compute and I/O are measured separately from steady
execution; a cheap workload must expose dispatch and bookkeeping directly.
Dynamic scheduling, adaptive updates and necessary freshness checks remain
in the timed scope when that case requires them.

| Cost case | Matched baseline and acceptance observation |
| --- | --- |
| Cheap action with 1, 4 and 16 selected operations | Compare prepared execution with direct Python using the same inputs, exception boundary, addressed outputs and observations. Record absolute excess microseconds and relative cost; heavy computation cannot be the sole performance case. |
| Whole-run dispatch and dormant work | Compare one due activity in arrangements with 0, 256 and 1,024 dormant actions. Startup may scale with accepted structure; ordinary dispatch must scale with due work and its relevant dependencies, not scan the entire arrangement. |
| Admission/freshness and lookup | Vary the relevant dependency count and separately time lookup and admission. Match actual required checks in the direct reference; do not obtain a lower number by removing current-use guards. Transition revalidation is distinguished from routine unchanged-state execution. |
| One, two and sixteen optimization units | Keep computation fixed while varying addressed contributions/results. Separate coordination, gradient handling and outcome bookkeeping from optimizer/backend compute. Cost may grow with the units actually involved. |
| Dynamic owner state and policies | Compare the same changing schedule/observation-driven decision and state effects in direct and prepared execution. Assert runtime decisions still change; static construction and strategy discovery must have zero calls in repeated execution. |
| Granted imperative region | Compare equivalent direct region execution with the same phase ownership, scope checks, exclusion and clean-handback requirements. Separate region math/backward from entry, handback and retained-action overhead. A synthetic grant alone is insufficient. |
| Governed change | Measure stale rejection, dependency invalidation and accepted transition separately from preparation/rebuild. Include precise unrelated reuse and conservative invalidation; ordinary weight/state evolution must not force structural rebuilding by convention. |
| Representative active training | Compare the comparable portion of the current loop and an equivalent direct reference under the same selected algorithm/backend. Report total latency/throughput alongside isolated framework cost. Additional capabilities and different algorithms require their own matched scope, not an invented old-loop equivalence. |

Repeated measurements must state Python/backend versions, environment, fixture
and source revisions, warmup, iteration count, timing scope and retention
policy. Use at least three independent process runs with multiple samples and
alternate comparison order. Report spreads and paired excess as well as
medians; overlapping distributions cannot establish a small regression.
Do not put noisy wall-clock thresholds in ordinary unit tests. Allocation
diagnostics run separately from timing and distinguish peak traced bytes,
post-collection retained state and cumulative allocation rate; one does not
measure the others. At fixed queue/history/gradient-window bounds, retained
framework state must not grow with elapsed attempts. Any accepted durable
history retention is a separately measured policy, not an accidental list.

The following numerical gates are **production regression acceptance policy
for this migration**, not permanent semantic requirements of accepted training
execution. The initial [G3.7 measurements](research/training-mechanism-sketch.md#g37-conformance-and-cost-check)
establish a same-environment reference envelope for the common measured
scope. Before this migration's production cutover, the proposed implementation
must run the matched cheap-action checks against the retained experiment in
the same environment. The reference fixture and measurement probe must remain
available until those production-cutover checks have passed. Its median
excess over direct Python must not exceed that
reference excess by more than the greater of **1 microsecond per action or
20% of reference excess**, across three process runs. This is a regression
allowance for this shared scope, not a universal hardware latency limit.
New required coordination/guards must be present in both compared paths and
have a separately justified budget before their implementation milestone;
they cannot be omitted to fit the old scope, and their budget cannot replace
these migration gates for comparable scope. The dormant-work check must show
no growing cost trend with arrangement size; an increase exceeding the greater
of **10% of the zero-dormant median or three times its within-run median
absolute deviation** requires investigation and correction or evidence of a
changed relevant-work scope before acceptance. These are measurable gates,
not a choice of graph storage or lowering. After this migration is accepted,
its reference fixture and numerical tolerances do not become a permanent
architecture contract. Later changes establish their own documented regression
policies while retaining the enduring requirements for semantically matched
comparisons, required runtime guards, relevant-work scaling, and bounded
retention.

G3.7 completes the test matrix and initial cost evidence. Production
conformance, extension/backend measurements, capability composition and
restoration are not closed by these CPU fixtures. G3.8 files the completed
exchange requirements; G4 completes their capability/recovery protocols;
G5 assigns concrete tests, environments and budgets to implementation work.

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

G2.6 selects the semantic representation described below: a hierarchical run
structure whose cross-owner dependencies and handoffs are explicit, with
selected Python behavior inside its work boundaries. It is graph-like where
the run must coordinate distinct owners, not a graph of every tensor or worker
operation. G5 still chooses concrete Python types and lowering.

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

The standard profile keeps training-action time and progress, accumulation,
synchronization, backward, clipping, advancement, zeroing, triggers,
observation, interruption, and cleanup under Trainer ownership. A custom
structured operation can replace standard decomposition while satisfying that
boundary. A different owner for one of those mechanics requires an explicit
extension/profile that states its actual executor and exchanges, rather than
assuming “the strategy takes over.”

G2.6 selects the common execution meanings below. G3 completes the step and
optimization exchange; G5 chooses its concrete Python representation rather
than letting the experimental classes become production APIs by default.

#### Input production and handoff

The accepted arrangement preserves which input sources, selection and
transformation policies, packing rules, consumer representation requirements,
producer dependencies, and bounded runtime choices were authored or selected.
Training coordinates producer startup/shutdown, readiness, admission, delivery,
and failure; selected input behavior may advance independently and owns its
internal live-source traversal, queue, selection, packing, and continuation
state. Cache-production traversal remains with caching coordination; the
cache-backed input boundary is reconciled at tasks 2.7/4.1.
An ordinary computation receives admitted values, not an unchecked blocking
read hidden inside its operation. This division does not require input to be
an optional capability or every provider-internal task to be a graph node.

Readiness and admission are different: a produced value can be available yet
wrong for the requested work or unacceptable under the current producer,
participant, or representation dependencies. The handoff preserves the work
and logical-example identities, provenance, and boundaries or masks that the
selected consumer, accounting policy, or restoration claim needs. Dataset
records, logical examples, packed sequences, and physical microbatches are not
one universal `batch`; the concrete payload remains selected domain behavior.
Out-of-order completion cannot pair a value with the wrong request. A bounded
change of accepted selection or packing policy is ordinary runtime state
evolution, while a change outside those bounds follows the applicable governed
transition or new fulfillment path.

Selection, production, readiness, admission, handoff, action consumption,
per-unit optimization outcomes, progress recording, and snapshot durability
are distinct events. Training's run-level accounting must be able to correlate
the relevant input delivery, action attempt, and unit outcomes without making
the input provider own optimization results or treating one action as an atomic
optimizer transaction. When exact same-run restoration is claimed, a
coordinated cut must include the input owner's continuation state, accepted
dependency revisions, handed-off but unfinished work, and an explicit
replay/skip policy consistent with the other run-state owners. A saved cursor,
provider pause, or unverified revision label alone does not prove that claim.
The concrete handoff, attempt-correlation, snapshot, and recovery types remain
G3–G5 decisions; the executable experiments establish these boundary needs,
not their final Python API.

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

#### G4.3 product coverage and use

SDXL, SD3, adapter, and learned-sidecar products are migration examples, not
four product categories or the limits of persistence. A declared product
selects what its intended consumer needs, not everything loaded, optimized,
or used during training. It may select raw or EMA state and omit training-only
teachers, discriminators, losses, and optimization machinery. Selection does
not require a universal EMA slot or a new participant for every saved value.

The declaration specifies the product's supported use and required state and
information. The plan embeds or explicitly references the weights, buffers,
normalization statistics, codebooks, construction configuration, installation
relationships, and implementation/representation dependencies needed for that
use. Capability-owned contributions remain owned by their domain. This does
not require pickling algorithms, embedding every dependency, or promising
compatibility with arbitrary consumers.
Implementation dependencies here are requirements for reconstruction/use,
such as architecture/configuration versions or required custom operations;
they do not automatically include the training Python class/object identity,
wrapper, or exact compiled artifact.

Consumer dependencies, training/preparation dependencies, and historical
provenance are distinct. A teacher can remain in provenance without being
required at inference. An adapter's base dependency must describe the state
it actually requires. In joint base-plus-PEFT training, an adapter-only product
does not contain the base updates; the product's embedded coverage or an
identified compatible external dependency must account for them when required
for the declared use. The original load path alone does not establish that fact.

#### G4.3 capture, export, and publication exchange

- **Declaration and request:** accepted product behavior names coverage,
  supported representations and transformations, required versus optional
  constituents, dependencies, consistency, and completion conditions. A
  lifecycle request selects a declared product, destination, and permitted
  choices; it does not rediscover coverage from current wrappers or mode.
- **Plan and readiness:** persistence coordination resolves a product-specific
  plan under the accepted obligations and source dependencies. Known semantic
  incompatibilities are rejected during fulfillment; concrete source or
  backend obligations are checked when authoritative evidence exists. No
  serializer invents another validity rule. Training compatibility alone does
  not prove that merging, pruning, or quantizing a product is supported.
- **Stable capture:** infrastructure coordinates the boundary and ranks;
  contributors provide their declared state and consistency evidence. Capture
  preserves the selected raw/EMA state, topology, relationships, dependencies,
  and lineage at that boundary. A borrowed live tensor or object reference is
  not automatically a stable capture. Required reads must remain protected
  through actual backend completion, not merely a Python call returning.
- **Export work:** selected implementations transform and serialize the
  protected source. Export is not permission to change canonical run state.
  Any temporary live-state effect must be explicitly permitted, coordinated
  against conflicting use, and restored before use resumes. A failed cleanup
  gates unsafe use without erasing known artifact output. Canonical changes
  require the existing run-authority transition path.
- **Actual result:** writing, local resource completion, remote publication,
  and observation have distinguishable outcomes where applicable. Results
  associate actual resources with semantic members and report required or
  optional omissions, known partial effects, pending/uncertain completion,
  and failures. Checksums identify what was measured; a tensor/model hash is
  not silently reported as a whole-resource checksum. An absent result or a
  raised exception does not establish that no output was produced.
- **Resource lifetime:** persistence infrastructure uses the actual resource
  set for retention and cleanup, respecting sharing and outstanding reads,
  including asynchronous publication. It must not delete or overwrite a
  resource still needed by accepted pending work. This is not a guarantee of
  permanent availability for every external dependency.

Freshness is checked when acquiring current source state. Once a stable
capture satisfies the accepted product obligations, later serialization or
publication can finish after the live arrangement changes. The result still
describes the captured state, not the arrangement current at completion. A
request requiring the latest state can impose a stronger accepted condition;
historical captured products do not become current-state projections.

Required shards, manifests, reconstruction information, and dependencies must
satisfy the product's declared completion conditions before it is reported
complete. Independently valid members can be reported even when another fails.
An external dependency satisfies completion according to the accepted policy:
embedded state, an available/resolvable resource, or a stable compatible
external reference can each be a supported requirement. Under a reference-only
policy, an adapter product can be complete even if its base is currently
unreachable. Completion does not imply self-containment, current usability
without resolving dependencies, or continued availability of external resources.
Neither multi-file writing nor remote publication is assumed transactional,
and product success remains independent of a simultaneous runtime snapshot.

#### G4.3 research checks and limits

| Recorded case | Requirement on the same persistence exchange |
| --- | --- |
| Student export and EMA autoencoder export (`research/teacher-student-distillation.md`, `research/autoencoders-vae.md`) | Select the intended state variant; omit training-only networks without dropping the state needed to use the selected product. |
| VQ codebook, latent normalization/packing, and control installation (`research/autoencoders-vae.md`, `research/conditioning-and-control-networks.md`) | Preserve required non-parameter state and construction/representation/installation information, without making every constituent a participant. |
| Joint direct-plus-PEFT training and low-precision merge (`research/precision-and-quantization-training.md`, accepted PEFT composition) | Describe the actual base dependency and export transformation; separate training compatibility from product compatibility. |
| Distributed members and resources (`research/distributed-training-execution.md`) | Preserve logical coverage across physical shards and report missing required resources, rather than promote rank-local fragments into semantic identities. |
| Dense-to-MoE and installed attention changes (`research/model-surgery-and-staged-topology.md`) | Identify the captured topology and construction meaning; do not relabel old captured state using the later live stage. |
| Compound control and AR/NAR adapter products (`research/conditioning-and-control-networks.md`, `research/video-audio-training.md`) | Permit selected cross-component coverage and external dependencies without SD-shaped roles or treating every preparation dependency as an inference dependency. |

Current writers provide additional migration evidence: SDXL combines components
or uses directory packaging; SD3 emits several files; VeRA includes shared/local
state and reconstruction settings; learned sidecars are selected domain
products. These implementations do not demonstrate stable distributed capture,
complete publication reporting, or the wider cases above. G5 must establish
those guarantees for each production implementation it deliberately supports.

Persistent dataset indices, cached representations, generated experience, and
unfinished producer work retain their owning capability's meaning. Shared
storage or publication infrastructure does not turn them all into trained-model
products. Exact continuation of such work and coordinated snapshot recovery
remain G4.4; metadata consumption remains G4.5. Concrete API, capture, backend,
and publication mechanisms remain G5, not new language constructs at this gate.

#### Restoration remains a separate exchange

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
| Teacher/student distillation | Roles do not determine participant count; separately declared participants stay distinct despite shared sources, while asymmetric execution, relationships, trajectories, optimization participation, and products remain explicit |
| Pixel or non-latent representation | No fake VAE or latent-shaped universal field; representation-specific behavior remains selected implementation |
| Video, audio, or other tensor shapes | No universal image-batch or fixed-axis assumption; temporal, channel, sequence, and representation meanings remain selected behavior |
| Added side network or LLM-containing model | Additional executable/trainable participant without new universal Trainer slots |
| Scheduled and observation-adaptive behavior | Runtime schedules, feedback-driven state, bounded decisions, observations, persistence, and restoration remain dynamic after authoring ends |
| Custom structured operation | Replaces maintained decomposition while keeping standard Trainer authority |
| Strong imperative research region | Requests actual backward, gradient, or advancement authority through an explicit supported profile; acceptance and rejection are testable |
| Replacement/preparation failure | Stale optimistic candidate, destructive failure, relationship invalidation, backend-group rebuild |
| Artifact and exact restoration | Product identity and lineage remain distinct from same-run authority/participant restoration |

### G2 worked case 1: ordinary SDXL fine-tune

This is a trace of one **accepted direct-parameter SDXL run**, not a proposed
universal SDXL-shaped interface. Its authored choices include the SDXL model
and objective, selected direct-training subjects (denoiser and, if requested,
text-encoder parameters), optimization policy, and any validation, sampling,
and full-model product requests. Contract-guided fulfillment settles what is
knowable from those choices and leaves explicit obligations for facts that
only loading or backend preparation can establish. The completed authoring
object is not the Trainer's per-step collaborator.

| Lifecycle use | Current owner and exact evidence | Target exchange: changing input, owned dependency, result, and failure |
| --- | --- | --- |
| Materialize and bind | `library/strategies/sdxl/loading.py:24–95` loads two text encoders, VAE, and UNet into `LoadedModelComponent`s; it also keeps source format, `ckpt_info`, and `logit_scale` on the strategy. `library/training/runners/trainer.py:417–425,970–975` currently invokes that loader and keeps compatibility component views. | Training lifecycle coordination requests the selected SDXL loader using accepted source intent and current authority revision. The loader returns candidate components and evidence, including product-relevant source facts; the run authority checks obligations and publishes participant bindings/relationships once. Load or evidence failure leaves the prior published state intact; a stale candidate cannot replace it. Neither Trainer slots nor the authoring object become the binding store. |
| Prepare and realize optimization | `library/training/modes/finetune_mode.py:84–158,190–273` selects live parameters, changes `requires_grad`, constructs optimizer groups, and wraps trained models plus optimizer/scheduler (or a DeepSpeed composite). `library/training/phases/model_prep.py:53–112` and `library/training/phases/optimizer.py:61–185` split casting, scheduler, backend, and restoration work across phase/mode/strategy calls. `library/strategies/sdxl/model_preparation.py:12–53` supplies CLIP-specific preparation and the TE1-tail trainability constraint. | A complete, revision-pinned preparation job uses the current bindings, accepted trainable subjects/constraints, required execution participants, and backend policy. Selected SDXL/CLIP operations supply their specific constraints; Trainer optimization resolves live parameters and builds its runtime; training preparation coordinates casting, wrapping, and route publication. The VAE and untrained encoders still need execution-ready views even when absent from optimizer membership. The successful result publishes coherent prepared views and optimization state; allocation, compatibility, or stale-revision failure publishes no mixed current runtime. DeepSpeed's composite handle is an execution representation, not replacement participant identity. |
| Execute one training step | `library/strategies/sdxl/diffusion.py:33–242` performs latent/conditioning preparation, DDPM or RF noise and prediction, and loss computation. `library/training/phases/training_loop.py:461–618` supplies batch/step, invokes `process_batch`, updates the objective from observations, applies a loss modifier, and owns synchronization, backward, clipping, optimizer/scheduler stepping, zeroing, progress, and triggers. | Trainer supplies the current batch, time/step coordinates, prepared participant routes, and approved objective/operation state. Accepted SDXL representation, conditioning, objective, and prediction behavior performs model-specific computation and returns the distinct values needed for optimization, observation, and accounting, plus declared owned-state effects. Trainer performs the standard-profile mechanics. A computation failure cannot silently advance optimization or publish an undeclared canonical transition; a Trainer/backend failure follows the retained mechanics' failure rule. This does **not** prescribe `process_batch()` or one loss-return protocol for all models. |
| Validate when requested | `library/training/phases/training_loop.py:133–183` selects triggered work and temporary evaluation; `library/strategies/sdxl/validation.py:106–196` currently mixes seeded RNG handling, validation-dataloader traversal, averaging/recorder updates, and selected `process_val_batch` computation. | Training validation coordination receives a trigger, input cursor/RNG policy, current evaluation-ready routes, and accepted evaluation behavior. It owns traversal and aggregation; SDXL computation remains selected behavior. It publishes a typed observation and validation state only for completed evaluation. Failure restores temporary evaluation/RNG projections and does not report a successful observation. |
| Sample when requested | `library/training/phases/training_loop.py:133–183` and `library/training/phases/orchestration_helpers.py:20–101` schedule and enter temporary evaluation; `library/strategies/sdxl/sampling.py:67–150` currently repeats trigger checking, unwraps/moves components, constructs the SDXL pipeline, calls shared sample traversal, and restores devices in `finally`. | Sampling coordination receives the accepted request/destination and current prepared predictor, conditioning, and representation routes. It owns trigger, traversal, temporary projections, output accounting, and cleanup; the selected SDXL generation operation owns pipeline/generation semantics. The result names outputs actually produced. Failure restores temporary projections and reports partial external writes rather than inventing complete output. |
| Persist a trained product | `library/training/phases/training_loop.py:133–183` triggers step saves; `library/training/runners/trainer.py:471–507` delegates serialization to the mode; `library/training/modes/finetune_mode.py:359–403` delegates full-model serialization to the strategy; `library/strategies/sdxl/checkpointing.py:49–121` reads whole-Trainer fields, unwraps components, converts/writes SDXL formats, and optionally uploads. | Persistence coordination receives a declared full-model product, a coherent participant/relationship projection, destination and consistency policy, and the source facts needed for SDXL conversion. Domain serialization performs conversion; infrastructure coordinates timing, writing, upload, and actual-result reporting. A write/upload failure records actual resources and omissions, not a fictitious complete product. This does not claim that a trained artifact is an exact runtime snapshot. |

Two dynamic paths make the execution requirement concrete. Before a batch,
`library/training/phases/training_loop.py:473–484` advances the objective at the
current step; `library/timesteps/runtime.py:135–143` changes its active timestep
range when a scheduled threshold is crossed. Separately, the SDXL computation
returns `sampling_loss` distinct from its weighted/accounting loss
(`library/strategies/sdxl/diffusion.py:204–242`); the loop supplies that
observation to the objective before applying its backward loss modifier
(`library/training/phases/training_loop.py:516–535`). The timestep runtime
updates its sampler from those observations and uses the updated state on
later batches (`library/timesteps/runtime.py:145–190`,
`library/timesteps/samplers/adaptive_log_snr_sampler.py:6–115`). Thus accepted
objective behavior retains its own advancing, restorable state after strategy
authoring ends. Trainer supplies step coordinates and routes observations; it
does not own the SDXL objective algorithm or freeze either policy at
fulfillment. The accepted state dependency and freshness rules for scheduled
and feedback-driven operations are completed across the remaining G2 cases,
not inferred from this single sampler.

This case establishes the required distinctions, not a final operation API:
participant membership versus optimizer membership, authority binding versus
backend representation, selected model/algorithm behavior versus Trainer
mechanics, and computation/observation/accounting/product results. SD, SD3,
non-latent and research cases must still test whether the eventual vocabulary
expresses those distinctions without SDXL fields or an active-strategy call.

### G2 worked case 2: SDXL PEFT and joint direct training

Consider two SDXL runs built from the same maintained strategy definition:
one trains a selected PEFT method alone; the other trains that adapter **and**
selected base-model parameters. This is a target-architecture test, not a claim
that today's `AdapterMode` supports the joint run. The current mode constructs
and attaches an adapter, then freezes the denoiser and text encoders
(`library/training/modes/adapter_mode.py:115–208`). Its working PEFT path is
evidence for required behavior, not the desired ownership or a reason to keep
the two training treatments mutually exclusive.

1. **Author and check the choice.** The SDXL strategy deliberately adopts the
   maintained PEFT integration and declares the methods it currently supports.
   For this run, construction selects one supported method, its method-local
   settings, an adapter participant, intended host relationships and semantic
   target intent, training subjects, continuation intent, and desired products.
   Fulfillment rejects an undeclared method or known unsupported target or
   overlap before realization. Today
   `library/adapters/methods/peft/config_resolution.py:131–191` selects a
   registered method from config branches and builds runtime settings, while
   `library/training/modes/adapter_mode.py:115–131` makes that choice inside the
   mode. Repository registration is evidence of availability, **not** automatic
   support by every authored strategy.

2. **Resolve hosts before constructing adapter state.** After the SDXL hosts
   materialize, governed PEFT realization resolves authored target intent
   against one current authority-qualified host projection. The result carries
   host participant/component identity, component-local paths, source
   revisions, and scoped live target views. Shared target references describe
   *what was resolved* without deciding PEFT selection or optimizer policy.
   Today `library/optimization/grouping.py:197–228` instead chooses host scope
   from learning rates and `library/adapters/runtime/targets.py:108–232`
   traverses selected components into module targets. The reusable
   component/path/ref shape is useful; optimization's ownership of semantic
   host selection and unqualified live-object identity is not. Stale host
   structure or unsupported targets reject the candidate rather than letting
   a method silently choose different targets.

3. **Realize and attach the selected method.** The selected method consumes
   those resolved targets, constructs or loads its own state, and returns
   adapter trainables with source-target provenance. The run authority accepts
   the adapter binding and the host-effect relationship; attachment changes
   the host's effective execution route without creating a new host participant
   or requiring an independent adapter forward route. Today
   `library/training/modes/adapter_mode.py:132–179` combines construction,
   `apply_to()`, export-weight loading, and Trainer-field assignment.
   `library/adapters/runtime/build.py:9–35` dispatches to registered methods;
   LoRA builds target-local modules
   (`library/adapters/methods/peft/lora/runtime.py:175–225`), whereas VeRA also
   has a shared projection bank
   (`library/adapters/methods/peft/vera/runtime.py:212–299`). Shared PEFT
   integration owns the common participant, target, relationship, lifecycle,
   and product meanings; each method owns its actual settings, algorithm,
   state shape, constraints, and serializer. Attachment or loading failure
   cannot advertise a bound, active, ready effect; an in-place host mutation
   follows D8's prior-guarantee withdrawal and recovery rule, not a fictional
   rollback.

4. **Resolve optimization after realization.** In the PEFT-only run, accepted
   optimization subjects are adapter-owned parameters while hosts remain
   executable but frozen. In the joint run, selected base-parameter
   substructures and adapter-owned parameters are both subjects; host
   participation in an adapter effect does not by itself imply either set is
   trainable. Trainer optimization resolves the live parameters, aliases,
   non-overlap, logical/execution groups, and backend preparation from the
   accepted subjects and the adapter realization result. It does not discover
   adapter targets. Today `library/adapters/shared/trainables.py:10–88` carries
   trainable refs and source-target provenance, and
   `library/optimization/grouping.py:258–308` groups them only after the
   adapter exists; `library/training/modes/adapter_mode.py:211–290` still owns
   grouping and backend installation. The eventual choice of one or multiple
   coordinated optimization units and advancement policy belongs to G3, not
   this case. Unsupported parameter overlap fails under the standard profile.

5. **Execute and handle specialized lifecycle behavior.** SDXL's selected
   computation still invokes the host's current route, through which an active
   adapter effect operates; it does not ask an `AdapterMode` what to do each
   step. Trainer retains standard timing and optimizer mechanics. A declared
   method-specific action, such as maximum-norm regularization, runs at its
   accepted lifecycle point and reports its effects/metrics without whole-
   Trainer access. Today `library/training/modes/adapter_mode.py:318–371`
   supplies step, train/eval, clipping, and maximum-norm hooks, while
   `library/strategies/sdxl/diffusion.py:33–142` invokes the denoiser with
   the currently attached behavior. Train/eval and backend projections are
   Trainer-coordinated; the specialized operation remains adapter-domain
   behavior, not a new generic Trainer algorithm.

6. **Keep product, new-run loading, and exact resume separate.** An adapter
   export selects adapter state, relevant host relationships/dependencies,
   method representation, and actual emitted resources; a full or merged
   product is available only when explicitly supported and selected. In a
   joint run, an adapter-only export does not silently claim to include the
   changed base weights needed to reproduce that run. Loading an adapter
   artifact into another run creates a new adapter participant with lineage
   to the artifact and source state, whether the artifact continuation policy
   is strict or a supported initialization mode. Exact same-run restoration
   instead restores the authority's identities and revisions plus host,
   adapter, optimization, backend, progress/input, and method-owned
   continuation state needed to resume. Today
   `library/training/modes/adapter_mode.py:51–72,132–174,306–314,373–410`
   mixes continuation, export, and resume hooks;
   `library/adapters/shared/state_io.py:45–113` already separates export I/O
   from Accelerate checkpoint hooks. Neither current weight-load path proves
   exact restoration, and the adapter-only hook is insufficient to define
   joint-run completeness.

The governed meanings stay separate: the adapter has its own participant
identity; its source artifact supplies **lineage**, not that identity; each
host effect has a relationship identity and lifecycle; prepared host routes
and unwrapped/product views have their own freshness; method-specific
operations and selected product capabilities have bounded authority; and
Trainer optimization owns the selected parameter membership and advancement.
An adapter's target-local modules or VeRA's shared bank do not automatically
become extra participants. None of these meanings is supplied by a replacement
`AdapterMode` or by a single `train_adapter` Boolean.

### G2 worked case 3: SD and SD3 variation (limited check)

This comparison checks whether the preceding cases accidentally assumed
SDXL's anatomy. All three families are still diffusion models: passing this
check is **not proof of a general Trainer**. The later non-latent, non-image,
compound-participant, and research cases must challenge the abstraction more
directly, and implementation/tests must establish that it works.

| Difference in current code | Evidence | Requirement placed on the target, not a copied implementation |
| --- | --- | --- |
| Component count and availability | SD loading binds one text encoder, VAE, and UNet (`library/strategies/sd/loading.py:25–61`). SD3 declares three encoder positions plus VAE and MMDiT; a position may contain `None` when its source is absent (`library/strategies/sd3/loading.py:22–69`, `library/models/sd3/loader.py:385–428`). | Participant and preparation requests use the accepted, keyed participants and their readiness; Trainer does not expose one or two universal text-encoder slots. An allowed absence, an unbound promise to materialize later, and a failed required load have different meanings. |
| Conditioning and prediction | SD conditioning is a list of hidden-state tensors (`library/strategies/sd/conditioning.py:11–94`); SD3 uses a named CLIP/T5 payload with pooled output and masks (`library/strategies/sd3/encoding.py:86–140`, `library/strategies/sd3/conditioning.py:12–106`). Their denoiser calls differ (`library/strategies/sd/denoiser.py:9–70`, `library/strategies/sd3/denoiser.py:9–73`). | Accepted representation, conditioning, and predictor operations own those payloads and calls. Trainer supplies prepared routes and changing inputs, without a universal `text_encoder1/2`, CLIP-mask, pooled-vector, or diffusion-specific batch field. |
| Objective compatibility | Current config validation requires DDPM for SD1/2 and rectified flow for SD3 (`library/config/config_validation.py:575–593`); SD3's batch computation directly uses the flow objective (`library/strategies/sd3/diffusion.py:50–121`). | The training contract makes supported model/objective combinations visible during strategy authoring and rejects a known invalid selection at fulfillment. Trainer does not choose an objective by branching on model family. This records current support, not a permanent rule that no future SD implementation may support another objective. |
| Component-specific preparation | SD's preparation delegates to CLIP helpers (`library/strategies/sd/model_preparation.py:12–46`). SD3 distinguishes CLIP from T5, currently rejects T5 FP8 preparation, and rejects training T5 while caching its outputs (`library/strategies/sd3/model_preparation.py:13–70`). | Accepted component constraints and selected preparation operations govern these differences. A known unsupported authored combination fails during fulfillment; realized/backend-specific incompatibility fails at its evidence checkpoint. Generic Trainer preparation still coordinates the coherent job. |
| Supported trained products | SD does not override the base full-model save method, which raises `NotImplementedError` (`library/strategies/sd/checkpointing.py:15–53`, `library/strategies/base/contracts.py:302–336`). SD3's full-model writer currently supports only safetensors (`library/strategies/sd3/checkpointing.py:70–120`). | A strategy offers only products and formats its selected implementation supports. A requested unsupported product is rejected before Trainer starts; a serializer failure after acceptance is a separate runtime failure. Adapter products, where declared, are not inferred from the existence of a family metadata resolver. |
| Deferred materialization | The generic loading contract mentions a deferred module and exposes a lazy-load hook; training preparation checks for a missing denoiser (`library/strategies/base/contracts.py:34–74`, `library/training/phases/model_prep.py:30–51`). The inspected SD3 loader itself returns an MMDiT and may return absent encoders; it is **not** evidence of a working deferred-MMDiT path (`library/models/sd3/loader.py:435–481`). The deferred SD3 denoiser is explicitly a design scenario in `docs_design/models-strategy-trainer/strategy_contract_exchange_design.md`. | An accepted deferred participant remains declared and unbound until an allowed readiness checkpoint; later loading materializes the same identity. A phase that requires it cannot proceed on `None`, while an explicitly allowed earlier phase may. This is a target obligation to implement and test, not behavior credited to today's SD3 path. |

At this limited level, D5's ownership categories can describe SD and SD3
without adding their encoder counts, conditioning fields, objective types, or
serializers to Trainer. That checks for one kind of SDXL leakage; it does not
establish that the eventual executable representation can handle other model
or data domains. G2 cases 2.4–2.6 must do that work before G3/G5 choose code
shapes.

### G2 worked case 4: non-SD and lifecycle pressure

The external implementations collected under [research](research/README.md)
are pressure cases, not another source of contract authority. They show work a
strategy might author *if* the intended Trainer, recognized pipeline
capabilities, and resulting contract permit it. "Recipe" in those sources is
not a Trainer-owned runtime type. The goal is one Trainer engine that can
execute different accepted training meanings through selected behavior, not a
separate Trainer or wrapper per model or method. Representability here does
not claim that every cited model, algorithm, backend, or artifact format will
ship in the first migration. Unsupported selections must be rejected by
contract-guided fulfillment unless an explicitly supported research extension
grants the required authority; they must not be silently approximated by
generic Trainer branches.

| Pressure case and evidence | Concrete behavior | Constraint on the accepted run |
| --- | --- | --- |
| Teacher/student: [distillation cases](research/teacher-student-distillation.md), including [DeiT's loss](https://github.com/facebookresearch/deit/blob/main/losses.py) and [progressive diffusion distillation](https://github.com/openai/consistency_models/blob/main/cm/train_util.py) | DeiT executes a distinct frozen CNN teacher and trainable Transformer student; other cases use one physical model in two roles or an offline-produced training corpus with no live teacher. The compared values may be classes, tokens, or denoising trajectories, not one universal logits tensor. | The authored strategy identifies actual participants, execution roles, comparison behavior, and required representation compatibility. A role label does not create a participant; two independently evolving declared participants do not become one because they share source weights. Frozen live teachers still need execution preparation, whereas an offline teacher is a data-provenance dependency rather than a current prepared route. No `teacher_model` Trainer slot is required. |
| Direct and packed pixel training: [pixel-space cases](research/pixel-space.md), including [PixelFlow's training path](https://github.com/ShoufaChen/PixelFlow/blob/8805204d1be8df22382280959b3063974ff3d77d/train.py) | Direct image diffusion trains without a VAE. PixelFlow starts with pixels but presents variable-length packed patch sequences from different resolutions to one model; its per-example stage is not a run-topology transition. | Representation and objective are separate selected meanings. Neither a VAE, a latent tensor, BCHW model input, fixed sequence length, nor one global "stage" is a core Trainer field. Accepted input dependencies and operation outputs carry the needed shapes/axes without asking Trainer to implement pixel or latent mathematics. |
| Video/audio: [temporal and multimodal cases](research/video-audio-training.md) | MiniMax-H3 jointly predicts video and audio streams with different representations and related but nonidentical noise times; YuE2 combines discrete-token autoregressive loss and continuous audio-flow loss with an explicit gradient boundary. Some examples have no audio for a particular sample. | Accepted behavior owns modality axes, masks/absence, time mappings, model calls, and loss composition. The Trainer's generic batch/step and standard optimization mechanics must not imply one image tensor, one timestep, one prediction, or one loss producer. Preparation membership includes frozen representation producers when needed, even if their output is later cached; exact step/optimization exchange remains G3. |
| Added side network: [control-network cases](research/conditioning-and-control-networks.md), including [ControlNet's training implementation](https://github.com/kohya-ss/sd-scripts/blob/main/train_control_net.py) | A trainable control branch supplies effects to a frozen base model. The base weights receive no optimizer update, yet the base forward remains on the differentiable path from loss to control branch. Control preprocessing, side state, injection, and exported side artifact have different boundaries. | Trainability, gradient participation, execution readiness, and artifact membership are distinct. The accepted strategy supplies the selected control/injection behavior and relationship; Trainer optimization selects only declared trainables and prepares every needed executable. A side network is not a universal Trainer slot, and an injected submodule is not automatically a new participant. |
| In-run stage transition: [progressive distillation](research/teacher-student-distillation.md) and its [training loop](https://github.com/openai/consistency_models/blob/main/cm/train_util.py) | At a scale boundary the implementation copies student weights into the teacher, rebuilds the optimizer, resets EMA state, and restarts stage-local progress. These are observable coordinated changes, not ordinary per-batch branching. | An accepted stage policy may request a governed transition. The run authority evaluates participant/binding/relationship continuity under current obligations; Trainer optimization owns optimizer replacement; accepted EMA behavior owns its state. The coordinator checks which prepared routes and dependencies remain valid and republishes a coherent current state before the next dependent step. Copying weights does not itself decide whether a participant reference is preserved or replaced. |

Together these cases test three levels of commitment. The common Trainer and
contract must **support** their shared meanings—explicit participants and
roles, selected computation, varied representations, execution versus
optimization membership, declared effects, and coherent current-state
transitions. The design must **prepare for** implementations that add new
model-specific operations, backend requirements, or staged changes without
adding another Trainer authority. It must remain **neutral** toward algorithms
and services not selected or implemented by this repository; neutrality means
no image-shaped barrier and clear unsupported-authority rejection, not a
promise that every external implementation runs unchanged.

These sources do not select a graph, region, schedule, or Python API. They also
do not establish our participant identity from upstream object copies,
unloading, or class names. G2.5–G2.6 test custom and imperative execution and
choose the smallest common meanings; G3 completes multi-output optimization
and stage-transition mechanics; G4 handles caching and product/restoration
details. This case remains a design pressure test until those boundaries have
been checked, not a claim of production support.

The common representation can cover these cases without adding teacher,
student, VAE, audio, or side-network slots to Trainer: independently managed
state has participant identity; each use names its required prepared view;
selected operations carry representation, axis, and comparison meaning;
optimization names only its accepted subjects; and products select their own
semantic coverage. For example, a student-only product may retain a teacher
source dependency without embedding a live teacher, while a side-network
product may depend on a frozen base without claiming to contain it. A stage
transition changes current authority and preparation state before dependent
work resumes; it is not inferred from an upstream weight-copy instruction.

### G2 worked case 5: adaptive operation and stronger research authority

An accepted custom operation may replace several maintained model/objective
calls while leaving the standard Trainer mechanics intact. Consider one that
selects its next sampling policy from prior loss observations. Its selected
implementation receives current inputs, coordinates, prepared routes, and its
own state; it returns computation values, optimization input, observations,
and declared state changes. Its state has an accepted initialization rule and
restoration contribution. The operation does not receive the whole Trainer or
optimizer handles as its normal interface, and the authoring object is not
consulted each step. This remains standard-profile behavior even though its
computation and state evolve during the run.

For a deliberately stronger counterexample, consider a two-pass operation
that requests control of backward and a temporary parameter perturbation
before Trainer's final optimizer advancement. Under the standard profile,
that request is rejected: declaring an ordinary computation operation does
not transfer backward or parameter-mutation authority. If the active contract
offers a suitable research extension, fulfillment may accept an explicitly
bounded region whose profile transfers the two backward passes and temporary
perturbation plus intermediate gradient handling to its executor, retains
final advancement with Trainer, names its allowed parameters and required
state, and prevents Trainer from performing those transferred actions in the
same accepted action. If no such profile is supported, the
same filing is rejected rather than run through a hidden callback. Failure
while parameters are perturbed cannot be described as harmless merely because
the region intended to restore them; uncertain live state must stop dependent
work and follow the accepted recovery rule.

This tests ownership transfer, not a commitment that the first production
contract supports this particular method. Scoped access and declared effects
make ordinary misuse harder and allow checks at the exchange boundary, but
arbitrary selected Python is not a sandbox. Fulfillment can reject an
unsupported authority request; realized gradient compatibility and hidden
mutation may require runtime evidence or trusted implementation discipline.

#### G2.5 ownership and recovery resolution

The adaptive operation above remains a **standard-profile** operation: its
accepted owner initializes its sampling state from the declared rule, consumes
declared loss observations, and contributes that state, relevant RNG, and its
dependency revisions to exact restoration. It does not gain backward or
optimizer authority by being stateful. If an observation or state update may
have occurred before an operation fails, the run cannot infer from a missing
result that the owner's state is unchanged; it stops dependent work unless an
accepted owner-specific recovery rule establishes a valid state. The common
freshness and restoration rules are in G2.6 below.

The stronger test is a *possible* two-pass perturbation extension, not a
promise to ship SAM. The [original SAM method](https://arxiv.org/abs/2010.01412)
and a [concrete PyTorch example](https://github.com/davda54/sam#usage) show
why this case needs two forward/backward passes, temporary parameter edits,
and clearing or replacing first-pass gradients before the second pass. For
this **chosen test split**, the accepted region does the two passes, scoped
temporary edit, intermediate gradient handling, and restoration of the
unperturbed parameters; Trainer retains final clipping, optimizer/scheduler
advancement, and final cleanup. Trainer may perform initial clearing before
the region starts. A wrapper that can only perform restoration *and* optimizer
advancement inside its own `step()` does **not** fit that split unchanged: it
needs a different supported grant that also transfers advancement, or is
rejected. G3 defines the concrete advancement policy and exchange.

| Checkpoint | Required meaning for this test split |
| --- | --- |
| Fulfillment | The active contract already offers the extension profile; the authored region requests the exact two-pass backward, gradient-handling, and temporary-edit authority, names its unit and parameter substructure, result/effect bounds, state contributors, and recovery claim. Standard-profile filing is rejected. The region's executor is accepted runtime behavior, not the strategy's authoring role. |
| Realization and readiness | The current prepared route, parameter projection, backend gradient/synchronization behavior, and exclusive-use requirement satisfy those accepted obligations. A syntactically accepted profile is not proof that a backend can perform the edit or gradient exchange safely. |
| First pass and edit | Trainer owns initial setup; the region owns its first backward, gradient inspection, bounded perturbation, and intermediate clearing/replacement. Trainer neither repeats these actions nor supplies an unrestricted optimizer handle merely because the region is imperative. |
| Second pass and handback | The region owns its second backward and proves the scoped parameters are again in the accepted unperturbed state before Trainer may clip or advance the due unit. Trainer then owns only its retained final actions and reports their own outcomes. |

Within the run's selected ownership profile, a region's grant is **scoped to
an accepted action and action phase**. Trainer's
initial or final zeroing and the region's *intermediate* gradient reset are
different actions; both cannot own the same reset. A coarse grant of
backward and temporary parameter editing without intermediate gradient
control is not enough for this case. The current `ExecutionProfile` experiment
checks only declared non-overlap of a probe-sized mechanic set. Its acceptance
of a two-pass declaration does not prove this full exchange, backend support, or
recovery. Unsupported access, duplicate ownership, or a missing restoration
claim is rejected during fulfillment when knowable; a backend-only conflict
fails at governed realization before the region runs. Arbitrary Python is
not a sandbox, so trusted implementations and narrow scoped access remain
necessary even after declaration checks.

The region's intermediate reset must also respect the accepted accumulation
and synchronization policy. It cannot erase gradients accumulated for the
same unit from earlier microbatches. Fulfillment must either accept a
specified two-pass accumulation boundary that preserves both passes' intended
contributions, or reject this split when ordinary accumulation is requested;
governed realization must verify the backend can honor the selected boundary.
G3 specifies the concrete policy, not a universal SAM accumulation schedule.

Temporary perturbation creates an exclusive execution interval over the
affected current views and parameters. Conflicting execution, preparation,
capability reads, and snapshots cannot observe its intermediate state. A
successful edit-and-restore does not replace the participant, rebind a route,
or revise the accepted optimization definition; ordinary final advancement
changes weights under the same current semantic ownership. If restoration is
uncertain, however, the affected current-use guarantees are withheld until
valid state is re-established. This resembles D8's safety response to
destructive preparation but is an **execution failure**, not a preparation
attempt or an automatic authority revision for every training step.

| Failure point | Honest continuation claim |
| --- | --- |
| Before the temporary edit | Gradients, RNG, input position, or operation-owned state may already have advanced. No blind retry follows merely from unchanged parameters; an accepted reset/discard rule would need to cover all of them. |
| During edit, second pass, or restoration | Parameters or gradients may be uncertain. Stop dependent work and withhold affected current-use guarantees. A region-specific cleanup may permit in-process continuation only if it proves restoration **and** accounts for every other changed state/effect; otherwise recover from a prior coherent snapshot. |
| After verified restoration but before Trainer advancement | The model view is clean, but gradients and owner state may not be. Trainer may advance only after the accepted handback checks; failure or missing evidence stops the action rather than silently retrying it. |
| During or after Trainer advancement | Use the G3 per-unit outcome/failure exchange. A returned, skipped, failed, or uncertain optimizer action is not an atomic action result and never implies numerical parameter change or rollback. |
| Process/rank loss at any point | The current attempt is not resumed mid-region. Restore the last coherent same-run snapshot and its accepted input/action position, or report exact continuation unavailable. |

For this minimal extension, an exact-restoration snapshot is taken only at a
quiescent, unperturbed boundary. It must cover accepted region definition and
grant, scoped participant/binding/route and unit revisions, persistent
region-owned adaptive state, model and optimizer/backend state, relevant RNG,
input and progress positions, and the status of any unfinished effects. The
temporary perturbation itself is *not* a durable participant state or a
trained-artifact member. Mid-region snapshots or automatic in-process replay
would require a separately accepted, demonstrably complete protocol; they are
not promised by this profile. G3 supplies the action/advancement result and
attempt correlation, and G4 supplies the coordinated snapshot mechanism.

### G2.6 synthesis: one run structure, selected Python work

The representative cases and the bounded [execution experiment](research/training-mechanism-sketch.md#executable-representation-check-bounded-g2g3-evidence)
support a **hierarchical semantic run structure**. At the run level it names
work with different lifetimes: repeated training actions, independently
progressing input production, requested capabilities, and governed
transitions. At an action boundary it connects selected computations, their
values, observations, state, and optimization inputs. The connections form a
graph of *meaning and coordination*, including feedback through versioned
state; they are not a demand to graph every PyTorch operation. Selected Python
implementations perform the model mathematics and may encapsulate internal
worker or backend work. This is the chosen semantic shape, not a commitment
to the experiment's `Action`, `Value`, or `PreparedRun` classes.

An **operation** is selected executable work with a declared boundary. A
**region** groups work that has its own scheduling, lifecycle, state owner,
failure boundary, or granted authority; it need not be a separate Python
class. An **action** is one accepted coordination boundary at which selected
computation offers results to the run and due optimization units. It is not
an atomic parameter-update transaction. The authored strategy explicitly
selects and connects these meanings; fulfillment checks them under the
pre-existing contract. Preparation binds the accepted structure to current
routes and backend/optimization runtime without changing its authored meaning.

| Meaning visible at the accepted boundary | Why the run needs it | What may stay inside selected behavior |
| --- | --- | --- |
| Work and lifetime | Name due actions, long-lived producers, capability work, and transition points without one universal step clock. | Internal model calls, worker queues, and physical microbatch schedules. |
| Values and dependencies | Distinguish value flow, required ordering, current participant/route/relationship dependencies, observation feedback, and versioned handoffs. A tensor shape or role label alone is not compatibility evidence. | Domain payload layout and mathematical transformation. |
| Readiness and admission | Distinguish a produced or prepared value's availability from its current acceptability under accepted obligations and dependency revisions. | Owner-internal waiting and construction details. |
| Changing inputs and bounded decisions | Associate admitted input and current coordinates/state with an accepted choice among known actions, policies, or representations. | The selected policy's algorithm and private buffers. |
| Owned state | Name its owner, initial source/rule, update inputs, dependency set, continuity rule, and restoration contribution. | The owner's internal storage format and update mathematics. |
| Results, effects, and failures | Distinguish computation values, addressed optimization inputs, observations, accounting, owner-state updates, external effects, and authority-governed transition requests; report the stage and possible effects of failure. | Model-specific result fields not consumed across the boundary. |
| Authority | State which Trainer mechanics remain standard-owned and which exact actions an accepted extension grants elsewhere. | An implementation's private control flow within that grant. |

Expose a relationship only when fulfillment, preparation, another state
owner, run coordination, observability, or restoration needs to judge it.
This rule keeps the structure small without allowing an opaque Python call to
hide a second training loop or optimizer owner. Scoped access and declared
effects support checking, but arbitrary Python with a live mutable object is
not sandboxed; the standard path relies on narrow interfaces and trusted
selected implementations, and an imperative path needs an explicit authority
profile. A declared gradient path may still need realized evidence.

The following are readable *meaning sketches*, not proposed builder APIs:

```text
Authored SDXL action                         Prepared repetition
  admitted image/caption input                 receive admitted input + current views
  -> selected latent and conditioning work     run pre-resolved selected computations
  -> selected predictor/objective              offer addressed loss to denoiser unit
  -> backward value + timestep observation     route observation to objective owner
                                                Trainer applies accepted optimization policy

Authored joint-unit action                   Prepared repetition
  one selected objective -> one loss            compute once; offer same addressed value
  matrix unit + other unit both due              to two distinct semantic units
  accepted shared-gradient/order policy         Trainer coordinates backward and reports
                                                each unit outcome separately

Authored stateful input policy               Run-level execution
  accepted sources, mixture/pack choices        provider advances on its own clock
  accepted consumer representation              training admits identity and dependencies
  bounded stage changes                         action receives model-ready value and
                                                required logical-example boundaries

Authored two-use model action                Prepared repetition
  one accepted backbone participant              invoke its teacher/base view without
  two selected invocation roles                  turning that role into a new participant
  explicit view and gradient requirements        invoke its adapted/student view with the
                                                required differentiable path
```

If choosing those views toggles mutable adapter state, the accepted runtime
must serialize incompatible invocations and restore the required view after
failure or stop dependent work if restoration is uncertain; view selection
alone does not revise the participant binding. Restoring the view re-establishes
its usability, not the success or recoverability of the failed action.

The current experiment pre-resolves action-local value wiring and executes
the SDXL-like and alternating actions with the same mechanism; a test-side
Trainer consumer handles one shared loss offered to two units. A separate
input provider progresses ahead and out of order, and another selected policy
changes stage and packs logical examples without changing the prepared action.
Those checks justify the boundary split above. They do **not** establish real
backend preparation, distributed agreement, an exact snapshot, or the final
authoring syntax. The prototype currently lives in
`library/training/execution.py` but is not imported by the active Trainer or
launcher; G5 decides whether to retain, relocate, or replace it. In
particular, the composition test has no run-level
reporter joining input delivery, action attempt, and per-unit outcomes; that
join is required by D10. The later [G3.3 whole-run experiment](research/training-mechanism-sketch.md#g33-check-one-candidate-language-spans-run-level-input-and-repeated-actions)
adds a test-only attempt report for that correlation; its production storage
and API remain G5 work.

#### Operation-owned state across change and recovery

Every stateful operation or region has an accepted owner and initialization
rule. If initialization needs a materialized participant or prepared route,
it waits for that evidence and readiness; neither Trainer nor the former
authoring role invents a value. The owner alone applies ordinary accepted
updates, including scheduled changes and feedback from observations. Those
updates do not by themselves rebuild the run structure or invalidate a
prepared route.

Each state definition declares the participant, binding, route, relationship,
observation-source, and other dependencies that determine its *meaning*,
separately from ordinary live inputs it is expected to observe. Without a
more precise valid set, its authority-state dependency is conservatively the
whole snapshot. A relevant dependency change makes dependent state unusable
until an accepted continuity rule has been applied: preserve it with evidence,
migrate it, reinitialize it from an accepted rule, or retire it. It is never
silently carried forward because a Python object or field still exists.
Compatible replacement may keep participant identity while invalidating
weight-dependent state; a route-only wrapper change need not reset state
that does not depend on that route. A relationship change is judged against
the declared relationship dependency. A new ordinary observation updates
state; a change to the observation producer's meaning or missing required
observation is a separate compatibility/readiness question, not an automatic
reset on every feedback value. A new participant incarnation needs an
explicitly accepted state transfer rather than identity-by-copy.
The producer owner publishes any changed source meaning; fulfillment checks
authored compatibility, and the receiving state owner applies its accepted
continuity rule against current realization/readiness evidence before new
feedback is routed. A structural source change also passes through the run
authority's transition boundary.

When a permitted structural transition changes accepted work, the authority
and the relevant state owners establish the new obligation/operation-state
mapping and current prepared view before dependent work resumes. D8's
replacement-only and destructive paths retain their different failure
guarantees. An authored interval in which old and new paths intentionally
coexist is an accepted current structure, not a half-published replacement.

For exact same-run restoration, each required owner contributes its current
state, accepted state-definition/arrangement revision, dependency evidence,
relevant RNG and progress, and any unfinished work or effect status needed
to continue. Trainer/pipeline restoration coordinates those contributions
with authority, input, optimization, and backend state into one coherent
recoverable cut. If a required contribution or accepted replay rule is
missing, the snapshot cannot claim exact continuation. Trained-artifact
products include operation state only when their own declared product meaning
selects it; a model-weight product is not a substitute for runtime state.
After a failed operation that may have changed owned or external state,
dependent work stops unless an accepted operation-specific recovery rule can
establish a valid current state. Uncertain parameter or optimizer effects do
not receive generic rollback or blind replay. G2.5 above specifies the
stronger region's recovery obligations for its chosen test split; G3 and G4
still owe the concrete advancement and coordinated snapshot exchanges.

#### Representation choice and hot path

| Candidate | Case result | Decision |
| --- | --- | --- |
| Flat per-step DAG | Makes SDXL value flow visible but cannot by itself represent long-lived producers, different clocks, governed transitions, or cross-step feedback. | Not the whole-run representation. |
| Fine-grained graph of every operation | Could expose dependencies but would force domain math, queues, and backend internals into an oversized common vocabulary. | Use graph-like links only at the accepted coordination boundary. |
| Regions alone | Give work a clear owner/lifetime but do not express value, freshness, observation, or handoff relationships between regions. | Keep regions with explicit typed relationships. |
| Schedule or callbacks alone | Select due work but hide its inputs, state dependencies, authority, and failure effects. | Use bounded selected policies to schedule accepted work, not as the whole contract. |
| Opaque lowered Python alone | Runs quickly, but an unchecked closure cannot supply fulfillment, preparation, restoration, or cross-owner evidence. | Lower already-checked work to Python for the hot path; retain its accepted description. |
| Hierarchical hybrid | Keeps run-level work and cross-owner links visible while selected Python implements domain algorithms and prepared actions pre-resolve repeated wiring. | Chosen semantic direction; G5 determines storage, classes, and lowering. |

Fulfillment checks static selections, permitted choices, dependencies,
authority, and state definitions. Governed realization checks concrete and
backend evidence. Repeated execution uses admitted changing inputs, current
prepared views, coordinates, and owner state; it invokes a selected due policy
within accepted bounds and already-resolved work. It must not traverse the
complete authored strategy, rebuild family anatomy, or repeat contract
negotiation each step. It **must** still check current freshness/admission
where state may change and run dynamic schedules or adaptive algorithms.
Performance and conformance of the eventual lowering remain task 3.7.

This semantic choice leaves the exact standard advancement policies and
attempt/result exchange to G3, capability-specific and coordinated snapshot
protocols to G4, and Python types/module placement to G5. None is license to
replace the hierarchy with an active strategy callback or to claim that the
experimental runner already implements the whole run.

## Design Gates Before Production Implementation

### G1. Complete Trainer consumption meanings

For each D5 row, establish the common Trainer-consumption frame: the categories
of request input, result, readiness evidence, canonical state effect, permitted
external effect, and failure that its eventual exchange must make explicit.
Trace each responsibility to one or more delta requirements without promoting
current family slots. Name the owner that derives each governed-realization job
and the coordinator of final publication across authority-owned and
Trainer-owned state; do not leave cross-owner atomicity in passive voice.
At the common-frame level, locate each row's result and state-effect categories
in the D5 run-state sections and name their responsible writers and projection
consumers. G2–G4 must finish that mapping for each concrete execution,
optimization, or capability result as those exchanges are specified. Do not
use the run-state aggregate as a reason to make the participant authority,
Trainer, or one new facade the universal writer.

G1 locates ownership, evidence checkpoints, and the common exchange questions.
It does not complete the domain-specific caching, validation, sampling,
persistence, or restoration protocols. G4 fills in those capability-specific
requests, results, readiness rules, effects, and failures after execution and
optimization semantics are settled.

### G2. Derive accepted execution from representative cases

Inspect SDXL fine-tune and adapter first, then the materially different SD and
SD3 paths and the strong custom/imperative cases. Select the minimum operation
or region meanings, dependency/effect vocabulary, and execution/ownership
profile rules. Do not choose the representation by analogy alone. A bounded
executable experiment may compare candidate Python shapes, but it does not
become the production acceptance boundary merely because its tests pass.

### G3. Complete optimization and step exchanges

Define supported standard advancement policies and the explicit extension
boundary. Construct one bounded whole-run language experiment from G2.6's
selected meanings before completing backend-flexible optimization
candidates/requests/results and the connection among accepted execution,
final optimization input, observations, state effects, and advancement. The
experiment tests a common coordinator over different accepted arrangements;
it neither selects the production Python API nor cuts over the active Trainer.
Preserve the settled unit identity, non-overlap, preparation, and publication
rules.

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
3. Construct the training-side run-state, realization, preparation,
   optimization, capability, and Trainer-consumption surfaces against explicit
   test-only accepted inputs. This is a dependency order for the code, not a
   requirement that every intermediate repository revision remain runnable,
   and it must not introduce a production bypass or an old-strategy adapter
   disguised as the new boundary.
4. Establish contract-guided authoring and strategy fulfillment that produces
   those same accepted inputs for one maintained SDXL path.
5. Cut over one coherent SDXL run path so the accepted arrangement and its
   authority become the only current-state source; remove the displaced mode,
   strategy callback, and Trainer projection writes for that path in the same
   milestone.
6. Add adapter behavior through the same accepted participant/relationship,
   optimization, persistence, and restoration boundaries; do not create a
   second adapter orchestration authority.
7. Migrate SD and SD3 with conformance scenarios proving that the core did not
   inherit SDXL cardinality or objective assumptions.
8. Complete capability and research-extension migration, remove compatibility
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
