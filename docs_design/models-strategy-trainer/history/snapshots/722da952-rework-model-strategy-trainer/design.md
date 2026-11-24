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

| Responsibility | Request and readiness | Successful result and state effect | External effect and failure boundary |
| --- | --- | --- | --- |
| Fulfill an authored strategy | Complete authored selections under the active contract/profile and every fact knowable before realization | Detailed acceptance containing one semantic arrangement, one authoritative run-specific obligation revision, and seeded run authority | No ordinary external runtime effect. Rejection returns detailed diagnostics and produces no Trainer-acceptable arrangement. |
| Materialize and bind | Accepted declarations, relationships, lifecycle permissions, source intent, obligations, and expected revisions | Training lifecycle coordination derives the request from accepted obligations and current state; a selected domain producer returns a candidate plus typed evidence; the run authority alone accepts and publishes a new or revised binding | Loading or construction may perform declared I/O before publication. Failure or stale evidence publishes no candidate; destructive replacement follows the separately accepted transition rules. |
| Prepare runtime | One coherent revision-pinned projection of every required participant, route/view, constraint, joint group, mutation permission, and obligation | Training-side preparation coordination derives the complete job and coordinates verification and final installation; its candidate separates authority-owned route/view changes, Trainer-owned backend state, and optimization runtime | Backend work may allocate, wrap, shard, compile, or communicate before publication. Replacement-only and destructive failures follow D8 and never expose a mixed current state. |
| Realize optimization | Accepted training subjects and units, grouping/policy meaning, current bindings, and preparation constraints | Trainer optimization infrastructure resolves live membership, trainability, logical/execution groups, optimizer/scheduler state, clipping, synchronization, and lifecycle projections | Optimizer/backend construction may allocate or communicate. Failure installs no partial runtime; later relevant revisions invalidate or require re-realization. |
| Execute training | Accepted execution structure, current prepared projections, prepared optimization runtime, changing inputs/coordinates, and granted authority | Accepted behavior returns declared computation outputs, optimization inputs, observations, owned-state updates, effects, or transition requests; Trainer performs retained mechanics | Only accepted effects may cross the execution boundary. Failure follows the active ownership profile, stops unsafe advancement, and cannot mutate canonical state outside an accepted transition. |
| Coordinate caching | Selected capability, data scope, representation/conditioning behavior, writable destination, and coherent dependency projection | Pipeline traversal/storage and selected codecs produce cache records plus dependency/readiness facts recorded as capability state | Cache writes are external effects. Failure or partial output is reported explicitly and cannot advertise stale or incomplete cache readiness. |
| Coordinate validation | Selected capability, trigger, evaluation-ready projections, deterministic input/RNG state, and aggregation policy | Trainer/pipeline traversal and accepted evaluation behavior produce typed evaluation results, observations, and updated validation state | Evaluation may consume resources but has no undeclared training-state effect. Failure restores temporary runtime projections and cannot fabricate a successful observation. |
| Coordinate sampling | Selected capability, trigger, destination, request set, and prepared conditioning/predictor/representation projections | Trainer/pipeline orchestration and accepted generation behavior produce typed sample results and updated sampling state | Image or media writes and tracker publication are explicit effects. Failure restores temporary projections and reports which requested outputs did or did not materialize. |
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
| Execute training | Trainer's loop supplies time/input and current prepared projections; accepted behavior executes only its granted operations, while Trainer commits its retained mechanics and the authority handles any requested canonical transition. | `accepted-training-execution`: “Accepted execution survives the authoring object”, “Execution uses current prepared state”, “The standard profile retains generic Trainer mechanics”. |
| Coordinate caching | The caching capability coordinator derives work from an accepted request and current data/representation dependencies; it publishes readiness only for completed, fresh cache results. | `training-capability-coordination`: “Capability readiness derives from accepted state”, “Caching separates semantics from storage orchestration”. |
| Coordinate validation | Trainer/pipeline scheduling derives the accepted request from trigger and ready projections; validation coordination publishes observations and capability-owned state only after traversal/evaluation succeeds. | `training-capability-coordination`: “Validation separates evaluation semantics from traversal”, “Capability results feed observation without transferring ownership”. |
| Coordinate sampling | Trainer/pipeline scheduling derives a request from selected trigger/destination and ready projections; sampling coordination reports actual outputs and capability state after generation/publication. | `training-capability-coordination`: “Sampling separates generation semantics from orchestration”, “Capabilities have named request and result semantics”. |
| Persist trained artifacts | Persistence coordination derives the accepted product plan from a coherent state projection; publication of the artifact result follows actual writes and does not publish a fictitious product on partial failure. | `training-artifact-persistence`: “Artifact persistence has declaration, request, plan, and result stages”, “Artifact results report actual output”, “Persistence uses one accepted boundary”. |
| Restore runtime | Trainer/pipeline restoration coordination derives the job from the accepted snapshot contract and registered state contributors; it coordinates one coherent final publication across authority, Trainer/backend, optimization, input, and contributor-owned state. | `training-capability-coordination`: “Persistence and restoration remain distinct capabilities”; `run-participant-state`: “Runtime identity and lineage remain separate”; `training-runtime-preparation`: “Process failure belongs to restoration”. |

The capability rows intentionally stop at a common request/evidence/ownership
frame. Their failure statements are common-frame obligations here, not a
claim that every capability delta already contains its full failure protocol.
G4 must make the corresponding domain-specific failure requirements explicit
before production implementation. None requires SDXL-shaped universal inputs,
and no coordinator may bypass the authority's binding and revision checks.

#### Run state is composed, not universally owned

The training process needs one coherent per-run state aggregate, but that does
not make every value part of the participant authority or give one object
unrestricted mutation rights. The aggregate distinguishes at least:

- governed semantic state: accepted arrangement meaning, obligations,
  participants, relationships, bindings, routes, revisions, and freshness;
- Trainer execution state: step/epoch progress, accumulation position,
  synchronization and backend state, interruption, and cleanup progress;
- input state: manifests, prepared input sources, dataloaders, deterministic
  cursor/RNG facts, and current epoch input;
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
