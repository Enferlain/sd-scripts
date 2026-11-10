## Context

This is the governing design record for the model–strategy–Trainer rework. See
[proposal.md](proposal.md) for motivation and scope.

Production implementation is not ready merely because this OpenSpec has all
standard artifact files. The settled architecture is recorded here now; the
remaining Trainer-consumption, execution, optimization, capability, and
production-shape decisions are completed through the numbered design gates
below.
Production tasks are added only after those gates have evidence, scenarios,
and acceptance tests.

A bounded, non-integrated executable experiment may inform G2/G3's choice of
execution representation and D14's language evidence for tasks 5.3/5.4. It is
design evidence, not a production
milestone or an accepted public API: the active Trainer and launcher must not
consume it, and G5 decides which parts, if any, become production interfaces.
This exception does not relax the production-readiness gate.

### Source authority and carry-forward rule

The pre-OpenSpec records have different jobs. They are not interchangeable:

| Record | Authority in this change |
| --- | --- |
| [Strategy system direction](../../../docs_design/archive/models-strategy-trainer/strategy_system_direction.md) | Normative input for the settled architecture and Q1–Q5 decisions. This design must not silently weaken or reopen it. |
| [Contract exchange design](../../../docs_design/archive/models-strategy-trainer/strategy_contract_exchange_design.md) | Detailed derivation of binding, preparation, identity, optimization, and the execution frontier. `SETTLED INPUT` is inherited; working decisions are adopted here only where this design or a delta spec states them. |
| [Production inventory](../../../docs_design/archive/models-strategy-trainer/strategy_system_inventory.md) | Evidence about active code and pressure scenarios, not authority over the target. |
| [Implementation mapping](../../../docs_design/archive/models-strategy-trainer/strategy_system_implementation_mapping.md) | Source-backed current-to-target responsibility map and design dependency order. |
| [Chronological notes](../../../docs_design/archive/models-strategy-trainer/notes.md) | History. Later dated corrections supersede earlier positions. In particular, the accepted-run-arrangement correction supersedes the active-strategy topology. |
| [Framework comparison](../../../docs_design/archive/models-strategy-trainer/framework_pattern_comparison.md) | Subordinate prior art. It does not select this repository's ownership model. |
| [xDSL language 0.1 proposal](research/xdsl_language_0_1_proposal.md) | Research input. D14 and the execution/contract deltas adopt the bounded shared-lifecycle decisions; its wider operation catalogue, identifiers and Python sketches are not accepted production APIs. |
| [Executable spike](../../../docs_design/models-strategy-trainer/executable_spike/) | Behavioral evidence for binding/preparation failure cases. Its class names and storage shapes are not a production prototype. |
| [Original discussion](../../../docs_design/archive/models-strategy-trainer/strategy_discussion.md) and [external discussion](../../../docs_design/archive/models-strategy-trainer/strategy_discussion_external.md) | Raw historical context. External comments are opinions; neither transcript is a requirement source. |

The language proposal stays unchanged as supplied research input. Where this
design departs from or adds to it, D14's "Amendments to the language proposal"
records the adopted rule; its provenance is tracked in bead
`sd-scripts-syv.31.26`.

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

The new system keeps these meanings. It does not wrap the current mutable
Trainer hub in a new all-purpose runtime object.

#### Training-first source disposition

The current `library/training/` package mixes several different owners. The
target keeps training as the package for the training process, but it does not
move these files wholesale or make the central Trainer class the owner of
everything that moves under training coordination. D21's coverage table
lists what in the new system covers these files; this table records the
analysis behind it.

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
| Concrete strategy validation, sampling, checkpointing, and preparation facets | Generic outer workflows become triggers, requests and handlers that the IR runtime coordinates (D21). Model-, objective-, serializer-, and family-specific computation remains selected accepted behavior and is not copied into Trainer branches. |
| `strategies/base/context.py` | Common phase, coordinate, route, and changing-input facts become part of the accepted execution invocation rather than being published by an active authoring strategy. Domain consumers retain only their accepted use of those facts. |

The sweep also exposes two state concerns that the provisional training shape
must not leave hidden in Trainer fields:

- input state, including manifests, prepared input sources, dataloaders,
  deterministic cursor/RNG facts, and the current epoch input; and
- observation/lifecycle state, including metric accumulators, validation
  history, runtime traces, reporting-session status, and cleanup progress.

The code that creates or advances those values lives with its owner:
producers, capability implementations, the observation owner, and the data
or logging packages (D21). Their
current per-run values belong to the coherent run-state aggregate rather than
requiring new top-level orchestration authorities.

### Previously tried draft

An earlier September draft is retained only as
`docs_design/archive/models-strategy-trainer/packages/mst_pre_astra.zip`. It correctly recognized the governing
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

G4.6 reconciles two concrete conflicts identified in milestone 1. The companion
metadata loading delta now agrees that accepted authority owns current bindings
and loading/provenance evidence is retained in scoped projections, not a
competing Trainer collection. Its target-ref delta now permits accepted
consumers rather than naming optimization as the universal builder, and limits
live access to revision-pinned projections. Typed loading evidence, structural
identity, and observation-local alias evidence survive. Their concrete type
integration remains the shared G5/metadata implementation boundary.

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
| `optimization-target-refs` | Its namespace and mode scenarios imply optimization/modes build all targets, live objects are embedded in unqualified refs, and universal component ancestry excludes admitted standalone participant state. | `MODIFIED` policy-neutral refs and consumer scenarios separate semantic identity from revision-pinned live projections. Trainer optimization resolves trainable parameters; governed PEFT realization resolves authored host-target intent. Modified selector/expansion applicability preserves existing component-backed labels, keys, selectors, grouping, and provenance while permitting participant-owned substructure without invented family components. |
| `component-qualified-selectors` | Unqualified inspection/group-matching and shared-ref requirements can force component prefixes and internal keys onto standalone participant-owned state, despite the target-ref delta. | Four `MODIFIED` requirements scope component-qualified inspection, matching, target selectors, and internal keys to genuine component provenance. Existing model selectors, regex matching, and public-label/internal-key separation survive; standalone targets retain accepted participant/substructure meaning without fabricated ancestry. Concrete standalone syntax remains G5. |
| `adapter-system` | Optimization-owned targeting, two `AdapterMode` ownership requirements, and adapter-path persistence assign durable authority to old actors. | Four requirements are `REMOVED`; realization, method configuration, and adapter-facing optimization boundaries are `MODIFIED`. Added requirements assign target intent to authoring, resolution to governed PEFT realization, timing to Trainer/pipeline, and product/loading/restoration to separate exchanges. Broad applicability, typed helpers, and parameter-native grouping survive. |
| `adapter-module-targeting` | Optimization owns host-module selection; `AdapterMode` appears in target passing, trainable handoff, and `loha` export/merge. Mixed-method overlap was deferred. | Ownership, deferral and the migration compatibility fields are `REMOVED`; governed resolution and unsupported-overlap rejection are `ADDED`. Provenance, parameter grouping, `loha`, method settings, continuation, and declared-component scope are `MODIFIED`. Method behavior, selector compatibility, strict artifact-continuation default, and non-SD component support survive. |
| `repo-owned-lora-method` and `repo-owned-vera-method` | Build scenarios name `AdapterMode` and call resolved targets optimization-owned. | Each build/loading requirement is `MODIFIED` to consume governed PEFT results. Method-local algorithm, configuration, state, export, and trainable-provenance requirements survive; methods still do not own host traversal or semantic selection. VeRA's malformed delta-shaped main-spec headings are normalized before validation. |
| `training-observability` | Producer and startup/resource scenarios allow mode-owned filtered state and mode/strategy authority. | Three requirements are `MODIFIED`: accepted projections/results supply facts, Trainer or accepted behavior choose when to emit, and observability owns formatting/routing without becoming authority. Sinks, ordering, resource separation, trackers, and provenance diagnostics survive. |
| `model-family-metadata` | Existing realization identity can be misread as live participant identity; companion composition and filing rules assume loader/preparation success establishes current history. | Shared `MODIFIED` distinction/filing requirements use accepted publication and checkpoint-scoped finalization, preserving all existing scenario identifiers. `ADDED` requirements distinguish runtime and durable identity, accepted history, actual product/restoration outcomes, and required delivery from optional telemetry. Existing typed catalog/build/emission/projection and external-parity requirements survive. |

Two inherited rules need explicit limits. The existing
`optimization-target-refs` and `component-qualified-selectors` component-backed
selectors and existing grouping behavior continue through policy-neutral refs,
including the accepted
component learning-rate policy for adapter trainables with genuine host
associations. They do not force standalone participants into family
declarations, and neither authorizes optimization to choose PEFT host targets.
The `adapter-module-targeting` compatibility rule, which allowed legacy
`component`/`component_key`/`path`/`module` fields as aliases during a
migration, is `REMOVED`: the new system is built new, and adapter realization
reads only `TargetRef` and its revision-pinned projection. The strict
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
| Catalog identity and source provenance | Durable catalog assignment, portable evidence, source selections, ordered materialization evidence, and lineage remain metadata-owned. | Ownership disposition reconciled in G4.6. Metadata foundation tasks 2–3 remain independent. Source evidence may describe loader candidates, while accepted composition requires authority publication. Metadata task 4.1 verifies concrete G5 integration before tasks 4.2–4.6; family task 5 and composition task 6 consume accepted transitions. |
| `loaded-model-components` | One typed loading-evidence shape, family-declared components, successful source decisions, deferred updates, and limitations remain required. | Both changes now carry the same modified loading requirement and scenario bodies. Authority publication establishes current bindings; accepted scoped projections retain loading/catalog evidence without a competing Trainer collection. Concrete projection/type compatibility remains metadata task 4.1, not an unresolved ownership decision. |
| `optimization-target-refs` and structural metadata | Qualified structural paths, distinct live-object/storage observations, alias evidence, optional catalog links, and observation-local execution deduplication remain required. | Companion structural associations now use policy-neutral refs, accepted consumers, and revision-pinned live projections. Alias evidence does not globally deduplicate semantic owners; same-unit/group aliases and unsupported cross-unit/group overlap retain distinct treatment. Metadata structural tasks 9 onward and consumer task 12.1 preserve this boundary without requiring catalog resolution for a valid live view. |
| `model-family-metadata` and resource intelligence | Typed model/realization/artifact facts, append-only composition history, registry filing, qualified resource owners, and bounded structural queries remain required. | Both changes now carry matching modified distinction/filing requirements, including `Model loading completes`. Companion provenance/design/tasks use accepted publication, named checkpoint finalization, original capture associations, and required-history delivery separate from reached effects. G4.5 supplies metadata/resource consumer rules; concrete schema/query integration remains G5 and metadata tasks 6–7/9–12. Resource intelligence remains an observer. |

These are boundary dependencies, not a whole-change landing order. G4.6 files
the semantic dispositions in both existing changes and repairs the companion's
omitted `Model loading completes` scenario; it does not complete the companion
implementation gate or authorize sync/archive. Metadata task 4.1 remains the
explicit verification against concrete G5 authority, projection, and evidence
types. Both changes must pass strict validation and required review at that
integration gate. Shared loading and model-fact modified requirements must
retain matching scenario identifiers and bodies; an edit or rename on one
side cannot silently diverge from the other.

The G4.6 audit also corrects older participant-state wording that would turn
restoration lacking identity facts into a new run. Required identity/revision
state missing from a same-run request means incomplete restoration and gated
execution; new identities require a separately requested new run with lineage.
Capability and product deltas retain G4.1–G4.5's admitted dependencies, protected
use, actual partial/uncertain outcomes, external-dependency completion policy,
and separation of artifacts, runtime recovery, required metadata, and optional
observations. Adapter/method deltas already displace mode/targeting ownership;
their selector, provenance and method-local behavior remain as recorded
above, and the migration compatibility rule is removed. Task 4.6's validation/review gate applies to these
dispositions before the milestone can be marked complete.

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
  ABC, or family-shaped Trainer fields for compatibility.
- Making arbitrary model/component assembly safe in this change.
- Adopting, vendoring, or recreating Lightning, Fabric, or another complete
  framework as part of this change.
- Making metadata, observability, a backend wrapper, or a checkpoint file the
  owner of live participant identity.
- Renaming provisional spike vocabulary into production without deriving the
  actual consumers and exchanges first.

## Glossary

These are the design's own terms, each with one meaning. Kuroiro's language
words are listed in D16's vocabulary. Where a term comes from a spec, the spec
stays normative.

| Term | Meaning | Defined in |
| --- | --- | --- |
| Accepted arrangement | The specs' name for what fulfillment accepts: the strategy's meaning with its obligations and authority seed. The design calls it the accepted run | D4, D17 |
| Accepted run | The accepted strategy with its obligations, pinned versions and authority seed; only fulfillment produces one | D17 |
| Activity | A named piece of work with its own lifetime, such as a producer, repeat, handler or parallel | D16 |
| Advancement | What one window's completion did to a unit: synchronize, clip, step, schedule and zero | D9, D20 |
| Artifact | What a run writes for a declared product: its members and physical resources, with an identity and lineage distinct from the participants it captured. Runtime snapshots and caches are not artifacts, and a strategy saved as text is its written form | D11, D21 |
| Assembly | A model as released: the components it uses and the conventions they were trained with. It references its components and owns none of them | D21 |
| Attempt | A governed job with a basis: a materialization or a preparation | D19 |
| Authoring | Writing strategies and their reusable pieces against the contract. It never includes fulfillment | D18 |
| Binding | A participant's bound state | D6, D19 |
| Capability | A kind of activity the contract recognizes, such as sampling, validation, saving a product, resume or caching, with a defined request, result, readiness and failure. It is not training-specific, and the IR runtime does its generic coordination | D3, D21 |
| Checkpoint | Only the user-facing name of a save handler, not a design term; see cut and readiness checkpoint | D17 |
| Claim | An item taken from a stream, settled exactly once by a disposition | D16 |
| Component | A part of a model with its own identity, such as an autoencoder, text encoder or denoiser. Arranged by kind in `library/models`; the model that introduced it does not own it | D21 |
| Contract | The composed, versioned registered vocabulary: everything a strategy may use. Bare "contract" always means this | D18 |
| Cut | A coherent recoverable capture across owners, for exact restoration; each owner supplies its cut state | D16, G4.4 |
| Declaration | What a component or feature states about itself, in three layers: its meaning, what a use requires, and provenance or guidance | D21 |
| Definition | One contract entry: an operation, type, capability, feature contract or ownership profile. Not a Python `def` | D16, D18 |
| Dependencies | What a projection or prepared realization relies on | D6, D19 |
| Derived word | A word defined by expanding into other words during normalization, such as `window`. The research called it a recipe | D16 |
| Diagnostic | A source-attributed problem report: rule code, names, line, fragment and fix | D18 |
| Evidence | Typed facts that a producer supplies later for an open obligation | D7, D19 |
| Executable run | The realized run: prepared strategy, published generation and device adapters | D17 |
| Feature | Anything a strategy can take and reuse, as it is or modified: an approach or its math, such as DDPM, a conditioning technique, an adapter method or EMA. Its code lives where its kind of code belongs, and it declares itself. The repository's features are listed in a generated listing in `strategies/features/`, where reusable strategy-level pieces also live. Machinery is not a feature, even when it offers a choice | D21 |
| Feature contract | What a feature offers and requires: the meaning and requirements in its declaration | D3, D21 |
| Fulfillment | The IR system's acceptance step: read, normalize, verify and derive obligations. Launch runs it | D18 |
| Generation | The coherent published state across owners | D19 |
| Guidance | What is known rather than required: provenance, tested combinations, recommended settings and community practice. It produces help and logs, never a refusal | D21 |
| Implementation | Selected code that does the math, declared with its contract and version | D16, D18 |
| IR system | Kuroiro on xDSL: the language, its front end, fulfillment, reference semantics and runtime, in `library/workflow`. Older text calls it the workflow layer | D14, D16 |
| Lease | The handle an access scope gives its body; it cannot leave the scope | D16 |
| Machinery | Code that runs whatever approach a strategy chooses, such as preparation, FP8 casting, optimizer construction and data loading | D21 |
| Obligation | Something verification requires the run to discharge, attached to a readiness checkpoint | D7, `training-contract` |
| Operation contracts | An operation's region, flow, resource, realization and decision contracts: the parts of its entry that the IR system checks. Always written with the qualifier | D14 |
| Owner | Holds one kind of state and realizes access to it at the owner seam | D16, D19 |
| Participant | A model resource with identity | D6, D16 |
| Product | A declared kind of artifact for an intended use: what it covers, its representations and transformations, its dependencies and its completion conditions. The owners of what it saves, such as assemblies and adapter methods, declare the products they support | D11, D21 |
| Projection | One consumer's coherent bundle of state, pinned to a generation or to exact dependencies | D6, D19 |
| Readiness checkpoint | The named point at which an obligation must first be fulfilled | `training-contract` |
| Realization | Concrete objects that realize an accepted meaning, such as a unit's optimizer objects. Governed realization is seed, materialize, prepare and publish | D9, D17 |
| Reference | The opaque identity of one incarnation | D19 |
| Revision | A number that identifies one published state of something an owner holds, such as an obligation set, a binding, a route, a relationship or a unit definition. A change of meaning or realization advances it; an ordinary numeric update does not | D6, D19 |
| Route | A named way to call a participant | D6 |
| Run | One execution of a strategy: first the accepted run, then the executable run as it executes | D17, D21 |
| Run authority | The owner of participant and relationship state | D6, D19 |
| Snapshot | A fixed published version of some state, and the access mode that reads one | D16, D19 |
| Strategy | Python that an author writes to define a run: a strategy function, its activities and any implementations of its own. Its read, accepted and prepared forms are the same strategy at later stages. Training, evaluation, caching and sampling runs all have strategies. A maintained strategy ships with the repository for one thing trained, with adapters, EMA and similar as choices inside it | D18, D21 |
| Transition | A typed proposal for a governed change | D7, D19 |
| Unit | An independently advanced optimization responsibility | D9, D20 |
| Version | Which state a use saw | D19 |
| View | A typed look at one participant that is not a route | D6 |
| Window | Contributions that advance a unit together | D17 |

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
       meaning + run-specific obligations + authority seed
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
of requirements for realizing that run, and its authority seed; failure
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

### D4. The authored strategy is a definition, not the ordinary runtime peer

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
- a passive definition whose missing behavior is reconstructed with family
  branches inside Trainer.

### Current-code evidence for the intended Trainer boundary

The existing [implementation mapping](../../../docs_design/archive/models-strategy-trainer/strategy_system_implementation_mapping.md)
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
| Fulfill an authored strategy | Complete authored selections under the active contract/profile and every fact knowable before realization | Detailed acceptance containing one semantic arrangement, one authoritative run-specific obligation revision, and the authority seed | No ordinary external runtime effect. Rejection returns detailed diagnostics and produces no Trainer-acceptable arrangement. |
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
| Fulfill an authored strategy | Contract-guided strategy-side fulfillment checks the completed strategy against the already active contract and establishes the obligation revision and accepted arrangement; no partial arrangement is published on rejection. | `training-contract`: “Strategy fulfillment produces acceptance or rejection”, “Obligations are evaluated at the earliest authoritative evidence point”, “One accepted contract governs one run authority”. |
| Materialize and bind | Training lifecycle coordination derives a candidate job from accepted obligations/current revisions; the selected domain loader or constructor supplies evidence; only the run authority publishes binding/relationship changes. | `run-participant-state`: “Initial realization and later transitions share one enforcement direction”, “Transitions are revision-checked and atomic”; `loaded-model-components`: “Model loading returns the loaded-component surface”. |
| Prepare runtime | Training-side preparation coordination derives one complete attempt-scoped job and coordinates final publication of authority-owned routes/views with Trainer-owned backend/optimization state; neither side exposes a half-installed result. | `training-runtime-preparation`: “Preparation derives one coherent attempt-scoped job”, “Preparation results separate ownership surfaces”, “Fallible work precedes final publication”, “Stale results never publish against newer state”. |
| Realize optimization | Trainer optimization derives candidates from accepted subjects/units and current projections; the joint preparation coordinator publishes the prepared optimization runtime with the corresponding authority routes and Trainer backend state. | `training-optimization`: “Trainer owns standard realization and mechanics”, “Realization and preparation form one publication attempt”, “Binding changes invalidate affected optimization runtime”. |
| Produce and hand off input | Training input coordination manages producer lifecycle, readiness, admission, delivery, and failure routing under accepted obligations; selected input behavior owns live-source traversal, selection/packing policy, and private continuation state. Cache-production traversal remains with caching coordination. An admitted handoff is not proof of action completion or optimization advancement. | `training-contract`: “Authors assemble strategies explicitly”, “Obligations are evaluated at the earliest authoritative evidence point”; `accepted-training-execution`: “Input handoff retains identity and admission meaning”, “Cross-owner action facts remain correlatable”; `training-capability-coordination`: “Caching separates semantics from storage orchestration”, “Cache publication and consumer admission are distinct”, “Independent cache production has bounded lifecycle and owned continuation”. G4.1 files the cache-production exchange below; task 4.4 completes coordinated exact-restoration protocols. |
| Execute training | Trainer's loop supplies time/input and current prepared projections; accepted behavior executes only its granted operations, while Trainer commits its retained mechanics and the authority handles any requested canonical transition. | `accepted-training-execution`: “Accepted execution survives the authoring object”, “Execution uses current prepared state”, “The standard profile retains generic Trainer mechanics”. |
| Coordinate caching | The caching capability's implementation derives work from an accepted request and current data/representation dependencies; it publishes readiness only for completed, fresh cache results. | `training-capability-coordination`: “Capability readiness derives from accepted state”, “Caching separates semantics from storage orchestration”, “Cache requests and results preserve the selected computation”, “Cache freshness follows actual computation dependencies”, “Cache publication and consumer admission are distinct”, “Cache storage lifecycle preserves outstanding use”, “Independent cache production has bounded lifecycle and owned continuation”. |
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
alone does not establish a coherent recoverable cut with model/optimizer/input state.
G4.4 defines that cross-owner snapshot protocol; G5 chooses concrete types and
backend publication/access mechanisms.

**Evidence and implementation limits.** The following checks complete the
caching meanings, not a claim that the current backend implements them:

| Pressure case / current evidence | Required consequence |
| --- | --- |
| `Trainer.train()` (`library/training/runners/trainer.py:256–301`) invokes caching before the loop; `CachingEngine.cache_dataset()` (`library/data/caching_engine.py:252–348`) processes a finite manifest and synchronizes ranks. | Preserve useful batch/encoding machinery without making full-manifest completion a universal execution prerequisite. Independently progressing production needs its own supported implementation and lifecycle tests. |
| `CacheBackend` (`caching_engine.py:71–208`) accepts image tensors and per-entry paths; `CacheEntry` (`library/data/structures.py:54–112`) stores one image, caption, and latent/TE paths. | Decouple cached representation and storage access from image/sample identity; allow shared/variant records and sharded or bounded storage without requiring those backends in this change. |
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
editing, or every asynchronous backend ships in this change. This
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
continuation needs; G4.4 defines a coherent recoverable cut, not this exchange.

**Evidence and implementation checks.** These complete the semantic exchanges,
not production capability implementations:

| Bounded evidence / pressure case | Consequence and remaining check |
| --- | --- |
| `ValidationScheduler` (`library/training/phases/validation.py:38–95`) and `_run_step_side_effects()` (`library/training/phases/training_loop.py:133–183`) provide useful due logic; `run_sampling_and_validation()` (`library/training/phases/orchestration_helpers.py:60–114`) hard-codes sequencing and broad strategy calls. | Preserve accepted scheduling without carrying that fixed sequence or active strategy boundary into the engine. Test independently due capabilities and their declared dependencies. |
| SDXL evaluation (`library/strategies/sdxl/validation.py:18–196`) combines selected timestep/L2 meaning with RNG switching, cyclic loader traversal, batch-mean aggregation, and history updates; SD3 (`library/strategies/sd3/validation.py:19–154`) repeats the workflow with different conditioning/representation. | Keep domain evaluation selected while moving outer coordination out. G5 must test unequal weights/masks, partial or empty coverage, distributed reduction when supported, and input-provider continuation without assuming the current loss formula is universal. |
| `temporarily_in_eval_mode()` (`orchestration_helpers.py:49–57`) changes mode before its `try` and restores training unconditionally; validation and `sample_images_common()` (`library/training/sample_generation.py:435–549`) restore global RNG only on their normal path. | G5 must inject partial setup, evaluation/generation, cancellation, and cleanup failures and verify prior projections or unsafe-use gating. Existing helpers are not proof of scoped restoration. |
| SD, SDXL, and SD3 sampling (`library/strategies/sd/sampling.py:19–83`, `library/strategies/sdxl/sampling.py:67–150`, `library/strategies/sd3/sampling.py:116–180`) unwrap/move family components and construct generation implementations. SDXL/SD3 flow backends contain selected inference algorithms. | Preserve family algorithm differences without universal component slots; test stale inference views and reads during conflicting temporary mutations using the chosen backend access protocol. |
| `SamplingRequest` (`sample_generation.py:45–59`) is image-shaped; `sample_image_inference()` (`sample_generation.py:552–625`) saves an image before direct tracker logging. | G5 must distinguish generated, saved, and reported outcomes, including save failure, later sink failure, and a partial multi-request result. Selected non-image/multi-output schemas must not require new Trainer family branches. |
| Recorded [BD3LM evaluation](research/llm-training.md), [video/audio cases](research/video-audio-training.md), and [train/inference-view coordination](research/preference-and-rl-post-training.md). | Preserve adaptive reactions and domain-specific measurement/generation semantics; these research cases establish requirements/neutrality pressure, not that every algorithm or separate inference backend ships in this change. |

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
but each section keeps its declared writer. Producers, capability
implementations, accepted operations, and Trainer mechanics create or advance
the state they own. Observability formats, routes, buffers, and persists accepted facts; it
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
| Fulfill | Governed semantic state; contract-guided fulfillment establishes obligations and the authority seed. | Training lifecycle coordination, Trainer, selected operations, capability implementations. |
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

#### G4.5 Metadata and observability consume accepted run facts

The metadata system is a recently built foundation, not a fixed constraint on
the new training system. Its central typed-item, builder, validation, registry,
backend, query, and projection responsibilities remain useful. Its schemas,
identity associations, and accumulation/query logic can change when the
accepted run needs distinctions that they do not preserve today. Keeping that
foundation does not mean forcing a dynamic whole run into its current
one-session, one-loaded-realization, step/epoch-facing inputs.

**Origin truth and durable representation are different authorities.** The run
authority establishes participants, relationships, accepted transitions, and
current bindings. Optimization, input, selected behavior, capabilities, and
backend owners establish their own state, progress, and results. They expose
scoped facts at the boundary where those facts become authoritative. Central
metadata builders translate explicit inputs into typed items; metadata owns
item validation, durable identity qualification, relationships, accumulation,
and export. That validation does not establish a new live participant or judge
whether an execution route is currently usable. Metadata may retain historical
binding/route facts, but not live module handles or a competing canonical
binding collection. Observability formats and routes observations rather than
reconstructing the arrangement through the old Trainer hub.

| Origin-owned meaning | Consumer input and retained distinction |
| --- | --- |
| Logical run and runtime identities | Authority-qualified run, participant/relationship incarnations, and optimization-unit identities remain distinct from execution-session/attempt, authored address, catalog/source/revision, realization/composition, and artifact identities. Available associations are explicit, not guessed from a name, role, path, or Python object. |
| Accepted transitions and current projections | Accepted arrangement/obligation, binding, relationship, route/view, unit-definition, and relevant owner-state dependencies qualify the observation. Failed/stale attempts remain attempt outcomes, not accepted topology history. Retired references retain historical meaning. |
| Activity progress and outcomes | Producer/request/input/operation/unit/capability facts retain their required cross-owner associations, independent coordinates, actual provenance, and known/skipped/failed/uncertain/unattempted outcomes. No global step substitutes for them all. |
| Product and snapshot publication | Descriptive export facts and requested scope are distinguished from actual product/member/resource publication, required/optional coverage, local/remote outcomes, and snapshot capture/publication/restoration guarantees. |
| Diagnostics and resource evidence | Accepted component and optimization projections supply subjects, labels, order, aliases, membership, and preparation state. Resource observations keep measurement basis, process/rank/device scope, owner evidence, and validity rather than promoting a display label into ownership. |

**History is associated with the state observed, not the time it was filed.**
Within one run, compatible replacement keeps the participant incarnation while
recording changed binding/state dependencies; retirement followed by a new
participant at the same authored address does not overwrite the old identity.
Separately authority-established teacher or EMA participants remain distinct
even when roles or sources overlap. Teacher/student labels can instead name
execution roles of one participant, and selected EMA state can be an artifact
or state variant rather than a separately accepted participant. Metadata records
the supplied identity and state provenance; neither label creates an incarnation.
Wrappers, replicas, access views, and named
routes do not independently decide participant identity. A durable model
realization/composition observation can be associated with these runtime facts
without becoming the participant reference itself.

Recorded history preserves its observed revision/scope and source identity.
An older observation arriving after a newer one remains older evidence; backend
arrival order or wall-clock time is not an authority revision. Where no supported
ordering/correspondence exists, consumers report ambiguity or incompleteness
rather than inventing a current state. Ordinary weight updates, adaptive state
changes, and produced-data versions are not all topology/binding revisions.
Their required state provenance is supplied by the owner or capture result.
This does not require a universal event log, serializing every runtime update,
or retaining unrestricted live state in metadata.

**Restoration preserves the logical run and changes the execution session.**
D11's restored participant, relationship, and unit identities and applicable
revisions retain their meaning. Observations from the new session are
distinguishable from historical attempts in the earlier session; rebuilding
modules does not create new incarnations. A new run initialized from an
artifact instead records explicit lineage between different identities.
Observation or tracker identifiers cannot replace the saved logical run
identity. The existing `MetadataSnapshot` is a collected-record view for
metadata queries/projections, not the G4.4 snapshot of required execution state.
Metadata may describe or reference a recovery snapshot, but cannot certify its
continuation coverage merely because its own store was saved successfully.

**Actual results drive product and run reporting.** Header/export facts may be
prepared before a serializer writes; they do not establish that a resource was
written or a product completed. Post-write reporting consumes D11's captured
state/provenance and actual publication result, not whatever participant happens
to be current when the report is built. Local success, pending/failed upload,
optional omissions, partial/uncertain output, and independent snapshot results
remain separate. Similarly, a producer's ready result is not a consumed input,
and a returned optimizer call does not prove numerical parameter change or
all-unit success. Reports can summarize these facts, but cannot strengthen
their guarantees or silently discard uncertainty to fit a single completion
flag. Public labels, family order within a declared view, and adapter provenance
remain useful presentation facts, not runtime topology selectors.

**Resource accounting receives evidence, not inferred ownership.** A component
key or phase label alone cannot qualify an owner across incarnations, sessions,
or activities. Estimates and measurements remain distinguished; the accepted
scope and validity of owner-bearing evidence determine how resource consumers
associate them. Physical sharing/alias deduplication is observation-local and
does not merge logical participants or optimization identities. Missing owner
evidence produces an accounting gap, not fabricated ownership. Bounded module
inspection by an authorized measurement producer remains possible; downstream
reports consume its supplied facts, not an unrestricted Trainer or strategy.

**Observation delivery is not algorithmic feedback or recovery publication.**
Optional telemetry can use bounded buffering and declared drop/degradation
policies outside latency-critical storage paths. Sink failure cannot erase an
accepted transition or known execution/product outcome, and retrying a filing
does not retry training work. Required history, provenance, product metadata,
or restoration state follows its accepted completeness/durability policy and
cannot be silently dropped as optional telemetry. If a required filing fails,
the relevant publication or subsequent dependent use remains incomplete or
unavailable under that policy; an already reached effect is not rolled back.
Observation gaps remain visible rather than making missing evidence mean that
an operation never ran. Delivery/retry must preserve fact identity and must not
manufacture a second semantic transition.

Feedback required by an adaptive algorithm remains an accepted owner-to-owner
execution exchange even if a copy is logged. Disabling/dropping tracker metrics
must not suppress that feedback or advance the algorithm twice on log retry.
That required algorithmic state belongs to its continuation owner, not a
telemetry buffer. These distinctions permit swappable sinks without making
backend ingestion ordering part of training semantics.

**Current-to-target evidence and remaining work.** The G4.5 graph/source pass
found current session-derived realization IDs and once-retained filing in
`Trainer._file_model_realization_metadata`, pre-write artifact fact filing in
`Trainer._build_checkpoint_metadata`, and broad mode/strategy/Trainer inputs in
`build_trainer_diagnostic_rows`, `build_training_startup_summary`, and report
context construction. The central model builder already returns module-free
typed facts from explicit inputs, while the resource layer already preserves
measurement basis and evidence-qualified accounting. These are evidence of
useful responsibilities and required input changes, not final APIs or proof
of the target runtime. In particular, the current buffer's degradation policy
does not establish required-history durability.

The existing `model-family-metadata` and `training-observability` deltas carry
the consumer requirements and positive/negative scenarios. Catalog/source,
structural, and revisioned-composition work in the companion metadata change
remains required where applicable; its runtime-facing ownership is reconciled
with accepted transitions, not independently implemented around old Trainer
bindings. A composition described as final/prepared is final for its named
checkpoint, not a promise that the run can never change again. G4.6 reconciles
the overlapping deltas. G5 and the metadata implementation work choose concrete
fact fields, identity links, query/accumulation rules, delivery policies, and
migration tests; existing metadata weak points may be corrected there rather
than preserved for their own sake. No new execution-language construct or
production metadata implementation is established by this filing.

### D6. One run authority owns participant and relationship state

The Q1–Q5 conclusions are carried forward as one coherent rule set.

This authority owns the governed semantic section of the broader run-state
aggregate described in D5. It does not thereby own Trainer progress, input
resources, optimizer/backend objects, accepted-operation state, capability
state, or observation buffers. Cross-section publication uses an explicit
coordinator; it does not collapse those owners into the participant authority.

#### Identity

- An authored participant key is an address within the strategy. It is not a
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
mutable or permanently frozen. A permitted arrangement amendment is one the
accepted strategy declares (D17). Fulfillment derives its obligations under
the same contract version and execution/ownership profile, and the amendment
brings them into force and evaluates them. Every proposal and derived projection identifies
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

Shared target references cover both component-backed and independently managed
participant-owned module or parameter substructure. Genuine model-component
provenance retains its declared keys, public labels, local paths, and existing
selector/grouping compatibility. A standalone learned participant, such as an
auxiliary loss-weight vector, instead has its own authority-qualified target
meaning and current scoped parameter projection; it needs no invented family
component, borrowed host ownership, or raw-pointer-only optimization path.
Trainer optimization consumes those same shared meanings when resolving its
accepted semantic subjects to live trainables. Genuine adapter host-target
associations remain provenance and do not determine ownership of the adapter's
learned state. Catalog and model-structural associations remain applicable
optional evidence, not an admission prerequisite for otherwise valid targets.

This resolves audit finding P1-REV-01's applicability gap. It does not select a
reference class, field layout, selector encoding, or broader metadata catalog
scope; those concrete choices remain G5 work.

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
strategy-authored instruction for physical construction.

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
coherent recoverable cut; this section fixes only the optimization and
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
For each applicable backend/profile, G5's milestones must identify which
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
for this rework**, not permanent semantic requirements of accepted training
execution. The initial [G3.7 measurements](research/training-mechanism-sketch.md#g37-conformance-and-cost-check)
establish a same-environment reference envelope for the common measured
scope. Before the new system's SDXL run is accepted (`tasks.md` section 18),
the proposed implementation must run the matched cheap-action checks against
the retained experiment in the same environment. The reference fixture and
measurement probe must remain available until those checks have passed. Its
median
excess over direct Python must not exceed that
reference excess by more than the greater of **1 microsecond per action or
20% of reference excess**, across three process runs. This is a regression
allowance for this shared scope, not a universal hardware latency limit.
New required coordination/guards must be present in both compared paths and
have a separately justified budget before their implementation milestone;
they cannot be omitted to fit the old scope, and their budget cannot replace
these gates for comparable scope. The dormant-work check must show
no growing cost trend with arrangement size; an increase exceeding the greater
of **10% of the zero-dormant median or three times its within-run median
absolute deviation** requires investigation and correction or evidence of a
changed relevant-work scope before acceptance. These are measurable gates,
not a choice of graph storage or lowering. After this rework is accepted,
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

#### G4.4 Coordinated runtime snapshot and restoration exchange

“Exact same-run restoration” means restoring the same accepted logical run
while preserving every state, effect, identity, and progress distinction
required by its accepted continuation policy. It does not imply bit-identical
replay of transient execution, regenerated disposable work, process-local
objects, or backend scheduling unless the selected guarantee explicitly
requires those properties.

Runtime restoration has a different purpose. Exact same-run restoration
preserves the logical authority, participant and optimization identities,
accepted arrangement, relevant revisions, mutable state, and run coordinates
only when the snapshot actually carries and restores them. New Python objects,
processes, or wrappers do not break identity; a new execution session has
its own session and attempt identity for observation and correlation. If required
identity or revision facts are missing, same-run restoration remains incomplete
and dependent execution stays unavailable. A separately requested new run,
including one initialized from an artifact, establishes new identities and
records lineage instead of claiming continuity.

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

**Coverage follows the accepted run, not a model file or one step counter.**
The accepted restoration policy identifies required owners, their state
definitions and dependencies, and the continuation guarantee requested. Each
owner supplies its state or an accepted reconstruction rule; stateless or
reconstructible behavior does not need a serialized object merely to appear in
the coverage. The snapshot must preserve enough accepted meaning to reconstruct
selected dynamic behavior, wiring, policies, and grants without retaining the
strategy-authoring object. It does not pickle the whole running process.

| Contributor | Continuation meaning it owns |
| --- | --- |
| Run authority | Logical authority, accepted arrangement and obligations, participant and relationship incarnations, bindings, routes, and relevant revisions, including current stage/topology |
| Selected operations and regions | Owned adaptive/stateful behavior, state-definition dependencies, relevant RNG, progress, unfinished work, and effects under the accepted authority profile |
| Input and producer capabilities | Source/work identity, traversal and allocation positions, dependency provenance, admission/handoff/consumption state, ready results, and accepted treatment of in-flight work |
| Optimization | Unit identity and definition revisions, semantic membership, optimizer/scheduler state, advancement coordinates, and required unfinished gradient windows |
| Backend integrations | Required numerical, scaling, synchronization, distributed/sharded state and its correspondence to semantic owners; reconstruction of physical execution representations |
| Trainer/pipeline infrastructure and other capabilities | Owned scheduling/progress, lifecycle and continuation state, cross-owner associations, and required outstanding requests/results |

These are responsibility categories, not mandatory files or a new universal
state container. Owners keep their writer rights. Logging history is not
automatically required, but feedback used by an adaptive algorithm is required
when its owner needs it to continue. Dependency evidence may be derived or
represented by an accepted generation; it need not be copied into one giant
revision tuple on every runtime object.

**Capture request, coordinated capture, and snapshot result.** A request names
the accepted guarantee, required coverage, consistency policy, and publication
requirements. Infrastructure coordinates protected reads and owner
contributions. A *coherent recoverable cut* means their saved positions are
compatible with the accepted relationships: for example, the saved input
position cannot omit a result that the saved optimizer state already used.
Required contributions across participating ranks or replicas belong to that
same coherent recoverable cut; accepted relationships can still permit different
state versions or progress coordinates.
Activities can retain independent clocks. Depending on the supported policy,
coordination can wait for safe boundaries, protect particular state versions,
or record unfinished work with an accepted recovery rule. There is no universal
end-of-step boundary, global pause, or transaction across training updates.
Protection lasts through actual backend reads/completion, not just Python
return. A snapshot cannot combine inconsistent pre- and post-transition owner
positions or capture a temporary execution edit outside its supported policy.
An intentionally published coexistence arrangement remains one coherent current
arrangement and can be captured at a supported boundary.

The result distinguishes capture, required durable publication, partial or
uncertain writes, and the guarantee actually established. A directory, model
file, or successful rank is not evidence that every required contribution was
captured coherently. Local success and remote publication can differ under the
accepted storage policy. Trained-product results remain independent. A complete
snapshot at publication also does not promise that an external base, data
source, selected implementation, or compatible backend remains available
forever; those dependencies must be identified and checked when restoring.

**Unfinished work follows its owner's policy.** Produced, ready, handed-off,
consumed, gradient-contributing, and advanced positions remain distinct where
continuation depends on them. Ready results can be stored; disposable pending
work can be reissued only when the accepted rule preserves source/work identity,
dependency provenance, and its relationship to consumption and advancement.
Reissue must not train an already consumed result twice. Protected identities,
snapshots, or structured provenance can account for work spanning multiple
encoder/policy states. Recreating arbitrary data under the same pathname is
not a substitute for the required source meaning.

For example, saved Trainer policy P18 and producer view P17 can be compatible
if the accepted input policy allows that lag and the snapshot also accounts for
finished, running, and pending work. Saving only P18 and the optimizer is not
enough. An accepted rule can regenerate disposable in-flight work; this supports
same-run continuation under that rule, not bit-identical replay of the
interrupted world. A stronger requested guarantee cannot be silently weakened.
Restoring a partial interaction with a remote environment/tool requires its
state and effects to be recoverable too. Unknown external effects do not imply
exactly-once execution, rollback, or safe retry.

Required unfinished gradients, derivative continuations, retained numerical
versions, or backend work must be preserved or reconstructible with evidence;
otherwise capture waits for a supported boundary or reports that coverage is
unavailable. This is not a requirement to serialize autograd graphs. The bounded
two-pass profile in G2.5 captures only at an unperturbed, quiescent boundary.
Similarly, a unit that advanced before another failed is not rolled back by
snapshotting: capture needs a supported, fully accounted-for position, or
recovery uses the last prior coherent snapshot rather than replaying the
uncertain attempt.

**Restore request, owner reconstruction, and readiness result.** Infrastructure
checks snapshot coverage and required dependencies against the accepted
obligations, restores logical identity/arrangement and owner contributions,
and establishes fresh execution readiness. This is dependency-ordered
coordination, not a fixed model-load/optimizer-load/step-reset sequence.
Correspondence uses semantic identity, membership, state definitions, and
revisions—not tensor ordering, authored addresses alone, or wrapper identity.
Current stage and installed relationships must be restored before dependent
construction; a completed transformation must not accidentally execute twice.

Recreated objects and backend representations can realize the same saved
definitions and revisions. Saved route meaning and dependency associations
survive; process-local handles and old readiness evidence do not become valid
merely by loading them. Required preparation establishes current evidence
before affected use. Backend/world-size portability or state migration requires
explicit supported correspondence; it is not implied by logical identity.
Required missing state cannot silently become a fresh optimizer, reset sampler,
or different input stream while reporting exact restoration.

A new execution session has its own observation and attempt identity while
same-run participant, relationship, and optimization identities remain intact.
Late results from old attempts are not automatically current. Captured work or
externally completed work can be adopted only through the accepted recovery and
admission rule, with its actual provenance and outcomes.

**Earlier capture versus later restore eligibility.** Suppose snapshot K at
accepted scope A1 contains live participant P, and a later accepted scope A2
retires P, possibly declaring successor Q at the same authored address. Once
that retirement is established, same-run restoration cannot publish K's P as
live again: retirement is terminal for that incarnation, and neither a new
execution session nor Q's matching address overrides it. K retains its original
capture guarantee; the incompatible continuation is reported unavailable,
while A2 and known or uncertain post-cut effects retain their historical meaning.
Late observations cannot erase that retirement or substitute Q for P. A later
eligible coherent cut may be used if one exists and its required correspondence
is established. This does not prohibit every earlier cut or numerical-state
restoration; other cuts remain subject to the accepted continuation policy,
effect reconciliation, and fresh readiness. Branching around terminal same-run
retirement would require an explicit different recovery/identity policy, not a
G5 storage or type choice. This is Pass 7's derived consequence of D6 and the
existing restoration rules, not a new universal rollback guarantee.

Fallible reconstruction happens while affected execution is unavailable.
Coordinated publication/readiness exposes only compatible restored state, never
a mixture of restored and newly initialized required owners. Each activity
still uses its applicable readiness checkpoint; restoration does not force all
optional routes to be prepared before unrelated work. A failed restore reports
which requirements remain unsatisfied and leaves dependent work stopped.
Process/rank loss does not certify a partial installation or live rollback.
Initializing a new run from whatever weights are available is a separate
explicit request with new identities and lineage, not an automatic fallback.

**Evidence and implementation boundary.** The existing identity, optimization,
preparation, and execution specs already establish the constituent ownership,
revision, readiness, and failure rules; `training-capability-coordination`
now files their whole-run snapshot/restoration exchange. The research sketch's
independent producer and staged-topology cases expose why input, producer,
stage, and progress contributions matter. The experiments demonstrate local
handoff/failure/preparation facts, not integrated durable capture or recovery.
Current Accelerator state hooks are useful owner-contribution evidence, not
proof of complete accepted-run coverage. G5 must choose formats, contributor
interfaces, capture protocols, backend mappings, and executable conformance
tests without inheriting the old fixed setup order.

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

### D14. xDSL is the working substrate for a bounded shared language slice

#### Substrate decision and its evidence boundary

On 2 November 2026 the user selected **xDSL as the working substrate for the
system machinery**. Python authoring and ordinary composition of maintained
operations are fixed premises for either candidate. This choice is about the
machinery's representation, analysis, transformation and lowering capabilities,
not authoring-language preference, setup convenience or user-written passes.

The [matched machinery record](../../../docs_design/models-strategy-trainer/machinery_comparison.md)
and [comparison plan](../../../docs_design/models-strategy-trainer/matched_comparison.md)
provide the bounded six-case evidence. Both xDSL and Pliron represented those
meanings with independent frontend verification/preparation. They reused the
existing owner runtimes; neither implemented a second ownership system or
established a general composition language. The recorded CPU observations and
test totals do not establish an intrinsic framework ranking. Historical
measurement/provenance qualifications and the unresolved legacy isolation issue
`sd-scripts-syv.31.23` remain as recorded.

xDSL supplies actual nested regions, typed def/use graphs, IRDL definitions,
traits, verification, rewriting, reconstruction and pass organization. Its
MLIR-aligned model and documented interchange offer a useful multi-level path:
retain high-level workflow contracts while lowering eligible local regions into
supported compiler representations. This is an architectural opportunity, not
qualification of an MLIR numerical target or automatic availability of MLIR
passes for custom workflow operations.

Pliron remains a credible alternative. Its explicit graph/SSA region kinds,
interface-driven conversion and managed analysis preservation/invalidation are
concrete strengths. Those do not currently establish a capability advantage
that outweighs the selected xDSL representation and lowering direction. Shared
Rust dependencies with a numerical backend also require an actual reusable
interface, analysis or artifact before they count as an integration advantage.

The working baseline is xDSL 0.74.0, matching the retained experiments. Version
changes require the affected structural, semantic and correspondence checks.
The runtime, numerical backend, production module placement and public authoring
API remain G5 decisions. In particular, this decision does not select Burn or
require ordinary numerical work to leave its Python/PyTorch implementations.

Reconsider the substrate if a required region/flow/resource contract cannot be
represented or preserved by the chosen framework, if a required analysis or
transformation is materially constrained by its structural facilities, or if
another framework demonstrates relevant reusable machinery at our actual
boundaries. State the specific obstruction and version, preserve the evidence,
and bound that investigation first. A version-specific workaround or an unused
MLIR opportunity alone is not evidence that the whole architecture must change.

#### Adopted shared contract boundary

Use the [language proposal](research/xdsl_language_0_1_proposal.md), especially
sections 2–5 and its small next slice, as amended below, as the input for one shared language
mechanism. Adopt the semantic obligations below, rather than its illustrative
class fields, operation spellings or full 0.1 catalogue.

Maintained semantic definitions expose typed descriptions derived from the
actual IR. They are shared knowledge used by verification, analysis,
normalization and preparation. Ordinary authors select/wire these definitions;
they do not supply validators or arbitrary scheduling callbacks. Registration
may select an operation's semantic implementation; case names, source spellings
and one fixture's exact operation counts cannot decide whether a whole strategy is valid.

| Contract surface | Required information in this slice |
| --- | --- |
| Region | Entry arguments and captures; invocation sources; child protocol; admission and closure; required completion, interruption and settlement relationships; declared quiescence boundaries, or a declaration that none is supported |
| Flow | Producer and consumer references; item/request association requirements; pending-product/disposition obligations; relationships needed to check producer-stop coverage; version admission policy over the actual production version |
| Resource | Selected access realization from the closed set below and a separate freshness/version-selection policy; required projections/leases, permitted uses and captures, effect/authority envelope, last required use and supported transfer/release boundary |
| Decision descriptor shared by region/flow/resource contracts | Reproducibility declaration for policies, admission rules and access modes; rank-consistency declaration for decisions that affect collective work; no separate conformance authority |
| Realization | Installed implementation/contract requirements, supported protocol and target limits, prepared fragments and correspondence to semantic operations/obligations |

The initial normalized descriptions cover once/sequence, parallel join, serial
repetition and requested-handler protocols, with explicit producer/control and
access relationships. These describe static templates and their permitted
runtime invocations; records, current model contents, queues, futures, mutable
owner records and physical handles are not semantic IR objects. An ordinary
leaf call does not need a new long-lived owner merely to participate.

Finite choice, richer flow/cardinality operators, grants and governed transitions
remain part of the broader design vocabulary. Tasks 6–9 implement only the
protocols necessary for the first shared slice and its declared target. A valid
contract that a target cannot realize must produce an explicit target/readiness
limitation at the earliest authoritative boundary, not a weaker algorithm or a
new language-wide restriction inferred from the fixture.

#### Invocation, pending work and settlement rules

An operation origin identifies a static definition. An invocation/request
identity identifies one occurrence under an enclosing invocation. Neither a
training step, model epoch nor an item key substitutes for that identity.
Capture mappings explicitly connect enclosing values/references to the body's
entry arguments; independent invocations do not share locals by convention.

For the bounded protocols:

- A sequence admits dependent work only after its declared successful child
  boundary. The first target uses settled child success unless an explicit flow
  relationship provides earlier availability.
- A parallel consumer group joins its required drivers and requested work.
  A handler template is not itself an admitted request. Its request sources
  must close, and every accepted pending/admitted request and entered use must
  settle or receive an accepted disposition/transfer before the group settles.
- Acceptance of a request creates an enclosing ownership obligation before
  handler execution admission. Waiting for input does not remove that debt.
  Completion-generated requests are registered, or their accepted dispositions
  become owned, before that completion becomes visible for parent settlement.
  A bounded-capacity wait retains the emitting invocation's obligation.
- Orderly stop follows the selected semantic admission boundary. Already
  admitted work retains its promised completion-generated requests and may
  finish them. Run cancellation can close broader admission, but pending work
  then has an explicit cancelled/unfulfilled disposition; it is not reported
  as executed. Required physical drain and cleanup continue.
- A completion-triggered producer stop must cover the permitted consumer and
  pending-request lifetime closure, unless an explicit disposal/transfer rule
  makes the earlier boundary valid. Consumer-group settlement requests producer
  stop without waiting for producer termination. Root settlement separately
  waits for both. Natural exhaustion, failure and cancellation retain their
  own accepted disposition policies.
- A lease or protected view cannot escape as an ordinary yielded value
  or capture. All admitted uses retain protection through required physical
  work, validation and cleanup, or through an explicitly supported transfer.
  Detached products can outlive a scope only when their contract discharges or
  transfers the remaining dependency. Settlement can coexist with failure and
  does not establish restored resource usability.

The verifier queries these relationships through registered contracts over
actual regions, captures and references. It checks known whole-lifetime cycles
separately from input waits and initialized stream feedback. It does not claim
general deadlock freedom or termination of arbitrary selected computation.
Open total production is permitted with bounded live obligations and a supported
control-progress contract. Fixture capacities and a particular selection quantum
are parameters, not universal language constants.

#### Amendments to the language proposal

The following rules extend the research proposal. Each is cheap to fix in the
contract algebra now and invasive to retrofit once operations and runtimes
assume otherwise. The shared rules are normative in the `workflow-composition`
capability; training composes them and adds only domain meaning.

**Shared contracts are not training contracts.** Region, flow, settlement,
access-mode, ownership, reproducibility, quiescence and rank-consistency rules
live in `workflow-composition`. `accepted-training-execution` keeps units,
windows, grants, derivative routing and other training-domain meanings. A
data-only composition must be valid against `workflow-composition` alone.

**Lifecycle semantics have an executable reference.** A deterministic,
Torch-free reference interpreter over normalized IR defines each invocation
protocol as a state machine over the proposal's control skeleton
(created → pending → admitted → running → closing → settling → terminal), with
the command/event protocol of proposal §2.1 as its transitions. Parent
completion and settlement are computed from child states. Selected computations
are replaced by declared stand-ins that honor their contracts and supply fixture
values. The verifier and prepared runtimes are checked against it, including
across systematically or randomly explored interleavings at suspension points.
Neither the verifier nor a runtime is the sole definition of lifecycle
correctness.
Determinism is relative to explicitly supplied external completion, input and
scheduling choices. The reference derives legal admissions and request
obligations itself; a runtime's admitted-work trace is not the oracle's input.
Comparison preserves owner-visible meaning while allowing private scheduling
and bookkeeping differences.

**Access realizations are a closed, versioned set.** No mode is implicit;
every resource use selects a realization and states its freshness requirement
separately. `current_at_start` is a version-selection policy, not a competing
access realization: an evaluation can select both current-at-start freshness
and protected live access.

| Mode | Meaning | Evidence a target must establish |
| --- | --- | --- |
| `snapshot` | Read an immutable published version independently of later owner updates | Coherent publication, selected version identity, isolation and the snapshot's lifetime |
| `protected_current` | Read live owner state while excluding incompatible updates over the required use extent | Atomic freshness/protection acquisition, physical-use completion with device ordering, and collective ordering when sharded |
| `owner_update` | Mutation through the owning protocol | The owner's ordinary update protocol |
| `grant(profile)` | Scoped delegated authority | The profile's phase and handback evidence |

Snapshot selection names a published-version rule or current-at-start
materialization; the latter requires atomically resolving and isolating that
version before use. Protected live reads select current-at-start freshness,
or the version at the request for requested work (below);
owner updates and grants carry their owning protocol's freshness requirements.
Concurrent producer and update activities may progress between protected uses,
but an incompatible mutation cannot race a protected use. A target rejects an
unsupported realization/selection combination rather than substituting one.
The existing owner engine's protected production epochs realize
`protected_current`; snapshot realization is new work.

**Requested work may select the version at its request.** A use inside a
handler may select the version of owner state that was current when its
request was made (working spelling `at_request`), in place of the version
current when the handler starts. A request is made at the point its
obligation is first owned: by the emitting invocation while it waits for
request capacity, or by acceptance. It is never the later point at which the
handler is admitted. From that point two things exist, with different
lifetimes:

- The request's dependency on that version stays with the request and its
  results until they settle.
- A reservation keeps the live state at that version. The coordinator holds
  it with the request, as it holds other reservations and protection (D15);
  it is not participant or authority state. The owner stays the authority
  for which version exists and is current, and cooperates by making
  conflicting updates wait while the reservation stands. Where the use
  selects protected live access, the reservation ends when the work's uses
  complete. Where a detached realization holds the version, such as a
  selected snapshot or a product's capture, it ends as soon as that
  realization is complete, and updates proceed.

No lease leaves its scope, and the handler still takes its lease when it
runs. A request that ends with a cancelled or unfulfilled disposition
releases the reservation. A target that can establish neither a live
reservation nor a detached realization rejects the selection. Products still
record the version actually observed, which here is the requested one. The
selection removes the dependence on when the handler runs. It does not remove
a dependence on when the request is made: where another activity can update
the same state around that point, the composition's reproducibility class
still reports it.

**State selection, admission and placement are independent.** Which version a
use observes, whether work may be admitted onto a device beside other work
(D15), and where work runs are three separate decisions. None is inferred
from another: work on the same device does not thereby see the same version,
a snapshot does not make work concurrent, and work on another device does not
thereby see an older version. They are independent in meaning, not in
progress, because a reservation and a device admission can wait on each
other. Two rules cover this. Preparation rejects a combination whose wait
cycle it can prove, as a target limit. At run time, work never waits while
holding a device for something that may need that device to progress: it
obtains its state conditions and admitted inputs first, or gives up the
device while it waits.

**Flows declare version admission.** A flow whose products depend on versioned
owner state declares exact-current, bounded-lag or any-with-provenance
admission. Admission evaluates the version actually observed during production.
A rejected product receives an explicit disposition, not a consumed outcome.
The policy names each relevant owner/incarnation and version lineage, its lag
metric, and the admission boundary at which comparison state is captured.
The first target can use owner update ordinals within one lineage; an ordinal
is not a global training clock. Products depending on several owners retain
the required version map. Incomparable versions require an explicit policy
disposition. Exact-current at admission does not promise freshness at later
execution entry; a demand for that guarantee needs an atomic entry check under
the corresponding resource-use contract.

**Compositions report a reproducibility class.** Access modes, policies and
admission rules declare whether their observable outcomes are
schedule-independent given the same inputs, seeds and configuration.
Verification derives a composition's class, names the operations that make it
schedule-dependent, and reports unknown where a declaration is missing.
A live read permits schedule dependence when writes can change its selected
version; accepted ordering can establish a fixed version instead. Deterministic
snapshot selection establishes only the version-selection part of the claim.
Admission, input association, RNG and other decision contracts still contribute
to the composition's class. State the assumptions and distinguish a proven
schedule-independent result from possible schedule dependence or unknown.

**Ownership is a type discipline.** Admitted item claims are affine: consumed,
transferred, explicitly forked or given a disposition exactly once on every
path. Leases are region-scoped and cannot escape except through an
accepted owner transfer. Both are checked from types and region contracts as
one generic rule set, not as per-composition validators.

**Quiescence and rank consistency are declared before they are realized.**
Every maintained region contract declares where a consistent cut may be taken,
or that none is supported; exact restoration and governed transitions use only
declared boundaries. Every decision affecting collective work under a
replicated target is declared replicated-deterministic or decided once and
distributed. Tasks 6–9 record these declarations and reject claims that depend
on boundaries or agreement they do not provide; implementing coordinated cuts and replicated execution is outside
this slice. The first target is single-process, where rank consistency imposes
no restriction.
Rank-local randomness may feed a decision made once and distributed before
dependent collective work. Rejection applies to decisions without the required
agreement, not to the mere presence of a rank-local random draw.

#### Compiler stages and implementation placement

Keep accepted meaning, normalized descriptions, prepared forms and runtime
instances distinct under the same revisioned run obligations. Normalization
resolves explicit captures, expands maintained derived words and exposes request/stop
relationships; it cannot invent input, lifecycle or authority meaning. Preserve
the accepted document and origin ancestry. Semantic summaries and saved
projections do not authorize execution independently of verified IR/contracts.

Preparation consumes those checked relationships and installed implementations to
derive inspectable fragments for the selected realization policy. Cold loading
rechecks source coverage, captures/types, contracts, request/stop/access claims
and policy correspondence before dependent readiness or effects. Legitimate
pure construction or backend preparation before final verification remains
permitted under D8; the old experiments' pre-factory rejection checks are not
a universal construction order.

Use an isolated research package at
`docs_design/models-strategy-trainer/xdsl_language/`. Its recurring stage is
shared language construction/verification/lowering for multiple compositions;
the existing `xdsl_candidate` and Pliron packages retain the bounded experiment
evidence. Anticipated internal seams are `contracts.py`, `dialects.py`,
`normalize.py`, `reference.py`, `verify.py`, `prepare.py` and `target.py`, with fixtures,
independent expectations, tests and a scope README. Task 5.5 still selects production
placement and dependency direction; these research filenames are not public APIs.

The target reuses the original owner engine's structured primitives and
protection/physical services. Existing composition math, implementations, gates and
oracles can be reused; calling the current fixed-profile
`pliron_candidate/composition_runtime.py` whole-case builder is not the new
generic lowering. Domain behavior remains in selected implementations. One shared
core lowering may dispatch on registered semantic protocols, but may not return
a preassembled whole-case graph or start another whole-run coordinator for the
data-only control.

Language-only construction/verification stays Torch-free. The existing owner
target may retain its current Torch import dependency; record it rather than
inventing a dummy model/optimizer for data-only meaning or claiming a new
Torch-free runtime. Use the existing qualified environments, observe actual
GIL state without forcing it, and preserve backend qualification limits.

Definition-analysis validity, prepared-realization validity and compiled/data
validity are distinct dependency systems. Ordinary numerical/state updates do
not change the accepted definition merely because they create another runtime
epoch. Binding/contract/structural changes invalidate affected dependent facts
and realizations under the governing transition rules. Begin with conservative
recomputation of inexpensive summaries rather than introducing a second cache
authority.

The selected MLIR direction is an explicit closed computational-island boundary
with qualified types/ABI, numerical/effect/failure policy, source correspondence
and physical completion. Design those contracts as needed; actual MLIR codegen,
whole-run async lowering and a numerical-backend migration are outside tasks
6–9. No generic purity/DCE inference may erase observable coordination,
validation, requests or outcomes just because local arithmetic is pure.

#### Selected effects, filtering and a local transformation

The next bounded advance after tasks 6–9 is task group 10. It corrects two
acceptance gaps and tests whether one additional data-composition boundary and
one transformation enter through reusable contracts. It does not reopen the
substrate comparison, implement the wider language catalogue or settle G5.

**Availability follows execution.** Each maintained ordered body makes its
entry arguments and preceding local results available. Captures must be
available at the capturing operation; sibling-role entries, self-use and later
local definitions are not available. Initialized repetition carries are entry
arguments, not forward references. Graph execution and other recurrence need
their own availability rules. Diagnostics identify the offending operand/use
and its definition. Apply these checks before and after derived words are
expanded.

**Selected work has one minimal effect contract.** Begin with declared resource
reads, declared updates and possible additional unknown effects. Requirements
name actual argument positions and the resource type/identity relationship
they require; reusable implementations do not acquire a global owner by name. Resolve
each invocation's actual leases through its SSA captures and issuing
accesses. Preparation and local transformation legality consume the same
authoritative declaration. Keep declaration, granted authority, actual returned
effects/outcomes and physical settlement distinct. Unknown effects constrain
rewriting and do not authorize undeclared resource mutation. Preserve runtime
active-authority/current-version checks. Missing legacy effect information is
conservative, never evidence of purity; preserve existing saved forms or
reject an explicitly unsupported format without rewriting historical evidence.

**Filtering is actual zero-or-one production.** An independently live source
feeds an intermediate filter producer and a retained-output flow. A filter
invocation owns one admitted input claim and returns either no downstream item
or one detached retained output with its actual input/version provenance. Drop
and forwarding settle that input exactly once after its required physical use;
an output reservation that produces no item also releases its capacity. The
source and retained-output horizons count all owned reservations, ready items
and admitted claims under their respective contracts. Transport capacity alone
does not establish the aggregate bound.

The concrete data fixture retains keys 0, 2 and 4 and drops intervening keys.
The first retained consumer requests two additional retained consumptions.
Dropped admitted inputs do not fulfill that demand or finish the consumer
group. A small local chain `(x + 1) * 2` yields independent expected products
2, 6 and 10. Consumer-group completion requests filter stop; filter settlement
then requests source stop. The group does not await producer termination to
request stop. Hold actual filter/source work to distinguish group completion,
physical drain, late failure and root cancellation. Bounded run-ahead is allowed
with explicit dispositions; no identical scheduling trace is required.

**One local rewrite must earn its legality.** Use actual xDSL rewriting and SSA
result remapping to combine compatible adjacent unary pure stages. Eligibility
requires explicit known effects, compatible types and wiring, and preserved
authority/completion/validation contracts. A generated local realization may
remove generic local dispatch while retaining each stage's logical invocation,
entry guard, physical completion, validation and failure outcome. This does not
collapse distinct physical calls or move an effect across a protected boundary.
Retain an executable opaque identity/offset between otherwise compatible
stages; unknown effects block grouping, and a forged group crossing that
barrier rejects on cold loading. Purity does not make coordination, validation
or stage outcomes removable.

The extension retains independent reference/owner checks, live and cold paths,
diagnostic erasure, wrong-implementation and corrupted-claim controls, and
normal/optimized child qualification. Existing scalar-gradient conventions and
optimization-unit construction remain bounded target realizations rather than
language rules. No host dispatch count establishes framework/device performance.

#### Scoped resource-backed products

Task group 11 and bead `sd-scripts-syv.31.29` extend the language/owner evidence
with one lexical retained-product contract. A physical call may return one
explicitly typed retained payload whose declared dependency positions name
actual leases or earlier retained products. Borrowers receive the actual
payload, without cloning away storage aliases or autograd dependencies. They
have read-only use; selected implementations remain trusted to honor that
contract. The product retains its dependencies and observed versions until an
explicit release after every borrower and derived product settles.

This first profile keeps products within their defining ordered body. A
product cannot be yielded, captured into another region, published to a flow or
implicitly copied; broader ownership transfer requires another explicit
contract. Each normal path releases exactly once. Failure/cancellation cleanup
releases remaining products in reverse dependency order after entered physical
work drains. Known updates to a retained owner reject before execution; runtime
checks remain authoritative for active products, lease versions, dependent
uses and reported updates. Actual storage stays outside IR. Detached results
retain their existing isolation convention; failed isolation is not permission
to infer an ownership contract.

Use a read-only buffer view and a real CPU autograd product passed between
physical calls. Independent stand-ins establish arithmetic and lifetime facts;
target evidence establishes actual aliases and graph retention. Preserve cold
correspondence, diagnostic erasure, interruption and late failure. Retention
boundaries block the existing detached-only fusion. This bounded profile feeds
G5 types and representation choices; it does not qualify device ordering,
mutable aliases, arbitrary escaped autograd graphs or production reclamation.

#### Bounded milestones and success criteria

Tasks 6–11 and Beads `sd-scripts-syv.31.26` / `sd-scripts-syv.31.28` / `sd-scripts-syv.31.29` provide
evidence for tasks 5.3/5.4:

| Milestone | Bead | Delivered boundary |
| --- | --- | --- |
| 6 | `sd-scripts-syv.31.26.2` | Typed contract descriptions, including access-mode, version-admission, reproducibility, quiescence and rank-consistency fields; actual xDSL structure and normalization; the Torch-free reference lifecycle interpreter |
| 7 | `sd-scripts-syv.31.26.3` | Shared pending-work/source-closure, stop-coverage and lifetime verification; ownership checked as affine claims and region-scoped leases; access-mode, admission and reproducibility-class checks; each verified against the reference |
| 8 | `sd-scripts-syv.31.26.4` | Derived prepared setup and live/cold execution through existing owner services, realizing both `snapshot` and `protected_current` explicitly; runtime lifecycle facts compared differentially with the reference |
| 9 | `sd-scripts-syv.31.26.5` | Integrated cancellation, late failure, archive-erasure and fresh-process qualification, with interleaving exploration against the reference across the supported compositions |
| 10 | `sd-scripts-syv.31.28` | Earlier SSA and implementation-position rejection, one shared effects contract, actual zero-or-one filtering across requested demand, and one justified local rewrite with an executable barrier |
| 11 | `sd-scripts-syv.31.29` | Scoped resource-backed products, explicit dependency/release contracts and actual storage/autograd use through cancellation and physical settlement |

Each milestone includes its own tests, scope documentation, relevant quality
gates and required review before the next starts. Milestone 9 integrates prior
checks; it does not defer earlier milestones' tests. Use the existing
producer/driver/requested-evaluation meaning, a data-only producer/consumer
control, and at least one valid rearrangement such as an extra consumer scope.
The shared verifier/normalizer/preparer must handle that arrangement without a
new whole-case rule. Retain invalid early-stop and lease-escape controls,
actual independent numerical/causal facts, physical draining and distinct late
failure outcomes. Diagnostic histories may be erased without changing execution.

Failure to express the two domains through the shared protocol is evidence to
revise that protocol, not permission to add case-name branches. This scope does
not implement the full language 0.1 catalogue or freeze serial-request limits
as the final language. Section 5's production-shape tasks and production apply
remain separately gated even after this research completes.

### D15. Production architecture

This decision begins G5. It gives the established system its production shape
at package level: layers and dependency direction, runtime owners, the
execution model on real hardware, the Trainer's role, and what existing code
and the research packages become. Tasks 5.1–5.5 complete the types,
representation, contract and authoring API, and module placement inside this
frame; task 5.6 turns it into numbered milestones. Those tasks stay open until
their own acceptance criteria and the G5 review are complete.

The system is designed as a whole. Maintained families and data-only
workflows are strategies, and PEFT methods and capability implementations are
registered implementations within it, not inputs that reshape it. As D14
already requires for the language, a case-specific branch in the workflow,
run-state or training layers is a defect in the shared design.

#### Layers and dependency direction

```text
  launch        train.py and the run config; composes the contract
                     |
  authoring     library/strategies: strategies and their reusable pieces
                     |
  training      library/training: training's words, Trainer, optimization,
                governed realization and preparation
                     |
  run state     run authority: arrangement and obligation revisions,
                participants, relationships, bindings, routes, views
                     |
  workflow      library/workflow, the IR system: Kuroiro, its Python
                front end, fulfillment, reference semantics, runtime

  libraries: artifacts, models (components and assemblies), conditioning,
    objectives, losses, timesteps, data, optimization, adapters, pipelines,
    evaluation, performance
  observability: metadata, logging
  base: utils, constants, vendor    external: xDSL, PyTorch, backends
```

A layer imports only layers below it, observability and base. Launch,
strategies and training also import the libraries; the run state and the IR
system never do. Libraries import neither training nor strategies, and
training receives features only as installed implementations. D21 states the
exact rules. Observability receives typed facts and results, owns no live run
state (G4.5) and imports only base and the fact types it consumes.

| Layer | Owns | Does not own |
| --- | --- | --- |
| Launch | Reading the run config; composing the contract and the installed implementations; selecting a maintained strategy and its options; running fulfillment; starting the Trainer | Automatic assembly, validity rules, run state |
| Authoring | The strategy system: strategies and their reusable pieces, written against the contract, with the readable contract and the guide | The contract's composition, fulfillment, execution, current run state |
| Training | Training's words in the contract (its operations and ownership profiles) and their realization; the Trainer; optimization realization; governed realization and preparation coordination | Fulfillment, participant state, input policy and coordination, capabilities, features |
| Run state | The run authority (D6–D7): arrangement and obligation revisions, participants, relationships, bindings, routes, views, freshness and transitions | Loading, backend work, optimizer state |
| Workflow | The IR system (D14, D16): Kuroiro's common, data-flow and model-use operations, contract interfaces, the Python front end, fulfillment (reading, normalization, verification and obligation derivation), the accepted-run type, diagnostics, artifacts and cold loading; the reference semantics; the runtime | Domain meaning, live models |

This direction removes today's import cycles. Training and strategies import
each other; config imports adapters, data, models and optimization while each
of those imports config; objectives, adapters, data and pipelines import
training or strategies. In the target:

- Only launch reads the run config, and it applies the config's values to the
  strategy's declared parameters (D18). Below launch, code receives accepted
  settings through the accepted run and its selected implementations, and
  launch infrastructure such as output locations and logging sinks from the
  launcher.
  Feature and method settings types belong to their owners; config composes
  them.
- Nothing below launch imports a strategy. Training defines its words. The
  launcher registers them with the other definitions as the contract,
  imports the strategy and hands the strategy function to fulfillment in the
  IR system. Fulfillment produces the accepted run, and the Trainer consumes
  it.
- Selected implementations are registered by explicit composition, never by
  import-order discovery (D2). The runtime calls them by their registered
  names without importing them.
- The run authority sits below training. A workflow that uses models without
  training does not import the training layer.
- The workflow language and reference semantics never import Torch. The
  runtime core does not either. Torch-facing pieces such as execution context,
  device completion and tensor isolation for snapshots are adapters that load
  only when a prepared target selects them.

#### Runtime model

The system runs on the CPU and drives the GPU. One coordinator per process
executes the prepared run. Selected implementations perform the numerical work
and launch device work. Owners hold state.

- **Coordinator.** There is one per process, so one per rank. It admits work,
  starts and settles activities, routes requests, and tracks claims,
  reservations and protection. That control state is runtime state, never
  semantic authority, and it stays with the coordinator.
- **Visible boundaries.** The coordinator acts at the boundaries the strategy
  exposes: activities, training actions, requests and protected uses. Inside a
  boundary, prepared straight-line code runs the selected computation without
  per-operation dispatch. Static choices are resolved before the hot path
  (D10).
- **Physical calls.** Selected implementations run on worker threads, and the
  coordinator never blocks on them. Each call establishes its execution
  context from its prepared call: grad mode, module mode, precision, device
  stream and RNG. It never inherits ambient thread state.
- **CPU/GPU hand-off.** A Python call returning means device work was queued,
  not finished. The physical layer obtains completion from the device through
  a backend adapter, using stream ordering and events rather than a CPU
  synchronization after every call. Protection, snapshots and retained
  products release on that completion fact. This is the device-ordering
  evidence that `workflow-composition` requires for protected live reads.
  Adapters cover host execution, where return is completion, PyTorch CUDA
  streams and events, and the project's own backend through its completion
  contract. Flow bounds and retained-product release also bound the device
  memory held by produced items.
- **Device capacity.** What fits on a device is not a question about state,
  so it is decided apart from which version a use observes (D14). A strategy
  states only meaning: whether work of different activities may overlap. It
  cannot command that the work fits. Work is presumed not to overlap unless
  the strategy permits it. Preparation and the coordinator decide whether
  permitted overlap is realized on the selected target, from the facts the
  device adapter supplies and the measured peaks in resource observation;
  otherwise the runtime runs the work in turn. A strategy whose meaning
  requires work to be live together on a device that cannot hold it is
  rejected as a target limit (D18). Implementations declare that they use a
  device (D18). Work that holds no device memory needs no admission: a save
  holds the device only while it captures state, not while it serializes and
  writes. Preparation and the coordinator also decide the extent that one
  admission covers, such as a call, a window or a handler's traversal. The
  adapter only supplies device facts and realizes acquisition and
  completion. The extent is settled in `tasks.md` section 14, under D14's
  rule on waiting. A participant resident on a device counts toward its
  capacity, and a strategy may declare that a participant leaves the device
  between uses; how that is realized is settled in the same section.
- **Collectives.** On multi-rank targets, every rank must issue collective work
  in the same order, or the ranks hang. The runtime orders collective-issuing
  work identically on every rank, for example through one ordered lane per
  process group. Scheduling decisions that determine what enters that order are
  replicated-deterministic or decided once and distributed, realizing D14's
  rank-consistency declarations. A single loop gets this ordering for free;
  concurrent activities must establish it. Activities that issue no collectives
  stay independent.
- **No-GIL.** Running without the GIL is the performance target, never a
  correctness mechanism. Correctness rests on the ownership rules, which hold
  either way: one writer, protected access, explicit hand-offs and affine
  claims. A selected implementation is presumed unsafe for concurrent use of
  the same participant unless its declaration says otherwise, and the runtime
  serializes such uses. Imports that would re-enable the GIL stay off the
  default path and load only when a selected implementation needs them; a
  run reports its actual GIL state. The
  [blockers report](../../../cpython-314t-blockers-report.md) records the
  current Triton, OpenCV and gRPC limits.
- **Input production.** Producers run as activities in the same process, in
  parallel with training when the GIL is off. A source may still use private
  worker processes internally.

#### The Trainer

The Trainer has two roles:

1. **Host.** It receives a fulfillment result, seeds the run authority from
   it and sequences governed realization and preparation through their
   coordinators, then publication, execution, interruption, failure
   handling, cleanup and finalization. It does not own the state those steps
   publish.
2. **Standard-mechanics owner.** Under the standard profile it realizes the
   training operations: progress, accumulation, synchronization, backward,
   clipping, optimizer and scheduler advancement, zeroing, triggers and
   observation (D10). An extension profile transfers named mechanics only
   through an explicit grant (G3.2).

The strategy's repeated training actions are the loop; there is no separate
fixed loop. D5's "Trainer's loop" row is realized this way. Participant state
belongs to the run authority, input policy to producers, capability semantics
to capability implementations and model behavior to selected implementations.

#### Runtime owners

| State | Writer | Notes |
| --- | --- | --- |
| Arrangement, obligations, participants, relationships, bindings, routes, views, revisions | Run authority | Single writer; consumers receive scoped projections (D6) |
| Interruption, cleanup | Trainer | |
| Optimization units and their definition revisions, realizations, optimizer and scheduler state, and each unit's progress and accumulation window | Trainer optimization | Optimizer and backend objects are realizations (D9); progress is counted per unit (D20) |
| Backend coordination: wrappers, inseparable groups, distributed state | Trainer runtime infrastructure | Published with routes and optimization runtime in one coordinated publication (D8) |
| Input policy, cursors and buffers | Producers, in `library/data`, each with its cut state | D5 input row |
| Input readiness, admission and hand-off records | The IR runtime's streams, in `library/workflow` | D5 input row; D21 |
| Operation-owned state, such as adaptive objective or sampling state | Its accepted operation | Continuity rules in G2.6 |
| Capability readiness, requests and results | The IR runtime's request and handler protocol, with each capability's implementations | G4 exchanges; D21 |
| Metrics, validation history, traces | Observation owner | Metadata and logging consume facts |
| Invocations, claims, reservations, requests, protection | Workflow coordinator | Runtime control state only |

#### Representation and authoring

These set the direction that tasks 5.3 and 5.4 complete and verify.

- **Accepted run.** The accepted IR document, its obligation revision, the
  contract and profile identity, the selected implementations with their
  versions, and the authority seed. The Trainer seeds the run authority from
  it. Normalized and
  prepared documents
  derive from it. The prepared strategy is inspectable but never an independent
  authority, and cold loading re-derives correspondence before any effect
  (D14).
- **Executable run.** The lowered runtime instance: the coordinator, owner
  services and prepared code over currently published state. Live models,
  tensors and optimizers never enter the IR; owners hold them and expose
  scoped projections.
- **Contract.** The contract is the IR system's registered vocabulary,
  composed once by the launcher and read in one place. Each operation
  definition carries its region, flow, resource and realization contracts.
  Training adds its operations and ownership profiles; capabilities and
  features declare themselves from their own packages. Together these are
  D3's three surfaces. One body of knowledge serves author guidance,
  verification and preparation.
- **Fulfillment.** The IR system's acceptance step, which the launcher runs.
  The authored strategy becomes proposed IR, which is normalized and verified
  under the shared rules and the training rules registered with training's
  definitions. The result is an accepted run, or diagnostics attributed to
  their source. The normal launcher constructs the Trainer only from a
  fulfillment result.
- **Authoring.** Authors write strategies and their reusable pieces against the
  contract; fulfillment is never part of that work. Users select a maintained
  strategy and set its options through the run config. D18 records how
  strategies are written.

#### What existing code becomes

D21 lists what in the new system covers today's code, and where the research
packages go.

#### Acceptance and sequencing for the milestones

- **Generality.** New families, methods, capabilities and workflows enter as
  registered operations, implementations and reusable strategy pieces. A
  milestone that changes the workflow, run-state or training layers records
  why; a case-specific branch there fails review.
- **Feature cost.** Each milestone records what its typical feature required:
  the layers touched and the declarations its author wrote. The design's
  distinctions belong in shared machinery and defaults rather than in every
  operation author's work.
- **GPU early.** The CUDA completion adapter and one real SDXL training action
  on GPU land in the earliest milestones, alongside the workflow core, rather
  than after a complete CPU path. The Migration Plan's order otherwise stands.
- **Conformance and cost.** G3.7's conformance and execution-cost criteria
  apply to every milestone, measured on GPU for training paths.

#### Work this frame leaves open

- **Tasks 5.1–5.5:** concrete state and optimization types, the
  representation, the contract and authoring API, module placement within
  these packages, and the run authority's package. D16 and D17 record the
  representation, D18 the contract and authoring API, D19 and D20 the state
  and optimization types, and D21 the module placement and the run state's
  package.
- **Task 5.6:** the numbered implementation milestones.
- **Targeted qualifications that do not block this frame:**
  - the CUDA and own-backend completion adapters;
  - the collective-ordering mechanism across ranks;
  - concurrency safety of selected implementations without the GIL;
  - the remaining vocabulary: grants, governed transitions, batching and
    correlation, parallel merging, nested resources and coordinated cuts;
  - exact restoration across owners;
  - bounded retirement of workflow bookkeeping (`sd-scripts-syv.31.30`).

### D16. The workflow layer and the Kuroiro language

This decision gives D15's workflow layer its production design. It covers the
workflow part of task 5.3: how the shared language turns an accepted
description into a running run, and how domains extend it. The training side
of 5.3, meaning what the Trainer consumes, builds on it, as does the Python
authoring face that task 5.4 designs.

#### Name and identity

The shared language is called **Kuroiro**, a working name. Its operations print
under the prefix **`iro`**. The name reads as 黒色, "kuro's color", the
recognizable look of the project's own language. It also contains IR, with the
"o" for ownership, which is what this IR adds over an ordinary one.

The `iro` prefix stays separate from the `kuro.*` metadata namespace already
written into saved models, so a later repository rename changes neither.
`workflow-composition` remains the descriptive spec name, and D15's
package name stays provisional.

xDSL is the machinery underneath. Kuroiro's vocabulary, rules, written form,
diagnostics and authoring surface belong to the project.

#### What the layer does

The layer turns a description of a run into a running run, then keeps the
description's promises while it runs. It has no training, data or model
meaning of its own beyond the shared data-flow and model-use operations D15
places here. Domains plug in through four seams.

```text
  Python authoring
     |  builds
  proposed strategy
     |  normalize  (expand derived words; explicit captures, requests, stops)
     |  verify     (lifetimes, ownership, access, admission)
     |             -> source-attributed diagnostics, or:
  ACCEPTED STRATEGY + obligations         saved with the run
     |  prepare    (match installed implementations and a target;
     |              check effects; fuse local chains)
  PREPARED STRATEGY                       inspectable, never authoritative
     |  lower      (runtime scopes, wiring, straight-line code)
  RUNNING: coordinator and owner services; devices do the numerical work

  reference semantics: executes the accepted strategy's lifecycles
  independently; every runtime realization is checked against it
```

#### Four seams

The layer's boundary consists of four seams. Its analyses query contracts
through them and never select behavior by operation name, fixture or case.

| Seam | What a domain supplies | Examples |
| --- | --- | --- |
| Definitions | Operations and types, each with region, flow, resource, realization and decision contracts (D14) | Training units, windows and advancement; data production and filtering; model invocation and views |
| Owners | Resource owners that realize the access modes (snapshot, protected live read, owner update, grant) with versions and evidence | The run authority for participants; Trainer optimization for weight updates; producers for their cursors and policies |
| Implementations | Selected code, each piece declaring its inputs, outputs, effects, concurrency safety and completion | Family conditioning and denoiser calls, objectives, codecs, serializers |
| Devices | Execution context and physical completion for one backend, and what it knows of the device's capacity | Host execution; PyTorch CUDA streams and events; the project's own backend |

Registration through every seam is explicit composition (D2).

#### Extending the language

The two kinds of extension have different costs:

- **A new operation built on existing protocols.** This is the common case:
  a new objective, transform, family, method or capability operation. It needs
  a definition, an implementation and, where useful, a reusable strategy
  piece. The workflow layer does not change.
- **A new lifecycle protocol.** This is a new way for work to live, wait or
  end, such as grants, governed transitions or correlation. It changes the
  shared language and ships as one unit: the protocol's reference semantics,
  verifier rules and runtime realization, with differential and interleaving
  conformance tests.

An implementation cannot introduce a lifecycle, an admission rule or an
authority that its operation's protocol does not define (`training-contract`,
`workflow-composition`).

#### What an access covers

Two shared rules fix what a lease protects and for how long.

- **Until the last use of what depends on it.** An implementation declares
  when its result keeps a dependency on leased state, such as a loss that
  still carries its graph (D18). That result is a retained product (D14). It
  keeps the state protected after the access body ends, until the result's
  last use. Authors write no release: normalization places it after the last
  use, which for a loss is the end of backward inside `contribute`, and the
  normalized form shows it. Such a result may leave its access body but not
  its activity; it is never yielded into a stream or captured by another
  activity. The lease itself still never leaves its scope. This extends
  D14's first retained-product profile by one step.
- **Everything its route depends on.** An access to a participant goes
  through a route, the normal one unless another is named, and covers what
  that route depends on (D19). The lease protects all of it, the use records
  each version, and a reservation for requested work holds each version
  (D14). An adapter that acts inside its host makes the host's normal route
  depend on it, so a host access covers the adapter. A use that must not see
  the adapter goes through a different route. A unit's evaluation point is a
  route too (D20).

Neither rule limits what a participant can be. An adapter that is called on
its own has its own route and is accessed like any participant, and one that
does both is reached both ways.

#### Correctness and saved forms

- **Conformance.** Each protocol has one runtime realization. It is checked
  against that protocol's reference state machine under explored
  interleavings. Facts visible to owners must agree; private scheduling and
  bookkeeping may differ (D14). A concurrency defect in a protocol's
  realization then shows up as a conformance failure, not only in production.
  That matters once the GIL is off.
- **Obligations.** D7's run-specific obligation set is what verification
  derives from the accepted strategy under the pinned definition and
  implementation versions. The accepted strategy and those versions never
  change during a run. The set's revision advances when a declared transition
  brings further obligations into force or ends some (D17). A saved
  obligation summary is a projection and never authorizes execution on its
  own.
- **Saved forms.** The accepted strategy is saved in its written form with the run's outputs,
  and snapshots reference it. The normalized and prepared forms are derived
  from it and saved for inspection. Cold loading and restoration parse them
  and re-derive them, checking correspondence before any effect (D14).

#### Written form and vocabulary

The research dialect prints in xDSL's generic form, which buries meaning in
attribute dictionaries. Each production operation declares its own written
form, following these rules:

- Nesting reads as lifetime.
- Relationships read as statements, for example
  `iro.stop @producer after @consumer`.
- Access modes appear on each use, and bounds and admission policies appear on
  each stream.
- Defaults are omitted, and source locations appear only on request.
- Saved strategies are parsed back on load, so the readable form is the saved
  form. It must round-trip exactly.

The vocabulary follows the design:

- Every first-class concept has exactly one word, and nothing else gets one.
  That word is the same in the design, in printed strategies, in Python
  authoring and in diagnostics.
- Research names are aligned to that rule. For example, `program` becomes the
  strategy. Claims receive a disposition rather than being "retired", because
  retirement already means ending a participant's or unit's use.
- Domain vocabularies print under the same name as sub-namespaces, for example
  `iro.train.window`.
- Diagnostics use the authored names, for example
  `iro verify: stop @producer after @training does not cover @evaluation's
  pending request`.

#### The Kuroiro vocabulary

The words come from the design's concepts. In the table, "exists" means the
research dialect already has the concept, under the name shown; "planned"
means production adds it.

Applying the one-word rule turned up four collisions:

- **capability:** D3 already uses it for Trainer-recognized capabilities such
  as caching and validation. The handle an access scope gives its body is
  therefore a **lease**.
- **binding:** D6 already uses it for a participant's current bound object.
  Selected code is therefore an **implementation**. Calls name it, and the
  strategy header pins its version.
- **retire:** D6 and D9 already use it for ending a participant's or unit's
  use. A claim is therefore **disposed**, with a disposition.
- **region:** this is xDSL's word for an operation's body. Kuroiro's word for
  a named piece of work with its own lifetime is **activity**.

| Area | Word | Meaning | Research name | Status |
| --- | --- | --- | --- | --- |
| Structure | `strategy` | The accepted strategy: the root operation, which one run executes | `lang.program` | Exists |
| Structure | activity | A named piece of work with its own lifetime, such as a named producer, repeat, handler or parallel; a concept, not an operation | — | Exists |
| Structure | `once`, `sequence`, `parallel`, `repeat`, `choice` | Run a body once; run children in order; run children concurrently until all settle; repeat a body under a policy; run exactly one accepted alternative | Same names | Exists |
| Structure | `derived` | A derived word: one defined by expanding into other words, such as `window` | `lang.recipe` | Exists |
| Structure | `fused` | A prepared-only group of local stages; authors never write it | `lang.fused_local` | Exists |
| Lifetimes | `producer` | An activity that keeps producing items into a stream | `lang.producer` | Exists |
| Lifetimes | `handler`, `request` | An activity that serves requests from declared sources; an owned pending request for one | `lang.requests`, `lang.request` | Exists |
| Lifetimes | `stop` | Close an activity's admission once another settles, or on an interrupt the strategy names (D17) | `lang.stop_after` | Exists; the interrupt source is planned |
| Lifetimes | `completed`, `settled` | Logical completion; everything finished, including physical work and dispositions | Boundary values | Exists |
| Data | `stream` | A bounded flow of items with an admission policy; the spec's "flow" | `lang.stream`, `endpoint` | Exists |
| Data | `take`, `claim`, `borrow`, `dispose` | Take the next admitted item as a claim; read a claim without owning it; settle the claim exactly once with a disposition such as consumed, forwarded, filtered_out or cancelled | Same names, `retire` | Exists |
| Data | `filter`, `optional` | Turn one claim into zero or one items | Same names | Exists |
| Data | `fork`, `transfer`, `batch`, `correlate` | Duplicate or hand over a claim; group items; join items by key | — | Planned |
| Access | `access`, `lease` | Use a resource within a scope; the handle the scope gives its body, which cannot leave it | `lang.access`, `capability` | Exists |
| Access | `snapshot`, `protected_current`, `owner_update`, `grant` | The closed set of access modes (D14); grant profiles come from domains | Mode values | Exists; `grant` planned |
| Access | `current_at_start`, `published`, `initial` | Version selection, separate from the mode | Freshness values | Exists |
| Access | `at_request` | Version selection for requested work: the version current when the request was made; a reservation held with the request keeps it until the work completes or a detached realization holds it (D14); a working spelling | — | Planned |
| Access | `exact_current`, `bounded_lag`, `any_with_provenance` | Stream admission policies | Admission values | Exists |
| Access | `retained`, `release` | A result that keeps its storage alive beyond its call; its explicit release | `retained`, `release_product` | Exists |
| Calls | `call`, `constant`, `yield` | Invoke an implementation by name; a literal; return from a body | Same names | Exists |
| Calls | `uses` | The strategy header that pins each implementation's version | Per-call version | Exists, moved to the header |
| Model | `resource`, `materialize` | Declare something an owner holds, such as a participant; name the implementation that materializes it | `lang.resource` | Exists; `materialize` planned |
| Model | `participant`, `view`, `route`, `invoke` | A model resource with identity; its typed projections and execution routes (D6); invoke a route under an access mode | — | Planned |
| Model | `relationship` | A declared relation between participants, such as an attachment, with its own identity and state (D6) | — | Planned |
| Training | `unit`, `window`, `contribute` | An independently advanced optimization responsibility (D9), whose attributes include its members, optimizer policy, joint windows and clipping scope (D20); contributions advanced together; offer an addressed input to a unit | — | Planned |
| Training | `advance`, `zero`, `trigger` | Advancement and zeroing, written only where a profile grants them and otherwise performed by the Trainer; a policy that makes requested work due | — | Planned |
| Outputs | `persist`, `transition`, `cut` | Write a durable output and report what was written; propose a governed change that the strategy declares; take a consistent cut at a declared boundary | — | Planned |
| Declarations | reproducibility, rank rule, `quiescence` | Printed only when not the default | Attribute values | Exists |
| Runtime | invocation, obligation, cancel | One occurrence of an activity; what verification requires the run to discharge; run cancellation, which closes admission while entered work settles | — | Exists |

Conventions for the written form:

- Activities list their captures explicitly. A capture keeps its name inside
  the activity unless `as` renames it.
- Local bodies such as access, derived and choice use enclosing values
  directly; parsing turns those uses into the same explicit operands.
- Each statement in a `parallel` is one concurrent branch.
- Types print by name. Literals print inline. Unused entry arguments and
  default attributes are omitted. `//` starts a comment.

The data-only strategy below is the research's accepted data-only fixture
printed in this form. The training strategy is the research lifetime fixture
written with the planned training words: its hand-written gradient and update
calls become a window, a contribution and Trainer-owned advancement.

```text
// Data-only: score records under a selection policy that learns from
// feedback.
iro.strategy @score_records uses(@select v1, @score v1, @feedback v1) {
  %policy  = iro.resource @policy
  %records = iro.stream @records : record
               bounds(outstanding 3, ready 2)
               admit any_with_provenance(@policy)

  iro.parallel @activities capture(%records, %policy) {
    iro.producer @producer into %records capture(%policy) {
    ^(%index: index):
      %record = iro.access %policy protected_current current_at_start {
      ^(%lease):
        iro.call @select(%lease, %index) -> record
      }
      iro.yield %record
    }

    iro.repeat @consumer 4 times capture(%records, %policy) {
      %claim = iro.take %records
      %value = iro.borrow %claim
      %score = iro.derived @score.derived {
        iro.call @score(%value) -> scalar
      }
      iro.access %policy owner_update {
      ^(%lease):
        iro.call @feedback(%lease, %score)
      }
      iro.dispose %claim consumed
    }

    iro.stop @producer after @consumer
  }
}
```

```text
// Training: learn from produced records, then evaluate once on request.
iro.strategy @learn_then_evaluate
    uses(@encode v1, @objective v1, @equal v1, @measure v1) {
  %encoder = iro.resource @encoder : participant
  %model   = iro.resource @model : participant
  %unit    = iro.train.unit @learner of %model optimizer @sgd
  %records = iro.stream @records : encoded_record
               bounds(outstanding 3, ready 2)
               admit any_with_provenance(@encoder)

  iro.parallel @activities capture(%records, %encoder, %model, %unit) {
    iro.producer @producer into %records capture(%encoder) {
    ^(%index: index):
      %encoded = iro.access %encoder snapshot published initial {
      ^(%lease):
        iro.call @encode(%lease, %index) -> encoded_record
      }
      iro.yield %encoded
    }

    iro.parallel @consumers capture(%records, %model, %unit) {
      iro.repeat @training 2 times capture(%records, %model, %unit) {
      ^(%iteration: index):
        // The Trainer advances %unit each time a window completes.
        iro.train.window %unit members 2 {
          %claim = iro.take %records
          %value = iro.borrow %claim
          %loss = iro.access %model protected_current current_at_start {
          ^(%lease):
            iro.call @objective(%lease, %value) -> loss
          }
          iro.train.contribute %loss to %unit
          iro.dispose %claim consumed
        }
        %due = iro.call @equal(%iteration, 1)
        iro.choice %due {
          iro.request @evaluation(%iteration) from @training at settled_success
        }
      }

      iro.handler @evaluation sources [@training] max 1
          capture(%records, %model) {
        %claim = iro.take %records
        %value = iro.borrow %claim
        iro.access %model protected_current current_at_start {
        ^(%lease):
          iro.call @measure(%lease, %value)
        }
        iro.dispose %claim consumed
      }
    }

    iro.stop @producer after @consumers
  }
}
```

#### Production work in this layer

Milestones 6–11 established the core. The research dialect has structure,
producer, request, filtering, access, call, claim and retained-product
operations, with verification, reference semantics, preparation and cold
loading. Production work:

1. Rebuild the core as `iro`, with the four seams as explicit interfaces and a
   declared written form for every operation, starting from the vocabulary
   above. The written-form grammar is completed in this step.
2. Build the domain vocabularies, training first. The connected candidate
   already established training's semantics: units, windows, gradient routing,
   advancement and grants. The xDSL slice drove training only through fixture
   implementations.
3. Add what a real run needs:
   - durable outputs that report what was actually written;
   - a consistent cut for checkpoint and resume, realizing the quiescence
     declarations;
   - rank-consistent collective ordering (D15).
4. Add the device adapters, starting with PyTorch CUDA (D15).
5. Bound runtime bookkeeping for long runs (`sd-scripts-syv.31.30`). The
   research keeps every finished invocation's facts for inspection, which a
   long run cannot afford.

### D17. Accepted and executable training runs

This decision completes the representation half of task 5.3 for training. It
covers what fulfillment hands the Trainer, what the Trainer publishes and
executes, and how a strategy, read into Kuroiro, becomes a training run on
GPU. The workflow layer (D16) supplies the language, verification and
runtime; this decision adds the training content and the Trainer's use of
it.

#### What the Trainer consumes

The Trainer consumes exactly these inputs:

1. **An accepted run.** Only fulfillment produces one; in tests, a test-only
   builder may (D1).
2. **Launch settings** from the launcher: output locations, logging sinks, and
   the device and rank topology.
3. **The installed implementations**, which the launcher composes explicitly
   (D2).
4. **For a resume only, the snapshot to restore from.**

It receives no strategy object, no run config and no mode. Every choice those
used to carry is now in the accepted strategy.

#### The accepted run

An accepted run is immutable and holds no live objects. It records a content
identity for its strategy and pinned versions. The run's own identity comes
from the run authority it seeds, so two runs of the same strategy remain
distinct runs (D6).

| Part | What it is |
| --- | --- |
| Strategy | The accepted strategy, in Kuroiro: participants and how each one materializes, relationships, units with their optimizer policy, streams, activities, transitions, triggers and capability handlers. Configuration values are attributes in the strategy. |
| Pinned versions | The contract and ownership profile, and every implementation with its version |
| Obligations | The revisioned set that verification derives from the strategy under those versions (D16), including the readiness checkpoints that fulfillment derives |
| Authority seed | The declared participants and relationships, still unbound, under the strategy and obligation revisions. It is derived from the strategy, and the Trainer creates the run authority from it at the start of realization (D6, D19) |

The strategy is the source. The other parts are derived from it and kept for
convenience, never as independent authority.

An accepted run says that the run is meaningful and permitted, and what must
be shown before each use. It says nothing yet about loaded models or devices.

#### Realization and the executable run

As host, the Trainer carries an accepted run to execution:

```text
  accepted run
     |  seed         create the run authority from the strategy's
     |               declarations
     |  materialize  run each participant's declared materializer as a
     |               governed job; the run authority publishes the bindings
     |  prepare      one coherent attempt (D8): placement, casting, wrapping,
     |               compilation and unit realization (optimizers, groups,
     |               schedulers), plus workflow preparation of the strategy
     |  publish      one coordinated publication of routes and views (run
     |               authority), optimization runtime (Trainer optimization)
     |               and backend state
  executable run
     |  execute      the coordinator runs the lowered strategy; owners serve it
     |  finalize     settle and report outcomes
```

Participants that the strategy's first activities use are materialized
before execution. A participant first used later is materialized by a
declared transition (below). Resources that name the same materializer are
materialized in one attempt, as when one released checkpoint supplies several
participants. A materializer that needs another participant's loaded
structure, such as an adapter's after its host, takes that participant's
inspection view, so the order follows from that access (D19).

An executable run exists only after publication. It holds:

- the prepared strategy and its lowered runtime instance;
- the published projections it runs against, identified by the authority,
  obligation and preparation-attempt revisions they came from;
- the selected device adapters;
- the evidence accepted for its readiness checkpoints.

An executable run is single-use:

- **Governed transitions.** A transition, for example attaching an adapter
  mid-run, re-enters preparation under D8 and publishes a new executable
  state. Results from the old state cannot publish against the new one.
- **Restoration.** Restoration realizes the same accepted run from a cut,
  using contributors' state instead of fresh initialization (G4.4). The cut's
  current stage says which transitions have taken effect, and so which
  obligations are in force. Creating a new run instead requires a separate
  request.

#### Everything a run can become is declared

A run changes only in ways its strategy already wrote. Every participant,
relationship and unit a run may ever have is declared in the accepted
strategy, and every change to the arrangement is a `transition` the strategy
states. Nothing is declared while a run executes.

- **Fulfillment verifies the whole strategy, including each transition's
  after-state,** as it verifies every alternative of a `choice`. It derives
  the obligations each transition brings and keeps them in the accepted run,
  each with the readiness checkpoint at which it first has to be met.
- **A transition brings those obligations into force.** The run authority
  evaluates evidence against them in the same atomic transition (D19). It
  derives nothing, so the Trainer needs neither the contract nor the verifier
  while a run executes (D18), and the accepted run stays immutable.
- **What depends on run-time facts is still declared.** When a transition
  happens, which declared targets it applies to within stated bounds, and
  which external file it loads may all be decided while running. The file is
  evidence, checked when it arrives.
- **Growth uses declared-but-unbound participants.** A strategy that adds an
  expert whenever error passes a threshold declares its experts up to a
  maximum, all unbound, and materializes the next one when the condition
  holds (D19).

Two kinds of change cannot be declared: one nobody planned, such as a person
deciding mid-run to add an adapter, and one that needs code or definitions
the run did not pin. Each is a new run, with its own fulfillment, started
from the earlier run's products with lineage (D11).

#### How training operations reach the Trainer

The Trainer joins the workflow layer through the owner seam. Trainer
optimization owns every unit and realizes the training operations as its
owner protocol:

| The strategy says | Trainer optimization does |
| --- | --- |
| `iro.train.contribute %loss to %unit` | Backward from the loss into the unit's gradient accumulation, deferring synchronization until the window's last member |
| The end of `iro.train.window` | Synchronize, clip, step the optimizer and scheduler, zero, and record progress (standard profile) |
| `iro.train.trigger @every_500 on %unit` | Evaluate the trigger policy over training progress; the strategy then requests the capability work that is due |
| `iro.access %unit grant(profile) { ... }` | Hand the profile's named mechanics to the granted implementation for that scope, and verify the handback before resuming its own mechanics (G3.2) |

A window is a maintained derived word over existing protocols: a repeat over
members, then an owner update of the unit. It needs no new lifecycle protocol.

Capability work such as validation, sampling and persistence runs as handlers
that triggers request; the IR runtime coordinates the requests, and each
capability's implementations own their state (D5, D21). Reads
of model weights use `protected_current`, so no advancement races a read in
another activity, such as validation's.

Which weights such work sees is stated, never left to scheduling. The shared
rhythm in `strategies/features/` requests validation, sampling and saving
with the version at the request (D14), so work requested at one point in a
unit's progress observes the weights of that point. The rhythm's handlers
read with protected live access by default, so advancement waits until that
work completes. A strategy that selects a snapshot instead lets training
continue once the snapshot holds that version, on the same device or
another. A save reserves the version only until its capture completes (D11).
A strategy that wants the weights current when the handler starts selects
`current_at_start` instead.

A trigger may follow a producer's progress, such as the end of a pass over
its data, as it follows a unit's. Reporting a value, such as a step loss or a
validation result, is an owner update on the observation owner (D15); it
needs no word of its own.

An interrupt is an orderly stop, not a cancellation. The host closes
admission of the activity the strategy names for it, usually its training
repeat, and the rest of the strategy, such as a final save, runs as written.
A second interrupt cancels the run.

#### On GPU

- **Execution.** A window's body runs as prepared straight-line code on a
  worker thread: take, forward, loss, contribute. The coordinator acts only at
  window and request boundaries.
- **Completion.** Completion comes from the device. Releasing protection after
  an optimizer step waits on that step's event, not on a CPU synchronization
  (D15).
- **Hand-off.** Items carrying device tensors are handed over with an event.
  The consumer's stream waits on that event, and the producer's memory stays
  reserved until the consumer's use completes.
- **Multiple ranks.** DDP's gradient reduction, validation metrics and
  checkpoint gathers all issue collectives in the rank-consistent order that
  D15 requires.

#### An SDXL fine-tune, in outline

```text
iro.strategy @sdxl_finetune uses(...) {
  %unet  = iro.resource @unet : participant materialize(@load_sdxl_unet)
  %te1   = iro.resource @text_encoder_1 : participant materialize(...)
  %te2   = iro.resource @text_encoder_2 : participant materialize(...)
  %vae   = iro.resource @vae : participant materialize(...)
  %unit  = iro.train.unit @denoiser of %unet optimizer @adamw
  %batch = iro.stream @batches : latent_batch bounds(outstanding 4, ready 2)

  iro.parallel @activities capture(...) {
    iro.producer @data into %batch capture(%vae, %te1, %te2) {
      ... decode, augment, encode latents and conditioning ...
    }
    iro.parallel @consumers capture(...) {
      iro.repeat @training until @max_steps capture(...) {
        iro.train.window %unit members 4 {
          ... take, predict with @sdxl_denoise, loss with @ddpm,
              contribute, dispose ...
        }
        ... triggers request @validation, @sampling and @checkpoint ...
      }
      iro.handler @validation sources [@training] max 1 ...
      iro.handler @sampling sources [@training] max 1 ...
      iro.handler @checkpoint sources [@training] max 1 ...
    }
    iro.stop @data after @consumers
  }
}
```

#### Acceptance and readiness stay distinct

The two types make the distinction structural:

- An accepted run cannot execute.
- An executable run exists only after the evidence for its readiness
  checkpoints is accepted and its state is published.
- Fulfillment rejects known incompatibilities. Contingent failures in
  loading, the backend or devices stop realization, and nobody reinterprets
  the accepted strategy (D5).

| Task 5.3 requirement | Where it lives |
| --- | --- |
| Acceptance distinct from readiness | The accepted run versus the executable run |
| Authoritative obligation revision | Derived by verification, pinned in the accepted run, referenced by every published projection |
| Maintained and custom structure | Derived words expand during normalization; custom operations enter through D16's first kind of extension |
| Imperative regions | `grant` access with a profile and a verified handback |
| Capability state | Each capability's implementations, acting as owners in the executable run, with the IR runtime coordinating requests (D21) |
| Binding authority | The run authority, seeded from the accepted run |
| Prepared routes | Preparation proposals that the run authority publishes |
| Optimization runtime | Trainer optimization's unit realizations, published with the routes |
| Readability | The printed accepted strategy and the inspectable prepared strategy |
| Hot path | Static resolution before execution, coordination only at boundaries, and G3.7 measurement |

D19 and D20 give the concrete run-state and optimization types for tasks 5.1
and 5.2. D18 covers how fulfillment builds the accepted run and how
construction outside fulfillment is prevented.

### D18. The training contract, fulfillment and authoring

This decision answers task 5.4. It covers what the contract is, how
maintainers write strategies against it, and how fulfillment turns a strategy
into an accepted run (D17). It also places each piece of that logic.

#### The contract

The contract is the IR system's registered vocabulary: everything a strategy
may use, in one versioned list that is read in one place. In xDSL, a parser
and verifier know only the dialects registered with them, and the contract is
that registration. The launcher makes it once, explicitly, before any
strategy is read. `train.py`, help, the doctor and tests all use the same
composition.

Each entry describes one piece: what it does, what it takes, what it connects
to and its rules. There are four kinds of entry:

- **Operations and types,** such as `stream`, `window` and `unit`. An
  operation's region, flow, resource and realization contracts (D14, D16) are
  the parts of its entry that the IR system checks.
- **Recognized capabilities:** validation, sampling, caching, persistence and
  restoration, each with its request, result, readiness and failure meanings
  (D5, G4).
- **Feature contracts:** what a feature offers and requires, from its
  declaration (D21), such as an adapter method, EMA or distillation (D12).
- **Ownership profiles:** the standard profile's advancement policies (G3.1)
  and the extension grants (G3.2), each with its phase owners, state access,
  effect bounds, handback and failure meanings.

Each entry's rules are the rules the verifier runs, so what a reader sees is
what gets checked. An entry's code sits with the piece it describes, in the
module D21 gives, and nobody needs to look there to read the contract. One
place to read does not mean one large object: the contract has no rules of
its own beyond its entries.

#### Authoring: the Python face of Kuroiro

Writing a strategy is where everything trainable starts, so starting one must
not be a challenge. A strategy is still code that someone writes: the
repository's maintainers, or whoever is training something. It states how the
run works, and it is not a detailed configuration for a hidden engine. Not all
such code is part of the repository, and a strategy may hold code that is
relevant only to it, such as an implementation that no other strategy uses.
That code is part of the strategy.
Adding to what the system offers every strategy is the separate topic of
extending the system (D21). Configuration is a user-facing filter: it only
narrows the choices and parameters a strategy declares.

A strategy contains three kinds of code, and an author always knows which kind
they are writing:

- **Activities** are functions, written in a subset of Python, that say what
  one piece of work does over its lifetime. Examples are a producer that
  yields items, a training activity that takes batches and contributes, and a
  handler that answers requests. The system reads their source into Kuroiro,
  the way Triton and Taichi read kernels. Their loops, branches, takes, yields
  and requests become the strategy's structure in Kuroiro. An activity is
  read, never run as written: what a run executes is the strategy as it was
  read and accepted (D16).
- **Implementations** are ordinary Python and PyTorch functions that do the
  math, such as encoding, denoising and evaluation. They run as written and
  are called through their declarations.
- **The strategy function** is ordinary Python that runs once while the
  strategy is read. It declares participants, units and streams, and starts
  activities together with their relationships. Its parameters are what
  configuration can filter.

```python
@iro.producer
def produce(dataset, vae, text):
    for item in dataset.shuffled():           # runs ahead of training
        yield encode(vae.snapshot, text.snapshot, item)


@iro.activity
def train(batches, unet, unit, max_steps):
    for step in range(max_steps):
        for batch in unit.window(batches, members=4):
            unit.contribute(denoise(unet.protected_current, batch))
        if step % 500 == 0:
            validate.request(step)


@iro.handler
def validate(step, unet, heldout):
    return evaluate(unet.protected_current, heldout)


@iro.strategy
def sdxl_finetune(model_path, dataset, heldout, max_steps: int = 20_000):
    unet, text, vae = sdxl.participants(model_path)   # declared, not loaded
    unit = iro.train.unit("denoiser", of=unet.attention,
                          optimizer=adamw(lr=1e-4))
    batches = iro.stream("batches", prefetch=8)
    with iro.parallel():
        produce(dataset, vae, text, into=batches)
        train(batches, unet, unit, max_steps)
        validate.serve(unet=unet, heldout=heldout)
        iro.stop(produce, after=train)
```

Reading activities follows fixed rules:

- `for step in range(n)` becomes a repeat.
- `yield` in a producer emits into its stream.
- An item taken from a stream or a window is a claim. Passing it to an
  implementation borrows it. Unless the activity disposes it explicitly, it is
  disposed as consumed when its scope ends normally. If the scope ends early,
  it gets the failure or cancellation disposition instead.
- `handler.request(...)` is a request.
- An `if` on a runtime value is a choice.
- Attribute access such as `.protected_current` or `.snapshot` on a
  participant selects the access mode. The lease lasts for the call it is
  passed to, and a result declared to depend on it keeps the state protected
  until that result's last use (D16).

Activities are written in a restricted subset of Python. They may call
implementations, use participants and use the language's words, and may do
nothing else with hidden effects. Anything outside the subset is rejected,
with its line. Code that has to run as ordinary Python belongs in an
implementation or in the strategy function.

Pieces connect only to what the author hands them. Activities and strategies
can be reused, and a strategy can build on another without forking it.

Implementations are declared with a decorator that records their contract:
signature, version, effects, concurrency safety and completion. Which
participants a call reads or updates is not written a second time. The lease
types in the signature say it, and verification checks each call site's
access mode against them (D14). The decorator states only what the signature
cannot show, such as use of randomness, external input or output,
collectives, use of a device, anything kept beyond the call, and state
private to the implementation. The decorator has no registration side effect;
the launcher installs implementations explicitly (D2).

```python
@iro.implementation(version=1, effects=["device"])
def denoise(unet: ReadLease, batch: LatentBatch) -> Loss:
    ...  # the selected PyTorch computation; the lease type says it reads
```

Implementations remain trusted numerical code (D14). The system guarantees
coordination according to these typed and declared effects; it does not
sandbox Python. Qualification tests may detect some undeclared mutation, such
as an in-place write through a read lease, and reject the implementation, but
that detection is never part of the guarantee.

The design reads source rather than recording builder calls, because
recording would need a special call for every decision made at run time. The
choice affects only the front end. Both approaches produce the same Kuroiro
form, so verification and the runtime do not depend on it. The first real
maintained strategy confirms the direction, and the front end defines the
subset's exact boundaries.

The authored activity is never executed as the run's code, although that
would make debugging direct. What a run executes must be exactly the accepted
strategy that was printed and saved, on a first launch and on a resume.
Executing the Python beside it would make the function a second authority
that could differ from what was accepted without anyone seeing it. Instead,
every operation keeps its origin through normalization and preparation (D14),
and lowered code carries the authored file and line wherever one authored
line corresponds. Runtime failures, traces and debuggers then point at
authored lines.

#### Authoring support

Four aids help an author write a strategy that works with what is allowed:

- **The contract** itself, readable as a reference: every entry with its
  parameters, connections, needs and rules. It is a readable file,
  `library/strategies/CONTRACT.md`, generated from the composition, and it
  points to where available features are listed, which D21 leaves open.
- **Help and the guide.** CLI help lists the strategies, their parameters and
  each word. The written guide explains how to write a strategy, with worked
  examples. The guide is not part of the contract: it explains the contract,
  and the contract decides. Its reference parts are generated from the
  contract and the doctor checks its examples, so it cannot drift. Help also
  lists every method in the repository with its status: allowed under the
  current contract, or present but not yet allowed, with the reason.
- **The doctor.** It runs fulfillment's checks on a strategy without loading
  anything, prints the strategy as read, and explains each problem in the
  strategy's own names, with its line and a fix. It can also step through
  the run's lifecycle in the reference interpreter with stand-in
  implementations. That
  shows, for example, that data runs ahead, that training takes batch 3, and
  that validation is requested at step 500, all before any GPU is involved.
- **Guided construction, later.** Every entry declares what it connects to,
  so a guided builder can offer only the choices that fit at each point.

Something the contract allows cannot fail a run just because a strategy uses
it. Fulfillment refuses only what would fail, and each refusal is an early
error that names the failure it prevents: an interface that cannot connect,
an authority the profile does not grant, or a construct the language cannot
express. Everything else runs. Guidance, such as provenance, tested
combinations and community practice, produces help and logs, never a refusal
(D21): an untested pairing, or a use that conflicts with a component's
declared meaning, such as fine-tuning an eps model toward v-prediction, is
allowed and logged. What depends on loaded state is checked at its readiness
checkpoint: for loaded models before training starts, and for a capability
when it is requested. Some spec rules still reject choices that would not
fail, such as a PEFT method outside a strategy's declared set
(`adapter-system`). The rejection-rule audit (task 5.7) brings each such rule
under this one or records the failure it prevents.

#### From run config to accepted run

```text
  run config        strategy name, plus values for its declared parameters
     |  launcher    compose the contract; resolve a preset, such as today's
     |              `mode`, to a maintained strategy; import the strategy;
     |              hand the strategy function and the config's values to
     |              fulfillment
     |  fulfill     the IR system: run the strategy function once and read
     |              its activities into the proposed strategy; normalize;
     |              verify under the shared rules and the training rules;
     |              check parameters against their declared bounds; derive
     |              obligations and readiness checkpoints
     v
  accepted run (D17), or a rejection with every diagnostic found
     |  Trainer     seed the run authority and begin realization
```

Fulfillment is the system's step, not the author's. The launcher runs it,
and the doctor runs the same checks earlier. Fulfillment never fills a gap.
It never selects a missing implementation, adds a dependency, or calls the
strategy back. Config can set only the choices the strategy declared, and
only within their bounds (`training-contract`). One contract version and
profile govern the run for its lifetime.

#### Diagnostics

A rejection reports every independent problem it finds. Each problem gives
its rule code, the authored names involved, the Python source line where the
offending element was written, the printed fragment and a suggested fix:

```text
iro verify [iro-life-004] strategies/sdxl/finetune.py:88
  stop @data after @training does not cover @validation's pending
  request; @validation still takes from @batches.
  fix: stop @data after @consumers, or give @validation its own stream.
```

A diagnostic distinguishes invalid meaning, which is a semantic error, from a
valid strategy that the selected target cannot realize. The second is a target
limit, reported at the earliest point where it is known (D14).

#### Bypass prevention

Only fulfillment produces an accepted run on the normal path, and the normal
launcher reaches the Trainer only through fulfillment. A saved strategy, for a
resume or a cold load, goes back through the same checks at the launcher,
under the versions it pins, before the Trainer sees it (D16). The Trainer
re-derives nothing, because it has neither the contract nor that job. It
checks only what it can know: that the installed implementations it receives
match the versions the accepted run pins. Otherwise it refuses to start.
Tests may build accepted runs with a test-only builder, which the normal
launcher never uses (D1). The builder lives under `tests/`, beside the IR
system's conformance suite, and no production module imports it. Selected Python is trusted (D14), so this is a
boundary on the normal path, not a defense against deliberately forged
objects.

#### Placement of contract logic

| Concern | Where it lives |
| --- | --- |
| The contract | One composition, made by the launcher before any strategy is read, versioned and read in one place |
| The code of each entry's rules | With the piece it describes, in the modules D21 gives. Readers never need it, because the composed contract shows every entry |
| Fulfillment | The IR system (D21). The launcher runs it, and the doctor runs the same checks earlier. Authors never do |
| Checks that need later facts | Some rules can be checked only once things exist, such as whether the loaded UNet has the layers a LoRA targets. The rule's entry includes its check. The run authority or preparation runs it when the facts arrive, at the rule's readiness checkpoint |
| Diagnostics | One shared format in the IR system. Each rule carries its own code and message, such as `iro-life-004`, and whichever tool ran the check prints them |

None of these is a separate validator system. Together they form one contract
system, organized by its entries.

This settles the deferred question of how custom operations are authored, as
a narrow combination:

- strategy functions in ordinary Python, with activity functions that the
  system reads;
- a declaration-only decorator for implementations;
- for new operations, xDSL definitions with contract traits, each with its
  own Python form.

### D19. Run state types

This decision answers task 5.1. It gives run state concrete types: an owner
pattern shared by every owner of run state, and the run authority's instance
of it. D20 gives Trainer optimization's instance for task 5.2. Both decisions
are written against D15–D18, so they already include the reconciliation that
the G5 gate requires of independently developed shapes.

The names follow the design's vocabulary (D6–D9, D16). Classes from the
research candidates, such as `Stamp`, `Image`, `RunState`, `UnitRuntime` and
`UnitOutcome`, remain evidence and do not carry over. The fields named here
are the ones the settled rules require, and D21 places them in modules.
This decision also answers three deferred production-shape choices: the type
names, how references are represented, and installation as a generation
switch over immutable sections.

#### What changes, and how often

Participants and optimization units have the same four layers, and each layer
changes at its own rate:

| Layer | Participant | Optimization unit | Changes when |
| --- | --- | --- | --- |
| Identity | A reference to one incarnation | A reference to one incarnation | Never. A change of identity creates a new incarnation |
| Meaning | The declaration: what it is and how it materializes | The definition: members, groups and policies | A governed change: rare, checked and all-or-nothing |
| Realization | The binding, which is the loaded state, and the routes and views prepared from it | The prepared unit: optimizer, scheduler and groups | A materialization or preparation attempt |
| Live state | The numeric version: which weights | Progress, optimizer state and the gradient window | Every step, by the owner with update access; the authority's records stay unchanged |

An optimizer step changes live state only. The authority's records change
when meaning or realization does: on materialization, bound-state
replacement, attachment, preparation, retirement and restoration. Those
happen at startup, at governed transitions and on resume, outside the hot
path, so the authority can check every change in full.

#### Where the IR system meets these types

These types realize what the accepted strategy declares:

- Participants are the strategy's participant declarations, such as
  `iro.resource @unet : participant`, and units are its `iro.train.unit`
  operations. At the start of realization the Trainer creates the run
  authority from those declarations (D17). Trainer optimization reads its
  unit definitions from the same strategy (D20).
- The run authority and Trainer optimization are owners at D16's owner seam.
  When the run reaches `iro.access %unet protected_current`, the
  coordinator obtains the lease from the authority's current generation.
  `iro.train.contribute` and the end of a window go to Trainer
  optimization's owner protocol (D17).
- A governed change and a resume cut appear in strategies as the planned
  `iro.transition` and `iro.cut` words. At run time they are a `Transition`
  and each owner's cut state.
- The IR system never holds live objects (D15). It refers to state through
  references, versions and dependencies, which access and admission check.

#### The owner pattern

Every owner of run state uses the same shared types: the run authority,
Trainer optimization, and later owners such as a capability's
implementation. A new owner fills these types in and adds no machinery to the
run-state layer (D15).

| Type | What it is |
| --- | --- |
| Reference | One incarnation of something an owner manages, as an opaque value made of the run identity, a kind and a serial number. Only its owner creates one, by minting it or by restoring it from a cut; other code only compares references, and none is ever reused. The authority mints the run identity when the accepted run seeds it (D17), and mints `ParticipantRef` and `RelationshipRef`; Trainer optimization mints `UnitRef` (D20). Cuts and facts store a reference as a string of its three parts and parse it back exactly. |
| `Version` | Which state a use saw: a reference, the revision of what is bound and a numeric version. Snapshot reads return it, produced items carry it as provenance, and stream admission policies compare it with the current version (D16). |
| `Dependencies` | What a projection or prepared realization relies on. The default is its whole source section; an exact set of revisions is the narrower alternative (D6). |
| Evidence | A typed value, one kind per obligation kind, declared beside the evaluator that judges it (D18). Each value records the attempt that produced it and the revisions it was produced under. A producer's own claim, or a Boolean, is never evidence. |
| Attempt | A governed job's identity, its basis (the revisions it started from) and whether it replaces state or mutates it in place. Its result is bound to it. `MaterializationAttempt` and `PreparationAttempt` are the two kinds. |
| `Generation` | The coherent published state: one section per owner, under a generation number. |
| `Publication` | The single swap that installs the next generation. |
| Cut state | An owner's part of a coherent recoverable cut (D16's `cut`; G4.4), with the identities and revisions its continuation needs. |

#### Publication

Each owner builds and checks its section of the next generation off to the
side: the authority its records, Trainer optimization its prepared units, and
backend coordination its wrappers and inseparable groups. One `Publication`
then installs every section by swapping the current generation. Every
fallible step has already run, so the swap does no fallible work and calls no
observers (D8).

- **Reading.** Code holds a generation without locking. A freshness check
  compares revision numbers, so routine checks at boundaries stay cheap
  (G3.7). A lease taken from one generation stays valid for its scope. A
  publication that conflicts with an outstanding protected use waits for it,
  is rejected or uses supported isolation (`training-runtime-preparation`).
  A reservation held for a pending request counts as outstanding use here
  (D14).
- **Withdrawal.** Each owner's section can withdraw the realizations that
  depend on changed revisions without fallible work. An authority transition
  and the withdrawals it causes therefore publish in one generation, and
  dependent work pauses until a preparation attempt publishes replacements
  (G3.5).
- **Layering.** `Generation` lives in the run-state layer, and owners register
  their section types with it. Training-side preparation coordination drives
  publication for training runs (D8). A workflow without training has no
  optimization section, and the run-state layer never imports training (D15).
- **Ranks.** Every rank publishes the same sequence of generations with the
  same revisions; only process-local objects differ. A publication swaps only
  after rank agreement, in the collective order that D15 requires. A rank lost
  at the swap aborts the in-process run, and recovery goes through
  restoration.
- **Concurrency.** Each section has one writer, published sections never
  change in place, and the swap is a single store. Correctness therefore does
  not depend on the GIL. The types hold live objects without importing Torch;
  Torch-specific evidence and evaluators live with the definitions that need
  them (D15).

#### The run authority's records

| Type | What it is |
| --- | --- |
| `Participant` | The authority's record: reference and authored address, declaration, lifecycle (declared, bound or retired), binding, routes, views and lineage |
| `Binding` | A participant's bound state: the loaded object, its binding revision and the evidence it was accepted with |
| `Route` | A named way to call the participant, such as `normal` or a route that a capability needs: its current prepared callable, route revision, dependencies and refresh rule. Its dependencies include participants that act inside this one through a relationship, and an access through the route covers them (D16) |
| `View` | A typed, revision-pinned look at one participant that is not a route: original, unwrapped, inspection, metadata or artifact |
| `Relationship` | Reference and address, endpoints, kind, operational state, revision and history. It changes independently of its endpoints |
| `TargetRef` | A reference into a participant's structure: the participant, the selected component, module or parameter substructure, its public selector and any declared component provenance. It holds no live object (`optimization-target-refs`). Where `expand-model-metadata-identity-and-structure` has resolved them, it also carries optional catalog and structural path associations, which never replace the semantic identity. Optimization, PEFT targeting and persistence share it, and sharing it makes none of them the owner of targeting (D12) |
| `Obligations` | The obligation set at its revision. Each entry names its subject, the readiness checkpoint at which it must first be fulfilled, the evidence kind it needs, its evaluator and the strategy location it came from (`training-contract`) |
| Authority section | The authority's part of a generation: these records at one authority revision. The specs call it the authority snapshot |

Four details follow from D6 and D7:

- "Binding" means only the bound state (D16). D6's one current binding per
  participant/route pair is one current prepared callable per route over the
  participant's single binding.
- A state-only participant, such as an injected adapter, has no route; its
  state is reached through the routes that depend on it (D16). An adapter
  that is called on its own has a route like any participant. A derived
  route under the same participant states its refresh rule.
- `declared` is D7's declared-but-unbound state. It is never a fake `None`
  candidate.
- Lineage records succession, merge or fold, and initialization from another
  run's artifact, separately from identity.

#### Changing the records

Producers never write records. They submit a `Transition`, the typed proposal
that D7 describes: the target identity, its kind, the attempt identity, the
expected revisions, the candidate and the evidence. The authority builds the
prospective section, checks the expected revisions and runs the evaluator of
every affected obligation. It returns `Published`, with the new revisions and
the realizations that the change invalidates, or `Rejected`, with diagnostics
and no change.

The kinds are a closed set: D6's operations plus D8's withdrawal.

| Kind | Effect |
| --- | --- |
| Declaration | Creates a declared, unbound incarnation of a participant or relationship that the accepted strategy declares. The authority seed supplies the initial ones; a later one is a successor at a declared address, never a new declaration (D17) |
| Materialization | Gives a declared participant its first binding |
| Bound-state replacement | Installs a new binding that still fulfills the participant's accepted meaning; otherwise retirement and a new declaration are required |
| Route rebinding | Installs a route's new prepared callable |
| Relationship transition | Changes a relationship's operational state |
| Arrangement amendment | Applies a change the accepted strategy declares as a `transition`, such as retiring participants and declaring their successors together. It brings the obligations that fulfillment derived for that change into force in the same transition and never derives new ones (D17) |
| Merge or fold | Folds one participant's state or effect into another, with lineage and optional retirement |
| Retirement | Ends current use of a participant or relationship; its reference and history remain |
| Withdrawal | Withdraws guarantees before destructive work, under the attempt's own identity (D8) |

#### Attempts

A `MaterializationAttempt` runs declared materializers, such as loaders, as
one governed job. One materializer may supply several participants, and one
that needs another participant's loaded structure reads its inspection view
(D17). Its basis records the authority and obligation revisions it
started from. It produces materialization or bound-state replacement
transitions with their evidence. Loading weights into an existing bound
object is in-place work, so such an attempt publishes its withdrawal first.

A `PreparationAttempt` is D8's one coherent job, derived from one generation.
The run-state layer defines the generic attempt, which a workflow without
training also runs. Its request names:

- the requested routes and views;
- every required participant, including frozen ones that still need
  movement, casting, wrapping or compilation;
- the permitted mutation;
- what each participating owner contributes.

Training's preparation coordination contributes the unit definitions to
realize (D20) and every inseparable backend group the attempt overlaps,
complete.

Its `PreparationResult` keeps its parts apart by owner: route rebinding
proposals and typed views, both for the authority; each contributing owner's
part; and the source revisions, constraints, guarantees, facts and structured
failures. In a training run the contributed parts are prepared units for
Trainer optimization and backend coordination state, which gives D8's five
parts. A result publishes
only while its basis is current. An older optimistic result can never restore
guarantees that a destructive attempt withdrew.

#### Projections

Consumers never receive the authority, a section or a generation. Each kind
of consumer receives its own projection type, pinned to one generation or to
exact dependencies:

| Consumer | What its projection holds |
| --- | --- |
| Execution | The leases that the coordinator hands out at boundaries, for call sites wired before the hot path (D16) |
| Preparation and Trainer optimization | The bindings, routes, target structure and backend groups that preparation and unit resolution need |
| Persistence | For each product, the artifact views and semantic state that the product covers (D11) |
| Capabilities | The accepted state that their requests are declared over, read through protected access (G4) |
| Observability and metadata | Accepted facts and history, without live objects (G4.5). Live structural resolution reads a revision-pinned inspection view under a scoped lease and keeps no live object |

No method returns a participant by its address. Addresses only locate
declarations during fulfillment and transitions. Downstream code sees the
accepted strategy's consequences, never the authoring object or a general
query over run state, so it has nothing to reinterpret.

#### The authority's part of a cut

The authority's cut state holds the run identity, the accepted strategy's
content identity, its obligation revision and the current stage. It also
holds every participant and relationship reference with its lifecycle and
revisions, each participant's named routes, lineage, and each binding's
contents at the cut. A snapshot read captures those contents; a participant
whose state never changed may record the source it re-materializes from
instead. Restoration re-materializes bindings into new objects under the same
references, then prepares and publishes as usual. Missing facts leave
restoration incomplete, and it never becomes a new run
(`run-participant-state`, G4.4).

#### Checking the run-state rules

| Settled rule | Where the types enforce it |
| --- | --- |
| One authority, with no competing canonical copies (`run-participant-state`) | Only the authority writes its section; everyone else holds read-only projections |
| Address and incarnation differ, and roles never establish identity (D6) | The authority mints `ParticipantRef`; addresses only locate declarations; references carry no role |
| Materialization, replacement, preparation, wrapping and compilation keep the reference (D6) | They change a `Binding` or a `Route` under the same `Participant`; only an amendment, a redeclaration or an incompatible replacement mints a reference |
| Retired references are never revived or reassigned (D6) | Minting is monotonic; retirement is a lifecycle state kept in history |
| One binding per participant/route pair, views apart from routes and no fake routes (D6) | One `Binding` per participant, one prepared callable per `Route`, `View` as a separate type, and optional routes |
| Participant, optimization and relationship lifecycles stay independent (D6) | Lifecycle on `Participant`, trainability on units (D20) and operational state on `Relationship` |
| Transitions are revision-checked and atomic (`run-participant-state`) | `Transition` carries expected revisions; the authority publishes the whole prospective section or nothing |
| Freshness is conservative by default (D6) | `Dependencies` defaults to the whole source section |
| Consumers get scoped projections instead of lookup (D6) | One projection type per consumer, and no lookup by address |
| Initial and later changes share one enforcement direction (D7) | Both are transitions judged against the same `Obligations` |
| Evidence is typed, not a fact dictionary or a Boolean (D7) | Evidence kinds declared beside their evaluators |
| Deferred state is declared-but-unbound, never `None` (D7) | The `declared` lifecycle state |
| Preparation is one attempt with separated results (D8) | `PreparationAttempt`, and the five parts of `PreparationResult` |
| Fallible work precedes an unobservable publication (D8) | Owners build sections before the swap; `Publication` only swaps |
| Destructive work withdraws guarantees first, and older candidates cannot restore them (D8) | The withdrawal transition, and results bound to their attempt's basis |
| Inseparable groups are prepared whole and evicted together (D8) | Preparation requests cover every overlapped group; retiring a member withdraws the whole group |
| Exact restoration keeps identities, and missing facts leave it incomplete (`run-participant-state`) | The authority's cut state; restoration re-materializes under the same references |

### D20. Optimization types

This decision answers task 5.2. Trainer optimization is an owner in D19's
pattern: it mints unit references, publishes prepared units in its section of
each generation and supplies its part of each cut. Optimizer, scheduler and
backend objects remain realizations (D9).

#### The types

| Type | What it is |
| --- | --- |
| `UnitRef` | One unit incarnation, minted by Trainer optimization. A split, merge, retirement or recreation mints new references (D9) |
| `UnitDefinition` | The accepted meaning at a definition revision: members as `TargetRef`s, logical groups, optimizer and scheduler policy, accumulation and completion rule, scheduler trigger, clipping scope, synchronization requirements, zeroing rule, parameter averaging such as EMA, and semantic dependencies |
| `OptimizationPlan` | The accepted strategy's units as typed definitions, plus what spans units: joint windows with their advancement order, joint clipping scopes and the standard rule that ownership does not overlap. It derives from the strategy and is never a second definition |
| `UnitCandidate` | Provisional resolution inside one preparation attempt: resolved parameters with provenance, tied aliases consolidated, execution groups mapped to logical groups, trainability, clipping and synchronization membership, the optimizer and scheduler (built or as a recipe), and correspondence evidence after backend transformations (G3.4) |
| `PreparedUnit` | The runtime that a preparation result carries for one unit and that a generation publishes: its reference and definition revision, resolved members, optimizer, scheduler, execution groups, membership in a backend synchronization group, optimizer-local lifecycle transitions and dependencies |
| `Advancement` | One window's record: the contributions it included; each mechanic reached, namely synchronization, unscaling, clipping, the optimizer call (returned, skipped, uncertain or not attempted), the scheduler and zeroing; which owner performed each; and the unit's progress afterwards |
| Unit cut state | The unit's part of a cut: reference, definition revision, member correspondence, optimizer and scheduler state, averaged weights, progress and window position. Pending gradients appear only when the cut is not taken at a quiescent boundary (G3.1). Backend scaling and synchronization state belongs to the backend coordination owner's cut state, together with its correspondence to the unit (G4.4) |

The strategy's structure decides where contributions come from and when
windows run: its contributions, windows and triggers (D17). D9's "checkpoint
ownership" is the unit's cut state together with the other owners' parts.

```text
  accepted strategy  units, windows, contributions and triggers
     |  Trainer optimization derives
  OptimizationPlan   unit definitions at their definition revisions
     |  preparation attempt (D19)
  UnitCandidate      members resolved, optimizer built, evidence collected
     |  preparation result, then publication
  PreparedUnit       current in optimization's section of the generation
     |  each window end, under the standard profile
  Advancement        mechanics reached, their outcomes and progress
     |  at a cut
  unit cut state
```

Trainer optimization derives the plan from the accepted strategy; fulfillment
does not produce it, and the IR system never imports training (D21). Overlap
is rejected at two points. Verification, under the training rules registered
with training's words, rejects overlap that the declared members already
show. Candidate resolution rejects overlap that appears only once members are
resolved against the loaded structure.

#### Live state and progress

A unit's optimizer state, progress and open gradient window are live state,
and only Trainer optimization changes them. An ordinary advancement changes
them and its members' numeric versions. It publishes nothing and triggers no
re-preparation (D9, G3.5).

Each unit counts its own progress. Attempted optimizer calls and
backend-confirmed non-skipped advancements are counted apart, because a
scheduler policy may follow either (G3.1). Triggers and reports name the unit
whose progress they follow (D17), and today's `global_step` becomes that
progress. A one-unit run reads as before, and a multi-unit run gets no
invented shared clock.

#### Train and evaluation modes

Module runtime mode is not unit state. Validation runs as its own activity
alongside training (D17), so a run-wide switch into evaluation mode has no
right moment. Each call's prepared execution context sets its module's mode
on entry, as it sets grad mode, and changes it only when the previous call
left a different mode (D15). The runtime serializes calls on one participant,
so no activity sees another activity's mode. Optimizer-local transitions,
such as a schedule-free optimizer's switch to its evaluation point, remain
with the unit. They are owner updates at accepted lifecycle points, under
protection. Parameter averaging such as EMA works the same way: the averaged
weights are unit state, sampling and validation read them as the unit's
evaluation point, and a product may save them (G4.3). The evaluation point
is a route on the participant whose dependencies are that unit state (D16).
Trainer optimization realizes it with separate averaged weights where they
exist. Where an optimizer can only swap in place, the swap is made when the
lease is taken and undone when it ends, so it is the route's realization and
never a competing update.

#### What covers today's flags and mode queries

Each current flag, use of the primary trainable, synchronization handle and
repeated mode query has exactly one replacement:

| Today | Replacement |
| --- | --- |
| `Trainer._train_denoiser`, `_train_text_encoder`, `FineTuneMode._te_train_flags`, `should_train_text_encoder()` | Unit membership in the accepted plan. Config only filters the strategy's declared parameters (D18) |
| `train_text_encoder=` and `train_denoiser=` passed through conditioning, validation and batch processing | Grad mode from each call's prepared context (D15). Live versus cached encoding is strategy structure, and verification rejects a detached path through a trained encoder (G3.6) |
| `Trainer._primary_trainable`, through `trainable_model`, when saving | Product persistence over a persistence projection (D11) |
| `trainable_model` passed into `process_batch` and `calculate_val_loss` | Leases for the participants each call declares (D16) |
| `all_reduce_trainable(accelerator, trainable_model)` | Synchronization in the window, through the unit's backend synchronization group |
| `Trainer._grad_sync_handle` and `accelerator.accumulate(...)` | The prepared unit's membership in its backend synchronization group; the window defers synchronization until its last member (D17). The handle itself is backend coordination state |
| `Trainer._accumulation_counter` | The unit's window position, in its live state and its cut state |
| `TrainingMode.get_trainable_params()`, for clipping | The prepared unit's resolved members within its clipping scope |
| `TrainingMode.set_train()`, `set_eval()` | Module mode from each call's prepared context; optimizer-local transitions as unit owner updates |
| `TrainingMode.on_epoch_start()`, `on_step_start()` | Repeats, windows and triggers in the accepted strategy. A method's own epoch or step behavior becomes a selected implementation call that the strategy makes at that point |
| `TrainingMode.on_step_end()`, such as weight-norm regularization | Selected post-advancement behavior under its declared effect authority, reporting observations (G3.1) |
| `TrainingMode.prepare_trainables()`: materialization and attachment | Materialization of declared participants including adapter state, and relationship transitions for attachment (D19) |
| `TrainingMode.prepare_trainables()`: merged base weights | Merge or fold transitions (D19) |
| `TrainingMode.configure_trainable_precision()`, `setup_gradient_training()` | Preparation under component constraints (D8). Freezing and trainability come from unit membership |
| `TrainingMode.build_optimizer_params()` | `UnitCandidate` construction from the plan |
| `TrainingMode.prepare_with_accelerator()` | The preparation attempt and its publication |
| `TrainingMode.register_state_hooks()` | Each owner's cut state (D19) |
| `TrainingMode.save_checkpoint()`, `checkpoint_artifact_role`, `resolve_checkpoint_artifact_format()` | Product persistence and its serializer (D11) |
| `TrainingMode.get_diagnostics_components()` | Observation facts (G4.5) |
| `OptimizationPlan`, `LogicalParameterGroup` and `ParameterGroup` in `library/optimization/types.py` | Their logical and execution split moves into candidate resolution and `PreparedUnit`. The name `OptimizationPlan` passes to the semantic plan in the same milestone (D13) |
| `OptimizationTargetRef.obj` | Leaves the reference. Live objects exist only in revision-pinned projections and candidates (`optimization-target-refs`) |

#### Checking the optimization rules

| Settled rule | Where the types enforce it |
| --- | --- |
| Address, incarnation, definition revision, realization and mutable state stay separate (D9) | `UnitRef` with its address, the `UnitDefinition` revision, `PreparedUnit` and live state |
| Membership is authority-qualified, never `nn.Parameter` identity (D9) | Members are `TargetRef`s; parameters appear only in candidates and prepared units |
| Only changes of meaning revise a definition; new wrappers, parameters, optimizer objects and processes do not (D9) | Only a governed change revises `UnitDefinition`; preparation publishes new `PreparedUnit`s at the same revision |
| Split, merge, retirement and recreation make new identities (D9) | New `UnitRef`s |
| Standard ownership does not overlap, and tied aliases consolidate (`training-optimization`) | Candidate resolution applies the plan's no-overlap rule and keeps alias provenance |
| Logical and execution groups stay distinct (`training-optimization`) | Logical groups in `UnitDefinition`, execution groups in the candidate, and the mapping between them |
| Accepted meaning precedes physical construction, in either order (G3.4) | The plan is fixed before the attempt; a candidate may build optimizer-first or wrapper-first |
| Every optimizer targets its final members or has an evidenced rebind (`training-runtime-preparation`) | Correspondence evidence in the candidate, judged before publication |
| Realization and preparation form one publication (`training-optimization`) | `PreparedUnit`s publish in the same generation as routes and backend state |
| Binding changes invalidate the affected runtime (`training-optimization`) | `PreparedUnit` dependencies; the next generation withdraws it |
| Identity continuity does not imply state continuity (`training-optimization`) | A transition states the preserve, migrate, reset or unavailable treatment, and readiness requires its evidence |
| Trainer owns standard mechanics, and only profiles transfer them (G3.1, G3.2) | `Advancement` records who performed each mechanic; grants come only from accepted profiles |
| Contributions and reached mechanics are reported separately, with no atomic joint step (`training-optimization`) | One `Advancement` per unit, with returned, skipped, uncertain and not-attempted outcomes |
| Exact restoration keeps unit identity within one run (`training-optimization`) | The unit's cut state; restored optimizer state is matched to restored members before continuation |

G3.5's first trace reads directly in these types. The denoiser's compatible
replacement is a bound-state replacement that moves its binding from revision
4 to 5. The `main` unit's prepared unit depends on binding 4, so the
generation that publishes the replacement withdraws it. A preparation attempt
based on binding 5 builds a candidate for incarnation 10 at definition
revision 3, once the accepted optimizer-state treatment is satisfied, and the
next publication installs it. A result from an attempt based on binding 4 can
no longer publish, because its basis is stale.

### D21. System map and module placement

This decision answers task 5.5. It is the one place that says where each
piece of the production system lives. The other decisions define what the
pieces are and point here for their packages and modules. Every package and
module name below is a working name; renaming one later changes no boundary.
Following the repository's placement rule, a module exists only for a stage
with several instances and shared rules.

#### Placement principles

A folder says what its code is. Whether a strategy can choose something is a
property that the thing declares, not a reason to put it in a folder. No
placement is assumed from today's layout; each one below states what the
code is.

- **Components** are the parts of models: autoencoders, text encoders,
  denoisers and the like. They live in `library/models`, arranged by kind.
  The model that introduced a component does not own it, and anyone may pair
  it differently. An **assembly** records a model as released: the
  components it uses and the conventions they were trained with.
- **Model code** is a component's or an assembly's intrinsic meaning and
  canonical interface, whatever a strategy does with it: its modules, its
  weights and their canonical formats, and the conventions it was trained
  with, such as CLIP's token window or SDXL's use of penultimate hidden
  states. Using something else, such as an earlier hidden state for
  clip-skip on SD, is a strategy's choice and leaves the model unchanged.
- **Features** are anything a strategy can take and reuse, as it is or
  modified: an approach or its math. The set is open. A feature's code lives
  where its kind of code belongs: DDPM and rectified flow in
  `library/objectives`, a loss weighting in `library/losses`, an adapter
  method in `library/adapters`. Each feature declares itself, and the
  repository's features are listed in a generated listing in
  `strategies/features/`. Reusable strategy-level
  pieces that more than one strategy uses, such as the shared
  sample/validate/save rhythm, the EDM2
  learned-weight setup or adapter use, are written like strategies and live
  in `library/strategies/features/`. EMA is a feature used as a choice on
  the unit rather than as such a piece: its math is in
  `library/optimization`, and its averaged weights are unit state (D20).
- **Machinery** runs whatever approach a strategy chooses: the IR system,
  preparation, FP8 casting, gradient checkpointing, optimizer construction
  and data loading. It lives with its owner. When it offers a choice, such
  as FP8 for one participant, the choice is declared, and the code still
  lives with its owner.
- **Behavior aimed at one component** goes where its subject belongs. How
  the component works as trained goes with the component. A choice that
  only applies to some components is a feature: its code goes with its kind,
  grouped by the component kind it needs, and its declaration says what it
  needs. Running a
  component well without changing what it computes is machinery, using
  facts the component declares. Features use what a component kind offers
  through a repository-owned interface, not a named component, so a feature
  works with every component that offers it.
- **Saving a model, writing a product and restoring a run are separate.** A
  component's canonical serialization belongs to the component, and the
  owner of what a product saves declares that product. Persistence, in
  `library/artifacts`, resolves and writes a requested product and reports
  what was written (D11); when to request one is a trigger in the strategy.
  Restoration is the IR system's cut protocol with each owner's cut state
  (D19).
- **Conventions, techniques and inputs are separate.** A model's canonical
  interfaces and the conventions an assembly was released with, such as
  SDXL's size and crop id layout, are model code. Reusable techniques that
  change how a model is used, such as prompt weighting, are features in their
  library. The realized inputs, such as the captions a run actually uses and
  the image facts behind those ids, come from data. Values that vary per item
  never become part of a cached computation that does not depend on them.

#### Declarations and guidance

Components and features declare themselves in three layers:

| Layer | What it says | For the SDXL UNet |
| --- | --- | --- |
| Meaning | What the thing is and how it reads its inputs | Eps prediction; four-channel latent input at the SDXL VAE's scale; how it reads its context, pooled vector and time ids |
| Requirements | What a use must provide | A 2048-wide context, a 1280-wide pooled vector, and the size and crop time ids |
| Provenance and guidance | What it was trained, tested or recommended with | Trained on CLIP-L and OpenCLIP-G penultimate states; the maintained settings |

The launcher composes the contract from the definitions and declarations
(D18), and help, the doctor, guided construction and fulfillment all read
that one composition. A listing of the repository's available features,
grouped by kind with their meaning, requirements and guidance, is generated
from the same composition into `strategies/features/`, and the readable
contract points to it. Help shows the live composition.

Rules and guidance are different things:

| Situation | Result |
| --- | --- |
| A use would fail: an interface that cannot connect, an authority the profile does not grant, or a construct the language cannot express | An error. Fulfillment may raise it early, and its diagnostic names the failure it prevents (D18) |
| A use conflicts with a declared meaning, such as fine-tuning an eps model toward v-prediction | Allowed and logged |
| Nobody has tested the combination | Allowed and logged |
| The combination is known to work, maintained or recommended | Shown as guidance by help and the doctor |

Research skips guidance, not rules. The repository guides what it is
prepared for with maintained assemblies, strategies and presets, help that
marks a combination as maintained or untested, and doctor warnings. All of
them read the same declarations the checks use, and none of them refuses
anything.

#### The system map

```text
  strategies and their reusable pieces                  library/strategies
     |
  launch                                         train.py, library/launch
     read the run config; compose the contract and the installed
     implementations; import the strategy and hand it to fulfillment
     |
  the IR system: Kuroiro on xDSL                        library/workflow
     fulfillment: read, normalize, verify, derive obligations
     -> the accepted run, or diagnostics
     |
  the Trainer, as host                                  library/training
     seed the run authority; materialize; prepare; publish
     |
  the run, as it executes
     the IR runtime: coordinator, streams, leases,
     requests and handlers                              library/workflow
     owners that serve it:
       the run authority: participants                  library/runstate
       Trainer optimization: units                      library/training
       backend coordination, the observation owner      library/training
     implementations it calls:
       components                                       library/models
       approaches, their math and machinery             the other libraries
     facts for metadata and logging                     observability
```

| Component | Role | Package | Defined in |
| --- | --- | --- | --- |
| Strategies, their reusable pieces and the readable contract | Strategies and the pieces they share, written by authors and maintainers; the readable contract and the guide | `library/strategies` | D18, D21 |
| Run config | Written by users and read only by launch | `library/config` | D18 |
| Presets, config application, contract composition, help and the doctor | Run by the system at launch | `library/launch`, `train.py` | D18 |
| Kuroiro, its front end, fulfillment, the accepted run, diagnostics, the reference interpreter, preparation and lowering, the runtime and device adapters | Run by the system | `library/workflow` | D14, D16–D18 |
| The owner pattern, the run authority, generations and publication | Holds state | `library/runstate` | D19 |
| Training's words, the Trainer, Trainer optimization, preparation and backend coordination and the observation owner | Run by the system; holds state | `library/training` | D17, D20 |
| Components and assemblies | What models are: their modules, weights, interfaces and declarations | `library/models` | D21 |
| Domain libraries | Approaches and their math, such as objectives, losses, timesteps, conditioning techniques, adapter methods, optimizers, sampling and evaluation, with the machinery that runs them; and products with their persistence | `library/objectives`, `losses`, `timesteps`, `conditioning`, `adapters`, `optimization`, `pipelines`, `evaluation`, `data`, `performance`, `artifacts` | D15, D21 |
| Metadata and logging | Consume facts | `library/metadata`, `library/logging` | G4.5 |
| Utilities, constants and vendored code | Base | `library/utils`, `library/constants.py`, `library/vendor` | D15 |

#### Dependency rules

These rules make D15's direction exact:

- Launch may import everything below it. `library/config` belongs to launch:
  it composes the settings types that their owners define, and nothing below
  launch imports it.
- Strategies may import training's words, the run state, the IR system, the
  libraries and base. Inside strategies, strategies import the reusable
  pieces in `features/`, and a piece may import others without a cycle.
- Training may import the run state, the IR system, the libraries,
  observability and base. It never imports strategies or config; it receives
  features only as installed implementations and declarations.
- Libraries may import the run state, the IR system, observability fact
  types, base and lower libraries. From the bottom, the library order is
  artifacts, performance, models, conditioning, losses, timesteps,
  objectives, data, optimization, adapters, pipelines and evaluation.
  Libraries never import training, strategies, launch or config.
- The run state may import only the IR system, observability fact types and
  base. It never imports the libraries or Torch.
- The IR system may import only xDSL, observability fact types and base.
  Torch loads only in its device adapters.
- Observability imports only base, and base imports nothing in `library`.
- New code never imports the code it replaces, or a `_deprecated/` folder.

The earlier module map was checked mechanically: its intended imports formed
93 package edges, with no upward import and no cycle. Today's code has 81
package edges, and all 16 packages sit in one import cycle; the coverage rows
below note today's crossings. The placement principles move family behavior
out of `strategies` and capability coordination into the IR runtime, so the
repository dependency test (`tasks.md` 13.2) checks the revised map. The
rules apply to the modules in the module map from their first line, and to a
kept library module once the new system uses it. Today's code is never
edited to follow them.

#### Module map

`library/workflow`, the IR system, is rewritten from the research
`xdsl_language` package and the `execution_candidate` runtime:

| Module | Holds | Origin |
| --- | --- | --- |
| `words/` | Kuroiro's core operations and types: structure, lifetimes, data, access, calls and model use | `dialects.py`, `filter_contracts.py`, `product_contracts.py` |
| `operation_contracts.py` | Region, flow, resource, realization and decision contracts | `contracts.py` |
| `composition.py` | The contract: registration of definitions and declarations, version, queries, and the readable form written to `library/strategies/CONTRACT.md` | New (D18, D21) |
| `frontend/` | The `iro` namespace, the one entry point authors import, with its decorators; and the reader that reads a strategy into Kuroiro | New (D18) |
| `normalize.py` | Expansion of derived words, and explicit captures, requests and stops | `normalize.py` |
| `verify.py` | Verification, the `Obligations` type and readiness checkpoints | `verify.py` |
| `diagnostics.py` | The shared diagnostic type | New (D18) |
| `fulfill.py` | Fulfillment, from strategy function to accepted run | New (D18) |
| `accepted.py` | The accepted run and its authority seed | New (D17) |
| `text.py` | The written form: printer, parser and cold loading | New (D16) |
| `reference/` | The reference semantics, for conformance and the doctor's dry run | `reference.py`, `product_reference.py`, `standins.py` |
| `prepare/` | Implementation matching, effect checks, fusion and lowering | `prepare.py`, `local_rewrite.py`, `target.py`, `target_bindings.py` |
| `runtime/` | The coordinator, invocations, streams and admission, claims, leases and protection, requests and handlers, the cut protocol and bounded bookkeeping | The research `*_runtime.py` modules; `access.py`, `progress.py` and `workers.py` from `execution_candidate` |
| `seams.py` | The owner, implementation and device seams | New (D16) |
| `devices/` | Device adapters for the host, PyTorch CUDA and the project's backend | New (D15) |

The research fixtures, stand-ins and qualification cases become the IR
system's conformance suite under `tests/`. Research names, fixture vocabulary
and the reference target's limits do not carry over as API.

`library/runstate`, the run state, is new; D15 left this package open:

| Module | Holds | Origin |
| --- | --- | --- |
| `owners.py` | The owner pattern: references, `Version`, `Dependencies`, evidence, attempts and cut state | New (D19) |
| `generation.py` | `Generation`, section registration, `Publication` and the executable run that reads through them | `execution_candidate` `state.py`, split |
| `preparation.py` | `PreparationAttempt`, `PreparationResult`, and the route and view preparation that a workflow without training also needs | `execution_candidate` `prepare.py`, split |
| `targets.py` | `TargetRef` | New; checked against `library/optimization/targets.py` |
| `projections.py` | One projection type per consumer | New (D19) |
| `authority/` | `Participant`, `Binding`, `Route`, `View`, `Relationship`, the authority section, transitions, `MaterializationAttempt` and the authority's cut state | `execution_candidate` `state.py`, split |

The authority holds the `Obligations` that the IR system's verifier defines.

`library/training`:

| Module | Holds | Origin |
| --- | --- | --- |
| `words/` | Training's words (unit, window, contribute, advance, zero, trigger and grant) and ownership profiles | New (D17, D18) |
| `trainer.py` | The Trainer: its host sequence and the standard mechanics | New; checked against `runners/trainer.py` |
| `optimization/` | Trainer optimization: D20's types, candidate resolution and grouping, and the owner protocol for training's words | New; checked against `library/optimization/grouping.py`, `phases/optimizer.py`, `phases/training_loop.py` and `modes/` |
| `preparation.py` | Training's preparation coordination, which adds unit candidates and backend groups to D19's attempt | New; checked against `phases/model_prep.py`, `trainer_utils.py` and each mode's `prepare_with_accelerator` |
| `backend.py` | Backend coordination: wrappers, inseparable groups, distributed state and their cut state | New; checked against `trainer_utils.py` and the modes' synchronization handles |
| `observation.py` | The observation owner: metrics, validation history and traces, as facts | New; checked against the Trainer's metric fields and `training/metadata.py` |
| `interrupts.py` | Interruption handling for the host | New; checked against `interrupts.py` |

`phases/`, `modes/`, `diffusion.py`, `noise_utils.py` and `execution.py`
retire from the package, as the coverage rows say.

Capabilities have no coordinator package. A capability is a kind of activity
the contract recognizes, such as sampling, validation, saving a product,
resume or caching, with a defined request, result, readiness and failure; it
is not training-specific. The IR runtime does the generic coordination
through triggers, handlers and requests, `persist` and `cut`. Where the specs
say Trainer or delegated pipeline orchestration, or a capability's
coordination, they mean this coordination by the IR runtime. A strategy
wires capabilities in, usually through the shared rhythm in
`strategies/features/`, and exposes their settings as its parameters. Each
capability's definition and implementations live with its domain: sampling
with `library/pipelines`, caching with `library/data`, persistence with
`library/artifacts`, validation with `library/evaluation`, and resume with
the cut protocol and each owner's cut state. Triggers in the strategy decide
when capability work is requested. Resume is started at launch from a saved
cut.

Where an input comes from is the strategy's choice. For each source, such as
one text encoder's outputs, the strategy states whether the value is
computed live, read from a cache or produced by an approximate
implementation, and what happens on a miss. Streams and admission check that
a value corresponds to the selected input and computation; they do not
choose. Results keep the provenance of each source, so one frozen encoder can
be served from a cache while another, being trained, runs live. A cached
value's meaning, schema and dependencies belong to whatever computed it: the
autoencoder components for latents, `library/conditioning` for text. Storage
and production traversal belong to `library/data`. "Exact" and "approximate"
always name the computation being reproduced.

`library/strategies` is the strategy system:

| Module | Holds |
| --- | --- |
| `CONTRACT.md` | The readable contract: its requirements and what the system can do, generated from the composition, with a test that fails when it is out of date. It points to where available features are listed |
| `guide/` | How to write a strategy, with worked examples. Its reference parts are generated from the contract, and the doctor checks its examples |
| `features/` | Reusable strategy-level pieces that more than one strategy uses, such as the shared sample/validate/save rhythm, the EDM2 learned-weight setup and adapter use; and the generated listing of the repository's available features, wherever their code lives |
| Maintained strategies | One for each thing trained, such as SDXL, SD and SD3, with adapters, EMA and similar as choices inside them. They are what the repository has made ready and proven |

Authors import the `iro` namespace and the pieces in `features/`; there is no
separate authoring module. Downstream users only touch the run config.
Launch also runs strategies the repository did not make, written from what
the contract offers; how it finds them is settled when launch is built
(`tasks.md` 17.1). Extending the system with new features, implementations
or operations from outside the repository is a different topic and is still
open. A strategy's own code, including an implementation only it uses, is
part of that strategy and is not an extension.

`library/launch` is new:

| Module | Holds | Origin |
| --- | --- | --- |
| `compose.py` | The one composition of the contract and the installed implementations | New (D18) |
| `presets.py` | Transitional presets, such as today's `mode`, resolved into a maintained strategy before fulfillment | New; checked against `training/modes/factory.py` |
| `run.py` | Config application, fulfillment and starting the Trainer | New; checked against `train.py` |
| `help.py`, `doctor.py` | The help and doctor commands | New (D18) |

The libraries below the strategy system:

| Package | Holds once the rework is done |
| --- | --- |
| `library/models` | Components arranged by kind, for example `autoencoders/`, `text_encoders/` and `denoisers/`, each with its modules, materializer and materialization evidence, canonical serialization and conversion, repository-owned interface and declaration; and assemblies, the models as released, such as SD, SDXL and SD3, whose materializers load a released checkpoint into its components |
| `library/conditioning` | Conditioning techniques, such as long prompts, prompt weighting, clip-skip and per-encoder dropout, and the meaning of cached conditioning values: their schema and dependencies. It uses the components' interfaces and the assemblies' conventions, and holds no storage or traversal |
| `library/objectives`, `losses`, `timesteps` | Formulations such as DDPM and rectified flow, losses and their weightings such as EDM2's, and timestep sampling |
| `library/data` | Sources, manifests, bucketing, realized captions and their transforms, augmentation, image facts, cache storage and the traversal that produces caches, with the caching capability |
| `library/optimization` | Optimizers, schedulers and parameter averaging such as EMA's, and their construction and wrappers, without target references, a plan or a grouping policy |
| `library/adapters` | Adapter methods, shared targeting, attachment and merge mechanics, their declarations and evaluators, and their file formats |
| `library/pipelines` | Sampling: the generation procedures |
| `library/evaluation` | Validation and other measurements: what a measurement means, its contributions, reduction and coverage, and its partial and empty results, starting with validation loss |
| `library/artifacts` | Products and their persistence: the product declaration types that owners use, plan resolution, capture, export, results, retention and publication, for declared products in every kind of run. Runtime snapshots and cache storage are not here |
| `library/performance` | Backend infrastructure for preparation and device adapters, including the FP8 and gradient-checkpointing rules that read component declarations |

The packages keep their roles as organizational concerns and improve where an
improvement is identified, as `models` does here. Assemblies expose the
intermediate stages that composition needs, such as each text encoder's
states before they are combined, without turning their conventions into
flags for a central routine. Training, cache production and sampling reuse
equivalent conditioning implementations: equivalent prompt-weighting copies
become one, and intentionally different methods stay distinct. A technique
applied to cached values works on its own copy and never changes the shared
cached value. `library/data` is expected to be reworked again; the new
system depends only on what its row above lists.

Observability (`library/metadata`, `library/logging`) and base
(`library/utils`, `library/constants.py`, `library/vendor`) keep their
packages.

#### Public and internal surfaces

| Package | Public to the layers above | Internal |
| --- | --- | --- |
| `library/workflow` | The `iro` words and decorators, composition, fulfillment, the accepted run, the runtime entry, the seams and diagnostics | Passes, verifier and lowering internals |
| `library/runstate` | The owner pattern, the authority's proposal and projection interfaces, and publication | Record storage and section internals; nothing returns a record by address |
| `library/training` | Training's words, for strategies; the Trainer entry, for launch | Optimization, preparation and backend |
| `library/strategies` | Maintained strategies, their reusable pieces, the readable contract and the guide, for launch and for authors | The pieces' internals |
| The libraries | Their declared implementations, component interfaces, evidence types and declarations | Everything else |
| `library/launch` | The launch, help and doctor entry points | Composition details |

#### Coverage of today's code

This table is the checklist for not forgetting anything. Each row names a
piece of today's code and what in the new system covers its
responsibilities. The new system is built new. Today's code is read, and may
be run for comparison, only to check these rows. It never decides how the
new code is shaped, and new code never imports it.

Today's code may stay where it is while the new system is written, as long as
it is identifiable and stays out of the way:

- The module map above is the boundary. Code it does not list is not part of
  the new system, and nothing in the new system depends on it.
- Checks for the new work, such as the dependency rules, lint, type checks
  and tests, apply to the new system. When the new work breaks one of today's
  tests, that test moves to `tests/_deprecated/` instead of being fixed.
- A new module may take an old file's path. If the old file is still wanted
  for reading, it is first copied into its package's local `_deprecated/`
  folder; comparisons can run it from the pinned baseline revision.

Once an area's rows are checked, its replaced code moves to its package's
`_deprecated/` folder. Old code moves earlier when new work would otherwise
break it, and its rows are then checked against the `_deprecated/` copy or the
pinned baseline (Migration Plan). Final validation deletes the folders this
change creates. Lint, type checks and test collection already exclude them.
The data pipeline rework shows why nothing may import them:
`library/config/config_util.py` and three tools still import
`library/data/_deprecated`, so that retirement is unfinished.

D20 covers the Trainer's fields and the `TrainingMode` methods individually.
Rows note today's import crossings; none of them exists in the new code.

| Today | Covered by |
| --- | --- |
| `train.py` | A thin entry over `library/launch` |
| `cfg.mode` and `training/modes/factory.py` | A transitional preset in `launch/presets.py`, resolved to a maintained strategy before fulfillment. No mode reaches the accepted run |
| `TrainingStrategy` and `strategies/base/` | Strategy functions and their reusable pieces cover the facets, contract entries cover `contracts.py`, and invocation inputs carry the facts in `context.py` |
| `strategies/factory.py` | Launch selection and fulfillment |
| Behavior modules in `strategies/<family>/` | Split by what each piece is: conventions a model was trained with go to its components and assembly in `models`; choices such as prompt handling and encoding options go to `library/conditioning`; validation's measurement goes to `library/evaluation`; what a cached value means goes with whatever computed it, its storage to `library/data`, and per-batch cached-or-live resolution becomes the strategy's stated choice of source; sampling procedures go to `library/pipelines`; each family's composition becomes its maintained strategy |
| `strategies/shared/clip/` | CLIP's token window and special-token handling go to the CLIP text encoder component. Long-prompt chunking and prompt weighting, including the copies in `data/prompt_utils.py` and the LPW pipelines, go to `library/conditioning`; weighting then works for any encoder with per-token states. The FP8 and gradient-checkpointing fixes become machinery rules that read each encoder's declared embedding modules, which covers T5 as well |
| `library/models/<family>/` (`sd/`, `sdxl/`, `sd3/`) | Components by kind, such as the UNets, MMDiT, the VAEs and the tokenizer setup, and assemblies for SD, SDXL and SD3 with their loaders, conversions and declarations |
| `TrainingMode`, `FineTuneMode` and `AdapterMode` | Unit declarations cover subjects, and Trainer optimization covers mechanics. A reusable piece in `strategies/features/` covers adapter intent, an authority transition covers resolution, and `library/adapters` covers realization (D12, D20) |
| `training/runners/trainer.py` | `training/trainer.py`, holding only the host sequence and the standard mechanics |
| `training/phases/` | The strategy's repeats and windows cover the loop, and Trainer optimization covers its mechanics. Capabilities cover caching, validation and sampling, and preparation covers model preparation |
| `training/checkpointing.py` | Two capabilities. Persistence in `library/artifacts` and the owners' serializers cover products, and the strategy's triggers cover the step and epoch schedules; each owner's cut state and the IR system's cut protocol cover resume state (D11) |
| `training/diffusion.py` and `noise_utils.py` | The autoencoder components' encoding covers latent preparation, as the strategy chooses, and `library/objectives` covers noise regularization (today's `objectives → training`) |
| `training/execution.py` | The IR runtime. Its experiment tests remain research evidence |
| `training/metadata.py` | The observation owner's facts |
| `training/interrupts.py` | `training/interrupts.py`, written new for the host |
| `training/sample_generation.py` | The shared rhythm in `strategies/features/` for triggers, prompts and destinations, and `library/pipelines` for the generation procedure |
| `training/trainer_utils.py` | Training's preparation and backend coordination |
| `LoadedModelComponent` | Materialization evidence from loaders; the canonical bound state is the authority's `Binding` |
| `OptimizationTargetRef` | `TargetRef` in `runstate/targets.py`, which keeps the semantic part and holds no live object (today's `adapters ↔ optimization` cycle) |
| Today's adapter target fields (`component`, `component_key`, `path`, `module`) | Not kept: adapter realization reads only `TargetRef` and its revision-pinned projection (`adapter-module-targeting`) |
| `optimization/grouping.py` | Trainer optimization's candidate resolution and grouping |
| `OptimizationPlan` and its groups | `UnitCandidate` and `PreparedUnit` cover the logical and execution split, and the name passes to the semantic plan (D20) |
| `ModelConditioning`, in `strategies/base/contracts.py` and used by `data` | The denoiser components' conditioning interface types in `models` (today's `data → strategies`) |
| `pool_workaround`, in `strategies/sdxl/encoding.py` and used by `pipelines` | OpenCLIP-G's pooled output as trained, in its text encoder component; `library/pipelines` reads it through the component interface (today's `pipelines → strategies`) |
| TLoRA's call to `current_strategy_context()` | Not needed: the method receives what its call declares (today's `adapters → strategies`) |
| The use of `training.checkpointing` in `adapters/shared/state_io.py` | Adapter state's cut state, through the owner pattern; serializers stay in adapters (today's `adapters → training`) |
| EDM2's learned loss weights, whose optimizer `losses/edm2/factory.py` builds | A learned participant with its own unit (D9), set up by a reusable piece in `strategies/features/`, while `library/losses` keeps the weighting math (today's `losses → optimization`) |
| Imports of `library.config` below launch | Each owner defines its settings type, and config composes them (today's 12 crossings into config, including `utils → config`) |
| `library/config` dataclasses, schemas and validation | Each owner's settings type, composed by the launch-only config; checks become each settings type's own and fulfillment's diagnostics |
| The import of metadata keys in `constants.py` | Not needed: base imports nothing above it |
| `metadata/builders/run.py` and `logging/summaries.py` reading adapters, data, objectives, models and optimization | Facts and projections only (G4.5; today's six crossings) |
| The import of `TrainingStrategy` in `training/runners/trainer.py` | Not needed: the new Trainer receives an accepted run, not a strategy (today's `training → strategies`) |
| Research `xdsl_language` | Rewritten as `library/workflow` |
| Research `execution_candidate` | Its owner machinery is rewritten as the IR runtime and the owners. Its asyncio coordination and worker model are the starting implementation, qualified on GPU in the first milestones. `RunState` splits into the run state and training's owners, `Lowering` into the IR system's preparation and runtime, and its older `graph.Contract` language is superseded |
| `xdsl_candidate`, `pliron_candidate` and `executable_spike` | Research evidence; no production code imports them |

#### Adapter placement

Task 5.5 names these placements explicitly. Adapters are features, one kind
among others, so none of these placements puts adapter-specific code in the
IR system, the run state or the Trainer. How a particular run uses them is
the strategy's business:

| Item | Home |
| --- | --- |
| Adapter methods, attachment and merge mechanics, evaluators and file formats | `library/adapters` |
| Bringing adapters into a run: the adapter participant, its attachment and its unit | A reusable piece in `strategies/features/` |
| Per-strategy method support | The strategy's declared method parameter; help shows which methods are maintained. `adapter-system` still requires a declared supported set and rejects other methods, which the rejection-rule audit (task 5.7) reviews under D18's refusal rule |
| Semantic target intent | Target selectors written in the strategy or a reusable piece |
| Governed target resolution | An authority relationship transition, judged by the adapters package's evaluator against the loaded structure |
| Policy-neutral target references | `TargetRef` in `runstate/targets.py`, outside the `optimization` namespace |
| Method realization | `library/adapters` |
| Parameter grouping | Trainer optimization |
| `FineTuneMode` | Dissolved into unit declarations plus standard optimization and preparation |
| Artifact product choice | Declared products in `library/artifacts`, chosen independently of how training treats the participant |
| Direct plus adapter training | Allowed wherever the contract allows it, with units covering base substructure and adapter participants |
| A mode axis | None: launch consumes presets before fulfillment |

#### Choices flagged for review

- Extending the system: adding features, implementations or operations from
  outside the repository, and a way to contribute them, such as a ledger or
  a GitHub-based process. D16's two extension kinds cover the language side;
  installing and sharing outside code is open. This is separate from running
  a strategy the repository did not make, which only uses what the contract
  offers. Task 5.6 assigns it to a milestone.
- What `library/evaluation` holds beyond validation loss, such as further
  measurements, evaluation sets and reward models, and how results feed later
  decisions.
- `library/data`'s internals, which another rework is expected to change.
- Sampling's engine may become external, such as ComfyUI, as another
  implementation of the sampling capability; it would use a product rather
  than live state.
- Working names: the component kinds, the assemblies' folder,
  `library/workflow`, `library/runstate`, `library/launch`,
  `library/conditioning`, `library/artifacts`, `library/evaluation`, and the
  spellings `iro.derived` for the derived-word operation and `at_request`
  for the version selection of requested work.

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
| Deferred materialization | The generic loading contract mentions a deferred module and exposes a lazy-load hook; training preparation checks for a missing denoiser (`library/strategies/base/contracts.py:34–74`, `library/training/phases/model_prep.py:30–51`). The inspected SD3 loader itself returns an MMDiT and may return absent encoders; it is **not** evidence of a working deferred-MMDiT path (`library/models/sd3/loader.py:435–481`). The deferred SD3 denoiser is explicitly a design scenario in `docs_design/archive/models-strategy-trainer/strategy_contract_exchange_design.md`. | An accepted deferred participant remains declared and unbound until an allowed readiness checkpoint; later loading materializes the same identity. A phase that requires it cannot proceed on `None`, while an explicitly allowed earlier phase may. This is a target obligation to implement and test, not behavior credited to today's SD3 path. |

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
ship in this change. Unsupported selections must be rejected by
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
launcher; D21 covers it with the IR runtime, and its experiment tests remain
research evidence. In
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

#### Concern audit disposition — 2026-10-05

The external concern passes 0–7 examined pinned commit
`6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48`; their reports remain evidence,
not normative authority. Pass 7's seven combined source traces established no
new blocker beyond inherited findings. They support G5 production-shape work
conditionally, not production implementation or an unconditional verdict on
the corrected working tree. The bounded local close-out checks the corrections
below against their neighboring requirements; it does not repeat the whole
audit or promote inspected experiments into integrated execution evidence.

| Finding | Corrected meaning and home |
| --- | --- |
| P1-REV-01 | D9, `loaded-model-components`, and `optimization-target-refs` explicitly admit standalone participant-owned substructure without invented component ancestry; genuine component selectors and host provenance remain intact. Concrete reference/selector encoding remains G5. |
| Post-close-out selector applicability | A follow-up against commit `b19717fa4eab3d6b31ad0a3ae4fe1c24bfa6f42c` identified a surviving independent obligation in unchanged `component-qualified-selectors`. Its matching delta now scopes inspection, matching, shared selectors, and internal keys to genuine component provenance; it closes the same P1 applicability concern rather than changing the architecture or selecting standalone syntax. |
| P2-REV-01 | G4.4's introduction now agrees with `run-participant-state` and capability restoration: missing required same-run facts leave continuation incomplete; new identities require a separate new-run request. |
| P4-REV-01 | `training-capability-coordination` rejects incompatible requests at their earliest authoritative evidence point before readiness/use; a runtime trigger may derive a bounded request but cannot validate it or authorize prohibited resource creation. |
| P6-REV-01 | The metadata foundation rejects ModelSpec extension collisions with canonical facts and the projection-owned version at item filing and final projection of stored records, while retaining custom extensions. Regression coverage checks required/optional fields, bare/prefixed stored keys, and artifact scope. This repairs existing conformance, not a G5 implementation milestone. |
| P6-NOTE-01 | The companion design's migration and open-question wording now retains the first fingerprint policy's candidate-only status after fixtures pass; automatic equality or merging needs a future explicit accepted policy/spec change. |

Passes 5–6's earlier-cut question is recorded explicitly in G4.4 and
`run-participant-state`: capture success does not override later established
terminal retirement. Required history and effect knowledge survive rejected
restoration. The example leaves other eligible earlier cuts available under
their accepted policy; it does not add branching, replay, or rollback authority.

G5 still must choose concrete types/APIs, executable representation and
lowering, protection/backend mechanisms, owner-contribution and recovery
protocols, and traced conformance and performance tests. Autonomous
multi-input coordination, backend-complete handback, and coherent durable
whole-run restoration have not been demonstrated by the bounded experiments.
The companion's concrete integration, typed-query, and qualified resource-owner
gates remain open; planning reconciliation does not complete them.

The combined corrections passed both strict OpenSpec validations, scoped
metadata lint/format/type checks, and all 246 metadata tests under UTC (the
existing family-parity fixtures assume UTC). The three shared loading/model-fact
requirement and scenario blocks still match across the changes. `review-mcp`
approved the corrected working-tree diff against the pinned baseline with
minor non-blocking notes; this completed review supersedes the earlier
quota-blocked attempt. The close-out and Pass 6 fix are tracked by
`sd-scripts-syv.3` and `sd-scripts-syv.2`.

That completed review predates the follow-up selector delta and does not cover
it. The selector correction and its proposal/ledger reconciliation are tracked
separately by `sd-scripts-syv.4`; required review remains outstanding because
the user reports `review-mcp` quota exhaustion. Both changes pass strict
OpenSpec validation; scoped document checks preserve all seven existing
selector scenario identifiers and the inspection/regex examples, and whitespace
checks pass. These checks are not executable selector conformance or a
substitute for the required review. The earlier audit reports and completed
review remain historical evidence, not approval of this later edit.

G5 selector work must keep optional catalog/structural qualification distinct
from genuine family-component ancestry; a catalog association alone does not
justify a fabricated public component prefix for standalone state. Pre-existing
duplicate custom-extension spellings and cosmetic error-type differences do
not reopen this canonical-claim repair or introduce a new implementation task.
G5 tasks remain unchecked and the main rework stays open. This bounded audit
close-out permits the next production-shape design task; it does not satisfy
task 5.7 or authorize production apply, spec sync/archive, or a new runtime API.

### G5. Choose production shapes and traced implementation milestones

D14 records the working xDSL substrate choice. The bounded language milestones
6–11 supply additional evidence for tasks 5.3/5.4 before those production choices
are completed. Their research contract types and runtime adapters do not close
G5 or become production interfaces by default.

D15 records the production architecture: layers and dependency direction,
runtime owners, the execution model, the Trainer's role, and what existing code
and the research packages become. Tasks 5.1–5.6 complete its types,
representation, contract and authoring API, module placement and milestones.
D16 records the workflow layer and the Kuroiro language. D17 records accepted
and executable training runs. Together they cover task 5.3's representation.
D18 records the training contract, fulfillment and authoring for task 5.4.
D19 and D20 record the run-state and optimization types for tasks 5.1 and
5.2, and D21 the system map and module placement for task 5.5. None of these
decisions satisfies this gate by itself.

Tasks 5.1/5.2 can develop independently from the completed exchanges; the
6–11 dependency applies to tasks 5.3/5.4, not every production-design task.
Any independently developed shapes still require reconciliation with those
choices before production placement and the milestones are finalized.

Only after G1–G4, choose concrete authority/binding/preparation types, accepted
arrangement/executable-run representation, authoring/strategy-check boundary,
modules,
and names. Every production milestone must identify the code it writes, the
replaced code whose coverage rows it checks, requirements/scenarios, tests,
performance checks, and review boundary.

The gate is satisfied only after strict OpenSpec validation, user review of the
architecture, required review tooling, and a complete trace:

```text
Trainer responsibility
  -> normative requirement
    -> accepted-arrangement exchange
      -> representative scenario
        -> implementation milestone
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
- **Old and new machinery mix** → They never do. New code never imports,
  bridges to or keeps alive the code it replaces; today's code is only read or
  run as a sanity check (D21, Migration Plan).
- **Metadata identity and runtime identity collapse** → Runtime authority owns
  participant incarnations; metadata stores intentional durable projections;
  artifacts and catalog/model revisions retain their own identities.
- **Current main specs contradict the target** → Carry explicit delta specs for
  every known conflict rather than relying on prose precedence.
- **Backend behavior forces accidental semantic ownership** → Backend order and
  composites constrain realization but never define participant or unit
  identity.
- **No trainer works in the tree until `tasks.md` section 18** → The pinned
  baseline revision stays runnable outside the working tree for real
  training, and sections 16 and 18 bring a GPU step and a full run as early
  as the dependency order allows.
- **The source-reading front end has no prototype** → The choice affects
  only the front end (D18). The first maintained strategy confirms it in
  section 17, and verification and the runtime do not depend on it.
- **xDSL changes underneath the IR system** → The baseline is pinned (D14),
  a version change reruns the structural, semantic and correspondence checks,
  and the written form is the project's own.
- **The runtime model is unproven on GPU and without the GIL** → The CUDA
  adapter and one real SDXL training action land in the earliest milestones
  (D15). Running without the GIL is a performance target, never a
  correctness mechanism, and every run reports its actual GIL state.

## Migration Plan

This section defines how the work is done. The numbered milestones are
`tasks.md` sections 12–28, which replace the outline below. Despite the
section's name, nothing is migrated.

The new system is built new. Today's code is a checklist for not forgetting
anything, through D21's coverage table, and it never shapes the new design.
It is not kept working, bridged to or maintained, and new code never imports
it. It may stay in the tree while the new system is written, as long as it
stays identifiable and out of the way (D21). Reading it, or running it to
compare results, is a sanity check. Kept libraries, such as model components,
data sources and optimizers, are used where they fit and follow D21's rules
once the new system uses them.

Before any production code is written, a baseline is pinned (`tasks.md`
section 12). The last revision where today's trainer works is tagged, and
short runs, a few training steps' inputs and outputs, and the G3.7 cost-probe
numbers are recorded on the target GPU. Every later sanity check compares
against those recordings, or reruns the pinned revision outside the working
tree.

New code is written in place, at its target path. Replaced code moves to its
package's local `_deprecated/` folder once its coverage rows are checked, or
earlier when new work would break it (D21). The folders this change creates
are deleted after final validation. Intermediate revisions need not train,
and no run ever mixes old and new machinery.

1. Complete and review G1–G4 inside this change. No production code changes.
2. Choose production types and an accepted-arrangement boundary from the
   completed consumers rather than copying the isolated spike.
3. Write the training-side run-state, realization, preparation,
   optimization, capability, and Trainer-consumption surfaces against explicit
   test-only accepted inputs. This is a dependency order for the code, and it
   must not introduce a production bypass or a bridge to today's strategies.
4. Write contract-guided authoring and strategy fulfillment that produce
   those same accepted inputs for one maintained SDXL strategy.
5. Make one SDXL training run work end to end on the new system. Check its
   coverage rows, then move the replaced code to `_deprecated/`.
6. Add adapter behavior through the same accepted participant/relationship,
   optimization, persistence, and restoration boundaries; do not create a
   second adapter orchestration authority.
7. Write the SD and SD3 strategies with conformance scenarios proving that the
   core did not inherit SDXL cardinality or objective assumptions.
8. Complete the capability and research-extension work, check the remaining
   coverage rows, and delete the `_deprecated/` folders after final
   validation. Archive the change only after all main specs and acceptance
   tests agree.

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

D15 now fixes package-level placement. Names and modules within those
packages, and the remaining items above, are answered by tasks 5.1–5.5. D18
answers the custom-operation authoring item.
