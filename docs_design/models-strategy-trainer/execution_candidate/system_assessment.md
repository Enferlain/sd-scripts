# Whole-candidate assessment — 2026-10-27

This refreshes the October 24 assessment with the implemented `.19`–`.26`
findings and the subsequent `.27` input-association fix. Bead
`sd-scripts-syv.25` tracks the assessment. The earlier
source-access/publication cycle is resolved by `.19`; its reproducer is now a
rejection regression, not a remaining defect. Historical measurements and
their source/environment limits remain in the [research record](README.md).

The authority is the [governing design](../../../openspec/changes/rework-model-strategy-trainer/design.md),
especially D4–D10, G2.6 and G3.6, and the delta requirements for
[accepted execution](../../../openspec/changes/rework-model-strategy-trainer/specs/accepted-training-execution/spec.md),
[participant state](../../../openspec/changes/rework-model-strategy-trainer/specs/run-participant-state/spec.md),
[preparation](../../../openspec/changes/rework-model-strategy-trainer/specs/training-runtime-preparation/spec.md),
[optimization](../../../openspec/changes/rework-model-strategy-trainer/specs/training-optimization/spec.md)
and [capabilities](../../../openspec/changes/rework-model-strategy-trainer/specs/training-capability-coordination/spec.md).
This assessment recommends a foundation; it does not amend those requirements,
adopt the experimental Python API, migrate production training or complete G5.

## Judgment

Retain the graph-derived execution mechanism as the working foundation. The
hierarchy and relationships construct executable work **and its coordination**.
Changing an arrangement changes the derived calls, input connections,
lifetimes, access and completion routing; the engine is not a fixed training
loop receiving a different objective callback.

The recent investigations strengthen that judgment. Optimization has one
pending-work owner shared by ordinary and granted execution. Calls, providers
and retained sources share current-use protection. Preparation derives access
checks and can specialize unused bookkeeping. Worker placement permits useful
parallel Python work while keeping control state with the coordinator. The
precision findings expose backend obligations without requiring another
training architecture.

The assessment reproduced one cross-owner gap: unfinished optimization kept
its outcome but could lose its input association after diagnostic eviction.
`sd-scripts-syv.27` now preserves those associations through immutable
references held by the existing execution, unit, request and failure owners.
The four counterexamples pass. This strengthens the shared mechanism without
changing the accepted graph direction; the production qualifications below
remain open.

## How the system fits together

There is one accepted description. Its nesting gives activities their
lifetimes, and its relationships connect work inside and across those
lifetimes. Preparation derives the executable realization of that description.

The adaptive [diffusion-style example](examples.py#L188) has this shape:

```text
accepted run: Together
|
+-- inputs: Together
|   +-- pixel Producer ------------ image -----------+
|   +-- caption Producer -------- conditioning ------+
|       +-- provider-owned workers, when selected    |
|                                                   v
+-- learning: Repeat <-- adaptive state <-- observation
|   +-- selected bounded policy                     ^
|   +-- learn: Block <-- correlated input Join       |
|       objective ------+----------------------------+
|                       +--> addressed differentiation --> advancement
|                                  |
|                           unit-owned window/outcomes
|
+-- validation: Requested <-- learning Completion
    +-- protected evaluation Block --> validation-owned results

Across these scopes:
  current bindings/views <--> protected calls, providers and retained uses
  input ports           <--> readiness, admission, handoff and capacity
  owned state           <--> selected policy/computation and continuity
  shared wait evidence  <--> protected access and bounded publication
```

The objective, adaptive algorithm and encoders remain selected Python
behavior. The run structure exposes the relationships that other owners need.
Internal tensor operators and provider worker schedules do not become common
graph nodes. The [adversarial example](examples.py#L270) instead selects D/G
actions with different derivatives and unit destinations through the same
construction and engine.

### From accepted meaning to execution

| Stage | Current mechanism | What becomes concrete |
| --- | --- | --- |
| Checking | `Contract.accept()` in [graph.py](graph.py#L399) | Known scope/value dependencies, ownership, bounded alternatives, offered two-pass authority and run-scoped participant/unit references. This is a probe contract, not the finished authoring or obligation system. |
| Realization | `prepare()` and `RunState.candidate()` in [prepare.py](prepare.py#L337) and [state.py](state.py#L649) | CPU-target access readiness, loaded/verified participants, resolved parameters and aliases, trainability, unit runtimes and views. Target checks precede loading; all candidates remain unpublished. |
| Lowering | `Lowering._connect()`, `block()`, `sequence()` and `_scope()` in [prepare.py](prepare.py#L421) | Request queues, whole-lifetime stop events, consumer relationships, direct value wiring, selected-policy loops, activity tasks and scoped access/cleanup. |
| Publication | `Preparation.publish()` and `RunState.install()` in [prepare.py](prepare.py#L287) and [state.py](state.py#L726) | One checked installed image after source-generation, current-use and unfinished-work guards. No external callbacks run in final installation. |
| Execution | `Engine.run()` in [prepare.py](prepare.py#L300) | Entry into the derived root, session-active/completed state and failure propagation. The derived coordination and state owners are also engine machinery; the small entrypoint is not the entire mechanism. |

The prepared callables are a realization of the accepted graph, not a second
authored run. `Ready.source` is inspection output. Ordinary execution does not
interpret those source strings or traverse the whole graph to find its work.
Static ordering, selection and connection construction finish in preparation.
Live policies, admission, current projections and relevant freshness checks
remain runtime work.

The `calls` and `inline` emissions share invocation, output-check and failure
rules. The second specializes ordinary-block dispatch; it does not separately
reimplement Sequence or run coordination. The `general` and `prepared` access
variants likewise use the same acquisition protocol: the latter omits empty
retention borrowing and empty input-demand registration for ordinary blocks.
Neither comparison establishes a numerical compiler or a general lowering
pipeline with multiple backend targets.

### Owners and interactions

| Concern | Owner and shared boundary |
| --- | --- |
| Participants and current representations | The binding section of `RunState` owns the installed image. Scoped projections supply selected models/views and actual stamps; participant identity is not a role, Python object or optimizer handle. |
| Optimization | Each `UnitRuntime` owns entered work, verified handback, readiness and reached per-unit outcomes. `RunState` coordinates addressed contribution/advancement. Ordinary differentiation and the bounded grant enter and hand back through that same owner. |
| Input delivery | `Feed`/`join()` own addressed readiness, protected admission and the local all-member claim. Coordination records immutable handoff facts; region, unit, request and failure owners retain the references they need. A provider owns traversal, private workers and continuation. |
| Policy and operation state | Accepted owner objects supply initialization and selected updates. Declared users acquire state access; active producer state stays pinned through cleanup. Ordinary feedback does not require rebuilding the graph. |
| Capabilities | Prepared completion/request connections and selected policies make evaluation due. Evaluation has its own protected invocation and result owner. Learning completion is not evaluation completion. |
| Physical execution | `Workers` owns capacity and physical futures. The coordinator keeps reservations through actual CPU completion and cleanup; providers using private executors must uphold their own equivalent boundary. |
| Progress | `Progress` relates task-attributed leases, prepared consumer relationships and live demands. It rejects closed sets of observed access/publication waits; it does not select an algorithm's next work or infer private waits. |
| Failure | Reached work and its owners retain known outcomes and uncertainty; dependent state is withheld. Scope cleanup drains supported uses without turning release or parameter restoration into successful execution. |

Sources: [current use](state.py#L742), [provider access](state.py#L823),
[retention](state.py#L855), [optimization](state.py#L1001),
[input handoff](state.py#L442), [progress](progress.py#L36),
[worker completion](workers.py#L75), [failure evidence](prepare.py#L130), and
[the bounded grant](prepare.py#L187).

`RunState` and `Lowering` remain large internal holders with several distinct
responsibilities. Production placement should make those owners and target
construction stages readable without duplicating their invariants. Ordinary
calls already receive a narrow `Context`, not the binding authority or
optimizer services. Read-only maps do not freeze their tensor/object contents;
selected Python remains trusted rather than sandboxed.

## Different arrangements through the same mechanism

**Adaptive computation and requested validation.** Two producers can advance
ahead and finish out of order. A selected key identifies the required members;
`Join` checks every member against the protected current projection before
claiming any. The objective produces a gradient source and a separate
observation. Explicit ordering delivers the observation before its dependent
backward/policy decisions. Standard mechanics then address the selected unit.
Normal Repeat completion records action progress and routes a completion to
the requested capability, whose own policy may request protected evaluation.
There is no need for the authoring object during these decisions.

**Alternating and joint optimization.** D/G alternation selects distinct
computations and destinations. The G computation differentiates through D
without adding a D advancement. Joint work instead requests one shared source
for two disjoint units. Both use the same unit-owner lifecycle. If A's optimizer
and reset return before B's optimizer fails, A's known outcome and B's
uncertainty remain separate and live; readiness cannot be reused. The fixed
CPU optimizer is plain SGD, not Muon/AdamW or a distributed backend.

**Multiple views and retained derivatives.** `routed_views()` preserves one
frozen participant across distinct invocation roles and keeps its required
gradient path to the selected subjects. A Sequence derives cross-block value
links and per-work access. Live old-state retention prevents conflicting
advancement until explicit last use; an actual isolated copy can permit an
earlier update in the qualified fixture. A stamp is not the retained numerical
state. Dynamic two/three-contribution windows use the same Sequence machinery,
and state-only work needs no fabricated optimizer action.

**Changed authority.** The offered two-pass profile owns its two backwards,
temporary edits and intermediate reset. Its checked handback enters the same
pending-work owner as ordinary differentiation. Standard infrastructure still
owns final clipping, advancement and reset. Failed handback or restored
parameters alone do not authorize advancement. This is one fixed grant, not
evidence for arbitrary imperative ownership profiles.

**Compatible replacement.** A typed proposal checks expected binding revision,
fresh realization/storage, current use, retained/contribution completion and
owner continuity. Candidate construction finishes before publication; affected
unit runtimes are rebuilt while their semantic identity/progress survives.
Other plain-CPU units can be reused with member correspondence evidence. This
is a global quiescent replacement-only boundary, not general arrangement
amendment, destructive preparation or migration of a live continuation.

Evidence: [examples](examples.py), [candidate cases](test_candidate.py),
[routed views](test_routed_views.py), [repeated regions](test_repeated_regions.py),
[optimization ownership](test_optimization_ownership.py),
[current use](test_current_use.py), [producer lifecycle](test_producer_lifecycle.py),
[physical lifetimes](test_worker_lifetimes.py), and
[same-graph access comparison](test_access_lowering.py).

## What the later investigations establish

| Finding | Current consequence and limit |
| --- | --- |
| Shared runtime progress (`.19`) | The former source-read/full-feed/conflicting-consumer cycle now rejects without changing accepted work or releasing state early. Compatible reads, per-work acquisition and an alternative claimant still progress. Private queues, worker-slot waits and device/collective completion are not inferred. |
| Eager optimizer (`.20`) | The candidate can run without the incidental Dynamo/Triton import and keep the GIL disabled. `EagerSGD` is a bounded adapter to the installed Torch functional SGD API, not the production optimizer choice or a consequence of producer live sharing. |
| Useful parallel work (`.21`) | Bounded Python text processing/packing used nearly two cores and about doubled throughput in the retained synthetic GIL-off comparison. This is actual useful parallel work, not a real-model training speedup. The source-stamped timings predate the subsequent worker policy and are not renewed here. |
| Free-threaded boundaries (`.22`) | Control state stays on one coordinator. Worker-visible owner state, contexts, source access and physical drain have deterministic concurrent cases. This does not establish arbitrary cross-loop control access, transitive payload isolation or safety of arbitrary module/native internals. |
| Precision/cache ownership (`.22`–`.24`) | Mixed bf16/fp16 readers exposed a shared Torch cache collision. The worker default disables that cache. Region-local casts and private module structures show qualified reuse, but mixed-use weights, normalization, buffers and module aliases expose real limits. Those helpers are not integrated execution views or an adopted AMP replacement. |
| Prepared access (`.26`) | Some unused repeated bookkeeping can disappear while the same protections remain. Matched cheap-block savings were roughly 6–7 microseconds, 14–16%; numerical results were noisy. These numbers neither price live sharing as a whole nor qualify full-engine performance. |

The [README checkpoints](README.md#shared-runtime-progress-2026-10-25) retain
the tests, samples, source hashes and counterexamples. This assessment does not
rerun performance measurements or pool different measurement generations.

### Live access and snapshots

The design requires actual production provenance, admission and protected use;
it does not mandate concurrent producer access to trained weights. Live access,
frozen sources, protected snapshots and supported isolation are realization
options subject to those obligations. The candidate has not compared their
target-workload throughput or selected a producer default.

In `producer_continuity()`, the provider's encoder write deliberately creates
old/new provenance cases; that encoder is not an optimization subject. This is
not evidence that an ordinary encoder producer must write trained weights.
Detached pre-encoding also cannot replace a required encoder derivative path.
The common protection machinery additionally serves replacement, mode changes,
owned state and retained derivatives, so its entire cost cannot be attributed
to live-sharing producers. The correlation regression below uses no
producer access to weights at all.

## Input correlation: reproduced gap and implemented lifetime

The governing execution spec requires input identities/dependencies,
invocations and unit-local outcomes to remain correlatable after partial
failure. At the `.25` checkpoint, the unit owner preserved live origins
independently of its 32-entry completed history. Input handoffs existed only in
`Feed.handed_off` (32 entries) and admission diagnostics in `RunState.trace`
(256 entries). `WorkFailed` copied unit outcomes, not their input associations.

The case in [test_system_progress.py](test_system_progress.py) executes this
accepted arrangement through the ordinary engine:

```text
source Producer --> one bounded input port --> learning Repeat
                                |               +-- Sequence
                                |                   claim training input
                                |                   create gradient contribution
                                |                   release old-state read
                                |                   await independent consumption
                                |                   attempt advancement --> failure
                                v
                        independent Repeat
                            claim 48 other inputs
```

The independent consumer performs 48 real handoffs. Before `.27`, the injected
optimizer failure preserved the pending contribution and its uncertain
optimizer/unattempted reset outcome, but input/admission histories had evicted
the original handoff. No owner retained the missing association. Physical
access drained and capture rejected; correct withholding did not repair it.

The original four strict expected failures have been replaced with positive
checks under `calls`/`inline` and `general`/`prepared`. They still confirm
diagnostic eviction, then check the live unit, `WorkFailed` and Engine facts.
Additional cases cover cross-block inputs, actual source provenance after
encoder changes, long windows, grants, independent capability requests,
failure/cancellation and release. Run the original regression:

```bash
UV_PROJECT_ENVIRONMENT=.venv-wsl uv run --locked --no-sync pytest --noconftest docs_design/models-strategy-trainer/execution_candidate/test_system_progress.py -k unfinished_input -q
```

Coordination now records metadata-only `InputHandoff` facts at protected
admission. Immutable prefix references pass to the existing region, pending
unit, capability-request and failure owners. Preparation bounds each region's
admissions by its Join member slots; pending windows and bounded request
queues govern in-flight retention. Completed unit history stays at 32, and
reference release reclaims unused links. No larger diagnostic deque, new
author declaration, producer-owned optimization or universal step packet was
introduced. See the [implementation and bounds](README.md#input-associations-follow-execution-lifetimes-2026-10-27).

### Reproducibility remains a separate qualification

Owned RNG streams and matched synthetic worker runs have repeatability
evidence. Whole-run reproducibility across arbitrary scheduling, live-source
version choices and numerical backends is unproven. The same seed does not
determine which changing source an asynchronous policy uses. Preserving actual
provenance makes that decision inspectable; a supported reproducible policy
must additionally control semantically relevant choices. No reproducibility
waiver or universal snapshot/live default is inferred. Same-run restoration
and its accepted replay/regeneration policy are separate and unimplemented.

## Production decisions versus later qualification

The following are concrete production-shape work; the candidate's restrictions
must not silently become architecture restrictions:

- **Region composition and extension.** The hierarchy supports recursive
  `Together`, but Repeat alternatives are Block/Sequence, Requested bodies are
  Block, and the two-pass grant is block-local. Work kinds and lowering rules
  are a closed probe vocabulary. Choose how maintained/custom work, region
  combinations and granted profiles are checked and lowered without separate
  algorithm engines. General nested control is not demonstrated here.
- **Authority and revision coverage.** The candidate has participant/unit
  references, binding revisions, numerical epochs and prepared runtime keys.
  It has no complete relationship lifecycle, obligation revision system,
  unit-definition amendments, deferred materialization or destructive attempt
  protocol. Operation state initializes before materialization; evidence-based
  later initialization and general state migration remain to be constructed.
  Global replacement quiescence is a target restriction, not a universal rule.
- **Prepared target and owner boundaries.** Keep accepted meaning separate from
  parameter/view realization, supported derivative/retention demands, actual
  completion and granted handback. The current synchronous CPU `nn.Module`
  and SGD choices are not sufficient production backend interfaces. Ordinary
  authoring must eventually compose maintained integrations rather than make
  authors repeat all intermediate declarations visible in these fixtures.

The broader [research cases](../../../openspec/changes/rework-model-strategy-trainer/research/README.md)
remain the coverage frame. Their relationship to this candidate is explicit:

| Research concern | Evidence here | Still required beyond this candidate |
| --- | --- | --- |
| Different objectives/modalities, side networks, learned auxiliary state | Selected Python, arbitrary payload values, distinct uses and addressed units; no universal latent/timestep fields. | Real family/representation integrations, maintained compatibility knowledge and product semantics. |
| Async conditioning, packed/streamed inputs, generated experience | Independent producers, correlated handoffs, selected adaptive state and bounded Python workers. | Real storage/cache protocols, snapshot/lag policies, mixed-state provenance, external failure and durable continuation. No required live-sharing default. |
| Same participant in several roles, delayed derivatives, early updates | Distinct views and live/isolated retained-state fixtures. | General adapters, recomputation and pipeline/backward schedules with backend-supported state lifetime. |
| Topology, trainability or stage changes | Fresh compatible replacement with simple owner reset/preserve/reject behavior. | Relationship/participant amendments, obligation/unit-definition changes and accepted continuation migration. |
| FSDP/ZeRO, parallelism, quantization and compiled regions | Semantic identity is distinct from physical runtime, and one description has alternative CPU lowerings. | Real joint/inseparable backend preparation, rank agreement, collective ordering, stream/event completion and numerical conformance. Backend independence is a design obligation, not a qualified integration claim. |
| Capabilities, products, metadata and exact restoration | Requested evaluation and a finished, quiescent in-memory `capture()` check. | Caching/sampling/product coordination, durable fact delivery, contributor-based snapshots and actual restoration. There is no active-run snapshot/recovery implementation. |

Native Windows, real-model free-threaded safety, general failure isolation,
non-cooperating worker shutdown and full G3.7 conformance/performance remain
separate qualification. Arbitrary Python waits and undeclared mutations remain
outside the trusted boundary. The supported fail-fast TaskGroup policy is not
evidence for all accepted failure relationships.

## Next construction step

With `.27`'s bounded association gap addressed, use the connected mechanism to
choose the production structure **as a
whole**: the accepted region/relationship description, its prepared target,
the state owners and their interactions. Show ordinary maintained authoring
and the existing contrasting arrangements against that same structure before
finalizing public types or module placement. That is a concrete production
design step, not another survey or an assumption that the experimental files
can simply be moved into `library/training`.

G5's individual type lists can check the result after this system-level
construction; they cannot determine it in isolation. Nothing in this pass
selects Accelerate, a numerical compiler, snapshot/live sharing, FSDP2 or a
module-casting policy as the mandatory foundation.

## Verification

Graph discovery/traces and exact source reads were used together. Index coverage
reported no recorded issue and matching source metadata for the existing
evidence paths; this is best-effort coverage, not proof of completeness. The
regression was read and executed directly. The assessment-only `.25` pass
left runtime unchanged; `.27` subsequently updates only this isolated candidate
and its tests. Governing design/spec/task files and production training remain
unchanged.

Fresh checks in the existing uv-managed Python 3.14.5t environment:

- **639 passed**, with no expected failures, across the candidate and unchanged
  execution tests. Independent standard-optimizer oracle tests load Triton
  and enable the GIL in this full-suite process.
- **580 passed** in a separate candidate process which excludes the two
  standard-optimizer oracle files, plus both changed long-window cases from
  one excluded file in another process. The GIL stays disabled after all
  included tests, with neither Dynamo nor Triton imported and no forced-off
  override. Every new/changed `.27` case is covered.
- Scoped Ruff lint/format, strict `ty`, new local links/source-line bounds,
  introduced whitespace and bounded changed-path checks pass. Performance
  probes were not rerun; earlier source-stamped costs do not qualify the
  added input coordination.
- The installed OpenSpec validator reports **53 existing requirement-length
  warnings and zero errors** for the unchanged change. `--strict` exits
  nonzero because those requirement texts exceed 500 characters. This is not
  recorded as a strict-validation pass; the assessment does not rewrite
  governing requirements to remove that advice.
