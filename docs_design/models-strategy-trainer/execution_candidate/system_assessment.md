# Whole-candidate assessment — 2026-10-24

> **Status note, 2026-10-25:** this preserves the assessment at its stated date.
> The publication/access cycle below was subsequently addressed by
> `sd-scripts-syv.19`; see the [runtime progress findings](README.md#shared-runtime-progress-2026-10-25).
> Later GIL-disabled, useful-parallel-work and numerical-view investigations
> are recorded in the README (`.20`–`.24`). `sd-scripts-syv.25` will refresh the
> connected-system assessment; do not reopen the historical cycle or treat the
> later CPU evidence as general backend/production qualification.

This is a source-backed assessment of candidate commit `21e356e9`, plus the
six assessment cases in [test_system_progress.py](test_system_progress.py).
It is a checkpoint, not another governing design or an adoption of these Python
types. Bead `sd-scripts-syv.18` tracks this pass. Production code, normative
requirements and G5 checkboxes are unchanged.

Subsequent qualification: `sd-scripts-syv.19` connects runtime input-capacity
and protection waits and converts the cycle below to explicit rejection
regressions. See [shared runtime progress](README.md#shared-runtime-progress-2026-10-25)
for that implementation and its limits. The assessment below records the
original `21e356e9` finding; its other qualifications are not closed by this fix.

The governing references are the OpenSpec [D10 and G2.6](../../../openspec/changes/rework-model-strategy-trainer/design.md#d10-accepted-execution-is-structured-and-authority-bounded)
and [accepted execution requirements](../../../openspec/changes/rework-model-strategy-trainer/specs/accepted-training-execution/spec.md).
The earlier [direction](../strategy_system_direction.md#authoring-accepted-arrangement-and-runtime-state)
establishes the authoring/runtime distinction; the later OpenSpec establishes
the hierarchical whole-run meaning. Older README checkpoints are historical;
their superseded limitations are not the current support list.

## Judgment

The graph-derived execution approach is a credible foundation to develop
further. It does real construction work: changing accepted relationships
changes executable dependencies, lifetimes, request connections and access.
It is not a drawing above a fixed training loop, and the examples do not each
supply a separate engine.

Recent consolidation is substantive. Ordinary and granted gradient work use
one optimization lifecycle. Calls, providers and retained sources use one
current-state access protocol. Preparation uses that protocol's access facts
for its bounded readiness checks. There is no further demonstrated need to
move an example's entire workflow into another general coordinator.

One concrete cross-owner progress gap remains: a producer can hold source
access while blocked publishing into a bounded feed, while the consumer needs
incompatible access before it can claim the product already in that feed.
The assessment reproduces this below. Fixing that boundary is justified;
designing a general scheduler or expanding the IR for every private wait is
not established by this finding.

This supports continued development of the mechanism, not production readiness,
approval of the raw authoring surface, or an unchanged promotion of these files.

## The actual connected system

The accepted description has two kinds of structure. Nesting gives activities
their lifetimes; value, order, input, completion and access relationships join
work and owners within and across those lifetimes.

```text
accepted run: Together
├── inputs: Together
│   ├── pixels: Producer ─────────── image ──────────┐
│   └── captions: Producer ─────── conditioning ─────┤
│       └── private encoding workers               │
├── learning: Repeat                               │
│   ├── selected live policy                       │
│   └── learn: Block ◄──── correlated Join ─────────┘
│       objective → observation → derivative → advancement
│           └──── updates adaptive state ───→ later policy/computation
└── validation: Requested ◄──── learning completion
    └── evaluate using protected current participant state

Shared owners and services across those scopes:
  participant bindings / prepared views ← scoped access → calls and providers
  handoff ports ← publication / admission / claim → producers and consumers
  optimization units ← contribution / advancement → ordinary or granted work
  owned policy/provider/capability state ← declared access → its accepted user
```

That is the actual shape of [diffusion()](examples.py#L188), using tiny CPU
models. [adversarial()](examples.py#L270) instead gives the repeat policy two
accepted D/G alternatives and distinct unit destinations. There is no universal
input/objective/backward/optimizer sequence outside the work each arrangement
selects. The producer can progress without any derivative or optimizer call.

### Construction and execution

| Stage | Actual mechanism | What it does not establish |
| --- | --- | --- |
| Target offer and checking | `Contract.accept()` checks known vocabulary, scope/value dependencies, ownership and offered two-pass authority; creates run-scoped participant/unit identities. | It is a small probe contract, not the finished readable/queryable strategy contract or full obligation system. |
| Concrete preparation | `prepare()` runs target access readiness before loading; `RunState.candidate()` verifies loaded CPU objects, resolves parameters/aliases, checks ownership, constructs units and views. | It is not distributed joint preparation, destructive preparation, or proof of every private wait. |
| Derive execution | `Lowering._connect()` binds request queues and demand-end events. Scope lowering recursively builds task lifetimes and selected-policy loops. Block/sequence lowering resolves value edges into generated direct local calls. | The derived machinery is not a second authored run. `Ready.source` is inspection output, not another executable authority. |
| Publication | `Preparation.publish()` calls the state owner's checked `install()`, then returns the callable root with its state. | An unpublished attempt is not runnable. A generation number is not rollback or distributed commit. |
| Run | `Engine.run()` enters that root. Derived scopes coordinate live decisions, inputs, protected calls, capabilities, outcomes and cleanup. | The twenty-line entrypoint alone is not all the engine's machinery. |

Source: [checking](graph.py#L388), [preparation and publication](prepare.py#L246),
[scope connections and block emission](prepare.py#L380),
[scope execution](prepare.py#L825), [candidate/install](state.py#L524).

The accepted graph is retained as meaning and identity; the callables are its
prepared execution. Static dependency sorting, implementation selection and
scope construction do not run every iteration. A live policy lookup among its
prebound alternatives, current-source projection and required freshness checks
still run when that work needs them. Avoiding rediscovery does not mean avoiding
all runtime checks.

Both `calls` and `inline` emissions use `bind_parts()` for invocation, output
and failure rules. Only the ordinary block's dispatch shell differs; Sequence
and run-level machinery do not become a separate implementation. This is
concrete selective lowering, not numerical fusion or proof of all possible
future transformations.

### Runtime ownership and interaction

| Concern | Current owner/mechanism | Connection to other work |
| --- | --- | --- |
| Current participants, bindings, modes and last use | `RunState` access/retention methods and installed binding/view image. | Lowered calls and `ProviderContext.use_source()` borrow or acquire protected projections through the same protocol. Retentions release at actual supported last use. |
| Optimization | Each `UnitRuntime` owns one unfinished-work lifecycle; `RunState` coordinates addressed contribution and advancement operations. | Ordinary differentiation and the granted two-pass path enter/hand back through that owner. A contribution is not an optimizer handle or proof that an update occurred. |
| Input handoff | `Feed` plus readiness/admission/claim coordination. | Provider publishes addressed products; selected admission checks their actual provenance against current protected state; joined members are checked before all are claimed. |
| Selected state | Declared owner objects, provider-lifetime registration and scoped state access. | Feedback can change later computation and bounded policy decisions; producer continuation is not a training-step counter. |
| Capability scheduling | Prepared completion/request connections and selected request policy. | Learning completion can request protected evaluation work, with its own invocation and result state. The source completion is not the capability's completion. |
| Execution placement | Prepared host-worker choice and `Workers` lifetime/drain protocol; provider-private workers remain provider-owned. | Coordinator protection lasts through physical completion. Queueing for a worker can conservatively retain model access. |
| Failure and stop | Per-work checks, owner evidence, derived scope cleanup and TaskGroup failure propagation. | Returned, uncertain and unattempted unit outcomes survive; dependent state is withheld; physical users are drained before access is released. |

Source: [shared access](state.py#L617), [retentions](state.py#L740),
[provider owners and optimization](state.py#L866), [handoffs](state.py#L377),
[granted authority](prepare.py#L162), [worker completion](workers.py#L73).

`RunState` is currently a large internal holder with distinct ownership sections,
not a polished production interface. `Lowering` likewise combines connection
construction, emission, runtime wrappers and supported profile handling.
These concentrations deserve attention when choosing production modules, but
file length alone does not justify scattering their shared invariants again.
Selected ordinary calls receive narrow `Context` projections; they do not
receive `RunState`, optimizer services or the original authoring interface.
Live Python objects are still trusted, not sandboxed.

## Different arrangements through the same mechanism

| Arrangement | What changes in accepted meaning | What remains shared |
| --- | --- | --- |
| Diffusion-style adaptive work | Two input producers, correlated admission, observation feedback, learning policy, requested validation. | Same scope lowering, access, derivative/advancement ownership and failure handling. |
| Alternating adversarial work | D/G computations, destinations and selected alternation policy. | No GAN branch in the entrypoint or general loop; separate unit clocks and outcomes. |
| Joint units | One loss/gradient request addresses A and B together. | Destination-specific preparation and outcomes; A can return before B fails without inventing an atomic update. Plain SGD realizes both in this target. |
| Routed derivatives | One frozen participant supplies different views; cross-block values and explicit old-state lifetime connect two destinations. | Same identities, source-access protocol, prepared views and unit owners. A real isolated copy can satisfy old-state use; stamps alone cannot. |
| Repeated accumulation regions | A live policy selects accepted two/three-contribution Sequences; feedback changes later selection; explicit links extend a window. | Same repeat and Sequence lowering; one pending-work owner, final advancement and distinct block/region progress. |
| Granted two-pass work | An explicitly offered profile transfers two backwards, temporary edits and intermediate reset to selected execution. | Checked handback returns to shared contribution ownership; final advancement stays standard-owned. |
| Compatible replacement | A typed proposal selects a fresh compatible physical realization. | Same authority/publication path; identity survives while dependent runtime is rebuilt and selected owner continuity is checked. |

The examples are inspectable in [examples.py](examples.py). Executable evidence
also lives in [candidate tests](test_candidate.py),
[routed views](test_routed_views.py), [repeated regions](test_repeated_regions.py),
[optimization ownership](test_optimization_ownership.py),
[current use](test_current_use.py), [producer lifecycle](test_producer_lifecycle.py),
[worker lifetimes](test_worker_lifetimes.py) and
[provider workers](test_provider_workers.py).

These are materially different arrangements, but not representative production
models or backend qualifications. `TwoPass`/`GrantedAccess` implement one offered
profile with a fixed phase protocol, not arbitrary imperative authority. Repeat
accepts Block/Sequence alternatives and Requested accepts a Block body; recursive
general-purpose control regions are not demonstrated by this target grammar.
Replacement is not a general topology amendment.

## What the examples still supply manually

| Example code | Assessment |
| --- | --- |
| Objectives, model calls, adaptive rules, loss normalization and D/G selection | Selected algorithm/policy behavior; it should not move into the generic engine. Ordinary authoring should later select maintained integrations rather than reproduce all this low-level wiring. |
| `correlated()` and `Conditioning` | Domain admission knowledge and payload. Required checks belong in maintained selected behavior, not universal caption/image fields or hand-written validators for every ordinary author. |
| Provider workers, private work status and cleanup | Legitimate selected provider ownership. The framework supplies lifecycle, access and handoff seams, but the provider must actually join its workers. |
| `stop_when` and remaining-work policy | Explicit accepted relationships, consumed by preparation; not an example-specific shutdown callback to Trainer. |
| `ahead` / `staged` events and `wait-ahead` in `producer_continuity()` | Fixture staging to expose ahead-of-training and unfinished-work states deterministically. Two new cases remove the entire staging Block and still complete one admitted update and a quiescent capture through the ordinary Join/demand-end machinery. Exact private staging is not preserved or needed in those controls. |
| Direct `Contract → prepare → publish → Engine` launcher in `execute()` | A common research entry boundary, not a family-specific coordinator or the final production launcher. |

No additional general coordinator is justified by those fixture events. If a
real algorithm requires another owner's milestone to gate execution, that
required relationship must be represented or realized under its accepted
contract; this fixture check does not prove arbitrary hidden event waits safe.

## Concrete unresolved progress case

Both CPU emissions accept, prepare and publish this arrangement:

```text
Producer owns model READ throughout one source interval
  publish demanded product 1 → feed(capacity=1) now contains 1
  publish product 2          → waits for feed space, still owns READ

Consumer asks for key 1
  prefix readiness          → succeeds: key 1 exists
  acquire model exclusion   → waits for producer READ before eval-view use
  protected Join/claim      → cannot run, so cannot free the slot
```

The consumer holds **no writer lease** during this cycle. The producer's read
prevents acquisition. This is different from impossible simultaneous read/write
ownership, and different from a full queue containing only the wrong key.
The existing reorder-capacity rejection does not fire: the requested key exists.

`check_waits()` describes declared source acquisition and consumer readiness/
reservation points. It does not describe a provider's publication wait while
holding a source interval. The missing edge is between two framework-visible
protocols, even though the provider chooses where to use them. Cancellation
successfully releases source access and closes the activity; that is cleanup
evidence, not successful execution or safe retry.

[test_system_progress.py](test_system_progress.py) adds six cases:

- Two limitation cases expose both wait edges and verify that no consumer call
  or handoff occurred, then cancel and drain. No elapsed-time guess is used to
  infer the cycle; the harness deadline is only a leak guard.
- Two positive controls release each product's source interval before its
  publication wait. They complete with the existing mechanism. That is a
  different source-use protocol, not a license to split another accepted
  interval automatically.
- Two controls remove the producer example's cross-owner fixture staging and
  exercise ordinary input coordination and shutdown, as described above.

The smallest justified follow-up is `sd-scripts-syv.19`: make the supported
source-access/publication protocol handle or honestly reject this combination
before indefinite waiting. Determine the required information at that shared
boundary. Do not silently release source protection, claim unadmitted input,
evict queued work, invent a protected snapshot, or make authors change the
algorithm's placement. This finding does not require a universal scheduler or
an IR construct for every queue operation.

## Remaining qualifications and decision boundary

The following are deliberately not closed by this pass:

- Finished ordinary strategy authoring and the full queryable contract. These
  examples write intermediate records directly; they are not a usability proof.
- General structural changes, relationship-aware shared storage, destructive
  preparation, active continuation migration and nontrivial backend groups.
- GPU/device completion, distributed coordination, compiled numerical regions,
  Windows and genuinely GIL-disabled native-stack safety.
- Durable snapshots, restoring a live run and recovery from uncertain outcomes.
  `capture()` checks a finished quiescent in-memory cut; it is not restoration.
- Full G3.7 conformance/performance qualification. Earlier local lowering cost
  measurements predate the latest access/lifetime integrations. They cannot
  establish the present whole-system overhead or production migration gates.
- All possible scheduling relationships. TaskGroup sibling failure and the
  offered completion/request/stop connections are the demonstrated policies;
  configurable isolation, distributed failure agreement and general liveness
  do not follow from them.

Recommendation: retain this graph-derived mechanism as the working foundation,
resolve the reproduced publication/access boundary, then use the connected
mechanics to choose production structures and the author-facing surface.
G5's list of types is not a substitute for this system-level construction and
must not dictate isolated interfaces before their interactions make sense.
Move toward production design without first implementing every future
capability in this candidate.

## Verification

The six new assessment cases, spanning both lowering configurations, pass. Two
cases deliberately preserve evidence of an unresolved limitation; they must
not be counted as successful progress conformance. Runtime implementation and
the nine existing examples are unchanged by this assessment.

Fresh scoped verification: **423 tests pass** (411 candidate cases plus 12
unchanged execution cases), all nine examples execute, and scoped Ruff lint,
format and strict `ty` checks pass. The expected Triton GIL-enablement warning
is present; these are CPU correctness checks, not free-threaded-stack or
performance qualification.

```bash
uv run --locked pytest --noconftest docs_design/models-strategy-trainer/execution_candidate tests/unit/training/test_execution.py -q --tb=short
uv run --locked python -m docs_design.models-strategy-trainer.execution_candidate.examples
uv run --locked ruff check docs_design/models-strategy-trainer/execution_candidate
uv run --locked ruff format --check docs_design/models-strategy-trainer/execution_candidate
uv run --locked ty check --python ./.venv-wsl/bin/python --extra-search-path docs_design/models-strategy-trainer docs_design/models-strategy-trainer/execution_candidate --error-on-warning
```

Code-graph generation `2026-10-24T09:01:16Z` supplied symbol discovery and traces;
coverage checks reported matching source metadata for the six implementation/
example files and ten existing test files used here. Exact source reads and
the executable probes check material conclusions. Graph coverage is best-effort,
not a completeness proof; the scoped excluded `__pycache__` is not code evidence.
