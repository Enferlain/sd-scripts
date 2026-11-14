# Shared xDSL language slice: milestones 6–11

This isolated package implements structural construction, normalization,
shared semantic verification, an executable lifecycle reference and derived
owner-target preparation for the bounded slice in
[D14](../../../openspec/changes/rework-model-strategy-trainer/design.md) and
[workflow-composition](../../../openspec/changes/rework-model-strategy-trainer/specs/workflow-composition/spec.md).
Training and data-only work use the same operation definitions and reference
protocols. The extra consumer scope is another composition of those definitions.

**Normalization alone is not whole-lifetime acceptance or executable readiness.**
[verify.py](verify.py) supplies the shared lifecycle/ownership verifier.
[prepare.py](prepare.py) and [target.py](target.py) derive and execute the bounded
owner target. [Qualification](qualification.md) exercises integrated interruption,
history erasure, detached optimized execution and bounded interleavings.
This package does not train a real model or select public APIs, a numerical
backend or a production migration.

## Representation and contract queries

[dialects.py](dialects.py) defines actual xDSL operations, types and regions.
`OrderedBody` is an execution contract selected by the program and maintained
structured region definitions. Each role starts with its own entry arguments;
an operation may use those arguments and earlier local results. Explicit
captures must already be available at the capturing operation. Repetition's
initialized carry arguments remain valid recurrence; forward local SSA uses,
self-use and sibling-role entry arguments reject with use/definition origins.
Graph execution would need its own availability rule.
Persistent resource and endpoint references are typed SSA values; records and
live model state are outside the IR. Independently invoked bodies use explicit
capture operands and block arguments. xDSL verifies types, region structure,
isolation and use tracking; local project traits check capture signatures.

The installed traits in [contracts.py](contracts.py) return typed descriptions:

| Query | Derived information |
| --- | --- |
| `RegionSemantics` | Once/sequence, parallel join, serial repeat, requested-handler, producer, scoped-access and finite-choice protocols; actual entry/yield/capture types; invocation sources and bounds; closure/interruption/settlement rules |
| `ResourceSemantics` | Actual resource operand, explicit access realization, separate freshness/version-selection policy and required use extent |
| `FlowSemantics` | Actual payload type, aggregate outstanding/ready bounds and owner-specific admission policy with lineage metric and comparison boundary |
| `ReferenceSemantics` | Typed request-source, request-destination, version-owner and stop references exposed for normalization and later dependency analysis |
| `RealizationSemantics` | Versioned selected binding, actual input/output types and required physical/validation boundary |
| `ItemSemantics` / `EndpointSemantics` | Claim admission, borrow, consumption and detached payload results; actual production/admission endpoint operands |
| `CompletionSemantics` / `DecisionSemantics` | Source/target/boundary, registration before source completion and retention of admitted promises; the shared descriptor for leaf calls and completion decisions |

A shared `Decision` description carries reproducibility/rank declarations and
assumptions through region, flow and resource queries. Region contracts also carry explicit quiescence declarations;
an empty declaration means none is offered. These are preserved contract
parameters. Verification derives a reproducibility class and checks declared
cut/rank requirements; actual distributed execution and coordinated restoration
remain deferred target evidence. Missing decision declarations mean `unknown`.

Semantic symbols (`SymbolOpInterface` / `sym_name`) are distinct from source
origins. This first slice uses one program-wide flat symbol namespace;
namespaced modules and cross-program linking are not implemented. Changing a
diagnostic origin does not retarget a semantic reference. Invocation identities
combine parent invocation, static origin and a local ordinal; template/service,
repeat-iteration, production-selection and accepted-request entry are recorded
separately.

[normalize.py](normalize.py) clones the accepted document, expands operations
offering the maintained sequence-expansion interface and uses xDSL's rewriter
to remap result uses. It resolves typed references, checks request payload/entry
correspondence and derives ancestry from the actual region tree. Source coverage,
contract versions, bounds, policy references and links survive normalization.
The normalizer checks before/after region, flow, resource and realization
contracts, semantic parameters and actual SSA-provider wiring. Cloned resource
uses compare by their real SSA provider, not Python object identity. Each
completion-request source must belong to its target handler's declared set.
The authoring object is detached. Parsing consumes the whole input and requires
one program; neither trailing garbage nor a second root becomes an accepted
document. Unknown semantic fields and missing registered definitions reject.

The normalizer checks local structure only. It does not certify producer-stop
coverage, all-path affine ownership, capability escape, hold/wait legality,
whole-lifetime cycles, source completeness or target feasibility.

## Shared verifier and obligation boundaries

`verify(normalize(module))` recomputes from a detached IR copy, using installed
contract queries. It does not invoke bindings or read runtime observations.
Its report contains source-attributed obligations, separate input relationships,
whole-lifetime wait edges and reproducibility contributors. It is a recomputable
analysis, not a readiness token; mutation requires verification again.

| Obligation | Actual query and check | Independent reference witness / remaining evidence |
| --- | --- | --- |
| Pending work and source closure | Region and completion contracts require source closure, owned registration before completion, pending/admitted dispositions and preservation after orderly stop | Late input keeps the group open; no request creates no handler invocation; cancellation disposes pending work |
| Producer stop | Endpoint operands resolve through actual SSA captures; the stop source must settle every dependent consumer/request lifetime | Training-only stop loses evaluation's late input; group stop retains evaluation and producer drain separately |
| Known lifetime cycles | Child settlement, serial entry, source closure, pending ownership, stop and transferred-claim drain requirements form a wait graph | A producer preceding its own stop blocks; a transferred claim cannot wait for producer termination before its disposition; initialized bounded feedback remains valid |
| Affine claims | Item types, borrow/consume descriptions and region execution multiplicity require one consume/transfer per path | Both exclusive branches may retire a claim; duplicate consumption, use after transfer and repeated/concurrent capture reject; explicit carry/return retains one claim |
| Scoped references and capabilities | Capture maps resolve actual owners; capability types retain issuing access scope; retained extent and detached call/borrow results are explicit contracts | Escaped capabilities fail their later use; resource/endpoint references cannot outlive their declaration scope without a defined transfer; the program has no outgoing owner/claim transfer |
| Access and physical extent | Resource query selects realization, separate freshness, version selector, profile and retained use extent; realization query requires physical completion then validation | Held reads retain protection through cancellation and late failure; acceptance does not establish physical settlement or resource usability |
| Nested acquisition | Actual owner aliases and access modes must permit entry while an enclosing scope retains its live protection | Same-owner read/write and write/read/write/current-snapshot nesting reject; read/read, different owners and independently published snapshots remain valid; ambiguous aliases report an analysis limitation |
| Production versions and admission | Flow owners must match actual production accesses; policies compare observed owner/incarnation/lineage maps at `item_admission`, using owner update ordinals | Independent reference admission matches exact-current, bounded-lag and provenance-only decisions; rejected products receive dispositions and never become consumed |
| Reproducibility | Region, flow, access and leaf declarations combine with current-version/update overlap and competing admission/input relationships | Ordered protected/current-snapshot reads observe a fixed epoch; concurrent current selection can observe different epochs; fixed snapshots do not remove admission races |
| Quiescence and rank claims | Cut requirements name a supported declared boundary; replicated decisions require deterministic replicated state or distribution before dependent work | Missing cuts/agreement reject as unavailable target claims; local randomness followed by distribution is valid; coordinated cuts/distribution are not implemented |

`SemanticError` identifies invalid accepted meaning. `TargetLimit` identifies
a valid contract or claim the selected analysis/target cannot establish.
`REFERENCE_TARGET` describes the reference's explicit access/policy/scope limits;
matching this profile is not device, thread or production qualification.
The first version analysis supports `item_admission` and update ordinals within
one owner lineage. Other boundaries/metrics require an installed realization.
An open producer needs a declared progressing normal-stop control to establish
normal settlement; external cancellation is not that proof. Conditional-only
stop alternatives currently report a progress-analysis limitation. The analysis
does not prove general deadlock freedom or termination of selected computation.

The input report's `required_initial_owners` names external owner initialization
requirements, not already initialized runtime state. Initialization, installed
bindings, output validation, physical/device/collective ordering and owner
failure/restoration protocols remain explicit preparation/execution evidence.
Ordinary leaf calls return detached values. The explicit scoped retained-product
contract below supports storage/derivative dependencies through physical uses.
No arbitrary Python return value becomes safe merely because its static payload
type is copyable.
Input consumers are potentially reachable templates, not admitted invocations.
A handler body without a possible request adds no fictional admission/drain
obligation. Claim transfers are followed to their final dispositions; producer
settlement retains those drain dependencies. Parallel branch bodies also own
their transferred locals and dispose them on cancellation/failure, even without
a separate region-operation invocation.
Reference provenance follows explicit captures, yields and initialized loop
backedges. It includes initial carries for zero-iteration/orderly-stop results
and owners introduced by later iterations. These are conservative possible
subjects; execution must select the actual SSA-supplied owner and observed
version, never choose the first owner from an analysis summary. A changing owner
carry can therefore prevent a schedule-independence claim.

## Reference semantics and observable comparison

[reference.py](reference.py) is a single-caller, Torch-free interpreter over
normalized IR. Each invocation follows the control skeleton
`created -> pending -> admitted -> running -> closing -> settling -> terminal`.
Logical success/failure, stop/cancellation requests, physical call outcomes,
cleanup and product dispositions remain separate facts.

`enabled()` exposes legal next logical commands; `step(choice)` supplies an
explicit scheduling choice. `held` and `failures` supply external physical
completion/failure choices. `request_stop(symbol)` closes new repeat/producer
admission at its selected boundary. `cancelled = True` is an external run
cancellation input. `run()` uses a deterministic round-robin choice and returns
when settled or blocked; callers must check `settled`. No actual threads,
futures, device calls or elapsed-time delays are modeled.

The interpreter derives admissions and requests from IR and owned protocol
state. A handler template creates no admitted request. A completion-triggered
request explicitly names an enclosing source and its `settled_success`
boundary. The source retains the emission obligation; request registration
precedes publication of that source's completion. A request waiting for input
or registration capacity keeps its enclosing lifetime open. Orderly stop
preserves already admitted iterations and their requests; cancellation records
unfulfilled work and drains modeled entered calls. Failure-triggered sibling
closure is distinct from a user cancellation request.
Request-source closure and internal stop relationships resolve within their
actual enclosing invocation. A source that closed in an earlier consumer
iteration cannot close the next iteration's handler.

[observations.py](observations.py) fixes the first comparison contract:

- A fact key contains owner/source origin, invocation path, kind and the
  operation-local occurrence or owned item/request identity.
- Facts retain input associations, request cause, observed resource version,
  logical/physical outcome, update, cleanup and disposition. Trigger provenance
  and the handler's input identity are distinct.
- Required causal edges cover local sequential interactions, child/parent
  settlement, product selection/publication/admission/disposition, completion
  requests and their physical use, and consumer completion before producer stop.
- Absolute timestamps, private queue order, global call counters and transient
  internal state observations are excluded. Values use exact fixture comparison;
  a future numerical target needs its separately qualified numerical policy.

The reference is an oracle, not a runtime trace replay. It never consumes an
observed admitted-work trace to determine expected admissions or requests.
Diagnostic history can be disabled without changing its progress or numerical
results. Its fact projection is test evidence and never execution authority.
Parent settlement also waits for accepted pending requests when no handler body
was invoked. An incomplete source/handler composition can remain blocked;
direct self-referential settled-source waits receive a typed diagnostic.
Structural normalization alone is not the shared completeness/cycle check.
Cancellation gives pending requests an explicit disposition.

## Fixture computations and supported reference target

[fixtures.py](fixtures.py) constructs the IR. [standins.py](standins.py) supplies
versioned, typed fixture implementations and domain payload validators through
the same binding interface; the reference has no case-name dispatch or model
arithmetic. The computations operate on copied fixture values and explicitly
declared updates. They are not Python/PyTorch model or optimizer substitutes in
an owner-integrated run.

The model fixture retains the earlier independent arithmetic: two two-member
windows update `0.8 -> 0.77 -> 0.635`, then requested evaluation consumes record
4 at model epoch 2, predicts `6.35` and reports loss `0.91125`. The producer has
an aggregate outstanding bound of three and a separate ready bound of two.
The data-only fixture selects under changing policy, scores `(2*x+1)^2`,
acknowledges feedback and updates the policy through its owner protocol.

The reference currently supports serial count-bounded repeats, serial bounded
handlers, unbounded-total/bounded-live producers, no-result parallel joins,
two-alternative finite choice and declared enclosing-source completion requests.
Its persistent resource/endpoint declarations belong to the program root;
nested persistent declarations are rejected as a reference-target limitation.
It models protected current reads, current-at-start isolated snapshots, retained initial
published snapshots and ordinary owner updates. Published snapshots remain
available independently of subsequent live-owner writes. It records actual owner,
incarnation, lineage and update ordinal. It realizes any-with-provenance item
admission, exact-current admission and bounded lag. The reference implements
admission independently of `evaluate_admission()` and emits comparison,
rejection and disposition facts. One rejected candidate is processed per logical
command, preserving scheduling/cancellation opportunities. Grant profiles,
arbitrary published-version selection, parallel claim transfer, parallel result
merging, distributed targets and coordinated cuts produce an explicit reference
limitation or remain declaration-only; they do not become weaker semantics.
Invocation outcomes are retained for inspection in this finite exploration
tool; its live-obligation bounds do not claim bounded total oracle memory.

Protected access excludes incompatible updates through all modeled entered
uses. A snapshot owns an isolated copy after coherent selection and does not
block subsequent updates. Modeled completion, host return, a version stamp or a
successful reference result do not qualify a real resource as usable or prove
device ordering, thread safety, backend interoperability or performance.

## Prepared artifacts and existing owner target

`prepare(module, manifest, Policy())` is Torch-free. It verifies a detached
normalized document, checks installed binding signatures/versions and explicit
target bounds, then derives a fragment for each actual operation. Fragments
retain SSA providers, result types, role/capture maps, source ancestry, region,
flow/access/completion contracts and their selected primitive. The prepared
recipe also contains source-attributed obligations, input relationships, wait
edges, reproducibility contributors and binding/resource/target manifests.

`Artifact` contains separate accepted, normalized and prepared documents.
`load()` parses complete documents and recomputes all correspondence from the
accepted IR and installed definitions. It checks normalized structure and the
entire prepared recipe; saved summaries and hashes cannot authorize execution.
Initial published snapshots are materialized only for declared published-snapshot uses; protected-current-only resources acquire no copying obligation.
`Runtime` repeats that barrier against the actually installed registry before
initializing resources or publishing its owner image. This is this target's
chosen ordering; pure construction before final verification is not prohibited
by the language. Cold loading starts a new run with explicit initial resources;
it is not exact restoration of a prior runtime.

The checked-in [artifact examples](evidence_prepared/qualification.json), whose hashes and cold execution are tested directly, cover
the model lifetime, extra consumer scope and data-only compositions. The
prepared JSON is an inspectable lowering recipe, not an independent semantic
authority. The target emits its local closures and owner primitives from the
reverified normalized IR, installed contracts and matching realization policy.

| Derived meaning | Existing service / target adaptation |
| --- | --- |
| Root readiness and run | One `RunState`, installed image and `Engine`; no second root coordinator |
| Parallel lifetime | `Lowering.scope(Together(...))` supplies TaskGroup failure/cancellation and joins |
| Serial repetition | `Lowering.scope(Repeat(...))` selects iterations and supplies completion; the prepared invocation guard admits a selected attempt, and locals/carries remain invocation-owned |
| Requested activity | `Lowering.scope(Requested(...))` supplies serial request execution; the adapter registers completion debt before source terminal publication and bounds pending plus active work together |
| Independent producer | `Lowering.scope(Producer(...))` supplies stop, cancellation, physical cleanup and port closure; the provider owns selection and bounded publication |
| Local bodies and choice | Prepared local closures interpret actual SSA captures/results and finite choice; they do not create another scheduler |
| Ready and admitted items | Existing `Feed` owns bounded ready products; a first-eligible claim adapter evaluates actual versions under its condition. A separate aggregate reservation remains live through physical work, ready state and final claim disposition |
| Protected current model | `RunState.use` acquires actual live protection and current stamp together; protection survives physical completion, validation and cancellation drain |
| Current snapshot model | `RunState.retain(isolated=True)` coherently copies, verifies separate storage and preserves the acquired stamp; the isolated copy outlives the live read lease and is released after its uses |
| State cells | Existing owner-state acquisition protects a domain value and its update ordinal; snapshots copy under acquisition, then release it before use. This target serializes live cell reads as well as writes |
| Physical selected calls | Existing `Workers.execute` owns dispatch entry and completion; only selected arguments enter workers, with event-loop-owned observations and effect application |
| Model advancement | Existing contribution, advancement, optimizer and reset protocols; their returned/uncertain/not-attempted evidence survives failure |

The old `graph.Contract` accepts the original experimental language. The new
shared verifier remains the semantic authority: the target constructs the
owner engine's `Accepted` image descriptor from checked declarations and emits
its structured primitives, rather than fabricating an old-language producer
tree for that contract. This is reuse of one ownership implementation, not
independent validation of another owner engine. Engine counters, resource
leases, pending requests and claims remain live control state; diagnostic fact
lists do not select execution.

[target_bindings.py](target_bindings.py) reuses the existing composition's scale
loader, encode, objective and measurement functions, never its whole-case
`lower()` builder. The scalar fixture's `gradient` returns a detached numerical
gradient. `advance` feeds that declared gradient through a linear scalar loss
to the existing contribution/step/reset owner. This is a bounded realization
of these bindings, not general autodiff or derivative routing. The model uses
CPU float64; the data-only workflow has no model binding or optimization unit.

The first target supports root persistent declarations, serial bounded-count
repeats and one live request per handler, no parallel result/claim merging,
initial-published or current-at-start snapshots, protected-current reads and
ordinary owner updates. Completion sources with multiple service invocations
inside one relationship scope, multiple completion stops for one producer,
nested live acquisition of the same state-cell owner, grants and other
unsupported profiles reject as explicit target limitations. Snapshot and
protected-current selection come from the IR; a target never substitutes one.
Runtime instances are single-use. Selected Python implementations remain trusted
and independently checked for numerical conformance.

Preparation checks each installed binding's declared update positions against
the actual SSA-supplied capability, its issuing access and possible resource
identities. An enclosing writer does not authorize mutation through a captured
read capability. Snapshot/protected-read updates reject before resource factories
or physical entry. Runtime checks still enforce the actual capability's active
authority, resource version and reported effects. This validates the current
update contract and the shared selected-binding effect requirements described below.

## Selected effects, filtering and local fusion

[BindingSpec.contract](prepare.py) supplies one typed `BindingContract` for
preparation and [local transformation legality](local_rewrite.py). Its minimal
effect summary contains resource read/update argument positions, optional resource
kinds and possible unknown effects. Preparation checks those requirements against
actual capability issuers, resource identities and modes. The same descriptor
permits fusion only for known-pure detached unary stages with compatible SSA
wiring. An explicit empty resource set and `unknown_effects=False` declare purity;
missing legacy effect information remains opaque. Neither unknown effects nor a
declaration creates invocation authority. The runtime still checks actual reported
updates, active authority/current versions, output validation and physical settlement.

The original `shared-owner-fragments-v1` documents retain their original binding
and policy spelling: absent new effect fields stay unknown, and absent fusion policy
means unfused. Explicit effects/fusion add fields to that inspectable recipe. Cold
loading compares the full installed descriptor and rederives the rewrite; it cannot
silently reinterpret an old artifact as pure or fuse it. No prior evidence is rewritten.

`lang.filter` consumes exactly one admitted claim. Its body receives a detached
copy of the input payload and yields a boolean; the operation returns a typed
`lang.optional` zero-or-one selection. A false result settles the input as
`filtered_out`; a true result settles it as `forwarded`. Both retain protection
through physical predicate completion. A kept output retains the original payload,
observed upstream versions and separate provenance, even if a predicate changes its
own copy. Selection envelopes originate from the cardinality protocol, never an
arbitrary physical binding. Run failure/cancellation prevents a new selection after
physical completion; interrupted inputs settle as discarded/cancelled respectively.
A producer reserves aggregate capacity before this work;
an absent result retires that reservation as `no_output` without publishing an item.
Actual upstream version dependencies are followed through the forwarding contract.

The [concrete case](filtering.md) uses an independent producer, a filtering producer,
one initial retained consumer and a completion-requested two-iteration consumer.
Odd inputs drop; keys 0, 2 and 4 produce 2, 6 and 10. Drops release capacity while
requested demand remains pending. Consumer completion stops filtering; filtering
settlement stops source production; the run joins both. Cursor/ticket/queue mechanics
remain runtime details. Flow settlement receipts retain actual dispositions, versions,
provenance and forwarding destinations after diagnostic erasure. These are finite-case
owner receipts, not a production retention policy.

`Policy(fuse_local=True)` selects actual xDSL rewriting with result-use remapping.
Derived `lang.fused_local` groups have the enclosing invocation's lifetime and no
completion symbol. Their generated coroutine calls each original stage directly,
preserving source identity, physical entry/return, validation, failure, interruption
and item associations. A transparent group adds no decision contributor: original
stage reproducibility and rank requirements remain authoritative. It replaces three local interpreter dispatches with one in
this case while retaining all three physical calls. Cold loading recomputes legality
and canonical correspondence. Unknown/resource effects, extra SSA uses and enclosing
authority/completion/cardinality boundaries constrain grouping. The valid opaque
identity variant runs successfully and rejects a forged group crossing its barrier.
Dispatch counts establish a distinct path; they do not establish a speed advantage.

Focused live/cold tests cover the independent arithmetic and request/input
associations, pending work and late input, orderly stop after window admission,
cancellation through held physical use, late physical failure and a partial
optimizer effect. Physical settlement is separate from resource usability:
failed live use withholds that owner; a failed optimizer may have changed the
weight while its outcome remains uncertain. Producer cleanup waits for
consumer-owned claims through their physical uses before retiring them. Disposition removes a product from both live reservations and ready storage; a disposed request packet cannot enter its handler. A nested driver group retains child-generated pending requests until handoff to its enclosing handler. Failure/cancellation can dispose that debt. Settlement checks precede a region's own completion-generated emissions, preserving their registration before terminal publication.

External producer stop wakes its native stop service even while aggregate capacity is exhausted; admission rechecks the signal after acquiring a permit. Ready work can be discarded, while a consumer-owned claim remains live through its physical use. Stop before producer service entry selects no work. These boundaries have independent reference controls.

A three-window variant stops after window-two admission and verifies that the third window is not admitted while window two and its request finish; its consumer and stop facts are compared with the reference. Scoped producer-stop flags are retained.

Differential checks compare model consumer facts and causal requirements with
the independent interpreter, allowing harmless producer run-ahead. Data-only
checks use an aggregate horizon of one to fix feedback order. Two further
comparisons supply the same external phases independently to each execution:
old prefill, two completed owner writes, then admission under bounded lag.
Both snapshot and protected-current variants reject records 0 and 1 and admit
record 2 using its actual observed version. A held production read separately
shows that snapshot use permits an independent update while protected-current
use excludes it. Version selection alone does not remove admission-related
schedule dependence. Neither execution is replayed from the other's trace.

The invocation admission guard now rechecks local stop after selection and before
body entry. A real selected/pending second-window attempt can settle as
`stopped_unadmitted`, matching the reference's separate admission command.
Stopping after admission retains that window and its requested evaluation.
Stopping before selection creates no attempt in either execution. Bounded
producer run-ahead is classified separately; score, update and consumer facts
permit no truncated surplus. Bounded interleavings establish these tested
boundaries, not equivalence under every possible schedule.

The existing owner target imports Torch, including its worker context and
optimizer adapter. Language-only checks remain Torch-free. The target checks
reuse the two existing environments without installation or root dependency
changes: the qualified xDSL site directory is added to the existing project
interpreter's search path. Record actual GIL state before imports and after
checks, without forcing it. These CPU tests do not qualify GPU/device ordering,
replicated execution, native-library concurrency generally or performance.
The [qualification record](qualification.md) adds owner-target history erasure,
fresh normal/optimized children and seeded suspension checks. Earlier
reference-only children remain separate evidence.

## Scoped resource-backed products

[The retained-product case](products.md) adds `lang.retained` results with
explicit capability/product dependency positions and `lang.release_product`.
A product belongs to its defining ordered body, permits read-only physical
borrowing and retains actual dependencies/observed versions until release.
Derived products release before their parents. Yields, nested captures,
publication, use after release and known conflicting live-owner updates reject;
release of a snapshot product does not depend on the live owner's current epoch.

Preparation checks the same installed retention declaration and derives cold
correspondence before factories. The reference models logical products and
independent arithmetic; the existing owner target carries actual memory views
and non-leaf Torch graphs without copying them. Cleanup retains entered uses
through repeated cancellation and late failure, then clears live product
references before access protection ends. The existing detached-only fusion
keeps these product boundaries. No new scheduler or production API is introduced.

The two witnesses exercise read-only buffer aliases and real CPU differentiation
through a frozen base into a trainable leaf. This is explicit product lifetime
evidence, not general model/optimizer routing, escaping graph support, mutable
alias safety, device qualification or bounded historical bookkeeping.

## Run the bounded checks

Reuse the existing qualified environment; no root dependency or lockfile update
is needed. A fresh isolated environment can use the existing pinned
[requirements](../xdsl_candidate/requirements.txt), including xDSL **0.74.0**:

```bash
uv venv /tmp/sd-scripts-xdsl-b-env --python .venv-wsl/bin/python
uv pip sync --python /tmp/sd-scripts-xdsl-b-env/bin/python \
  docs_design/models-strategy-trainer/xdsl_candidate/requirements.txt

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/sd-scripts-xdsl-b-env/bin/python -m pytest \
  --noconftest docs_design/models-strategy-trainer/xdsl_language \
  -q -p pytest_asyncio.plugin
/tmp/sd-scripts-xdsl-b-env/bin/ruff check docs_design/models-strategy-trainer/xdsl_language
/tmp/sd-scripts-xdsl-b-env/bin/ruff format --check docs_design/models-strategy-trainer/xdsl_language
# Existing owner target, with its existing Torch dependency:
PYTHONPATH=/tmp/sd-scripts-xdsl-b-env/lib/python3.14t/site-packages \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-wsl/bin/python -m pytest \
  --noconftest docs_design/models-strategy-trainer/xdsl_language \
  -q -p pytest_asyncio.plugin
/tmp/sd-scripts-xdsl-b-env/bin/ty check --python .venv-wsl/bin/python \
  --extra-search-path /tmp/sd-scripts-xdsl-b-env/lib/python3.14t/site-packages \
  --extra-search-path docs_design/models-strategy-trainer \
  docs_design/models-strategy-trainer/xdsl_language
```

Checks include three genuine IR compositions, registered operation expansion,
SSA remapping, explicit-capture corruption, source/symbol separation, complete
document parsing, independent numerical expectations, late input/pending work,
iteration admission versus orderly stop, held physical uses under cancellation,
late failure, retained partial updates, missing/corrupted observable facts and
fresh normal/optimized children with authoring absent. Verifier controls include
early stop, pending coverage, known cycles, actual capture remapping, claim
duplication/branching/carry, capability escape, owner lifetime, access selection,
multi-owner admission, missing decision metadata and unavailable cut/rank claims.
Seeded schedules and the integrated controls qualify the explicitly bounded
slice, with the limits in [qualification.md](qualification.md).
Observe actual GIL state before/after imports and tests without
forcing it. Interpreter and IR objects remain confined to their calling thread.
