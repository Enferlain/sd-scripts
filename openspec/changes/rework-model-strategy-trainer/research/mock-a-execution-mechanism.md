# Mock A: retained semantic graph with prepared executable regions

Recorded: 6 October 2026.

Status: non-normative candidate captured from the design walkthrough in chat.
This is not an adopted architecture, a finalized language/API, a production
implementation, or experimental evidence. It does not complete a G5 task.
Names and pseudocode illustrate responsibilities, not proposed public classes.

Purpose: reconstruct Mock A without recovering the entire conversation, while
preserving its corrections, implementation choices, worked interactions and
limitations. This is a snapshot of the candidate, not another governing design.

## 1. Authority and the defining choice

The [governing design](../design.md) and applicable delta specs supply the
requirements. In particular:

- [Accepted training execution](../specs/accepted-training-execution/spec.md)
  supplies whole-run structure, dynamic behavior, equivalent execution,
  authority, correlation, completion/failure and protected-use requirements.
- [Runtime preparation](../specs/training-runtime-preparation/spec.md) supplies
  coherent preparation, ownership separation, support evidence and publication.
- [Participant state](../specs/run-participant-state/spec.md) and
  [optimization](../specs/training-optimization/spec.md) distinguish semantic
  identity, revisions, runtime realizations and their owners.
- [Capabilities](../specs/training-capability-coordination/spec.md) and
  [persistence/restoration](../specs/training-artifact-persistence/spec.md)
  govern their respective cross-owner interactions.

The pre-OpenSpec
[direction](../../../../docs_design/models-strategy-trainer/strategy_system_direction.md)
remains normative input as described by the governing design. Existing
research, including [answer-edited.md](answer-edited.md),
[trainer-architecture-research.md](trainer-architecture-research.md),
[trainer-boundary-adversarial-review.md](trainer-boundary-adversarial-review.md)
and [trainer-architecture-concrete.md](trainer-architecture-concrete.md), is
supporting material rather than authority for this mock.

The defining implementation choice is:

> Retain the accepted hierarchical semantic graph. Preparation derives directly
> executable regions and the coordination machinery their relationships need.
> A notification-driven supervisor coordinates externally relevant work while
> long-running selected work executes outside its serialized coordination loop.

The graph determines the run. The engine is not a fixed numerical training
sequence with configurable helpers. Training machinery and supported
capabilities shape the pre-existing contract; authoring is guided by that
contract, not an independently invented strategy API that dictates the engine.

| Inherited architectural direction | Mock A's candidate realization |
| --- | --- |
| Hierarchical whole-run meaning and explicit relevant relationships | Retained semantic graph with derived executable attachments and indexes |
| Selected behavior remains dynamic after authoring ends | Bound implementations and policies operate on current inputs and owned state |
| One governed participant authority, other owners retain their state | Coordination records associate owner facts without becoming a competing binding map |
| Required interactions survive lowering | Direct region executables plus guards, reports and relationship handlers |
| Independent progress and shared current-use protection | Serialized coordination loop with backend-compatible execution contexts |
| Explicitly supported alternative authority | Prepared granted-region executors with scoped services and checked handback |

The serialized supervisor, notification transport, attachment storage and
execution-context layout are candidate choices, not settled requirements.

## 2. Whole-system path and the two-form clarification

```text
Pre-existing training contract
            |
     guided authoring
            |
Authored strategy definition
            |
   contract fulfillment
            |
Accepted run: semantic graph + selected behavior + policies + ownership
            |
   governed preparation / lowering
            |
Executable form derived from that graph
  region executables + coordination handlers + dependency evidence
            |
Engine coordinates execution through supported backend mechanisms
```

The authored strategy is not retained as the ordinary runtime collaborator.
Its selected executable behavior, policies, state definitions and permissions
survive in the accepted run. The authoring object can be released.

There are semantic and executable forms, but not two independently authored
descriptions of the run. An "entry" is a callable entry point into a lowered
region, not another recipe assembled between graph construction and execution.
The graph's cross-owner and lifecycle relationships are lowered too.

For the concrete storage sketch, keep one semantic graph and a derived table
of executable attachments. This does not require a second complete persistent
control IR. Exact storage, classes and the Python/native split remain open.
Preparation may fuse or partition work, so attachment correspondence need not
be one callable per graph node.

The retained graph records accepted meaning. Its current executable form
records how that meaning runs with the established bindings and backend.
Each executable remains associated with its source meaning and preparation
dependencies. Permitted changes invalidate or rebuild affected attachments;
runtime code must not independently rewrite the meaning.

## 3. Reference graph and relationships

The ordinary illustration has a frozen text encoder producing conditioning,
a trainable predictor, adaptive objective/sampler state and requested
validation. The payload names are domain examples, not common-language fields.

```text
RUN
 |
 +-- P: independent input production
 |       source -> caption selection -> selected encoding
 |                                      |
 |                        product identity and provenance
 |                                      |
 |                               admitted handoff
 |                                      v
 +-- L: repeated learning
 |       selected computation
 |           |             |
 |           |             +-- observation -> adaptive state S
 |           v                                      |
 |       differentiation                 later accepted decisions
 |           |
 |       optimization unit U
 |           |
 |           +-- unit outcome -> selected validation policy
 |                                      |
 +-- V: requested validation <-----------+
```

Requested sampling, product persistence, snapshot capture and governed
transitions can also be work in the whole-run structure. Their eventual
coordination cannot be reduced to a universal end-of-step hook. They were
discussed conceptually, not worked through as implemented activities here.

Connections have distinct meanings:

| Relationship | Meaning |
| --- | --- |
| Value flow / handoff | Work consumes identified values produced elsewhere |
| Ordering | A particular boundary must complete before dependent work |
| State access and lifetime | Work needs current or retained state, protected against conflicting use |
| Feedback | An observation affects subsequent accepted decisions |
| Lifecycle / failure | Starting, stopping or failing work has accepted consequences for other work |

An update waiting for a derivative's last use is not an ordinary tensor edge.
Private tensor mathematics and provider worker schedules need not enter the
common graph. Expose relationships when fulfillment, preparation, another
owner, coordination, observation or restoration needs to judge them.

## 4. What preparation produces

Preparation derives executable entries and the coordination information needed
at their boundaries. It does not infer algorithms or invent admission policy.

| Accepted meaning | Derived executable attachment | Why it survives |
| --- | --- | --- |
| Selected implementation | Bound Python call or supported compiled executable | Avoid repeated selection and lookup |
| Producer/consumer identity and admission rule | Matching/admission handler and readiness notification | Products arrive during execution and can be inadmissible |
| Observation recipient and required order | Bound receiver and ordering enforcement | Feedback must reach its owner before the dependent decision |
| Participant uses and unit membership | Checked references to owner-managed realizations | Physical handles do not establish semantic identity |
| Numerical-state lifetime | Access and completion machinery | Python return need not end actual use |
| Activation policy | Bound policy with its owned state | Decisions remain dynamic |
| Preparation dependencies | Evidence associated with affected executables | Current state can change while a handle still exists |

Private intermediate values may become local variables. Required boundaries
may survive inside combined code as guards, scoped access, completion facts or
reports rather than separate engine dispatches. Generating training-owned
mechanics into that executable does not grant them to selected computation.

Preparation is also backend realization, not only code generation:

```text
Coherent accepted state and obligation revision
        |
Resolve participants, views, units and supported relationships
        |
Required ordered / joint backend preparation
        |
Bind or lower regions and coordination relationships
        |
Check correspondence, support, source freshness and complete evidence
        |
Coordinated installation across the relevant owners
```

Required frozen participants still participate in preparation. Optimizers must
target the final accepted parameter realization or have evidenced rebinding.
Expected-fallible work precedes final publication; no owner publishes its
portion and leaves another owner to catch up. Unsupported derivative, retention
or granted-phase support fails readiness without weakening accepted meaning.

Maintained domain integrations supply knowledge such as caption/encoder
provenance checks. Backend integrations supply support and completion evidence.
Runtime checks remain runtime checks: preparation installs admission machinery
rather than predicting whether every future product passes it.

## 5. Worked lowering: explicitly feedback-first learning

The initial informal diagram did not establish whether feedback preceded
differentiation or advancement. The earlier pseudocode placed it before them
without deriving that placement. The correction was to make the source
arrangement explicit for the walkthrough.

Feedback-first is this example's accepted order, not a proposed repository
default or universal requirement:

```text
Admitted input
      |
Choose using current adaptive state S
      |
Selected computation
      |
Apply observation to S
      |
Differentiation
      |
Advance unit U
      |
Invocation completes
      |
Next learning invocation becomes eligible
```

Here the objective/sampler implementation owns S, learning repeats one
invocation at a time, standard differentiation/advancement remain
training-owned, and failure gates the remaining and dependent learning work.
The independent producer can continue under its accepted lifecycle/capacity
policy. Routine source details can be supplied by maintained composition.

The candidate transformation rule is to combine ordered region work only when
all required interactions survive. Preserve source order, dynamic decisions,
ownership/scope, outcomes and failure boundaries, actual-use protection and
required external interaction opportunities.

Illustrative lowered body:

```python
def learning_entry(admitted_input, protected_use, attempt):
    choice = choose_policy(sampler_state, admitted_input.coordinates)
    result = compute(admitted_input, protected_use, choice)
    apply_feedback(result.observation, sampler_state, attempt)
    backward(result.backward_value, protected_use, attempt)
    advance_unit(protected_use, attempt)
```

Names and result members belong to this example. They are not a universal
process-batch interface or a finalized result class. The calls represent bound
implementations with required owner accounting/completion protocols; the body
alone does not implement those protocols.

| Failure boundary | Required retained facts |
| --- | --- |
| Feedback fails after possible mutation of S | Feedback effects may have occurred; differentiation and advancement are unattempted; dependent learning stays gated |
| Differentiation fails | Feedback completed; gradient state may be incomplete/uncertain; advancement is unattempted |
| Optimizer returns, scheduler work then fails | Known optimizer outcome remains distinct from scheduler failure and remaining unattempted work |

Moving feedback after advancement changes what can already have happened when
feedback fails. Equal successful numerical results do not justify that move.
Likewise, precomputing several invocations' choices before their intervening
feedback violates the source's adaptive behavior.

A deterministic check could start with S selecting A, then make the first
observation change S to select B. Required choices are A then B, not A then A.
Injecting partial feedback failure should retain possible state effects,
leave advancement unattempted and prevent the dependent invocation. These are
proposed checks, not tests run or evidence obtained during this mock.

## 6. Concrete engine candidate

The chosen engine realization has a serialized coordination loop with work
executing outside it. "Supervisor" names that coordination role; it does not
require a dedicated thread or one thread per activity.

```text
Prepared region executables + derived coordination handlers
                         |
                         v
+------------------------------------------------------+
| Engine coordination                                  |
| Region lifecycles and activation requests             |
| Readiness, admission and current-use reservation       |
| Completion, failure and dependency tracking           |
+------------------------------------------------------+
       |                         ^
       | submit permitted work   | owner/backend facts
       v                         |
+------------------------------------------------------+
| Backend-compatible execution contexts                 |
| Numerical work | Provider workers | Capability work   |
| Preparation work                                     |
+------------------------------------------------------+
```

Long computation and blocking provider reads must not monopolize coordination.
Actual context placement follows backend support: serialized numerical work,
provider-private workers or supported native/asynchronous execution are possible.
Physical concurrency is not promised merely because activities are independent.

Schematic loop:

```python
while run_is_active:
    notice = notifications.receive()
    installed_handlers.route(notice)
    for request in affected_pending_work():
        reservation = current_use.try_reserve(request)
        if reservation is not None:
            register_attempt_and_submit(request, reservation)
```

A request means accepted work wants to start, not that it ran or is ready.
Reservation combines applicable admission/readiness with protected access.
Relevant freshness must be re-established under the same current-use protocol
that controls conflicting publication; a prior standalone check is insufficient.
Fallible policy/backend work is not implicitly executed inside a long-held
coordination lock. The exact acquisition implementation remains unbuilt.

Attempts are associated before submission. Dispatch rejection is distinct from
an entered call that failed or had an uncertain dispatch/execution outcome.
The loop routes notices to indexed affected relationships, not a full graph
scan. It is not an interpreter for every numerical operation.

Handlers are derived from the accepted connections:

- P product -> input readiness -> identity/admission check -> reconsider L.
- U outcome -> selected validation policy -> register any V request.
- L completion -> selected repetition policy -> possible further L request.
- L failure -> gate unsafe dependencies -> accepted failure/lifecycle policy.

Repeated learning, independently progressing production and requested
validation retain different lifetimes/clocks. One root contains them without
imposing one progress counter or universal completion boundary. Selected
policies decide due work; the engine does not consult the authoring object.

## 7. State ownership, handoff and protection

| Engine coordination tracks | Relevant owner retains |
| --- | --- |
| Producer lifecycle and outstanding handoffs | Queue, traversal, workers and continuation state |
| Learning invocation associations | Objective/sampler algorithm state |
| Which unit outcomes belong to an invocation | Optimizer/scheduler state and unit progress |
| Pending validation requests | Validation algorithm state and results |
| Outstanding protected uses and their relationships | Canonical bindings, backend realizations and owner-controlled state |

There is no unrestricted global run object passed to every implementation.
The authority retains participant/relationship truth; coordination records do
not establish another binding map or another conformance authority.

For the candidate access mechanism, preparation resolves relevant semantic
scopes to physical uses, including aliases and inseparable backend groups.
The runtime reserves the required scope together or leaves the request waiting,
rather than indefinitely holding partial acquisition. Exact modes/granularity
remain implementation work; conservative exclusion is not proof of optimal
concurrency or support for every arrangement.

```text
Reserve current use
        |
Execute within accepted scope
        |
Required backward / recomputation / transfer / device use finishes
        |
Establish usable state, or withhold affected ordinary use
        |
Release execution rights
```

Python return is not necessarily backend completion. Completion can wake waiting
work only according to established facts. Releasing rights after failure does
not certify uncertain state as usable. Authorized recovery must remain able to
restore or replace withheld state. Keep primary and cleanup failures distinct.

Notifications associate source work, invocation and prepared generation with
owner-supplied facts. Late older-generation facts remain historical evidence but
do not make a newer executable ready by matching an address. The transport is
not automatically a durable log or origin truth for optimizer state.

## 8. Independent-progress walkthrough

Assume this consumer's accepted ordering wants input A before B:

| Moment | Required behavior |
| --- | --- |
| Startup | Start P; L waits for required input |
| B finishes first | Record B's identity/provenance; do not substitute it for A |
| A finishes | Check the handoff, reserve required use and submit L |
| L computes | P can progress within supported access/capacity constraints |
| V is requested | V waits if its access conflicts with outstanding learning use |
| Required learning use ends | Release only when applicable state usability is established; reconsider eligible work under accepted policies |

Startup acknowledgement, product readiness, admission, handoff, consumption,
gradient contribution, advancement and durable capture are different facts.
This input ordering is an example policy, not universal FIFO.

P owns its private queue; handoff capacity must also be bounded. This mock
suggests bounded delivery capacity administered by input coordination. A full
downstream buffer slows/pauses production under the provider's accepted policy.
The notification transport must not become an unbounded tensor queue.

Products carry the identity and actual state provenance the consumer needs,
possibly structured provenance rather than one version. In-flight work across
encoder changes requires supported protection/provenance and admission policy;
neither automatic discard nor automatic acceptance is a universal answer.
Precomputation must not silently replace a required gradient through a trainable
encoder. Ordinary computations receive admitted values, not hidden blocking
queue reads.

## 9. Changes, persistence and shutdown

For a supported replacement-only path:

```text
Permitted replacement request
        |
Gate new conflicting activations
        |
Build candidate under the protections preparation itself requires
        |
Outstanding conflicting uses finish
        |
Recheck source dependencies, obligations and candidate evidence
        |
Coordinated publication across relevant owners
        |
Install affected executable attachments; reconsider waiting work
```

Protection may be needed before backend preparation starts, not only at final
publication. Dependency evidence identifies affected work; unrelated guarantees
may remain usable. Failed unpublished replacement preserves only still-valid
prior state. Destructive in-place work instead withdraws affected guarantees
before mutation and cannot regain usability by restoring flags. Backend-group
closure and unit-definition dependencies remain relevant even when participant
identity is unchanged.

Ordinary weight, adaptive-state and optimizer-state evolution does not
automatically revise accepted topology or invalidate preparation. Produced-data
validity can nevertheless change under its own dependency policy. Binding,
composition, unit-definition and prepared-generation changes are not the same
as numerical-state evolution. An identity/revision label does not preserve
old numerical contents.

Snapshot/restoration was outlined, not implemented: a capability obtains a
supported recoverable cut across required owners, using their agreed boundaries,
protected versions and unfinished-work rules. A conservative first candidate
uses required-owner quiescent cuts, not a universal end-of-training-step cut.
Supported disposable in-flight work may be regenerated without collapsing
handoff, consumption and update-contribution distinctions. Unknown effects do
not acquire rollback, exactly-once or replay guarantees.

Restore the same accepted logical run and required owner contributions, then
re-establish backend readiness and executable attachments in the new execution
session. Do not deserialize process-local callables/handles and assume they are
ready. Same-run participant identity and new-run artifact lineage remain
distinct; trained-product completion is not exact runtime continuation.

Shutdown gates relevant admission, follows accepted producer cancel/drain rules
and accounts for outstanding actual use and effects. Cancellation does not
guarantee immediate backend interruption or safe state. Recovery and metadata
consume established owner facts, not a fabricated global success flag.

## 10. Other arrangements and explicitly granted work

These are mappings the candidate proposes, not demonstrations of support:

- Joint units: one prepared computation may feed shared or distinct derivative
  work under the accepted plan. The prepared mechanics preserve separate unit
  responsibilities/outcomes, including one returning while another is uncertain.
  Physical composites do not replace semantic unit identities. Aliases within
  the same accepted membership are not automatically cross-unit overlap.
- Alternating adversarial work: distinct accepted actions/units have their own
  activation policies. The engine does not need a G/D-specific numerical loop,
  and an alternating cycle is not an atomic transaction.
- Granted two-pass work: reserve the scoped interval, run the accepted executor,
  verify handback, then perform retained training-owned mechanics. The chosen
  narrow profile gives the region two computations/backwards, gradient inspection,
  temporary edits, intermediate gradient control and restoration, but not final
  optimizer/scheduler advancement. Protection spans handback and retained uses
  without a gap. No second standard backward is applied to handed-back gradients.
  Capability readers/snapshots cannot enter an excluded intermediate state.
  Pending-gradient and backend-support conditions are checked before execution.

Neither arbitrary Python nor a native implementation is inherently sandboxed.
Supported grants need trusted implementations and appropriate evidence. The
candidate does not imply that every current Accelerate/DeepSpeed combination
can implement the chosen split or reveal all required outcomes.

## 11. Python, native execution and performance

The mock does not require a project autodiff engine or a graph of every tensor
operation. Existing numerical mechanisms own kernels, derivative formulas and
backend execution; preparation establishes their supported realization.

Free-threaded Python was discussed as potentially useful for parallel CPU-side
Python work, not as a prerequisite. The entire pipeline need not be Python for
its Python portions to benefit. If coordination and other CPU portions become
native, the interpreter restrictions affect less work. No current dependency
compatibility, 3.13t/3.14t comparison or performance result was verified here.

Not every local observation/update must round-trip through the supervisor.
Closed-region feedback can remain a direct call. Combining repeated work or
chunking is permitted only when required admission, capability, transition,
snapshot and failure interactions remain available. A cheap workload must
expose framework overhead; model compute cannot be assumed to hide it.

## 12. Assessment, open details and reconstruction guide

Current assessment: Mock A is coherent and worth keeping as a serious
candidate. Confidence is stronger in graph-derived execution with retained
meaning than in a single central notification loop as the final runtime.

Its strengths are connected responsibilities, selected dynamic behavior,
owner-preserving realization, independent activities, explicit research
authority and room for Python/compiled/native execution. Ordinary authoring
should compose maintained behavior, not manually implement notification,
reservation and failure bookkeeping.

Concrete risks and unfinished details:

- Exact semantic constructs, representations and transformation rules are
  still unspecified. "Generate handlers" must become a small faithful
  translation, not special cases reconstructing intent.
- A centralized supervisor can bottleneck or impose unnecessary serialization.
  One coherent protection/coordination protocol does not require every logical
  decision or update to pass through one dispatcher.
- Physical alias correspondence, completion tracking and access granularity
  must have usable implementations. Coarse protection can obstruct independent
  work; unsupported narrow protection is unsafe.
- Long executables can remove required interaction opportunities.
- Retained meaning and derived attachments can diverge if source dependencies,
  invalidation and installation are wrong.
- Public API proliferation can make ordinary strategy code cumbersome even
  when each internal semantic distinction is justified.

No language implementation, handler generator, live backend integration,
distributed publication/recovery, snapshot protocol or hot-path benchmark was
built or validated in this walkthrough. Successful ordinary-region illustrations
do not prove the whole-run, joint-unit or granted-region claims.

To reconstruct the candidate, read sections 2-3 for the whole-run meaning,
section 4 for the derived execution form, section 5 for the ordering/failure
transformation, sections 6-8 for runtime coordination, and sections 9-10 for
change/recovery and nonordinary-work connections. Keep this status/assessment
alongside the diagrams so illustrative restrictions do not become requirements.

A suggested later comparison was a candidate lowering more whole-run control
into explicit executable control flow with a smaller outer coordinator. That
alternative has not been mocked, selected or shown superior. No production
shape should be frozen merely because Mock A was the first complete discussion.
