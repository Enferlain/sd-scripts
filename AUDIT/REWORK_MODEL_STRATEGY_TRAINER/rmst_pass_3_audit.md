Pass 3: execution, optimization and failure.

Apply the shared general audit protocol. This is a read-only concern audit,
not an implementation task or another architectural survey.

Audit commit:
6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48

Confirm that the examined sources correspond to that commit. Identify any
additional working-tree content separately; do not silently combine revisions.

Use previous reports as navigation, not authority. P1-REV-01 and P2-REV-01
have local corrections outside this baseline. Reference those findings where
relevant rather than assuming their corrections exist in the audited sources.

Objective

Determine whether execution and optimization semantics support the intended
hierarchical whole-run composition, dynamic selected behavior and explicitly
granted alternative authority.

Follow contributions, state lifetimes, meaningful effects, completion,
handback and failure across owners. Compare the architectural claims and
task acceptance criteria with what the experiments actually establish.

Do not interpret the target as a configurable universal
input → action → optimizer pipeline.

Sources

- Read the governing source-authority rules and relevant design decisions,
  especially D5, D9 and D10.
- Read the complete G3.1–G3.7 discussions relevant to this concern, including
  standard policies, changed authority, differentiated work, equivalence,
  cross-owner handoffs and failure.
- Read complete requirements/scenarios in accepted-training-execution and
  training-optimization.
- Follow applicable requirements through training-contract,
  training-runtime-preparation and run-participant-state.
- Include capability, persistence/restoration and observation requirements
  where execution depends on them.
- Consult the working sketch, concrete architecture proposal and experiments
  where a claim depends on their evidence. Proposals and experimental APIs
  are not automatically governing decisions.
- Include relevant unchanged main-spec obligations and companion requirements.

Examine these questions

1. Whole-run structure and dynamic execution

Does the accepted representation preserve:
- repeated and independently progressing activities;
- hierarchical regions, lifecycle boundaries and cross-owner dependencies;
- work that produces state/effects without an optimization offer;
- schedules, feedback, branching and bounded runtime choices;
- permitted transitions affecting later execution?

Verify that static selection leaves the hot path without freezing accepted
runtime behavior or requiring callbacks into the authoring interface.

Check that operation-local wiring is not presented as proof of run-level
coordination. Internal mathematics and worker schedules may remain private,
but externally relevant dependencies, effects and failure boundaries cannot
disappear inside an opaque operation.

2. Execution equivalence and preparation

Check what prepared/lowered execution must preserve when work is fused,
partitioned, chunked, compiled or scheduled differently.

Follow input associations/admission, numerical policy, gradient meaning,
observation order, adaptive decisions, effects, authority, current-use
dependencies, partial outcomes and required coordination opportunities.

Equal final losses or weights must not alone establish equivalence.
Conversely, preserving those interactions must not require every internal
call to remain a separately dispatched node.

Verify that generating Trainer-owned mechanics into an executable does not
silently transfer their authority to selected computation.

3. Differentiated work and numerical lifetime

Follow derivative demands, seeds where applicable, intentional cuts,
subject/input-path routing, contribution destinations, retained numerical
state, supported recomputation, completion and release.

Check:
- gradients through frozen participants;
- one participant used with different views and gradient requirements;
- several derivatives relying on earlier state;
- derivative work crossing owner or region boundaries;
- cancellation/failure while retained state or device work remains in use.

A ready contribution must not by itself authorize conflicting mutation.
A revision identifier must not be treated as retained numerical contents.

Do not require a project-owned autodiff engine, public tensor-operation graph
or continuation object for every call. Determine whether the necessary
relationships are specified independently of their eventual representation.

4. Optimization offers, windows and multiple units

Distinguish computation output, accounting value, optimization offer,
completed contribution, usable gradient, eligible advancement and backend
outcome.

Verify accepted policies can express:
- one unit;
- independently due units and their independent windows;
- jointly due disjoint units;
- shared backward work where valid;
- distinct sources requiring controlled gradient routing.

Follow window completion, synchronization, scaling/unscaling, clipping scope
and timing, advancement ordering, scheduler triggers, zeroing/discard rules
and progress.

Check that work on one unit cannot silently clear another unit's pending
contributions. Shared computation must not silently introduce gradients into
unselected destinations.

Separate semantic unit identity from optimizer handles and tuple position.
Check alias consolidation versus competing ownership.

Backend limitations must reject unsupported readiness rather than silently
change the accepted policy. Do not infer support for arbitrary accumulation,
routing or distributed arrangements merely because multiple units exist.

5. Explicitly granted alternative authority

Verify that supported research regions can genuinely own the actions granted
to them, while every retained action still has an identified owner.

Examine scope by region, phase, affected state and transferred mechanics.
Check coexistence of standard work and granted work without dual backward,
advancement or zeroing ownership.

Use the bounded two-pass case to follow:
- entry requirements and pending-window treatment;
- temporary parameter edits;
- intermediate backward and gradient handling;
- restoration and produced-gradient handback;
- retained Trainer clipping, advancement and cleanup.

Do not turn that one profile's quiescent-window restriction into a universal
restriction on all extensions.

Distinguish declared/trusted Python behavior, boundary-observable enforcement,
and guarantees the framework cannot prove. Scoped interfaces do not establish
a general sandbox against hidden mutations through live objects.

6. Failure and completion

Trace failures before invocation, during selected work, after return,
during backward or optimizer work, at handback and during cleanup.

Verify that:
- pre-invocation rejection does not claim that selected work ran;
- post-return output-validation failure retains possible effects;
- missing success does not prove unchanged state;
- returned, skipped, uncertain and unattempted mechanics remain distinguishable;
- scheduler/zeroing failure does not erase an earlier optimizer outcome;
- restoring a temporary edit does not erase other effects of a failed region;
- primary and cleanup failures remain available together;
- dependent unsafe work stays unavailable until accepted recovery;
- unaffected work may continue only under its dependencies and failure policy;
- authorized recovery is not prevented by withholding ordinary use.

A returned optimizer call is not proof of numerical parameter change.
Jointly due units are not automatically an atomic update.
Do not invent rollback, exactly-once effects or safe replay.

7. Actual backend completion and handback

Check whether protection lasts through required backend reads, transfers,
backward/recomputation, outstanding device work and retained mechanics.

A Python return, clean visible tensor or successful cleanup callback must not
alone certify synchronization, restored master/sharded/offloaded state,
usable gradients or completed device work.

Verify readiness and handback demand the applicable evidence and can reject
unsupported combinations. Concrete locking, completion tokens and backend
adapters remain implementation choices unless their necessary meaning is
actually missing.

Concrete cases

Use recorded cases to reason through:
- ordinary maintained training with adaptive feedback;
- a frozen participant transmitting gradients to a trainable subject;
- one participant used with and without an adapter;
- jointly due Muon/AdamW-style units versus alternating G/D actions;
- independent/nonconsecutive contribution windows;
- several independently produced inputs feeding one consumer;
- a bounded two-pass region beside standard work;
- a stage change with pending contributions and retained derivative state.

Include failure variants:
- invalid output after a possible owned-state update;
- one optimizer returns and another fails or has an uncertain outcome;
- scheduler or zeroing fails after advancement;
- second-pass failure, failed restoration or uncertain handback;
- cancellation/cleanup failure while state is still required.

These are semantic source checks, not a requirement to build new experiments.

Evidence and completion claims

For relevant experiments, distinguish:
- assertions actually exercised;
- fixtures inspected but not rerun;
- behavior simulated by trusted callbacks or flags;
- architecture established through source-backed case analysis;
- production/backend capabilities not demonstrated.

Examine G3 completion claims against their actual acceptance criteria.
Missing production integration does not invalidate an architectural derivation
task by itself. Narrow experiments also cannot certify broader guarantees.

Check the conformance and hot-path criteria where they bear on execution:
matched semantics, retained dynamic/safety checks, due-work scaling,
bounded retained state and explicitly scoped measurements.
Do not promote migration-specific numerical budgets into permanent execution
semantics or infer full-run performance from dispatch-only measurements.

Pass 0 leads

Investigate P0-INV-04, P0-INV-08 and P0-INV-09 within this concern.
Use Pass 2's dispositions as context, but independently check execution-time
correlation, protection, completion and handback.

The leads are not confirmed findings or the complete checklist.

Scope boundaries

Follow preparation/revision requirements where needed, without repeating
Pass 2 wholesale.

Route detailed producer lifecycle, backpressure and capability coordination
to Pass 4; coherent capture and recovery mechanisms to Pass 5; durable schema
and reporting integration to Pass 6; integrated case synthesis to Pass 7.

Do not replace a missing relationship with an unstated universal sequence,
result object, event log, global clock or new architecture.

Return

- a concise source-backed map of execution/optimization owners and handoffs;
- findings with P3-prefixed IDs under the general reporting protocol;
- concrete success and partial-failure traces;
- an evidence map separating requirements, case analysis, experimental proof
  and explicitly untested guarantees;
- dispositions of the named leads;
- verified areas, coverage limitations and questions routed to later passes.

Do not edit files or tasks, prescribe concrete G5 interfaces, or issue an
overall G5-readiness verdict.
