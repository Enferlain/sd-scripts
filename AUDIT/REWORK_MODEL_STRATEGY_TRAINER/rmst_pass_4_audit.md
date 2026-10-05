Pass 4: independent progress and capabilities.

Apply the shared general audit protocol. This is a read-only concern audit,
not an implementation task or another architectural survey.

Audit commit:
6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48

Confirm that the examined sources correspond to that commit. Identify any
additional working-tree content separately; do not silently combine revisions.

Previous reports are navigation, not authority. P1-REV-01 and P2-REV-01 have
local corrections outside this baseline. Reference them where relevant rather
than assuming those corrections are present.

Objective

Determine whether independently progressing input/cache production, training,
validation, sampling and other selected activities can be coordinated under
the accepted whole-run structure without a universal training-step sequence.

Follow ordering, admission, protected access, capacity/backpressure,
cancellation, adaptive feedback and separate continuation owners.

Independent progress does not automatically require simultaneous execution
of conflicting work. Check the accepted consistency and progress requirements,
rather than assuming either universal concurrency or universal serialization.

Sources

- Read the governing source-authority rules and applicable design decisions,
  especially D5, G4.1, G4.2, composed run state, G3.6 and D10.
- Read complete relevant requirements/scenarios in
  training-capability-coordination and accepted-training-execution.
- Follow dependencies through training-contract, run-participant-state,
  runtime preparation, optimization, persistence and observability.
- Include applicable companion and unchanged main-spec obligations.
- Consult the research index, relevant data/streaming, conditioning, distributed,
  teacher/student and asynchronous rollout cases, and adopted future-idea
  discussions where the design relies on them.
- Examine experiments and current production only for relevant evidence,
  migration assumptions or feasibility claims.

Research cases establish support or architectural-neutrality pressure according
to their recorded disposition. Do not assume every researched backend or
algorithm must ship in the first migration.

Examine these questions

1. Capability selection and runtime coordination

Distinguish repository availability, explicit strategy provision, accepted
configuration/policy requests, trigger eligibility and executable readiness.

Verify that:
- availability alone does not select or schedule a capability;
- known incompatibility is rejected at its earliest authoritative point;
- bounded requests supplied later are checked before use;
- coordination consumes existing obligations rather than inventing another
  acceptance policy;
- selected domain algorithms remain outside family branches in Trainer;
- runtime requests and stateful behavior survive completion of authoring.

Do not infer support from method presence, family names or raising defaults.

2. Independent activities and ordering

Follow startup, due selection, waiting, progress, handoff, shutdown and failure
relationships across independently progressing work.

Check that:
- production can progress before or without a due training action;
- activities can have distinct clocks and completion boundaries;
- required dependencies and observation order are preserved;
- a waiting activity does not implicitly establish a universal global barrier;
- capability work is not forced through an optimizer offer;
- private workers and internal scheduling need not become individual IR nodes.

Where accepted policies require eligibility, fairness or progress guarantees,
verify their meaning. Do not invent a universal fairness policy or demand a
particular scheduler implementation.

The prototype's first-waiting-item behavior or manually published producer
events must not become architectural constraints.

3. Cache production versus live input consumption

Identify the owners of:
- cache-production traversal, workers, storage and publication;
- live-source/cache selection, mixture, packing and exposure;
- consumer admission and delivery;
- each owner's continuation state.

Outer coordination must not silently take over provider selection/packing
or equate a cache index with the training input position.

Check streaming, partial readiness and bounded scopes without assuming a
complete finite manifest, epoch clock or dataset-completion barrier.

Cache arrival order must not silently determine the training distribution.
Miss, capacity, fallback and ready-subset selection follow accepted policies.

4. Work identity, provenance and admission

Distinguish sample/example identity, requested work, caption/transformation
variant, produced value, storage location and consumer association.

Follow out-of-order completion, joins of independent inputs, partial handoffs,
shared stored values and retries.

Verify that readiness or physical readability does not imply admission.
Training, validation and sampling may have different consumer obligations;
one consumer's admission must not authorize another's use.

Check actual producer-state provenance, not merely launch/request/completion
labels. Provenance may describe a protected state or accepted structured
associations across states/segments; attribution alone does not authorize
mixed-state production.

Ordinary encoder updates can invalidate reuse without changing participant
identity or binding revision. Conversely, unrelated downstream updates need
not invalidate an evidenced upstream cache.

Detached cached values must not replace required live derivative paths.
Caching must not silently freeze stochastic transformations or substitute
approximate computation for accepted exact behavior.

5. Publication, storage lifetime and backpressure

Follow payload production, indexing/locator publication, readiness and
consumer use.

Check:
- complete publication units and accepted independently usable chunks;
- incomplete bundle/job coverage;
- payload-written but index/publication-failed outcomes;
- shared storage with distinct logical handoffs;
- eviction, compaction and relocation during outstanding reads;
- bounded pending/ready work, resource use and backpressure;
- supported distributed data ownership.

A file, successful encoder call or queue position is not sufficient readiness.
A whole-dataset transaction or crash-safe storage guarantee must not be
invented.

Required source/storage access must remain valid through actual dependent use.
Bounds and storage pressure must not silently change accepted input selection.

6. Validation and sampling

Verify the separation between pipeline coordination and selected evaluation
or generation semantics.

For validation, follow:
- requested coverage/stopping rules, including bounded streams;
- provider continuation separate from training, unless sharing is accepted;
- weighting, masks, denominators and distributed reduction;
- empty, partial and missing contributions;
- actual evaluated source-state provenance;
- derivative needs and declared adaptive effects.

Aggregation must execute selected measurement meaning, not guess an unweighted
mean of batch means. Validation is not universally no_grad or state-free, but
does not receive implicit optimizer or participant-transition authority.

For sampling, follow:
- selected generator/objective compatibility;
- request/output associations and random-input policy;
- generation, publication and reporting as separate outcomes;
- partial multi-request failure and policy-governed retry;
- non-image and associated outputs without image-shaped universal fields.

Known output publication must not be erased by a later reporting failure,
and reporting retry must not silently regenerate outputs.

7. Shared access and temporary projections

Follow training updates, retained derivative work, cache reads, evaluation,
generation and temporary research edits when their source/storage uses overlap.

Check accepted wait, rejection, synchronization or isolation policies.
Different participant labels alone are not proof of physical independence.

Protect consistency through actual backend use/completion, not only Python
return or trigger-time freshness. Trigger coordinates must not substitute for
the states actually evaluated or used.

Check temporary modes, placement, dtype and relevant RNG:
- partial setup failure;
- execution failure and cancellation;
- restoration of prior mixed projections;
- unsupported concurrent RNG/mutable-state overlap;
- cleanup failure and uncertain handback.

Neither unwrapping/moving a live object nor releasing exclusion establishes
readiness. A clean visible tensor does not prove all backend state restored.

8. Adaptive feedback and continuation owners

Separate required algorithmic feedback from optional logging.

Follow observations into subsequent adaptive decisions, including validation-
driven reactions. Preserve required delivery/order, state ownership and effects
without creating generic Trainer knowledge of the algorithm.

Measurement completion, adaptive reaction and failed reaction can have separate
outcomes. Observation retry must not repeat adaptation or computation.

Identify the continuation contributions needed from producer, input provider,
capability coordinator and selected algorithm. Produced, ready, handed-off,
consumed and update-contributing positions remain distinct where required.

A paused producer, saved cursor or cache index alone must not claim a coherent
runtime snapshot. Detailed capture and restoration mechanisms belong to Pass 5.

Concrete cases

Use recorded cases, including:
- asynchronous encoding with changing captions and encoder state;
- frozen upstream cache followed by a trainable conditioning adapter;
- streaming/sharded production consumed before total completion;
- two independently produced inputs joined by one consumer;
- several consumers sharing storage under different admission requirements;
- compaction/eviction overlapping an admitted read;
- bounded-stream validation with independent provider continuation;
- validation feedback changing later accepted behavior;
- generation overlapping training or a temporary two-pass perturbation;
- partial generation/publication followed by cleanup or reporting failure;
- evolving producer/policy state with late results.

Include cancellation, capacity saturation, missing inputs and uncertain
source/storage state where relevant. Do not require new experiments merely
to perform this audit.

Evidence and neighboring passes

Investigate P0-INV-04 and P0-INV-08 within this concern, and P0-INV-09 for
capability completion/evidence claims.

Distinguish source-backed semantics, recorded case analysis, inspected
assertions, freshly executed tests and untested production guarantees.

Bounded producer simulations and one-feed actions do not prove autonomous
scheduling, general joins, cancellation races or real backend isolation.
Conversely, those missing implementations are not semantic defects when their
necessary meaning and conformance obligations are already specified.

Avoid repeating Passes 2–3 wholesale. Route detailed artifact completion and
coherent recovery to Pass 5; durable schema/reporting integration to Pass 6;
combined whole-run sufficiency to Pass 7.

Return

- a concise source-backed map of activity, admission, access and continuation
  owners;
- findings with P4-prefixed IDs under the general reporting protocol;
- concrete overlapping-work and failure traces;
- dispositions of the named leads;
- an evidence map, verified areas and coverage limitations;
- unresolved questions routed to later passes.

Do not edit files or tasks, introduce a universal scheduler/result/state
container, prescribe concrete G5 interfaces, or issue an overall G5-readiness
verdict.
