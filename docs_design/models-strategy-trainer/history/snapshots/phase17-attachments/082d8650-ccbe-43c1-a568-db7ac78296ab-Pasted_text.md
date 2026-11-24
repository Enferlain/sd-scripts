Retain the graph-derived execution approach, but revise producer ownership and lifecycle handling before treating the current shapes as a production foundation. The candidate genuinely derives work and coordination from accepted relationships; it does not merely decorate a fixed training loop. Its strongest evidence is the connected lowering, preserved dynamic behavior, and separate optimization outcomes. Its weakest evidence is continuity across long-lived activities and changes to their state.

I reviewed revision [`5218fdb`](https://github.com/Enferlain/sd-scripts/tree/5218fdb1546f7c485647f4f1ba86470f7c88a867), using the governing design and delta specs as authority. Historical proposals supplied pressure cases, not additional requirements. All **51 candidate tests and 12 existing execution tests passed** in the review environment. Additional small probes reproduced defects outside that coverage. No code, requirements, task checkboxes, or issue status were changed.

**Demonstrated strengths worth retaining**

The diffusion arrangement provides a complete, credible trace from accepted meaning to execution:

| Stage | What the accepted relationships produce |
|---|---|
| Acceptance | Participant and unit identities, checked local value dependencies, declared owner state, producer/channel ownership, bounded block alternatives, and completion/request connections. |
| Preparation | Concrete participant verification, physical-member and alias checks, optimizer membership, ordered block bodies, producer ports, request queues, and scope callables. |
| Publication | One installed image with checked generation correspondence. Loading or constructing an optimizer alone does not establish readiness. |
| Independent production | Pixels and conditioning progress separately. Caption workers acquire encoder access and retain the actual source stamp used to produce each detached value. |
| Training admission | The selected policy chooses a work key. Prefix readiness waits without participant leases; protected admission then checks every product before claiming the pair. |
| Training work | Relationships produce `join → objective → feedback → differentiate → advance`, despite the deliberately reversed declaration order. Feedback changes the adaptive state used by later computation. |
| Capability and run completion | Learning completion feeds the requested validation activity. Validation acquires protected current state. Hierarchical task groups determine shared completion and failure propagation. |

The declaration is in [examples.py:168–223](https://github.com/Enferlain/sd-scripts/blob/5218fdb1546f7c485647f4f1ba86470f7c88a867/docs_design/models-strategy-trainer/execution_candidate/examples.py#L168-L223); the derived ordering, leases and call chain are in [prepare.py:332–384](https://github.com/Enferlain/sd-scripts/blob/5218fdb1546f7c485647f4f1ba86470f7c88a867/docs_design/models-strategy-trainer/execution_candidate/prepare.py#L332-L384).

That machinery is genuinely derived. Changing unit membership changes parameter resolution and protected access. Changing value or `after` relationships changes the generated call chain. Changing hierarchy or request connections changes activity machinery without changing `Engine.run`.

The prepared form is also an attachment to the accepted graph: executable closures, queues and generated source are produced from it rather than independently authored. The diagnostic source strings are not another authoritative program. This conclusion is bounded to the supported, initially accepted structure; general topology amendment is not demonstrated.

Several other choices are strong:

- **Authoring completion preserves runtime behavior.** Adaptive computation, bounded policy choices, suspending implementations and completion-dependent requests remain live. Ordinary execution does not revisit the strategy-authoring interface.
- **Identity, binding and numerical progress are separated.** Compatible replacement preserves participant and unit identity while revising bindings and rebuilding affected optimizer runtime. Ordinary advancement does not manufacture a topology revision.
- **Input claim is distinct from optimization completion.** All-member admission precedes claim, and claim is not presented as a multi-optimizer transaction.
- **Failure knowledge survives partial advancement.** A returned first optimizer and an uncertain second optimizer remain distinguishable; protection release does not automatically restore usability.
- **The granted two-pass region has a meaningful ownership split.** Handback must restore the protected view and supply verified gradients; clipping, final advancement and final reset remain retained mechanics.

These are useful architectural results, not universal proof from tiny models.

**Actual defects or contradictions**

**1. Replacement can publish new owner state while a producer continues using the old object.**

[ProviderContext:169–178](https://github.com/Enferlain/sd-scripts/blob/5218fdb1546f7c485647f4f1ba86470f7c88a867/docs_design/models-strategy-trainer/execution_candidate/prepare.py#L169-L178) captures its state objects during lowering. [RunState.replace:450–478](https://github.com/Enferlain/sd-scripts/blob/5218fdb1546f7c485647f4f1ba86470f7c88a867/docs_design/models-strategy-trainer/execution_candidate/state.py#L450-L478) can subsequently reset those owners and replace `self.owners`. Provider continuation is not represented in `state_locks`.

A concrete accepted trace reproduced this:

1. A producer retains its declared owner list and suspends outside a participant lease.
2. Another activity replaces that owner’s dependent participant, with `on_replace="reset"`.
3. Publication installs generation 2 and a new empty owner list.
4. The producer resumes and appends to its original list.

The old list became `["continued"]`; the current authoritative owner remained `[]`.

This is a coherent-publication defect, not merely missing topology support. Execution and capture can now describe different continuation state.

The smallest safe correction is to **reject or defer a reset affecting an active producer until a governed owner boundary is established**. A fuller realization could quiesce and migrate/rebind the continuation. Merely changing the context to perform fresh dictionary lookups is insufficient: selected behavior may already hold references across suspension.

**2. Declared producer writes do not advance numerical provenance.**

`Producer.uses` accepts write access, but `ProviderContext.use_source()` only acquires access. Epoch advancement exists for returned `Call` writes and retained optimizer advancement, not producer writes: compare [prepare.py:176–178](https://github.com/Enferlain/sd-scripts/blob/5218fdb1546f7c485647f4f1ba86470f7c88a867/docs_design/models-strategy-trainer/execution_candidate/prepare.py#L176-L178) with [prepare.py:439–447](https://github.com/Enferlain/sd-scripts/blob/5218fdb1546f7c485647f4f1ba86470f7c88a867/docs_design/models-strategy-trainer/execution_candidate/prepare.py#L439-L447).

The reproduced trace used an honestly declared write:

- The producer computed a value from weight `1` and retained its source stamp.
- Under its write lease, it changed the weight to `2`.
- After release, it published the old value.
- Exact-stamp admission accepted it because the current numerical epoch was still `0`.

This undermines the admission guarantee without requiring hidden mutation or malicious Python.

For this target, the smallest correction is to reject producer writes until their effect/completion protocol exists. The broader correction is a consistent mechanism for reporting declared effects across owners. That audit should include granted work affecting participants beyond the temporarily restored unit.

**3. Cleanup can become the reported cause and obscure reached work.**

The evaluation wrapper restores modes in a plain `finally` block: [prepare.py:391–403](https://github.com/Enferlain/sd-scripts/blob/5218fdb1546f7c485647f4f1ba86470f7c88a867/docs_design/models-strategy-trainer/execution_candidate/prepare.py#L391-L403).

In a probe where selected work raised “primary failure” and mode restoration raised “cleanup failure”, `WorkFailed.cause` reported the cleanup failure. Python retained the primary failure in the implicit exception context, so it was not completely erased. Nevertheless, the structured result did not distinguish primary failure, cleanup failure and restoration uncertainty. Similarly, a returned value followed by cleanup failure never reaches the outer “returned” record.

That falls short of [accepted-training-execution:344–347](https://github.com/Enferlain/sd-scripts/blob/5218fdb1546f7c485647f4f1ba86470f7c88a867/openspec/changes/rework-model-strategy-trainer/specs/accepted-training-execution/spec.md#L344-L347).

The correction is a scoped outcome that retains execution results/failure separately from cleanup outcomes, attempts the remaining cleanup where appropriate, and withholds uncertain state.

**4. Some completely knowable target incompatibilities are rejected after materialization.**

`prepare()` loads models and constructs the candidate before creating `Lowering`; target wait-cycle checks and the prohibition on replacement within protected numerical/state work occur afterward: [prepare.py:242–266](https://github.com/Enferlain/sd-scripts/blob/5218fdb1546f7c485647f4f1ba86470f7c88a867/docs_design/models-strategy-trainer/execution_candidate/prepare.py#L242-L266).

A graph accepted by `Contract.accept()` invoked its loader before preparation rejected a potential lease/source cycle. No concrete model evidence was needed: this target’s whole-block lease policy was already fixed.

This conflicts with the earliest-authoritative-evidence rule in [training-contract:122–137](https://github.com/Enferlain/sd-scripts/blob/5218fdb1546f7c485647f4f1ba86470f7c88a867/openspec/changes/rework-model-strategy-trainer/specs/training-contract/spec.md#L122-L137). Shared target-conformance checks should run before loading, ideally during acceptance when the necessary evidence is already authoritative. Concrete storage, member and backend checks should remain later.

**Candidate tradeoffs and plausible alternatives**

The current vocabulary is a coherent small seed, but its smallness partly comes from omitting relationships that the governing design requires.

| Construct | Assessment |
|---|---|
| `Block`, `Together`, `Repeat`, `Requested`, `Producer` | Useful lifetime distinctions. Restricting repeat alternatives and capability bodies to blocks is an experimental restriction, not a general region model. |
| Value links and `after` | Correctly distinguish data dependence from required ordering. Stable declaration order resolves otherwise unordered work; it should not accidentally carry an undeclared semantic dependency. |
| `Join`, `Channel`, `Product` | Good separation of availability, admission, provenance and claim. One shared key is adequate for these examples, but richer member correlation must retain each member’s identity. |
| `Use` | Necessary, but participant plus read/write cannot describe independently prepared routes/views, derivative cuts or numerical last use. |
| `OwnedState` | Makes state visible, but initialization, dependency and continuity declarations are too narrow for broader owner preparation. |
| `Differentiate`, `Advance`, `TwoPass` | Legitimate supported mechanics. Their scalar-source and local-window assumptions must remain profile choices rather than universal training semantics. |

The design’s boundary rule is useful here: expose relationships needed by another owner, preparation, coordination or recovery; leave private mathematics and worker topology inside selected behavior. See [design:2922–2963](https://github.com/Enferlain/sd-scripts/blob/5218fdb1546f7c485647f4f1ba86470f7c88a867/openspec/changes/rework-model-strategy-trainer/design.md#L2922-L2963).

**Whole-run coordination is credible within its narrow target, but lifecycle is incomplete.**

The prefix-readiness correction is sound. It avoids this cycle:

> Consumer holds a model write lease → waits for input → producer needs model read access to produce that input.

Later joins can still wait under whole-block protection. Cold analysis rejects potential direct and indirect cycles. That is a reasonable conservative target choice, explicitly acknowledged in [prepare.py:286–330](https://github.com/Enferlain/sd-scripts/blob/5218fdb1546f7c485647f4f1ba86470f7c88a867/docs_design/models-strategy-trainer/execution_candidate/prepare.py#L286-L330).

It is not evidence that every rejected arrangement is semantically impossible. Alternatives include finer access intervals, safe reservations, isolated views, or moving a join before unrelated work when effect evidence permits it. Shorter leases must still protect saved numerical state through derivative last use and actual completion.

End-of-demand exposes a different missing relationship. With channel capacity one:

1. A consumer requests and consumes item 1, then finishes.
2. The producer publishes item 2.
3. Publishing item 3 blocks on capacity.
4. `Together` waits for the producer, so the run never completes normally.

The probe timed out with one completed consumer invocation and item 2 still ready. This is **documented unimplemented behavior**, not an undisclosed defect: [README:238–243](https://github.com/Enferlain/sd-scripts/blob/5218fdb1546f7c485647f4f1ba86470f7c88a867/docs_design/models-strategy-trainer/execution_candidate/README.md#L238-L243). It nevertheless shows why demand closure, stop/drain policy and outstanding-work disposition belong at the accepted owner boundary.

The full-feed rejection policy is also defensible for this target. It preserves queued products and fails rather than indefinitely waiting for a demanded key that cannot currently enter. Demand-aware submission or reserved capacity could execute some rejected arrangements. Queue fullness alone does not establish global deadlock.

Finally, channel capacity bounds **ready products**, not all producer-private workers, completed unpublished values or retained resources. Maintained providers must supply credible total-resource bounds. The earlier [producer pressure test](https://github.com/Enferlain/sd-scripts/blob/5218fdb1546f7c485647f4f1ba86470f7c88a867/tests/unit/training/test_input_producer_experiment.py#L29-L76) explicitly counted pending, produced and ready stages; that is useful evidence, not a mandate for its implementation.

**Dynamic behavior survives, but ownership declarations are partly trusted.**

Bounds on selected alternatives, declared publication ports, contribution correspondence and two-pass phase order are checked. Mapping projections limit which objects are supplied through the public seam.

Those restrictions do not enforce read-only tensor access or prohibit independent optimizer construction. Live modules remain mutable; granted access internally retains a unit runtime containing the optimizer. Python privacy and mapping proxies are not authority isolation.

That agrees with the governing design’s trusted-implementation boundary. The useful correction is to distinguish:

- mechanically checked bounds and correspondence;
- maintained implementation conformance;
- backend evidence needed for protection and handback.

The two-pass profile should remain explicit. Its exact phase sequence is legitimate for the offered experiment, not a universal algorithm lifecycle. Stateful grants can receive explicit input values today, but they lack the direct declared owner-state/context seam available to `Call`. Stateful request policies likewise lack an owned-state declaration. Extending those seams is preferable to encouraging hidden closure state or restoring a broad strategy interface.

**The specialization path is credible, but not demonstrated.**

Generated direct calls establish that static graph interpretation is unnecessary on each invocation. Suitable regions could lower into compact synchronous execution, supplied compiled/native bodies, or residency/transfer schedules while preserving the accepted boundary.

Current couplings would obstruct that path if frozen:

- every work item becomes an awaited Python wrapper;
- tensor mechanics, leases, publication and owner bookkeeping are closely coupled through `RunState`;
- participant use resolves directly to one `nn.Module`;
- completion means cooperative coroutine completion plus synchronous CPU tensor completion;
- derivative work assumes immediate framework gradients and a local pending window;
- every `Call` enumerates its declared module trees for training-mode snapshots, even when `evaluation=False`.

The last point is avoidable implementation work, not a measured speed claim. Mode capture should occur only when needed; any cached membership must remain tied to valid binding/topology evidence.

Measure dispatch and projection allocation, module inspection, admission/lease contention, queueing, grant copies, retained numerical memory, preparation amortization and feedback/capability latency. Keep execution entrypoints, diagnostic retention, access scheduling and numerical adapters replaceable. Fewer calls alone cannot establish improvement.

**Understandability and maintainability**

The connected system is understandable: `graph.py` describes and checks meaning, `prepare.py` derives execution, and `state.py` supplies current bindings and mechanics. The examples are particularly useful because their relationships explain why the machinery exists.

The concern is responsibility concentration. `state.py` combines authority, optimizer realization, handoff coordination, access protection, continuity and capture. `prepare.py` combines control lowering, conformance-like target checks, capability setup/cleanup and failure classification. Adding work kinds requires coordinated changes to input/output helpers, acceptance, lease derivation, wait analysis and binding.

Simply splitting those files would move complexity. The useful correction is to establish one source of construct semantics and explicit owner exchanges, so each consumer derives its needs from the same meaning.

Ordinary authors should select participants, maintained computations, policies, capabilities, units and meaningful relationships. Maintained integrations should normally supply concrete verification, routine effects, route/view requirements, derivative behavior and preparation evidence. Internal machinery should derive correspondence keys, leases, queue connections and generated locals.

For example, an author could select a maintained conditioning computation and connect its output to an objective. The integration would supply its encoder-view, provenance and gradient obligations; preparation would choose the appropriate protected executable attachment. That preserves explicit composition without making every author reproduce compatibility validators.

**Deliberately unimplemented features—and where shapes need revision**

These distinctions matter more than a production-feature checklist:

| Pressure | What the present candidate can support | What needs additional representation |
|---|---|---|
| Repeated roles for one participant | Several calls can use the same participant with different mathematics. | Independently prepared views/routes and their validity cannot be identified through `Use(participant, access)` alone. |
| Topology and policy changes | Compatible binding replacement with stable participant/unit identity. | Relationship identity, obligation revisions, unit definition revisions, amendments and dependency invalidation. |
| Contribution windows | One local contribution and final advancement per unit per block. | Nonconsecutive or cross-block windows and their completion/continuity rules. Acceptance currently excludes them explicitly. |
| Shared differentiation | One loss can address several disjoint units. | Distinct routed derivatives sharing retained state, recomputation conditions and numerical last-use evidence. |
| Owner initialization | Constant or independently initialized objects. | Initialization/migration requiring governed current views or prospective replacement evidence. |
| Recovery | A finished, quiescent in-memory capture. | Contributor-complete cuts, unfinished-work disposition, RNG and backend continuation, restoration validation and execution-session identity. |

These are not reasons to abandon the hierarchy. Some require substantive revision of fields and relationships rather than merely implementing more methods. Hiding them in closures would not represent them faithfully.

The frozen-conduit pressure case is especially discriminating: the same frozen participant supplies different views, distinct losses route to different units, and one gradient can be ready while another derivative still requires old numerical state. The [concrete proposal’s fixture B](https://github.com/Enferlain/sd-scripts/blob/5218fdb1546f7c485647f4f1ba86470f7c88a867/openspec/changes/rework-model-strategy-trainer/research/trainer-architecture-concrete.md#L238-L248) supplies a useful oracle. It remains a proposed trace, not completed candidate evidence.

CPU/asyncio restrictions, SGD-only construction, lack of distributed preparation and lack of durable restoration are deliberate target omissions. They should not become architectural requirements.

**Genuine semantic decisions still to establish**

I did not find a conflict among the governing requirements. Most major questions are already settled: one authority, retained selected behavior, explicit ownership, actual provenance, protected numerical lifetimes and honest partial outcomes.

Two policy questions remain consequential when extending this candidate:

- **Demand completion:** what must happen to pending, unpublished and ready work when demand ends—retain, drain, explicitly discard or transfer to another consumer? The design requires an accepted lifecycle and truthful outcomes; it does not supply one universal policy.
- **Requested measurement consistency:** does a request require the state at its trigger, a later protected current state, or an explicitly permitted multi-state measurement? The design already forbids treating a trigger coordinate as provenance. The example records only trigger position and value, so it demonstrates protection more strongly than result attribution. Recording actual `Context.stamps` separately would improve that evidence.

These need selected policy meaning, not another universal clock or lifecycle.

**The three most valuable next steps**

1. **Run one connected producer-continuity and shutdown fixture.** Combine bounded production, declared source writes, dependent owner reset, early demand completion, cancellation and cleanup failure. Observe old/new owner identity, actual provenance, outstanding work, reached effects and usability after release. This would settle the minimum producer lifecycle/publication protocol and which unsupported transitions this target must reject.

2. **Execute the frozen-view, routed-derivative pressure case with a cross-block window.** Compare required old-state retention with an explicitly isolated-state variant; include a transition while contributions are pending. Verify routed gradients, independent unit progress, access release and partial outcomes. This would settle whether the current IR can be extended cleanly or needs restructuring around view, window and lifetime relationships.

3. **Compare a second lowering against the same accepted arrangement and observable trace.** Use a compact realization for a suitable region while retaining dynamic feedback, admission opportunities, capability coordination, ownership and failure milestones. Measure runtime overhead, memory and preparation cost under matched conditions. This would settle which current wrappers and bookkeeping are necessary boundaries, which are replaceable implementation choices, and whether specialization offers a worthwhile measured benefit.