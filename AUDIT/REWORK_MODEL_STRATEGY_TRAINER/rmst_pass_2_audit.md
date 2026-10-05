Pass 2: revisions, lifecycle and preparation.

Apply the shared general audit protocol. This is a read-only concern audit,
not another orientation pass or an implementation task.

Audit commit:
6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48

Confirm that the sources examined correspond to that commit. Report any
additional working-tree content separately; do not silently combine revisions.

Use the Pass 0 report as navigation, not authority. The Pass 1 report may help
locate ownership distinctions, but verify the actual sources independently.
P1-REV-01 has been corrected locally outside this audit baseline. If encountered,
reference the existing finding rather than treating that correction as part
of the pinned sources.

Objective

Determine whether the recorded system consistently distinguishes identity,
accepted meaning, changing numerical state, prepared usability, and historical
observations. Follow what changes, what remains the same, what becomes stale,
and what may become current across all relevant specifications.

These distinctions need coherent semantics, not necessarily separate classes,
universal counters, or explicit revision tuples on every runtime object.

Sources

- Read the governing design's source-authority rules and relevant decisions,
  especially D5–D9, including composed run state and the G3.4/G3.5 findings.
- Read complete requirements and scenarios in run-participant-state,
  training-runtime-preparation and training-optimization.
- Follow relevant dependencies through training-contract,
  accepted-training-execution, capability coordination, artifact persistence,
  and metadata/observability.
- Read applicable companion requirements for model catalog identity,
  source provenance, structure metadata and shared targeting.
- Include unchanged main-spec obligations wherever the deltas rely on them.
- Consult inherited direction, explicit historical corrections and experiments
  where a claim depends on their evidence.

Examine these questions

1. Identity and revision meanings

Distinguish:
- authored addresses and role labels;
- logical run authority and participant/relationship incarnations;
- binding, route/view, arrangement and obligation revisions;
- ordinary numerical, optimizer, schedule and adaptive-state evolution;
- prepared generations and their dependency evidence;
- optimization-unit identity, accepted-definition revision and physical handles;
- realization-composition revisions and checkpoint-scoped finalization;
- catalog/model revisions and source-selection identity;
- execution sessions, preparation attempts and work identities.

For each material distinction, establish its scope, who establishes it,
what changes it, and which consumers rely on it.

Check for accidental equivalence between these meanings. Conversely, do not
require distinct storage mechanisms merely because the semantics differ.

2. Lifecycle and continuity

Follow materialization, wrapping, casting, compilation, compatible replacement,
incompatible replacement, retirement, relationship changes and stage changes.

Verify that:
- identity-preserving changes preserve the correct incarnation;
- retirement ends current use without erasing historical meaning;
- new participants and new runs use explicit identity and lineage;
- freezing, relationship deactivation and retirement remain different actions;
- routes, access views and individual invocations do not become competing
  participant identities;
- optimization-unit continuity is separate from optimizer-state continuity.

Check unit membership changes, physical parameter surgery, split/merge,
recreated handles and ordinary policy evolution against these distinctions.

3. Invalidation and continued use

Follow dependency changes across target projections, prepared routes/views,
backend groups, optimization runtimes, pending gradient/derivative work,
selected-operation state, input products and capability work.

Verify that ordinary accepted numerical evolution does not automatically become
a semantic amendment or require preparation rebuilding. It may nevertheless
make produced data inadmissible under its accepted dependency policy.

Check both conservative snapshot freshness and justified narrower dependencies.
Do not assume precise minimal invalidation is universally required.

Distinguish checking freshness when work starts from protecting the state
actually needed until execution, derivative work or handback completes.
A valid starting revision must not alone permit a conflicting later mutation.

4. Preparation and backend correspondence

Verify that preparation membership is broader than optimization membership,
including frozen execution participants and independently managed state.

Check that one coherent attempt-scoped job:
- may require multiple ordered backend calls;
- preserves participant and optimization-unit identity despite composites;
- separates authority-owned routes/views from optimization/backend state;
- retains correspondence between accepted subjects and final parameters;
- supports the accepted construction-order flexibility;
- includes the closure required by inseparable or overlapping backend groups.

Distinguish semantic acceptance from executable readiness. Unsupported backend
requirements may fail readiness; acceptance must not imply universal support.

Do not treat lightweight experimental handles or cleanup flags as proof of
distributed publication, retained-state protection or real backend handback.

5. Publication and failure

Trace candidate creation, evidence collection, obligation evaluation,
freshness validation and final installation across the relevant owners.

Verify that:
- stale, incomplete or superseded candidates cannot become current;
- applicable obligation changes matter even if physical bindings did not change;
- expected-fallible work precedes final publication;
- partial owner state cannot be advertised as a complete prepared result;
- reporting failure after publication does not become rollback authority.

Keep replacement-only preparation distinct from destructive in-place work.
For the latter, examine withdrawal of guarantees, attempt identity, completion
basis, failure and late optimistic results.

Check process/rank failure at publication without inventing rollback,
transactional optimizer updates or automatic recovery.

6. Composition, catalog and checkpoint boundaries

Follow authority publication into immutable composition observations.

Verify that loader success is not accepted composition, and that delayed
filing, retries or duplicate observations cannot redefine accepted ordering
or make an older state appear current.

Check checkpoint-scoped finalization:
- what exactly is finalized;
- how consumers select the corresponding checkpoint;
- how later transitions coexist with earlier final observations;
- how products retain the composition actually captured.

Catalog/source identity must not substitute for live participant identity,
numerical-state provenance or prepared readiness.

Follow same-run restoration far enough to verify identity and revision
continuity alongside a new execution session and freshly established physical
readiness. Missing required facts must not silently initialize another run.
Route detailed capture/recovery mechanisms to Pass 5.

Concrete cases

Use recorded cases to trace the relevant transitions, including:
- a frozen participant requiring preparation;
- one participant with several execution uses or prepared representations;
- compatible replacement while a preparation candidate is in flight;
- parameter or membership changes with pending accumulation contributions;
- a preparation job overlapping an inseparable backend group;
- a stale candidate after an obligation or relationship amendment;
- replacement-only failure versus destructive preparation failure;
- an encoder's numerical state changing while produced inputs remain in flight;
- an older composition observation arriving after a newer publication;
- a finalized checkpoint followed by another transition and later artifact capture;
- same-run restoration in a new session with old handles or attempts still recorded.

These are source-backed semantic traces, not a demand for new experiments.
Where producer work spans state changes, check whether its actual provenance
can express the accepted dependency meaning without assuming one universal
state/version field.

Pass 0 leads

Investigate P0-INV-05 and P0-INV-08 explicitly.
Follow P0-INV-01 where restoration identity intersects this pass.
Use P0-INV-03 and its Pass 1 disposition as context for distinguishing ordinary
owned-state evolution from coordinated transitions; reopen it only if concrete
conflicting evidence appears.

The leads are neither confirmed findings nor an exhaustive checklist.

Boundaries and evidence

Distinguish established architectural direction from:
- unresolved necessary semantics;
- contradictory requirements;
- unsupported guarantees or completion claims;
- wording with a concrete interpretation risk;
- representation and implementation choices deliberately deferred to G5.

Do not demand production backend integration to validate a task whose actual
acceptance criteria were architectural derivation or a bounded experiment.
Equally, do not use an action-local prototype as proof of broader run-level
coordination or backend safety.

Inspect adjacent execution, recovery and metadata requirements where needed,
but route detailed scheduling to Pass 3, independent activity coordination to
Pass 4, snapshot/recovery mechanisms to Pass 5, and schema/reporting integration
to Pass 6.

Return

- a concise source-backed identity/revision and dependency map;
- findings with P2-prefixed IDs using the general reporting protocol;
- concrete transition traces showing continuity, invalidation and publication;
- dispositions of the named Pass 0 leads;
- verified areas, coverage limitations and questions routed to later passes.

Do not apply corrections, change tasks, prescribe concrete G5 types, repeat
the entire Pass 0 map, or issue an overall G5-readiness verdict.
