Pass 6: metadata, observability and spec compatibility.

Apply the shared general audit protocol. This is a read-only concern audit,
not an implementation task or another architectural survey.

Audit commit:
6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48

Confirm that the examined sources correspond to that commit. Identify any
additional working-tree content separately; do not silently combine revisions.

Previous reports are navigation, not authority. P1-REV-01, P2-REV-01 and
P4-REV-01 have local corrections outside this baseline. Reference them where
relevant rather than assuming those corrections are present.

Objective

Determine whether metadata and observability faithfully represent the accepted
whole run without becoming competing runtime authorities.

Check origin truth versus recorded facts, delayed observations, required
durability, optional telemetry, resource evidence and compatibility projections.

Cross-compare both active changes and applicable unchanged main-spec
requirements. Do not restrict the audit to recently modified clauses.

Sources

Primary changes:
- openspec/changes/rework-model-strategy-trainer/
- openspec/changes/expand-model-metadata-identity-and-structure/

Read:
- source-authority rules and the reconciliation ledgers;
- D5/G4.5, G4.6 and relevant identity, execution and persistence decisions;
- complete model-family-metadata and training-observability deltas;
- companion catalog identity, source provenance, composition, structural
  metadata, target-reference and resource requirements;
- complete relevant main-spec requirements/scenarios, especially
  model-family-metadata, training-observability, resource-intelligence,
  resource-visibility, loaded-model-components, optimization-target-refs
  and affected adapter/method obligations;
- companion integration gates and tasks where the reconciliation depends
  on them.

Follow supporting notes, research, experiments and current production only
where needed to verify a claim, migration assumption or feasibility concern.

The recently built metadata system is useful groundwork, not an immutable
constraint. Its responsibilities may remain valid while its current schemas,
query behavior and runtime inputs require adaptation.

Examine these questions

1. Origin authority versus metadata authority

Identify who establishes:
- participant, relationship and accepted-transition truth;
- optimization, input, operation, capability and backend outcomes;
- catalog/source identity and provenance evidence;
- typed-item validity, durable qualification, relationships and projections;
- resource interpretation and observation delivery.

Check that “accepted typed fact” does not silently mean metadata has accepted
a live binding, validated a training algorithm or established runtime readiness.

Metadata may validate, qualify and retain supplied facts. It must not infer
runtime incarnations from names, roles, paths or Python objects, retain live
handles as canonical bindings, or reconstruct a competing arrangement.

Conversely, the runtime must not bypass central metadata ownership for durable
filing, invent catalog identities or duplicate registry/routing authority.

Distinguish a valid live projection from a filing that requires catalog
resolution under its accepted policy. One must not automatically require or
authorize the other.

2. Identity associations and historical meaning

Follow the required associations among:
- logical run and execution session/attempt;
- authored address and participant/relationship incarnation;
- optimization unit and its definition;
- catalog lineage/revision, serialized representation and source selection;
- realization/composition;
- artifact, member, resource and captured state.

These distinctions need not imply one universal record or giant identity tuple.

Check compatible replacement, retirement and address reuse, multiple prepared
views, physical replicas, independently evolving teacher/student participants,
and several roles of one participant.

Labels such as teacher, EMA or denoiser must not themselves create or merge
identities. Ordinary numerical updates and produced-data versions must not
automatically become topology or binding revisions.

Same-run restoration preserves required logical identities while establishing
a new execution session. New-run initialization records lineage between
different identities. Late old-session observations must retain their origin.

3. Delayed filing, revision allocation and captured scope

Follow an accepted transition through fact production, delayed delivery,
validation, accumulation, query and export.

Check that:
- rejected/stale/unpublished attempts remain attempts;
- accepted composition derives from authority publication, not loader success;
- revision allocation preserves accepted ordering and required uniqueness;
- arrival order and wall-clock time do not redefine semantic order;
- delivery retry preserves fact identity without creating another transition;
- missing correspondence yields ambiguity/incompleteness rather than guessed
  current state.

Finalized composition belongs to a named readiness checkpoint. An earlier
finalization does not establish later readiness or permanent immutability.

An artifact reported after a live transition must retain its actual captured
composition, state provenance and required finalized context—not whichever
observation was filed most recently.

Check both schema/query requirements and the documented reconciliation.
Do not supply an unstated ordering mechanism to repair an actual gap.

4. Required durability versus optional telemetry

Separate:
- required history, provenance and product metadata;
- required runtime-restoration state;
- required algorithmic feedback;
- optional tracker/resource telemetry.

Examine buffering, capacity, drops, sink failure, retry and shutdown.

Optional telemetry may have bounded degradation policies. Required coverage
cannot silently inherit those policies merely because it uses the same
delivery infrastructure.

When required filing fails after an effect has occurred, preserve the reached
effect while reporting the relevant incomplete coverage or unavailable
dependent use. Do not infer rollback or erase successful output.

Missing optional telemetry must not prove that work never ran or authorize
replay. Retrying a filing must not repeat computation, optimizer advancement,
participant publication or adaptive feedback.

Check that disabling a tracker does not disable feedback required by a selected
algorithm, and that feedback state belongs to its continuation owner.

Do not require synchronous optional storage on every hot-path invocation.
Do not infer that all required durable facts must therefore be asynchronous
or synchronously persisted: examine their actual accepted guarantees.

5. Observation and reporting fidelity

Trace actual execution/capability results into summaries, trackers and reports.

Check preservation of:
- independent activity coordinates;
- many-input and cross-owner associations;
- actual producer/evaluated-state provenance;
- known, skipped, failed, uncertain and unattempted outcomes;
- per-unit advancement versus action completion;
- local output, remote publication and reporting outcomes;
- snapshot capture, publication and restore readiness.

Reports must not strengthen the origin's guarantees or imply numerical parameter
change merely because an optimizer call returned.

Descriptive header/export facts prepared before writing are not proof that
resources exist or publication succeeded.

Metadata snapshots are collected-record views, not evidence that missing
execution state can be reconstructed.

6. Resource evidence and bounded inspection

Cross-check unchanged resource requirements with the new accepted projections.

Preserve distinctions between measured observations, structural quantities,
estimates, derived profiles and accounting statements.

Verify:
- process/rank/device/host and collection-frame scope;
- measurement source, derivation version and validity;
- qualified owner evidence and explicit accounting gaps;
- relationships to supporting facts, including relevant cross-run evidence;
- physical sharing/alias deduplication without merging logical owners;
- source selection and declared fallback/degradation;
- bounded collection cost and explicitly enabled expensive diagnostics.

A phase label, coincident memory movement or device reading must not establish
persistent participant ownership.

Structural queries must not trigger unbounded model inspection implicitly.
Authorized measurement producers may inspect within their accepted scope;
ordinary reports must not regain unrestricted Trainer/strategy access.

Check stale measurements are not re-emitted as newly collected facts, and that
compatibility JSONL remains a projection/fallback rather than a second canonical
database.

7. Catalog evidence and compatibility output

Check that source evidence supports only the claims its policy establishes.

Distinguish exact representation equality, selected-state meaning, catalog
identity, revision equality, lineage and behavioral compatibility.

Follow conflicting carried identities, weak aliases, same bytes with different
selected states, conversions and normalization policies.

Inspect P0-INV-06: experimental canonical-state fingerprints must not gain
automatic identity-merge authority merely because fixtures pass. Compare the
operative policy with any broader or older wording; use explicit supersession,
not presumed intent.

Compatibility keys and unverified external metadata must not become canonical
internal truth merely because they are present in a file.

Check explicit artifact export scope, deterministic identity-qualified output,
missing/ambiguous relationships, applicable ModelSpec validation, omission
behavior and retained external-parity obligations.

Include the unchanged no-metadata migration requirement. Do not assume an
export toggle waives required internal history, or use this audit to settle
its separately deferred long-term product policy.

8. Effective requirements across both changes and main specs

Build an explicit compatibility map for relevant requirements:
- unchanged main-spec obligations;
- requirements replaced by each MODIFIED delta;
- added or removed obligations;
- shared requirements requiring matching scenario identifiers/bodies;
- documented reconciliation and remaining integration gates.

Read surviving scenarios as well as replacement text.

Check matching shared loading and model-fact requirements where the ledger
requires parity. Do not demand that every related spec be textually identical.

Verify that unchanged typed-fact, builder/emitter, registry, projection,
identity, console, sink, family-ordering, adapter-provenance and resource
obligations remain coherent with the new runtime topology.

Legacy mode/strategy wording may be superseded where explicitly modified.
Do not silently treat every old statement as superseded merely because the
new architecture would prefer that.

Identify any obligation neither preserved nor explicitly displaced, and any
independent sync/archive or implementation path that would reintroduce
contradictory shared semantics.

Concrete traces

Use source-backed cases including:
- a binding accepted before required composition filing fails, then retried;
- an older observation arriving after a newer accepted revision;
- a stable artifact capture reported after topology changes;
- retired and successor participants sharing an authored address;
- one participant with several roles/views and shared physical storage;
- two units with different outcomes and missing optional completion telemetry;
- dropped tracker metrics alongside required adaptive feedback;
- a resource measurement with timing evidence but no established owner;
- a failed optional collector or an expensive structural query;
- an artifact projection from a long-lived multi-artifact metadata snapshot;
- restored logical identities with a new session and late prior-session facts;
- conflicting source identity or candidate-only fingerprint evidence.

Carry forward Pass 5's older-snapshot/history question: examine how recorded
history and origin correspondence remain honest if recovery considers a cut
predating a later retirement or transition. Do not invent a rollback/branching
policy or assume all older snapshots are unusable. Route the combined recovery
eligibility scenario to Pass 7 where necessary.

Evidence and neighboring passes

Investigate P0-INV-05 and P0-INV-06, relevant implications of P0-INV-01/02,
and P0-INV-09 for metadata/observability completion claims.

Reference inherited findings without duplicating them. Their local fixes are
outside this audit baseline.

Distinguish specified semantics, inspected code, recorded experiments, freshly
executed checks and untested durability/concurrency guarantees.

Do not re-audit all preparation or recovery mechanisms. Focus on their recorded
truth, consumer correspondence and surviving specification obligations.

Return

- a source-backed ownership and fact-flow map;
- an effective-requirement compatibility map across both changes and main specs;
- findings with P6-prefixed IDs under the general reporting protocol;
- concrete delayed-delivery, failure, projection and resource-evidence traces;
- dispositions of relevant inherited findings/leads;
- verified areas, evidence and coverage limitations;
- unresolved questions routed to Pass 7 or concrete G5/metadata integration.

Do not edit files or tasks, prescribe a universal event log or metadata schema,
introduce another runtime authority, or issue an overall G5-readiness verdict.

Do not dismiss a necessary missing guarantee as an implementation detail.
Equally, distinguish that from deliberately deferred field layouts, storage
mechanisms, delivery interfaces and production conformance work.
