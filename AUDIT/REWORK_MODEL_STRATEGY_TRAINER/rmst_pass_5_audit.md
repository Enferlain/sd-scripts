Pass 5: products, snapshots and recovery.

Apply the shared general audit protocol. This is a read-only concern audit,
not an implementation task or a new architectural survey.

Audit commit:
6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48

Confirm that the examined sources correspond to that commit. Identify any
additional working-tree content separately; do not silently combine revisions.

Previous reports are navigation, not authority. P1-REV-01, P2-REV-01 and
P4-REV-01 have local corrections outside this baseline. Reference them where
relevant rather than assuming those corrections are present.

Objective

Cross-compare cached values, trained products, runtime snapshots and metadata
snapshots. Determine whether their capture, dependencies, completeness,
publication and recovery guarantees are coherent and sufficiently specified.

Follow unfinished work, restoration of the same logical run and explicit
initialization of a new run.

Do not turn these distinctions into a closed product taxonomy. Shared timing,
resources or writing infrastructure may be legitimate; shared infrastructure
must not imply interchangeable semantics or outcomes.

Sources

- Read the governing source-authority rules and applicable design decisions,
  especially D11/G4.3/G4.4, D5/G4.1, composed run state and G3.5–G3.6.
- Read the complete training-artifact-persistence requirements/scenarios and
  the relevant persistence/restoration and caching sections of
  training-capability-coordination.
- Follow dependencies through run-participant-state, runtime preparation,
  optimization, accepted execution, metadata and observability.
- Include applicable companion-change and unchanged main-spec obligations,
  particularly metadata snapshots, model identity, lineage and resources.
- Consult the working sketch, research cases and experiments where the
  governing documents rely on their evidence.
- Use current production as behavior/migration evidence, not target authority.

Research cases have different recorded support dispositions. Do not assume
every researched algorithm, backend or recovery mechanism must ship in G5.

Examine these questions

1. What each saved result actually establishes

Build a source-backed comparison of:

- cached values: production, publication, dependency provenance and admission;
- trained products: declared consumer/use and required semantic coverage;
- runtime snapshots: accepted continuation coverage and recoverable positions;
- metadata snapshots: the metadata state actually captured or exported.

Check that success in one category does not silently establish another:
a readable cache is not necessarily admissible; a complete trained product
does not imply resumable execution; a metadata snapshot does not reconstruct
missing model, optimizer, input or operation state.

Conversely, do not prohibit legitimate shared resources or contributors.

2. Capture and source consistency

Follow declaration/request, semantic planning, capture, writing/publication
and actual results without requiring one universal implementation sequence.

Verify that capture preserves:
- selected participants, state variants, topology and relationships;
- relevant numerical state and dependency provenance;
- required non-parameter and domain-owned state;
- the accepted consistency relationships between contributors.

Check protection through actual source reads and backend completion, rather
than only Python return or capture-request time. Borrowed live tensors alone
must not establish a stable capture.

Examine later model updates, replacement or stage transitions while an older
capture is still being serialized or uploaded. Historical captures must retain
their actual source meaning; stronger latest-state policies must be explicit.

Different owners or ranks may have different valid coordinates. Do not assume
one global revision, identical progress or a universal end-of-step barrier.

3. Product dependencies and completeness

Distinguish product, semantic member and physical resource. Check many-to-many
coverage, sharding, optional constituents and shared resources.

Verify that the declared use determines required weights, non-parameter state,
construction/configuration, installation, representation and relevant
implementation requirements. Do not assume a state dictionary is sufficient
or that the training Python wrapper must be preserved.

Separate:
- consumer dependencies;
- training/preparation dependencies;
- historical provenance.

Check external-dependency completion against the accepted policy:
embedded state, available/resolvable resources and stable compatible external
references are not equivalent requirements.

A reference-only adapter product may legitimately be complete without embedding
or currently fetching its base when its policy permits that. Completeness must
not falsely promise current usability, self-containment or perpetual access.

Include base-plus-adapter changes, selected teacher/student/EMA variants,
auxiliary state and derived/merged products. Baseline product examples must
not become a restrictive taxonomy.

Check that export transformations are declared, preserve their resulting
meaning and do not silently mutate the running arrangement.

4. Publication, partial results and retention

Follow capture success, local writing, required remote publication, reporting
and restore readiness as distinguishable outcomes.

Examine:
- missing required shards, members, manifests or contributors;
- payload written but publication/indexing failed;
- known local output with pending or uncertain remote output;
- reporting or metadata failure after successful publication;
- cleanup failure after known output;
- retention overlapping reads, uploads or restoration.

Plans must not substitute for measured post-write facts. Exceptions or absent
return values must not imply that no resources exist.

Known product success must survive a later reporting failure, while required
reporting/publication obligations remain visibly unsatisfied.

Do not invent atomic multi-resource publication, rollback or permanent external
dependency availability. Check that retention respects actual membership and
outstanding accepted use.

5. Coherent snapshots and unfinished work

Determine what makes a supported snapshot a coherent recoverable cut across
the accepted whole-run relationships—not merely several individually durable
owner snapshots.

Follow produced, ready, admitted, handed-off, consumed, contributed, advanced
and durable positions where continuation depends on them.

Check the evidence or accepted reconstruction rules needed for:
- producer/input-provider continuation and outstanding handoffs;
- adaptive algorithms, RNG and required observations;
- pending gradients and contribution windows;
- derivative continuations and retained source versions;
- optimizer, scheduler, scaler and backend state;
- installed stage transformations and topology;
- unfinished external interactions.

Disposable work may be regenerated only under an accepted rule preserving
required identity, provenance and consumption distinctions.

Do not demand serialization of every transient object. Supported safe capture
boundaries or evidenced reconstruction can satisfy the guarantee. Conversely,
a saved cursor, paused producer or cache index alone must not prove coherence.

“Exact same-run restoration” must preserve the accepted logical run's required
continuation semantics, without implying bit-identical replay unless explicitly
required.

6. Failure and uncertain effects

Trace failure before and after meaningful effects.

Include:
- one optimization unit advanced while another failed or remains uncertain;
- temporary two-pass perturbation when capture is requested;
- an external interaction with unknown completion;
- partial rank/contributor publication;
- required restoration failure after other owners reconstructed successfully.

Check whether the supported policy waits, reports capture unavailable,
reconstructs evidenced unfinished work or uses an earlier coherent snapshot.

Missing completion records must not authorize blind replay. A clean visible
parameter tensor must not prove that all backend state is restored.

Do not infer rollback, exactly-once effects or an atomic multi-unit update.
Preserve known outcomes and uncertainty while gating dependent unsafe work.

7. Same-run restoration versus new-run initialization

Verify that restoration uses saved semantic identities, state definitions,
membership, revisions and dependency evidence—not wrapper equality, authored
addresses or tensor ordering.

Same-run restoration may recreate physical objects and establish a new
execution session while preserving logical participant and optimization
incarnations. Old handles and readiness evidence must not automatically become
current.

Check dependency-appropriate reconstruction, fresh readiness and explicitly
supported backend/state remapping. Already-installed transformations must not
be repeated merely because Python objects are new.

Late prior-session results require valid recovery/admission correspondence.

Missing required state must leave continuation incomplete, not silently reset
an optimizer, substitute input defaults or initialize another run.

A separately requested new run establishes new identities and lineage.
Artifact loading and restoration must not be conflated.

Failed restoration must not expose an incompatible mixture of restored and
fresh owner state. Unrelated activity need not wait for optional routes unless
its accepted dependencies require them.

Concrete traces

Use recorded cases to cover at least:
- an adapter product referencing an updated external base;
- a multi-member/sharded product with partial publication;
- a stable capture published after a topology transition;
- asynchronous production with ready and in-flight work across encoder changes;
- independently progressing owners with compatible different coordinates;
- partial multi-unit advancement and unsupported mid-perturbation capture;
- product publication followed by metadata/reporting failure;
- complete snapshot publication followed by an unavailable dependency;
- restoration without the authoring object, followed by a late old-session result;
- explicit new-run initialization from an earlier product.

These are source-derived traces unless actually executed. Do not require new
experiments merely to perform the audit.

Evidence and neighboring passes

Follow P0-INV-01, P0-INV-04, P0-INV-05 and P0-INV-08 where relevant, and
P0-INV-09 for persistence/restoration completion claims.

Distinguish specified semantics, recorded analysis, inspected experiments,
freshly executed checks and untested production guarantees.

Do not repeat revision or execution-authority audits wholesale. Route detailed
metadata schema/delivery reconciliation to Pass 6 and combined whole-run
sufficiency to Pass 7.

Return

- a concise comparison of saved-result meanings, owners and guarantees;
- findings with P5-prefixed IDs under the general reporting protocol;
- concrete capture, publication, failure and restoration traces;
- dispositions of relevant inherited findings/leads;
- verified areas, evidence and coverage limitations;
- unresolved questions routed to later passes.

Do not edit files or tasks, introduce a universal snapshot/result container,
prescribe G5 storage formats or issue an overall G5-readiness verdict.

Separate actual missing semantics from implementation choices deliberately
left to G5. The absence of a production recovery mechanism is not itself a
semantic defect; an unsupported claimed guarantee may be.
