# Model–strategy–Trainer: Pass 6 audit

**Concern:** metadata, observability, and effective specification compatibility.  
**Repository:** `Enferlain/sd-scripts`, requested branch `model-strategy-trainer`.  
**Examined revision:** `6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48` (`4.6`, 2026-10-04 04:39:22 UTC).  
**Report date:** 2026-10-05.

The target metadata/observability boundary is substantially explicit: origin owners establish runtime truth; central metadata validates and retains durable facts; observability delivers and presents them. The requirements cover delayed delivery, reached effects after filing failure, historical captures, independent progress, required feedback, and evidence-limited resource accounting. I found **one concrete compatibility/conformance issue in the existing metadata foundation**, P6-REV-01: ModelSpec extensions can overwrite the projection-owned specification version and bypass the validity of the final exported claim. This is a migration issue, not evidence that the new accepted-run ownership model is contradictory.

The shared loading and model-fact requirement blocks match across both active changes. No new semantic failure was established for composition ordering or first-fingerprint promotion. Existing code and tests do **not** establish the new durability, distributed ordering, or whole-run recovery guarantees. The older-snapshot/retirement/history combination remains routed to Pass 7. This report makes no overall G5-readiness verdict.

## 1. Snapshot, instructions, and evidence discipline

I used the three uploaded files as assigned: general protocol, Pass 0 navigation, and Pass 6 scope. Repository `AGENTS.md`, `DEVELOPMENT_GUIDE.md`, and the changelog top were read. The read-only assignment does not implement an OpenSpec milestone, change task status, or trigger the implementation review workflow.

This was a **remote pinned-source audit**, not a checkout of the user's working tree. GitHub's commit response confirms the requested SHA, message, timestamp, and root tree `a46f1fb0e1edc9cda5e29dec873ae0cb78f8de46`. Its recursive tree was not truncated. All repository retrievals used the full commit ref. Sixty-three materialized source files were checked against their pinned Git blob hashes; all matched. Scratch evidence copies are not a production checkout. No additional local repository content, dirty user checkout, or post-baseline correction was incorporated. The user's uncommitted state was unavailable and is not claimed clean. No remote or repository file/task was changed.

The governing hierarchy is the rework design's source-authority rule and reconciliation ledgers, with unchanged main requirements retained unless explicitly replaced or removed. The direction supplies settled input; dated corrections explain supersession; working exchange notes and research do not override adopted requirements. The companion change governs catalog/source/structural semantics and consumes the rework's runtime boundary. Code is evidence of the migration starting point. A checked task is a recorded design/evidence claim, not production conformance. [R-D], [M-D], [R-T], [M-T]

In particular, the 2026-08-26 topology correction explicitly supersedes the active authoring-strategy runtime peer, and the 2026-08-27 decision makes the active change the home of new normative decisions. G4.5 explicitly permits adaptation of the metadata foundation. Neither its current identifier strings nor its one-session inputs constrain the target. [NOTES], [R-D]

## 2. Ownership and fact-flow map

| Meaning | Establishing owner | Metadata/observability responsibility | Boundary that must survive |
|---|---|---|---|
| Participant, relationship, current binding, route/view, obligation and accepted transition | One contract-governed authority in the accepted arrangement | Consume explicit durable projections; qualify and retain history | A valid typed item is not another live binding acceptance or readiness decision. Loader success is candidate evidence. [RS], [LC] |
| Optimization unit definition, physical realization, window and reached mechanics | Trainer optimization under the accepted authority profile; granted executor for transferred phases | Retain unit incarnation/definition correspondence, contribution and per-mechanic outcomes | Returned optimizer call does not prove numerical change; action is not an atomic all-unit update. [OU], [OB] |
| Input production, admission, delivery, consumption, operation and capability effects | Respective input, selected behavior, capability and backend owners; run coordination preserves relationships | Keep actual producer/evaluated-state provenance, many-input joins and independent coordinates | Ready production is not consumption; missing telemetry is not proof of non-execution. [EX], [CC], [OB] |
| Catalog identity, source representation/selection, claim trust and lineage | Central persistent catalog authority; loaders supply truthful source decisions | Validate recognition evidence and policy, register durable facts, maintain derived indexes | A run filer cannot invent catalog IDs; bytes, selected state, lineage and behavioral compatibility are distinct claims. [CAT], [SRC] |
| Typed-item validity, durable qualification, records, relationships, accumulation/query and projection | Central metadata | Builders convert explicit inputs; registry selects emitters; public views and projections serve consumers | No live handle as canonical durable binding; no parallel type/routing registry or catalog database. [FM], [R-FM] |
| Resource measurement, structural quantity, profile and accounting interpretation | Authorized resource producers/interpreters | Central metadata validates, stores and shapes accepted resource exports; reports render evidence | Timing/device scope alone is not owner evidence; physical alias deduplication does not merge logical owners. [RI], [RV], [M-RI], [OB] |
| Stable capture, actual member/resource writing, local/remote publication, snapshot readiness | Persistence/snapshot coordinators and their contributors | Describe supplied capture/result and achieved coverage | Pre-write header facts are not proof that a resource exists. Metadata-store success is not runtime recovery. [AP], [CC], [R-FM] |
| Optional delivery, tracker and presentation | Observability concerns and selected sinks | Format, route, buffer and report degradation under policy | Required history and feedback cannot inherit optional drop policy; delivery retry repeats neither effects nor feedback. [OBS], [OB], [R-FM] |

The fact-flow is not a mandatory synchronous call chain. Authoritative source facts become available at their own boundaries. A valid live projection can exist before catalog resolution. A dependent durable filing must resolve applicable catalog/source identity under its policy, then pass through central builders, validation, registry/emitter and backend. Queries and exports consume the resulting explicit correspondence. Delivery can fail after the runtime effect; it cannot undo or repeat that effect. [R-FM] (“Model realization facts use the shared metadata runtime path” and “Metadata delivery cannot change origin-owned truth”), [R-D] (G4.5)

The two authorities therefore complement each other: runtime acceptance establishes what happened and what is current; metadata acceptance establishes the validity and qualification of the supplied record. Neither validates the other's unrelated responsibilities.

## 3. Effective requirement compatibility

The following is the **combined target**, not a claim that the active deltas have already been archived into the main specs. Appendix A inventories every requirement in the ten main capabilities examined, including its exact replacement/removal disposition and additions.

| Capability | Surviving main obligations | Explicit delta effect and compatibility assessment |
|---|---|---|
| `model-family-metadata` | Typed canonical facts; family-owned meaning and versioned contributions; declaration order and distinct same-role components; builder/emitter split; qualified identities; explicit artifact relationships; scoped deterministic Kuro/ModelSpec/ss output; required/optional ModelSpec facts; external parity and no-metadata migration behavior; removal/deferment ledger | Both changes replace distinction and shared filing requirements identically. Rework adds runtime/durable identity separation, history/order, actual publication/restoration outcome and required delivery rules. These additions qualify what builders and retained identities mean; they do not transfer runtime acceptance to metadata. P6-REV-01 concerns surviving ModelSpec output validity in the baseline code. [FM], [R-FM], [M-FM] |
| `training-observability` | Separate console/metrics/summaries/reports/resources; stdlib logging and Rich; early/runtime console continuity; swappable sinks with run initialization, metrics persistence and finalization; adapter provenance and public labels; additive resource schema; target modules; declared family ordering | Three named requirements are replaced: structured signals, training/observability separation, backbone structure. New whole-run scope/result, evidence-constrained diagnostics and delivery/feedback requirements remove broad strategy/mode/Trainer reconstruction. Old mode words in replaced scenario bodies do not survive. Unchanged sink and presentation duties remain. [OBS], [OB] |
| `resource-intelligence` | Distinct measured/structural/profile/accounting classes; metadata ownership; bounded observable ingestion; scoped relational identity and frames; versioned derived products; explicit gaps; public resource-run views and cross-run supporting evidence; canonical projections; operational collector policy, budgets and stale-reading protection | Companion adds qualified structural owners and shared cost/source query policy. No existing requirement is removed. Optional sampled telemetry degradation remains valid; the rework prohibits applying it silently to required coverage. “Export schema” ownership does not transfer resource interpretation to metadata or human presentation to runtime. [RI], [M-RI], [OB] |
| `resource-visibility` | Question-driven host memory; source-qualified aggregate/per-device GPU facts; live/profile/accounting/presentation distinctions | Unchanged in both changes. Device scope does not create ownership, and display-only summaries need not become canonical fact types. [RV] |
| `loaded-model-components` | Family-owned keys, labels, order, roles/capabilities and distinct same-role components | Shared loading replacement makes evidence/candidates separate from publication and forbids a competing Trainer collection. Rework replaces the highest-control-surface requirement to permit independently managed non-family participants while preserving model-substructure provenance. It does not fabricate family components for all participants. [LC-MAIN], [LC], [M-LC] |
| `optimization-target-refs` | Component-qualified selector compatibility | Rework replaces shared refs, policy neutrality, grouping, component derivation and lower-level expansion with authority-qualified semantic refs and pinned live views. Companion adds optional catalog/structural associations, shared-query reuse and observation-local aliases. A valid live projection does not require catalog resolution. The standalone-participant target coverage lead remains inherited P1/P0-INV-02. [TR-MAIN], [TR], [M-TR] |
| `adapter-system` | Broad future applicability; contract-based parameter grouping; typed production helpers | Four old ownership/persistence requirements are removed. Realization, method configuration and optimization-facing contract requirements are replaced. Authored intent, products versus initialization/restoration, and pipeline coordination are added. Retained general grouping duties do not reauthorize optimization-owned host targeting. [AS-MAIN], [AS] |
| `adapter-module-targeting` | Declared component provenance | Old optimization-owned targeting and spike-local overlap deferral are removed. Governed resolution and unsupported-overlap rejection are added. Provenance, parameter grouping, LoHa seams, method settings, strict continuation, migration fields and component scope are replaced. Transitional `module`/path fields are aliases of the same pinned projection, not another binding authority. [AT-MAIN], [AT] |
| `repo-owned-lora-method` | Repo-owned math/state/export; method-local config; source-target provenance and grouping independence | Build/loading requirement is replaced with governed PEFT targets and pipeline materialization. Method-local save/reload remains specialized mechanics, not overall persistence authority. [LO-MAIN], [LO] |
| `repo-owned-vera-method` | Shared projection bank and target-local views; supported wrapper layout; method-local config/persistence; deterministic omitted-projection reconstruction; trainable provenance | Build/loading requirement is replaced. Main file has ordinary requirement headings at this baseline. Shared physical state is not proof that target provenance or logical ownership can be collapsed. Reconstruction metadata is a product-use obligation where applicable, separate from optional telemetry. [VE-MAIN], [VE] |

Fresh comparison found no missing main requirement for a `MODIFIED`/`REMOVED` block and no lost existing main scenario identifier in the modified blocks examined. This checks delta correspondence, not semantic completeness or official OpenSpec validation.

| Shared block required by G4.6 ledger | Scenario count | Fresh result |
|---|---:|---|
| Loading: “Model loading returns the loaded-component surface” | 4 | Exact normalized requirement/scenario body parity |
| Model facts: “Family declaration, model realization, and artifact facts remain distinct” | 6 | Exact normalized requirement/scenario body parity |
| Model facts: “Model realization facts use the shared metadata runtime path” | 4 | Exact normalized requirement/scenario body parity, including `Model loading completes` |

Whole files are intentionally different. Companion target-reference additions and rework runtime replacements are complementary, not parity blocks. The rework ledgers forbid independent implementation/sync/archive of the overlap. Companion task 4.1 is still unchecked and requires concrete G5 types, strict validation and review of both changes; sections 6–7 retain composition/lineage work, and section 8 gates structural work on shared query and qualified resource-component identity. These gates remain substantive obligations. G4.6 did not complete them. [R-D] (reconciliation ledgers), [M-T] (4.1, 6, 7, 8, 12)

### No-metadata compatibility

The unchanged `No-metadata policy is requested` scenario preserves current observable export behavior and explicitly defers the long-term policy. At this baseline, `build_checkpoint_metadata(no_metadata=True)` excludes related realization context, training/checkpoint facts and ss output, but still projects the selected model-artifact Kuro facts and ModelSpec. The named parity test asserts that behavior. This is source/test evidence, not a fresh family-runtime test. [CKPT], [PARITY] (`test_no_metadata_preserves_modelspec_and_model_identity_only`)

It is therefore inaccurate to equate this option with “no metadata whatsoever.” It also does not waive required internal history, required product reconstruction information, restoration contributions or adaptive feedback. Their requiredness follows accepted policy and surviving product/method requirements. Whether a particular VeRA omitted-projection product is complete must be evaluated against its required reconstruction information; changing the long-term export policy is outside this pass. [FM], [VE-MAIN], [AP], [R-FM]

## 4. Findings

### P6-REV-01 — Extension overrides can falsify the final ModelSpec claim

**Impact:** Medium for compatibility consumers; concrete implementation/migration nonconformance. It does not establish a new runtime-authority contradiction or an overall G5 blocker.

**Exact sources:**

- Main `model-family-metadata`, requirement **“ModelSpec claims validate required artifact facts”**, scenarios **“Projection stamps its supported ModelSpec version”**, **“Required ModelSpec fact is missing”**, and **“Optional family fact is inapplicable”**. [FM] (lines 183–198)
- Main **“Compatibility metadata is projected from canonical facts”**, particularly **“Projecting SAI ModelSpec metadata”** and **“Rendering safetensors metadata”**. [FM] (lines 140–181)
- `library/metadata/projections.py`, `ModelSpecCompatibilityProjection.project`, lines 156–204; `library/metadata/keys.py`, `MODELSPEC_VERSION = "1.0.1"`. [PROJ], [KEYS]
- `library/metadata/validation.py`, `ModelArtifactFacts` branch, lines 179–205: canonical mandatory fields are checked and prefixed extension keys rejected, but reserved unprefixed extension names are accepted. [VALID]
- `tests/unit/metadata/test_projections.py`, **`test_modelspec_projection_maps_canonical_artifact_facts_and_applies_extensions_last`**, lines 110–145, explicitly expects `modelspec.sai_model_spec = "user-version"`. [PROJ-TEST]

**Relationship:** Surviving normative requirement versus inspected/tested baseline behavior. The projection stamps its implemented version, copies validated canonical fields, then applies arbitrary string extension fields last. Thus `sai_model_spec` replaces the version. Other reserved fields can replace required canonical claim values after those canonical values passed validation. Final safetensors conversion stringifies the result; it does not restore claim validity.

**Concrete case:** An otherwise valid artifact has canonical title `Valid`, architecture and implementation facts, and extension fields `{"sai_model_spec": "user-version", "title": ""}`. The exported header advertises `user-version` and an empty title. The canonical facts have not changed, but the final compatibility claim no longer faithfully reflects the projection-owned version or the validated required claim. The existing test already confirms the version override; a fresh focused probe of the pinned projection method produced both overrides.

**Evidence limits:** The focused probe extracted the unchanged projection method/helper/constants and supplied lightweight scope/validation collaborators. It verifies the final mapping behavior, not the entire registry/backend pipeline. The inspected production validator independently permits these unprefixed string extension names. I did not show that the ordinary Trainer currently exposes this combination through its user configuration; its inspected builder call supplies no extension mapping. The public central projection and baseline test nevertheless exercise the behavior that the migration must assess. Current production is not the target authority, so this is not reported as a contradiction created by G4.5.

**Smallest correction:** Reserve the projection-owned specification/version key against extension override. For extensions overlapping typed claim fields, ensure the final emitted values obey the selected claim's validation/omission rules, or reject those collisions. Preserve valid non-conflicting extension fields. Update the test that currently endorses the version override. Check both standalone and long-lived checkpoint projections, and record any real external-parity exception needed rather than silently broadening the existing exception list. No alternative metadata schema or runtime authority is necessary.

**Neighbors affected:** Main external-parity requirement, typed artifact presentation/extension validation, standalone artifact export, final safetensors projection and compatibility tests. The rework/companion consumer integration must not treat the current passing extension test as proof of final ModelSpec conformance.

### Advisory P6-NOTE-01 — Fingerprint prose should retain the explicit promotion gate

This is **not a confirmed permission to merge** and is not a second correctness finding. Companion design line 345 says experimental fingerprints stay non-merging “until” fixtures prove semantics; its open question line 356 says the fixture milestone decides policy-by-policy. Read alone, that prose can suggest passing fixtures is sufficient. However, the explicit decision at lines 315–317, the structural spec's **“First canonical state fingerprint passes its fixtures”** scenario, and tasks 13.3/13.5 expressly require the first policy to remain candidate-only even after passing. Promotion requires a later accepted policy/spec change. This is direct qualification, not supersession inferred from file dates. [M-D], [STRUCT], [M-T]

The smallest prose alignment is to make the migration/open-question wording refer to evidence for a **future explicitly accepted policy**. The operative first-policy prohibition already exists; this advisory does not reopen it or demand another architecture.

## 5. Concrete fact-flow and failure traces

These are source-backed semantic traces. They are not claims that an integrated implementation has executed them.

### T1. Accepted binding, required composition filing failure, then retry

The authority validates and publishes the binding transition. Its accepted runtime effect remains reached. Loading/catalog evidence and the origin's state correspondence are retained for the required durable observation. If catalog-dependent filing or backend delivery fails, coverage is incomplete/unavailable under the accepted policy; dependent use requiring that coverage cannot claim readiness. A retry files the **same fact/transition correspondence**, not another authority publication or another semantic composition change. This does not require a synchronous optional sink at publication and does not establish a universal storage protocol. Crash-safe retention/retry details remain integration work. [R-FM] (`Required composition filing fails after publication`, `Metadata filing fails after a binding was accepted`, `One accepted transition is delivered again`), [SRC], [M-T] (6.1–6.7)

### T2. Older accepted observation arrives after a newer revision

Accepted composition changes C2 and C3 occur in that order; C2's delivery is delayed. Allocation must be unique and monotonic for the realization **with correspondence to originating accepted state**, not the arrival sequence. C3's earlier delivery cannot entitle later-arriving C2 to a higher semantic rank. A latest view uses validated accepted revision/state correspondence; missing correspondence produces ambiguity/incompleteness. Redelivery preserves identity. A failed or rejected candidate consumes no accepted composition revision. [R-FM] (`Older observation is ingested last`), [SRC] (immutable observations and semantic-order views), [M-D] (composition decision)

**P0-INV-05 disposition:** No additional semantic assumption is needed to prohibit arrival-assigned inversion: the design expressly ties monotonic allocation to origin order and both specs forbid arrival from redefining it. Choosing an allocation/reservation/validation mechanism is G5/metadata work. The rules do not prove that an eventual concurrent implementation satisfies them. An allocator issuing C3 a lower ordinal because its filing arrived first would violate the specified correspondence; uniqueness alone would not save it.

### T3. Stable artifact capture is reported after topology changes

A protected dense-state capture selects its actual variant, participant/relationship coverage, dependencies and composition at boundary B. Required reads finish before conflicting changes. The live arrangement then changes to MoE or replaces a component. Historical serialization/upload can finish where the product policy permits. Post-write facts retain B's state provenance and actual captured composition, plus its corresponding finalized checkpoint where required. An arbitrary later `final` record is invalid substitution. Header facts prepared before writing remain descriptive; actual resource results supply completion/partial status. [AP] (capture and actual output), [SRC] (`Artifact is produced from a realization`), [R-FM] (captured participant and pre-write scenarios)

### T4. Retirement and authored-address reuse

Participant P at address `student` is retired. A later declaration at `student` establishes successor Q. Metadata preserves P and Q separately, with succession/lineage where supplied; address equality cannot overwrite P. A late measurement of P stays scoped to P and its valid session/state, not Q. Compatible binding replacement of a continuing P instead preserves incarnation while recording changed dependencies. [RS], [R-FM] (retired-address and compatible-replacement scenarios), [OB] (address reuse)

This trace does not assume that a locally corrected P1 finding is present in the baseline. The general identity requirement exists, but the inherited target-reference coverage issue remains separately referenced.

### T5. Several roles/views and shared physical storage

One authority-established P may serve several roles and expose wrappers, replicas, inspection views and routes. Labels and physical objects do not create participant identities. Separately declared student and independently evolving teacher are distinct even with the same source. An authorized measurement can associate multiple qualified paths/views with one observation-local object/storage group and deduplicate supported physical quantities within that observation, while retaining each logical association. Same-unit/group optimization aliases and unsupported cross-unit/group overlap retain separate rules. [R-D] (G4.5), [R-FM], [STRUCT], [M-TR], [OU]

### T6. Two units, different outcomes, missing optional completion telemetry

An action consumes admitted inputs I1 and I2. U1's call returns, U2 enters its backend and has an uncertain outcome, and later mechanics are unattempted. Owners report those reached facts and the common attempt relationships. Losing the optional overall metric does not erase U1, prove U2 never ran, imply numerical change, or permit action replay. Required correlation remains available or is honestly reported incomplete. Unsafe dependent execution stops until accepted recovery establishes validity. [EX] (cross-owner facts and failure), [OU] (separate reached mechanics), [OB] (two-unit and missing-observation scenarios)

### T7. Dropped tracker copy and required adaptive feedback

The accepted operation supplies feedback to its adaptive owner through the runtime exchange. A metric copy enters an optional bounded buffer and is dropped or fails ingestion. The adaptive owner still receives the required feedback and owns the continuation state it creates. Flushing/retrying the copy cannot update that owner again. Disabling the tracker cannot disable the algorithm. Restoration covers required adaptive state, not a copy assumed recoverable from a telemetry queue. [OB] (adaptive-loss scenario), [R-D] (G4.5), [CC] (continuation coverage)

### T8. Resource movement with timing evidence and no owner

A collector records GPU/process memory during phase F with device, rank, source, value and frame context. That can be an honest observation without participant ownership. Coincident movement and phase F do not establish persistent ownership. Structural accounting requires its qualified owner, quantity/scope, validity, derivation version and supporting facts; otherwise an explicit gap remains. Supporting evidence may be cross-run or artifact evidence without becoming a resource owned by the viewed run. [RI] (relational facts, frames, evidence-constrained accounting and source evidence), [RV], [OB] (phase-only scenario)

### T9. Optional collector failure and an expensive missing inventory

An unavailable optional collector degrades with status/source evidence; a supported fallback records its concrete source and quality. No measurements means status/degradation, not an empty frame disguised as successful collection. A warning-budget breach is evidence of degradation unless hard cancellation is actually supported. CUDA peak resets belong to explicit diagnostic-window behavior. A later frame cannot relabel cached old samples as newly co-collected. [RI] (operational policy)

A resource query lacking an inventory uses the shared allowed cost/source policy or remains incomplete. Existing immutable descriptors can be reused only for the exact supported owner/policy/coverage. Expensive live inspection requires explicit scope/diagnostic enablement; ordinary reports cannot regain unrestricted Trainer/strategy access. [M-RI], [STRUCT], [R-D] (G4.5)

### T10. Artifact export from a long-lived multi-artifact metadata snapshot

The request identifies artifact A explicitly. Its native and compatibility context follows accepted relationships to A's realization, components and applicable family contributions, preserving actual captured scope. Missing/ambiguous required target relationships fail or remain unavailable under the relevant contract; unavailable provenance identity is omitted where the main requirement allows omission, not invented. Repeated entities use deterministic identity-qualified Kuro output independent of backend iteration order. ModelSpec validates its applicable claim and omits inapplicable optional facts; P6-REV-01 shows why the final extension stage also matters. [FM] (compatibility, identity and ModelSpec requirements), [SRC]

The baseline has explicit artifact scope, deterministic multi-component keys and ambiguity tests, but its `_select_model_artifact_record` selects the last record for a stable artifact identity. That is migration evidence, **not** a permitted general whole-run semantic-order rule. New composition/capture/session correspondence must reach artifact queries; current snapshot iteration does not prove that integration. [PROJ], [PROJ-TEST], [R-D] (G4.5), [M-T] (6.3/6.6)

### T11. Same logical run, new session, late old-session facts

A supported same-run restore preserves required authority, participant, relationship and unit identity/revision state and establishes fresh preparation/readiness and session S2. Late observations from S1 retain S1 and their actual attempts; arrival during S2 does not make them current. Late work results require the accepted recovery/admission rule before adoption. Metadata snapshots remain record views, not evidence that missing input/optimization/adaptive state can be reconstructed. Missing required identity facts gate same-run continuation; they do not create a new run automatically. [RS], [CC] (coordinated restoration and late result), [R-FM], [OB]

### T12. Conflicting carried identity or a fixture-passing normalized fingerprint

Identical bytes resolve the same exact representation under policy, but incompatible publisher lineage claims remain separate claims/conflict evidence. EMA versus non-EMA selections remain distinct even within those bytes. Unsigned/invalid carried identities cannot independently merge or redirect unrelated identities. Accepted conversions can relate distinct representations to a revision only with their declared evidence; uncertain transformation meaning remains unresolved. A fixture-passing first canonical fingerprint is still candidate-only. No role, architecture, parameter count or similarity grants identity-merge authority. [CAT], [STRUCT], [M-D]

## 6. Older snapshot, later retirement, and recorded history

Consider snapshot K at accepted scope A1 containing P. The later session reaches A2, retires P and possibly declares Q at the same address; history H records those outcomes. Recovery then considers K for the same logical run. Three things must remain distinct:

1. K's saved identity/state and capture claim.
2. The actual later reached transitions and effects recorded in H.
3. Whatever restored current scope a supported recovery policy can legitimately publish.

Metadata delivery cannot erase A2, invent that it never occurred, or interpret late A2 evidence as work newly performed in the restored session. Equally, selecting the numerically largest historical revision across every session cannot by itself certify the restored current state. Queries claiming current meaning require correspondence to the authority's applicable restored scope; otherwise they report ambiguity/incompleteness. [R-FM] (history and origin truth), [OB] (session scope), [SRC] (valid requested-scope correspondence)

The baseline also says a retired participant reference is never revived, while recovery may use a prior coherent snapshot. **This pass does not resolve whether this particular K is eligible, how post-cut effects are reconciled, or what a supported recovery branch would mean.** It does not assume every older snapshot is unusable. Those combined policy/identity questions belong to Pass 7 with Pass 5's question and the inherited P1/P2 recovery findings. Once the allowed recovery meaning is established, metadata integration must demonstrate honest history/current/capture queries across that case without rewriting history or selecting by ingestion order. [RS], [R-D] (G4.4)

## 7. Inherited findings and investigation leads

| Lead/finding | Pass 6 disposition |
|---|---|
| P0-INV-05 — delayed composition ordering | Specified necessary meaning is present; origin-corresponding monotonic uniqueness, semantic selection and idempotent redelivery must be implemented and tested. No new semantic finding from mechanism absence. |
| P0-INV-06 — fingerprint promotion | First-policy prohibition survives fixture success explicitly. Residual broad prose is P6-NOTE-01; no current automatic promotion permission established. |
| P0-INV-01 / P2-REV-01 — recovery wording | Baseline G4.4 introduction still contains the broad absent-facts/new-identities wording. RS and later explicit restore rules prohibit silent fallback; metadata must follow the actual accepted request/outcome. The supplied assignment says a local correction exists outside this baseline; it was not incorporated. Refer to inherited finding rather than duplicate it. |
| P0-INV-02 / P1-REV-01 — standalone participant targets | Runtime non-family participants are explicitly allowed, but target-ref/component coverage is the inherited issue. Metadata must not fabricate family components or substitute bare keys/pointers. Local correction is outside this baseline. Verify structural/resource consumers against the actual later correction at integration. |
| P4-REV-01 | Assignment reports a local correction outside the baseline. No Pass 4 report or correction body was supplied/read here, so its precise diagnosis or resolution is not inferred. Required origin/evaluated-state provenance is independently checked in this pass's sources; any inherited capability deficiency remains with Pass 4/7. |
| P0-INV-09 — checked completion versus evidence | G4.5/4.6 establish planning/consumer semantics and ledgers, not integrated durable publication, all-owner snapshots or concurrent revision allocation. Companion implementation remains largely unchecked. Do not inflate those checkboxes into production metadata support. |
| Pass 5 older-cut/history question | Preserving actual history is specified; combined recovery eligibility, permanent retirement and post-cut effect reconciliation are routed to Pass 7. No rollback/branching policy invented. |

Only Pass 0 and the assignment's inherited-finding notice were available as prior-report evidence. I did not reconstruct unseen reports from their IDs.

## 8. Code, tests, and verification limits

| Evidence | What was inspected or freshly checked | What it does not establish |
|---|---|---|
| Pinned retrievals | Commit/tree metadata and all 63 materialized Git blob hashes freshly verified | User's dirty checkout or local fixes; latest branch equivalence |
| Delta compatibility | Fresh requirement parser checked modified/removed homes, preserved main scenario identifiers and all three parity blocks | Official strict OpenSpec validation, review completion or full semantic proof |
| Model builder/dataclasses | Module-free explicit facts; qualified realization/component identities and declared order; baseline facts lack new composition/session/incarnation associations | Current schemas already representing the dynamic accepted run |
| Trainer filing/checkpoint assembly | Session-derived `training-target` realization retained after successful filing; artifact facts filed before writing; no-metadata projection selection | Same-run logical identity restoration, actual product publication or revised composition history [TRAINER] |
| Metadata runtime/backend | Direct `file`/batch path and bounded buffer; failed telemetry ingestion counts losses; snapshot `record_for` returns last inserted | Required-history retry/durability, crash-safe retention, semantic ordering or catalog resolution [RUNTIME], [BACKEND] |
| Existing telemetry tests | Selected bodies cover bounded batches, both drop policies and failed-ingestion degradation; failure test expects `pending_items == 0` and an ingestion loss | Durable retry of required facts or required-feedback delivery; tests were inspected, not run [TELEMETRY-TEST] |
| Projection tests/code | Artifact scoping, ambiguity, deterministic ordering and extension override behavior; focused extracted-method override probe freshly run | Full family parity/registry/backend execution or new capture-aware historical query conformance [PROJ], [PROJ-TEST] |
| Resource fact production/view | Canonical frame versus compatibility fallback is exclusive; measurement source/scope fields; view checks frame's run and declared measurement order; explicit cross-run/artifact evidence resolution | All collectors' cost bounds, physical-owner accuracy, session reuse or distributed stale-reading prevention [RESOURCE-CODE], [RESOURCE-VIEW] |
| Family parity/no-metadata test | Named assertions and checkpoint projection body inspected | Fresh training/writer parity for SD/SDXL/SD3 or every unsupported/combined product |

No production suite, GPU/distributed test, process-restart catalog test, official OpenSpec CLI validation, or review-MCP run was performed. The report relies on complete relevant main/delta requirements and scenarios, selected design sections, and targeted source/test bodies. Materialized files not substantively examined are not claimed fully audited. Supporting architecture and execution experiments were not rerun; no fingerprint experiment was treated as identity authority.

## 9. Concrete integration and Pass 7 routes

The following closes the audit's evidence loop without prescribing a universal log, tuple, schema or allocator:

- **G5/metadata composition integration:** demonstrate T1/T2 with required coverage failure, retry, concurrent acceptance/allocation and delayed arrival. Validate origin order and fact identity separately from unique database keys. [M-T] (4.1, 6.1–6.7)
- **Capture and query integration:** demonstrate T3/T10 with at least two artifacts, more than one accepted composition and finalized checkpoint, ordinary numerical state changes without topology revision, and delayed filing. No last-inserted/current-live substitution. [M-T] (6.3/6.6, 7.1–7.4), [FM]
- **Identity/session integration:** consume actual G5 and inherited P1/P2 corrections for address reuse, several roles/views, standalone learned participants and same-run restoration. Check qualified target, structural and resource associations without invented family components or catalog prerequisites for live access. [M-T] (4.1, 9–12)
- **Delivery policy integration:** required facts need an accepted retention/failure/retry/shutdown policy that cannot silently use the inspected lossy telemetry semantics. Verify effect/report separation and feedback exactly once under the accepted runtime exchange. The representation and storage mechanism remain open. [R-FM], [OB], [R-D] (G4.5)
- **Compatibility integration:** address P6-REV-01; preserve applicable omission, external parity and current no-metadata behavior; maintain ModelSpec version authority through final output. [FM], [PROJ]
- **Resource integration:** verify actual selected collectors' cost/source/degradation behavior, no stale re-emission, explicit owner gaps, cross-run supporting evidence and query cost limits. Shared query/qualified identity dependency gates remain required. [RI], [RV], [M-T] (8, 11, 12)
- **Pass 7 combined recovery case:** decide permitted older-cut recovery meaning against permanent retirement and known/uncertain post-cut effects, then demonstrate metadata/history/current-query correspondence. Keep the snapshot's original achieved guarantee separate from later restore eligibility. [RS], [CC], [R-D] (G4.4)

These are concrete conformance obligations already implied by the sources, plus the narrow compatibility correction. They do not authorize implementation in this audit.

## Source index

All repository links below are pinned to the examined commit. Abbreviations identify exact files; requirement/scenario names in the report narrow the cited section.

[R-D]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/design.md
[M-D]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/design.md
[R-T]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/tasks.md
[M-T]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/tasks.md
[NOTES]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/docs_design/models-strategy-trainer/notes.md#L1145-L1247
[FM]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/model-family-metadata/spec.md
[R-FM]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/model-family-metadata/spec.md
[M-FM]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/model-family-metadata/spec.md
[OBS]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/training-observability/spec.md
[OB]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-observability/spec.md
[RI]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/resource-intelligence/spec.md
[RV]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/resource-visibility/spec.md
[M-RI]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/resource-intelligence/spec.md
[CAT]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/model-catalog-identity/spec.md
[SRC]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/model-source-provenance/spec.md
[STRUCT]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/model-structure-metadata/spec.md
[RS]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/run-participant-state/spec.md
[OU]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-optimization/spec.md
[EX]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/accepted-training-execution/spec.md
[CC]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-capability-coordination/spec.md
[AP]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-artifact-persistence/spec.md
[LC-MAIN]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/loaded-model-components/spec.md
[LC]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/loaded-model-components/spec.md
[M-LC]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/loaded-model-components/spec.md
[TR-MAIN]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/optimization-target-refs/spec.md
[TR]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/optimization-target-refs/spec.md
[M-TR]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/optimization-target-refs/spec.md
[AS-MAIN]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/adapter-system/spec.md
[AS]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/adapter-system/spec.md
[AT-MAIN]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/adapter-module-targeting/spec.md
[AT]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/adapter-module-targeting/spec.md
[LO-MAIN]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/repo-owned-lora-method/spec.md
[LO]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/repo-owned-lora-method/spec.md
[VE-MAIN]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/repo-owned-vera-method/spec.md
[VE]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/repo-owned-vera-method/spec.md
[TRAINER]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/library/training/runners/trainer.py
[RUNTIME]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/library/metadata/runtime.py
[BACKEND]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/library/metadata/backends.py
[PROJ]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/library/metadata/projections.py#L156-L204
[KEYS]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/library/metadata/keys.py
[VALID]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/library/metadata/validation.py#L179-L205
[PROJ-TEST]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/tests/unit/metadata/test_projections.py
[CKPT]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/library/metadata/emitters/checkpoint.py
[PARITY]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/tests/unit/metadata/test_model_family_metadata_parity.py
[TELEMETRY-TEST]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/tests/unit/metadata/test_telemetry_ingestion.py
[RESOURCE-CODE]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/library/logging/resource_monitor/fact_production.py
[RESOURCE-VIEW]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/library/metadata/views/resource.py


## Appendix A. Requirement-by-requirement compatibility inventory

R = rework change; M = companion metadata change. “Retained” means the full main requirement and its scenarios survive; replacements use the complete delta body, not a mixture of old and new scenarios. Shared replacements are applied once after reconciliation. This inventory is a structural cross-check accompanying the semantic assessment in section 3.


### model-family-metadata

| Requirement | Combined target disposition |
|---|---|
| [Typed model facts are the authoritative internal representation](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/model-family-metadata/spec.md#L11) | Retained unchanged, including all scenarios |
| [Family declaration, model realization, and artifact facts remain distinct](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/model-family-metadata/spec.md#L24) | R: MODIFIED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/model-family-metadata/spec.md#L3)); M: MODIFIED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/model-family-metadata/spec.md#L3)) |
| [Family-owned semantics feed central metadata ownership](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/model-family-metadata/spec.md#L37) | Retained unchanged, including all scenarios |
| [Model component facts derive from family declarations](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/model-family-metadata/spec.md#L60) | Retained unchanged, including all scenarios |
| [Central builders separate source conversion from emission](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/model-family-metadata/spec.md#L78) | Retained unchanged, including all scenarios |
| [Model and component identities are durably qualified](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/model-family-metadata/spec.md#L96) | Retained unchanged, including all scenarios |
| [Model realization facts use the shared metadata runtime path](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/model-family-metadata/spec.md#L114) | R: MODIFIED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/model-family-metadata/spec.md#L45)); M: MODIFIED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/model-family-metadata/spec.md#L45)) |
| [Artifact relationships reuse accepted model identities](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/model-family-metadata/spec.md#L127) | Retained unchanged, including all scenarios |
| [Compatibility metadata is projected from canonical facts](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/model-family-metadata/spec.md#L140) | Retained unchanged, including all scenarios |
| [ModelSpec claims validate required artifact facts](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/model-family-metadata/spec.md#L183) | Retained unchanged, including all scenarios |
| [Active family exports preserve external parity](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/model-family-metadata/spec.md#L200) | Retained unchanged, including all scenarios |
| [Replaced active dictionary seams are removed](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/model-family-metadata/spec.md#L238) | Retained unchanged, including all scenarios |
| Runtime participant identity is projected from the accepted run authority | R: ADDED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/model-family-metadata/spec.md#L81)) |
| Metadata keeps runtime and durable model identity meanings distinct | R: ADDED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/model-family-metadata/spec.md#L101)) |
| Metadata history preserves observed revisions and accepted outcomes | R: ADDED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/model-family-metadata/spec.md#L126)) |
| Artifact and restoration observations report their own achieved guarantees | R: ADDED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/model-family-metadata/spec.md#L155)) |
| Metadata delivery cannot change origin-owned truth | R: ADDED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/model-family-metadata/spec.md#L182)) |

### training-observability

| Requirement | Combined target disposition |
|---|---|
| [Training observability uses repo-owned structured signals](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/training-observability/spec.md#L9) | R: MODIFIED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-observability/spec.md#L3)) |
| [Observability ownership is organized by concern](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/training-observability/spec.md#L21) | Retained unchanged, including all scenarios |
| [Console observability preserves the repo logging stack and presents one user-facing surface](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/training-observability/spec.md#L32) | Retained unchanged, including all scenarios |
| [Observability supports swappable sinks and future repo-owned back-ends](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/training-observability/spec.md#L51) | Retained unchanged, including all scenarios |
| [Adapter diagnostics derive from repo-owned adapter provenance](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/training-observability/spec.md#L72) | Retained unchanged, including all scenarios |
| [Resource monitoring remains a distinct observability sub-concern](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/training-observability/spec.md#L89) | Retained unchanged, including all scenarios |
| [Resource-monitor report surfaces preserve additive live schema growth](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/training-observability/spec.md#L114) | Retained unchanged, including all scenarios |
| [Training and observability ownership stay separate](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/training-observability/spec.md#L126) | R: MODIFIED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-observability/spec.md#L19)) |
| [First-pass implementation lands in the target observability modules](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/training-observability/spec.md#L138) | Retained unchanged, including all scenarios |
| [Backbone observability derives from declared loaded components](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/training-observability/spec.md#L151) | R: MODIFIED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-observability/spec.md#L37)) |
| [Observability preserves family-declared top-level ordering](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/training-observability/spec.md#L164) | Retained unchanged, including all scenarios |
| Whole-run observations preserve owner scopes and independent progress | R: ADDED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-observability/spec.md#L55)) |
| Diagnostics and resource accounting use accepted subject and ownership evidence | R: ADDED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-observability/spec.md#L86)) |
| Observation degradation is separate from execution and required feedback | R: ADDED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-observability/spec.md#L117)) |

### resource-intelligence

| Requirement | Combined target disposition |
|---|---|
| [Resource intelligence preserves fact semantics](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/resource-intelligence/spec.md#L6) | Retained unchanged, including all scenarios |
| [Resource facts use the metadata backbone](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/resource-intelligence/spec.md#L15) | Retained unchanged, including all scenarios |
| [Resource telemetry ingestion is bounded and observable](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/resource-intelligence/spec.md#L31) | Retained unchanged, including all scenarios |
| [Resource facts preserve relational identity](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/resource-intelligence/spec.md#L41) | Retained unchanged, including all scenarios |
| [Co-collected observations preserve frame identity](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/resource-intelligence/spec.md#L55) | Retained unchanged, including all scenarios |
| [Resource profiles are durable derived products](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/resource-intelligence/spec.md#L72) | Retained unchanged, including all scenarios |
| [Resource accounting is evidence-constrained](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/resource-intelligence/spec.md#L83) | Retained unchanged, including all scenarios |
| [Reports and analyses consume a resource-run view](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/resource-intelligence/spec.md#L109) | Retained unchanged, including all scenarios |
| [Resource artifacts are projections](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/resource-intelligence/spec.md#L131) | Retained unchanged, including all scenarios |
| [Runtime integration remains low-coupling](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/resource-intelligence/spec.md#L163) | Retained unchanged, including all scenarios |
| [Collection capabilities carry operational policy](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/resource-intelligence/spec.md#L172) | Retained unchanged, including all scenarios |
| Structural model resources use accepted qualified owners | M: ADDED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/resource-intelligence/spec.md#L3)) |
| Resource queries do not trigger unbounded structural inspection | M: ADDED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/resource-intelligence/spec.md#L21)) |

### resource-visibility

| Requirement | Combined target disposition |
|---|---|
| [Resource monitor records broader host-memory facts](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/resource-visibility/spec.md#L6) | Retained unchanged, including all scenarios |
| [Resource visibility can grow beyond aggregate-only GPU views](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/resource-visibility/spec.md#L17) | Retained unchanged, including all scenarios |
| [Report/debug summaries remain distinct from live resource facts](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/resource-visibility/spec.md#L27) | Retained unchanged, including all scenarios |

### loaded-model-components

| Requirement | Combined target disposition |
|---|---|
| [Training-capable model families declare top-level loaded components](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/loaded-model-components/spec.md#L7) | Retained unchanged, including all scenarios |
| [Loaded components declare generic semantics explicitly](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/loaded-model-components/spec.md#L20) | Retained unchanged, including all scenarios |
| [Model loading returns the loaded-component surface](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/loaded-model-components/spec.md#L33) | R: MODIFIED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/loaded-model-components/spec.md#L3)); M: MODIFIED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/loaded-model-components/spec.md#L3)) |
| [Top-level components remain the highest generic control surface](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/loaded-model-components/spec.md#L46) | R: MODIFIED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/loaded-model-components/spec.md#L37)) |

### optimization-target-refs

| Requirement | Combined target disposition |
|---|---|
| [Shared optimization target refs](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/optimization-target-refs/spec.md#L6) | R: MODIFIED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/optimization-target-refs/spec.md#L3)) |
| [Target refs preserve selector compatibility](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/optimization-target-refs/spec.md#L29) | Retained unchanged, including all scenarios |
| [Target refs are policy-neutral](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/optimization-target-refs/spec.md#L43) | R: MODIFIED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/optimization-target-refs/spec.md#L26)) |
| [Existing grouping behavior remains stable](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/optimization-target-refs/spec.md#L58) | R: MODIFIED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/optimization-target-refs/spec.md#L42)) |
| [Top-level component target refs derive from declared loaded components](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/optimization-target-refs/spec.md#L72) | R: MODIFIED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/optimization-target-refs/spec.md#L55)) |
| [Module and parameter refs expand from declared components](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/optimization-target-refs/spec.md#L85) | R: MODIFIED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/optimization-target-refs/spec.md#L69)) |
| Target refs can reference accepted structural identities | M: ADDED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/optimization-target-refs/spec.md#L3)) |

### adapter-system

| Requirement | Combined target disposition |
|---|---|
| [Optimization-owned adapter targeting](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/adapter-system/spec.md#L6) | R: REMOVED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/adapter-system/spec.md#L128)) |
| [Adapter-system realization layer](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/adapter-system/spec.md#L19) | R: MODIFIED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/adapter-system/spec.md#L3)) |
| [AdapterMode owns the training-side adapter path](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/adapter-system/spec.md#L34) | R: REMOVED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/adapter-system/spec.md#L151)) |
| [Explicit adapter-type configuration](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/adapter-system/spec.md#L48) | R: MODIFIED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/adapter-system/spec.md#L26)) |
| [Adapter persistence belongs to the adapter-training path](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/adapter-system/spec.md#L63) | R: REMOVED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/adapter-system/spec.md#L140)) |
| [Broad adapter-system applicability](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/adapter-system/spec.md#L78) | Retained unchanged, including all scenarios |
| [Optimization-to-adapter boundaries stay contract-based](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/adapter-system/spec.md#L94) | R: MODIFIED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/adapter-system/spec.md#L42)) |
| [Grouping code integrates as grouping](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/adapter-system/spec.md#L111) | Retained unchanged, including all scenarios |
| [Production helpers stay typed and production-first](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/adapter-system/spec.md#L131) | Retained unchanged, including all scenarios |
| [AdapterMode remains explicit orchestration](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/adapter-system/spec.md#L147) | R: REMOVED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/adapter-system/spec.md#L162)) |
| Authored adapter intent becomes accepted participants and relationships | R: ADDED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/adapter-system/spec.md#L59)) |
| Adapter products, artifact loading, and exact restoration are distinct | R: ADDED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/adapter-system/spec.md#L82)) |
| Pipeline coordination replaces mode ownership | R: ADDED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/adapter-system/spec.md#L107)) |

### adapter-module-targeting

| Requirement | Combined target disposition |
|---|---|
| [Optimization-owned adapter module targeting](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/adapter-module-targeting/spec.md#L6) | R: REMOVED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/adapter-module-targeting/spec.md#L185)) |
| [Repo-owned adapter target provenance for module-resolved methods](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/adapter-module-targeting/spec.md#L39) | R: MODIFIED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/adapter-module-targeting/spec.md#L45)) |
| [Parameter-native optimization grouping remains the training boundary](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/adapter-module-targeting/spec.md#L59) | R: MODIFIED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/adapter-module-targeting/spec.md#L62)) |
| [First absorbed `loha` method fits the repo-owned adapter runtime seams](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/adapter-module-targeting/spec.md#L77) | R: MODIFIED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/adapter-module-targeting/spec.md#L76)) |
| [Mixed-method overlap semantics remain out of scope for this slice](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/adapter-module-targeting/spec.md#L108) | R: REMOVED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/adapter-module-targeting/spec.md#L175)) |
| [Method-local adapter config owns runtime settings translation](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/adapter-module-targeting/spec.md#L118) | R: MODIFIED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/adapter-module-targeting/spec.md#L100)) |
| [Continuation intent defaults to strict continuation](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/adapter-module-targeting/spec.md#L142) | R: MODIFIED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/adapter-module-targeting/spec.md#L123)) |
| [Adapter target refs remain compatible during migration](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/adapter-module-targeting/spec.md#L157) | R: MODIFIED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/adapter-module-targeting/spec.md#L147)) |
| [Adapter component scope derives from declared loaded components](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/adapter-module-targeting/spec.md#L169) | R: MODIFIED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/adapter-module-targeting/spec.md#L159)) |
| [Adapter target provenance preserves declared component identity](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/adapter-module-targeting/spec.md#L181) | Retained unchanged, including all scenarios |
| Governed PEFT realization resolves authored adapter target intent | R: ADDED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/adapter-module-targeting/spec.md#L3)) |
| Overlapping adapter target assignments require explicit support | R: ADDED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/adapter-module-targeting/spec.md#L31)) |

### repo-owned-lora-method

| Requirement | Combined target disposition |
|---|---|
| [Repo-owned LoRA runtime realizes from resolved adapter targets](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/repo-owned-lora-method/spec.md#L6) | R: MODIFIED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/repo-owned-lora-method/spec.md#L3)) |
| [Repo-owned LoRA method owns LoRA module behavior](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/repo-owned-lora-method/spec.md#L25) | Retained unchanged, including all scenarios |
| [Repo-owned LoRA config stays method-local and policy-light](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/repo-owned-lora-method/spec.md#L43) | Retained unchanged, including all scenarios |
| [Repo-owned LoRA trainable refs preserve target provenance](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/repo-owned-lora-method/spec.md#L59) | Retained unchanged, including all scenarios |

### repo-owned-vera-method

| Requirement | Combined target disposition |
|---|---|
| [Repo-owned VeRA runtime realizes from resolved adapter targets](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/repo-owned-vera-method/spec.md#L8) | R: MODIFIED ([delta](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/repo-owned-vera-method/spec.md#L3)) |
| [Repo-owned VeRA runtime owns shared projection state](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/repo-owned-vera-method/spec.md#L27) | Retained unchanged, including all scenarios |
| [Repo-owned VeRA method owns method-local config and artifact policy](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/repo-owned-vera-method/spec.md#L52) | Retained unchanged, including all scenarios |
| [Repo-owned VeRA trainable refs preserve target provenance](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/repo-owned-vera-method/spec.md#L79) | Retained unchanged, including all scenarios |

### New companion catalog/source/structure capabilities

No corresponding main capability exists in the examined tree. Each item below is an ADDED obligation in M; none implicitly displaces an unrelated surviving main requirement.

| Capability | Added requirement |
|---|---|
| `model-catalog-identity` | [Catalog identity separates logical models from representations and realizations](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/model-catalog-identity/spec.md#L3) |
| `model-catalog-identity` | [Portable identity is carried or deterministically derived at the correct identity layer](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/model-catalog-identity/spec.md#L37) |
| `model-catalog-identity` | [Catalog registration has one durable authority](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/model-catalog-identity/spec.md#L93) |
| `model-catalog-identity` | [Recognition policy is progressive and versioned](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/model-catalog-identity/spec.md#L131) |
| `model-catalog-identity` | [Catalog preserves typed relationships, immutable revisions, and lineage](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/model-catalog-identity/spec.md#L164) |
| `model-catalog-identity` | [Produced artifacts can carry portable catalog identity](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/model-catalog-identity/spec.md#L197) |
| `model-source-provenance` | [Every family uses one uniform loading-result schema](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/model-source-provenance/spec.md#L3) |
| `model-source-provenance` | [Materialization provenance is selection-aware, component-aware, and ordered](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/model-source-provenance/spec.md#L31) |
| `model-source-provenance` | [Provenance and runtime observations use an explicit boundary](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/model-source-provenance/spec.md#L54) |
| `model-source-provenance` | [Realization composition observations are immutable and revisioned](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/model-source-provenance/spec.md#L71) |
| `model-source-provenance` | [Materialization attempts are events rather than composition state](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/model-source-provenance/spec.md#L100) |
| `model-source-provenance` | [Composition views use semantic order rather than insertion accidents](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/model-source-provenance/spec.md#L127) |
| `model-source-provenance` | [Provenance facts remain central and family-extensible](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/model-source-provenance/spec.md#L150) |
| `model-structure-metadata` | [Model structure separates path bindings, live objects, storage, and source keys](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/model-structure-metadata/spec.md#L3) |
| `model-structure-metadata` | [Structural inventories declare dimension-scoped coverage](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/model-structure-metadata/spec.md#L36) |
| `model-structure-metadata` | [Tensor descriptors are typed and selectable](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/model-structure-metadata/spec.md#L64) |
| `model-structure-metadata` | [Live and artifact resolvers preserve source limitations](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/model-structure-metadata/spec.md#L82) |
| `model-structure-metadata` | [Model structural queries use the shared metadata query capability](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/model-structure-metadata/spec.md#L100) |
| `model-structure-metadata` | [Structural fingerprints are versioned evidence](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/model-structure-metadata/spec.md#L117) |
| `model-structure-metadata` | [Existing structural consumers converge on accepted identities](https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/model-structure-metadata/spec.md#L145) |
