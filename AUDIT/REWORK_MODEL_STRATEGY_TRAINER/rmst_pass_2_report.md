# Model–strategy–Trainer: Pass 2 — revisions, lifecycle and preparation

**Audited snapshot:** `Enferlain/sd-scripts`, requested branch `model-strategy-trainer`, detached checkout at **`6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48`**; commit message `4.6`, author time `2026-10-04T06:39:22+02:00` (`04:39:22 UTC`). Git tree: `a46f1fb0e1edc9cda5e29dec873ae0cb78f8de46`. Report date: 2026-10-04.

The GitHub plugin's recursive tree response identifies the requested commit, and the local checkout's `HEAD` identifies the same commit. `git status --porcelain=v1` was empty before substantive inspection and at final source verification. No user working-tree additions or local corrections were supplied or examined. This is an audit of that clean pinned snapshot, not the user's unseen local checkout or the current moving branch tip. The report and transfer helper are outside the checkout; repository files, task checkboxes, issue state and attachments remain unchanged.

**Result:** the audited requirements establish coherent distinctions between logical identity, accepted definition, ordinary numerical evolution, physical realization, prepared usability and historical observations. **One low-impact wording finding, P2-REV-01, remains in the restoration introduction.** It does not reopen the explicit normative prohibition on automatic new-run fallback. No additional necessary semantic gap or incompatible guarantee was established in the assigned concern. The existing **P1-REV-01** targeting applicability gap remains visible in the baseline and is referenced rather than duplicated; its reported local correction is outside this snapshot.

This report does not certify production backend support, integrated recovery, or overall G5 readiness.

## 1. Source authority and audit method

The governing design's source hierarchy was applied directly. Active design/delta requirements govern new decisions; the direction supplies inherited normative inputs; the exchange document's **SETTLED INPUT** is inherited, while its working proposals bind only where adopted. Chronological notes establish explicit supersession. Unchanged main-spec requirements remain applicable. Production code and experiments provide bounded evidence, not target authority. [Authority] [SettledInput]

The 2026-08-26 optimization and topology corrections preserve Trainer-owned ordinary mechanics and place accepted runtime behavior in the arrangement rather than the authoring strategy. G4.6 explicitly corrects incomplete same-run restoration becoming another run. These are documented corrections, not conclusions drawn from timestamps. [Corrections26] [Reconciliation]

I read the complete requirements/scenarios of `run-participant-state`, `training-runtime-preparation`, `training-optimization`, `training-contract`, `accepted-training-execution`, `training-capability-coordination`, and `training-artifact-persistence`; the rework metadata, observability, loading and targeting deltas; applicable companion catalog, provenance, structure, loading and targeting requirements; and the affected main loading, targeting, family-metadata and observability specs. Relevant design sections, inherited decisions, correction records, test fixtures and selected production code were followed for the assigned relationships. Pass 0 and the targeted Pass 1 excerpts were navigation and finding deduplication only.

The central requirements were checked as complete units rather than inferred from headings. Long supporting reads were sometimes truncated; relevant central passages were subsequently read in smaller ranges. This report does not claim full-file verification of every historical or research document.

## 2. Identity, revision and dependency map

These are different meanings, not a demand for separate classes, global counters or revision tuples on every object. The design explicitly permits evidence through a generation, snapshot or justified narrower dependencies. [D6] [D7] [OptimizationDependencies] [RestorationDesign]

| Meaning and scope | Establishing owner | What changes it / what preserves it | Consumers and source |
| --- | --- | --- | --- |
| Authored participant/relationship address and role | Explicit strategy authoring establishes declarations and uses. | Reusing an address does not reuse a retired incarnation. Role, type, source and object equality do not define identity. | Fulfillment, transition proposals, diagnostics; RS “Authored address and runtime incarnation are distinct.” [RS] |
| Logical run authority | Fulfillment seeds the accepted authority under one contract/version/profile. | Complete same-run restoration preserves it; separately requested new run establishes another, with lineage. A process/session is not the run. | Every authority-qualified consumer; TC “One accepted contract governs one run authority”; RS identity/lineage. [TC] [RS] |
| Participant incarnation | Run authority establishes an independently addressable semantic participant. | Materialization, compatible replacement, wrapping, casting, compilation and preparation preserve it. Retirement closes use permanently; incompatible replacement/redeclaration requires a new incarnation. | Execution, optimization, capabilities, products, history. [RS] [D6] |
| Relationship incarnation and operational revision | Run authority establishes an independently addressable effect and accepts its transitions. | Activation/deactivation changes relationship state/revision without creating or retiring endpoints. Endpoint replacement/retirement can also invalidate relationship-dependent preparation. | Routes/views, unit dependencies, persistence and history. [RS] [RPRelationships] |
| Binding, route and view state | Authority publishes candidate bindings and scoped route/view guarantees. | Compatible physical replacements can advance affected revisions without changing the participant. Several invocations are not several bindings; inspection/unwrapped/artifact views are not extra forward routes. | All live projections and readiness checks. [RS] [D6] |
| Arrangement and applicable obligation revision | Fulfillment derives obligations; authority evaluates permitted complete-state amendments under the same contract/profile. | Changed accepted meaning re-evaluates obligations atomically. Physical bindings can remain unchanged while old evidence becomes inadmissible. | Initial realization, transitions, preparation and use; no downstream reinterpretation. [D7] [TC] [RPStale] |
| Ordinary numerical, schedule and adaptive state | Optimization, selected behavior, input and capability owners advance their own sections under accepted rules. | Ordinary advancement/bounded choices preserve participant and unit-definition identity. They can invalidate produced values under their dependency policy without automatically rebuilding preparation. | Subsequent decisions, produced-data admission, capture and continuation. [ComposedState] [G35] [CacheFreshness] |
| Optimization-unit address, incarnation and accepted-definition revision | Accepted semantic plan establishes independently managed responsibilities; optimization realizes them within governed coordination. | Changed subjects, optimizer-significant groups, policies or dependency definitions revise a continuing unit. Split/merge/retire/recreate establishes new units. New physical handles alone do neither. | Contributions/windows, advancement, preparation, products/recovery and reports. [OUIdentity] [G35] |
| Mutable optimizer/scheduler/window state | Trainer optimization under standard ownership; an extension transfers only its specified duties. | Continuity after replacement/replanning requires accepted preserve-with-evidence, migrate, reset or unavailable treatment. Unit identity is insufficient proof. | Advancement and exact restoration; pending contributions cannot silently cross. [OUContinuity] |
| Prepared generation / dependency evidence | Preparation coordinator assembles evidence; authority, optimization and backend owners install their respective surfaces together. | A relevant broken dependency removes current usability. Precise evidence can spare unrelated state; full-snapshot evidence conservatively cannot. Physical existence is not readiness. | Execution and capabilities; evidence can be indirect rather than stored in every runtime. [RP] [OptimizationDependencies] |
| Backend composite, replica and inseparable group | Trainer/runtime infrastructure owns coordination state. | Returned composites do not merge participants or units. Affected inseparable groups lose usability as a whole; later jobs cover the full overlap closure. | Preparation, synchronization and actual-use protection. [RPIdentity] [RPGroups] |
| Preparation attempt | Coordination derives one job from coherent state/obligations. Destructive entry is authority-recognized. | Multiple ordered calls remain one attempt. Destructive withdrawal advances freshness while its distinct coordination identity preserves a completion basis. That identity does not waive other current obligations. | Candidate validation, publication, failure/recovery; older optimistic results cannot restore guarantees. [RPDestructive] [RPStale] |
| Work, invocation, request and execution session | Each accepted activity's owner supplies its facts; run coordination preserves required joins. | Multiple inputs can feed one attempt; production may predate it. Same-run restoration creates a new session/attempt identity; matching old addresses does not admit late results. | Admission, failure correlation, metadata and restoration. [Input] [Correlation] [RestoreReadiness] |
| Realization and composition observation revision | Metadata owns durable qualification, allocation and validation; accepted publication supplies origin truth. | Stable realization has immutable ordered observations. Loader attempts do not become composition. Filing/retry does not redefine accepted ordering. Ordinary weight evolution is separate capture provenance. | Latest/final views, structural associations and artifact provenance. [FM] [Composition] [CompanionComposition] |
| Checkpoint-scoped composition finalization | Accepted preparation publication establishes composition for a named readiness checkpoint; metadata records that correspondence. | Later transitions append history without erasing earlier finals. An old final does not prove a later checkpoint ready. | Consumers select the requested checkpoint/captured scope, not any final marker. [Composition] [FM] |
| Catalog lineage, immutable model revision, representation and selection | Central durable catalog authority applies versioned evidence/recognition policy. | Equal complete bytes establish representation equality under policy, not EMA/non-EMA selection, live incarnation or readiness. Execution-only casts/moves/wrapping preserve source identity; persisted changes get supported lineage/revision treatment. | Loading/provenance and optional structural associations; no catalog prerequisite for an otherwise valid live projection. [Catalog] [Provenance] [TargetsCompanion] |
| Structural path binding, observed object/storage and source key | Metadata qualifies model paths/inventories; accepted live consumers resolve current objects. | Different paths can alias one observed parameter without becoming one durable path identity. Source/runtime mappings may be non-bijective; object/storage evidence is observation-local. | Alias-aware realization, grouping and resource evidence; not semantic participant deduplication. [Structure] [TargetsCompanion] |

**Important boundary:** a prepared route can remain current across ordinary weight advancement while a cached encoding produced from earlier weights becomes inadmissible. Conversely, a new compiled route can invalidate a physical runtime while a stored encoding remains reusable if the accepted computation/numerical dependencies are evidenced equivalent. Neither conclusion requires one universal “model version.” [G35] [CacheFreshness] [CacheRealization]

The shared structural-target contract's component-free applicability remains the already reported **P1-REV-01**. The baseline permits independent learned participants but some shared target scenarios require family-component ancestry. I did not fabricate such ancestry in this map or treat the reported out-of-baseline correction as present. The lifecycle distinctions themselves do not resolve that separate target applicability gap. [Targets] [Loaded] [OUIdentity]

## 3. Finding

### P2-REV-01 — Low: restoration introduction retains an unqualified new-run consequence after missing restore facts

**Classification:** wording ambiguity / concrete drift risk, concerning semantic intent. It is not an unresolved recovery API, evidence demand or new architecture. It follows **P0-INV-01**; Pass 1 routed that lead onward without issuing a separate finding.

**Exact source relationship:**

1. `openspec/changes/rework-model-strategy-trainer/design.md`, **G4.4, introductory restoration identity paragraph**, lines **2267–2275**, couples missing restored facts and artifact initialization: “If those facts are not restored—or an artifact starts another run—the new run establishes new identities and records lineage instead of claiming continuity.” [RestoreIntro]
2. The same design's **G4.6 reconciliation**, lines **256–259**, expressly corrects older wording: missing required identity/revision state makes a same-run request incomplete; new identities need a separately requested run. G4.4's **Restore request, owner reconstruction, and readiness result**, particularly lines **2400–2408**, also prohibits automatic fallback. [RestoreCorrection] [RestoreExplicit]
3. `specs/run-participant-state/spec.md`, **“Runtime identity and lineage remain separate”** and **“Same-run restoration lacks required identity facts”**, lines **160–181**, requires incomplete restoration and unavailable dependent execution, not invented identities or silently initialized state. [RSRestore]
4. `specs/training-capability-coordination/spec.md`, **“Failed restoration cannot expose a mixed continuation”**, lines **557–577**, requires a separate explicit new-run request. [RestoreFailure]
5. Historical `docs_design/models-strategy-trainer/notes.md`, **“Durable participant identity and lineage settled (2026-08-25)”**, lines **840–847**, contains the older fallback wording. It is historical and the active correction governs; it is not independent permission to fall back. [OldRestore]

**Concrete interpretation risk:** a caller requests same-run restoration from weights and optimizer state but lacks required incarnation/revision facts. An implementer relying on the G4.4 introduction could mint new identities with lineage and proceed. That changes the requested operation and can discard required continuation semantics. The complete normative sources require reporting incompleteness and gating the dependent execution instead.

**Why this is low impact:** source authority and the explicit G4.6 correction resolve the intended semantics. Reading the whole requirement does not leave a legitimate automatic-fallback option. The defect is the retained unqualified introductory branch, which can mislead a partial reader or be copied into a public resume explanation. It is not a new necessary semantic blocker.

**Smallest plausible correction:** limit the introduction's new-identity consequence to **a separately requested new run** and state that a same-run request missing required facts remains incomplete. No types, recovery mechanism, policy or task completion need change. Neighboring identity/lineage, failed-restoration, metadata session and artifact-initialization requirements should retain their existing separation. Historical wording can remain historical when its supersession is clear; it should not be promoted back into active documentation.

**No corrections were applied.**

## 4. Concrete transition traces

Example numbers below illustrate established semantics; they are not required counters or production fields. These are source-backed derivations, not claims that the full traces execute in the current Trainer.

### A. Frozen participant needing preparation, with several uses

Declare frozen participant `E`, required by accepted execution, and trainable participant `A`. Fulfillment records required routes, derivative meaning and readiness checkpoints. Optimization can exclude `E`; preparation still includes it if movement/casting/wrapping/compilation is required. If derivatives to `A` traverse `E`, freezing does not permit detachment. [RPJob] [Differentiated]

Training and validation through the same prepared callable share its route. A separately compiled sampling callable may be a second named route with refresh dependencies; an unwrapped persistence handle is a view. Each invocation keeps its use/gradient meaning and all representations keep `E`'s incarnation. Unsupported concurrent mutable view selection must serialize/restore or fail readiness. [RSRoutes] [ParticipantUses]

**Consequence:** execution/preparation membership, optimization membership and individual uses are distinct. One participant with several representations is not several independently evolving participants. No additional identity mechanism is needed to derive this trace.

### B. Compatible replacement while an optimistic candidate is in flight

1. Attempt `J` resolves participant `M`, binding 4, obligation 7, and unit `U`, definition 3.
2. Authority accepts compatible replacement at binding 5, preserving `M`'s incarnation. If `U`'s semantic subjects/groups/policy remain unchanged, it also preserves `U`'s identity and definition 3.
3. Affected routes, unit runtime and any inseparable group lose current usability. Old optimizer state still requires an accepted continuation treatment; matching unit identity does not validate momentum or gradients.
4. `J` returns after replacement. Its binding-4 evidence cannot publish against binding 5. It must be rejected or explicitly re-resolved, then verified as a current candidate.
5. Dependent work resumes only after complete current preparation and the selected state-continuity treatment succeed. [RSChanges] [RPStale] [OUContinuity] [G35]

If the replaced state was still required by an outstanding derivative, publication cannot jump from step 1 to step 2 on freshness alone. It waits, rejects, or uses supported isolation through that actual use. This is additional protection, not another definition revision. [Protection] [RPPublish]

### C. Parameter surgery or membership amendment with pending contributions

A unit with an accepted dynamic-substructure selector can resolve new physical parameters after allowed structure-preserving surgery at the **same definition revision**. Its physical members/aliases/trainability/backend realization must nevertheless be resolved again. Changing the accepted selector, optimizer-significant grouping, policy or semantic dependencies revises the continuing unit. Splitting the responsibility into independent units creates new incarnations. [OUIdentity] [OptimizationDependencies] [G35]

Before the transition takes effect, pending gradient-window contributions follow the accepted complete/preserve/discard treatment. The owners separately establish preserve/migrate/reset/unavailable treatment for optimizer, scheduler and backend state. Old gradients cannot silently enter the new runtime, and a reordered parameter list is not state correspondence evidence. If no accepted continuation rule is established, dependent work waits or remains unavailable. [OUContinuity]

**Consequence:** identity continuity and mutable-state continuity can have different answers without contradiction. Neither automatic momentum migration nor transactional optimizer rollback follows.

### D. Preparation overlapping an inseparable group

Suppose current backend group `G1` covers `M` and `A`; another covered group `G2` covers `A` and `B` where the backend supports that arrangement. A later job touching `M` must include `A`; that overlap in turn requires `B`. This closure follows “include every member of an overlapped inseparable group,” not a new minimal-invalidation algorithm. Disjoint unrelated groups need not be included under justified precise dependencies. [RPGroups] [G34] [G35]

The backend may return a composite handle, but `M`, `A`, `B` and their units retain separate semantic identities. Retiring one member evicts its group and removes surviving dependent usability until a coherent rebuild. A destructive job omitting an overlapped member is rejected before mutation. [RPIdentity] [RPGroups]

**Evidence limit:** the historical spike checks fake joint coverage, retirement eviction and pre-mutation destructive omission rejection. Neither it nor the preparation-order fixture proves arbitrary real overlapping-group integration or rank agreement.

### E. Stale candidate after obligation or relationship amendment

`J` has valid physical parameters but was derived under obligation 7. A permitted arrangement amendment publishes obligation 8. Unchanged bindings do not make `J` current: its evidence must be judged against the current accepted obligations, and the superseded candidate must be rejected or explicitly re-resolved. A relationship revision similarly invalidates dependent views whether caused directly, by endpoint replacement or by retirement. [RSAtomic] [RPStale] [RPRelationships]

A pre-existing precise projection can survive an unrelated transition only with evidence that its applicable obligations and inseparable-group dependencies remain satisfied. A projection declaring only the complete source snapshot becomes stale. “Precise” is not permission to ignore an obligation change. [RSFreshness] [OptimizationDependencies]

### F. Two backend construction orders, one publication attempt

| Physical order inside the coherent attempt | Evidence needed before publication |
| --- | --- |
| Resolve semantic members → construct optimizer/scheduler → transform/wrap participants | Optimizer targets final prepared members, or an evidenced rebind establishes that correspondence. |
| Transform/wrap participants → resolve members using transformation provenance → construct optimizer/scheduler | Post-transform objects preserve the accepted semantic subjects, groups, aliases and policies. |

Both include frozen execution participants and required infrastructure; both preserve unit identity/definition. Same-unit/group aliases consolidate with provenance, while standard cross-unit/group overlap is rejected. Several backend calls remain one attempt. A backend need support only the ordering/policies it can evidence. [RPCorrespondence] [G34]

Candidate assembly, backend calls, rank agreement, evidence collection, obligation evaluation and freshness checks precede final visibility. The coordinator installs the verified authority route/view surfaces and Trainer-owned optimization/backend surfaces without exposing a partial prepared result. A component hook or fallible observer cannot be deferred into that final installation. [RPPublish]

### G. Replacement-only failure versus destructive failure

**Replacement-only:** build unpublished candidate state off to the side. Failure discards it. Prior state remains usable only if its guarantees independently still hold; “no new result published” does not override a concurrent invalidation. [RPReplacement]

**Destructive:** before permitted in-place mutation, establish the recognized attempt and withdraw affected guarantees. The withdrawal changes freshness, but its distinct attempt identity allows its own completion to be evaluated rather than invalidating itself merely because it withdrew source usability. Other amendments and obligations still matter. Failure leaves affected use unavailable; restoring flags is not restoration of contents. [RPDestructive]

An older optimistic candidate cannot restore the withdrawn state during mutation or after failure. Relationship state cannot change while an endpoint is under destructive preparation. Authorized recovery/replacement can still occur under its accepted path; gating ordinary use does not prohibit recovery. [RPDestructive] [RPRelationships] [Protection]

**Consequence:** one atomic final publication does not imply rollback of earlier external mutation. The destructive safety transition is intentionally earlier and distinct.

### H. Encoder weights change while produced inputs are in flight

An encoder `E` can remain the same participant/binding and retain prepared execution while ordinary weight state advances from `S1` to `S2`. A produced value records what actually computed it, not merely the requested launch state or whatever is current on completion. [CacheFreshness] [Input]

If production uses protected `S1`, a result finishing after advancement remains an `S1` result. If explicitly permitted production spans states, its provenance associates relevant portions with the actual states; a single launch version must not relabel the whole value. Provenance does not authorize mixed-state work. Publication and each consumer's admission apply their current accepted schema/freshness/lag/gradient obligations. Old-state values can remain physically stored or admissible under accepted lag while failing a current-state guarantee. [Input] [CacheAdmission]

The required source/storage access stays protected through actual use. A detached cache value cannot replace a live derivative path through a trainable encoder. Upstream frozen cache can remain valid when only a downstream adapter evolves. No universal numerical counter, binding revision per update or proactive invalidation broadcast is required. [CacheFreshness] [Protection]

### I. Older composition evidence arrives after newer publication

Authority publishes accepted composition states `C2` then `C3`. Metadata receives `C3` first and `C2` later. The observed accepted-state correspondence remains `C2` before `C3`; arrival must not assign `C2` a valid revision that makes it appear latest. Retry also preserves fact identity instead of allocating a second semantic transition. [FMHistory] [Composition] [CompanionComposition]

“Highest valid accepted observation revision” operates together with origin correspondence, not independently of it. If the ordering/correspondence cannot be established, the view reports ambiguity or incompleteness. The implementation can choose how to reserve/validate ordering or handle a gap; the sources do not permit using arrival order to fill that gap. [FMHistory] [Composition]

Loader success or an unaccepted candidate is attempt evidence and allocates no accepted composition. Required filing failure after live publication leaves that transition reached and durable coverage incomplete; retrying metadata does not rerun publication. [FM] [Provenance]

### J. Finalized checkpoint followed by transition and later artifact capture

Accepted preparation establishes composition `C2` at readiness checkpoint `K1`; its final observation remains immutable. Later accepted replacement produces `C3`. `C2`'s final marker does not make `C3` prepared for a later checkpoint. A consumer asking for a finalized checkpoint must select its corresponding observation or receive incomplete/unavailable context. [Composition] [FM]

An artifact captured at `C2` retains `C2` and `K1` where required even if serialization/reporting finishes after `C3`. An artifact captured after the transition uses its actual captured composition and any finalized context required by that product; it cannot borrow `K1` merely because that is the latest filed final. Ordinary weight updates without topology changes are recorded through supplied capture state provenance rather than manufactured composition revisions. [Composition] [Capture] [FMHistory]

**Consequence:** finalization fixes a checkpoint's composition observation, not all future run state or numerical content. Detailed query keys and schema integration belong to Pass 6/G5; no specific checkpoint-ID representation is required here.

### K. Granted two-pass work and safe handback

Fulfillment accepts the narrow region's parameter/gradient scope, phase owners, quiescent-window rule, effects and recovery meaning. Preparation must establish scoped access, completion, synchronization/scaling, retention and restoration evidence. A backend lacking that support fails readiness before entry; acceptance does not promise universal execution. [Grant] [RPGranted]

The region performs its two backwards, temporary edit, intermediate gradient treatment and restoration. Trainer retains final clipping/advancement. Protection continues or transfers without a gap through handback verification and retained mechanics. A clean parameter tensor or successful callback is not proof that all relevant master/sharded/offloaded state is restored, the final gradient is valid, or device work has completed. [Grant] [Protection] [WorkedSAM]

Failed second pass, failed restoration and uncertain handback do not authorize advancement. Even successful parameter restoration after failure leaves other reached effects/gradients subject to recovery. Capability readers and exact snapshots cannot consume the uncertain affected state. Independent work can continue only where its dependencies and accepted failure policy permit it. [Grant] [RestoreCut]

**Evidence limit:** the local SAM fixture checks phase ownership, one CPU parameter, produced gradient and local failure flags. The worked proposal expressly leaves concrete scaler, gradient and real-backend handback evidence unfinished. This is a declared support boundary, not a claim that checking those flags solves backend safety.

### L. Same-run restoration in a new session

A complete supported snapshot preserves logical authority, participant/relationship incarnations, accepted arrangement/obligations, unit definitions and required owner state. Restoration reconstructs objects while affected execution is unavailable, validates semantic correspondence and establishes fresh physical readiness in a new session. Saved handles and old attempts do not become current merely because semantic identities match. [RSRestore] [OUIdentity] [RestoreReadiness]

A late result from the earlier session may be adopted only under an accepted recovery/admission rule establishing actual provenance and outcome correspondence. Missing identity or required state blocks same-run continuation; it does not silently reset state or establish another run. An optional sampling route can remain unready while independently satisfied training resumes. [RestoreReadiness] [RestoreFailure]

Process/rank loss during preparation installation aborts the in-process run; it cannot expose partial current state to surviving work or claim in-process rollback. A later continuation follows restoration. Detailed capture/recovery protocols remain Pass 5. [RPProcess]

## 5. Named Pass 0 leads and neighboring findings

| Lead | Disposition in this pass | Evidence and limit |
| --- | --- | --- |
| **P0-INV-05 — composition order versus delayed filing** | **Semantically resolved; no new finding.** | Provenance views require validated origin/revision correspondence; G4.5 and companion design expressly exclude arrival order and duplicate semantic transitions. Missing correspondence means ambiguity/incompleteness. Allocation/query implementation remains G5/metadata work; no global clock or reserved ordinal algorithm is assumed. [FMHistory] [Composition] [CompanionComposition] |
| **P0-INV-08 — backend support and safe handback** | **Semantically resolved, with unproven implementation support explicitly retained.** | RP requires differentiated/granted-work readiness evidence and rejection of unsupported backends; EX requires actual-use protection through retained mechanics. G3.2 and the worked case acknowledge fixture/scaler/backend limitations. There is no requirement to certify every backend. [RPGranted] [Grant] [Protection] [WorkedSAM] |
| **P0-INV-01 — recovery wording** | **Low wording finding P2-REV-01; normative intent resolved by explicit correction.** | Same-run incompleteness cannot automatically become new-run intent. Detailed capture, mapping and recovery cases remain Pass 5. [RestoreCorrection] [RSRestore] [RestoreFailure] |
| **P0-INV-03 — ordinary owned state versus governed transition** | **Pass 1's distinction independently corroborated; not reopened.** | Composed-state owners and G3.5 separate bounded schedule/adaptive/optimizer evolution from changes to accepted binding/definition/obligation meaning. Temporary effects can gate use after uncertain cleanup without manufacturing a topology revision. [ComposedState] [G35] [ParticipantUses] |
| **P1-REV-01 — component-free structural targeting** | **Existing finding only; local correction not audited.** | Baseline targeting still contains the identified component applicability wording. Optional catalog association does not supply missing family ancestry. Its lifecycle/alias implications should remain visible in later integration, but no duplicate P2 finding is issued. [Targets] [Loaded] [TargetsCompanion] |

## 6. Evidence, verified areas and limitations

### Verification performed

- Confirmed the pinned checkout and clean tracked/untracked state; read `AGENTS.md`, `DEVELOPMENT_GUIDE.md` and the top of `CHANGELOG.md`. Repository review requirements apply to significant changes/implementation milestones; this read-only audit changes neither.
- Ran `PYTHONDONTWRITEBYTECODE=1 timeout 60 python docs_design/models-strategy-trainer/executable_spike/scenarios.py`: **28 tests passed**. This rerun supports local binding/identity, stale results, destructive-withdrawal, fake group and history/lineage behavior only. [Spike]
- Read `test_preparation_exchange_experiment.py` completely: both physical construction orders, frozen membership, same-unit aliases, optimizer/final-member correspondence, stale source and incomplete candidate checks are present. Publication is modeled as one local replacement; only one execution group per unit is modeled. [PreparationFixture]
- Read `test_imperative_authority_experiment.py` completely: phase ownership, pre-mutation backend/window rejection, second-pass failure with restoration, failed restoration and missing-gradient handback are covered locally. No exclusion service, asynchronous completion or snapshot capture is demonstrated. [GrantFixture]
- Inspected production realization filing, pre-write artifact fact assembly and resume hooks for the specific migration claims: current realization filing uses the session and is retained once; artifact facts are filed before the write result; resume delegates backend loading and reads epoch/step. Those paths corroborate migration needs, not target semantics. [CurrentRealization] [CurrentArtifact] [CurrentResume]

### Verified source-level areas

Participant retirement, relationship deactivation and freezing are separate; routes/views/invocations do not create competing incarnations. Optimization-unit definitions survive physical re-realization while mutable-state continuity remains a separate obligation. Ordinary numerical/adaptive progression does not automatically require preparation rebuilding, but actual producer dependencies govern reuse. Complete joint publication, explicit destructive withdrawal, group-wide invalidation and no post-publication observation rollback are all recorded. Historical composition, checkpoint finalization and actual artifact capture remain distinguishable. Same-run identities survive a new session while old physical readiness does not. [RS] [OUContinuity] [G35] [RP] [FM] [Capture] [RestoreReadiness]

### Limits

The environment's default Python and supplied primary Python both lacked `pytest`. Consequently the preparation-order and torch SAM pytest fixtures were **inspected, not rerun**. No dependencies were installed and no substitute test harness was used. Their historical passing status was not recertified by this audit.

No real Accelerate/DeepSpeed/FSDP/scaler integration, GPU/device completion, concurrent readers/publication, separate-store visibility, distributed agreement, rank-kill injection, production fulfillment, integrated runtime snapshot or restoration was executed. Passing fake-backend tests cannot establish those guarantees. No new benchmark, OpenSpec validation, review-mcp invocation or Beads completion-history audit was performed. Checked tasks are completion claims; tasks 3.4/3.5 concern architectural exchange/continuity meanings, so lack of production backend integration is not alone evidence that those tasks are falsely complete. [G34] [G35] [Tasks]

Not every research recipe, main adapter-method scenario, historical exchange register entry or upstream framework was re-audited. The read-only concern scope is not a general production implementation review. No supported numerical equality, automatic recovery, atomic optimizer update, replay guarantee or precise minimal invalidation was supplied as an unstated assumption.

## 7. Follow-up routing

| Destination | Concrete question to retain |
| --- | --- |
| **Pass 3 — execution/optimization** | Verify remaining derivative state, scaler/unscale/sync semantics, pending-window transitions, reached-call outcomes and protection through retained mechanics for supported profiles. Readiness must reject an unsupported combination. |
| **Pass 4 — independent activities/capabilities** | Check admission/read/protection lifetimes, capacity/backpressure, cancellation and source/storage overlap with numerical evolution and temporary effects. Mixed-state attribution does not itself grant mixed-state consistency. |
| **Pass 5 — capture/recovery** | Carry P2-REV-01; examine coherent cuts, contributor mapping, new-session readiness, old-attempt adoption and missing identity/state without default reset. Verify no capture of unsupported destructive/perturbed state. |
| **Pass 6 — metadata integration** | Verify actual allocation/validation and queries preserve accepted order under late delivery, duplicate retry and missing required history; checkpoint selection and product capture must survive later transitions. Follow existing P1-REV-01 without assuming its local fix is in this baseline. |
| **Pass 7 — combined cases** | Combine pending-window stage change, independent producer work, scoped research authority, finalized capture and failed restoration under one whole-run arrangement. This pass's constituent traces do not certify an integrated run. |

Concrete storage, API, locking, generation representation, checkpoint keys, adapters and backend publication protocols remain deliberately deferred G5 choices. They become defects if they fail the recorded meaning; their absence from the planning snapshot is not itself another finding. No overall G5-readiness verdict is issued.

## Source references

All repository references below are pinned to the audited commit. Requirement and scenario names in the report identify the relevant units; file-wide links are used where several complete requirements contribute.

[Authority]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/design.md#L19-L68
[Reconciliation]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/design.md#L140-L274
[RestoreCorrection]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/design.md#L256-L259
[D6]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/design.md#L1069-L1139
[D7]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/design.md#L1140-L1186
[ComposedState]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/design.md#L870-L933
[G34]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/design.md#L1483-L1570
[G35]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/design.md#L1571-L1701
[Grant]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/design.md#L1383-L1482
[RestorationDesign]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/design.md#L2258-L2421
[RestoreIntro]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/design.md#L2267-L2275
[RestoreExplicit]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/design.md#L2400-L2408
[SettledInput]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/docs_design/models-strategy-trainer/strategy_contract_exchange_design.md#L296-L329
[Corrections26]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/docs_design/models-strategy-trainer/notes.md#L1098-L1247
[OldRestore]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/docs_design/models-strategy-trainer/notes.md#L840-L847
[RS]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/run-participant-state/spec.md
[RSChanges]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/run-participant-state/spec.md#L52-L71
[RSRoutes]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/run-participant-state/spec.md#L72-L89
[RSAtomic]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/run-participant-state/spec.md#L113-L131
[RSFreshness]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/run-participant-state/spec.md#L132-L144
[RSRestore]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/run-participant-state/spec.md#L160-L181
[RP]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-runtime-preparation/spec.md
[RPJob]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-runtime-preparation/spec.md#L10-L43
[RPIdentity]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-runtime-preparation/spec.md#L44-L78
[RPCorrespondence]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-runtime-preparation/spec.md#L79-L102
[RPGranted]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-runtime-preparation/spec.md#L103-L121
[RPPublish]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-runtime-preparation/spec.md#L122-L144
[RPReplacement]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-runtime-preparation/spec.md#L145-L154
[RPDestructive]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-runtime-preparation/spec.md#L155-L180
[RPStale]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-runtime-preparation/spec.md#L181-L199
[RPGroups]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-runtime-preparation/spec.md#L200-L213
[RPRelationships]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-runtime-preparation/spec.md#L214-L224
[RPProcess]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-runtime-preparation/spec.md#L225-L233
[OUIdentity]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-optimization/spec.md#L9-L51
[OptimizationDependencies]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-optimization/spec.md#L132-L159
[OUContinuity]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-optimization/spec.md#L160-L176
[TC]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-contract/spec.md
[Input]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/accepted-training-execution/spec.md#L254-L308
[Correlation]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/accepted-training-execution/spec.md#L309-L348
[Protection]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/accepted-training-execution/spec.md#L439-L463
[Differentiated]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/accepted-training-execution/spec.md#L112-L137
[ParticipantUses]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/accepted-training-execution/spec.md#L138-L160
[CacheRealization]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-capability-coordination/spec.md#L81-L118
[CacheFreshness]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-capability-coordination/spec.md#L119-L159
[CacheAdmission]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-capability-coordination/spec.md#L160-L251
[RestoreCut]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-capability-coordination/spec.md#L443-L473
[RestoreReadiness]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-capability-coordination/spec.md#L524-L556
[RestoreFailure]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-capability-coordination/spec.md#L557-L577
[FM]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/model-family-metadata/spec.md
[FMHistory]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/model-family-metadata/spec.md#L126-L154
[Composition]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/model-source-provenance/spec.md#L71-L149
[Provenance]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/model-source-provenance/spec.md
[CompanionComposition]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/design.md#L186-L225
[Catalog]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/model-catalog-identity/spec.md
[Structure]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/model-structure-metadata/spec.md
[TargetsCompanion]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/optimization-target-refs/spec.md
[Targets]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/optimization-target-refs/spec.md
[Loaded]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/loaded-model-components/spec.md
[Capture]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-artifact-persistence/spec.md#L125-L156
[WorkedSAM]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/research/trainer-architecture-concrete.md#L375-L469
[Tasks]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/tasks.md
[Spike]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/docs_design/models-strategy-trainer/executable_spike/scenarios.py
[PreparationFixture]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/tests/unit/training/test_preparation_exchange_experiment.py
[GrantFixture]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/tests/unit/training/test_imperative_authority_experiment.py
[CurrentRealization]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/library/training/runners/trainer.py#L737-L768
[CurrentArtifact]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/library/training/runners/trainer.py#L529-L579
[CurrentResume]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/library/training/checkpointing.py#L45-L62
