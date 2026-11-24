# Model–strategy–Trainer: Pass 3 — execution, optimization and failure

**Audited snapshot:** `Enferlain/sd-scripts`, requested branch `model-strategy-trainer`, clean detached checkout at **`6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48`**. Commit message: `4.6`; author/committer timestamp: `2026-10-04T04:39:22Z`. Git tree: `a46f1fb0e1edc9cda5e29dec873ae0cb78f8de46`. Report date: 2026-10-04.

GitHub's Git-commit endpoint and the checkout identify the same commit and tree. `git status --porcelain=v1` was empty before substantive inspection and at final source verification. No user working-tree additions or local corrections were supplied or incorporated. The report is outside the checkout. Repository files, tasks, issues and attachments were not changed.

**Result: no new P3 correctness finding was established within the coverage below.** The governing execution requirements preserve whole-run composition, selected dynamic behavior, differentiated state lifetimes, unit-local windows and outcomes, and genuinely transferred region authority. Their supporting experiments establish substantially less, but the governing records expressly retain those limitations and future conformance obligations. This is a source-level conclusion, not certification of production support or overall G5 readiness.

Two inherited findings remain applicable in this baseline: **P1-REV-01**, the shared-target applicability gap for component-free participants, affects optimization member resolution; **P2-REV-01**, residual restoration-introduction wording, must not create automatic new-run fallback. Their reported local corrections are outside this audit. No duplicate P3 finding is issued.

## 1. Authority and method

The active design and complete delta requirements/scenarios govern new decisions. The direction supplies inherited normative inputs; exchange **SETTLED INPUT** supplies inherited identity and ownership constraints. Working proposals, research, experimental APIs and current production code do not become target authority. The documented 2026-08-26 optimization/topology corrections supersede blanket strategy-owned optimization and ordinary active-strategy runtime collaboration. G4.6 explicitly corrects incomplete same-run restoration becoming a new run. These relationships follow documented decisions, not modification times. [Authority] [Corrections] [Settled] [Reconciliation]

The general protocol and pass 3 assignment were read directly from their supplied scratch copies. Pass 0 supplied navigation. Targeted excerpts from the existing pass 1 and pass 2 reports supplied finding deduplication and pass 2's disposition of P0-INV-08; their claims were independently checked against pinned sources where used. Those reports were not treated as governing requirements.

Central requirement sets were read as complete requirements and scenarios: `accepted-training-execution`, `training-optimization`, `training-contract`, `training-runtime-preparation`, `run-participant-state`, `training-capability-coordination`, `training-artifact-persistence` and the observability delta. D5's responsibility frame, D9/G3.1–G3.7, D10, G2.5/G2.6 and relevant worked cases were followed across owners. [D10] [AP] Relevant unchanged main targeting/grouping/observability obligations and companion structural-target requirements were included. [MainTargets] [MainAdapters] [MainTargeting] [MainOB] [CompanionTargets] Long supporting reads were sometimes truncated; central relied-on passages were subsequently read in smaller ranges. Coverage is not a claim to have read every historical or research file completely.

## 2. Execution owners and handoffs

These are responsibility boundaries. They do not require one class per row, a universal event log or a fixed input → action → optimizer pipeline. [D5] [State] [G36]

| Meaning or action | Owner | Handoff and retained constraint |
| --- | --- | --- |
| Accepted algorithms, policies, bounds and effects | Explicit authoring selects them; contract fulfillment accepts them and derives obligations. | Selected executable behavior survives fulfillment. Ordinary execution does not query the authoring role for established choices. |
| Participant/relationship identity, bindings, routes, views and governed revisions | One accepted run authority. | Consumers receive scoped coherent projections. Selected work proposes canonical changes; it does not publish them directly. |
| Runtime realization/preparation | Training-side coordinator; authority, optimization and backend owners retain their surfaces. | One complete verified publication. Frozen execution participants are included when required; optimizer membership is not preparation membership. |
| Due selection and independent activity coordination | Trainer/run coordination invokes the accepted policy; the policy owner updates its own state. | Different clocks and waiting/completion boundaries remain possible. A due choice is not invocation or completion. |
| Live input selection, packing, private buffers and continuation | Selected input provider. | Consumer coordination admits exact work under its obligations; ready, handed-off and consumed remain distinct. Cache production has a separate traversal owner. |
| Model/objective/custom computation and ordinary adaptive state | Accepted selected implementation. | Produces computation, observations, state/effect facts, transition proposals and, when applicable, addressed optimization offers. No offer is a completed contribution. |
| Differentiated numerical work | Executor assigned by the accepted profile, realized through supported framework/backend behavior. | Preserve demands/seeds, routes/cuts, destinations/windows, retained state or recomputation, completion and release. Framework internals may remain private. |
| Standard windows, sync/scaling, backward, clipping, optimizer/scheduler calls and zeroing | Trainer optimization. | The accepted policy determines scope and order. Each unit retains contribution, readiness, reached-mechanic and progress facts. |
| Granted backward/edit/intermediate-reset phases | Accepted region executor, for exactly its accepted scope. | Trainer stops owning those phases there and retains everything not transferred. A produced-gradient handback must not trigger another standard backward. |
| Required feedback versus optional logging | Algorithm/state owner consumes feedback; observation infrastructure formats/routes its logged copy. | Dropped telemetry does not drop feedback. Observation retry cannot repeat computation, advancement or adaptive updates. |
| Failure coordination and safe availability | Responsible owners supply known outcomes/effects; run coordination gates dependent work. | Missing success does not prove unchanged state. Released access does not make uncertain state usable. Authorized recovery remains possible. |
| Products and runtime recovery | Respective pipeline coordinators and domain contributors. | Product capture/publication, coherent runtime snapshot and fresh restore readiness are separate outcomes. |

Sources: EX “Accepted execution survives the authoring object,” “Accepted execution covers the whole run,” “The standard profile retains generic Trainer mechanics,” “Cross-owner action facts remain correlatable”; OU “Trainer owns standard realization and mechanics” and “Advancement authority is profile-defined”; RP “Preparation results separate ownership surfaces”; CC/OB feedback and continuation requirements. [EX] [OU] [RP] [CC] [OB]

## 3. Findings and inherited boundaries

### New P3 findings

None established. In particular, missing locking algorithms, completion-token types, gradient-service APIs, node layouts, optimizer-state codecs or concrete backend adapters were not promoted to semantic defects where the required meaning and unsupported-readiness behavior are already specified.

This does not close implementation evidence. A claimed supported backend/profile must establish the recorded obligations; it cannot borrow the local fixtures' success as proof.

### P1-REV-01 — inherited Medium semantic applicability gap

The baseline permits independently managed participant state outside family component declarations, while shared module/parameter target scenarios require a declared model component and component-root expansion. OU permits participant/substructure subjects and additional learned state. Companion optional structural associations solve absent catalog evidence, not absent family ancestry. [Loaded] [Targets] [CompanionTargets] [OU]

**Execution consequence:** a learned loss-weight vector or auxiliary participant can have accepted optimization responsibility without genuine family-component ancestry. The audit cannot assume it is grouped by fabricated host ancestry or resolved through an unrecorded substitute target contract. Alias/member resolution must preserve its semantic owner independently from host-target provenance.

**Disposition:** retain P1-REV-01 and its smallest correction: clarify applicability of component-derived references and the relationship for participant-owned substructure without such ancestry. Verify the resulting member/group behavior at integration; do not select a concrete G5 representation here. This is not a claim that every standalone participant is untrainable.

### P2-REV-01 — inherited Low restoration wording risk

Design G4.4's introduction, lines 2267–2275, still couples missing restored facts with a new run establishing new identities. G4.6, RS “Same-run restoration lacks required identity facts,” and CC “Failed restoration cannot expose a mixed continuation” explicitly require incomplete same-run restoration, gated dependent execution and a separate new-run request. [RestoreIntro] [Reconciliation] [RS] [CC]

**Execution consequence:** failure to restore an uncertain action, gradient window or required identity/state cannot be converted into fresh defaults and new identities to resume ordinary work. The explicit correction governs; the misleading introduction remains a documentation drift risk.

**Disposition:** carry P2-REV-01 to Pass 5. Limit the introduction's new-identity consequence to a separately requested new run. No new P3 finding or recovery architecture is warranted.

## 4. Concrete success and partial-failure traces

These are derivations from recorded requirements and cases, not claims that the full arrangements execute in today's Trainer. Example names and numbers are illustrative. Policy-dependent choices below remain policy-dependent; no universal sequence is added.

### A. Ordinary maintained training with adaptive feedback

1. Author selects representation/conditioning, objective/predictor/loss, semantic subjects, an offered standard advancement policy, adaptive observation inputs and state initialization/continuation rules. Fulfillment accepts those meanings; preparation establishes current routes and unit runtime.
2. Run coordination supplies admitted changing input and coordinates. The selected schedule and stateful computation execute using current prepared and owned state; static implementation discovery is absent.
3. Computation produces distinct backward, accounting and observation values where required. Required feedback reaches its state owner before the later decision that depends on it. Trainer applies only the accepted unit/window mechanics and routes due capability work under its separate readiness.
4. A fused or chunked lowering may remove internal dispatches, but cannot postpone feedback past its dependent decision, erase required capability/transition/capture opportunities or weaken partial-outcome knowledge. Generated Trainer mechanics retain Trainer ownership. Equal final loss/weights alone is insufficient equivalence evidence. [EX] [TC] [G36] [G37] [G26]

**Current-code corroboration:** the live loop advances objective scheduling before `process_batch`, routes `sampling_loss` to objective adaptation before loss modification/backward, then performs optimizer mechanics. SDXL returns sampling loss separately from weighted backward/accounting values; timestep runtime updates sampler state. This corroborates the migration dependency, not the accepted-arrangement implementation. [CurrentLoop] [CurrentSDXL] [CurrentTimesteps]

**Invalid return after a possible update:** if selected work updates adaptive state/RNG and then returns an invalid result, EX requires a post-invocation failure with possible effects. Dependent decisions cannot consume assumed old state merely because no valid result exists. Required recovery must establish state/effect validity; computation is not automatically replayed. `test_execution.py` contains explicit effect-then-invalid-output assertions. [EX] [ExecutionTests]

### B. Frozen conduit, routed derivatives and retained earlier state

The recorded proposal's B fixture uses frozen `F(a)=k*a`, `k=1`, trainable `a=2` in U1 and `b=3` in U2: `L1=F(a)*b` routes to U1; `L2=F(a)^2*b` routes to U2. Authorized gradients are **(3, 4)**; summing both losses for both subjects gives **(15, 6)**. These are analytical case values, not an executed framework result in this audit. [ProposalB]

Frozen F remains outside optimization membership while transmitting the required derivative to a. Preparation must include its required executable representation and preserve that route. Separate derivative demands, seeds where applicable, cuts and destinations are accepted meaning; an optimizer's disjoint parameter list does not repair an incorrectly summed gradient. [EX] [RP] [OU]

For the fixture's **supported recomputation variant**, D1 is ready while D2 still needs old a/F state. U1 cannot mutate that state merely because its gradient is ready. It waits until the required remaining use completes. A **supported isolated-state variant** can allow earlier advancement only if it preserves D2's meaning and also satisfies window, sync/unscale, clipping, ordering and current-readiness obligations. A revision label supplies neither old numerical contents nor completion. [G36] [ProposalB]

**Cross-owner failure/cancellation:** retain originating invocation, outstanding derivative requests, required state and release conditions. Cancellation cannot dispose of storage still read by device work or certify partial gradients as usable. Resource release and mutation eligibility follow actual completion or supported isolation. Replay/recompute additionally follows accepted RNG and effect rules. These semantics are explicit; delayed-derivative/exclusion implementation is not demonstrated by the local routed-gradient test. [EX] [RP] [AdvancementTests] [G37]

### C. One participant with and without an adapter

One base participant supplies a no-gradient teacher use and a differentiable adapted-student use. Uses retain distinct views, input/output and gradient requirements while sharing base incarnation. The relationship remains accepted; ordinary per-use view choice is not detachment/rebinding or a new participant. [EX] [G26] [SketchUses]

If both views toggle shared mutable adapter state, incompatible uses must serialize or use supported isolation. Protection extends through required backward/recomputation, not only forward return. Uncertain restoration gates dependent use. Successful view restoration does not make the failed action, gradients, RNG or other state effects recovered. A frozen base can also transmit gradients to a side network without entering optimization membership. [EX] [RP] [G36]

This is established meaning, not an executed view-restoration/exclusion test. Component-free side-state targeting still inherits P1-REV-01 where applicable.

### D. Joint Muon/AdamW-style units, alternating actions and independent windows

**Joint arrangement:** one accepted action offers a shared objective to two disjoint responsibilities, such as matrix and other-parameter units. Semantic identities/groups/policies precede physical optimizer construction. A supported shared backward may serve both; separate sources need controlled routes. Synchronization, scaling/unscaling, clipping scope/timing, ordered eligible advancement, scheduler trigger and zero/discard treatment come from the accepted policy. Joint due status does not make the update atomic. [G31] [OU]

**Alternating arrangement:** due policy selects G or D; execution may use the other participant as a differentiable conduit without selecting its parameters for that contribution. No mandatory G-then-D cycle or global optimizer clock is introduced.

**Nonconsecutive window:** in the inspected G, D, G test, G accumulates two contributions while D completes one. D's zero/step leaves G's pending gradient intact; G advances only at its own completion boundary. The test explicitly detaches the unselected parameter in its local objectives. It therefore demonstrates that chosen routing/window case, not arbitrary unselected-destination suppression. [AdvancementTests]

**Distinct sources:** the other local test obtains routed gradients **4 and 6**, while naive summed backward gives **16 and 8**. Its “backend lacks isolation” rejection is a trusted test flag; real readiness must establish backend support rather than accept that flag as evidence. [AdvancementTests] [RP]

**First optimizer returns; second mutates then raises:** retain first returned outcome, second uncertain unless backend evidence narrows it, and later unentered mechanics as unattempted. Keep their input/attempt/unit associations. Missing action completion does not erase first-unit knowledge or prove either unit's numerical change. No rollback, automatic retry or exactly-once effect follows. [OU] [EX] [ExecutionTests] [RunTests]

**Scheduler or final zeroing failure:** preserve the already known optimizer outcome separately. Scheduler state or gradient storage may be uncertain; neither can be reused as though cleanup completed. Other units' reached/unattempted status depends on the accepted ordering, not an invented universal schedule. These failure obligations are explicit in OU, but the inspected fixtures do not inject scheduler/final-zero failures. [OU] [G37]

**Backend boundaries:** standard meanings include differently due units, but arbitrary overlapping accumulation, shared scaler state or distributed routing is not thereby supported. A backend unable to isolate/synchronize selected windows or report a required non-skipped scheduler trigger fails readiness. [OU] [G31]

### E. Several independently produced inputs feeding one consumer

Suppose a consumer requires caption encoding E17 and representation R17. R17 completes first; E17 completes later. Admission preserves each requested/produced identity, actual provenance and consumer association, then joins the intended pair. Queue position, one selected input ID or readiness alone cannot establish that join. Swapped, missing, stale or incompatible members block computation or follow an accepted wait/reissue/fallback policy. Partial handoffs cannot be silently dropped or duplicated. [EX]

After consumption, an action with several unit outcomes retains **both** input-to-attempt associations and separately reached operation effects/unit outcomes. Production may have completed before that action existed; it is still meaningful production completion. Correlation does not transfer producer or optimizer state ownership to the coordinator. [EX] [OB] [G36]

The one-feed candidate cannot demonstrate this trace. G3.7 explicitly retains a positive and rejected/failed multi-input conformance case. Producer lifecycle, fairness/backpressure and concrete split/join/cancel policy implementation belong to Pass 4; coherent recovery of partial handoffs belongs to Pass 5. The necessary execution-time correlation is specified independently of those representations. [G37] [Candidate] [SketchRun]

### F. Bounded two-pass region beside standard work

**Acceptance and entry.** The contract offers a supported region profile before authoring. Author selects its exact subject/unit, phase owners, effects, state/RNG and recovery contributions. The chosen narrow case requires one due unit and a quiescent window. Pending earlier contributions cause rejection before region mutation; they are not cleared to manufacture entry. Another accumulation treatment needs its own accepted policy/support. This restriction is local to this profile, not universal extension policy. [TC] [G32] [EX]

**Preparation.** Establish current routes/member correspondence, scoped parameter/gradient access, physical aliases and relevant backend representations, completion/retention, exclusive use, sync/scaling compatibility and restoration/handback evidence. If the selected backend cannot establish them, reject readiness before invocation. The proposal expressly leaves actual scaler/master/sharded/offloaded evidence unfinished. [RP] [ProposalSAM]

| Phase | Owner in the chosen split | Condition for safe progress |
| --- | --- | --- |
| Due/admission/readiness, protected entry and initial gradient clear | Trainer/run coordination | No incompatible pending window or outstanding old-state use; current scope and supported backend evidence. |
| First computation/backward, gradient inspection, bounded perturbation | Region executor | First gradient has the required numerical interpretation; required first-pass use is complete or isolated before conflicting edit. |
| Intermediate reset and second computation/backward | Region executor | Reset only granted gradients; preserve other windows. Keep temporary view protected through its actual required use. |
| Restoration and produced-gradient handback | Region executor supplies facts; receiving owners verify | Same attempt/current scope; required unperturbed representations restored; second-pass gradient usable; effects accounted; completion established. |
| Remaining accepted sync/unscale, clipping, advancement, scheduler and final cleanup | Retained Trainer/backend owners | Exactly the accepted remaining actions; **no additional standard backward**. Protection continues/transfers without a gap. |
| Settlement/release and later work | Corresponding owners/run coordination | Reached outcomes remain distinct; release follows required completion. Snapshot additionally needs the complete supported contributor cut. |

A selected SAM wrapper that restores **and steps** inside its own second step does not fit this split unchanged. Select another offered profile with the corresponding duties, decompose it or reject it; library availability cannot transfer final advancement. The executor can branch on current gradients/state and change an accepted radius without numerical capture, author callbacks or another training loop. Narrow access relies on trusted Python conformance and boundary-observable checks; it is not a hidden-mutation sandbox. [G32] [TC] [ProposalSAM]

**Failure variants:**

| Reached boundary | Facts that remain meaningful | Dependent work withheld |
| --- | --- | --- |
| First pass fails before edit | No edit only if that non-entry is established; gradients/RNG/input/owned-state effects may exist. | No blind retry, second pass or final step; gate affected state until accepted recovery. |
| Second pass fails, restoration is evidenced | Unperturbed parameter fact may be known; action/contribution and other effects remain failed or unresolved. | No final advancement. Narrow continuation requires recovery of all relevant effects, not just a clean tensor. |
| Restoration fails or has uncertain partial writes | Parameters/views/gradients and backend representations may be uncertain. | Affected standard/research work, capability/product reads, conflicting publication and exact snapshots remain unavailable. |
| Executor returns malformed/stale/incomplete handback | Preserve any established restoration fact; missing gradient/effect/completion facts remain missing. | No retained advancement; continue protection or gate affected usability before releasing exclusion. |
| Trainer mechanic fails after valid handback | Region restoration can stay known; retained mechanic reports its reached outcome separately. | Apply unit/effect dependency and recovery rules; no special SAM transaction. |
| Cancellation or cleanup failure while work remains in use | Preserve primary plus cleanup failures and known effects; host exit is not device completion. | Do not release still-used resources or admit uncertain state. Authorized restoration/replacement can still enter through recovery. |

Unrelated N or an isolated frozen encoder can continue only if its dependencies, physical aliases, RNG/scaler/backend resources and accepted failure policy permit it. “Different participant” alone is insufficient isolation evidence. [EX] [G36] [CC] [ProposalSAM]

For this bounded synchronous case, the proposal's direct handlers suffice **in principle** when prepared contracts and owner facts preserve these relationships. Retained executable phase structure could help yielding/asynchronous completion or inspection, but the sources do not establish a mandatory second IR or continuation object. Actual access/completion mechanisms and their proof remain G5/backend work. [ProposalSAM] [G36]

### G. Stage change with pending contributions and retained derivative state

A stage policy requests teacher binding, unit policy and dependent owner-state changes. Before a conflicting transition takes effect, outstanding derivatives/recomputation/device work either finish or use supported isolation. Pending contributions follow accepted complete/preserve/discard treatment. Identity continuity cannot validate old momentum, scheduler state, gradients or saved numerical contents. [G35] [EX] [OU]

The authority evaluates prospective canonical changes under current obligations; relevant owners establish optimizer/EMA/operation-state continuation. Dependent work waits for one coherent current view. Same-definition physical replacement can preserve unit definition revision while requiring new member/alias/backend resolution; changed accepted policy/membership revises the continuing definition; split/merge creates new responsibilities. [G35] [RP]

Replacement-only candidate failure preserves only still-valid prior state. Destructive preparation withdraws affected guarantees before mutation; failure leaves them withheld. An old optimistic result cannot restore them. Neither path may publish against a still-required numerical use merely because revision checks passed. [RP] [EX]

**Failed transition/cancellation:** old gradients do not silently enter the new runtime. Required owner-state mapping or publication not established means dependent work remains unavailable. A fresh optimizer object, copied weights or released lock supplies no completed-stage evidence. Snapshot/recovery mechanisms are routed to Pass 5, not invented here.

## 5. Evidence map and completion claims

“Inspected assertions” below means source inspection, **not a fresh test pass**. Default Python lacks `torch`, `pytest` and `hydra`. No dependencies or substitute harness were installed; no pytest fixture was rerun.

| Evidence layer | Actual scope examined | What it establishes or requires | What it does not certify |
| --- | --- | --- | --- |
| Governing requirements | Complete EX/OU/TC/RP/RS, applicable CC/AP/OB requirements and scenarios; D5, D9/G3 and D10. | Owner boundaries, dynamic behavior, gradient/lifetime relationships, windows, readiness rejection, effects and partial failures are specified. | Their production realization or every offered backend's support. |
| Recorded case analysis | G2 ordinary/variation/non-SD/adaptive/granted cases; G3 transition/interaction traces; proposal B and §5a. | Derives the required owner relationships without a universal pipeline or mandatory numerical capture. | An integrated executing run, race exclusion, compiled equivalence or recovery proof. |
| Action wiring and joint-unit fixtures | Entire `execution.py` and `test_execution.py` inspected. | Input/wiring/due rejection; selected work outside backward; post-invocation malformed outputs with effects; local shared backward/clipping/order; first returned/second uncertain. | Full authority/readiness; actual Muon; accumulated/scaled/distributed policy; scheduler/final-zero failure. SGD stands in for the matrix/Muon responsibility. |
| Advancement fixture | Entire `test_advancement_policy_experiment.py` inspected. | G,D,G independent windows; controlled local gradients; wrong summed gradients are observable. | A general window/router API, distributed isolation, AMP/scaler support, pending-window capture or delayed last use. Backend rejection is simulated by a Boolean. |
| Granted fixture | Entire `test_imperative_authority_experiment.py` inspected. [ImperativeTests] | Phase-set acceptance, pending-gradient/backend-flag rejection; one parameter restored with gradient 2.2; Trainer's sole step yields 0.78; failed second pass/restoration/missing gradient prevent finish. | Snapshot exclusion or a shared access protocol. The test named “withholds_live_use_and_snapshot” calls the finish gate and inspects a perturbed parameter; it does not execute live-use admission or snapshot capture. |
| Whole-run candidate fixture | Entire `test_run_language_experiment.py` inspected. | Same coordinator for ordinary/alternating actions; manually published independent input; exact local request/admission; correlation of partial unit outcomes; shutdown failure recorded separately. | Autonomous scheduling, multi-feed joins, selected-state continuity, capabilities/transitions, device completion or exact recovery. An escaped advancement exception/malformed report can conservatively lose narrower knowledge; this is a documented candidate limitation. |
| Preparation fixture | Entire `test_preparation_exchange_experiment.py` inspected. [PreparationTests] | Two physical orders, required frozen route, same-owner alias consolidation, cross-unit conflict, final-member correspondence, stale/incomplete rejection. | Multiple execution groups, separate owner-store installation, rank agreement, real backend provenance/rebind, destructive or overlapping-group preparation. |
| Producer/packing support | Selected provider/admission, paused-state and packing/restoration assertions inspected. [ProducerTests] [InputPolicyTests] | Explicit work/provenance and domain-boundary checks support the handoff distinction. | Credible actual source-state detection or a contributor-complete snapshot. Detailed lifecycle goes to Pass 4. |
| Current production | Selected loop, SDXL loss and timestep runtime passages; absence of experimental imports in active launcher/Trainer/loop. | Corroborates the ordinary adaptive migration case and separation of backward/accounting observations. | The target contract or production use of the candidate. |
| Recorded cost data | Raw JSON parsed; all three source hashes and stored paired medians verified; probe and comparison tests inspected. | Reproducible correspondence and bounded recorded CPU scope. | A new benchmark run, full-run speedup, cumulative allocation rate or long-run bounded memory. |

### G3 task claims versus their acceptance criteria

| Checked task | Audit disposition |
| --- | --- |
| **3.1 — define standard advancement policies** | G3.1 and OU cover one/independent/joint shapes, gradient routes, windows, sync/scaling, clipping, scheduler/zeroing, lifecycle, checkpoint and failure ownership. The two local experiments inform distinctions; distributed support is expressly not claimed. |
| **3.2 — define changed authority** | G3.2 and TC/EX/RP name exact phase owners, access/effects, entry/handback and recovery. The coarse experimental mechanic set is expressly insufficient. The completed design task is not a promise to ship SAM or a verified mutation sandbox. |
| **3.3 — bounded whole-run experiment** | Candidate and fixtures satisfy the bounded language question: ordinary/alternating arrangements share a coordinator and independently published input with attempt/outcome correlation. One-feed, head-of-line waiting, trusted provenance and missing transitions/capabilities/restoration are explicitly recorded limits. No broader completion inferred. |
| **3.4/3.5 — preparation and invalidation meanings** | Both construction orders and identity/state-continuity rules are recorded; pending-window and current-use treatment agree with execution. Narrow preparation tests do not claim distributed/publication implementation. P1-REV-01 still qualifies component-free member resolution. |
| **3.6 — complete owner handoffs** | EX/OU and G3.6 cover live decisions/state, outputs/offers, observations, transition requests, contribution/handback/outcomes and effects without a universal result/sequence. Multiple-input and delayed-lifetime behavior are required but not credited to the candidate. |
| **3.7 — criteria plus representative measurements** | G3.7 defines positive/rejected or failed cases, interaction-based equivalence, retained dynamic/safety checks and scoped cost gates. Initial measured dispatch/state-correlation scope exists; granted/backend/transition/full-run measurements remain expressly open. |
| **3.8 — file/validate/review/confirm** | Normative requirements are present in the core deltas. This audit did not rerun OpenSpec validation, inspect review/confirmation history or independently certify the administrative completion claim. |

Missing production integration does not invalidate these architectural derivation tasks by itself. Conversely, a checkbox does not substitute for G3.7's implementation conformance and support evidence. [Tasks] [G37] [SketchRun] [SketchCost]

## 6. Cost evidence: admitted scope and limits

The raw records contain three reported independent process runs on 2026-10-01, CPython 3.13.13/WSL2, seven 10,000-call samples per case. The inspected probe warms each case, reverses comparison order on alternate repeats, leaves GC enabled, excludes construction from timing and measures allocation separately. The raw fixture, candidate and current-loop SHA-256 values match this pinned checkout; stored paired medians were recalculated successfully. This verifies record consistency, not independent execution of the recorded runs. [CostData] [CostProbe]

| Recorded scope | Median of three process medians, µs/call |
| --- | ---: |
| Candidate / direct, 1 cheap operation | 4.834 / 2.674 |
| Candidate / direct, 4 cheap operations | 6.629 / 2.843 |
| Candidate / direct, 16 cheap operations | 14.406 / 3.574 |
| Extracted current-loop call, with direct fixture facade | 4.012 |
| Ready-input workflow tick, no real optimization | 18.453 |
| Same tick with 256 / 1,024 dormant actions | 18.197 / 17.956 |
| Tick reporting 2 / 16 addressed unit contributions | 19.290 / 32.675 |

The direct unary fixture retains input membership, selected operations, post-invocation exception wrapping, addressed offers and observations. Its narrow comparison tests include effect-then-failure and changing selected state after compilation functions are disabled. It does not supply real participant freshness or current-use protection; those are outside both compared fixture paths. [CostTests] [CostProbe]

The whole-run tick performs due selection, request/delivery checks, trusted admission, computation and report bookkeeping. Its provider is already ready; optimizer work is replaced by a constant reported contribution. Reports are drained after every tick. Dormant-trend results support this particular indexed dispatch scope, not autonomous activity scheduling or real optimizer throughput. The current-loop comparison extracts a real call expression but replaces its body with the direct fixture; it is argument/field-access dispatch evidence, not a matched complete legacy/new Trainer run.

Peak traced bytes and post-GC retained bytes do not measure cumulative allocation rate. A 100-call diagnostic plus immediate report draining cannot establish bounded retained state over elapsed attempts. G3.7 retains the fixed queue/history/window retention gate, additional dependency-count checks, granted access/handback, transition/replanning and actual backend measurements. Its greater-of-1µs-or-20% excess allowance and dormant-trend threshold are migration acceptance policy for comparable scope, not permanent execution semantics. Required new guards must exist in both comparisons and receive a separate justified budget; omitting them to fit the old envelope is explicitly nonconformant. [G37] [EX] [SketchCost]

## 7. Named leads and verified areas

| Lead | Pass 3 disposition | Source relationship and remaining evidence |
| --- | --- | --- |
| **P0-INV-04 — correlation and work lifetime** | **No new semantic finding; implementation evidence remains bounded.** | EX expressly covers many-input association, separate attempts/unit outcomes, actual production dependencies and partial handoffs. Differentiated/protected-use rules carry lifetime through other owners. G3.7 retains multi-input and delayed-use cases. The one-feed candidate cannot prove them. Detailed producer coordination → P4; coherent capture/reissue → P5. |
| **P0-INV-08 — backend support and safe handback** | **Pass 2's semantic disposition independently corroborated.** | RP requires actual derivative/grant completion/retention/protection support or readiness rejection. EX continues protection through handback and retained mechanics, gates uncertainty and permits recovery. Proposal §5a explicitly leaves scaler/master/sharded/offloaded proof unfinished. Boolean flags, clean tensors and callback return are not sufficient universal evidence. |
| **P0-INV-09 — checked coverage versus whole-run evidence** | **No unsupported broad completion claim established in G3's inspected records.** | Tasks distinguish defining exchanges and bounded experiments from production acceptance. G3.7's evidence column and sketch explicitly mark multi-input, lifetime, capability, transition, restored-state and backend guarantees untested. Administrative reviews/confirmations were not independently audited; P7 must assess integrated case sufficiency. |

Verified at the source level: static choice removal preserves runtime schedules/adaptation; state-only effects need no optimizer offer; frozen/differentiable/optimized roles differ; participants differ from uses/views; semantic units differ from handles/list positions; same-owner aliases differ from competing ownership; independent windows cannot be cleared by unrelated work; contribution/gradient readiness does not authorize unsafe mutation; produced-gradient handback does not repeat backward; protection follows actual numerical use; and partial failures preserve known/uncertain/unattempted outcomes. [EX] [OU] [RP] [G36]

Failure requirements also preserve primary and cleanup failures, prohibit interpreting missing results as unchanged state, separate scheduler/zeroing failure from prior optimizer return, and allow authorized recovery while ordinary use is withheld. Unaffected work remains subject to dependency and failure policy, not an automatic global stop or automatic independence assumption.

## 8. Coverage limits and later-pass questions

- **Pass 4:** verify detailed independent-progress eligibility/fairness, producer/capability lifecycle, work credits, backpressure, split/join/cancel correspondence and actual source/storage completion. The prototype's first-waiting-item break is a recorded restriction, not the target policy.
- **Pass 5:** verify coherent cuts and supported recovery for pending gradients, retained derivatives, partial handoffs, stage changes and uncertain external effects. Carry P2-REV-01. Restored identity does not restore numerical contents/readiness; current clean parameters do not establish contributor completeness.
- **Pass 6:** verify durable schema/report integration preserves many-input associations, unit incarnations/revisions and reached-mechanic knowledge. Required feedback/history cannot be reduced to optional telemetry. Carry P1-REV-01's semantic owner versus component/host provenance distinction.
- **Pass 7:** combine adaptive feedback, several producers, independent windows, outstanding derivatives, scoped research authority, capability readers and stage change. Constituent source traces and local tests do not certify that integrated arrangement.

No new experiments, full Trainer/backend execution, GPU/device tests, concurrency/cancellation race tests, scaler/sharding/offload support matrix, compiled/fused equivalence test, rank-failure injection, exact runtime restoration, benchmark rerun, OpenSpec strict validation, review-mcp invocation or Beads status/history verification was performed. Pytest fixtures were inspected only. Selected production source was checked for relevant migration claims; this was not a general implementation-quality audit. Broader research recipes and upstream links were not independently re-audited.

The remaining evidence requirements are substantial and explicit. This report neither replaces them with unstated guarantees nor treats their deferred implementation as a new architecture finding. It issues no overall G5-readiness verdict.

## Source references

All repository links below resolve to the audited commit. Requirement/scenario and section names above identify the applicable source units. Supporting proposal/experimental references remain subordinate evidence.

[Authority]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/design.md#L19-L65
[Reconciliation]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/design.md#L195-L274
[D5]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/design.md#L512-L570
[State]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/design.md#L870-L933
[G31]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/design.md#L1295-L1382
[G32]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/design.md#L1383-L1482
[G35]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/design.md#L1571-L1701
[G36]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/design.md#L1702-L1906
[G37]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/design.md#L1907-L2015
[D10]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/design.md#L2016-L2125
[G26]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/design.md#L2882-L3057
[RestoreIntro]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/design.md#L2258-L2275
[EX]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/accepted-training-execution/spec.md
[OU]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-optimization/spec.md
[TC]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-contract/spec.md
[RP]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-runtime-preparation/spec.md
[RS]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/run-participant-state/spec.md
[CC]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-capability-coordination/spec.md
[AP]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-artifact-persistence/spec.md
[OB]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-observability/spec.md
[Loaded]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/loaded-model-components/spec.md
[Targets]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/optimization-target-refs/spec.md
[Corrections]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/docs_design/models-strategy-trainer/notes.md#L1098-L1247
[Settled]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/docs_design/models-strategy-trainer/strategy_contract_exchange_design.md#L296-L329
[CompanionTargets]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/optimization-target-refs/spec.md
[MainTargets]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/optimization-target-refs/spec.md
[MainAdapters]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/adapter-system/spec.md#L78-L146
[MainTargeting]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/adapter-module-targeting/spec.md#L39-L76
[MainOB]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/training-observability/spec.md
[CurrentLoop]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/library/training/phases/training_loop.py#L465-L590
[CurrentSDXL]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/library/strategies/sdxl/diffusion.py#L204-L242
[CurrentTimesteps]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/library/timesteps/runtime.py#L135-L190
[Candidate]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/library/training/execution.py
[Tasks]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/tasks.md
[ProposalB]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/research/trainer-architecture-concrete.md#L238-L348
[ProposalSAM]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/research/trainer-architecture-concrete.md#L375-L530
[SketchUses]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/research/training-mechanism-sketch.md#L448-L477
[SketchRun]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/research/training-mechanism-sketch.md#L609-L628
[SketchCost]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/research/training-mechanism-sketch.md#L635-L665
[CostData]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/research/execution-cost-results.json
[ExecutionTests]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/tests/unit/training/test_execution.py
[AdvancementTests]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/tests/unit/training/test_advancement_policy_experiment.py
[ImperativeTests]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/tests/unit/training/test_imperative_authority_experiment.py
[RunTests]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/tests/unit/training/test_run_language_experiment.py
[PreparationTests]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/tests/unit/training/test_preparation_exchange_experiment.py
[ProducerTests]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/tests/unit/training/test_input_producer_experiment.py
[InputPolicyTests]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/tests/unit/training/test_input_policy_experiment.py
[CostProbe]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/tests/unit/training/execution_cost_probe.py
[CostTests]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/tests/unit/training/test_execution_cost_probe.py
