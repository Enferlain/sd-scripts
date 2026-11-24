# RMST Pass 4 — Independent progress and capabilities

**Audit date:** 2026-10-04 UTC  
**Repository:** Enferlain/sd-scripts, `model-strategy-trainer`  
**Audited commit:** `6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48` (`4.6`)  
**Concern:** independently progressing input/cache production, training, validation, sampling, protected access, feedback, and continuation ownership.

The examined requirements supply the essential ownership and safety relationships for this concern without imposing one training-step sequence. This audit identifies **one localized wording ambiguity, P4-REV-01**, concerning validation “before” a runtime trigger. It does not identify an additional missing join, admission, protection, or continuation relationship in the inspected governing requirements. Executable evidence is substantially narrower than those requirements; the design expressly records that boundary.

This is a read-only specification audit. No repository files or task statuses were changed, and no recommendations were applied. This report gives no overall G5-readiness verdict.

## 1. Baseline, authority, and method

The local detached checkout's HEAD is the exact requested commit. GitHub commit metadata and the local checkout agree on tree `a46f1fb0e1edc9cda5e29dec873ae0cb78f8de46`. `git status --short` was empty before and after inspection. There was no additional working-tree content in this checkout.

The three supplied audit files were used as instructed: the general protocol governs the review, Pass 0 supplies navigation, and Pass 4 supplies the concern. **P1-REV-01 and P2-REV-01 have local corrections outside this baseline, as the Pass 4 instructions state.** Those corrections and their detailed diffs were not supplied or incorporated. This report neither assumes their presence nor re-audits their detailed ownership/publication concerns.

Authority follows the governing [design's source rules][authority]: the design and delta specifications govern the target; explicit recorded corrections govern supersession; the normative direction is carried forward through their reconciliation; working exchange material contributes only where adopted. The [topology correction][correction] expressly supersedes the earlier broad strategy/model relationship. The [concrete two-pass proposal][twopass] remains a worked proposal and pressure case, rather than a production interface or additional authority grant.

Complete relevant requirements and scenarios in [capability coordination][cc] and [accepted execution][ex] were read, with dependencies through contract, participant state, preparation, optimization, persistence, and observability. Selected design and research sections were followed across the cases below. Current production was inspected only to check migration assumptions. Tests were inspected as assertions, not treated as newly executed evidence.

## 2. Source-backed ownership map

“Owner” below names a responsibility, not a prescribed class, service, lock, or universal container.

| Activity or boundary | Activity/state owner | Admission and access responsibility | Separate continuation contribution |
| --- | --- | --- | --- |
| Whole-run lifecycle and due selection | Trainer or delegated run/pipeline coordination retains startup, due/wait relationships, handoffs, shutdown, results, and failures. Selected implementations may keep workers and internal scheduling private. | Coordinate against the existing accepted obligations and current readiness; do not create a second acceptance policy. | Due/request and cross-owner correspondence needed for the selected recovery claim. [D5][d5], [D10][d10], [EX whole run][ex-run] |
| Cache production | Data/cache infrastructure owns production traversal, workers, queues, storage, and publication. Representation/conditioning behavior owns encoding, decoding, schema, and dependency meaning. | Protect actual producer-state access; publish only complete accepted units with usable storage/index and dependency evidence. | Production position, pending work, publication state, dependencies, and storage guarantees. [G4.1][cache-design], [CC production/publication][cc-cache] |
| Live input selection and delivery | Selected provider owns live/cache selection, mixture, packing, buffers, exposure, and delivery state. Production traversal is a different responsibility. | Training input coordination checks exact work association and consumer obligations before handoff/use; provider selection is not silently reassigned to outer orchestration. | Selection/packing/RNG as required; produced, ready, handed-off, and unfinished positions, rather than one cache cursor. [EX input][ex-input] |
| Training actions and optimization | Accepted computation/region owns its selected algorithm and declared state; Trainer owns standard mechanics. An extension transfers only explicitly granted phases. | Respect required derivative routes and numerical last use; optimizer eligibility and backend support are unit/profile-specific. | Algorithm state and separately owned unit windows/outcomes; receiving input is not advancement. [EX lifetime][ex-life], [optimization clocks][opt-clocks] |
| Validation | Pipeline owns scheduling, outer traversal, projections, aggregation execution, and routing. Selected evaluation owns measurement/reduction meaning and declared algorithm state. Provider-private selection stays provider-owned. | Capability coordination checks its own input admission and protected source/view, derivative, and reduction requirements. | Evaluation-provider continuation is independent unless sharing is explicitly accepted; coordination and algorithm effects remain separate. [G4.2][eval-design], [CC validation][cc-val] |
| Sampling | Pipeline owns due requests, outer traversal, access, destinations/publication, and result routing. Selected generation owns conditioning, objective-compatible inference, decoding, output meaning, and declared algorithm state. | Check generator/objective pairing and current protected dependencies; training admission does not authorize sampling or vice versa. | Request/publication continuation and selected generator state; generation, saved output, and reporting have separate outcomes. [CC sampling][cc-sample] |
| Shared source/storage use and temporary projections | Existing participant/preparation/backend and projection owners retain their authority. Run coordination coordinates conflicting uses. | Wait, reject before use, synchronize, or isolate as accepted; protect through actual dependent/backend completion, including handback into retained mechanics. Restore prior relevant modes, dtype, placement, and RNG or keep dependent use unavailable. | Required outstanding-use and restoration facts; exclusion release is not a validity certificate. [EX current use][ex-use], [CC protection/cleanup][cc-protect] |
| Required feedback, optional observation, and snapshots | The selected adaptive algorithm owns reactions/state; observation owns its recording/delivery concern; snapshot contributors retain their own state. | Required feedback must arrive before dependent decisions independently of optional logging. Snapshot coordination must establish a coherent recoverable cut. | Producer, provider, capability coordinator, selected algorithm, optimization, and backend contributions as required; no owner-local cursor alone proves recovery. [observability][obs-feedback], [CC recovery][cc-recovery] |

The accepted arrangement remains the canonical source for runtime participant/relationship/binding semantics. Durable metadata, loaded-component facts, and semantic target refs describe or project that state; they do not grant access to a live source. The companion change's source/model identities and unchanged main-spec typed facts therefore do not substitute for current producer-state provenance or revision-pinned consumer views. [Participant state][participant], [loaded components][loaded], [shared refs][refs], [metadata reconciliation][reconciliation], [main metadata][main-metadata], [companion provenance][provenance]

### Selection, eligibility, and readiness

The specified sequence of judgments is coherent except for the wording issue in §3:

1. Repository availability does not provide or select a capability.
2. Authored provision establishes the exposed capability and accepted bounds.
3. Configuration or selected policy requests work within those bounds.
4. Due/trigger eligibility derives from accepted progress and state.
5. Request validation and current executable readiness govern use.
6. Actual admission/access remains protected through the required lifetime.

Known incompatibility is rejected at fulfillment when authoritative evidence exists. Deliberately later requests and realization-dependent evidence are checked before readiness/use under the same obligations. Stateful policies, bounded runtime decisions, and adaptive behavior survive release of the authoring object. None of this permits support discovery through method presence, family names, or raising defaults. [CC selection][cc-select], [contract evidence timing][contract-timing], [EX authoring/dynamics][ex], [D5][d5]

Independent progress permits different clocks and meaningful production before a consumer is due. It does **not** promise simultaneous execution of conflicting numerical work. Conversely, one activity waiting does not establish a universal pause or an optimizer clock for all activities. Accepted dependencies and selected eligibility/progress policies determine order; no universal fairness algorithm is required. [EX whole run][ex-run], [D10][d10], [G4.2 shared work][eval-design]

## 3. Finding

### P4-REV-01 — “Before their runtime trigger” has ambiguous timing

**Impact:** Low; a localized wording ambiguity with a concrete implementation drift risk.  
**Classification:** Specification wording / ambiguous correspondence between selection validation, trigger eligibility, and use. The surrounding sources already state the needed late-request behavior; this is not a demonstrated missing safety guarantee.

**Exact sources**

- `training-capability-coordination/spec.md`, **Capability selection is validated before use**, lines 23–27: missing or incompatible requested capabilities must fail “before their runtime trigger.” [Selection requirement][cc-select]
- Its **Full-model persistence is unavailable** scenario, lines 29–32, expressly permits a bounded request supplied later and requires rejection before persistence readiness or checkpoint-resource creation. [Same source][cc-select]
- `training-contract/spec.md`, **Obligations are evaluated at the earliest authoritative evidence point** and **Readiness checkpoints are derived during strategy fulfillment**, lines 122–156: later requests are validated before capability readiness/use. [Contract timing][contract-timing]
- `design.md`, **G4.2 Validation and sampling exchanges**, validation request row: triggers may derive bounded requests during execution; the shared-access discussion distinguishes trigger eligibility from execution readiness. [G4.2][eval-design]
- `accepted-training-execution/spec.md`, **Capability work follows an accepted trigger**, lines 82–85: becoming due is followed by readiness checking and request routing. [Whole-run scenarios][ex-run]

**Concrete consequence**

An accepted evaluation policy derives a bounded request when evaluation becomes due. Alternatively, the capability spec's own persistence scenario supplies a request later. If that request is incompatible, authoritative request evidence may first exist at that point. A literal reading requiring failure before the eligibility trigger cannot be satisfied. An implementation could respond by freezing all requests during authoring, excluding permitted late requests, or by inconsistently treating “trigger” as actual capability invocation.

The latter interpretation may match the intended safety boundary, but the source uses trigger/due eligibility separately from execution readiness. The audit therefore does not silently equate them.

**Smallest plausible correction**

Replace the unconditional pre-trigger wording with rejection at the earliest authoritative evidence point: known selections during fulfillment; deliberately late requests before capability readiness/use and before any resources prohibited by the capability's request policy. Explicitly distinguish due/eligibility from beginning capability work. Retain the existing unavailable-full-model scenario and add or clarify a trigger-derived invalid-request scenario that performs no prohibited capability work.

**Affected neighbors:** contract evidence/readiness timing; EX's due-capability scenario; G4.2 trigger-derived requests; capability-specific resource prohibitions. No new validator, scheduler, or request structure is needed.

## 4. Concrete overlapping-work and failure traces

These are source-derived traces, not claims of newly executed integrations. Each uses an accepted policy where indicated; the audit does not supply an otherwise unstated fallback or concurrency guarantee.

### T1 — Out-of-order encoding, changing captions, and encoder updates

Producer A starts work for sample S, caption variant C1, and actual encoder state E1. The provider later selects S/C2. A's result cannot be paired with C2 merely because S matches or its file is readable. If the encoder advances to E2 before completion, the result must retain the state actually used; a launch/completion label cannot relabel it E2. Publication checks current obligations and admission applies the selected freshness/lag policy. Strict-current consumption waits, recomputes, rejects, or uses another explicitly accepted resolution; permissive lag requires its own accepted evidence.

If production spans several source states, structured portion/state associations may be valid only when mixed-state production is itself accepted. Provenance alone does not authorize it. Ordinary weight evolution can invalidate reuse while participant and binding identity remain unchanged. [CC computation/freshness][cc-cache], [EX late/mixed inputs][ex-input]

### T2 — Frozen upstream cache, live trainable adapter

Frozen upstream encoding publishes the complete schema required by a trainable conditioning adapter. Adapter updates do not invalidate precisely evidenced upstream dependencies, but the adapter output is recomputed live and its derivative path remains present. Detached encoding of a trainable upstream path cannot replace a required derivative merely because inference could use it.

The recorded Anima case's upstream states, masks, and token inputs illustrate why “one embedding tensor” is not a universal schema. Exact caption encoding and stochastic transformations cannot be replaced by approximate cached edits or frozen augmentation unless that different behavior is selected and accepted. [CC cache scenarios][cc-cache], [Anima research][anima], [conditioning research][conditioning]

### T3 — Streaming/sharded publication and capacity saturation

A producer has no complete finite manifest. A shard/chunk becomes ready while later work remains pending. Only a complete independently usable publication unit may be advertised; a consumer requiring the whole bundle remains unready. The provider may choose a ready subset only under its accepted selection/exposure policy. Arrival order and storage pressure cannot silently define training distribution.

When pending/ready resources reach accepted bounds, backpressure must honor those bounds without substituting a different mixture or dropping required work. Supported distributed ownership must be respected; the sources do not make one rank assignment algorithm universal. Workers need not become individual execution nodes. [CC publication/lifecycle][cc-cache], [data-pipeline research][data-research], [distributed research][distributed], [async-data discussion][async-data], [shard discussion][shards]

### T4 — Two independent inputs, a missing member, and partial handoff

A consumer needs A and B from independent producers. B completes first. Admission retains both requested/produced identities, dependency evidence, and consumer association; it cannot invoke joined computation with B substituted for A. If A is missing or stale, the accepted wait/reissue/fallback policy governs. An already acquired B cannot be silently duplicated or discarded.

If joined work later reaches one returned unit outcome and another uncertain outcome, both input-to-attempt associations and the separate unit facts remain correlatable. An action accepting a tuple is not evidence of this independently progressing join. [EX handoffs/correlation][ex-input], [EX outcome correlation][ex-correlation], [G3.7 join gate][conformance]

### T5 — Shared storage, different consumers, compaction during a read

Training, validation, and sampling may share one stored value and still have distinct logical work/handoffs and admission obligations. An inference consumer accepting detached data does not authorize training to omit a required live derivative.

After admission, compaction or eviction cannot invalidate outstanding source/storage access. The backend must preserve the required read through completion or provide accepted isolation; if safe access cannot be established, dependent use remains unavailable. A matching semantic identity or previously valid locator is insufficient. This is a lifetime rule, not a prescribed lease implementation or dataset-wide transaction. [CC consumer admission/storage][cc-cache], [EX protected use][ex-use], [shard discussion][shards]

### T6 — Payload written, index publication fails, producer is cancelled

Payload production succeeds, but required indexing/locator/readiness publication fails. The value is not ready. The result preserves known external effects and incomplete publication; accepted storage recovery may verify/adopt, reissue, or remove orphaned work. Complete independent records may remain usable under accepted partial-coverage policy.

Cancellation with other work in flight must preserve completed publication, pending/uncertain work, and cleanup failures. Producer shutdown/resource release does not certify uncertain source or storage state. Dependents wait, recover, follow an explicitly accepted fallback, or stop before invalid delivery. No crash-safe transaction or automatic rollback is inferred. [CC publication/cancellation][cc-cache], [EX failure facts][ex-correlation]

### T7 — Bounded-stream validation, unequal weights, and missing contributions

Evaluation requests a bounded coverage/stopping rule, using its own provider continuation unless sharing is accepted. It cannot advance the training cursor accidentally or require a dataset length/epoch. Its selected measurement determines weighting, masks, denominators, and distributed reduction. Pipeline aggregation executes that meaning; an unweighted mean of batch means is valid only when selected.

An empty reduction follows the selected policy. Missing/partial contributions cannot yield a fabricated completed score; an accepted partial measurement retains its actual scope. Stable-source evaluation needs protection across traversal; a trigger step is not actual evaluated provenance. Derivative-requiring evaluation is allowed under its profile without implicit optimizer or participant-transition authority. [CC validation][cc-val], [G4.2][eval-design]

### T8 — Validation feedback changes later accepted behavior

A validation measurement feeds a selected adaptive noise-range decision, as in the recorded BD3LM case. Required owner-to-owner feedback precedes the later dependent decision even if optional telemetry drops its copy. The adaptive owner retains its state and continuation after authoring ends.

Measurement completion, reaction effects, and reaction failure remain distinct. If the reaction may have changed state before failing, dependent decisions follow the accepted recovery rule. Retrying observation delivery must not repeat measurement, adaptation, advancement, or transition. This is selected algorithm behavior, not generic Trainer knowledge of BD3LM. [CC adaptive scenario][cc-val], [EX dynamics/order][ex], [required feedback][obs-feedback], [LLM research][llm]

### T9 — Generation overlaps training or a two-pass perturbation

A generator becomes due while training still requires retained numerical state or a two-pass region has temporarily perturbed shared parameters. Participant labels do not prove physical independence. Coordination waits, rejects, synchronizes, or uses an accepted isolated view. It cannot read the perturbation as ordinary state, unwrap/move live objects past readiness, or substitute a stale inference view.

Protection extends through backend last use, backward/recomputation, and handback into retained mechanics where required. A restored visible tensor and released exclusion do not prove restoration of scaler/master/sharded/offloaded state. Unsupported backend combinations must remain unready. Asynchronous rollout with lagged inference is an explicitly accepted consistency case, not permission for untracked stale generation. [CC protection][cc-protect], [EX lifetimes/grants][ex-life], [EX protected use][ex-use], [two-pass case][twopass], [rollout research][rollout]

### T10 — Partial setup, mixed projections, RNG conflict, and cleanup failure

Validation/sampling changes some modes, dtype, placement, or RNG, then setup fails or cancellation occurs. Cleanup restores the relevant prior state, including mixed module modes, or marks dependent use unsafe. Unconditional `train(True)` is not restoration. Saving a single global RNG generator cannot establish isolation for unsupported concurrent activity/device overlap.

If generation or a measurement already reached a known outcome and restoration then fails, retain that outcome plus the primary/cleanup failures and uncertainty. Releasing exclusion cannot admit later work until accepted recovery establishes validity. [CC scoped cleanup][cc-protect], [EX cancellation/uncertainty][ex-use], [G4.2][eval-design]

### T11 — Partial generation/publication and later reporting failure

For a multi-request job, output X publishes; Y generates but fails to save; Z is unattempted. Results retain those separate associations/outcomes. A planned destination is not a written resource. If X's optional tracker report fails, X remains published. Reporting retry does not regenerate X. Any reissue of Y follows the accepted retry policy and actual outcome evidence.

Text, audio/video, and related multi-output products use selected schemas/publication implementations without universal image dimensions or PNG/PIL fields. Sampling publication is not automatically a trained-product save or exact runtime snapshot. [CC sampling outcomes][cc-sample], [observability][obs-feedback], [persistence distinctions][persistence]

### T12 — Evolving producer/provider policy, late results, and continuation

The provider changes an accepted source-mixture/packing stage while production continues. Provider-owned buffers and boundaries remain meaningful; outer coordination does not reinterpret a cache index as input position. Late results preserve their actual work/provenance and are judged under current accepted obligations rather than accepted solely because they were once launched.

Produced, ready, handed-off, consumed, update-contributing, advanced, reported, and durable facts remain distinct where needed. A paused producer or saved cursor is not a coherent snapshot. The recovery claim requires a compatible cut across relevant owner contributions, pending work and effects; exact capture/recovery mechanics are routed to Pass 5. [EX input/state][ex-input], [EX correlation/failure][ex-correlation], [CC continuation/recovery][cc-recovery], [input-policy experiment][policy-test]

## 5. Dispositions of the Pass 0 leads

| Lead | Pass 4 disposition | Remaining evidence / neighboring concern |
| --- | --- | --- |
| **P0-INV-04 — multi-owner correlation and work lifetime** | Necessary meanings are specified: independent joins, exact associations, partial handoff policy, actual/mixed provenance, separate attempts/unit outcomes, and protection through dependent use. No additional semantic omission was established here. | The one-feed coordinator and manually supplied producer events do not demonstrate general joins or autonomous scheduling. G3.7 explicitly requires the missing joined-input and lifetime implementation cases. Pass 5: unfinished-work recovery. Pass 7: combined multi-owner sufficiency. [EX input/correlation][ex-input], [EX protection][ex-use], [G3.7][conformance] |
| **P0-INV-08 — backend support and safe handback** | Unsupported profiles can be rejected/unready; cleanup success, callback return, a clean tensor, and exclusion release cannot fabricate current-use or handback evidence. The shared cross-owner lifetime is specified. | Real backend isolation/restoration, retained derivatives, cancellation/acquisition races, and capability reads during handback remain unproved by inspected fixtures. This is recorded implementation/conformance work, not a newly missing rule or a requirement to support every backend. Pass 2: preparation evidence; Pass 5: recovery; Pass 7: imperative-plus-capability case. [EX grants/current use][ex-use], [G3.7][conformance], [two-pass case][twopass] |
| **P0-INV-09 — checked coverage versus evidence** | Tasks 4.1/4.2 claim completed semantic exchanges. G4.1/G4.2 and G3.7 expressly distinguish those from production implementation and identify the remaining checks. The inspected capability completion language does not claim production concurrency, general joins, or exact recovery. | Checkmarks alone do not prove review completion or executable integration. Review/validation logs and all combined cases were not independently reproduced. Pass 7 should judge materially different combined cases, using evidence at its actual level. [Tasks][tasks], [G4.1][cache-design], [G4.2][eval-design], [G3.7][conformance] |

The lead dispositions do not close implementation verification. They explain why acknowledged experiment limits are not separately promoted into unsupported specification findings.

## 6. Evidence map, verified areas, and limitations

### Evidence actually examined

| Evidence level | Sources / inspected scope | What it establishes and what it does not |
| --- | --- | --- |
| Governing semantics | Design authority/reconciliation, D5, G4.1, G4.2, composed run state, G3.6, G3.7, D10, applicable identity/metadata decisions; complete CC/EX requirements and dependency specs. | Ownership, accepted behavior, safety/failure relationships, and required conformance. This is source verification, not runtime proof. |
| Companion / unchanged obligations | Companion design, loaded components, policy-neutral target refs and source provenance; applicable main loaded-component, model-family-metadata, and observability requirements. | Runtime authority and historical typed facts remain separate; neither metadata identity nor external projection grants live access. This is not an end-to-end metadata implementation audit. |
| Recorded research and future discussions | Research index; data pipeline, Anima, distributed execution, teacher/student, relevant preference/RL rollout cases, BD3LM evaluation; applicable mechanism/two-pass analysis; asynchronous data/TE, streaming/shards, exposure/accounting, and conditioning discussions. | Pressure for distinct clocks, state/provenance, stream scope, exact computation, selected consistency, and separate continuation. Optional proposals are not adopted wholesale; recorded external implementations were not revalidated against today's upstream. |
| Producer experiment assertions | [`test_input_producer_experiment.py`][producer-test] | Out-of-order work, variant/revision/lag checks, pending/produced/ready capacity, local failure/shutdown, paused-state stages, and separate action/unit facts. Producer “actual revision” is fixture-supplied; publication is an in-memory simulation. No real storage verification, autonomous encoder provenance, cancellation race, or total long-run memory bound is proved. |
| Input-policy assertions | [`test_input_policy_experiment.py`][policy-test] | Finite-source mixture/stage changes, token-budget packing, boundary identity, local buffers/state restore, stale/missing-boundary rejection. Not a streaming recovery or distributed selection proof. |
| Whole-run assertions and coordinator | [`test_run_language_experiment.py`][run-test], [`library/training/execution.py`][coordinator] | Generic ordinary/alternating actions, manual readiness, local admission, separately known partial unit outcomes, and primary/shutdown failure handling. One feed per action, trusted publication, first-waiting-item behavior, and retained report history limit the evidence. No general joins, scheduler fairness/autonomy, or capability concurrency is established. |
| Imperative assertions | [`test_imperative_authority_experiment.py`][imperative-test] | Scalar/local SGD ownership split, explicit granted actions, quiescent-gradient checks, restored-gradient handback, no duplicate backward, and withheld retained advancement on invalid handback. Backend support is a fixture Boolean. Tests referring to live use/snapshot do not exercise an actual capability reader or snapshot coordinator. No real backend restoration or shared-use protocol is proved. |
| Current production source | [Trainer startup][trainer], [cache traversal][cache-code], [cache-backed delivery][delivery], [SDXL caching][sdxl-cache], [orchestration helper][helper], [SDXL validation][validation-code], [sample generation][sample-code]. | Independently corroborates relevant finite-manifest/startup cache, sequential sample/validation, cache-shape/hash checks, cyclic batch-mean evaluation, image-output, and normal-path RNG/mode assumptions. These are migration assumptions predating the target, not target semantic defects. |
| Fresh execution | Git commit/tree/status checks and source inspection; Python dependency probes. | **No audit tests were freshly run.** `pytest` import failed in the available Python probes, so assertions above are explicitly inspected evidence. No installation, OpenSpec validation, benchmark, real GPU/distributed/storage exercise, or concurrency test was performed. |

### Verified within this concern

The inspected requirements explicitly preserve:

- Provision/request/due/readiness distinctions and earliest authoritative validation, subject to P4-REV-01's wording clarification.
- Independent activity lifecycles, distinct clocks, private implementation scheduling, and state-only work without an optimizer offer.
- Production versus provider selection/packing ownership, per-consumer admission, actual provenance, variant identity, derivative paths, and accepted stochastic/exact behavior.
- Complete publication units, partial coverage, payload-versus-index failure, shared storage identity, outstanding-read lifetime, and bounded resource/backpressure policy.
- Validation scope/reduction/provenance/derivative/effect meaning; sampling compatibility and separate generation/publication/reporting.
- Shared access through actual last use, temporary projection cleanup, uncertainty gating, separate primary/cleanup failures, and required adaptive feedback independent of optional sinks.
- Distinct owner continuation contributions and the requirement for a coherent cut.

These checks establish specified meaning across the cited sources. They do not establish that every recorded algorithm/backend is supported, that a particular scheduler is fair, or that the current production Trainer realizes the target.

### Coverage limits

This was a concern audit, not exhaustive verification of all repository specs, every research case, or all production paths. Sampling family implementations and all other evaluation families were not each independently traversed. Current-code configuration restrictions, all launcher paths, detailed preparation/artifact schemas, and performance/cost claims were not re-audited. Strict validation/review completion and prior local correction diffs were not independently verified.

Bounded simulations do not settle autonomous progress, many-to-many joins, concurrent cancellation/publication, distributed reduction, real RNG isolation, physical alias/backend restoration, crash behavior, or coherent recovery. The design already names relevant conformance obligations; missing concrete mechanisms alone are deferred implementation choices, not findings of missing semantics.

## 7. Unresolved work routed to neighboring passes

| Route | Question to carry forward |
| --- | --- |
| **Pass 1 / Pass 2 follow-up** | When their local corrections are incorporated, recheck only affected accepted-arrangement/ref and preparation-publication correspondences. Do not infer them present at this SHA. |
| **Pass 5 — completion and coherent recovery** | Can producer, provider, coordinator, algorithm, optimization, backend, and resource contributions form the promised cut, including partial joins, late results, uncertain cancellation/handback, and unfinished effects? Are exact replay, reissue, skip, and unrecoverable work distinguished? Owner-local pause/index capture must not overclaim. |
| **Pass 6 — durable schema/reporting integration** | Do durable result/provenance associations retain actual work, source-state, output/resource, partial outcome, and cleanup/observation distinctions? Can required feedback and reporting retries remain separate without recomputation or duplicate adaptation? Detailed durable schema choices are outside Pass 4. |
| **Pass 7 — combined whole-run sufficiency** | Independently test the combined cases: streaming/cache consumption plus validation; two producers plus partial unit outcomes; shared storage lifetime plus cancellation; training/imperative handback plus generation; adaptive feedback plus continuation. Preserve source authority and evidence levels when assessing completion claims. |

Concrete access, scheduling, publication, and recovery mechanisms remain for the implementation that claims them. P4-REV-01 recommends only a localized timing clarification. No new universal scheduler, result/state container, concrete G5 interface, or expanded backend support is proposed.

## Source references

All repository links below are pinned to the audited commit. Requirement names in the report identify the governing semantic sections; ranges are navigation, not a substitute for their complete requirements/scenarios.

[authority]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/design.md#L19-L85
[correction]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/docs_design/models-strategy-trainer/notes.md#L1145-L1206
[d5]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/design.md#L512-L570
[cache-design]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/design.md#L571-L744
[eval-design]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/design.md#L745-L869
[d10]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/design.md#L2016-L2125
[conformance]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/design.md#L1907-L2015
[reconciliation]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/design.md#L85-L311
[cc]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-capability-coordination/spec.md
[cc-select]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-capability-coordination/spec.md#L8-L58
[cc-cache]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-capability-coordination/spec.md#L59-L250
[cc-val]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-capability-coordination/spec.md#L252-L298
[cc-sample]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-capability-coordination/spec.md#L299-L342
[cc-protect]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-capability-coordination/spec.md#L343-L392
[cc-recovery]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-capability-coordination/spec.md#L393-L577
[ex]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/accepted-training-execution/spec.md
[ex-run]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/accepted-training-execution/spec.md#L66-L111
[ex-life]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/accepted-training-execution/spec.md#L112-L160
[ex-input]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/accepted-training-execution/spec.md#L254-L308
[ex-correlation]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/accepted-training-execution/spec.md#L309-L348
[ex-use]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/accepted-training-execution/spec.md#L358-L463
[contract-timing]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-contract/spec.md#L122-L156
[participant]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/run-participant-state/spec.md
[opt-clocks]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-optimization/spec.md#L177-L227
[obs-feedback]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-observability/spec.md#L117-L146
[persistence]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/training-artifact-persistence/spec.md
[loaded]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/loaded-model-components/spec.md
[refs]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/specs/optimization-target-refs/spec.md
[main-metadata]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/specs/model-family-metadata/spec.md
[provenance]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/expand-model-metadata-identity-and-structure/specs/model-source-provenance/spec.md
[tasks]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/tasks.md
[anima]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/research/anima-llm-adapter.md
[conditioning]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/docs_design/future_ideas/text_encoder_conditioning_research.md#L1-L110
[data-research]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/research/training-data-pipeline.md
[distributed]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/research/distributed-training-execution.md
[async-data]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/docs_design/future_ideas/async_data.md#L321-L466
[shards]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/docs_design/future_ideas/data_shards.md#L470-L550
[llm]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/research/llm-training.md#L716-L784
[rollout]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/research/preference-and-rl-post-training.md#L310-L710
[twopass]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/openspec/changes/rework-model-strategy-trainer/research/trainer-architecture-concrete.md#L375-L471
[producer-test]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/tests/unit/training/test_input_producer_experiment.py#L1-L380
[policy-test]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/tests/unit/training/test_input_policy_experiment.py#L1-L309
[run-test]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/tests/unit/training/test_run_language_experiment.py#L1-L484
[imperative-test]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/tests/unit/training/test_imperative_authority_experiment.py#L1-L235
[coordinator]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/library/training/execution.py#L240-L608
[trainer]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/library/training/runners/trainer.py#L256-L307
[cache-code]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/library/data/caching_engine.py#L252-L355
[delivery]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/library/data/dataloader.py#L311-L337
[sdxl-cache]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/library/strategies/sdxl/caching.py#L472-L630
[helper]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/library/training/phases/orchestration_helpers.py#L49-L114
[validation-code]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/library/strategies/sdxl/validation.py#L18-L196
[sample-code]: https://github.com/Enferlain/sd-scripts/blob/6db3f1ce83482e7d3a5fd22bd7b0dfa01b5eca48/library/training/sample_generation.py#L435-L625
