# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Rules:
- Use proper sub titles "Added", "Changed", "Removed" and "Fixed"
- Keep proper track of days for where entries should go
- Be concise but mention all changes without necessarily detailing each one

## [2026-11-23]

### Added

- **A gradient norm that is not finite fails the run** — the window end reads each unit's gradient norm (clipping measures it; without clipping one more pass does) without the host waiting: the norm is copied to host memory and read at a later window end, before stepping. When its work has ended, an executable run asks each owner that checks late to settle (`runstate.generation.Settles`), so the last windows' norms are read too; a reason is reported as a `settle_failed` fact and fails a run that succeeded, and a run that failed or was cancelled keeps that outcome. A newly prepared unit takes over the norms still on their way. Reading the norm before the step instead was measured on the SDXL step: no step-time difference beyond noise, consistent with bitsandbytes' AdamW8bit step being bound by its host launches (about 360 ms per step, `sd-scripts-syv.70`).
- **Tests for Trainer optimization's owner** — a publication that changes a unit is refused while the unit's update lease is out and installs after its release, and leases taken on other threads never see their prepared unit change.

### Changed

- **The non-finite rule and the run's end in the delta specs** — `training-optimization` gains "A non-finite gradient fails the run unless an accepted policy skips its step": the check may lag, the run fails at the latest when its work ends, and the unit does not advance once the failure is found. `accepted-training-execution`'s "Completion and failure follow the accepted activity relationships" gains a scenario for an owner that checks the run's work late, and `training-artifact-persistence`'s capture requirement one for gradient norms still being read when a product captures a unit. Tasks 16.3 and 18.3 list them; the schedule decision stays in the design, since no spec covers formulations yet.
- **Chunk C's decisions in the rework design** — D15: precision is per role, and autocast's dtype is stated in the strategy on an activity, inherited and overridable; checkpointing is a preparation choice at launch, gated on grad mode; a selection that needs the GIL costs only its run, which reports the first selection that turned it on. D21: SD 1.x, 2.x and SDXL are one UNet component with the reference's parameters, each assembly shipping its configuration; a training loss is composed from calls cut where the choices vary independently, and a formulation takes its noise schedule as a value, so zero-terminal SNR is a transform on the schedule (decided 2026-11-23). D17 and D20: the late gradient-norm check and the run's end settle.
- **The retained-result release rule, written down** — the 2026-11-16 fix changed only the runtime and the reference semantics. D16 and the `workflow-composition` spec now state it: a result computed from a retained result keeps every lease that result keeps, so the earlier one may be released first. The verifier no longer refuses a normalized form that releases a kept result first, and still refuses an update between the two releases (`iro-lease-004`). Runtime, reference and verifier tests cover a loss that leaves its access while the prediction is released inside. D14's first retained-product profile points to D16's extension.

### Fixed

- **Grad mode and device residence at a filter** — preparation treated a filter body's keep decision as the filter's result. The deciding call got grad mode when the kept item reached a differentiated input, and the host did not wait for a decision computed on the device, which the filter reads on the host. A filter's yield is no longer treated as a result.
- **ty on the new system's tests** — three errors from 2026-11-16 and three from section 14; ty now passes on the new system with its tests.
- **A flaky run-state test** — `test_a_route_never_changes_under_a_lease_held_on_another_thread` failed when its workers finished before the publishing loop started, which then attempted nothing. The workers now lease until the publications are done.

## [2026-11-16]

### Added

- **Training's words (section 16, first pass)** — `library/training/words` extends the contract with `iro.train.unit`, `iro.train.window` and `iro.train.contribute` and the standard ownership profile. Each is a derived word over existing protocols: a unit becomes a resource Trainer optimization holds, a contribution an owner update of the unit, and a window a repeat over its body followed by one access that takes the unit's and its participants' update leases. Their written forms mirror the Python face of D18; a unit's gradient-norm limit is `max_grad_norm`.
- **Materializer settings** — values in the strategy, such as the checkpoint a materializer loads, follow its arguments (`materialize(@load_sdxl_unet, "…")`), reach the authority seed and are passed to the materializer as its inputs, checked against its declared inputs.
- **The training core (section 16, first pass)** — `library/training/units` holds Trainer optimization: the plan derived from the accepted strategy, candidate resolution in module traversal order (tied weights merged as aliases, overlap and empty units refused), prepared units with the optimizer built on the final parameters, live state with the standard profile's contribution (backward into the unit's own parameters only) and window end (clip to `max_grad_norm`, step, zero, progress, an `Advancement` record), and the owner of every unit at the owner seam with its own readiness. `library/training/preparation.py` prepares the routes and realizes the units in one in-place attempt and sets trainability from unit membership; `library/training/trainer.py` carries an accepted run from cold load through materialization, preparation and readiness to an `ExecutableRun` (in `library/runstate/generation.py`), which runs once. A small network trained through the whole host sequence matches the same step written by hand exactly. Two units declaring the same participant are rejected when the strategy is read.
- **Grad mode from the dataflow** — implementations declare which inputs they differentiate and which they are differentiable through; preparation gives a call grad mode when its results reach a differentiated input, records where its derivatives go, and no longer fuses calls whose contexts differ.
- **Per-call setup on the owner seam** — a lease may establish what its state needs for each call (`seams.Establishes`), on the worker before the call, and the call then uses that resource alone. Training's routes use it for module mode: train mode for the modules of the units a call's derivatives go to, evaluation mode for the rest, set only where it differs. The IR itself no longer sets module mode and carries no training rule.
- **In-place preparation before a participant's first route** — offered where there is nothing to withdraw; the authority refuses the result if a participant was leased while the work ran. A whole participant is a target kind, selected as `@name`.
- **`library/optimization/construction.py`** — builds an optimizer by its registered name and imports only that one, so bitsandbytes loads only when AdamW8bit is selected. Building any `torch.optim` optimizer still turns the GIL on in this environment (`torch._disable_dynamo` imports dynamo, which imports Triton).

### Changed

- **Section 16's opening decisions in the rework design** — D16: materializer settings are values in the strategy; D17 and D20: the end of a window takes the unit's and its members' update leases in one access; D19: in-place work withdraws the guarantees that exist, so before a participant's first route there is nothing to withdraw.
- **The baseline's pass criteria** — refined after an environment probe: the Windows baseline serves performance, resource usage and obvious problems, exact correctness is checked against a reference made in this environment with deterministic algorithms, and every throughput measurement records the GPUs' tensor-unlock state.
- **Normalization** expands later derived words first, so a word still finds what it names, and reports every expansion that fails (`iro-norm-009`); a definition's form is digested by its code, never by an object's address.
- **Chunk B's decisions in the rework design** — D15: grad mode from declared differentiation, and module mode as what a lease establishes per call; D16: the owner seam's per-call setup; D20: module mode follows trainability through the routes training prepares, a contribution backpropagates only into its unit's parameters, checkpointing is non-reentrant, empty units are refused, and data parallelism covers DDP and FSDP2 with per-unit sync scope; D21: Trainer optimization lives in `training/units/`.
- **The registry in `library/optimization`** imports its config type only for type checking, so the new system can build optimizers from it without depending on config.
- **The dependency rules** give each library its layer: it may import the run state, the IR, observability facts, base and the libraries below it.

### Fixed

- **A retained result kept by a result its body yields** was released inside the body, which both the runtime and the reference semantics refused, so a loss computed from a retained prediction could not leave its access. A result that keeps another holds its leases too, so the kept one may now be released first; releasing one a call is still using stays refused.

## [2026-11-15]

### Added

- **`library/runstate` begins (section 15, first pass)** — the owner pattern (references and their mints, versions, dependencies, evidence, attempts), generations with one-store publication that stays correct without the GIL, target references with component-qualified and participant-qualified selectors, and the run authority: seeding from an accepted run, materialization attempts that run the declared materializers as one job, route rebinding, readiness at a checkpoint, and leases at the owner seam through the execution projection. What this pass leaves out fails with its reason: relationships, views, snapshots of participants the run updates, grants and materializer settings.
- **Preparation attempts in `library/runstate`** — `PreparationAttempt` derives one job from one generation, reads its participants through the `Preparation` projection pinned to that generation, and publishes the authority's routes with every named contributor's section in one publication, or nothing when the result is stale, failed, partial or refused. A strategy that both snapshots and updates a participant is now refused at preparation and readiness instead of at the use.
- **The IR's admission, freshness and protected-use cases run against the real authority** — the conformance fixtures, exact-current and bounded-lag admission, and requested work reading its reserved version, with the participants bound, prepared and leased through the run authority.

### Changed

- **Section 15's opening decisions in the rework design** — D16: an owner's versions advance through the runtime, which tells the update lease after each declared update; D19: how a version maps onto the owner seam, that the runtime enforces publication conflicts while the authority refuses to publish over its own leases, and who judges which obligations; D20: an advancement updates its members' weights through update leases on them.
- **The IR's update path and obligations** — the runtime tells `seams.Lease.updated()` after a call instead of asking the owner for its version, and the reference semantics advances every declared update, even one that changes nothing; obligations carry a kind and an evaluator, a lease obligation carries its use, and a resource other than a participant is judged by its own owner.

## [2026-11-14]

### Added

- **`library/iro` gains its runtime and device adapters (section 14)** — preparation for a target (implementation matching, effect checks, fused call groups, device admissions, implementation manifests keyed on code, cold loading of prepared forms, participants' devices and which never change), lowering to numbered slots and bound steps, and a coordinator split into components (kernel, turns, device storage, protection, requests, flow, calls) that realizes every lifecycle protocol and is checked against the reference semantics under explored interleavings. Device adapters: the host, and a shared PyTorch base with CUDA first (per-call settings, one stream per device, completion as events that order later work). Owners' device storage is work the runtime queues on the device's stream: participants permitted to leave the device between uses do so, and snapshots of device state are copies ordered between the participant's calls, or the live storage for participants the run never updates. Implementation and definition versions are pinned with declaration digests.
- **Checks for the runtime** — witnesses for bounded bookkeeping and coordinator progress, GPU tests run on an RTX 3080, the G3.7 cost probe and `cost_gate.py` (dormant work passes; the cheap action is 181-244 microseconds over direct Python against 2-7 allowed, the target of the overhead pass before section 18), and a fresh-process test that importing the new system keeps the GIL off. The new system's tests run with the GIL off via `--confcutdir=tests/unit/iro`.

### Changed

- **Rework design, section 14 decisions** — D15 states the capture order, the in-place evaluation-point swap as runtime device work with other routes excluded (D20), that an implementation joins its own streams before returning, and that a copy read on another stream is marked for it; D14 says an `at_request` use belongs to its innermost handler and a request holds one version per resource and kind. Section 14's remaining items moved to the sections that first need them (capacity admission to 23, RNG ownership to 16, undeclared-mutation qualification to 26, the overhead pass to 18).
- **The research language package is retired** — its coverage row is checked and it moves to `docs_design/models-strategy-trainer/_deprecated/xdsl_language/`.
- **The permission to leave the device has its spelling** — D15 and D16: `may_leave_device` on a participant's declaration lets its device storage leave between uses; preparation rejects it on a trained participant, a derived route's copy gets a route form when a strategy first needs it, and where the storage waits stays a target choice (pinned host memory first), exact copies only.

### Fixed

- **Written form and requests** — printing no longer numbers values it does not show, so a strategy and its printed form normalize alike; an outer handler's requests no longer reserve a nested handler's reads; two `at_request` snapshots of one resource in one handler no longer lose a lease.

## [2026-11-13]

### Added

- **`library/iro` completes its language half (section 13)** — normalization (derived words expanded with their ancestry, releases placed after a retained result's last use, relationships resolved), verification (claims, leases, accesses, calls, lifecycles, streams and rank agreement) with obligations at their readiness checkpoints and a reproducibility report, the accepted run with its authority seed and cold loading, the owner, implementation and device seams, and the reference semantics, a deterministic lifecycle interpreter with stand-ins.
- **A conformance suite and a dependency test for the new system** — the research fixtures rebuilt as written forms and run against independently computed expectations, the test-only accepted-run builder, and `tests/unit/test_dependency_rules.py` checking D21's rules. AGENTS.md and CLAUDE.md describe the build-new approach and the rules.

### Changed

- **The written form marks normalized strategies** — `iro.strategy @name normalized`, with `from(...)` naming the words an expanded statement came from; printing goes through the contract's forms, and long choice headers wrap like other bodies.

- **Section 14's open decisions are settled in the rework design** — D15 now says what one device admission covers (one access body, from its taken inputs to the release of its last device-holding result), that work in turn shares one runtime-assigned stream per device (measured: separate streams force a device-wide synchronization per turn when memory is short), that completion orders later work instead of making the coordinator wait, the rules for overlap once realized, and how a participant's device storage leaves between uses, with the other offload mechanisms placed with their owners. D16 adds implementation and definition versions: what one covers, declaration digests, pins of only the definitions a run uses, and what happens when pins are gone. The GPU run on the research code (bead `sd-scripts-syv.35.4`) is dropped; its measurements move to task 16.2. Overlap's realization moves to section 23, and new-run-from-snapshot and provider identity in pins are recorded for sections 24 and 17.

- **Device adapters and owners' device storage are settled in the rework design** — D15 gives one device adapter per PyTorch backend on a shared base (per-call settings, completion through PyTorch's streams and events, and storage copies), each reporting what it offers so preparation refuses a run that needs a missing feature; CUDA comes first and backends no one has run are recorded as untested. Every read or copy of a participant's device storage outside a call is device work the runtime queues on the device's stream, so a snapshot or save captures exactly its version; an owner never copies device storage itself (measured: an off-stream copy saw the value from before a queued update). D16, D19, D21 and tasks 14.1 and 14.5 follow.

- **ROADMAP.md opens with the rework as its current direction** — the research progress log is replaced by a short description of the rework, links to its design, tasks, history and baseline, and the open topics it does not settle. The remaining items are unchanged, pending a detailed review.

### Fixed

- **A call inside an access found its pinned version** — the lookup stopped at the access's resource uses instead of the strategy's `uses` header.

## [2026-11-12]

### Added

- **`library/iro` begins with Kuroiro's core words and written form** — operation contracts, diagnostics, the core words with their shape checks, the composed contract, and the written form's reader and printer; D16's data-only example prints back exactly. xdsl 0.75 is now a dependency.

### Changed

- **Task 13.5's names and spellings are settled in the rework design** — `library/iro` and its modules are confirmed; `choice` yields values and can choose on an `optional`; one access takes several resources' leases; `transfer` is a hand-over, not an operation; `invoke` and `once` are dropped; value types use Python's names.
- **The baseline README explains how to read the historical timing range** — slow runs may reflect their conditions; the fast end is reachable, never an outlier.
- **Working instructions follow the rework** — CLAUDE.md and DEVELOPMENT_GUIDE.md drop the Windows-filesystem fallbacks, use current examples and describe the review → commit → next cadence; the guide marks the sections that describe today's config and launcher, and states the changelog rule and the roadmap's role as a horizon board.

### Removed

- **Stale root documents move to `docs_design/archive/`** — `TODO_PRIORITIZED.md`, `TODO_RESOLUTIONS.md`, `BENCHMARKING.md`, `INTEGRATION_TESTING.md` and `cpython-314t-blockers-report.md`.

## [2026-11-11]

### Added

- **The trainer baseline is pinned and recorded (section 12)** — `d880bd2e` is tagged `rmst-baseline`; three SDXL fine-tune and three SDXL PEFT timing runs, three recorded steps per mode with their tensors, the G3.7 cost probe, both environments, the multi-GPU error log and rerun instructions live under `docs_design/models-strategy-trainer/baseline/`.

### Changed

- **Baseline tasks cover the two configurations the trainer has been run with** — later comparisons without a recording are reworded, and throughput is read against the logged runs instead of a fixed allowance.

## [2026-11-10]

### Added

- **SDXL written-form reading probe** — `docs_design/models-strategy-trainer/sdxl_written_form_probe.md`, with its findings settled in the design.

### Changed

- **G5 is closed and the rework's production shapes are signed off** — requested work can select its version at request; state selection, device admission and placement are independent decisions; reads and updates come from lease types; every amendment a run can make is declared in the accepted strategy; an access covers what its route depends on, and a leased result stays protected until its last use. Review fixes, the rejection-rule audit and vocabulary alignment (implementation, lease, disposed claims, association) apply across both changes; design tasks 5.1–5.4, 5.6 and 5.7 are closed.
- **The IR system's package is renamed from `library/workflow` to `library/iro`.**

## [2026-11-09]

### Changed

- **The rework design gains its production shapes and G5 placements** — the design, proposal, delta specs and tasks describe the production system; `library/conditioning`, `library/artifacts` and `library/evaluation` are placed, along with data's boundary, strategy-stated input sources, the generated feature listing and launch of strategies the repository did not make; "recipe" becomes "derived word". Task 5.5 is closed, with matching updates to the metadata change.

## [2026-11-06]

### Changed

- **Historical rework sources now live in the design archive** — the history index connects the decision trail to byte-preserved originals and explicit path mappings. Active OpenSpec references follow the move; runnable experiments and retained evidence keep their existing paths.

## [2026-11-05]

### Added

- **Scoped products now retain storage and derivative dependencies through physical use** — explicit dependency/release contracts reject escape and premature reuse, while real buffer views and CPU autograd products execute through the independent reference and existing owner target. Cold correspondence, erased histories, cancellation and late physical failure are qualified; production ownership APIs and bookkeeping reclamation remain open.

## [2026-11-04]

### Fixed

- **Shared language acceptance now checks ordered SSA and binding update authority** — direct uses, captures and recipes reject unavailable definitions; preparation checks actual update operands, capability issuers and resource identities before factories or physical entry. Runtime authority/version checks and prior cold artifacts remain intact.

### Added

- **Shared workflow contracts now govern filtering and local fusion** — one selected read/update/unknown-effects descriptor supports preparation and transformation legality. Explicit zero-or-one filtering preserves demand, provenance and claim settlement; an xDSL SSA rewrite reduces local dispatch while preserving stage physical/validation boundaries. The independent reference, erased owner runs and cold negative controls qualify the bounded host case.
- **The first shared xDSL slice gains integrated qualification** — seeded reference comparisons, explicit owned results after history erasure, local-stop/cancellation and late physical-failure controls, and fresh normal/optimized cold runs retain independent negative controls and exact evidence. The CPU owner target and production/language limits remain explicit.
- **Shared xDSL workflows now derive prepared owner execution** — detached artifacts and recomputed cold correspondence drive model, data-only and rearranged compositions through the existing owner engine. Explicit snapshots, protected current access and actual-version admission retain request, physical-drain and partial-update evidence; production integration and broader schedule qualification remain open.

## [2026-11-03]

### Added

- **The shared xDSL language slice now verifies composition contracts** — typed IR, normalization and a Torch-free lifecycle reference support model and data-only workflows. The shared verifier checks pending work, stop coverage, ownership, access/version policies and decision declarations, with independent lifecycle and admission controls. Prepared owner-target execution remains the next milestone.

## [2026-11-02]

### Added

- **Matched machinery evidence now covers all six xDSL/Pliron cases** — source-linked framework/domain inventories, paired diagnostics and release-build CPU observations retain actual API boundaries, native request/serialization spans, process costs and separate peak-memory definitions. Collector corrections and failed probes remain recorded; the substrate recommendation is a separate task.
- **Pliron mixed workflows now connect feedback and changing model state** — typed producer/repeat regions, policy acknowledgements, protected production snapshots and checkpoint/drain dependencies independently prepare for the existing owner engine. Both frontends check saved plans before factories; numerical reconstruction, archive erasure, bounded exclusion progress and physical failure/cancellation controls pass. Matched diagnostics, costs and the final substrate decision remain open.

## [2026-11-01]

### Added

- **Pliron A now represents correlation and retained accumulation** — typed stream/model/unit relationships, nested local preprocessing, atomic admission and requested current-state evaluation are independently verified and prepared in native IR. Both frontends reuse the extracted owner target after their own cold checks; independent arithmetic, rejected-member non-consumption, retained partial contributions and physical/mode cleanup are qualified. Mixed ownership and final comparison costs remain open.
- **xDSL and Pliron now share the intended frozen-read grouping policy** — compatible typed unary stages group across the pair extension, while a valid opaque offset remains a barrier. Resource summaries drive rewriting and cold legality in both candidates; project-owned dependency caching in xDSL and native pass-managed invalidation in Pliron are recorded separately. Per-stage outcomes, source identity and physical draining remain checked; counts alone establish no performance advantage.
- **Pliron C adds reciprocal authority representation to the comparison** — actual typed phase and handback chains, independent native preparation and cold correspondence reuse the existing grant/optimization owners and selected numerical callable. Both C frontends validate saved plans before model factories; independent arithmetic, unverified restoration, physical cancellation/drain and partial retained outcomes are checked. Grouping alignment and the A/mixed ports remain open.
- **Matched controls and an xDSL lifetime port begin the framework comparison** — descriptive fixtures freeze the six existing meanings; live/cold B and valid opaque-offset controls use both candidates, while independent xDSL lifetime verification/preparation reuses the existing owner target. Pending evaluation, partial admission, physical failure/drain and cold correspondence are checked; the remaining ports and grouping alignment stay open.
- **The next xDSL/Pliron comparison has a bounded research plan** — six existing case meanings, reciprocal coverage work, aligned frozen-resource grouping, independent correctness gates and separate machinery/integration/cost evidence lead to a decision record. This preparation leaves language design open and starts no implementation ports.

## [2026-10-31]

### Added

- **Pliron now represents a bounded different-lifetime composition** — an independent producer serves two accumulation windows and requested evaluation through one existing owner engine. Native pending-work, stop, access and settlement contracts survive cold verification; independent checks cover late/prefetched input, partial-window stop/cancel, physical draining and failure without selecting a production language or framework.

## [2026-10-30]

### Added

- **Pliron's independent local extension now uses shared payload/resource contracts** — a separate pair dialect and immutable coefficient binding run through the existing scalar target, while cached resource analysis permits known frozen grouping and retains an executable opaque-effect barrier. Cold correspondence, diagnostics, domain outcomes and physical draining are checked without selecting a production language or framework.
- **Pliron 0.20.0 now has an isolated data-only feedback comparison** — real semantic/prepared operations, graph/SSA region checks, verified cold correspondence and local fusion use native framework infrastructure while reusing B's Python runtime and independent oracle. Analysis invalidation is exercised; xDSL remains the provisional baseline and native backend sharing remains unproven.
- **Candidate comparison now separates provisional choices from open language decisions** — the assessment maps original/xDSL evidence and reusable checks, identifies xDSL as a viable working baseline and protects readiness/execution/effects without prescribing preparation order; language design and final implementations remain unselected.

### Fixed

- **Pliron document verification now requires complete input** — semantic acceptance, preparation and cold loading reject trailing garbage or extra roots after permitted whitespace. The experiment's requirements file now has a local ignore exception so fresh-environment instructions can be included in a normal commit.
- **Mixed xDSL selection now yields after bounded filter work** — long excluded stretches permit sibling work, feedback and cancellation without filling the admitted-ticket horizon.
- **Local outcome contracts now require the runtime baseline** — acceptance and cold resolution reject matching operation/binding declarations that omit returned, invalid-output or raised outcomes.
- **Scalar ticket retirement now runs with Python assertions disabled** — removal and the identity guard no longer depend on `assert`; optimized B/extension execution and fused archive-free cancellation/failure cases are qualified.

## [2026-10-29]

### Added

- **xDSL local composition now admits an independent operation and payload schema** — common interfaces and decoded local signatures verify codecs, immutable resource identity, outcomes and lowering before execution. The same scalar workflow/runtime carries a typed pair; independent payload checks catch hidden corruption, and conservative fusion preserves resource/schema barriers.
- **xDSL research now combines continuous selection with shared model ownership** — one existing engine handles dynamic policy feedback, protected trainable-model preprocessing, atomic admission and ordinary updates; a real blocked-handoff/physical-read stop case retains failure and interruption evidence. The explicit checkpoint and CPU profile remain bounded research.
- **xDSL research now compares unfused and fused local data dispatch** — an actual prepared-IR rewrite groups eligible stages while retaining per-stage validation, source and failure facts. A/B independent conformance, physical interruption/drain checks and a C no-op guard the boundary; bounded raw local costs are recorded without a production speedup claim.
- **xDSL research now represents offered two-pass authority and retained advancement** — typed handback and explicit phase/authority contracts reuse the existing grant, protected-state and optimization owners. Independent numerical and failure checks distinguish restoration from verified handback, drain physical work through interruption, and retain uncertain optimizer/reset outcomes; the fusion comparison and framework choice remain open.
- **xDSL research now integrates correlated model/data work with existing owners** — explicit sample/window mappings, atomic two-path admission, two-contribution accumulation and current-at-start requested evaluation reuse the candidate's state, optimization and physical-drain services. Independent numerical, failure and reconstruction checks preserve the data-only path; framework selection remains open.
- **Isolated xDSL research now executes a data-only feedback workflow** — typed semantic and prepared IR reconstruct without the authoring object; independent checks cover actual policy provenance, aggregate outstanding bounds, feedback under blocked publication, partial failure and physical drain through cancellation. Production interfaces and framework selection remain unchanged.

### Fixed

- **Mixed xDSL feedback now follows the actual admitted input through the owner sequence** — control no longer reads the last contribution from a recorder. Archive-erasure checks preserve learning, feedback, physical draining and numerical results with source/learning observations disabled.
- **xDSL preparation now verifies decoded plan correspondence** — A/B reject altered target/bounds/control descriptors, substituted stage bindings, missing/duplicate calls and incorrect source ancestry. Fixed backend protocol fields are checked explicitly before runtime construction, with regression evidence for the external review's corruption cases.

## [2026-10-28]

### Added

- **Candidate conformance now has reusable independent checks** — test-only adapters expose input/source associations, adaptive feedback order, numerical results and partial optimization outcomes. Independent arithmetic and deliberately corrupted reports detect shared mistakes across lowerings while permitting private scheduling differences; runtime and governing requirements are unchanged.
- **Candidate RNG probes now vary worker timing explicitly** — job-owned randomness and ordered input handoff preserve training results across worker order/count changes. Global RNG, shared-stream and changing-source controls expose distinct limits; scoped dropout and protected source boundaries reuse the existing ownership mechanism. Production APIs and governing requirements remain unchanged.

### Fixed

- **Candidate worker submission now retains execution ownership through ambiguous errors** — a dispatch owns its entry gate and completion before executor handoff. Failed submission either prevents later selected entry or drains already-entered work while preserving effects, interruption and failure evidence; queued callbacks can no longer mutate state after its protection was released.

## [2026-10-27]

### Fixed

- **Candidate input associations now survive diagnostic-history eviction** — immutable identity/provenance references follow accepted regions, pending ordinary/granted contributions, capability requests and failure reporting. Prepared input-member bounds and existing owner lifetimes govern retention; the original four counterexamples now pass with cancellation, cross-block and release checks.

### Added

- **Connected candidate assessment now incorporates the completed concurrency and free-threading investigations** — traces graph-derived work, coordination and state ownership, distinguishes production decisions from backend qualification, and documents the four-variant input-association counterexample addressed by the lifetime fix. Governing requirements remain unchanged.

## [2026-10-26]

### Added

- **Candidate access-cost research now compares preparation-selected coordination** — ordinary blocks can omit unused retained-source borrowing and empty input-demand bookkeeping while keeping the same lease protocol. Conformance cases cover producer conflicts, publication, replacement, retained derivatives, cancellation and partial failure; matched GIL-disabled CPU measurements retain raw samples and limits. Production training and producer defaults remain unchanged.

## [2026-10-25]

### Added

- **A repository-local free-threading skill** — a pinned Tiran workflow includes its supporting references, license and CPython contributor guidance, with standalone links and repository-specific scope notes.

### Changed

- **The isolated execution candidate now uses eager functional SGD** — a small adapter reuses PyTorch's update implementation without the optimizer-base Dynamo/Triton import. Fresh-process checks keep the GIL disabled through the nine examples and bounded worker/provider cases; standard-optimizer numerical oracles remain independent. This changes neither production training nor the selected optimization semantics.
- **Candidate input capacity and protected access now share runtime progress evidence** — task-attributed leases, prepared consumer relationships and live demands detect observed cross-owner wait cycles without changing work order or input admission. Compatible reads and alternative consumers remain useful; unknown private waits remain unqualified.
- **A matched candidate probe now measures useful parallel Python work** — bounded text preprocessing/packing runs through the real provider, handoff and optimization machinery under GIL-on/off and one/two-worker configurations. Retained samples show about twice the throughput for this synthetic workload with two GIL-disabled workers; this is not a production training speedup claim.
- **CPU autocast research now distinguishes cache safety from useful reuse** — mixed-precision overlap tests and matched forward/gradient timings expose the cost of blanket uncaching, verify region-local differentiable cast reuse, and retain the limits of selective native caching. The conservative worker default and production interfaces remain unchanged.
- **Module-level precision-view research now exercises unchanged forward code** — private structures preserve source parameter identity and gradients in a qualified subset; negative cases expose shared functional-call mutation, precision-policy changes, shared buffers and submodule-alias restoration. No production view or cache API is selected.

### Fixed

- **CPU candidate workers no longer inherit Torch's shared autocast weight cache** — overlapping bf16/fp16 reads reproduced a dtype mismatch despite correct thread-local settings. The worker realization now disables that cache while preserving grad/inference/autocast modes and dtype; focused overlap, state-ownership and RNG cases qualify the supported GIL-disabled boundary.

## [2026-10-24]

### Changed

- **Isolated training-execution research now shares pending optimization ownership** — ordinary backward and granted two-pass work use one entry/handback lifecycle; retention, publication, replacement and capture consult the same unfinished-work evidence. Advancement consumes readiness before fallible mechanics and preserves per-unit partial outcomes. This does not change the live Trainer or select production APIs.
- **Candidate state access and lifetimes now share one owner protocol** — retained-read borrowing, provider source authority/effects and region cleanup move out of lowering; preparation uses the same acquisition/reuse rules for cooperative and worker targets, replacing the older coarse contract check. Closing ends token borrowing before physical release; failed release preserves unfinished evidence and withholds affected current state. Regression cases distinguish released sequence access from aggregate block leases without changing production training.
- **Candidate mode protection now includes provider-owned native workers** — preparation derives physical exclusion from declared provider source access as well as selected-call placement. Shared checks cover block, sequence and retained-source uses; deterministic thread cases verify compatible overlap, conflicting-view serialization, cancellation/drain and honest pre-load rejection without changing strategy authoring or production training.

## [2026-10-11]

### Changed

- **The project now uses a shared uv-managed Python 3.14t baseline** — interpreter, Ruff and ty targets align on Python 3.14; uv 0.12+ installs the project dependencies and default development tools from the root lock. Torch 2.14.1 / torchvision 0.29.1 use the CUDA 13.0 index, replacing the old Torch/CUDA profile matrix and temporary CPU-only research group.
- **Dependencies and tooling were refreshed for the new baseline** — updated the Hugging Face, numerical, image, optimizer and development stacks; selected Hydra and safetensors release candidates, configured the OpenCV source-build workaround, and enabled stricter pytest, Ruff and ty checks. Installation does not establish complete runtime/API compatibility. Native Windows support and optimizer migration checks remain outstanding; OpenCV, grpc and Triton can re-enable the GIL as noted in `pyproject.toml`.

## [2026-10-05]

### Fixed

- **ModelSpec extensions cannot overwrite canonical claims** — item validation and final projection reject extensions colliding with the specification version or canonical artifact fields, while preserving custom extensions. Stored records receive the same protection as newly filed facts.

## [2026-07-20]

### Added

- **Metadata query and granular model-structure research now have an explicit dependency path** — recorded the current snapshot-only read capability, promoted stable cross-run source-model provenance to a base requirement, defined the optional source-resolution boundary, and placed the queryable module/parameter/buffer catalog after source identity, query design, and qualified component linkage.
