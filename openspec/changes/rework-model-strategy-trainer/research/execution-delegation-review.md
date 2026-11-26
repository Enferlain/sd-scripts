# Review notes: delegated execution, requirements and resource-aware planning

Date: 2026-11-26

Status: research / non-normative. A review by Claude of `execution requirements and policies`, `resource-aware execution` and the live SDXL sketch (`sd-scripts-syv.77`). **Nothing proposed here is agreed.** It does not amend OpenSpec, approve an API or choose a scheduler.

Basis: branch `model-strategy-trainer` at `9055aa20`. Code and design were read, not run. External sources were only partly checked (section 7).

Labels: **Observed** (seen in the repository), **Inferred** (reasoning from it), **Proposed** (a suggestion to evaluate), **Unknown**.

## 1. Read with these notes

The SDXL sketch is the test case, so give agents the same copy of it as the two notes.

Paths below are under `openspec/changes/rework-model-strategy-trainer/` unless they start with `library/`. Line numbers refer to `9055aa20`.

| Priority | Read | Why |
| --- | --- | --- |
| 1 | `design.md` 2798-2893 | Access realizations, `at_request`, flow admission, reproducibility class |
| 1 | `design.md` from 3276, up to the **Collectives** bullet | Device admission, turns, overlap, leaving the device |
| 1 | `design.md` 3957-3961; `load_prepared` in `library/iro/prepare/prepared.py` | Saved prepared forms must match a fresh derivation |
| 1 | `design.md` 4906-4950 | What is refused, including silent substitution |
| 2 | `design.md` 3694-3716 and from 3894 | Core responsibilities; what an access protects |
| 2 | `specs/workflow-composition/spec.md` from 380 | Normative text on requested versions, overlap and leaving |
| 2 | `library/iro/runtime/protection.py`, `requests.py`, `turns.py` | Current reservations, snapshots and turns |
| 2 | `library/iro/verify.py` | Existing derivation of the reproducibility class |
| 2 | `tasks.md` 23.2 and 23.5 | Pending overlap, capacity, snapshot and reporting work |
| 3 | `library/iro/README.md` from about 195; `tasks.md` 14 notes | Conformance and the runtime cost gate |

## 2. Observed

1. The design asks the strategy for overlap permission and `may_leave_device`; the delta spec covers both. D14 has each use select an access realization. The sketch delegates these choices and states only the version read.
2. Streams require `outstanding` and `ready`; handlers require `max_outstanding` (`library/iro/words/data.py`, `lifetimes.py`). The sketch states neither.
3. The runtime branches on the authored access mode: `snapshot` captures, while a live `at_request` read reserves the version.
4. A snapshot lease can reuse live storage when the run never updates that resource (`Protections.hold`).
5. Preparation checks eligibility for a participant permitted to leave the device: not updated by the strategy, device work on one device, and copy support (`_leaving`). A write that no call declares is outside the design's ordering guarantee.
6. The device adapters define `capacity()`. A search of `library/iro` found no caller and no peak-memory recording in the library code. Task 23.5 says section 14 runs all work in turn.
7. A reproducibility class is already derived and recorded on the accepted run. It concerns schedule-dependent outcomes such as versions and admission. It makes no promise about computed bits.
8. `load_prepared` prepares the run again and refuses a saved form whose text differs.
9. Runtime `Fact` records have `owner`, `invocation`, `kind` and `values`. They have no timestamp or byte-count fields.
10. **Cost record, possibly stale.** `research/iro-runtime-cost-results.json` is recorded as of 2026-11-14. Its cheap action costs 181-244 µs over direct Python, against 2-7 µs allowed; dormant work passes. `library/iro/runtime/` changed afterwards, including `calls.py`. It was not rerun for this review. The project already tracks the fix as the overhead pass `sd-scripts-syv.53` (18.5), placed after sections 15 and 16 by the user, and the design requires the gate before section 18's SDXL run is accepted. It is a CPU measurement, not evidence about GPU step cost.

## 3. Inferred and proposed framing

- **Proposed: allowed runs.** Authored behavior, derived obligations and extra requirements limit the set of allowed runs. Preferences rank within it. The system chooses a plan that cannot leave it.
- **Inferred: allowed does not mean identical.** With bounded lag, prefetch depth may change the versions observed while every outcome stays legal. Matching a reference is a further requirement.
- **Proposed: define before representing.** State each candidate requirement over observable run events before choosing syntax, then identify how it can be established:

| Kind | Example | Establishing it |
| --- | --- | --- |
| Safety | No update waits on validation's live-state hold | Exclude that realization. This does not remove waits for capture or device turns |
| Liveness | Both activities keep progressing | Argue from scheduler and termination assumptions. FIFO turns give only a conditional guarantee |
| Hard time bound | Finish within a limit | Needs control or bounds over running work; delaying admission is not enough |
| Relation between runs | Throughput versus no validation; bitwise equality to a reference | Needs a reference and either a by-construction argument or measurement; measurement covers only what was tested |
| Static placement | Use separate devices | Filter plans. This alone does not prove isolation |
| Preference | Prioritize training | Rank eligible plans |

## 4. Gaps

### 4.1 Measured planning and saved plans

**Inferred:** if preparation depends on timings, free memory or prior runs that are not saved inputs, a fresh derivation may differ and be refused. Mid-run adaptation and per-rank measurements need records too.

Two options to compare:

- **A. Evidence as input.** Save the evidence snapshot and pass it to a deterministic preparation. Keeps re-derive-and-compare, but ties loading to reproducing the planner's decision.
- **B. Propose and validate.** Save the decision and evidence; a separate checker establishes eligibility on load. Allows heuristic planners, but needs a checker that is complete for the guarantees claimed, and changes the current D16 rule.

The review leans to B. Neither is designed. In both, current resources and target eligibility still need checking.

### 4.2 Unknown facts about code

**Proposed:**

- A fact needed to run the authored behavior correctly (what a result keeps alive, what a call mutates): use known evidence or the strategy's statement; otherwise ask.
- A fact needed only to enable an optimization (safe to overlap or recompute): leave that candidate out without asking.

Example: "the strategy never updates it" does not show that a library call never writes to it. If one does, restoring an old host copy loses the write. Automatic offload needs evidence of no writes, or a mechanism that handles them. This split depends on the sketch's direction that library code carries no IR declarations, so where that knowledge lives and how strong it must be are **Unknown**.

Running in turn with everything resident may need less evidence than other plans. That does not make it a privileged fallback, and it may not fit.

### 4.3 Validation retry and reporting

**Inferred, conditional:** D15 retries an overlapped out-of-memory failure only for work without owner effects. The sketch calls `iro.report` inside the batch loop, and task 23.5 lists reporting as an observation-owner update. If both apply, a failure after the first report cannot use that retry. Reporting is still open in the sketch. Options include publishing once at the end or making reports safely repeatable.

## 5. Open decision for the user

**If nothing else is stated, what must a delegated choice keep the same?** Agents should not treat this starting proposal as settled.

- Keep authored version and freshness rules, data order and association, owner effects, the precision recipe and complete required state.
- Allow no change to any call's random draws, at least until stream identity is designed.
- Promise no bitwise equality by default. The design notes that lower free memory can change kernel selection. A bitwise requirement excludes choices that cannot establish it.
- Where the strategy permits variation, such as bounded lag, accept any legal outcome and record it.

**Unknown:** the numerical standard. No bitwise promise must not mean that any error is acceptable. Each alternative needs a stated basis for computing the same operation, and it is undecided whether a kernel change needs a tolerance.

## 6. Suggested next case: `at_request` validation

Work through one GPU, for full fine-tune and adapter training.

| Config | Live-state hold | Snapshot |
| --- | --- | --- |
| Full fine-tune | No state copy; the next update waits for validation's uses to finish | Needs a second UNet, roughly 5 GB in bf16 (estimate, not measured) |
| Adapter | The trained adapter's updates still wait | Expected: unchanged base reuses live storage, changing state is copied. Confirm across route dependencies |

Both need validation working memory. The full state may include more than parameters.

Cover: the observed version; full state; memory exhaustion before and after a report; cancellation; wait cycles between the reservation and device turns; what is saved about the choice; and what cold loading checks. For each unmet need, say whether existing structure covers it before proposing a new representation.

**Proposed order:** settle section 5, work this case, and decide section 4.1 before building measurement-informed planning. Defer a Realm experiment until current measurements show a limit it would address. For fine-grained overhead, start from the tracked overhead pass and fresh measurements. An earlier discussion's user-reported measurements pointed mainly to thread handoffs; they may predate current code. GPU step costs need their own measurements.

## 7. Added external reading

| Source | Use here | How much was checked |
| --- | --- | --- |
| [Schneider, *Enforceable Security Policies*](https://personal.utdallas.edu/~kxh060100/Papers/EnfSecPols.pdf) | Limits of stop-on-violation monitors: enforceable policies must be safety properties | Search excerpts, including a later paper's summary |
| [Basin et al., *Enforceable Security Policies Revisited*](https://doi.org/10.1007/978-3-642-28641-4_17) | Separates controllable from observable-only actions, relevant to time and launched work | Abstract and excerpts |
| [Clarkson and Schneider, *Hyperproperties*](https://ecommons.cornell.edu/bitstream/1813/11660/4/hyperproperties-tr.pdf) | Properties over sets of runs | Tool summary of the first part |
| [Pnueli, Siegel and Singerman, *Translation Validation*](https://rd.springer.com/content/pdf/10.1007/BFb0054170.pdf) | Precedent for 4.1 option B: check each translation instead of proving the translator | Search excerpts |

These are about monitoring and compiler correctness. They do not show that a Kuroiro checker is feasible. The external claims in the two original notes were not systematically re-verified.
