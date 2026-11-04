# Review: milestone 9 qualification (tasks 9.1–9.2)

## Scope verified

I reviewed the nine focus files in full, plus the supporting code the new modules rely on (`prepare.load`, `observations.observe/compare`, `reference._invoke`, the owner engine's `Workers`/`Lowering`/`Feed`, and `fixtures`), and cross-checked the retained evidence record against the code paths that produce it.

## Core claims — verified sound

**Live/projection separation is complete.** `self.outcomes` is now write‑only in `target.py` (declared :197, written :437); every live-state read goes through `self._live` (:347), entries deleted in the invocation `finally` (:525), and the `drained` check in `summary()` includes `runtime._live`. Milestone‑8 tests that read `runtime.outcomes` still see terminal outcomes, so the projection contract is preserved.

**Erasure is behavior-safe.** Everything `erase()` clears is diagnostic: `Feed.handed_off` is a bounded diagnostic deque (`state.py:456`), `Lowering.source` is generated-source diagnostics (`execution_candidate/prepare.py:442,546`), `state.trace` entries are append-only observations. `summary()` reads only surviving owner ledgers (resources/units/handlers/flows/closed-source receipts), so recovery of results/epochs/step-reset/request/ticket/closure/drain evidence is genuinely independent of erased histories.

**Request registration & authorization.** `slot.issued` is retained at registration (:775–776) and unfulfilled disposal (:784); the packet-vs-owned-registration guard (:717–720) and the "no owned registration or disposition" rejection (:722–725) close the forged-packet path. Registration precedes queue publication.

**Stop/cancellation guards.** Repeat admission rechecks scoped stop after the admit checkpoint (:455–461); producer rechecks stop after capacity acquire (:654), after the reserve checkpoint (:662), and after the publish checkpoint (:680) with an interruption-aware disposition (:685–695) and capacity release on checkpoint exception (:657–661); `stop_after` is superseded by run closing (:566–568); call entry/complete rechecks merge cancellation while still applying the physically completed call's effects (:949–1009).

**Oracle independence.** `qualification_oracle.check` recomputes the window recurrence in exact `Fraction` arithmetic (I re-derived 0.8→0.77→0.635, loss 0.91125, and the data-only `(2x+1)² % 5` recurrence by hand — they match) and never reads runtime facts/traces. The reference derives its own admissions via `rng.choice(enabled())`; `pressure_reference` pins external phases by inspecting only reference state. The wrong-implementation control proves the oracle is not derived from runtime values.

**Optimized-mode safety.** The child path contains no control-flow `assert`s; the parent refuses to spawn without `__debug__` (`child()`), the oracle self-guards (`check` raises `RuntimeError` under `-O`), and the child's reported `optimized` flag is verified against the requested mode. Under `-O` the parent fails loudly at the first `check()`, including inside the wrong-implementation `except AssertionError` block (RuntimeError is not caught there).

**Forged prepared claim.** `load()` re-derives the entire recipe and compares (`prepare.py:304–314`); the `workers=99` forgery fails the exact comparison and raises `CorrespondenceError(ValueError)`, which the child's `except ValueError` catches — before any resource factory (`factory_calls == []` verified).

**Evidence/reproducibility.** `document()` uses `sort_keys=True`/`allow_nan=False`, schedules use `gzip.compress(..., mtime=0)`, seeds, base commit, per-child environment/GIL/module inventory, artifact+schedule SHA-256s, and the collector refuses non-empty outputs. Verdict/schedule counts (44/24) match the code paths that generate them.

**Differential strength.** `compare()` requires exact consumer-fact equality plus all reference-required causal edges against runtime positions; the 128-exclusion test compares the first 128 admission decisions between runtime and reference value-for-value.

## Findings

**F1 (moderate) — the new `stopped_unadmitted` runtime path has no direct verification of its fact shape.** The guard at `target.py:458–461` is new in this milestone, but every run that exercises it (`orderly-before` children, `test_integrated_pressure_choices...`) runs with `erased=True`, and `check()` only infers the guard fired indirectly (unit steps, model epoch, empty `issued`). The outcome *status* and its state/fact sequence (created→pending→closing→settling→terminal, outcome fact arity) are asserted only on the reference side (`test_reference.py:188`). A regression that, e.g., mislabeled the status or skipped a state fact would pass milestone 9. Recommend one non-erased orderly-before/after run with either an assertion on the outcome status or a `compare()` against `pressure_reference` facts.

**F2 (minor) — wrong-implementation control cannot attribute the oracle failure.** `qualification.py` (`try: check(...) except AssertionError: verdicts.append("rejected wrong arithmetic")`) and `test_wrong_selected_implementation...` treat *any* `AssertionError` from `check` as the numeric rejection. If `check` tripped on an earlier structural assert (e.g. `owner["drained"]`), the control would pass vacuously with a misleading label. Low risk since the child pre-asserts `completed`/no failures, but asserting the specific product-value mismatch (or inspecting the failure message) would make the control self-describing.

**F3 (minor) — pressure-mode differential is state-level only.** The pressure test compares resource values/epochs and request dispositions with the independently driven reference, not consumer fact streams + causal edges; fact-level comparison exists only in the 8-seed normal-mode runs (and milestone-8 tests). This is a defensible split of "seeded interleavings" vs "fixed external phases," but `qualification.md` doesn't state that the pressure comparison form is weaker; one sentence would prevent over-reading the record.

**F4 (minor) — freeze-guard scope.** `source_hashes()` covers only package `*.py` plus the two research reports. README/qualification docs and the evidence tree aren't hashed, and artifact files are hashed once at the end rather than compared against a pre-child snapshot. Tampering is still non-exploitable in practice (every child re-derives via `load()`), but the guard's claim should be read as exactly "package sources and the two reports."

**F5 (info) — private-attribute coupling.** `summary()` reads `runtime._version()`, `slot.capacity._value`, `runtime._live`; `lag_case` mutates `selection.regions[0].block.args[0]._type` directly (`qualification_cases.py:117`). Acceptable in a frozen research slice; fragile against xDSL/owner-engine refactors.

**F6 (info) — retained `Handler.issued` grows with total requests** (`target.py:150,776,784`). Correctly documented as bounded-case receipts, not a production retention policy; no action needed for this slice.

**F7 (info) — timing heuristics and budgets.** The held-mode witnesses use `asyncio.sleep(0.01)` polling, children get a 45 s timeout, and `until`/`wait_for` budgets are 10/15 s. The witnesses are logically sound by construction (gates block worker threads; the task cannot complete while held), but very slow CI could hit the child timeout in the 128-exclusion run.

**F8 (info) — `child()` hard-asserts GIL-disabled children** (`assert not observed["gil_before"] and not observed["gil_after"]`). Intended for the documented 3.14.5t environment; a friendlier explicit error would help anyone re-running the collector elsewhere.

## Conclusion

No blocking defects. The implementation matches the milestone 9.1–9.2 intent without semantic shortcuts or manufactured evidence: the oracle is independently recomputed, controls have teeth (the forged-claim and wrong-implementation paths fail in the right place and cannot silently pass), cancellation/admission/stop boundaries are rechecked at the new suspension points, ownership of live invocation state is genuinely separated from erasable projections, and the retained record is reproducible and honestly scoped. I'd ask for F1 (a direct check of the `stopped_unadmitted` runtime outcome/facts) before marking 9.1 complete, with F2–F4 as cheap hardening/doc clarifications; F5–F8 are informational.

Review artifacts: /tmp/review-mcp-run-r8deqgcp
