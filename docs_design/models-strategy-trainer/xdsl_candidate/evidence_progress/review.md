I now have a complete picture. Let me do one final arithmetic sanity check on the archive matrix's independently reconstructed weight (0.7749034762382507) against the fixture semantics (w₀=0.8, feature_i = 2i+1+w_epoch·i/4, error = w·feature − i/2, lr 0.001, epochs all 0):

- w₁ = 0.8 − 0.001·(0.8·1)·1 = 0.7992
- w₂ = 0.7992 − 0.001·(0.7992·3.2 − 0.5)·3.2 = 0.79261619...
- w₃ = 0.79261619 − 0.001·(0.79261619·5.4 − 1)·5.4 = 0.77490350...

This matches the recorded 0.77490347… (float32 vs float64 explains the tail), and the probe weight 0.8 − 0.001·0.8 = 0.7992 matches the new test's expectation exactly.

---

# Code Review: sd-scripts-syv.31.17 (progress/contract repair)

## Scope confirmation

The uncommitted diff touches exactly the focus files plus consistent documentation (`README.md`, `CHANGELOG.md`, `ROADMAP.md`) and new `evidence_progress/` artifacts. No production code, owner engine (`execution_candidate/`), OpenSpec, or dependency changes — matches the stated boundary.

## Fix 1: Selection quantum (mixed_adapter.py) — correct

The loop change is sound on all the semantics claimed:

- **Exclusions counted**: `selected_in_quantum += 1` executes before the filter invoke, so admitted *and* excluded attempts both consume quantum budget. After 64 attempts without a capacity wait, the counter resets, `await asyncio.sleep(0)` yields to the loop, and `continue` re-enters the `while` condition so `bridge.stop` is rechecked before any further selection. This exactly matches "at most 64 selection attempts between explicit yield/capacity waits, exclusions included, continue rechecks shutdown."
- **No new lost-wakeup/spin risk**: the pre-existing `clear()`-then-recheck pattern around `bridge.wake.wait()` is untouched; `stop()` sets both `stop` and `wake`, so a parked producer is released. The counter reset after `wake.wait()` is correct (a real await occurred). I traced the edge where the inner horizon condition is skipped (len dropped or stop set between outer check and recheck): either the loop exits on stop, or selection resumes with the counter still bounded — the next quantum check still yields. No unbounded selection stretch remains reachable.
- **Horizon accounting unchanged**: excluded keys never enter `source.active`, so excluded attempts cannot fabricate admissions or trigger the capacity path. Correct per the defect description.
- **Honest boundary**: the qualification explicitly notes this is cooperative — a blocking/non-returning filter invoke is not preempted. `SELECTION_QUANTUM` is a runtime scheduling parameter, not serialized plan meaning, consistent with the package's stated authority split.

The before/after evidence directly demonstrates the defect and fix (heartbeat at call 20000 → call 64; feedback observed at 513 with one admission; scheduled cancellation observed at 64 with 192 total calls).

## Fix 2: Mandatory baseline outcomes — correct and placed at both stated boundaries

- `prepare.local_maps` is the shared analysis used by `accept()` (B), and by `accept_mixed`/`accept_model` (`mixed_contract.py:79`, `model_contract.py:70`) plus all adapters, so semantic acceptance for B/A/mixed/extension now rejects baseline omissions with `Rejected("{origin}: missing mandatory baseline local outcomes")`.
- `payloads.resolve_local` independently rejects the same omission *before* codec lookup, so a caller bypassing `accept` is still defended. The existing `binding.outcomes != description.outcomes` equality then guarantees a matching binding can't launder an incomplete op description — which was precisely the reproduced defect (matching op+binding omitting outcomes the runtime emits).
- Consistency with existing checks is preserved: fusion eligibility (`fusion.py:34`) already demanded the exact baseline tuple; B non-map bindings already required exactly `DEFAULT_OUTCOMES` (`prepare.py:490`); reserved runtime labels remain rejected only at cold resolve, which matches the stated claim scope.
- The new regression patches **both** `PackContract.describe` and the binding outcomes jointly, across three omission patterns (all missing / missing two / missing `returned`), and exercises both surfaces (`Rejected` at accept, `ValueError` at resolve). This is the right test shape — it proves the check doesn't depend on a binding mismatch.

One naming nuance worth recording (not a defect, and pre-existing): the guarded runtime never emits the literal label `"raised"` (raise-path frames end as `entered_failed`, a reserved runtime label). `DEFAULT_OUTCOMES` is a declared *possibility contract* for the ABI rather than an enumeration of emitted strings. The new docstring frames it correctly.

## Fix 3: Assertion-independent retirement — correct

- `_retire` now performs `pop` unconditionally with an explicit identity guard raising `RuntimeError`. Behavior for the corrupted-invariant case is equivalent-or-better than the old assert (missing key → `pop` raises `KeyError`/`None`-mismatch `RuntimeError`, both loud).
- I searched for other side-effecting asserts in the runtime paths: the only remaining one (`runtime.py:350`) is a pure post-condition with no mutation, so nothing else silently changes under `-O`.
- The regression runs **full real B and extension sessions** in a `python -O` subprocess, asserts `optimized` (i.e. `not __debug__`) so the mode is proven, checks `outstanding == active_calls == 0`, and validates the complete report with the independent parent-side oracles (`check_b`/`check_extension`) where assertions are enabled. The `after.json` optimized reports are complete, consistent (5 tickets; dispatch counts 5 base / 15 extension match fused-group arithmetic), and the before-probe (`outstanding: 1` with `retired: 1`) demonstrates the original defect.

## Test and evidence quality

- **Progress probes** use individually fast predicates with a finite sentinel, and each assertion genuinely discriminates fixed vs. broken behavior: `heartbeats[0] < limit` fails on the old code (callback only ran at 20000), `first_feedback_at < limit` proves real model/learning feedback during exclusion, and `not any("failed to interrupt" ...)` proves cancellation preempted the finite fallback rather than the sentinel firing.
- **First-failure handling is correct**: the generic TaskGroup message in the consumer-facing report vs. the exact sentinel retained on each owning input port (`feed.failure` via `causes`) is checked at the right surface, and `causes(None)` returning `()` means the `all(...)` would fail loudly if a feed lost its failure — this is a real assertion, not a vacuous one.
- **Archive matrix**: 8 cases (calls/inline × fused/unfused × normal stop / repeated cancellation + physical failure) with all observation lists discarded, retained owner input facts reconstructing the parameter independently (I verified the arithmetic above), `released`/`steps`/`pending`/claimed/discarded accounting consistent, and both `CancelledError` and the physical failure retained in the cancelled reports. The repeated-cancellation loop asserts readers *and* active producers persist between interrupts, so protection-through-drain is actually exercised.
- **Count lineage** checks out: 457 + 6 archive + 3 progress + 3 outcome + 2 optimized = 471 full; 256 + 3 + 2 = 261 pure. Documentation updates (README section, CHANGELOG, ROADMAP) match the code and don't overclaim.

## Minor observations (non-blocking)

1. **`test_actual_session_completes_with_python_assertions_disabled` omits `cwd=`** — the sibling fresh-process test (`test_extension.py:315-338`) explicitly sets `cwd=Path(__file__).resolve().parents[3]` (and env). The new test relies on pytest being invoked from the repository root, which is the documented workflow, but running pytest from another directory would make the subprocess import fail. Adding the explicit `cwd` (and arguably `PYTEST_DISABLE_PLUGIN_AUTOLOAD`, though the script never imports pytest) would make it robust for free.
2. The quantum-reset placement after `wake.wait()` (rather than before entering the horizon branch) is fine as analyzed, but it is subtle enough that a one-line comment mirroring the existing inline comments could help future readers; not required.
3. `_retire`'s guard change swaps `AssertionError` for `RuntimeError` in an already-corrupt-ownership scenario; if it ever fired inside `_apply_feedback`, the subsequent consumer failure path would hit a `KeyError` on the second pop. That is acceptable (loud failure under a broken invariant), just noting the failure mode is now two-stage rather than one.

## Verdict

All three reproduced defects are fixed at the correct boundaries, with no new fairness, shutdown, or wakeup hazards introduced; horizon and outcome-contract semantics are unchanged apart from the added rejections; the regression suite genuinely discriminates the fixed behavior (including optimized-mode execution with parent-side oracles and the expanded archive-erasure matrix); and the evidence (before/after probes, matrix, qualification lineage) is internally consistent — I independently reproduced the recorded numerical results. No blocking correctness, ownership, or security issues found. The only actionable suggestion is the minor `cwd=` robustness fix in the new subprocess test; everything else is at most documentation polish.


## Parent follow-up

The optimized subprocess regression now supplies an explicit repository cwd. Both B/extension cases pass when pytest itself runs from /tmp; Ruff, format and full ty pass after this test-only change. No second review was requested for this small robustness correction. Raw generated-code Markdown from older evidence was preserved unchanged; executable Python source formatting is clean.
