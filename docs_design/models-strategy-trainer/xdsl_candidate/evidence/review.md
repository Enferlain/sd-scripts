# Review: xDSL data-only B experiment (`sd-scripts-syv.31.9`)

## Verdict

This is a well-executed, self-consistent research handoff. I traced every acceptance criterion from the task description into the actual semantic/prepared IR and runtime/test code (not just the generated evidence), and found **no correctness gaps, false conformance, or lost cancellation/failure ownership**. The findings below are minor/defensive; none blocks acceptance.

## Verified against the stated acceptance criteria

**Author detachment.** `accept()` returns a canonical dumped snapshot (`prepare.py:110-179`); `prepare()` re-parses that text and requires re-acceptance to be byte-identical (`prepare.py:220-222`). `test_snapshot_round_trip_and_detachment` mutates the author after acceptance and proves the snapshot is unaffected; `test_loaded_document_executes_in_fresh_process_without_torch` replaces `fixture.build` with a throwing stub in a fresh process and executes from the loaded document. Solid.

**Semantic rejection vs. CPU readiness.** Cleanly separated: `Rejected` (contract violations, origin-annotated) vs `NotReady` (target `scalar-async-cpu-v1`, capacity≠1, horizon>2, missing/incompatible binding role/effect/version). Horizon 3/99 accepted semantically but `NotReady` at prepare — exactly as the README claims.

**Versioned bindings and prepared recipes.** `prepare()` emits `exec.plan` with accepted digest, cloned owner entities, `exec.select`/`exec.handoff`/`exec.feedback` control recipes, and `exec.chain`-grouped `exec.call`s with ancestors/binding versions/`await_physical_then_validate` — verified against `evidence/prepared.mlir`, whose call origins, ancestor chains (`data.double` → `[data-session, data.composed]`; `data.offset` adds `data.nested`), and endpoint name (`mapped` from `compose.output.owner`) all correspond to `accepted.mlir`. Dispatch is genuinely re-derived by re-parsing the *prepared* text (`prepare.py:319-327`), not from the semantic module. The `--manifest` JSON-equality gate plus the subprocess version-mismatch rejection test pins versions across processes.

**Runtime semantics.** Horizon is `len(self._active)` from selection to retire-at-feedback-ack (`runtime.py:202, 273`), so selected calls, buffered mailbox work, and pending feedback are all counted; `high_water` matches and the oracle replays transitions and asserts `peak == high_water` and `peak <= horizon`. Control is a dedicated task; the mailbox slot is retained until after `acknowledgement` (`_consumer` peeks, pops only post-ack), so feedback progresses with the producer blocked (`test_feedback_progress_while_publication_is_blocked` asserts `buffered == 1` *after* feedback application). Receipts precede policy application (`publication_ack` set before queueing; `_apply_feedback` requires it), and actual revision/bias are stamped on the ticket and decision at selection (`tickets[1].policy_revision == 0` test). The `_aborting` check sits between the capacity wait and the mailbox append (`runtime.py:217-223`), so a consumer abort mid-producer-invoke correctly abandons the in-hand ticket after its physical uses — I traced this interleaving explicitly and there is no orphan/leak path; `run()`'s final `assert not self._active ... and not self._mailbox` is a real invariant check that holds on every exit path.

**Stop/cancellation/failure honesty.** `stop()` records `selection_closed` once; the producer stops selecting, the consumer settles buffered work, and the report exists only after all three tasks drain. `drain()` shields the gather and re-arms on repeated cancellation; session tasks and `_invoke` are never cancelled, so physical uses (real threads via `asyncio.to_thread`) complete before the report is built. Status logic (`cancelled` / `failed` / `stopped`) is honest in both directions and the oracle enforces it; failed tickets keep `entered_failed`/`invalid_output` attribution, wrong receipts yield no feedback decision, and already-acknowledged work survives a later failure. Consumer failure abandons only work it can no longer publish; producer failure lets published work settle.

**No Torch / no placeholders.** Grep confirms no `torch` anywhere in package code (only the README discussion and the fresh-process `sys.modules` assertion); no optimizer/training/numerical-compiler placeholders; nothing in the package imports `execution_candidate`, and nothing else imports the package.

**Diagnostic eviction.** All associations live in `_active`/`_completed`/decisions, independent of the bounded `_diagnostics` deque; conformance passes with history 0/1/32.

**Oracle independence.** `conformance.py` is stdlib-only, recomputes the fixture arithmetic, provenance, replayed policy transitions, causal orderings, and aggregate bound; 12 corruption cases plus the shared-wrong-implementation control (equivalent reconstructions, failing oracle) demonstrate sensitivity beyond self-agreement.

**Gates/claims.** Test count checks out by parametrization math: test_ir 25 (1+4+12+1+4+2+1) + test_runtime 53 (6+1+16+1+1+6+1+12+1+1+4+3) = **78**. GIL is only observed (`sys._is_gil_enabled()`); no `PYTHON_GIL`/switch forcing anywhere. README, `qualification.json`, CHANGELOG, and ROADMAP all explicitly bound claims (no A/C, no fusion, no framework/production selection, no GPU/perf/concurrency-safety claims), and the evidence chain is consistent (report digest == accepted.mlir digest == embedded prepared digest; sample 6 tickets / high-water 2 is consistent with stop-after-4 and settle semantics).

**CHANGELOG/ROADMAP.** Accurate, correctly dated/placed (2026-10-29 above 2026-10-28), no overclaim; the ROADMAP paragraph correctly defers A/C and fusion.

## Findings

1. **Low — `__main__.py:37`: oracle uses the default horizon.** `check_b(report)` hardcodes `horizon=2` instead of `prepared.horizon`. A horizon=1 document loaded via `--load` would be verified against a looser aggregate bound than its contract. One-line fix: `check_b(report, horizon=prepared.horizon)`.

2. **Low — `prepare.py:333-335`: dispatch split is positional.** `resolved[:-2]` / `resolved[-2:]` / `resolved[-1]` assume walk order and that the last two calls are score/publish, without checking the parent `exec.chain`'s role. Correct for this fixed contract, but a `ChainOp`-role assertion would make the "dispatch derived from prepared IR" step self-checking rather than order-dependent.

3. **Low — `__main__.py:41`: hardcoded `"xdsl": "0.74.0"`.** Could drift silently if the pin changes; `xdsl.__version__` (or asserting equality with the pin) would be safer.

4. **Nit — `runtime.py:31-33`: `drain`'s non-Cancelled `break` + gather first-exception semantics.** If a session task ever escaped its handler (currently unreachable: producer/consumer retire their in-flight ticket, control catches `BaseException`), `run()` would propagate without a `Report` while sibling tasks might still be running. Defensive only; no test reaches it.

5. **Nit — `__main__.py:25`: `--stop-after` ≤ 0** surfaces as a raw `ValueError` from `execute` rather than an argparse usage error.

6. **Nit — test coupling to privates.** `test_feedback_progress_...` and `test_duplicate_or_premature_...` reach into `_apply_feedback`/`_active`. Fine for an in-package research suite, but those are the spots where a refactor would break the control-progress/duplicate-admission guarantees silently.

## Notes

- `requirements.txt` is compiled against the repo's uv-managed interpreter and the package-local `.gitignore` re-includes it despite the repo `*.txt` rule — consistent with the README's "no lockfile refresh" claim.
- The single-loop confinement of `ScalarSession` (including `stop()`) is documented and respected; the shared `_changed` event with check-before-clear predicates has no lost-wakeup window on one loop.
- Per the task instruction I did not review the unrelated pre-existing worktree state; the only tracked diffs (CHANGELOG/ROADMAP) belong to this change.
