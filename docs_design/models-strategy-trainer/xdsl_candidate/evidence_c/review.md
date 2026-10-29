# Code Review: xDSL Experiment C (granted two-pass authority)

## Verdict

**No blocking correctness defects found.** C matches its stated intent on every checkpoint in the task description, reuses the A-baseline owner services unchanged, and its evidence artifacts are internally consistent. One explicitly requested assessment (the shielded-future log) is a pre-existing service diagnostic, not a new bug. Findings below are minor/informational.

---

## 1. Intent conformance (verified against code, not just claims)

**Source/authority correspondence — sound.** Global unique/non-empty `source_id` is enforced by the shared `accept()` walk (`prepare.py:137-143`), which runs *before* `extension.verify`, so the `identity` corruption case is rejected there. `verify_grant` then pins: `grant.model == model.entity` (participant kind), `grant.unit` (unit kind), `handback.type.unit == grant.unit`, `advance.unit == grant.unit`, `advance.handback is grant.handback`, exactly one use of the handback, exact `DELEGATED`/`RETAINED` splits, exact five-phase protocol, finite initial/lr/clip. The adapter re-verifies the decoded artifact: digest of the re-embraced semantic subtree, `ancestors == [session.source_id]`, origin↔implementation symbol correspondence, no duplicate origins, exactly two resolved bindings. Tests corrupt `completion`/symbol/ancestry/target/protocol/capacity via a patched `dump` and all reject before the owner program is generated.

**Acceptance vs concrete readiness — correctly separated.** Semantic acceptance imports no Torch (subprocess test). `prepare_grant` rejects missing/role-mismatched/empty-version bindings, wrong target, `offered=False`, and unsupported parameter *before* `backend.prepare` calls `declaration.load()` — verified by `assert not loads` in the test.

**Typed unique handback.** The `!grant.contribution<@weights>` edge survives print/parse (SSA identity re-checked on the parsed clone), and runtime readiness is decided only by `UnitRuntime.accepts` object identity against the live `_PendingWork.ready` — the forged-`Contribution` test proves a correctly addressed fabricated token cannot advance, and the static SSA edge is never consulted at runtime.

**Unchanged numerical callback, generated topology.** `grant_registry` extracts only `examples.two_pass().root.work[0].implementation`; the example's authored `Run`/`Block`/`Unit` are discarded and the filing is built from decoded xDSL (work names from `source_id`s, lr/clip/parameter from the digest-verified decoded `advance`). The `__code__`-identity test confirms the maintained callable is reused. The `observed` wrapper correctly decouples the binding's declared role key `"model"` from the semantic participant symbol via `MappingProxyType({"model": models[model_name]})`.

**One existing owner Engine; no fake flow fields.** Single `backend.Engine(ready)` per `PreparedGrant.publish()` with a one-lifetime guard. C *requires* `session.horizon is None` and *rejects* any `horizon`/`capacity`/`feedback` on the generated plan (line 213), with tests asserting the props are absent and `"wf.flow"`/`"feedback"` absent from the accepted text. The optional-props change in `dialects.py` is compensated: B's `accept` and A's `verify_model` both newly `require(session.horizon is not None)`, with regression tests for both fixtures; the positivity `verify_` check is retained when a horizon *is* present. A emits `session.horizon` into its plan (guaranteed non-None), and B asserts post-parse presence before use. No other `horizon` consumer is exposed (grep-verified).

**Observation vs live verification; failure/uncertainty/cancellation/drain; retention.** `GrantFacts` are written only in `observed` and read only by `report()`; advancement is gated solely by `GrantedAccess._check_handback()` + `complete_contribution` + `state.advance`. The failure matrix matches the README table and the six evidence reports: raise-after-restore, incomplete, missing-gradient, edit-after-restore, wrong phase order, forbidden `access.advance()` all yield `unverified_grant`/no optimizer attempt/`unusable={"model"}`/pending retained; suspended and physically-running grants hold the write lease through repeated cancellation; the retained-optimizer/reset after-effects faults keep `granted_ready` distinguishable from `uncertain` advancement/reset with pending evidence surviving, and retry raises. Active outcome retention is real: `_PendingWork.records` is independent of the bounded `outcomes` deque, and the test replaces the deque with `maxlen=0` after publish.

**Independent arithmetic and author-disabled reconstruction.** `grant_conformance.py` is stdlib-only and its math checks out by hand: first grad `4*(2w−1)=2.4`, temp `1.04`, final grad `4.32`, SGD→`0.584`; the clip formula `min(1, clip/(|g|+1e-6))` matches `torch.nn.utils.clip_grad_norm_`'s eps convention, giving `≈0.75` (evidence: exactly `0.75` in float32). The no-op-optimizer test demonstrates both lowerings agree while the oracle catches the corruption — the right negative control. The subprocess reconstruction test disables `build_grant` and reproduces from saved IR + manifest, byte-identical prepared text.

**Evidence/test-count consistency.** 23 IR tests (1+1+5+14+2) + 60 runtime tests (8+3+1+6+6+12+1+2+4+4+10+1+1+1) = 83; 158+83 = 241; 241−58 (A optional) −60 (C optional) = 123 passing with 2 module skips — all consistent with the README and `qualification.json` (`review.status: "pending"`). Six failure reports + clipped + success present; cause chains in the physical-failure report show `WorkFailed → PhysicalFailed → RuntimeError` plus the captured `CancelledError`. Root changes are documentation-only (CHANGELOG/ROADMAP); no `execution_candidate`, production, or OpenSpec changes.

---

## 2. Requested assessment: `RuntimeError exception in shielded future` log

**Old-service diagnostic, not a C bug.** Mechanism: on modern CPython (3.13+; evidence records 3.14.5), `asyncio.shield` reports an inner future's exception to the loop's exception handler when the *outer* shield was already cancelled (gh-97637 fix — previously the exception was silently dropped). In the existing `Workers.execute` drain loop, each retry creates a *new* shield; after the coordinator's cancellation is delivered, the abandoned shield observes the worker's `RuntimeError` completing on `pending` and logs it. Crucially, the coordinator then retrieves the same exception itself — either at the loop head (`cancelled is not None and pending.done()` → `pending.result()`) or via the next `await asyncio.shield(pending)` — and wraps it in `PhysicalFailed(error, cancelled)`. I traced every interleaving (second-cancel-while-shielding, completion-while-resuming, NotEntered entry-gate win); in all of them the exception is retrieved, so nothing is leaked:

- The run's error chain contains the `RuntimeError` cause and the `CancelledError` (evidence `physical_failure_after_repeated_cancel`).
- Drain and lifetime are real: the test asserts `finished.is_set() and pool.active == 0 and pool.executor is None`, and `released`/`pending`/`unverified_grant` ownership facts hold.

C introduces no changes to `Workers` or `asyncio.shield` usage (it only *uses* the service inside the selected binding, as the ABI requires). If the duplicate stderr noise ever matters, the fix belongs in `Workers.execute` (e.g., retrieve `pending.exception()` when done before re-shielding) — an A-baseline change outside C's scope, and it would change no observable ownership semantics.

---

## 3. Minor, non-blocking observations

1. **`grant_adapter.py:106`** — `report()` hardcodes `get_parameter("weight")` while `observed` uses `advance.parameter.data`. Currently safe because `prepare_grant` rejects any other parameter for this scalar target (line 163), but it's a latent inconsistency if that guard moves. Prefer storing the parameter name on `PreparedGrant`.
2. **`model_adapter.py` decoded-plan checks** — unlike B's `prepare` (which asserts `root.horizon/capacity` non-None post-parse), A never re-checks these on its decoded plan now that they're optional. Behavior-neutral (A always emits them from a contract-guaranteed session horizon), but a symmetric assert would harden the boundary the optionality change opened.
3. **`prepare_grant(…, offered=…)` uses `is not True`** — only literal `True` counts as offered; an intentional test seam, but `offered=1` silently rejects. Fine for the experiment; worth a type annotation stricter than `bool` if this ever generalizes.
4. **`GrantRun.run` swallows non-interrupted failures by design** (returns the report, does not raise) — mirrors A's `ModelRun` and is the observation harness contract; callers must inspect `errors`. No action needed, just be aware `grant_demo` will exit nonzero via the `check_c` assertion on a failing loaded document rather than surfacing the failure report.
5. **Unit↔parameter member binding lives only in the adapter** (`graph.Unit(unit_name, ((model_name, advance.parameter.data),))`); the semantic contract cannot express unit members. This is the same pattern A uses and is disclosed in the README ("xDSL does not … own parameters"), so it's an accepted limitation, not a defect.

---

## 4. Security / scope

No `eval`/`exec` on IR text; the only codegen `exec` is the pre-existing backend's dispatch shell built from accepted work items. The grant ABI is explicitly documented as authority restriction, not a sandbox. No production, root-dependency, OpenSpec, or `execution_candidate` (A baseline) modifications; the unrelated untracked `openspec/changes/rework-model-strategy-trainer/research/ir_architecture_report.md` was not inspected or altered, as instructed.

**Recommendation:** Approve for the C milestone. Items 1–2 above are optional polish; nothing blocks handoff or the pending qualification review.


## Parent follow-up

Applied the two small consistency suggestions: C reports through the parameter name stored from decoded IR, and A checks decoded horizon/capacity against the accepted recipe. Four corrupted A-plan cases were added. The final combined suite passes245 tests in14.50s with actual GIL state false before/after; Ruff/format/ty and diff checks pass. These small changes do not require another full review.

The shield diagnostic was independently reproduced using only installed Python3.14.5 asyncio, retained as shield_diagnostic.json. This confirms the extra message source for this installation; no broad claim about every Python version or arbitrary worker behavior is inferred. No legacy candidate source changed.
