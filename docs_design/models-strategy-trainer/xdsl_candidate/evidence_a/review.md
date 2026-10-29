# xDSL A Handoff — Read-Only Implementation Correctness Review

Reviewer: hcom OMP fallback (`labe`), default model/effort, after review-mcp provider429 failure.
Scope honored: baseline `2131f31d`; diff = `xdsl_candidate/{README.md,dialects.py,prepare.py}` (+7/+22/README), 7 new `model_*.py`/`test_model_*.py`, `evidence_a/`, plus CHANGELOG/ROADMAP doc entries. `execution_candidate/`, production, OpenSpec, root deps untouched (verified via `git status`/`git diff --stat 2131f31d`). No repo files edited; no Beads/git mutation. The untracked `ir_architecture_report.md` excluded as instructed.

## Verdict

**PASS — no current correctness gaps found.** Every concrete behavior claim in the handoff traces to code I read directly, and the full combined suite re-ran green under my session: **158 passed in 10.10s** (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync --with-requirements …/requirements.txt python -m pytest …/xdsl_candidate -q -p pytest_asyncio.plugin`). Two informational robustness notes below; neither is reachable divergence.

## Claim-by-claim verification

**1. Typed hierarchical snapshot reconstructs without author or Torch import — VERIFIED.**
`model_dialect.py` imports only xdsl; `model_contract.py` imports `math`/xdsl/local modules only. Fresh-process tests: `test_model_ir.py:27-37` asserts `'torch' not in sys.modules` after build+accept+round-trip; `test_model_runtime.py:289-312` reconstructs via `model_demo --load/--manifest` with `build_model` monkeypatched to throw, digest-equal and byte-equal prepared plan. `Accepted` carries text only; no author object or callback crosses the boundary (`prepare.py:95-107,145-147`).

**2. Optional installed model contract — VERIFIED.**
`dialects.py`, `prepare.py`, `fixture.py`, `runtime.py`, `__init__.py` never reference `model_*` (grep-verified); the dependency edge is one-directional. Shared-code changes are minimal and non-semantic for B: `PlanOp.feedback` → `opt_prop_def` (B always emits it at `prepare.py:325`), `parse(text, extensions=())` default-empty, `ContractExtension` hook selected explicitly at `accept(module, extension=None)` (`xdsl_candidate/prepare.py:120-148`); the B acceptance path is otherwise byte-identical logic. `test_model_ir.py:23-24` confirms B's acceptor rejects A documents.

**3. Prepared bindings/version/effect/completion + correspondence — VERIFIED.**
`model_adapter.py:349-374` emits `exec.call` records with origin/ancestors/implementation/binding_version/effect/role/`completion="protected_physical_return_then_validate"`; `:415-431` re-parses the dumped plan and cross-checks every (role, effect, version, completion) tuple against the live registry; `:432-445` re-derives the expected (window, slot, sample) list from `selected_keys` of the re-loaded plan, checks port-name/source correspondence and rejects aliasing; `:418-419` re-accepts the retained `RecipeOp` semantic clone and requires digest equality with the accepted snapshot. Tamper tests (`test_model_runtime.py:109-125`) prove NotReady on membership/binding corruption. `bindings.json` manifest records the resolved set.

**4. Old Run/Work records are backend artifacts; ONE existing Engine — VERIFIED.**
`PreparedModel.publish()` (`model_adapter.py:211-213`) constructs a single `backend.Engine(ready)` from the existing `execution_candidate/prepare.py:300-322` entrypoint; `ModelRun.run` (`:225-228`) enforces one owner lifetime. All `graph.Run/Producer/Repeat/Requested/Join/Retain/…` records are generated in `lower_model` (`:467-704`) and validated by the unchanged `graph.Contract().accept`. Zero diff under `execution_candidate/` — "no algorithm-specific cases in common B runtime" holds.

**5. Fixture semantics — VERIFIED, independently re-derived.**
`build_model` (`model_contract.py:142-247`): keys 1..6, excluded {3,6} before fanout (wiring `:102-110`; left = double→nested offset; right = frozen `scale_v1`@1.5 with `encode = model(k/4)`; learner initial 0.8; loss `(w·(l+r) − k/2)²/2` (`:83-84`); SGD 0.05 after each 2-sample window (`:626-664`, Advance in finish block after both slots, accumulate across slots). Hand-recomputed SGD: window 1 → **−0.60625**, window 2 → **10.419848** — matches `evidence_a/report.json` and the independent stdlib oracle (`model_conformance.py:21-34`).

**6. Trigger vs actual epoch independence — VERIFIED.**
Request fires from `Completion(position=1)` published only after the window block (including its advance) fully returns (`execution_candidate/prepare.py:1000-1006`); the measurement fact records `context.stamps[learner].numerical_epoch` at actual start (`model_adapter.py:599-616`). Oracle requires `(trigger_window, trigger_position) == (0, 1)`, `epoch ∈ {1,2}`, `weight == reference[epoch]`, and window-0 input membership regardless of epoch (`model_conformance.py:108-114`); evidence sample shows epoch 2 with window-0 membership. `test_evaluation_trigger_is_distinct_from_physical_completion` (`test_model_runtime.py:192-222`) blocks the physical call mid-flight and asserts the trigger fact exists before completion.

**7. Numeric return ≠ request completion before output/physical/mode cleanup — VERIFIED at lowering level.**
`measurement-completed` is ordered strictly after `measure` via `after=` (`graph.py:290-320` `ordered()`); the measure Call (evaluation=True, dispatched on a worker) returns only after: worker future quiescent, numerical return, `check_output`, and module-mode restore inside the worker's `synchronous()` before `CallReturn` (`execution_candidate/prepare.py:601-666`); any primary/cleanup/cancellation becomes `CallFailed` instead of a return. Cancellation tests assert `completed` stays False (`test_model_runtime.py:184-189`).

**8. Atomic two-member admission — VERIFIED.**
`state.join` (`execution_candidate/state.py:543-580`) checks EVERY member under the shared condition, then claims all without an intervening await; `Admission.__call__` (`model_adapter.py:165-181`) validates physical port identity, sample/window/slot/branch, scalar float32 finiteness, and current-source stamps (left: empty; right: encoder stamp vs join-time projection). 16 parametrized missing/bad/swapped/stale cases assert neither member consumed, prior work preserved, and pending/claimed/remaining evidence exact (`test_model_runtime.py:67-83`, `model_conformance.py:124-136`).

**9. Stop/failure drain; released vs pending separation — VERIFIED.**
Producer failure closes feeds with the failure and marks participants/states unusable; planned stop uses `end_demand`; cancellation shields task until cleanup joins (`execution_candidate/prepare.py:1052-1155`); `Workers.lifetime` raises on undrained submissions (`workers.py:85-108`); A's `produce` wraps its private pool in `pool.lifetime()` (`model_adapter.py:488,549-553`). Report's `released` is computed only from physical lease/registration/producer-lifetime state, `pending` from unfinished unit work (`model_adapter.py:275,280-286`) — separate flags as claimed. Diagnostic history is bounded post-lowering (`:455`, attribute rebind — safe, closures resolve `state.trace` at call time) while live input links and pending evidence survive eviction (`test_model_runtime.py:86-93`).

**10. External vs TaskGroup-peer cancellation; NotEntered vs entered failure — VERIFIED.**
`ModelRun.run` (`model_adapter.py:225-243`) detects external cancellation via `task.cancelling()` delta and re-raises only then; peer-failure groups surface as `status="failed"` with flattened causes (`causes()`, `:304-315`). `right()` re-raises `physical.NotEntered.cause` with the stage fact left `not_entered`, marks `PhysicalFailed` as `entered_failed`, and distinguishes `returned_cancelled` after a completed-but-cancelled physical call (`:504-527`).

**11. Deliberate target limits — VERIFIED.**
`prepare_model` (`model_adapter.py:334-339`) rejects target ≠ `candidate-owner-cpu-v1`, window count ≠ 2, >4 selected keys, horizon < len(keys), and non-"weight" parameters — while those same modules remain semantically acceptable (`test_model_ir.py:40-43`, `test_model_runtime.py:96-106`). Matches "other valid semantic counts or unavailable concrete parameters fail preparation". C/fusion/final selection correctly not claimed (README `:148-152,202-213`; qualification `limits`).

**12. Evidence integrity — VERIFIED.**
All 14 sha256 entries in `evidence_a/qualification.json` match the current worktree byte-for-byte; `accepted_digest bfa26dba…` == sha256(accepted.mlir); report GIL false; measured epochs/weights consistent. CHANGELOG/ROADMAP edits are documentation-only and accurate.

## Informational notes (no action required)

1. **`model_adapter.py:562`** — the objective's normalization argument is the literal `2` rather than a value re-read from the prepared IR. Unreachable divergence: `:334` rejects any window count ≠ 2 and `model_contract.py:86` requires `normalization == count`. Robustness nit only; deriving it from the decoded `WindowOp`/`ContributeOp` would remove the coupling-by-convention.
2. **`model_adapter.py:236-238`** — `interrupted` detection via `task.cancelling()` delta assumes `run()` is awaited as a task member; the non-task `await run.run()` path relies solely on the `isinstance(CancelledError)` check. No current caller hits the gap; noting for future callers only.

## Distinguished future work (not correctness gaps)

C granted-authority integration, semantics-preserving fusion, GPU/distributed/durability, dynamic filtering/prefetch beyond the static ≤4-key bound, and final framework selection — all explicitly out of scope and not claimed by this handoff.
