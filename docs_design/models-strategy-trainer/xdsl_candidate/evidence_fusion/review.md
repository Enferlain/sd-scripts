# Review: bounded prepared local dispatch fusion (sd-scripts-syv.31.12)

## Verdict

**Approve.** The fusion is a genuine xDSL `RewritePattern`/`PatternRewriteWalker` rewrite with real cold verification and a real runtime distinction, and I found no correctness, authority, or security hole. The checks compose correctly with the prerequisite `.31.14` repair, C is verifiably a no-op, and the evidence (including the timing numbers and the disclosed post-measurement hash delta) is internally consistent. Findings below are minor observations, none blocking.

## What I verified

**Rewrite correctness (`fusion.py`)**
- `rewrite()` clones first; the caller's plan and accepted document are untouched (test pins `dump(module) == before` and `unfused.accepted == fused.accepted`).
- `FuseLocal` groups only *consecutive* eligible calls (run broken by any ineligible child), requires runs ≥ 2, replaces the first call and erases the rest. With no SSA results on `exec.call`, no result remapping is needed — matching the README's precise limitation. Re-application reaches a fixed point (`rewrite(transformed, sources)` is stable; verified in the test and by inspection for A's plan shape).
- `eligible()` requires: source op known, `role=map`, `effect=pure`, an exact `LocalDescription` (record→record, one-to-one, `resources == ()`, exact outcome triple), and a `TransparentLocal` ancestor chain terminating at a `SessionOp`. Model (`role=model`), grant (`scoped_write`), score/publish (`score`/`publish`), encode, objective and measure calls are all excluded by role/effect/trait. A bare label with empty sources authorizes nothing (`rewrite(module, {})` pinned by test, and this is what makes C a no-op).

**Cold verification interactions**
- `check_calls` walks all `CallOp`s via `loaded.walk()` — *including leaves inside `exec.fused` bodies* — so the `.31.14` one-to-one source/symbol/version/role/effect/completion/ancestry correspondence is retained per leaf, and duplicates across group boundaries are caught by the `seen` set. This is the key interaction and it is sound.
- `groups()` enforces exact property set, `PROFILE`, closed body, ≥ 2 calls, `origins` ↔ body correspondence, and per-leaf eligibility. Nested `FusedOp`s fail (`any(not isinstance(call, CallOp))`).
- Both consumers enforce flattened **stage order** against the semantic source: B against `maps` from the accepted module plus per-chain role/transport checks; A against `local_maps(one(decoded, ComposeOp))` from the *re-accepted* recipe — closing the "A formerly used origin lookup" gap the task called out. The reorder-with-regenerated-origins corruption is pinned by test.
- `fuse=False` + any `FusedOp` anywhere rejects ("unrequested" case); `fuse=True` with ineligible stages legitimately degrades to unfused dispatch.

**Runtime equivalence and failure semantics**
- `invoke_stage` and the generated fused coroutine have identical guard structure: enter → await → `finally: physical_end()` → `validate` → exit, with validation raising *before* the next stage's `with` is entered. Failures, invalid outputs, and later-entry prevention are pinned for both realizations (`test_each_stage_validation_and_later_entry_survive_fusion` asserts identical failed-ticket facts across fuse modes).
- Cancellation: `ScalarSession.run` shields the task gather, so the fused coroutine is never cancelled mid-binding; `physical_end` is recorded only after actual physical return. The drain/repeated-cancel test covers both fuse modes and both stop/cancel × success/failure.
- `ScalarFrame` preserves the exact per-stage clock ordering (`entered < physical_end < released`) that `conformance.check_b` asserts — the independent oracle cannot distinguish fused from unfused except via `local_dispatches`, which is observation-only. A's `ModelLocalFrame` preserves the exact stage-fact shape `check_a` consumes.
- Codegen safety: only `f"stages[{index}]"` integer interpolation; origins passed as data; `"__builtins__": {}`. The `single()` closure binds its stage via a default argument (no late-binding bug).

**C no-op** — `prepare_grant` never calls `fuse_plan`; the new test applies `rewrite` externally with empty sources (byte-identical dump) and reruns `check_c`. Confirmed.

**Evidence and claims**
- Recomputed from `costs.json` raw samples: unfused median 6.575 ms, fused 6.392 ms, ratio 1.0287; prep medians 10.55/11.99 ms; hot dispatch counts 2000 vs 1000 with 2000 stage observations on both sides. The honest 8.97 ms fused outlier is retained, and the README/qualification explicitly disclaim statistical/production significance. A is not timed.
- Hash story checks out: `local_runtime.py` differs between `costs.json` (measurement-time `1202b5c9…`) and `qualification.json` (final `c938b853…`), exactly matching the disclosed "Protocol positional-argument annotation only" change; every other hash is identical between the two files.
- Test arithmetic reconciles: 309 (post-`.31.14`) + ~47 fusion-only cases = 356; no-Torch 167 + ~29 torch-free new cases = 196 (fusion modules and `test_fusion.py` import no Torch; the fresh-process tests assert it).
- Scope is respected: only the candidate directory plus CHANGELOG/ROADMAP; `ir_architecture_report.md` untouched; model/worker/optimization owner services unchanged (only `left()` in `lower_model` switched to `local.run`).

## Minor findings (non-blocking)

1. **A's stage-order check filters before comparing** (`model_adapter.py:489-497`): `dispatch_ops` drops interleaved non-map plan ops, so the check pins the order of map dispatches relative to each other but not their position among other plan-body operations. Execution is still fully pinned (non-map calls are consumed by origin, map dispatch positionally by `dispatch_groups`), so this is a document-order observation, not a semantic hole. B's chain-level check does not have this asymmetry.
2. **Manifest can misreport the lowering**: `"local_lowering": "fused-v1"` is emitted whenever `fuse=True`, even when nothing was eligible (the `may_effect` test) or a decomposed-group corruption leaves bare calls — which then execute unfused with a fused label. Diagnostics-only; consider recording whether any group actually formed.
3. **`groups()` does not enforce group maximality**: a corrupted partially-fused plan (bare call + `FusedOp`, or two adjacent `FusedOp`s) with identical flattened order and eligibility passes verification. Execution is equivalent, so only order/bindings are pinned, not the partition itself. Consistent with the stated contract, but worth remembering if the group partition ever gains meaning.
4. **`Prepared.maps` is now dead**: the producer dispatch goes through `Prepared.local`; no consumer of `maps` remains (searched).
5. **`missing_chain` attribution**: in `test_prepared_dispatch_checks_roles_and_binding_contracts`, the `missing_chain` case is actually caught earlier by `check_calls` ("missing semantic implementation selection" — the regex matches via `selection`), not by the chain-presence check. Behavior is correct; the test name slightly oversells which check fires (same pattern the `.31.14` review noted for `wrong_call_role`).
6. **A's stage facts carry no physical-end/release clocks** (`ModelLocalFrame.physical_end`/`__exit__` are no-ops). The README states this accurately ("retains sample/source/outcome facts"), but readers should not infer that A's per-stage facts could detect premature validation ordering the way B's clocks do for `check_b`.
7. `fusion_probe` uses the private `sys._is_gil_enabled()` — acceptable for a test-only probe, just noting it.

## Conclusion

The new fusion does what the task claims: a real pattern rewrite producing `exec.fused` regions that retain every ordered leaf call, gated by conservative trait-based eligibility, re-verified cold after decode (including the new positional stage-order checks A now needs), and realized as a genuinely different dispatch (one unrolled generated coroutine per group) under the same observer ABI. Failure, cancellation-drain, corruption, optional-Torch, fresh-process and C-no-op behaviors are all pinned by tests whose count arithmetic is consistent with the stated totals. The observations above are accuracy/hygiene items, suitable for follow-up rather than blocking this handoff.


## Parent follow-up

The manifest now records actual formed groups (zero for barrier-only plans) and the obsolete Prepared.maps inspection field is removed. All356 tests pass19.94s with GIL false before/after; no-Torch196pass+2optional skips6.42s and static gates pass. These are small diagnostic/unused-field corrections, not changes to the hot guard/binding path, and do not require a second full review.

Raw timing snapshots and their measured source hashes remain intact, including the fused outlier. final-*-bindings.json hold refreshed manifests; the fused A manifest is refreshed without resampling its numerical observations. Cold timing describes the measured source snapshot before these minor preparation/manifest cleanups; no exact final-code or production performance claim is inferred.
