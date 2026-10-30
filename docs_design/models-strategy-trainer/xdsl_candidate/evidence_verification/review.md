I have now reviewed the complete change set (tracked diff + untracked `test_plan_verification.py` and `evidence_verification/`), the surrounding fixtures/contracts, and the unchanged consumers. Here is my review.

# Review: prepared-plan correspondence repair (sd-scripts-syv.31.14)

## Verdict

**Approve.** The repair correctly closes both externally-found defects, is well-staged (all cold verification precedes model loading / runtime construction), and the descriptor strings are treated as checked constants rather than an interpreter, as claimed. I found no blocking correctness holes. A few minor observations and pre-fusion notes follow.

## Confirmation the two findings are actually closed

**B (horizon 0 / unsupported target / changed feedback accepted):**
- `prepare.py:256-278` (`check_plan`) rejects any decoded envelope whose digest/target/contract/stop_rule/feedback/horizon/capacity differ from the accepted session and the validated target, plus unknown properties and open bodies. The old code went straight from `parse`/`verify` to chain scanning with no envelope check at all — the gap is genuinely closed, not narrowed.
- `check_scalar_recipes` (prepare.py:324-372) requires exact property-set equality for `exec.select`/`exec.handoff`/`exec.feedback` (extra or missing props fail `properties != properties`), and the entity multiset comparison (len + set of sorted property tuples) correctly rejects clones-that-drop/duplicate entities.
- Chain checks (prepare.py:500-513) now pin transport, exact property set, and — importantly — **decoded origin order against semantic order**, so a whole-call permutation can't slip through even though `check_calls` alone is order-insensitive.
- `prepare.py:378` `accept(module) != accepted` (dataclass equality incl. obligations) closes the weakened-obligations path (tested).

**A (swapped compatible map symbols under unchanged origins):**
- `check_calls` (prepare.py:281-321) requires, per origin: exact property set, uniqueness/completeness against the spec set (`seen != expected.keys()`), **exact symbol equality with the semantic op's own `implementation`**, binding role/effect/version/completion match against the live registry, and exact ancestry recomputed from the accepted module. A swap between two same-role/effect/version bindings fails the symbol comparison — this is precisely the fixed defect, and `test_swapped_compatible_map_symbols_cannot_redefine_accepted_stages` (both lowerings) plus the B-side `swapped_symbols` case pin it.
- `model_adapter.py:415-478` stages everything before `lower_model`/`backend.prepare`: structural wrap → envelope → unsupported-op filter → exactly one recipe with exact property set → semantic body must re-accept to the same digest → recipe protocol/identity → `check_calls` → address property-set/membership/port checks. The load-probe test asserts no binding `invoke` ran on rejection.

## Staging, unknown/duplicate/missing descriptors, retained recipe, Torch separation

- **Failure staging** is clean: parse/verify failures are wrapped (`prepare.py:479-483`, `model_adapter.py:415-419`), semantic corruption is reported as semantic ("changed/invalidated the accepted meaning"), envelope/protocol/binding/address failures each have distinct messages. One nit below on the `except Rejected` narrowness.
- **Missing descriptors**: required `prop_def`s are caught by `loaded.verify()` inside the try/except → `NotReady("invalid structure")`; optional ones (`horizon`, `capacity`, `feedback`) fall through to explicit inequality. Covered by the `missing=True` parametrizations on both sides.
- **Retained semantic recipe** (`model_adapter.py:429-444`): the recipe body must be exactly one SessionOp whose re-accepted digest equals `accepted.digest` — this is the strongest possible check (byte-identical canonical text) and transitively protects everything nested inside it (nested CallOps/AddressOps/sessions would change the digest).
- **Optional-Torch separation**: the new cold-verification core (`check_plan`, `check_calls`, `check_scalar_recipes`) lives in Torch-free `prepare.py` and is reused by `model_adapter.py`; `test_plan_verification.py` imports no Torch, so the no-Torch run (165 pass + 2 skips) exercises the B checks without the optional backend. Good.

## Evidence consistency

- `mutation_results.json`: before/after match the stated findings exactly (A swap accepted + executed with oracle mismatch `12.0 != 11`; B horizon 0/target accepted; B feedback mutation passed even the original oracle). After: all five are `NotReady` with the messages the code actually produces.
- Test math checks out: 42 new cases in `test_plan_verification.py` + 17 in `test_model_runtime.py` = **59**; 245 + 59 = **304**. README/ROADMAP/CHANGELOG wording is consistent, and the README authority table matches the implementation field-for-field (including that A's `capacity=2` and B's `capacity=1` are hard target constraints and that runtime consumes decoded `root.horizon`/`root.capacity`).
- `test_ir.py:213` regex widening is correct: the `wrong_call_role` corruption now fails earlier in `check_calls` ("...binding/completion contract differs"); `duplicate_chain` still exercises the chain-role path itself.

## Minor findings (non-blocking)

1. **`model_adapter.py:433-436`** — the digest re-accept catches only `Rejected`. A decoded body that passes IRDL `verify()` but trips a non-`Rejected` exception in `accept_model` would surface as something other than `NotReady`. This is theoretical (extension verification is `require`-based), but for uniform readiness staging consider catching `Exception` there, mirroring the parse wrap a few lines above.
2. **Untracked evidence files** — `evidence_verification/` (and `test_plan_verification.py`) must actually be added to the commit; the qualification file records `"review": {"status": "pending"}` which should be updated at handoff once this review is accepted.
3. **`test_prepared_dispatch_checks_chain_roles`** — with the widened regex, the `wrong_call_role` case no longer proves the chain-role check fires; only `duplicate_chain` does. Acceptable, but the test name now slightly oversells its coverage.
4. **Coverage nit**: there is no direct B mutation test for a *third chain* with a novel role or a missing entire chain. The logic rejects both (`role not in (...)`, `set(resolved) != {...}`), and `duplicate_chain` covers the adjacent path, so this is only a gap in regression pinning, not in behavior.

## Pre-fusion observations (explicitly not requested to fix now)

- **C divergence**: `grant_adapter.prepare_grant` still hand-rolls its decoded-plan verification and does *not* reject unknown executable properties on `PlanOp`/`CallOp`/`RecipeOp` the way `check_plan`/`check_calls` now do (no exact property-set equality). Unconsumed fields only, so no execution hole today, but before fusion there will be three divergent verification regimes; C should adopt the shared checkers or equivalent hygiene.
- **A call order is unchecked** — fine today because A consumes calls by origin lookup, but if fusion ever consumes positional calls, port B's `expected_sources` order check. Similarly, `check_calls`' correspondence is keyed by flat `origin`; a merged multi-recipe plan with cross-recipe origin collisions will need qualified correspondence (the current "ambiguous semantic binding sources" guard only sees one spec list).
- **Descriptor/runtime coupling**: `SCALAR_UPDATE` (`score%5`) is checked as a constant while `runtime._apply_feedback` independently implements `% 5`. That is the documented "checked contract, not interpreter" design and acceptable, but a future edit to one without the other would not be caught by cold verification — worth remembering when the fixed protocol is ever revised.

No production, governing, legacy-runtime, or root-dependency files were touched (only the candidate directory plus CHANGELOG/ROADMAP entries, consistent with AGENTS.md). The bounded scope — no interpreter, no fusion, no effect algebra — is respected throughout.


## Parent follow-up

Applied uniform readiness classification for A semantic re-acceptance and C decoded structural verification; added missing/novel chain and unknown C property regressions. IRDL already rejects undeclared properties, so the C change classifies that failure rather than establishing a previously missing structural rejection. Explicit C profile-property checks retain the known no-flow/owner contract. An initial follow-up run had306 pass and3 exception-classification failures; after the C wrapper correction,309 tests pass17.36s with GIL false before/after. No-Torch167 pass+2optional module skips; Ruff/format/ty/diff checks pass. Small follow-up corrections only; no second full review required.
