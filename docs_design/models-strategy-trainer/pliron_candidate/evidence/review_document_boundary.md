# Code Review: Pliron whole-document parsing fix + requirements.txt gitignore exception

## Verdict

**No actionable findings.** Both fixes are correct, narrow, and adequately regression-tested. Evidence claims are internally consistent and verified against the repository where possible. Two minor, non-blocking observations below.

---

## 1. Parsing completeness/trivia behavior — correct

- **Single choke point, all boundaries covered.** All five parse call sites route through the shared `parse()` (`prepare.rs:85`): `accept` at :386 and the round-trip re-parse at :389, `prepare` at :696, and `load` parsing *both* documents at :750 and :753. Repo search confirms `top_level_parser` and `parse_from_str` appear exactly once. There is no bypass path — accept, prepare, and both cold-load documents now all require full consumption.
- **Composition semantics verified against the pinned dependency.** `Cargo.lock` pins `pliron 0.20.0` (matching `Cargo.toml`'s `=0.20.0`) with `combine 4.6.9`. In combine 4.6.x, `spaces()` is `skip_many(space())` where `space()` satisfies `char::is_whitespace` — Unicode-aware, which is exactly why the `\u2003` (em space) test case passes. `.skip(spaces()).skip(eof())` is the correct idiom; `eof()` fails on the first unconsumed non-whitespace character, so `"\nunchecked garbage"`, a second root, and `"// comment"` are all rejected as claimed.
- **Strictly tightening, no false rejections.** Previously-accepted documents (with or without trailing whitespace) still parse: `spaces()` is zero-or-more, and the printed canonical text re-parsed at :389 ends with a newline. The change cannot regress previously-valid input.
- **Deliberate trivia policy is documented.** README's "no comment syntax in this grammar; comment-like suffixes are rejected, not stripped" matches the pinned-source claim (the `src/parsable.rs` hash is now recorded in qualification.json; pliron source itself is not in-repo, so this is attested evidence rather than verifiable here — acceptable).
- Error path prefix `"parse: {e}"` matches the tests' `pytest.raises(Rejected, match="parse:")` through the JSON stderr channel in `main.rs`/`bridge.py`.

## 2. Regression adequacy — good

- **12 rejection cases** = 4 boundaries (accept, prepare, load_accepted, load_prepared) × 3 suffixes. Both cold-load documents are independently corrupted — this directly matches the "both documents" claim.
- **The monkeypatch guard is meaningful.** `monkeypatch.setattr(bridge_module.local_runtime, "compile_local", forbidden)` patches the same module object `_materialize` dereferences at call time, so an AssertionError fires if a bad document ever reaches executable projection. Before the fix, these tests fail either via `DID NOT RAISE` (accept) or via the `forbidden` AssertionError (prepare/load) — consistent with the recorded 12 pre-fix failures at 2.49 s.
- **3 whitespace controls** (`""`, ASCII, Unicode) exercise accept→prepare→load round-trip. The recorded `before.whitespace_tests_passed: 3` is honest: pre-fix, whitespace remainders were silently dropped, so those tests also passed before — correctly not counted among the 12 failures.
- Count math checks out: 103 + 15 = 118; 364 + 15 = 379 (261 existing + 3 Torch skips unchanged).

## 3. Packaging / fresh-env claims — verified

- Root `.gitignore:52` is indeed `*.txt`, confirming the original finding. The nested `!requirements.txt` in `pliron_candidate/.gitignore` is the correct remedy: patterns in deeper `.gitignore` files override parent-file patterns, and no parent *directory* is excluded (so the "cannot re-include under an excluded directory" caveat does not apply). No pattern in either file re-ignores it afterward.
- `requirements.txt` → `-r ../xdsl_candidate/requirements.txt`, which pins **exactly 12 packages** (immutabledict, iniconfig, ordered-set, packaging, pluggy, pygments, pytest, pytest-asyncio, ruff, ty, typing-extensions, xdsl) — the "12 packages installed offline" claim matches. Relative `-r` includes resolve against the including file's directory, and `uv pip sync ... pliron_candidate/requirements.txt` is documented in README (:107–110), satisfying "documented requirements".
- The file is currently untracked (absent from `list_changed_files`/diff), consistent with the stated no-commit posture; with the exception it is now visible to ordinary `git add` and to `ls-files --others --exclude-standard`.

## 4. Evidence accuracy — consistent

- `test_run.json` seconds (21.91028824600653) == qualification `final_passing_run_wall_seconds`; counts (379/118/261/3) match README and the `external_review_correction.after` block; `before` block matches README's "twelve failed / fifteen pass" narrative.
- `requirements_sha256` in the correction block matches the pre-existing `source_sha256["requirements.txt"]` entry (9ccc4f02…); `.gitignore` is now self-hashed; changed sources (`prepare.rs`, `test_b.py`, `README.md`) all have updated hashes.
- Review status correctly downgraded to `pending` with the prior completed review preserved verbatim under `previous_review` — matches README's "The correction's required review is pending".
- CHANGELOG/ROADMAP entries accurately describe both corrections; bead bump and reviewed commit `5a737c27` are recorded.

## 5. Narrow diff — confirmed

Exactly the 8 focus files changed; test_b.py changes are purely additive; no runtime/architecture/backend changes; no commit/push (untracked state preserved).

---

## Minor observations (non-blocking)

1. **The git-visibility "requirement" is recorded, not enforced.** "Evidence source hashes now also require every source path visible in `git ls-files --cached --others --exclude-standard`" exists only as prose plus the boolean `all_source_manifest_files_visible_to_normal_git_add: true` — no script, hook, or check in the repository contains `ls-files`/`exclude-standard`. If this is meant as a durable control ("cannot silently disappear again"), consider recording the exact command alongside the other gate commands in qualification.json, or a trivial check script. As a one-time manual verification it is adequate.
2. **Rejection suffixes always begin with `\n`.** No case appends non-whitespace directly after the final token (e.g., `accepted + "garbage"`), and no rejection case mixes Unicode whitespace with garbage. Both are logically rejected by the same `spaces()+eof()` composition, so this is a coverage nicety, not a gap.

Both observations are informational; neither blocks accepting these changes.
