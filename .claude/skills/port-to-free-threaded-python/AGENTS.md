<!-- Locally adapted from tiran/agent-skills@0c51c24122b33afed31fd8d50534e089534f77a7; see README.md#repository-installation. -->

# AGENTS.md — port-to-free-threaded-python

Cross-agent entry point for this skill (Codex CLI and any agent that reads
`AGENTS.md`). Claude Code loads `SKILL.md` directly via its frontmatter; both share the
same content and reference files, so there is nothing to keep in sync here.

## When to use

A package should work on the **free-threaded (no-GIL) CPython build** (PEP 703; the
`cp313t`/`cp314t` "t" ABI) and **declare** that it does, so importing it doesn't
silently re-enable the GIL. Pure-Python packages need the thread-safety pass and the
declaration; C/C++/Cython/pybind11/nanobind/Rust-PyO3 extensions additionally need
data-race fixes and a `cp3Xt` wheel. Use it to add free-threading / no-GIL / nogil
support, ship a free-threaded wheel, or stop the GIL being re-enabled.

**Scope guard:** the skill **detects** all unprotected shared state and **fixes only
the simple, local cases** (borrowed→strong reference swaps, a single atomic, a
self-contained critical section / small `PyMutex` region, the support declaration).
Complex C/C++ shared-state redesign, structural global→thread-local conversion, and
non-reentrant-library locking strategy are **out of scope to fix** — it reports those
to the user with location, cause, and a candidate fix.

**Not** the ABI mechanics of a single-wheel `abi3t` build (PEP 803 + PEP 793) — pair
with [`port-to-python-limited-api`](https://github.com/tiran/agent-skills/blob/0c51c24122b33afed31fd8d50534e089534f77a7/skills/port-to-python-limited-api/SKILL.md) for that.
Thread-safety and ABI stability are orthogonal.

## How to run it

1. Read **`SKILL.md`** — the ordered 8-step workflow. Follow it top to bottom.
2. Open a bundled reference only when a step cites it:
   - `reference/background.md` — what free-threading is, PEP 703 internals, detection,
     the GIL-re-enable trap, cost/benefit, abi3 vs abi3t (the "why").
   - `reference/declaring-support.md` — `Py_mod_gil` / `PyUnstable_Module_SetGIL` and
     the per-binding declarations + the trove classifier.
   - `reference/thread-safety.md` — detection heuristics, the detect-vs-fix scope
     boundary, borrowed-reference hazards, critical sections / `PyMutex` / atomics,
     global state and caches.
   - `reference/build-and-test.md` — `cp3Xt` wheels per backend + cibuildwheel; TSan,
     `pytest-run-parallel`, running the GIL-disabled interpreter.

## Definition of done

Importing on a free-threaded build prints no GIL-re-enable warning and
`sys._is_gil_enabled()` stays `False`; the suite passes under `pytest-run-parallel` and
TSan is clean (x86 and ARM); wheels carry `cp313t`/`cp314t` (or `abi3.abi3t`); the
classifier states the level actually verified; and any out-of-scope shared-state
findings are reported to the user rather than silently left or misdeclared as stable.

## Sources & acknowledgments

This skill is grounded in real projects and their maintainers — acknowledge them when
the output leans on their work, and preserve upstream license/attribution:

- **Porting guides & tooling** — **Quansight** and the **free-threading community**, for
  the [py-free-threading](https://py-free-threading.github.io/) porting guides and
  [`pytest-run-parallel`](https://github.com/Quansight-Labs/pytest-run-parallel).
- **Runtime & C-API docs** — the **CPython** developers, for the
  [free-threading HOWTOs](https://docs.python.org/3/howto/free-threading-python.html).
- **PEP 703** — **Sam Gross** (making the GIL optional).
- **PEP 803** — **Petr Viktorin** (`abi3t`).
- **Adoption data** — **Hugo van Kemenade**, for the
  [free-threaded-wheels](https://hugovk.dev/free-threaded-wheels/) adoption tracker.
- **Real-world ports** — the **numpy**, **scipy**, and **PyO3** teams.

See [`../reference-repos.md`](https://github.com/tiran/agent-skills/blob/0c51c24122b33afed31fd8d50534e089534f77a7/skills/reference-repos.md) for the full grounding list.
