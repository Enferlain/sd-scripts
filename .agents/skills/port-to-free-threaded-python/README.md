<!-- Locally adapted from tiran/agent-skills@0c51c24122b33afed31fd8d50534e089534f77a7; see README.md#repository-installation. -->

# Skill: port a package to free-threaded Python

## Repository installation

Imported on 2026-10-25 from
[`tiran/agent-skills` at `0c51c24122b33afed31fd8d50534e089534f77a7`](https://github.com/tiran/agent-skills/tree/0c51c24122b33afed31fd8d50534e089534f77a7/skills/port-to-free-threaded-python),
under the bundled [Apache-2.0 license](LICENSE). Upstream labels the skill experimental.
The original acknowledgments and supporting references are retained.

Local adaptations shorten the discovery description, add the repository-use and
CPython-guidance notes in [SKILL.md](SKILL.md#use-in-this-repository), keep the shared
[guardrails](reference/GUARDRAILS.md) inside this skill, distinguish normal-import
qualification from forced-off testing, and direct optional sibling-skill links to
the pinned upstream source. Other skills and runtime libraries are not bundled.


Take a package to **correct behavior on the free-threaded (no-GIL) CPython build**
([PEP 703](https://peps.python.org/pep-0703/); the `cp313t`/`cp314t` "t" ABI) and to
**declaring support**, so importing it
doesn't silently switch the GIL back on. For pure Python that's a thread-safety review;
for a C/C++/Cython/pybind11/nanobind/Rust-PyO3 extension it also means fixing data
races and shipping a `cp3Xt` wheel. The content is framework-neutral Markdown; the same
files drive multiple coding agents.

## Why do this (and why not)

**Payoff.** Real multi-core parallelism for threaded Python plus native code, without
multiprocessing's pickling/IPC overhead. It's already mainstream — a Sept 2026 scan of
the top 360 PyPI packages found **~70% of those shipping compiled wheels already ship a
free-threaded one**. And if an extension *doesn't* declare support, every user on a
free-threaded interpreter silently loses parallelism: importing it re-enables the GIL
process-wide.

**Two separate jobs, kept distinct.** *Declaring support* (a `Py_mod_gil` slot or
one-line binding flag) is what stops that GIL re-enable — cheap and a no-op on the normal
build. *Being race-free* is the real engineering: without the GIL, code that was
"accidentally thread-safe" (shared caches, module globals, borrowed references) now
races. A declaration not backed by safe code just ships a lie.

**Scope — detect broadly, fix conservatively.** The skill flags **all** unprotected
shared state but only fixes the simple, local cases (borrowed→strong swaps, a single
atomic, a self-contained critical section / `PyMutex`, the declaration). Structural
redesign — global→thread-local conversion, non-reentrant-library locking, multi-lock
ordering — is reported to the maintainer, not attempted.

**Costs.** ~1–8% single-threaded overhead on the free-threaded build (the normal build
is untouched); the audit is genuine work; and until Python 3.15's `abi3t` you ship a
*second* wheel per version. The full rationale, PEP 703 internals, and detection
heuristics live in [`reference/background.md`](reference/background.md) and
[`reference/thread-safety.md`](reference/thread-safety.md).

**When _not_ to reach for this skill.** The single-wheel **`abi3t`** ABI mechanics
([PEP 803](https://peps.python.org/pep-0803/) + [PEP 793](https://peps.python.org/pep-0793/)
`PyModExport`, opaque `PyObject`) belong to
[`port-to-python-limited-api`](https://github.com/tiran/agent-skills/tree/0c51c24122b33afed31fd8d50534e089534f77a7/skills/port-to-python-limited-api) — thread-safety and ABI
stability are orthogonal; run this for the former, that for the latter.

## Layout

The workflow (in `SKILL.md`) runs: get a free-threaded interpreter and a baseline → do
the pure-Python thread-safety pass → declare support → detect races and fix the simple
ones (report the rest) → build `cp3Xt` wheels → test under load (`pytest-run-parallel`,
TSan) → verify and declare done. The files:

| File | Role |
| --- | --- |
| `SKILL.md` | The workflow — 8 ordered steps + frontmatter. **Source of truth.** |
| `AGENTS.md` | Cross-agent entry point (Codex and other `AGENTS.md`-aware agents). |
| `reference/background.md` | What free-threading is, PEP 703 internals, detection, the GIL-re-enable trap, cost/benefit, abi3 vs abi3t — the "why". |
| `reference/declaring-support.md` | `Py_mod_gil` / `PyUnstable_Module_SetGIL`, the per-binding declarations (Cython/pybind11/nanobind/PyO3), the trove classifier. |
| `reference/thread-safety.md` | Detection heuristics, the detect-vs-fix scope boundary, borrowed-reference hazards, critical sections / `PyMutex` / atomics, global state and caches. |
| `reference/build-and-test.md` | `cp3Xt` wheels per backend + cibuildwheel; TSan, `pytest-run-parallel`, running the GIL-disabled interpreter. |

`SKILL.md` and `AGENTS.md` point at the same `reference/` files; only `SKILL.md`
carries the full step list, so there is no duplicated workflow to maintain.

## Related

- [`port-to-python-limited-api`](https://github.com/tiran/agent-skills/tree/0c51c24122b33afed31fd8d50534e089534f77a7/skills/port-to-python-limited-api) — the `abi3`/`abi3t`
  ABI work; pair with this skill to ship **one** `abi3.abi3t` wheel instead of a
  `cp3Xt` per version.
- [`pybind11-to-nanobind`](https://github.com/tiran/agent-skills/tree/0c51c24122b33afed31fd8d50534e089534f77a7/skills/pybind11-to-nanobind) — nanobind has first-class
  free-threading support (`FREE_THREADED`).
- [`choose-python-build-backend`](https://github.com/tiran/agent-skills/tree/0c51c24122b33afed31fd8d50534e089534f77a7/skills/choose-python-build-backend) — picks the backend
  whose free-threaded wheels this skill builds.

## Further reading

- CPython [Free-threading HOWTO](https://docs.python.org/3/howto/free-threading-python.html)
  · [Free-threading C-API guide](https://docs.python.org/3/howto/free-threading-extensions.html).
- The community porting guides:
  [py-free-threading.github.io](https://py-free-threading.github.io/) —
  [porting Python](https://py-free-threading.github.io/porting/),
  [porting extensions](https://py-free-threading.github.io/porting-extensions/),
  [tracking](https://py-free-threading.github.io/tracking/).
- [PEP 703](https://peps.python.org/pep-0703/) (making the GIL optional) ·
  [PEP 803](https://peps.python.org/pep-0803/) (`abi3t`) ·
  [PEP 793](https://peps.python.org/pep-0793/) (`PyModExport`).
- Adoption trackers: [hugovk.dev/free-threaded-wheels](https://hugovk.dev/free-threaded-wheels/).

## Acknowledgments

This skill distills the work of many people and projects:

- **Quansight** and the **free-threading community** — the
  [py-free-threading](https://py-free-threading.github.io/) porting guides and
  [`pytest-run-parallel`](https://github.com/Quansight-Labs/pytest-run-parallel).
- The **CPython** developers — the
  [free-threading HOWTOs](https://docs.python.org/3/howto/free-threading-python.html).
- **Sam Gross** — [PEP 703](https://peps.python.org/pep-0703/) (making the GIL optional).
- **Petr Viktorin** — [PEP 803](https://peps.python.org/pep-0803/) (`abi3t`).
- **Hugo van Kemenade** — the
  [free-threaded-wheels](https://hugovk.dev/free-threaded-wheels/) adoption tracker.
- The **numpy**, **scipy**, and **PyO3** teams — real-world ports.

See [`../reference-repos.md`](https://github.com/tiran/agent-skills/blob/0c51c24122b33afed31fd8d50534e089534f77a7/skills/reference-repos.md) for the full grounding list.

See the repository [`README.md`](https://github.com/tiran/agent-skills/blob/0c51c24122b33afed31fd8d50534e089534f77a7/README.md) for per-agent setup (Claude Code,
Codex, and others).
