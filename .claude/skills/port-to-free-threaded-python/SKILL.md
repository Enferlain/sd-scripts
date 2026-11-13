---
name: port-to-free-threaded-python
description: >-
  Develop, review, or port Python code for CPython 3.14t free-threading
  (PEP 703/779). Use for shared-state ownership, worker/coordinator boundaries,
  GIL re-enabling, native-extension compatibility, concurrent testing, or cp3Xt
  wheels. Covers pure Python and C/C++/Cython/Rust bindings; extension build and
  packaging steps apply only when relevant.
---

<!-- Locally adapted from tiran/agent-skills@0c51c24122b33afed31fd8d50534e089534f77a7; see README.md#repository-installation. -->

# Port a package to free-threaded Python

## Use in this repository

Repository instructions and the user's authorized task take precedence over the
upstream workflow's scope, approval and clean-tree defaults. Preserve unrelated
work and use the existing uv environment. Apply only the steps relevant to the
current task; native wheel builds and instrumented TSan matrices are not gates
for a pure-Python candidate change.

Verify actual GIL state after ordinary imports without `PYTHON_GIL=0` or
`-X gil=0`. Forced-off race testing is a separate, controlled native-extension
check; it cannot establish normal-import compatibility or dependency safety.

Use the [CPython contributor free-threading guidance](https://python.ca/cpython-guidance/free-threading/)
for reasoning about access guarantees, ownership, synchronization scope and tests
that actually reach conflicting accesses. Its C/interpreter-specific mechanisms
are not prescriptions for Python application architecture. The upstream skill is
experimental; check version-specific API claims against the installed versions'
official documentation.

Free-threaded CPython (PEP 703; experimental in 3.13, officially supported from **3.14**
per PEP 779) runs without the GIL, so threads execute Python in parallel. Target the
`3.14t` build. The catch: the GIL used to serialize every operation, so
code that was "accidentally thread-safe" no longer is. This skill takes a package to
**correct behavior** under the free-threaded build and to **declaring support**, which
are two separate jobs — an extension that imports fine can still silently re-enable
the GIL, and one that declares support can still race.

Read [`reference/GUARDRAILS.md`](reference/GUARDRAILS.md) first. It governs installs, builds, and
not committing/pushing without approval.

**Scope.** Pure-Python packages need only steps 1–3 and 8. Packages that ship a
compiled extension need all steps. This skill is language-agnostic at the binding
level (C/C++, Cython, pybind11, nanobind, Rust/PyO3); it does **not** teach the ABI
mechanics of a single-wheel `abi3t` build — for that, run
[`port-to-python-limited-api`](https://github.com/tiran/agent-skills/blob/0c51c24122b33afed31fd8d50534e089534f77a7/skills/port-to-python-limited-api/SKILL.md) (PEP 803 + PEP
793) after this one. Thread-safety and ABI stability are orthogonal.

## 1. Scope, prior work, and strategy

- **Prior-work check (GUARDRAILS).** Search the project's issues/PRs and the
  ecosystem tracker (<https://py-free-threading.github.io/tracking/>) for existing
  free-threading work before starting — much of the top 360 is already done or in
  flight.
- **Classify the package.** Pure Python → thread-safety review only. Ships an
  extension → the full workflow. List every compiled module and every C/Rust
  dependency (a thread-unsafe dependency blocks you — see step 5).
- **Pick a wheel strategy** (details in step 6): version-specific `cp313t`/`cp314t`
  wheels (the norm today), or — only once 3.15 is your floor — a single
  `abi3.abi3t` wheel via the limited-API skill.

## 2. Get a free-threaded interpreter and a baseline

Install a free-threaded interpreter and confirm the GIL is actually off:

```bash
uv python install 3.14t            # 3.14t is the baseline; 3.13t was experimental
uv run --python 3.14t python -c "import sys; print(sys._is_gil_enabled())"   # -> False
```

Then import your package under it and **watch for the warning**
`"The global interpreter lock (GIL) has been enabled to load module ..."`. If it
prints, the extension has not declared support and every user silently loses
parallelism (step 4 fixes it). `sysconfig.get_config_var("Py_GIL_DISABLED") == 1` is
the build-time check; `sys._is_gil_enabled()` is the runtime one.

## 3. Pure-Python thread-safety pass

Applies to every package. The GIL used to make these safe; now they race. See
[`reference/thread-safety.md`](reference/thread-safety.md).

- **Mutable module-, class-, and function-level state** — lazy caches, memoization
  dicts, registries filled on first use. The top source of bugs.
- **`contextvars.ContextVar`** for per-logical-thread state (asyncio-aware);
  `threading.local` only for genuine per-OS-thread state.
- **Copy-on-write reads** — snapshot a shared reference into a local before using it
  so a concurrent replacement can't tear it.
- **Locks** — `threading.Lock`/`RLock` as context managers around read-modify-write
  on shared objects.

## 4. Declare free-threading support

This is the single highest-value change for an extension: it stops the GIL being
re-enabled. It is a **promise**, so only make it after (or alongside) steps 3 and 5.
See [`reference/declaring-support.md`](reference/declaring-support.md).

- **Multi-phase init (C/C++):** add the `{Py_mod_gil, Py_MOD_GIL_NOT_USED}` slot
  (version-guarded, so it still compiles on 3.12).
- **Single-phase init (C/C++):** call
  `PyUnstable_Module_SetGIL(mod, Py_MOD_GIL_NOT_USED)` under `#ifdef Py_GIL_DISABLED`.
- **Cython** `# cython: freethreading_compatible=True` (3.1+); **pybind11**
  `py::mod_gil_not_used()`; **nanobind** `FREE_THREADED` CMake flag; **PyO3** free-threaded
  by default since 0.28 (write `gil_used = true` only to opt *out*; 0.23–0.27 need
  `gil_used = false`). All declare support only — none makes code safe.
- Add the trove classifier
  `Programming Language :: Python :: Free Threading :: 3 - Stable` (or `2 - Beta` /
  `1 - Unstable`) so users can see the status.

## 5. Detect unsafe shared state; fix only the simple cases

For a compiled extension, **detection is the primary deliverable**. Inventory every
piece of state reachable from more than one thread and classify each finding, then act
within the scope boundary below. Detection heuristics and the fix patterns are in
[`reference/thread-safety.md`](reference/thread-safety.md).

**Safe to fix in place** (local, self-contained; patterns and caveats in the reference):

- **Borrowed-reference hazards** → strong-ref getters (`PyList_GetItemRef`,
  `PyDict_GetItemRef`, `PyWeakref_GetRef`; backport via `pythoncapi-compat`). Don't
  blanket-convert — arguments and new references stay safe.
- **A single atomic counter or flag** → C11 `<stdatomic.h>` (C++/Rust std) with the
  weakest correct memory order, `#ifdef Py_GIL_DISABLED`-guarded. Anything wider than
  one word — a CAS loop or a hand-rolled lock-free structure — is a report, not a fix.
- **A self-contained read-modify-write on one object you own** → a critical section
  (`Py_BEGIN_CRITICAL_SECTION`) or a `static PyMutex`. Mind the two rules: never nest to
  lock two objects (use the `_2` form); a section is suspended across blocking calls.
- Optionally add a **borrowed-ref CI linter** (numpy's `check_c_api_usage.py`) so these
  fixes can't silently regress.

**Detect and report to the user — do NOT attempt** (out of scope; these need design
judgement and can introduce subtle deadlocks or performance cliffs):

- Shared mutable **data structures** that need re-architecting (custom containers,
  intrusive caches, object graphs with cross-references).
- **Global/module state** whose fix is a structural change (thread-local conversion
  across the codebase, ownership redesign).
- **Non-reentrant C/C++/Fortran libraries** and lock-hierarchy decisions — where to
  serialize, and the parallelism trade-off, is the maintainer's call.

For each such finding, tell the user **where** (file:line), **what** state is shared,
**why** it races without the GIL, and the **candidate fix** — then stop and let them
decide. Legitimate outcomes for this category, not just "add a lock" (scipy does all
three): **detect-and-refuse** concurrent use with a clear error, or **document a
subsystem as not thread-safe** and quarantine its tests. Report these clearly rather
than declaring the port done.

## 6. Build free-threaded wheels

The wheel tag follows the interpreter you build with — build with a `t` interpreter
and you get a `cp3Xt` wheel. See
[`reference/build-and-test.md`](reference/build-and-test.md) for meson-python,
scikit-build-core, maturin, and setuptools specifics.

- In CI, have **cibuildwheel** build the free-threaded targets:
  `CIBW_ENABLE=cpython-freethreading` (or `enable = ["cpython-freethreading"]`).
- Ensure the `Py_GIL_DISABLED`-guarded support declaration from step 4 is compiled in.
- **Python 3.15+ collapses the matrix to one wheel.** Under
  [PEP 803](https://peps.python.org/pep-0803/), a single **`abi3.abi3t`**-tagged wheel
  (a `*.abi3t.so`) loads on **both** the standard (GIL) and free-threaded interpreter
  from 3.15 up — no more separate `cp3Xt` build. That ABI work (opaque `PyObject` +
  PEP 793 `PyModExport`) lives in
  [`port-to-python-limited-api`](https://github.com/tiran/agent-skills/blob/0c51c24122b33afed31fd8d50534e089534f77a7/skills/port-to-python-limited-api/SKILL.md); pair it with
  this skill. Until your floor is 3.15, ship version-specific `cp314t` wheels.

## 7. Test under load

Concurrency bugs don't show single-threaded. See
[`reference/build-and-test.md`](reference/build-and-test.md).

- **`pytest-run-parallel`** (`--parallel-threads=auto`) runs the existing suite from
  many threads at once — the cheapest way to shake out races.
- **ThreadSanitizer** — build the extension with `-fsanitize=thread` and run the suite;
  it finds data races the tests don't assert on. Use a **TSan-instrumented CPython**
  (and instrument native deps like OpenBLAS too) — running TSan against a stock
  free-threaded interpreter floods false positives. **Test on ARM too** — some races
  only manifest on weakly-ordered CPUs.
- Write targeted multithreaded stress tests for the shared structures you touched.

## 8. Verify and declare done

- Importing the package on a free-threaded build prints **no** GIL-re-enable warning
  and `sys._is_gil_enabled()` stays `False`.
- The suite passes under `pytest-run-parallel` and TSan is clean on x86 **and** ARM.
- Wheels carry the `cp313t`/`cp314t` tag (or `abi3.abi3t`), and the classifier
  advertises the support level you actually verified — on every version you claim.
- Any complex shared-state findings from step 5 that were **out of scope to fix** are
  reported to the user with location, cause, and a candidate fix. Don't declare support
  (`3 - Stable`) while known races are unresolved — use `1 - Unstable`/`2 - Beta` or
  leave the declaration off until the maintainer addresses them.
