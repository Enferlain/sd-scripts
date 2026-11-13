<!-- Locally adapted from tiran/agent-skills@0c51c24122b33afed31fd8d50534e089534f77a7; see ../README.md#repository-installation. -->

# Background: what free-threading changes, and why

Read for the "why" — deciding whether to invest, or explaining a data race or a wheel
tag. Grounded in PEP 703, PEP 803, and the CPython free-threading HOWTO
(<https://docs.python.org/3/howto/free-threading-python.html>) and porting guide
(<https://py-free-threading.github.io/>).

## The build and how to detect it

Free-threaded CPython is a **separate build** (`configure --disable-gil`) that runs
threads through the interpreter in parallel instead of serializing them on the GIL. It
first shipped as **experimental in 3.13**; **3.14 is the baseline** — PEP 779 made
free-threading officially supported (no longer experimental) in 3.14. Target `3.14t`;
treat `3.13t` as a skippable experimental predecessor.

- **Build-time:** `sysconfig.get_config_var("Py_GIL_DISABLED") == 1` — the
  recommended check for build decisions. The C macro `Py_GIL_DISABLED` guards
  free-threaded-only code.
- **Runtime:** `sys._is_gil_enabled()` returns `False` when the GIL is actually off.
  `python -VV` / `sys.version` contain `"free-threading build"`.
- **ABI tag:** the interpreter's ABI gains a `t` — wheels are tagged `cp313t`,
  `cp314t`. These are **distinct** from the GIL build's `cp313`/`cp314`; a `t` wheel
  loads only on a `t` interpreter.

## The GIL can come back — that's the trap

A free-threaded interpreter **re-enables the GIL at runtime** if it imports a C
extension that hasn't declared support, printing a warning. So a package that "works"
(imports, passes tests) can still silently rob every user of parallelism. Declaring
support (SKILL step 4) is what prevents this; it is independent of whether the code is
actually race-free. Users can override with `PYTHON_GIL=0` or `-X gil=0` (force off) /
`PYTHON_GIL=1` (force on), but you cannot rely on that.

## Why removing the GIL breaks "accidentally safe" code

Under the GIL, only one thread ran bytecode at a time, so most operations on shared
objects were atomic by accident. Without it, two threads can mutate the same
`dict`/`list`/your-object concurrently → data races and undefined behavior. PEP 703
makes the *interpreter's own* structures safe (see below), but **your** global state,
caches, and object internals are your responsibility.

## What PEP 703 changed under the hood (why extensions must be rebuilt)

Not ABI-compatible with the GIL build — extensions are recompiled for `t`. The
mechanisms, useful for reasoning about hazards:

- **Reference counting** is now a mix of **biased reference counting** (each object
  has an owning thread that uses fast non-atomic ops; other threads use atomics),
  **immortalization** (interned strings, small ints, `None`/`True`/`False` never
  change count), and **deferred reference counting** (functions, modules, code
  objects reconciled at GC). Consequence: **borrowed references are unsafe** —
  another thread can drop the last reference while you hold a borrow.
- **`PyObject` grew** (`ob_tid`, `ob_mutex`, `ob_ref_local`, `ob_ref_shared`, …) and
  `ob_type` moved — this is why free-threaded needs its own ABI and why `abi3` wheels
  won't load on `t` (and why `abi3t` makes `PyObject` opaque; see PEP 803).
- **Per-object locks** (a one-byte `PyMutex` in the header) replace GIL protection for
  built-in containers; **critical sections** provide deadlock-safe locking around
  them. `dict[key]`/`list[idx]` reads use an optimistic lock-free fast path.
- **pymalloc → mimalloc** (thread-safe); the GC is now **non-generational** and
  **stop-the-world**.

## Cost / benefit

- **Benefit:** real multi-core parallelism for threaded Python + native code, no
  multiprocessing/pickling overhead. Adoption is already mainstream — a Sept 2026 scan
  of the top 360 PyPI packages found **57 of the 81 that ship compiled wheels already
  ship a free-threaded one** (~70%).
- **Cost:** single-threaded overhead on the free-threaded build (roughly 1–8% on
  pyperformance depending on OS/arch); the thread-safety audit is real engineering;
  and until 3.15's `abi3t` you ship a **second** wheel per Python version. The GIL
  build is unaffected — none of this slows down normal CPython.

## abi3 vs abi3t (the ABI axis, handed to the limited-API skill)

Until 3.15 there is **no** stable ABI for free-threading, so you ship a *separate*
version-specific `cp3Xt` wheel every year alongside your GIL-build wheels — and `abi3`
is simply ignored on a `t` interpreter (numpy and PyO3 both warn and fall back to a
version-specific FT build).

**[PEP 803](https://peps.python.org/pep-0803/) changes this for Python 3.15+:** `abi3t`
is the *stable ABI for free-threaded builds*, and a single **`abi3.abi3t`**-tagged,
`*.abi3t.so` wheel loads on **both** the standard (GIL) and free-threaded interpreter
from 3.15 up. That collapses today's four-way matrix (per-version × GIL/FT) back to one
wheel — the free-threaded analogue of what `abi3` already does across GIL-build Python
versions. It requires an opaque `PyObject` and PEP 793 `PyModExport`, and cannot wrap
variable-size types.

That ABI work lives in
[`port-to-python-limited-api`](https://github.com/tiran/agent-skills/blob/0c51c24122b33afed31fd8d50534e089534f77a7/skills/port-to-python-limited-api/SKILL.md); this skill
is about *thread-safety and declaring support*, which you need regardless of whether
you ship `cp3Xt` or `abi3t`. The two axes are orthogonal: this skill makes the code
race-free and declares support; the limited-API skill makes it *one wheel*.
