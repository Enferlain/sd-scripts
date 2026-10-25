# Making code thread-safe under free-threading

SKILL steps 3 and 5. Without the GIL, anything that mutates state shared between
threads can race. This is the actual engineering; declaring support (step 4) without
doing this ships a lie. The rule of thumb: **find every piece of state reachable from
more than one thread and give each a strategy** — make it immutable, make it
thread-local, or guard it with a lock.

## Scope: detect everything, fix only the simple cases

Detection is the deliverable; apply a fix only when it's local and obviously correct.
The full fix-vs-report boundary is in [SKILL step 5](../SKILL.md) — this file is the
*how* (heuristics and patterns). Throughout, **✅** tags a pattern that's safe to fix in
place and **🚩** one to report to the maintainer (file:line → what's shared → why it
races → candidate fix) rather than attempt.

## Finding shared state (detection heuristics)

Grep and read for state that outlives a single call and isn't per-thread:

- C/C++: `static` non-const variables and file-scope globals inside the extension;
  `PyObject *` module-level caches; `PyModule_GetState` fields mutated after init;
  writable data inside a linked C library.
- Cython: module-level `cdef` globals and mutable attributes on `cdef` classes.
- Any language: lazy singletons/memoization filled on first use, registries,
  free-lists/object pools, buffers reused across calls, and borrowed-reference getters
  (`PyList_GetItem`/`PyDict_GetItem`) whose result is used after a point another thread
  could mutate the container.

A finding is only a *bug* if the state is reachable from two threads and at least one
writes. Read-only-after-init state (populated during import, then immutable) is fine.

## Python level (all packages)

- **Mutable global / class / function state** is the top bug source: lazy caches,
  `@lru_cache`-style dicts filled on first use, registries, module-level mutable
  defaults. Two threads hitting an unpopulated cache race.
- **`contextvars.ContextVar`** for per-logical-thread state — it is asyncio- and
  thread-aware. `threading.local` only when you truly mean per-OS-thread.
- **Copy-on-write reads:** never mutate a shared container in place; publish a new
  one and have readers snapshot the reference into a local first
  (`local = self._shared; use(local)`), so a concurrent swap can't tear the read.
- **Locks:** `threading.Lock`/`RLock` as context managers around any
  read-modify-write. `lock.acquire(blocking=False)` can *detect* unexpected concurrent
  use and raise, if sharing is a bug in your API.

## Borrowed references (C extensions)

A **borrowed reference can be freed by another thread while you hold it** — the classic
new hazard. Even `Py_NewRef(PyList_GetItem(l, 0))` races: the item can be dropped
between the borrow and the incref. Fix by fetching a **strong** reference atomically.
All the `*Ref` getters are **new in 3.13**:

| Unsafe (borrowed) | Safe (strong) | Returns |
| --- | --- | --- |
| `PyList_GetItem` / `PyList_GET_ITEM` | `PyList_GetItemRef` | new ref, or `NULL`+`IndexError` |
| `PyDict_GetItem` / `PyDict_GetItemWithError` / `PyDict_GetItemString` | `PyDict_GetItemRef` | **int** 1/0/-1, writes strong ref via out-param |
| `PyWeakref_GetObject` / `PyWeakref_GET_OBJECT` | `PyWeakref_GetRef` | **int** 1/0/-1, out-param |
| `PyImport_AddModule` | `PyImport_AddModuleRef` | new ref |

Mind the two calling conventions: `PyList_GetItemRef` returns the object (or `NULL`),
while `PyDict_GetItemRef`/`PyWeakref_GetRef` return an `int` status and write the
result through an out-parameter.

- **Don't blanket-convert.** Function *arguments* stay valid (the caller holds a
  reference), borrows of immortal/immutable objects (e.g. `PyTuple_GetItem` on a tuple
  you own) are fine, and a *new* reference is safe until your thread releases it.
- Backport the `*Ref` spellings to older Pythons with **`pythoncapi-compat`**
  (header-only, vendorable).
- `PyList_SET_ITEM` / `PyTuple_SET_ITEM` do **no** locking — use only to populate a
  freshly created, not-yet-shared object.
- **Keep them from creeping back with a CI linter.** numpy's
  `tools/ci/check_c_api_usage.py` flags every borrowed-ref call (`PyList_GetItem`,
  `PyDict_GetItem*`, `PyDict_Next`, `PyWeakref_GetObject`, …) unless annotated
  `// noqa: borrowed-ref OK` (genuinely safe) or `// noqa: borrowed-ref - manual fix
  needed` (tracked). A ready template — recommend it for any nontrivial C extension.
  `PyDict_Next` has no strong-ref replacement; a deliberate borrowed loop is the
  `// noqa: borrowed-ref OK` case.

### `PyMutex` (3.13+)

A **one-byte**, statically-initializable lock (`static PyMutex m;` or `(PyMutex){0}` —
unlocked == zero, no init call). `PyMutex_Lock`/`PyMutex_Unlock`/`PyMutex_IsLocked`. It
releases the thread from the runtime while blocked, so it can't wedge the GC. It is
**not reentrant** (unlike a critical section) and is **not in the limited API at any
version** — including `abi3t`.

```c
static PyMutex lock = {0};          // valid zero-init, no PyMutex_Init needed
PyMutex_Lock(&lock);
/* ... touch shared state ... */
PyMutex_Unlock(&lock);
```

For a **limited-API / older-Python fallback**, use numpy's portable macro pattern —
`PyMutex` where available, `PyThread_type_lock` otherwise:

```c
#ifdef NPY_USE_LEGACY_LOCK
static PyThread_type_lock the_mutex;
#define LOCK()   PyThread_acquire_lock(the_mutex, WAIT_LOCK)
#define UNLOCK() PyThread_release_lock(the_mutex)
#else
static PyMutex the_mutex = {0};
#define LOCK()   PyMutex_Lock(&the_mutex)
#define UNLOCK() PyMutex_Unlock(&the_mutex)
#endif
```

### Critical sections

`Py_BEGIN_CRITICAL_SECTION(op)` / `Py_END_CRITICAL_SECTION()` lock an object via its
built-in per-object lock. **The macros open and close a brace block**, so they must be
paired in the same lexical scope, and they are **true no-ops** without `Py_GIL_DISABLED`
(bare `{ }`), so extension code can use them unconditionally. Two rules the CPython
headers spell out that ports routinely get wrong:

1. **A critical section can be suspended across blocking calls / I/O** (that is how it
   avoids deadlock). Your lock is dropped and reacquired, so **invariants do not hold
   across a blocking call** inside the section — another thread may have mutated the
   object.
2. **Never nest critical sections to lock two objects** — the inner one suspends the
   outer. To hold two at once use `Py_BEGIN_CRITICAL_SECTION2(a, b)` /
   `Py_END_CRITICAL_SECTION2()`.

The idiomatic refactor (CPython `listobject.c`, `_collectionsmodule.c`) keeps the real
work in a `*_lock_held` helper and wraps the public entry point:

```c
int
PyList_Insert(PyObject *op, Py_ssize_t where, PyObject *newitem) {
    int err;
    Py_BEGIN_CRITICAL_SECTION(op);
    err = ins1((PyListObject *)op, where, newitem);   // does the work
    Py_END_CRITICAL_SECTION();
    return err;
}
```

For a lazily-filled cache, the winning pattern is **critical section + double-checked
fill** (numpy `convert_datatype.c`): take the section, re-check whether another thread
filled it while you blocked, then populate.

> Do not use `Py_REFCNT(op) == 1` as a thread-safe fast path — it is **not** reliable
> on free-threaded builds; use `PyUnstable_Object_IsUniquelyReferenced()`.

### Atomics and memory ordering

For a **single shared word** — a counter, a flag, a lazily-set pointer — an atomic is
cheaper than a lock and can't deadlock. Use **C11 `<stdatomic.h>` directly** (MSVC:
`/experimental:c11atomics`), `std::atomic` (C++), or `std::sync::atomic` (Rust).
CPython's own `_Py_atomic_*` / `FT_ATOMIC_*` are private/internal; don't depend on them.

The one thing to internalize before reaching for an atomic: **it makes a *single*
operation indivisible, not a sequence.** `if (!p) p = build();` is still a race even if
`p` is atomic — two threads both see `NULL` and both build. That is a critical section
or a double-checked fill (above), not an atomic. Atomics fit read-one / write-one /
fetch-and-add, nothing wider.

**Memory ordering** is the second parameter of every atomic op — it controls what the
compiler and CPU may reorder *around* it (the "memory barrier"). Pick the weakest that
is correct:

| Order | Use for | Cost |
| --- | --- | --- |
| `memory_order_relaxed` | a standalone counter/statistic where only the final tally matters — no other memory is published through it | cheapest; no barrier |
| `release` (store) / `acquire` (load) | a flag or pointer that **publishes other data**: the writer fills a struct then `release`-stores `ready=1`; a reader that `acquire`-loads `ready==1` is guaranteed to see the filled struct | one barrier each side |
| `seq_cst` (the default) | when you can't reason out an acquire/release pairing — correct but a full fence; don't reach for it as a habit | most expensive |

The acquire/release pair is what makes the double-checked cache fill correct on
weakly-ordered CPUs (ARM, POWER): without it a reader can see the "ready" flag before
the data it guards. x86 hides this — **test on ARM** (see
[`build-and-test.md`](build-and-test.md)). Two portability idioms from numpy/CPython:

- **Guard the cost on the default build** — only go atomic under `Py_GIL_DISABLED`
  (CPython `_asynciomodule.c`): `#ifdef Py_GIL_DISABLED ... _Py_atomic_add ... #else ++x`.
- **Cast the field at the access site** rather than declaring the struct member
  `_Atomic`, so the same struct still compiles in non-atomic contexts:
  `atomic_load_explicit((_Atomic(uint8_t)*)&cache->initialized, memory_order_acquire)`.

### Lock-free / CAS — report, don't write (🚩)

Beyond a single atomic word lies **lock-free programming**: compare-and-swap
(`atomic_compare_exchange_*`) retry loops, atomic stack/list pushes, seqlocks, hazard
pointers, RCU. It suits a genuinely hot, contended structure — but it is also where the
ABA problem, missing barriers, and reclamation bugs live, and those failures are
non-deterministic and near-impossible to reproduce. It is a **🚩 report finding, not a
fix to apply.**

If safety seems to *require* a CAS loop or a hand-rolled lock-free algorithm, stop and
report it (location, why a plain lock is a contention concern, that lock-free is a
deliberate design choice). Steer the user toward existing tools, not new `unsafe` code:
CPython already provides the common lock-free fast paths (optimistic `dict`/`list` reads
with a per-object-lock fallback) — reuse them via the container APIs. The first move is
almost always a `PyMutex` or critical section; escalate only with measured contention
and a reviewer who knows the memory model (C++ `std::atomic` with a known-correct
pattern; Rust via an audited crate like `crossbeam`/`arc-swap`).

## Global state, caches, and unsafe libraries (usually report, don't fix)

These are the 🚩 category — the candidate fixes to **recommend in your report**; apply
one yourself only if it turns out to be a small, self-contained change:

- **Convert global state to thread-local:** `Py_tss_t` + `PyThread_tss_*` (PEP 539,
  portable), C `thread_local`, `threading.local()` in Python, or a portable macro. Both
  numpy (`NPY_TLS`) and scipy (`SCIPY_TLS`) define the same shim —
  `thread_local` / `_Thread_local` / `__thread` / `__declspec(thread)` — for scratch
  buffers and callback slots. Converting one global scratch buffer to `NPY_TLS`
  (numpy `dragon4.c`) is a cheap ✅ fix; a codebase-wide ownership change is a 🚩
  maintainer decision.
- **Caches:** disable on the free-threaded build, populate **once** during module init
  (which runs under an import lock), use `std::call_once` / `std::once_flag`, or protect
  the container with a `PyMutex`. In pure Python, replace a module-level cache dict with
  `threading.local()` (scipy does this in `_morestats.py`, `_mannwhitneyu.py`,
  `fftpack`). Populating-once at init is often the simple case; reworking a hot mutable
  cache is not.
- **Non-reentrant native libraries** — scipy's playbook, best option first:
  1. **Rewrite the SAVE/COMMON state into a caller-owned state struct** so the library
     becomes reentrant (scipy rewrote ARPACK → `ARNAUD_state_s`, and VODE/ZVODE from
     f2py-Fortran to stateless C). Best parallelism; a 🚩 maintainer-scale effort.
  2. **Serialize with a lock** — a per-handle lock for a stateful object (scipy Qhull:
     a per-instance `PyThread_type_lock`), or one `static PyMutex` for a library with
     global state.
  3. **Detect and refuse concurrent use** with a clear error, rather than corrupt
     silently (scipy `ode` raises `IntegratorConcurrencyError`). A legitimate outcome.
  4. **Document the subsystem as not thread-safe** and quarantine its tests (scipy says
     exactly this for `scipy.sparse`; see [`build-and-test.md`](build-and-test.md) on
     the `thread_unsafe` marker). Also legitimate — declaring support does not require
     every corner to be safe, only that you're honest about which corners aren't.
- **A dependency that isn't free-thread-safe blocks you** — a global-lock wrapper is a
  stopgap; report the lost parallelism and the upstream tracking issue.

## Cython specifics

Cython 3.1+ takes `# cython: freethreading_compatible=True` and emits the `Py_mod_gil`
declaration for you, but the directive **declares support — it does not make code
safe** ("it merely confirms that you have checked the logic"). Two things it does *not*
protect, both able to **crash the interpreter** via inconsistent refcounts:

- **Module-level `cdef` globals** and **mutable attributes on `cdef` classes** — audit
  exactly as hand-written C. Cython doesn't even reach CPython's built-in "won't crash"
  baseline for these.
- **`prange` with Python-object variables** — they are *shared*, not thread-safely
  refcounted; `prange` + `with gil` is "extremely unsafe."

`with nogil:` (releasing the thread state) is **orthogonal** to free-threading —
existing `nogil` blocks do not make a module free-thread-safe. Fix with the tools
Cython provides: `cython.critical_section` (context manager/decorator, ≤2 objects),
`cython.pymutex` (= `PyMutex` on 3.13+), and `py_safe_call_once` (C++ output only) for
lazy cache init. Cython 3.3+ auto-wraps *generated* accessors/pickle/dataclass methods
in critical sections, but that is a no-crash guarantee, not an atomic snapshot, and
covers none of your hand-written logic. Convert shared `cdef` counters to C
`_Thread_local` + a `critical_section` merge. On **Windows**, define `Py_GIL_DISABLED=1`
manually or the extension crashes on import.

## PyO3 (Rust) specifics

Rust's type system prevents many races, but `Sync` static data still needs protection
without the GIL. Since 0.26 the sync toolbox lives in `pyo3::sync`:

- **`GILProtected` was deprecated in 0.26** → use `std::sync::Mutex`/atomics for pure-Rust state,
  or `pyo3::sync::PyMutex<T>` (wraps CPython's `PyMutex`, releases the runtime while
  blocked). `GILOnceCell` → **`PyOnceLock<T>`** for one-time init.
- When you hold a lock **across arbitrary Python calls**, use `MutexExt::lock_py_attached`
  / `RwLockExt` / `OnceExt::*_py_attached` — they detach from the runtime before
  blocking, avoiding GC deadlocks.
- **Critical sections from Rust:** `pyo3::sync::with_critical_section(&bound, || …)` is
  the direct analogue of `Py_BEGIN_CRITICAL_SECTION` (and `_2` for two objects); a
  no-op on the GIL build.
- Runtime **"Already borrowed" panics** get much more likely: two threads calling a
  `&mut self` `#[pymethods]` method concurrently raise `RuntimeError` — add explicit
  locking. `'py` no longer implies exclusive access.
- Gate FT-only code on `#[cfg(Py_GIL_DISABLED)]` (add
  `println!("cargo::rustc-check-cfg=cfg(Py_GIL_DISABLED)")` in `build.rs`).

## nanobind (C++) specifics

nanobind's own internals (type/instance registries) are already thread-safe (sharded
locks + thread-local caches + immortalized functions/types), so the work is entirely
your C++ state. Tools, all no-ops on the GIL build:

- **`nb::ft_mutex` + `nb::ft_lock_guard`** — a member `PyMutex` on your bound class.
- **`nb::ft_object_guard(h)` / `nb::ft_object2_guard`** — RAII single/two-object
  critical sections.
- **Argument locking** (`.lock()` / `nb::lock_self()`) retrofits safety at the binding
  boundary without touching the C++ type — but at most 2 args, and it protects *only*
  calls through the binding (direct C++ callers are unprotected). Locked calls cost
  more, so annotate selectively.
- **Do not remove `gil_scoped_acquire`/`release`** on FT builds — they still set thread
  context and `release` drops the thread's argument locks.
- **All-or-nothing:** loading *any* extension that hasn't declared support disables
  free-threading process-wide — in a multi-extension project, every one must be ported.
