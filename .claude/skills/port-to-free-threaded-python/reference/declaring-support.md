# Declaring free-threading support

SKILL step 4. Declaring support tells the interpreter *not* to re-enable the GIL when
your module is imported. It is a **promise the runtime trusts** — make it only once the
module is actually safe (steps 3 and 5). It is cheap to add and, for a well-behaved
extension, the single most impactful change. The declaration compiles to a no-op on the
GIL build, so it is always safe to ship.

The macros/values below are all **new in CPython 3.13** (`Py_mod_gil`,
`Py_MOD_GIL_NOT_USED`, `PyUnstable_Module_SetGIL`), matched against the CPython 3.15
headers; target interpreter is **3.14t** (3.13t was experimental — see
[`background.md`](background.md)).

## C / C++ extensions

**Multi-phase init** (`PyModuleDef.m_slots` / `PyModExport`) — add a slot, guarded by
the *version* so it still compiles on 3.12 (which lacks the macro), not by
`Py_GIL_DISABLED` (which would compile the slot only into the `t` build). This is what
numpy and scipy do — the slot is harmless on the GIL build, so declare it everywhere:

```c
static PyModuleDef_Slot mymod_slots[] = {
    {Py_mod_exec, mymod_exec},
#if PY_VERSION_HEX >= 0x030d00f0    // Python 3.13+
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, NULL},
};
```

The default when the slot is absent is `Py_MOD_GIL_USED` — i.e. the GIL gets
re-enabled. (Real example: `numpy/_core/src/multiarray/multiarraymodule.c`,
`Modules/_queuemodule.c` in CPython.)

**Single-phase init** (legacy `PyInit_` returning a fully built module) — there is no
slot, so call `PyUnstable_Module_SetGIL`. This symbol exists **only under
`Py_GIL_DISABLED`**, so the guard here is mandatory (not just documentation):

```c
PyMODINIT_FUNC PyInit_mymod(void) {
    PyObject *m = PyModule_Create(&mymoddef);
    if (m == NULL) return NULL;
#ifdef Py_GIL_DISABLED
    PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED);
#endif
    return m;
}
```

`PyUnstable_*` is deliberately outside the limited API — fine for a version-specific
`cp3Xt` build; if you also target `abi3`/`abi3t`, move to multi-phase init and the slot.
(Real example: scipy's `_minpackmodule.c` uses both — the `#if Py_GIL_DISABLED`
`PyUnstable_Module_SetGIL` call *and* the version-guarded slot.)

## Bindings (the one-liner cases)

Every one of these **only declares support** — none makes your code thread-safe.
Cython's own docs put it bluntly: the directive "does not itself make the module safe
to run without the GIL; it merely confirms that you have checked the logic."

| Binding | How to declare | Version | Notes |
| --- | --- | --- | --- |
| **Cython** | `# cython: freethreading_compatible=True` (file), `-X freethreading_compatible=True` (CLI), or `compiler_directives={...}` (build) | **3.1+** | Cython emits the `Py_mod_gil` slot *and* `PyUnstable_Module_SetGIL` for you; 3.3+ auto-adds critical sections to *generated* accessors/pickle/dataclass methods only (a no-crash guarantee, not atomic snapshots). Windows needs `Py_GIL_DISABLED=1` defined manually. |
| **pybind11** | pass `py::mod_gil_not_used()` to `PYBIND11_MODULE(...)` | 2.13+ | — |
| **nanobind** | pass `FREE_THREADED` to `nanobind_add_module(...)` (CMake) | **2.2+** | The CMake flag sets `NB_FREE_THREADED` and emits the `Py_mod_gil` slot. Gate FT-only code on `NB_FREE_THREADED`, not `Py_GIL_DISABLED`. |
| **PyO3 (Rust)** | **0.28+:** free-threaded by default; write `#[pymodule(gil_used = true)]` only to *opt out*. **0.23–0.27:** must write `#[pymodule(gil_used = false)]` (or `PyModuleMethods::gil_used(false)`) | 0.23+ | Maps to `PyUnstable_Module_SetGIL`; no-op under abi3 and the GIL build. |

**PyO3 caveat:** the 0.26 release renamed `Python::with_gil`→`attach`,
`allow_threads`→`detach`, `GILOnceCell`→`PyOnceLock`, and **deprecated `GILProtected`**.
Examples using the old names predate 0.26. See [`thread-safety.md`](thread-safety.md)
for the replacement sync primitives.

## Advertise the status to users (metadata)

Add a trove classifier so installers and the ecosystem trackers can see where you are:

```toml
# pyproject.toml [project]
classifiers = [
    "Programming Language :: Python :: Free Threading :: 3 - Stable",
    # ... or :: 2 - Beta / :: 1 - Unstable while you stabilize
]
```

Use the level you have actually verified (step 8), not the one you aspire to.
`1 - Unstable` is the honest choice while races are still being found.

The classifier is **advisory only** — it is *not* what stops the GIL re-enabling (the
`Py_mod_gil` declaration is), and installers don't act on it. It's a signal to users
and the ecosystem trackers. Plenty of ported packages skip it: numpy and scipy ship
free-threaded wheels with no Free Threading classifier at all. Add it if you want to
advertise status; don't treat it as part of the mechanism.
