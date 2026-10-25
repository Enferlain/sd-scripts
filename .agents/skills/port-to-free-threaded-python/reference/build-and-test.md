<!-- Locally adapted from tiran/agent-skills@0c51c24122b33afed31fd8d50534e089534f77a7; see ../README.md#repository-installation. -->

# Building and testing free-threaded wheels

SKILL steps 6–7. The wheel's ABI tag follows the interpreter you build against: build
with a free-threaded (`t`) interpreter and the backend emits a `cp3Xt` wheel
automatically. There is no per-backend "free-threading flag" — the work is (a) having
the support declaration from step 4 compiled in, and (b) getting a `t` interpreter into
the build matrix.

## Build backends

For all of these, build under a `t` interpreter (`uv python install 3.14t`); the tag
follows.

- **meson-python** — no special config; produces a `cp3Xt` wheel under a `t` build.
- **scikit-build-core** — same; CMake finds the `t` interpreter. (If you *also* target
  a single stable-ABI wheel, `wheel.py-api = "cp315.cp315t"` gives `abi3.abi3t` — see
  [`port-to-python-limited-api`](https://github.com/tiran/agent-skills/blob/0c51c24122b33afed31fd8d50534e089534f77a7/skills/port-to-python-limited-api/reference/build-systems.md).)
- **setuptools** — same; the `bdist_wheel` tag follows the interpreter. Ensure the
  `Py_GIL_DISABLED`-guarded `Py_mod_gil` / `PyUnstable_Module_SetGIL` declaration is in
  the C.
- **maturin (PyO3)** — set `gil_used = false` (declaration) and build with `-i
  python3.14t`. abi3 and free-threading are independent PyO3 features.

## CI: cibuildwheel

Free-threaded targets are **opt-in**:

```toml
# pyproject.toml
[tool.cibuildwheel]
enable = ["cpython-freethreading"]     # add the cp3Xt builds
```

or `CIBW_ENABLE=cpython-freethreading` in the environment. Without it, cibuildwheel
skips the `t` interpreters and you silently ship no free-threaded wheel. Build the same
platforms you already cover; add ARM if you can (see testing).

## Testing for races

A clean single-threaded suite proves nothing about thread-safety — exercise the code
from many threads.

- **`pytest-run-parallel`** — runs your existing tests concurrently from N threads:
  ```bash
  uv run --python 3.14t pytest --parallel-threads=auto --iterations=10
  # or, quarantining known-unsafe tests: --parallel-threads=4 --skip-thread-unsafe=true
  ```
  The cheapest first pass; flaky failures under it are real races. Mark tests that are
  legitimately single-threaded (spawn their own threads, mutate a documented-unsafe
  subsystem) with `@pytest.mark.thread_unsafe`, and cap memory-heavy ones with
  `@pytest.mark.parallel_threads_limit(n)` — both are how numpy and scipy manage it.
- **ThreadSanitizer (TSan)** — the authoritative data-race detector. Build the
  extension with `-fsanitize=thread` and run the suite; it reports races even when the
  test doesn't assert on the corrupted result. **Use a TSan-instrumented CPython** (and
  instrument native deps — numpy's CI TSan-builds OpenBLAS): against a stock
  free-threaded interpreter, uninstrumented lock internals produce a flood of false
  positives. Ship a suppressions file for benign/upstream races (numpy suppresses e.g.
  the Fortran LAPACK `dlamch_` lazy-init race) rather than blocking on them.
- **Run on ARM as well as x86.** x86 is strongly ordered and hides memory-ordering
  bugs that surface on weakly-ordered ARM. If you only test one, test ARM.
- **Separate normal-import qualification from controlled forced-off testing.**
  First verify ordinary imports leave `sys._is_gil_enabled()` false without
  `PYTHON_GIL=0` or `-X gil=0`. A native-extension race investigation may separately
  force the GIL off with appropriate dependency review/instrumentation so a missing
  declaration cannot mask the path under test. That override does not prove import
  compatibility or make an unprepared dependency safe.
- Add **targeted stress tests** for each shared structure you changed in step 5. Use a
  `threading.Barrier` so every worker hits the code simultaneously (numpy's
  `run_threaded`, scipy's `_run_concurrent_barrier`), then assert an invariant.

## A release gate worth copying

numpy and scipy both fail the wheel build if importing the package re-enables the GIL —
the one check that catches a missing/uncompiled declaration in the shipped artifact:

```bash
FT="$(python -c 'import sysconfig; print(bool(sysconfig.get_config_var("Py_GIL_DISABLED")))')"
if [[ $FT == "True" ]]; then
    if [[ $(python -c "import mypkg" 2>&1) == *"The global interpreter lock (GIL) has been enabled"* ]]; then
        echo "Error: importing mypkg re-enables the GIL on the free-threaded build"; exit 1
    fi
fi
```

## Confirming the result

```bash
uv run --python 3.14t python -c "import mymod; import sys; print(sys._is_gil_enabled())"
# -> False, with NO "GIL has been enabled to load module" warning
ls dist/                       # expect *-cp314t-*.whl (and/or *-abi3.abi3t-*.whl)
```

A printed GIL-re-enable warning means the declaration (step 4) is missing or not
compiled into the wheel you built.
