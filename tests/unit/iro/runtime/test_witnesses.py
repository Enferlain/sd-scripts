"""Witnesses for a long run, for running without the GIL and for the coordinator's progress (14.2).

- Bounded bookkeeping (sd-scripts-syv.31.30): the coordinator's records and
  its peak traced memory stay flat as a run's iterations grow tenfold.
- Free-threaded: a run on the free-threaded interpreter keeps the GIL off and
  reports its actual state; PYTHON_GIL is never forced.
- Progress (sd-scripts-syv.31.23): while one worker spins in Python, the
  coordinator keeps admitting and completing another activity's calls, under
  whatever GIL state the interpreter actually has.
"""

import asyncio
import os
import subprocess
import sys
import sysconfig
import threading
import time
import tracemalloc
from pathlib import Path

import pytest

from library.iro.accepted import cold_load
from library.iro.devices import Host
from library.iro.frontend import implementation
from library.iro.prepare import Target, prepare
from library.iro.runtime import Coordinator
from tests.unit.iro.accepted_runs import accepted_run
from tests.unit.iro.runtime.harness import Memory

ROOT = Path(__file__).parents[4]

LONG = """iro.strategy @long uses(@make v1, @use v1, @see v1, @step v1, @check v1) {
  %m = iro.resource @m : participant
  %items = iro.stream @items : item bounds(outstanding 2, ready 1)
  iro.parallel @all capture(%items, %m) {
    iro.producer @source into %items {
    ^(%i: int):
      %item = iro.call @make(%i) -> item
      iro.yield %item
    }
    iro.repeat @loop TIMES times capture(%items, %m) {
    ^(%k: int):
      %claim = iro.take %items
      %value = iro.borrow %claim
      %r = iro.access %m protected_current {
      ^(%lease):
        iro.call @use(%lease, %value) -> retained int
      }
      %seen = iro.call @see(%r) -> int
      iro.access %m owner_update {
      ^(%lease):
        %w = iro.call @step(%lease) -> int
      }
      iro.dispose %claim consumed
      iro.request @report(%k) from @loop at settled_success
    }
    iro.handler @report sources [@loop] max 1 {
    ^(%x: int):
      %y = iro.call @check(%x) -> int
    }
    iro.stop @source after @loop
  }
}
"""


def _installed():
    def step(cell):
        cell.update(cell.value + 1)
        return 0

    return {
        "make": implementation(version=1, inputs=("int",), outputs=("item",), name="make")(lambda i: i),
        "use": implementation(version=1, inputs=("protected_current lease", "item"), outputs=("retained int",), name="use")(
            lambda lease, v: v
        ),
        "see": implementation(version=1, inputs=("retained int",), outputs=("int",), name="see")(lambda r: r),
        "step": implementation(version=1, inputs=("owner_update lease",), outputs=("int",), name="step")(step),
        "check": implementation(version=1, inputs=("int",), outputs=("int",), name="check")(lambda x: x),
    }


def run_long(times: int) -> tuple[dict[str, int], int]:
    installed = _installed()
    text = LONG.replace("TIMES", str(times))
    run = accepted_run(text, installed=installed)
    prepared = prepare(cold_load(run, installed=installed), installed, Target(workers=2))
    owner = Memory({"m": 0})
    largest: dict[str, int] = {}
    holder: list[Coordinator] = []

    def observe(fact) -> None:
        for key, value in sizes(holder[0]).items():
            largest[key] = max(largest.get(key, 0), value)

    coordinator = Coordinator(prepared, {"m": owner}, installed, {"host": Host()}, observe=observe)
    holder.append(coordinator)
    tracemalloc.start()
    outcome = asyncio.run(coordinator.run())
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert outcome.status == "succeeded", outcome.error
    assert owner.values["m"] == times
    left = {key: n for key, n in sizes(coordinator).items() if key in LIVE and n}
    assert not left, f"live state left after the run: {left}"
    return largest, peak


# Containers of live state, which must be empty after a run; the others are keyed by the program.
LIVE = frozenset(
    {"leases", "products", "outcomes", "request_of", "stops", "admissions", "_admission_of", "_productions"}
    | {"_started", "_live_in", "_failures", "outstanding", "ready", "pending", "_tasks"}
)


def sizes(coordinator: Coordinator) -> dict[str, int]:
    """The size of every container the coordinator holds, and of the streams' and handlers' queues."""
    found = {}
    for key, value in vars(coordinator).items():
        if isinstance(value, dict | list | set | frozenset) and key not in ("owners", "installed", "devices", "named", "groups"):
            found[key] = len(value)
    found["_ordinals"] = sum(len(v) for v in coordinator._ordinals.values())
    found["emissions"] = sum(len(v) for v in coordinator.emissions.values())
    found["outstanding"] = sum(len(s.outstanding) for s in coordinator.streams.values())
    found["ready"] = sum(len(s.ready) for s in coordinator.streams.values())
    found["pending"] = sum(len(h.pending) for h in coordinator.handlers.values())
    found["_tasks"] = sum(1 for t in coordinator._tasks if not t.done())
    return found


def flat(lengths: tuple[int, ...]) -> None:
    runs = {times: run_long(times) for times in lengths}
    largest = {times: sizes for times, (sizes, _) in runs.items()}
    assert len({tuple(sorted(v.items())) for v in largest.values()}) == 1, f"records grow with the run: {largest}"
    peaks = {times: peak for times, (_, peak) in runs.items()}
    assert peaks[lengths[-1]] - peaks[lengths[0]] < 256 * 1024, f"peak traced memory grows with the run: {peaks}"


def test_bookkeeping_and_peak_memory_stay_flat_as_a_run_grows() -> None:
    """A witness for sd-scripts-syv.31.30: largest records and peak traced memory as the run grows."""
    flat((60, 200))


@pytest.mark.slow
def test_bookkeeping_and_peak_memory_stay_flat_as_a_run_grows_tenfold() -> None:
    flat((60, 200, 600))


def test_the_runtime_keeps_the_gil_off_on_a_free_threaded_interpreter() -> None:
    """In a fresh process, so the test session's own imports cannot turn the GIL on; both witnesses run there."""
    environment = {k: v for k, v in os.environ.items() if k != "PYTHON_GIL"}
    code = (
        "import sys, asyncio\n"
        "from tests.unit.iro.runtime import test_witnesses as w\n"
        "w.run_long(20)\n"
        "w.test_the_coordinator_keeps_progressing_while_a_worker_spins_in_python()\n"
        "print(sys._is_gil_enabled())\n"
    )
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    enabled = done.stdout.strip().splitlines()[-1]
    free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    print(done.stdout.strip())
    print(f"free-threaded build: {free_threaded}; GIL enabled after a run: {enabled}")
    assert enabled == ("False" if free_threaded else "True")


PROGRESS = """iro.strategy @s uses(@spin v1, @tick v1) {
  iro.parallel @both {
    iro.sequence @long {
      %x = iro.call @spin() -> int
    }
    iro.repeat @short 20 times {
    ^(%i: int):
      %y = iro.call @tick(%i) -> int
    }
  }
}
"""


def test_the_coordinator_keeps_progressing_while_a_worker_spins_in_python() -> None:
    finished: dict[str, float] = {}
    ticks: list[float] = []

    all_ticked = threading.Event()

    def spin():
        deadline = time.perf_counter() + 30
        n = 0
        while not all_ticked.is_set() and time.perf_counter() < deadline:
            n += 1  # pure Python work: it holds the interpreter whenever the GIL is on
        finished["spin"] = time.perf_counter()
        return n

    def tick(i):
        ticks.append(time.perf_counter())
        if len(ticks) == 20:
            all_ticked.set()
        return i

    installed = {
        "spin": implementation(version=1, outputs=("int",), name="spin")(spin),
        "tick": implementation(version=1, inputs=("int",), outputs=("int",), name="tick")(tick),
    }
    run = accepted_run(PROGRESS, installed=installed)
    prepared = prepare(cold_load(run, installed=installed), installed, Target(workers=2))
    coordinator = Coordinator(prepared, {}, installed, {"host": Host()})
    outcome = asyncio.run(coordinator.run())
    assert outcome.status == "succeeded"
    before = sum(1 for t in ticks if t < finished["spin"])
    print(f"GIL enabled: {sys._is_gil_enabled()}; ticks completed while the worker spun: {before} of 20")
    assert all_ticked.is_set() and before == 20, "every short call completed while the long call still ran"
