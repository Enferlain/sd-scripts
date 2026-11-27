"""Keeping the GIL off for a run (D15): each case runs in a fresh process, since turning the GIL on cannot be undone.

The cases use Triton, the module known to turn the GIL on, which the
environment provides.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

pytestmark = pytest.mark.skipif(not hasattr(sys, "_is_gil_enabled"), reason="needs the free-threaded build")


def fresh(tmp_path: Path, script: str) -> str:
    """Runs the script in a fresh process and returns what it printed; fails the test if the process fails."""
    path = tmp_path / "case.py"
    path.write_text(textwrap.dedent(script))
    # Without PYTHON_GIL, the interpreter makes its own choice.
    env = {name: value for name, value in os.environ.items() if name != "PYTHON_GIL"} | {"PYTHONPATH": str(ROOT)}
    done = subprocess.run([sys.executable, str(path)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-3000:]
    return done.stdout


def test_a_run_keeps_known_causes_unavailable_and_the_gil_off(tmp_path: Path) -> None:
    out = fresh(
        tmp_path,
        """
        import importlib.util
        from library.iro import gil

        assert importlib.util.find_spec("triton") is not None, "the environment provides Triton"
        gil.keep_off()
        print("spec", importlib.util.find_spec("triton"))
        try:
            import triton
        except ImportError:
            print("unavailable")
        print("state", gil.state())
        """,
    )
    assert out.split() == ["spec", "None", "unavailable", "state", "off"]


def test_a_run_refuses_to_start_with_the_gil_on(tmp_path: Path) -> None:
    out = fresh(
        tmp_path,
        """
        import triton  # turns the GIL on
        from library.iro import gil

        try:
            gil.keep_off()
        except gil.GilTurnedOn as refused:
            print(refused)
        """,
    )
    assert "the GIL is on as the run starts" in out


def test_an_import_that_turns_the_gil_on_fails_and_so_does_the_next_check(tmp_path: Path) -> None:
    out = fresh(
        tmp_path,
        """
        import sys
        from library.iro import gil

        gil.keep_off()
        del sys.modules["triton"]  # code that gets around the block
        try:
            import triton
        except RuntimeWarning as failed:
            print("import failed naming triton:", "triton" in str(failed))
        try:
            gil.check("after the load")
        except gil.GilTurnedOn as found:
            print(found)
        """,
    )
    assert "import failed naming triton: True" in out
    assert "the GIL is on after the load" in out


RUN = """
import asyncio, sys, time
from library.iro import gil
from library.iro.accepted import cold_load
from library.iro.devices import Host
from library.iro.frontend import implementation
from library.iro.prepare import Target, prepare
from library.iro.runtime import Coordinator
from tests.unit.iro.accepted_runs import accepted_run

TEXT = '''iro.strategy @s uses(@load, @after) {
  iro.sequence @once {
    %x = iro.call @load() -> int
    %y = iro.call @after(%x) -> int
  }
}
'''


def load():
    sys.modules.pop("triton", None)  # code that gets around the block
    try:
        import triton  # noqa: F401
    except RuntimeWarning:
        pass  # code that swallows the import's error
    waited = 0.0
    while not COORDINATOR.kernel.failed and waited < WAIT:  # keeps working, while the watch has its turns
        time.sleep(0.01)
        waited += 0.01
    print("failed while the load worked:", COORDINATOR.kernel.failed)
    return 1


def after(x):
    return x + 1


installed = {
    "load": implementation(outputs=("int",), name="load")(load),
    "after": implementation(inputs=("int",), outputs=("int",), name="after")(after),
}
gil.keep_off()
run = accepted_run(TEXT, installed=installed)
prepared = prepare(cold_load(run, installed=installed), installed, Target())
COORDINATOR = Coordinator(prepared, {}, installed, {"host": Host()}, watch_gil=WATCH)
outcome = asyncio.run(COORDINATOR.run())
print(outcome.status)
print(outcome.error)
"""


def test_the_periodic_watch_finds_a_gil_turned_on_while_the_run_works(tmp_path: Path) -> None:
    """The load keeps working after turning the GIL on, and the watch fails the run meanwhile."""
    out = fresh(tmp_path, "WAIT = 5.0\nWATCH = 0.05\n" + RUN).splitlines()
    assert out[:2] == ["failed while the load worked: True", "failed"]
    assert out[2].count("the GIL turned on while the run worked") == 1


def test_the_check_as_the_run_settles_finds_a_gil_the_watch_had_no_turn_to_see(tmp_path: Path) -> None:
    """The watch reads only once an hour, so only the check as the run settles can find it."""
    out = fresh(tmp_path, "WAIT = 0.0\nWATCH = 3600\n" + RUN).splitlines()
    assert out[:2] == ["failed while the load worked: False", "failed"]
    assert "the GIL turned on while the run worked" in out[2]
