"""The GIL rule as the Trainer applies it (D15): its failure is a refusal, and it comes after the refusals that change nothing.

Each case runs in a fresh process, since turning the GIL on, or keeping
Triton unavailable, cannot be undone.
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
    path = tmp_path / "case.py"
    path.write_text(textwrap.dedent(script))
    env = {name: value for name, value in os.environ.items() if name != "PYTHON_GIL"} | {"PYTHONPATH": str(ROOT)}
    done = subprocess.run([sys.executable, str(path)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-3000:]
    return done.stdout


def test_a_run_whose_process_has_the_gil_on_is_refused_naming_the_way_to_go_ahead(tmp_path: Path) -> None:
    out = fresh(
        tmp_path,
        """
        import triton  # turns the GIL on
        from library.runstate.owners import Refused
        from tests.unit.training_core import test_realizations as case

        made = case.trainer("")
        made.keep_gil_off = True
        try:
            made.realize()
        except Refused as refused:
            print(refused)
        """,
    )
    assert "the GIL rule was refused" in out
    assert "the GIL is on as the run starts" in out
    assert "a run goes ahead with the GIL on only by declaring the selection that needs it" in out


def test_a_run_refused_for_its_implementations_leaves_the_process_as_it_was(tmp_path: Path) -> None:
    out = fresh(
        tmp_path,
        """
        import sys
        from library.training.trainer import Trainer
        from tests.unit.training_core import test_realizations as case

        made = case.trainer("")
        made.keep_gil_off = True
        made.installed = {name: made.installed[name] for name in made.installed if name != "objective"}
        try:
            made.realize()
        except Exception as refused:
            print("refused", type(refused).__name__)
        print("triton blocked:", "triton" in sys.modules and sys.modules["triton"] is None)
        """,
    )
    assert out.splitlines() == ["refused Rejection", "triton blocked: False"], out


def test_a_gil_the_checks_before_the_rule_turn_on_is_refused_as_theirs(tmp_path: Path) -> None:
    """The rule applies after the implementation check and cold load; the GIL is read before them too, so the refusal says which."""
    out = fresh(
        tmp_path,
        """
        import sys
        from library.runstate.owners import Refused
        from library.training import trainer as module
        from tests.unit.training_core import test_realizations as case

        checked = module.check_implementations

        def importing(*args, **kwargs):  # a check that loads a module that turns the GIL on
            import triton  # noqa: F401
            return checked(*args, **kwargs)

        module.check_implementations = importing
        made = case.trainer("")
        made.keep_gil_off = True
        try:
            made.realize()
        except Refused as refused:
            print(refused)
        """,
    )
    assert "the GIL is on after checking the implementations and loading the run" in out, out


def test_the_checks_before_the_rule_load_nothing_that_could_turn_the_gil_on(tmp_path: Path) -> None:
    """What the ordering relies on: implementation checking and cold loading load no module outside the standard library, xDSL and the new system."""
    out = fresh(
        tmp_path,
        """
        import sys
        from library.iro import gil
        from library.iro.accepted import check_implementations, cold_load
        from tests.unit.training_core import test_realizations as case

        made = case.trainer('realize(@mixed_precision, "bfloat16")', {"mixed_precision": case.mixed_precision})
        before = set(sys.modules)
        check_implementations(made.run, made.installed)
        cold_load(made.run, made.composition, installed=made.installed)
        loaded = sorted(set(sys.modules) - before)
        outside = [m for m in loaded if m.split(".")[0] not in sys.stdlib_module_names | {"xdsl", "library"}]
        print("outside:", outside)
        print("gil:", gil.state())
        """,
    )
    assert out.splitlines() == ["outside: []", "gil: off"], out
