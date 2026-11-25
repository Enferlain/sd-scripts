"""Evidence of what a piece of code is: its declaration and its source (D16)."""

import functools
import importlib.util
import subprocess
import sys
from pathlib import Path

from library.iro.evidence import CodeEvidence, setup_digest, setup_evidence, source_evidence


def local(x: object, scale: object = 1) -> object:
    return x


class Callable:
    def __call__(self, *args: object) -> None:
        return None


class Setup:
    name = "test.setup"


def logged(function):
    @functools.wraps(function)
    def wrapper(*args: object) -> object:
        return function(*args)

    return wrapper


@logged
def decorated(x: object) -> object:
    return x


def test_source_evidence_follows_where_the_code_is_defined() -> None:
    assert source_evidence(Callable()) == source_evidence(Callable), "an object is identified by its class"
    found = source_evidence(local)
    assert (found.module, found.qualname) == (__name__, "local")
    assert source_evidence(Callable).digest != found.digest, "two pieces defined in one module differ by their names"


def test_a_partial_is_its_function_with_the_values_it_binds() -> None:
    one, two = source_evidence(functools.partial(local, 1)), source_evidence(functools.partial(local, 2))
    assert one == source_evidence(functools.partial(local, 1)) and one != two, "bound values are part of the evidence"
    assert source_evidence(functools.partial(local, scale=2)) != source_evidence(functools.partial(local, scale=3))
    assert "bound (1)" in one.printout() and (one.module, one.qualname) == (__name__, "local")
    opaque = source_evidence(functools.partial(local, object()))
    assert opaque != source_evidence(functools.partial(local, object())), "a value without process-independent text never matches"


def test_a_wrapped_function_is_identified_through_every_layer() -> None:
    found = source_evidence(decorated)
    assert (found.module, found.qualname) == (__name__, "decorated")
    assert any(cover.startswith("wrapped by logged.<locals>.wrapper") for cover in found.covers)
    assert found != source_evidence(decorated.__wrapped__), "the wrapper is code the call runs, so it is covered too"


def test_compiled_code_is_identified_by_its_release_the_same_in_every_process() -> None:
    assert source_evidence(len) != source_evidence(abs), "builtins differ by their names"
    assert any(cover.startswith("compiled, python ") for cover in source_evidence(len).covers)
    script = "from library.iro.evidence import source_evidence; print(source_evidence(len).digest)"
    other = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True).stdout.strip()
    assert other == source_evidence(len).digest, "another process gives the same evidence"


def test_an_edit_to_the_defining_module_changes_the_evidence_under_the_same_name(tmp_path: Path) -> None:
    def load(text: str):
        path = tmp_path / "piece.py"
        path.write_text(text)
        spec = importlib.util.spec_from_file_location("piece", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.work

    before = source_evidence(load("def work(x):\n    return x\n"))
    after = source_evidence(load("def work(x):\n    return x + 1\n"))
    assert (before.module, before.qualname) == (after.module, after.qualname) == ("piece", "work")
    assert before.digest != after.digest


def test_a_record_prints_in_full_and_reports_its_differences_part_by_part() -> None:
    record = CodeEvidence("work", "name work\ninputs int", source_evidence(local))
    printed = record.printout()
    assert printed.splitlines()[:3] == ["@work", "  declares name work", "  declares inputs int"]
    assert f"{__name__}.local source " in printed
    assert record.digest != CodeEvidence("work", "name work\ninputs int").digest, "the digest names the whole record"
    other = CodeEvidence("work", "name work\ninputs float", source_evidence(Callable))
    assert record.differences(other) == ("what it takes or returns", f"its source in {__name__}.Callable")
    assert record.differences(record) == ()
    assert CodeEvidence("load").differences(CodeEvidence("other")) == ("its name (@other)",)


def test_a_setup_is_recorded_by_its_name_and_its_class_s_source() -> None:
    assert setup_evidence(Setup()) == CodeEvidence("test.setup", source=source_evidence(Setup))
    assert setup_digest(Setup()) == source_evidence(Setup).digest
