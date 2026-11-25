"""Evidence of what a piece of code is: its declaration and its source (D16)."""

import functools
import gc
import importlib.util
import subprocess
import sys
import weakref
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


class Scale:
    def __init__(self, factor: int) -> None:
        self.factor = factor

    def __call__(self, x: int) -> int:
        return x * self.factor

    def apply(self, x: int) -> int:
        return x * self.factor

    @classmethod
    def unit(cls, x: int) -> int:
        return x


def test_source_evidence_follows_where_the_code_is_defined() -> None:
    assert source_evidence(Callable()) == source_evidence(Callable), (
        "an object outside a selected implementation, such as a setup, is its class"
    )
    found = source_evidence(local)
    assert (found.module, found.qualname) == (__name__, "local")
    assert source_evidence(Callable).digest != found.digest, "two pieces defined in one module differ by their names"


def test_a_partial_is_its_function_with_the_values_it_binds() -> None:
    one, two = source_evidence(functools.partial(local, 1)), source_evidence(functools.partial(local, 2))
    assert one == source_evidence(functools.partial(local, 1)) and one != two, "bound values are part of the evidence"
    assert source_evidence(functools.partial(local, scale=2)) != source_evidence(functools.partial(local, scale=3))
    assert "bound (1)" in one.printout() and (one.module, one.qualname) == (__name__, "local") and not one.opaque
    assert source_evidence(functools.partial(local, ["a", "b"])).opaque == "binds a list, which can change while staying the same object"
    assert source_evidence(functools.partial(local, object())).opaque.startswith("binds an object")
    assert not source_evidence(functools.partial(local, ("a", ("b", 1)))).opaque, "tuples of fixed values are written out"


def test_a_wrapped_function_is_identified_through_every_layer() -> None:
    found = source_evidence(decorated)
    assert (found.module, found.qualname) == (__name__, "decorated")
    assert any(cover.startswith("wrapped by logged.<locals>.wrapper") for cover in found.covers)
    assert found != source_evidence(decorated.__wrapped__), "the wrapper is code the call runs, so it is covered too"


def make(scale: object):
    def work(x: object) -> object:
        return (x, scale)

    return work


def test_a_closure_is_its_function_with_the_values_it_captures() -> None:
    assert source_evidence(make(2)) != source_evidence(make(3)), "a captured setting is part of the evidence"
    assert source_evidence(make(2)) == source_evidence(make(2)), "two closures capturing the same value are the same code"
    assert "closes over (scale=2)" in source_evidence(make(2)).printout() and not source_evidence(make(2)).opaque
    assert source_evidence(make(local)) == source_evidence(make(local)), "captured code is included by its own evidence"
    mutable = source_evidence(make([]))
    assert mutable.opaque == "closes over scale, a list, which can change while staying the same object"
    assert "cannot be identified: it closes over scale" in mutable.printout()
    assert source_evidence(make(make([]))).opaque.startswith("closes over scale, code that closes over scale, a list"), (
        "what captured code captures counts too"
    )


def test_compiled_code_is_identified_by_its_release_the_same_in_every_process() -> None:
    assert source_evidence(len) != source_evidence(abs), "builtins differ by their names"
    assert source_evidence(len).compiled and not source_evidence(local).compiled
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


def test_state_bound_into_a_selected_implementation_must_be_represented() -> None:
    assert source_evidence(Scale(2), selected=True).opaque == "is a callable Scale object, whose state the system cannot see"
    assert source_evidence(Scale(2).apply, selected=True).opaque == "is a method bound to a Scale, whose state the system cannot see"
    assert source_evidence(functools.partial(Scale(2).apply, 1), selected=True).opaque.startswith("is a method bound to a Scale")
    assert not source_evidence(Scale.unit, selected=True).opaque, "a method bound to a class is the class's code"
    assert not source_evidence(local, selected=True).opaque and not source_evidence(make(2), selected=True).opaque
    assert source_evidence([].append, selected=True).opaque == "is a method bound to a list, whose state the system cannot see"


def test_a_captured_callable_object_or_bound_method_leaves_the_code_open() -> None:
    assert (
        source_evidence(make(Scale(2))).opaque
        == "closes over scale, code that is a callable Scale object, whose state the system cannot see"
    )
    assert source_evidence(make(Scale(2).apply)).opaque.startswith("closes over scale, code that is a method bound to a Scale")
    assert source_evidence(make([].append)).opaque.startswith("closes over scale, code that is a method bound to a list")
    assert not source_evidence(make(Scale)).opaque and not source_evidence(make(len)).opaque, "classes and plain functions are code"


def test_the_source_cache_keeps_no_closure_alive() -> None:
    class Payload:
        pass

    def make(payload: Payload):
        def run(x: object) -> object:
            return (payload, x)

        return run

    payload = Payload()
    held = weakref.ref(payload)
    first = source_evidence(make(payload))
    second = source_evidence(make(Payload()))
    assert first.module == second.module and first.qualname == second.qualname, "closures of one definition share their source"
    del payload
    gc.collect()
    assert held() is None, "the cache keeps neither a closure nor what it captures alive"


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
