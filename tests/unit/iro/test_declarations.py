"""Implementation declarations: what follows from the written types, the digest a pin records, and the decorator."""

from contextlib import contextmanager
from dataclasses import dataclass, field, replace

from xdsl.dialects.builtin import SymbolRefAttr
from xdsl.ir import Block

import pytest

from library.iro.frontend import implementation
from library.iro.seams import Declaration, Effect
from library.iro.words.calls import CallOp, retained_positions
from library.iro.words.types import lease, participant, retained, value, written


class Done:
    def done(self) -> bool:
        return True

    def wait(self) -> None:
        return None


@dataclass
class Host:
    name: str = "host"
    entered: int = 0
    inside: bool = False
    completed_inside: list = field(default_factory=list)

    @contextmanager
    def context(self, call=None, after=(), reaches=()):
        self.entered += 1
        self.inside = True
        try:
            yield
        finally:
            self.inside = False

    def completion(self):
        self.completed_inside.append(self.inside)
        return Done()

    def capacity(self):
        return {}


DENOISE = Declaration(
    "denoise",
    "1",
    ("protected_current lease", "owner_update lease", "retained loss", "latents"),
    ("retained loss",),
    frozenset({Effect.DEVICE}),
)


def test_reads_updates_and_retained_inputs_follow_from_the_written_types() -> None:
    assert DENOISE.reads == (0,)
    assert DENOISE.updates == (1,)
    assert DENOISE.retains == (0, 1, 2), "a retained result keeps every lease and retained input, as a call's does"
    assert replace(DENOISE, outputs=("loss",)).retains == ()


@pytest.mark.parametrize(
    "change",
    [
        lambda d: replace(d, inputs=d.inputs[:-1]),
        lambda d: replace(d, outputs=("loss",)),
        lambda d: replace(d, effects=frozenset({Effect.DEVICE, Effect.RANDOMNESS})),
        lambda d: replace(d, concurrency_safe=True),
        lambda d: replace(d, completion="own"),
    ],
    ids=["inputs", "outputs", "effects", "concurrency", "completion"],
)
def test_any_change_to_the_declaration_changes_its_digest(change) -> None:
    assert change(DENOISE).digest != DENOISE.digest


def test_the_implementation_version_is_not_part_of_the_digest() -> None:
    assert replace(DENOISE, version="2").digest == DENOISE.digest


def test_a_declaration_names_itself_and_its_version_and_a_known_completion() -> None:
    with pytest.raises(ValueError, match="implementation version"):
        Declaration("denoise", "")
    with pytest.raises(ValueError, match="completion"):
        Declaration("denoise", "1", completion="later")
    with pytest.raises(ValueError):
        Declaration("denoise", "1", effects=frozenset({"teleport"}))  # ty: ignore[invalid-argument-type]


def test_the_decorator_records_the_declaration_without_registering_anything() -> None:
    @implementation(version=2, inputs=("int", "int"), outputs=("int",), effects=["device"])
    def add(a, b):
        return a + b

    assert add.declaration == Declaration("add", "2", ("int", "int"), ("int",), frozenset({Effect.DEVICE}))
    host = Host()
    with host.context():
        results, completion = add(host, 2, 3)
    assert results == (5,) and completion.done() and host.entered == 1
    assert host.completed_inside == [True], "the completion covers the work submitted in the device's context"
    with pytest.raises(TypeError, match="declares 2 inputs and received 1"):
        add(Host(), 2)


def test_retained_inputs_match_what_a_call_retains() -> None:
    types = [lease(participant(), "protected_current"), value("latents"), retained(value("loss")), lease(participant(), "owner_update")]
    args = Block(arg_types=types).args
    result = retained(value("loss"))
    call = CallOp.create(operands=list(args), result_types=[result], properties={"implementation": SymbolRefAttr("f")})
    declared = Declaration("f", "1", tuple(written(t) for t in types), ("retained loss",))
    assert declared.retains == retained_positions(call)


def test_results_come_back_in_the_declared_arity() -> None:
    @implementation(version=1, outputs=("int", "int"))
    def pair():
        return 1, 2

    @implementation(version=1)
    def nothing():
        return None

    @implementation(version=1, outputs=("int", "int"))
    def short():
        return 1

    assert pair(Host())[0] == (1, 2) and nothing(Host())[0] == ()
    with pytest.raises(TypeError, match="declares 2 results"):
        short(Host())


def test_an_implementation_with_its_own_completion_returns_it() -> None:
    @implementation(version=1, outputs=("int",), completion="own")
    def submitted():
        return 7, Done()

    @implementation(version=1, outputs=("int",), completion="own")
    def forgot():
        return 7

    assert submitted(Host())[0] == (7,)
    with pytest.raises(TypeError, match="own completion"):
        forgot(Host())
