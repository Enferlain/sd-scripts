"""RNG ownership and job identity under deliberately changed worker timing.

These are selected test policies, not implicit engine RNG or scheduling rules.
"""

from __future__ import annotations

import asyncio
import random
import threading
from dataclasses import dataclass, field
from typing import Literal

import pytest
import torch

from .examples import scale
from .graph import (
    Advance,
    Block,
    Call,
    Channel,
    Choice,
    Contract,
    Differentiate,
    Join,
    NotReady,
    Output,
    OwnedState,
    Producer,
    Release,
    Repeat,
    Retain,
    Run,
    Sequence,
    Together,
    Unit,
    Use,
)
from .prepare import Context, Engine, ProviderContext, WorkFailed, prepare
from .state import Product
from .test_candidate import leaves
from .workers import Workers


SEEDS = (17, 29)
type LoweringKind = Literal["calls", "inline"]


class DrawOrder:
    """Force two already-entered workers to draw in either physical order."""

    def __init__(self, first: int):
        self.first = first
        self.entered = threading.Barrier(2, timeout=5)
        self.drawn = threading.Event()

    def before(self, key: int) -> None:
        self.entered.wait()
        if key != self.first:
            assert self.drawn.wait(5), "The prescribed first worker did not draw"

    def after(self, key: int) -> None:
        if key == self.first:
            self.drawn.set()


@dataclass
class RandomOwner:
    generator: torch.Generator
    python: random.Random
    values: list[torch.Tensor] = field(default_factory=list)


def generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


@pytest.fixture
def cpu_rng():
    """Only this fixture's joined work uses the global CPU generator."""
    previous = torch.random.get_rng_state().clone()
    python_previous = random.getstate()
    try:
        yield
    finally:
        torch.random.set_rng_state(previous)
        random.setstate(python_previous)


async def selected_trial(kind: str, first: int, lowering: LoweringKind):
    order = None if kind == "serialized-dropout" else DrawOrder(first)
    actual_order = []

    def sample(context: Context):
        key = int(context.choice.block)
        owner = context.states[str(key)]
        assert isinstance(owner, RandomOwner)
        if order is not None:
            order.before(key)
        if kind in {"python-global", "python-owned"}:
            stream = random if kind == "python-global" else owner.python
            value = torch.tensor([stream.random() for _ in range(32)])
        elif kind == "owned":
            value = torch.rand(32, generator=owner.generator)
        elif kind == "dropout":
            value = torch.nn.functional.dropout(torch.ones(32), p=0.5)
        elif kind == "serialized-dropout":
            # Every default-generator user in this arrangement declares the
            # same gate. The seed belongs to the job, not the acquired thread.
            with torch.random.fork_rng(devices=[]):
                torch.random.set_rng_state(owner.generator.get_state())
                value = torch.nn.functional.dropout(torch.ones(32), p=0.5)
                owner.generator.set_state(torch.random.get_rng_state())
        else:
            assert kind == "global"
            value = torch.rand(32)
        owner.values.append(value)
        actual_order.append(key)  # Ordered by the test gate or shared state lease.
        if order is not None:
            order.after(key)

    # The serial case reverses child submission rather than overlapping draws.
    # The actual-order assertion below verifies the resulting dispatch order.
    children = tuple(
        Block(str(key), (Call("sample", sample, states=(str(key), *(("default-rng",) if kind == "serialized-dropout" else ()))),))
        for key in (first, 1 - first)
    )
    filing = Run(
        Together("run", children),
        states=tuple(
            OwnedState(str(key), lambda key=key: RandomOwner(generator(SEEDS[key]), random.Random(SEEDS[key]))) for key in range(2)
        )
        + ((OwnedState("default-rng", lambda: None),) if kind == "serialized-dropout" else ()),
    )
    ready = prepare(Contract().accept(filing), lowering=lowering, worker_calls=("sample",), worker_limit=2).publish()
    torch.random.set_rng_state(generator(123).get_state())
    random.seed(123)
    async with asyncio.timeout(8):
        await Engine(ready).run()
    assert actual_order == [first, 1 - first]
    assert not ready.state.state_locks
    values = []
    for key in range(2):
        owner = ready.state.owners[str(key)]
        assert isinstance(owner, RandomOwner) and len(owner.values) == 1
        values.append(owner.values[0])
    return values, torch.random.get_rng_state(), random.getstate()


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("kind", ("global", "owned", "python-global", "python-owned", "dropout", "serialized-dropout"))
def test_selected_worker_rng_assignment_under_opposite_orders(lowering, kind, cpu_rng):
    async def scenario():
        forward, forward_state, python_forward = await selected_trial(kind, 0, lowering)
        reverse, reverse_state, python_reverse = await selected_trial(kind, 1, lowering)
        assert torch.equal(forward_state, reverse_state)
        assert python_forward == python_reverse
        if kind in {"global", "dropout", "python-global"}:
            assert all(torch.equal(forward[key], reverse[1 - key]) for key in range(2))
            assert not torch.equal(forward[0], reverse[0])
            serial = generator(123)
            if kind == "python-global":
                python_serial = random.Random(123)
                expected = tuple(torch.tensor([python_serial.random() for _ in range(32)]) for _ in range(2))
            elif kind == "global":
                expected = tuple(torch.rand(32, generator=serial) for _ in range(2))
            else:
                torch.random.set_rng_state(serial.get_state())
                expected = tuple(torch.nn.functional.dropout(torch.ones(32), p=0.5) for _ in range(2))
            assert all(torch.equal(value, wanted) for value, wanted in zip(forward, expected, strict=True))
        elif kind in {"owned", "python-owned"}:
            assert torch.equal(forward_state, generator(123).get_state())
            assert python_forward == random.Random(123).getstate()
            for key in range(2):
                assert torch.equal(forward[key], reverse[key])
                python_serial = random.Random(SEEDS[key])
                expected = (
                    torch.tensor([python_serial.random() for _ in range(32)])
                    if kind == "python-owned"
                    else torch.rand(32, generator=generator(SEEDS[key]))
                )
                assert torch.equal(forward[key], expected)
        else:
            assert torch.equal(forward_state, generator(123).get_state())
            for key in range(2):
                torch.random.set_rng_state(generator(SEEDS[key]).get_state())
                expected = torch.nn.functional.dropout(torch.ones(32), p=0.5)
                assert torch.equal(forward[key], reverse[key]) and torch.equal(forward[key], expected)

    asyncio.run(scenario())


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("other", ("global", "owned"))
def test_scoped_default_rng_can_overlap_owned_work_but_not_unscoped_default_use(lowering, other, cpu_rng):
    entered, unrelated_finished = threading.Event(), threading.Event()
    unrelated = []

    def isolated(context):
        with torch.random.fork_rng(devices=[]):
            torch.random.set_rng_state(generator(SEEDS[0]).get_state())
            entered.set()
            assert unrelated_finished.wait(5)
            torch.nn.functional.dropout(torch.ones(32), p=0.5)

    def unscoped(context):
        assert entered.wait(5)
        unrelated.append(torch.rand(32, generator=generator(SEEDS[1]) if other == "owned" else None))
        unrelated_finished.set()

    filing = Run(
        Together(
            "run",
            (
                Block("scoped", (Call("isolated", isolated, states=("declared-default-rng",)),)),
                Block("unscoped", (Call("unscoped", unscoped),)),
            ),
        ),
        states=(OwnedState("declared-default-rng", lambda: None),),
    )
    ready = prepare(Contract().accept(filing), lowering=lowering, worker_calls=("isolated", "unscoped"), worker_limit=2).publish()
    torch.random.set_rng_state(generator(123).get_state())

    async def scenario():
        async with asyncio.timeout(8):
            await Engine(ready).run()

    asyncio.run(scenario())
    assert len(unrelated) == 1
    if other == "global":
        assert torch.equal(unrelated[0], torch.rand(32, generator=generator(SEEDS[0])))
        assert not torch.equal(unrelated[0], torch.rand(32, generator=generator(123)))
        # The scoped restore also erased this operation's advancement.
    else:
        assert torch.equal(unrelated[0], torch.rand(32, generator=generator(SEEDS[1])))
    assert torch.equal(torch.random.get_rng_state(), generator(123).get_state())


@dataclass
class PipelineOwner:
    generator: torch.Generator = field(default_factory=lambda: generator(123))
    consumed: list[tuple[int, float]] = field(default_factory=list)


async def pipeline_trial(kind: str, first: int, workers: int, lowering: LoweringKind):
    order, lock = DrawOrder(first), threading.Lock()
    physical_order = []

    async def produce(context: ProviderContext):
        owner = context.states["production"]
        assert isinstance(owner, PipelineOwner)

        def draw(key: int) -> float:
            if workers > 1:
                order.before(key)
            with lock:
                stream = owner.generator if kind == "owner" else generator(SEEDS[key])
                value = float(torch.rand((), generator=stream))
                physical_order.append(key)
            if workers > 1:
                order.after(key)
            return value

        pool = Workers(workers)
        with pool.lifetime():
            async with asyncio.TaskGroup() as tasks:
                pending = tuple(tasks.create_task(pool.execute(lambda key=key: draw(key), threading.Event())) for key in range(2))
                for key, task in enumerate(pending):
                    result = await task
                    if result.cancellation is not None:
                        raise result.cancellation
                    # Selected publication order also respects a capacity-one
                    # Feed when a later job physically completes first.
                    await context.publish("input", Product(key, result.value))

    def objective(context: Context, value: float):
        owner = context.states["training"]
        assert isinstance(owner, PipelineOwner)
        key = context.choice.key
        assert isinstance(key, int)
        assert len(context.input_handoffs) == 1 and context.input_handoffs[0].key == key
        owner.consumed.append((key, value))
        return (context.models["model"](torch.tensor(value)) - (0.1, 0.7)[key]).square()

    block = Block(
        "learn",
        (
            Join("input", ("input",), (Output("value", float),), lambda *args: True),
            Call("objective", objective, ("value",), (Output("loss", torch.Tensor),), (Use("model"),), ("training",)),
            Differentiate("gradient", "loss", ("weights",), "gradient"),
            Advance("advance", "gradient", ("weights",)),
        ),
    )
    filing = Run(
        Together(
            "run",
            (
                Producer("production", ("input",), produce, states=("production",)),
                Repeat(
                    "learning",
                    (block,),
                    lambda counts, state: Choice("learn", counts.get("learn", 0)) if counts.get("learn", 0) < 2 else None,
                    "policy",
                ),
            ),
        ),
        (scale("model", 0.25),),
        (Unit("weights", (("model", "weight"),)),),
        (OwnedState("production", PipelineOwner), OwnedState("training", PipelineOwner), OwnedState("policy", lambda: None)),
        (Channel("input", 1),),
    )
    ready = prepare(Contract().accept(filing), lowering=lowering).publish()
    before = torch.random.get_rng_state().clone()
    async with asyncio.timeout(8):
        await Engine(ready).run()
    assert torch.equal(before, torch.random.get_rng_state())
    owner = ready.state.owners["training"]
    assert isinstance(owner, PipelineOwner)
    assert [key for key, _ in owner.consumed] == [0, 1]
    assert ready.state.feeds["input"].high_water == 1
    assert physical_order == ([0, 1] if workers == 1 else [first, 1 - first])
    assert ready.state.image is not None
    unit = ready.state.image.units["weights"]
    assert unit.steps == 2 and not unit.unfinished
    assert [record.input_handoffs[0].key for record in unit.outcomes] == [0, 1]
    weight = ready.state.image.bindings["model"].model.get_parameter("weight").detach().clone()
    expected = torch.tensor(0.25, requires_grad=True)
    for key, value in owner.consumed:
        (gradient,) = torch.autograd.grad((expected * torch.tensor(value) - (0.1, 0.7)[key]).square(), expected)
        with torch.no_grad():
            expected.add_(gradient, alpha=-0.05)
    assert torch.equal(weight, expected)
    return owner.consumed, weight


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("kind", ("owner", "job"))
def test_job_rng_survives_private_worker_reordering_and_worker_count(lowering, kind):
    async def scenario():
        serial = await pipeline_trial(kind, 0, 1, lowering)
        forward = await pipeline_trial(kind, 0, 2, lowering)
        reverse = await pipeline_trial(kind, 1, 2, lowering)
        assert serial[0] == forward[0] and torch.equal(serial[1], forward[1])
        if kind == "job":
            assert serial[0] == reverse[0] and torch.equal(serial[1], reverse[1])
            assert [value for _, value in serial[0]] == [float(torch.rand((), generator=generator(seed))) for seed in SEEDS]
        else:
            stream = generator(123)
            expected = [float(torch.rand((), generator=stream)) for _ in range(2)]
            assert [value for _, value in forward[0]] == expected
            assert [value for _, value in reverse[0]] == list(reversed(expected))
            assert not torch.equal(forward[1], reverse[1])

    asyncio.run(scenario())


async def source_trial(policy: str, timing: str, lowering: LoweringKind, *, require_initial: bool = False):
    source_done, updated, update_requested, entered = (asyncio.Event() for _ in range(4))
    release, loop = threading.Event(), asyncio.get_running_loop()
    observed = []

    async def produce(context: ProviderContext):
        if policy == "latest":
            await context.publish("permit", Product("", True))
            if timing == "update-first":
                await updated.wait()
        async with context.use_source() as source:
            if policy == "boundary":
                # Release the updater only after this accepted use pins the
                # initial view. Its conflicting advancement must then wait.
                await context.publish("permit", Product("", True))

            def encode():
                loop.call_soon_threadsafe(entered.set)
                if policy == "boundary":
                    assert release.wait(5), "The source lease was not inspected"
                with torch.no_grad():
                    return float(source.models["encoder"](torch.rand((), generator=generator(SEEDS[0]))))

            pool = Workers(1)
            with pool.lifetime():
                result = await pool.execute(encode, threading.Event())
                if result.cancellation is not None:
                    raise result.cancellation
            product = Product("caption-42", result.value, (source.stamps["encoder"],))
        source_done.set()
        await context.publish("text", product)

    async def arrive(context):
        update_requested.set()
        if timing == "source-first":
            await source_done.wait()

    def objective(context):
        return context.models["encoder"](torch.tensor(1.0)).square()

    def completed(context):
        updated.set()

    def admissible(product, projection):
        return len(product.provenance) == 1 and (
            product.provenance[0].numerical_epoch == 0 if require_initial or policy == "boundary" else True
        )

    def observe(context: Context, value: float):
        (fact,) = context.input_handoffs
        assert fact.key == context.choice.key == "caption-42"
        observed.append((fact.key, fact.provenance[0].numerical_epoch, value))

    update = Sequence(
        "update",
        (
            Block(
                "arrival",
                (Join("permit", ("permit",), (Output("allowed", bool),), lambda *args: True), Call("arrive", arrive, after=("permit",))),
            ),
            Block(
                "change",
                (
                    Retain("hold", ("encoder",), ("encoder-weights",), "source"),
                    Call("objective", objective, outputs=(Output("loss", torch.Tensor),), uses=(Use("encoder"),), retained="source"),
                    Differentiate("gradient", "loss", ("encoder-weights",), "gradient", retained="source"),
                    Release("release", "source", after=("gradient",)),
                    Advance("advance", "gradient", ("encoder-weights",), after=("release",)),
                    Call("completed", completed, after=("advance",)),
                ),
            ),
        ),
    )
    consume = Block(
        "consume",
        (Join("text", ("text",), (Output("encoded", float),), admissible), Call("observe", observe, ("encoded",))),
    )
    filing = Run(
        Together(
            "run",
            (
                Producer("production", ("text", "permit"), produce, (Use("encoder"),)),
                update,
                Repeat("consumer", (consume,), lambda counts, state: None if counts else Choice("consume", "caption-42"), "policy"),
            ),
        ),
        (scale("encoder", 1.0),),
        (Unit("encoder-weights", (("encoder", "weight"),)),),
        (OwnedState("policy", lambda: None),),
        (Channel("text", 1), Channel("permit", 1)),
    )
    ready = prepare(Contract().accept(filing), lowering=lowering).publish()
    before = torch.random.get_rng_state().clone()
    assert ready.state.image is not None
    unit = ready.state.image.units["encoder-weights"]
    task = asyncio.create_task(Engine(ready).run())
    try:
        async with asyncio.timeout(8):
            if policy == "boundary":
                await entered.wait()
                await update_requested.wait()
                assert ready.state.readers.get("encoder") == 1 and unit.steps == 0
                release.set()
            if require_initial:
                with pytest.raises(ExceptionGroup) as caught:
                    await task
                causes = [error.cause if isinstance(error, WorkFailed) else error for error in leaves(caught.value)]
                assert len(causes) == 1 and isinstance(causes[0], NotReady) and "Inadmissible" in str(causes[0])
                assert not observed and "caption-42" in ready.state.feeds["text"].ready
                assert not ready.state.feeds["text"].handed_off
            else:
                await task
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert unit.steps == 1 and ready.state.stamp("encoder").numerical_epoch == 1
    assert torch.equal(before, torch.random.get_rng_state())
    assert not ready.state.readers and not ready.state.writers
    return observed


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("policy", ("latest", "boundary"))
def test_rng_and_job_identity_need_a_source_version_policy_too(lowering, policy):
    async def scenario():
        early = await source_trial(policy, "source-first", lowering)
        late = await source_trial(policy, "update-first", lowering)
        noise = torch.rand((), generator=generator(SEEDS[0]))
        assert early == [("caption-42", 0, float(noise))]
        if policy == "boundary":
            assert late == early
        else:
            weight = torch.tensor(1.0)
            weight.add_(torch.tensor(2.0), alpha=-0.05)
            assert late == [("caption-42", 1, float(weight * noise))] and late != early

    asyncio.run(scenario())


@pytest.mark.parametrize("lowering", ("calls", "inline"))
def test_source_provenance_can_enforce_the_required_version_before_handoff(lowering):
    asyncio.run(source_trial("latest", "update-first", lowering, require_initial=True))
