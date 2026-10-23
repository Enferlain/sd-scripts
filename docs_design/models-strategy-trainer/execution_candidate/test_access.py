"""Readiness follows actual reservation lifetimes, not aggregate footprints."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from .examples import Scale, scale
from .graph import (
    Block,
    Call,
    Channel,
    Contract,
    Join,
    NotReady,
    Output,
    OwnedState,
    Producer,
    Release,
    Retain,
    Run,
    Sequence,
    Together,
    Use,
    View,
)
from .prepare import Engine, prepare
from .state import Product
from .test_specialized_lowering import LoweringKind
from .test_workers import run


def pair(*, modes=(True, True), release_first=False, isolated=False):
    seen = []

    def scope(name, other):
        def forward(context):
            context.views[f"{name}-view"](torch.tensor(1.0))

        compute = Call(f"compute-{name}", forward, uses=(Use(name, view=f"{name}-view"),), retained=f"old-{name}")
        read = Call(
            f"read-{other}", lambda _: seen.append(name), uses=(Use(other),), after=(f"release-{name}" if release_first else compute.name,)
        )
        release = Release(f"release-{name}", f"old-{name}", after=(compute.name if release_first else read.name,))
        return Sequence(
            f"sequence-{name}",
            (
                Block(
                    f"block-{name}",
                    (
                        Retain(f"hold-{name}", (name,), (), f"old-{name}", isolated=isolated),
                        compute,
                        *((release, read) if release_first else (read, release)),
                    ),
                ),
            ),
        )

    return (
        Run(
            Together("run", (scope("x", "y"), scope("y", "x"))),
            (scale("x", 1.0), scale("y", 1.0)),
            views=tuple(View(f"{name}-view", name, evaluation=mode) for name, mode in zip(("x", "y"), modes, strict=True)),
        ),
        seen,
    )


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("modes", ((False, False), (True, False), (False, True)))
def test_nonconflicting_reads_do_not_form_a_reservation_cycle(lowering: LoweringKind, modes):
    filing, seen = pair(modes=modes)
    ready = prepare(Contract().accept(filing), lowering=lowering, worker_calls=("compute-x", "compute-y"), worker_limit=2).publish()
    run(Engine(ready).run())
    assert sorted(seen) == ["x", "y"] and ready.state.completed
    assert not ready.state.readers and not ready.state.writers and not ready.state.active_retentions


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("isolated", (False, True))
def test_release_or_real_copy_breaks_the_cross_scope_cycle(lowering: LoweringKind, isolated):
    filing, seen = pair(release_first=not isolated, isolated=isolated)
    ready = prepare(Contract().accept(filing), lowering=lowering, worker_calls=("compute-x", "compute-y"), worker_limit=2).publish()
    run(Engine(ready).run())
    assert sorted(seen) == ["x", "y"] and ready.state.completed
    assert not ready.state.readers and not ready.state.writers and not ready.state.active_retentions


@pytest.mark.parametrize("reverse", (False, True))
def test_three_independent_scopes_have_a_diagnosed_cycle_before_loading(reverse):
    filing, _ = pair()
    assert isinstance(filing.root, Together)
    x, y = filing.root.children
    assert isinstance(x, Sequence) and isinstance(y, Sequence)
    z = replace(
        x,
        name="sequence-z",
        blocks=(
            Block(
                "block-z",
                (
                    Retain("hold-z", ("z",), (), "old-z"),
                    Call("compute-z", lambda _: None, uses=(Use("z", view="z-view"),), retained="old-z"),
                    Call("read-x", lambda _: None, uses=(Use("x"),), after=("compute-z",)),
                    Release("release-z", "old-z", after=("read-x",)),
                ),
            ),
        ),
    )
    y = replace(
        y,
        blocks=(
            replace(
                y.blocks[0],
                work=tuple(
                    replace(item, name="read-z", uses=(Use("z"),))
                    if item.name == "read-x"
                    else replace(item, after=("read-z",))
                    if isinstance(item, Release)
                    else item
                    for item in y.blocks[0].work
                ),
            ),
        ),
    )
    loads = []
    filing = replace(
        filing,
        root=Together("run", (z, y, x) if reverse else (x, y, z)),
        participants=tuple(replace(scale(name, 1.0), load=lambda: loads.append(True) or Scale(1.0)) for name in ("x", "y", "z")),
        views=(*filing.views, View("z-view", "z", evaluation=True)),
    )
    accepted = Contract().accept(filing)
    with pytest.raises(NotReady, match="reservation wait cycle") as caught:
        prepare(accepted, worker_calls=("compute-x", "compute-y", "compute-z"), worker_limit=3)
    assert all(f"sequence-{name}/" in str(caught.value) for name in ("x", "y", "z"))
    assert not loads


@pytest.mark.parametrize("sequential", (False, True))
def test_prefix_join_reservations_match_the_selected_execution_shape(sequential):
    async def source(context):
        async with context.use_source():
            value = Product("", 7)
        await context.publish("input", value)

    block = Block(
        "consumer",
        (
            Join("receive", ("input",), (Output("value", int),), lambda *_: True, uses=(Use("model", "write"),)),
            Call("compute", lambda _: None, after=("receive",)),
        ),
    )
    consumer = Sequence("sequence", (block,)) if sequential else block
    loads = []
    accepted = Contract().accept(
        Run(
            Together("run", (consumer, Producer("source", ("input",), source, uses=(Use("model"),)))),
            (replace(scale("model", 1.0), load=lambda: loads.append(True) or Scale(1.0)),),
            channels=(Channel("input", 1),),
        )
    )
    if sequential:
        # Sequence Join holds its own declared uses while waiting on input.
        with pytest.raises(NotReady, match="reservation wait cycle"):
            prepare(accepted, worker_calls=("compute",))
        assert not loads
    else:
        # Ordinary blocks wait for prefix input BEFORE the aggregate lease.
        ready = prepare(accepted, worker_calls=("compute",)).publish()
        run(Engine(ready).run())
        assert ready.state.completed and loads == [True]


def test_model_state_and_input_edges_participate_in_one_wait_analysis():
    async def source(context):
        async with context.use_source():
            value = Product("", 7)
        await context.publish("input", value)

    consumer = Block(
        "consumer",
        (
            Call("hold-journal", lambda _: None, states=("journal",)),
            Join("receive", ("input",), (Output("value", int),), lambda *_: True, after=("hold-journal",)),
        ),
    )
    sequence = Sequence(
        "sequence",
        (
            Block(
                "state-user",
                (
                    Retain("hold-model", ("model",), (), "old"),
                    Call("compute", lambda _: None, uses=(Use("model", view="eval"),), retained="old"),
                    Call("journal", lambda _: None, states=("journal",), after=("compute",)),
                    Release("release", "old", after=("journal",)),
                ),
            ),
        ),
    )
    reached = []
    accepted = Contract().accept(
        Run(
            Together("run", (consumer, sequence, Producer("source", ("input",), source, uses=(Use("model"),)))),
            (replace(scale("model", 1.0), load=lambda: reached.append("load") or Scale(1.0)),),
            states=(OwnedState("journal", lambda: reached.append("initialize")),),
            channels=(Channel("input", 1),),
            views=(View("eval", "model", evaluation=True),),
        )
    )
    with pytest.raises(NotReady, match="reservation wait cycle") as caught:
        prepare(accepted, worker_calls=("compute",))
    assert all(place in str(caught.value) for place in ("consumer/receive", "sequence/journal", "source"))
    assert not reached


@pytest.mark.parametrize("lowering", ("calls", "inline"))
@pytest.mark.parametrize("reverse", (False, True))
def test_different_phases_of_one_activity_cannot_be_simultaneous_waiters(lowering: LoweringKind, reverse):
    # Aggregate footprints suggest X -> Y -> Z -> W -> X. The two A phases
    # cannot coexist: B can read Z before A starts retaining it, releasing Y;
    # A then releases X, allowing C to release W before A's later request.
    seen = []

    def activity(name, pairs):
        blocks = []
        for held, requested in pairs:

            def forward(context, source=held):
                context.views[f"{source}-view"](torch.tensor(1.0))

            blocks.append(
                Block(
                    f"{name}-{held}",
                    (
                        Retain(f"hold-{held}", (held,), (), f"old-{held}"),
                        Call(f"compute-{held}", forward, uses=(Use(held, view=f"{held}-view"),), retained=f"old-{held}"),
                        Call(
                            f"read-{requested}",
                            lambda _, source=held: seen.append(source),
                            uses=(Use(requested),),
                            after=(f"compute-{held}",),
                        ),
                        Release(f"release-{held}", f"old-{held}", after=(f"read-{requested}",)),
                    ),
                )
            )
        return Sequence(name, tuple(blocks))

    children = (activity("A", (("x", "y"), ("z", "w"))), activity("B", (("y", "z"),)), activity("C", (("w", "x"),)))
    accepted = Contract().accept(
        Run(
            Together("run", tuple(reversed(children)) if reverse else children),
            tuple(scale(name, 1.0) for name in ("x", "y", "z", "w")),
            views=tuple(View(f"{name}-view", name, evaluation=True) for name in ("x", "y", "z", "w")),
        )
    )
    ready = prepare(
        accepted, lowering=lowering, worker_calls=("compute-x", "compute-y", "compute-z", "compute-w"), worker_limit=2
    ).publish()
    run(Engine(ready).run())
    assert sorted(seen) == ["w", "x", "y", "z"] and ready.state.completed
    assert not ready.state.readers and not ready.state.writers and not ready.state.active_retentions
