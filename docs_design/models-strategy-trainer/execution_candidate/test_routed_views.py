"""Frozen views, destination-specific derivatives and cross-block last use.

These are CPU candidate conformance probes, not production support claims.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
import torch
from torch import nn

from .examples import Scale, execute, routed_views, two_pass
from .graph import (
    Advance,
    Block,
    Call,
    Contract,
    Differentiate,
    NotReady,
    Output,
    Participant,
    Rejected,
    Release,
    Retain,
    Run,
    Sequence,
    Unit,
    Use,
    View,
)
from .prepare import Engine, WorkFailed, prepare
from .state import Contribution, PreparedView, Replacement
from .test_candidate import leaves, run


def sequence(filing: Run) -> Sequence:
    assert isinstance(filing.root, Sequence)
    return filing.root


def add_work(filing: Run, index: int, work: Call) -> Run:
    root = sequence(filing)
    blocks = list(root.blocks)
    blocks[index] = replace(blocks[index], work=(*blocks[index].work, work))
    return replace(filing, root=replace(root, blocks=tuple(blocks)))


@pytest.mark.parametrize("isolated", (False, True))
@pytest.mark.parametrize("recompute", (False, True))
def test_routed_gradients_match_direct_oracle_with_independent_unit_progress(isolated, recompute, monkeypatch):
    # Independent direct Python oracle. No custom derivatives or engine formulas.
    a, b = nn.Parameter(torch.tensor(2.0)), nn.Parameter(torch.tensor(3.0))
    frozen = Scale(1.0).requires_grad_(False)
    h = frozen(a)
    expected_a = torch.autograd.grad(h * b, (a,), retain_graph=True)[0]
    expected_b = torch.autograd.grad(h.square() * b, (b,))[0]
    naive_h = frozen(a)
    wrong = torch.autograd.grad(naive_h * b + naive_h.square() * b, (a, b))
    assert tuple(float(value) for value in wrong) == (15.0, 6.0)
    assert (float(expected_a), float(expected_b)) == (3.0, 4.0)

    filing = routed_views(isolated=isolated, recompute=recompute)
    observations = []

    def before_second(context):
        assert state.image is not None
        observations.append((state.image.units["A"].steps, state.image.units["B"].steps, state.image.units["A"].pending))

    filing = add_work(filing, 1, Call("inspect-independent-progress", before_second, after=("second-objective",)))
    root = sequence(filing)
    second = root.blocks[1]
    second = replace(
        second,
        work=tuple(
            replace(work, after=("inspect-independent-progress",)) if isinstance(work, Differentiate) else work for work in second.work
        ),
    )
    filing = replace(filing, root=replace(root, blocks=(root.blocks[0], second, root.blocks[2])))
    ready = prepare(Contract().accept(filing)).publish()
    state = ready.state
    assert state.image is not None
    gradients = {}
    original = state.contribute

    def record(*args, **kwargs):
        contribution = original(*args, **kwargs)
        assert state.image is not None
        for name in contribution.units:
            saved = []
            for parameter in state.image.units[name].parameters:
                assert parameter.grad is not None
                saved.append(parameter.grad.detach().clone())
            gradients[name] = tuple(saved)
        return contribution

    monkeypatch.setattr(state, "contribute", record)
    run(Engine(ready).run())
    assert torch.equal(gradients["A"][0], expected_a)
    assert torch.equal(gradients["B"][0], expected_b)
    assert observations == ([(1, 0, None)] if isolated else [(0, 0, 1)])
    assert float(state.image.bindings["a"].model.get_parameter("weight").detach()) == pytest.approx(1.7)
    assert float(state.image.bindings["b"].model.get_parameter("weight").detach()) == pytest.approx(2.6)
    assert state.image.bindings["frozen"].model.get_parameter("weight").grad is None
    assert not state.image.bindings["frozen"].model.get_parameter("weight").requires_grad
    assert state.generation == 1 and not state.active_retentions and not state.readers and not state.writers
    a_outcome, b_outcome = (state.image.units[name].outcomes[0] for name in ("A", "B"))
    assert a_outcome.region is b_outcome.region
    assert (a_outcome.invocation, b_outcome.invocation, b_outcome.advancement_invocation) == (1, 2, 3)
    assert a_outcome.advancement_invocation == (1 if isolated else 2)
    assert a_outcome.source_stamps == b_outcome.source_stamps
    assert all(stamp.numerical_epoch == 0 for stamp in b_outcome.source_stamps)
    assert state.coordinates["routed-work"] == {"first-loss": 1, "second-loss": 1, "finish-b": 1}
    state.capture()  # Only the finished quiescent cut is supported.


def test_views_are_distinct_prepared_calls_of_one_participant_and_refresh_on_replacement():
    ready = run(execute(routed_views()))
    state = ready.state
    assert state.image is not None
    target, conduit = (state.image.views[name] for name in ("target", "conduit"))
    assert target is not conduit and target.model is conduit.model is state.image.bindings["frozen"].model
    assert len(state.accepted.participant_refs) == 3 and len(state.accepted.unit_refs) == 2
    original_ref = state.accepted.participant_refs["frozen"]
    run(state.replace(Replacement("frozen", 1, Scale(2.0)), "frozen"))
    assert state.accepted.participant_refs["frozen"] == original_ref
    assert state.stamp("frozen").binding == 2
    assert state.image.views["target"].model is state.image.views["conduit"].model
    assert state.image.views["target"] is not target
    assert state.image.views["conduit"].model is not conduit.model


def test_prepared_views_restore_mixed_modes_on_return_and_failure():
    class Modes(Scale):
        def __init__(self):
            super().__init__(1.0)
            self.child = nn.Identity()
            self.child.eval()
            self.seen = []
            self.fail = False

        def forward(self, value):
            self.seen.append((self.training, self.child.training, torch.is_grad_enabled()))
            if self.fail:
                raise RuntimeError("view failed")
            return self.child(super().forward(value))

    filing = routed_views()
    filing = replace(filing, participants=(Participant("frozen", Modes, lambda model: None), *filing.participants[1:]))
    ready = prepare(Contract().accept(filing)).publish()
    assert ready.state.image is not None
    model = ready.state.image.bindings["frozen"].model
    assert isinstance(model, Modes)
    target, conduit = (ready.state.image.views[name] for name in ("target", "conduit"))
    value = torch.tensor(2.0, requires_grad=True)
    assert not target(value).requires_grad and conduit(value).requires_grad
    assert model.seen == [(False, False, False), (True, False, True)]
    assert model.training and not model.child.training
    model.fail = True
    with pytest.raises(RuntimeError, match="view failed"):
        target(value)
    assert model.training and not model.child.training


def test_direct_update_cannot_bypass_live_old_state_last_use(monkeypatch):
    async def check():
        entered, finish = asyncio.Event(), asyncio.Event()
        contribution = None

        async def pause(context):
            entered.set()
            await finish.wait()

        filing = add_work(routed_views(recompute=True), 0, Call("pause", pause, after=("a-gradient",)))
        ready = prepare(Contract().accept(filing)).publish()
        state = ready.state
        assert state.image is not None
        original = state.contribute

        def record(*args, **kwargs):
            nonlocal contribution
            result = original(*args, **kwargs)
            if result.units == ("A",):
                contribution = result
            return result

        monkeypatch.setattr(state, "contribute", record)
        task = asyncio.create_task(Engine(ready).run())
        await entered.wait()
        assert isinstance(contribution, Contribution)
        assert state.readers == {"frozen": 1, "a": 1, "b": 1}
        with pytest.raises(NotReady, match="old-state last use"):
            state.advance(contribution, ("A",), 1, region=contribution.region)
        with pytest.raises(NotReady, match="unverified gradient handback"):
            state.advance(contribution, ("A",), 1, region=object())
        assert state.image.units["A"].steps == 0 and state.image.units["A"].pending == 1
        finish.set()
        await task
        assert not state.readers and not state.active_retentions

    run(check())


def test_live_last_use_blocks_other_writer_until_recomputation_finishes():
    async def check():
        entered, finish, acquired = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def pause(context):
            entered.set()
            await finish.wait()

        filing = add_work(routed_views(recompute=True), 0, Call("pause", pause, after=("a-gradient",)))
        ready = prepare(Contract().accept(filing)).publish()
        task = asyncio.create_task(Engine(ready).run())
        await entered.wait()

        async def writer():
            async with ready.state.acquire((Use("a", "write"),)):
                acquired.set()

        writer_task = asyncio.create_task(writer())
        await asyncio.sleep(0)
        assert not acquired.is_set()
        finish.set()
        await task
        await writer_task
        assert acquired.is_set()

    run(check())


def test_isolated_recomputation_uses_real_old_storage_and_origin_stamps():
    seen = []
    filing = routed_views(isolated=True, recompute=True)
    root = sequence(filing)
    second = root.blocks[1]
    objective = second.work[0]
    assert isinstance(objective, Call)

    def inspect(context, *args):
        assert ready.state.image is not None
        live = ready.state.image.bindings["a"].model.get_parameter("weight")
        old = context.models["a"].get_parameter("weight")
        seen.append(
            (
                float(live.detach()),
                float(old.detach()),
                context.stamps["a"],
                old is not live,
                old.untyped_storage().data_ptr() != live.untyped_storage().data_ptr(),
            )
        )
        return objective.implementation(context, *args)

    second = replace(second, work=(replace(objective, implementation=inspect), *second.work[1:]))
    ready = prepare(Contract().accept(replace(filing, root=replace(root, blocks=(root.blocks[0], second, root.blocks[2]))))).publish()
    run(Engine(ready).run())
    current, old, origin, separate_object, separate_storage = seen[0]
    assert current == pytest.approx(1.7) and old == 2.0
    assert separate_object and separate_storage
    assert origin.participant_ref == ready.state.stamp("a").participant_ref
    assert origin.binding == ready.state.stamp("a").binding == 1
    assert origin.numerical_epoch == 0 and ready.state.stamp("a").numerical_epoch == 1


def test_pending_isolated_work_rejects_replacement_without_resetting_returned_unit():
    async def check():
        entered, finish = asyncio.Event(), asyncio.Event()

        async def pause(context):
            entered.set()
            await finish.wait()

        filing = add_work(routed_views(isolated=True), 0, Call("pause", pause, after=("advance-a",)))
        ready = prepare(Contract().accept(filing)).publish()
        state = ready.state
        assert state.image is not None
        old_image = state.image
        task = asyncio.create_task(Engine(ready).run())
        await entered.wait()
        assert not state.readers and state.image.units["A"].steps == 1
        with pytest.raises(NotReady, match="unfinished retained derivative work"):
            await state.replace(Replacement("frozen", 1, Scale(2.0)), "frozen")
        assert state.image is old_image and state.image.units["A"].steps == 1
        finish.set()
        await task
        assert state.image.units["B"].steps == 1

    run(check())


def test_released_old_state_does_not_make_pending_gradient_replacement_safe():
    async def check():
        entered, finish = asyncio.Event(), asyncio.Event()

        async def pause(context):
            entered.set()
            await finish.wait()

        filing = add_work(routed_views(), 1, Call("pause", pause, after=("advance-a",)))
        ready = prepare(Contract().accept(filing)).publish()
        state = ready.state
        assert state.image is not None
        task = asyncio.create_task(Engine(ready).run())
        await entered.wait()
        assert not state.active_retentions and not state.readers
        assert state.image.units["A"].steps == 1 and state.image.units["B"].pending == 2
        with pytest.raises(NotReady, match="unfinished gradients"):
            await state.replace(Replacement("b", 1, Scale(4.0)), "b")
        assert state.stamp("b").binding == 1
        finish.set()
        await task

    run(check())


@pytest.mark.parametrize("isolated", (False, True))
def test_late_unit_failure_preserves_cross_block_returned_uncertain_and_unentered_outcomes(isolated, monkeypatch):
    filing = routed_views(isolated=isolated)
    visited = []
    filing = add_work(filing, 2, Call("later-unentered", lambda context: visited.append(True), after=("advance-b",)))
    ready = prepare(Contract().accept(filing)).publish()
    state = ready.state
    assert state.image is not None
    optimizer = state.image.units["B"].optimizer
    original = optimizer.step

    def changed_then_failed():
        original()
        raise RuntimeError("B completion unknown")

    monkeypatch.setattr(optimizer, "step", changed_then_failed)
    with pytest.raises(WorkFailed, match="B completion unknown") as failure:
        run(Engine(ready).run())
    outcomes = {record.unit_ref: record for record in failure.value.unit_outcomes}
    a, b = (outcomes[state.accepted.unit_refs[name]] for name in ("A", "B"))
    assert (a.optimizer, a.reset, b.optimizer, b.reset) == ("returned", "returned", "uncertain", "not_attempted")
    assert (a.invocation, b.invocation, failure.value.invocation) == (1, 2, 3)
    assert a.region is b.region and a.source_stamps == b.source_stamps
    assert state.image.units["A"].steps == 1 and state.image.units["B"].steps == 0
    assert float(state.image.bindings["b"].model.get_parameter("weight").detach()) == pytest.approx(2.6)
    assert state.image.units["B"].pending == 2 and not visited
    assert not state.active_retentions and not state.readers and not state.writers
    assert state.unusable == {"frozen", "a", "b"}
    with pytest.raises(NotReady, match="usable"):
        state.capture()


def test_failure_or_cancellation_releases_lifetime_without_discarding_pending_gradient():
    async def check():
        entered = asyncio.Event()

        async def pause(context):
            entered.set()
            await asyncio.Event().wait()

        filing = add_work(routed_views(), 0, Call("pause", pause, after=("a-gradient",)))
        ready = prepare(Contract().accept(filing)).publish()
        assert ready.state.image is not None
        task = asyncio.create_task(Engine(ready).run())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not ready.state.readers and not ready.state.active_retentions
        assert ready.state.image.units["A"].pending == 1
        assert ready.state.image.units["A"].steps == 0
        assert ready.state.unusable == {"frozen", "a", "b"}
        with pytest.raises(NotReady):
            ready.state.capture()

    run(check())


def test_copy_that_only_relabels_current_storage_is_rejected_without_leaking_lease():
    class Borrowed(Scale):
        def __deepcopy__(self, memo):
            return self

    filing = routed_views(isolated=True)
    filing = replace(filing, participants=(Participant("frozen", lambda: Borrowed(1.0), lambda model: None), *filing.participants[1:]))
    ready = prepare(Contract().accept(filing)).publish()
    assert ready.state.image is not None
    with pytest.raises(WorkFailed, match="borrow current objects/storage"):
        run(Engine(ready).run())
    assert not ready.state.readers and not ready.state.active_retentions
    assert ready.state.image.units["A"].steps == ready.state.image.units["B"].steps == 0


def test_known_early_update_and_premature_release_are_rejected_before_loading():
    filing = routed_views()
    root = sequence(filing)
    first, second, third = root.blocks
    first = replace(first, work=(*first.work, Advance("too-early", "ga", ("A",))))
    with pytest.raises(Rejected, match="old-state last use"):
        Contract().accept(replace(filing, root=replace(root, blocks=(first, second, third))))
    first = replace(root.blocks[0], work=(*root.blocks[0].work, Release("too-early", "old", after=("a-gradient",))))
    with pytest.raises(Rejected, match="released"):
        Contract().accept(replace(filing, root=replace(root, blocks=(first, second, third))))


def test_parallel_block_local_values_do_not_silently_become_cross_block_links():
    from .graph import Together

    filing = routed_views()
    root = sequence(filing)
    with pytest.raises(Rejected):
        Contract().accept(replace(filing, root=Together("parallel", root.blocks)))


def test_views_require_genuine_participant_correspondence_and_calls_do_not_receive_coordination_handles():
    filing = routed_views()
    with pytest.raises(Rejected, match="execution view"):
        Contract().accept(replace(filing, views=(View("not-a-participant", "missing"),)))
    root = sequence(filing)
    first = root.blocks[0]
    call = first.work[1]
    assert isinstance(call, Call)
    bad_use = replace(call, uses=(Use("a", view="target"),))
    with pytest.raises(Rejected, match="participant use"):
        Contract().accept(
            replace(filing, root=replace(root, blocks=(replace(first, work=(first.work[0], bad_use, first.work[2])), *root.blocks[1:])))
        )
    bad = add_work(filing, 0, Call("hidden-loop", lambda context, old: None, inputs=("old",)))
    with pytest.raises(Rejected, match="coordination handles"):
        Contract().accept(bad)


def test_sequence_lowering_preserves_cross_block_locals_without_hot_path_graph_discovery(monkeypatch):
    from . import prepare as preparation_module

    ready = prepare(Contract().accept(routed_views())).publish()

    def rediscovery(*args, **kwargs):
        pytest.fail("Hot path rediscovered accepted work")

    monkeypatch.setattr(preparation_module, "sequence_order", rediscovery)
    monkeypatch.setattr(preparation_module.Lowering, "bind", rediscovery)
    run(Engine(ready).run())
    source = ready.source["routed-work"]
    assert "first-loss" in source and "second-loss" in source and "finish-b" in source
    assert source.index("a-gradient") < source.index("b-gradient") < source.index("last-old-state-use") < source.index("advance-a")


def test_missing_last_use_release_and_cross_sequence_window_are_rejected():
    filing = routed_views(isolated=True)
    root = sequence(filing)
    second = replace(root.blocks[1], work=tuple(work for work in root.blocks[1].work if not isinstance(work, Release)))
    with pytest.raises(Rejected, match="explicit last-use release"):
        Contract().accept(replace(filing, root=replace(root, blocks=(root.blocks[0], second, root.blocks[2]))))
    # Explicit same-sequence links are required; names alone do not grant use.
    foreign = Run(Sequence("foreign", (Block("advance", (Advance("step", "ga", ("A",)),)),)), filing.participants, filing.units)
    with pytest.raises(Rejected, match="missing value"):
        Contract().accept(foreign)


def test_borrowed_tensors_cannot_escape_last_use_into_later_work():
    filing = add_work(routed_views(), 2, Call("use-released-h", lambda context, h: None, inputs=("h",), after=("advance-b",)))
    with pytest.raises(Rejected, match="borrowed values outlive released"):
        Contract().accept(filing)


def test_retained_loss_requires_its_accepted_derivative_correspondence():
    filing = routed_views(isolated=True)
    root = sequence(filing)
    first = root.blocks[0]
    first = replace(first, work=tuple(replace(work, retained=None) if isinstance(work, Differentiate) else work for work in first.work))
    with pytest.raises(Rejected, match="matching retained correspondence"):
        Contract().accept(replace(filing, root=replace(root, blocks=(first, *root.blocks[1:]))))


def test_custom_backward_returning_source_alias_cannot_alias_pending_gradient_storage():
    class Alias(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b):
            ctx.save_for_backward(b)
            return a * b

        @staticmethod
        def backward(ctx, *grad_outputs):
            # This oracle has scalar unit seed. Return source storage itself.
            (seed,) = grad_outputs
            assert torch.equal(seed, torch.ones_like(seed))
            (b,) = ctx.saved_tensors
            return b, None

    ready = prepare(Contract().accept(routed_views())).publish()
    state = ready.state
    assert state.image is not None
    a, b = (state.image.bindings[name].model.get_parameter("weight") for name in ("a", "b"))
    state.contribute(Alias.apply(a, b), ("A",), 1)
    assert a.grad is not None and not a.grad.requires_grad
    assert a.grad.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()
    with torch.no_grad():
        b.add_(10)
    assert float(a.grad) == 3.0


def test_view_forward_failure_and_restoration_failure_are_both_retained():
    class Broken(Scale):
        def __setattr__(self, name, value):
            if name == "training" and value is True and getattr(self, "break_restore", False):
                raise RuntimeError("restore failed")
            super().__setattr__(name, value)

        def forward(self, value):
            self.break_restore = True
            raise ValueError("forward failed")

    filing = routed_views()
    filing = replace(filing, participants=(Participant("frozen", lambda: Broken(1.0), lambda model: None), *filing.participants[1:]))
    ready = prepare(Contract().accept(filing)).publish()
    with pytest.raises(WorkFailed) as failure:
        run(Engine(ready).run())
    assert {str(error) for error in leaves(failure.value.cause)} == {"forward failed", "restore failed"}
    assert not ready.state.active_retentions and not ready.state.readers


def test_repeated_cancellation_does_not_release_protection_before_lifetime_cleanup(monkeypatch):
    async def check():
        entered, cleaning, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def pause(context):
            entered.set()
            await asyncio.Event().wait()

        filing = add_work(routed_views(), 0, Call("pause", pause, after=("a-gradient",)))
        ready = prepare(Contract().accept(filing)).publish()
        original = ready.state.release

        async def delayed(retained, region, *, aborted=False):
            if aborted:
                cleaning.set()
                await finish.wait()
            await original(retained, region, aborted=aborted)

        monkeypatch.setattr(ready.state, "release", delayed)
        task = asyncio.create_task(Engine(ready).run())
        await entered.wait()
        task.cancel()
        await cleaning.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and ready.state.readers and ready.state.active_retentions
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not ready.state.readers and not ready.state.active_retentions

    run(check())


def test_sequence_cannot_accept_unprotected_forward_to_backward_gap():
    filing = routed_views()
    root = sequence(filing)
    first = replace(
        root.blocks[0],
        work=tuple(
            replace(work, retained=None) if isinstance(work, (Call, Differentiate)) else work
            for work in root.blocks[0].work
            if not isinstance(work, Retain)
        ),
    )
    # Per-work leases alone are insufficient if another activity can change
    # the source while this sequence suspends between forward and backward.
    with pytest.raises(Rejected, match="matching retained correspondence"):
        Contract().accept(replace(filing, root=Sequence("unprotected", (first, Block("finish", (Advance("step", "ga", ("A",)),))))))


def test_block_local_grant_is_not_falsely_accepted_as_cross_block_handback():
    filing = two_pass()
    assert isinstance(filing.root, Block)
    with pytest.raises(Rejected, match="grant remains block-local"):
        Contract(two_pass=True).accept(replace(filing, root=Sequence("not-yet-supported", (filing.root,))))


@pytest.mark.parametrize("isolated", (False, True))
def test_retained_modes_require_synchronous_view_scope_not_broad_async_evaluation(isolated):
    filing = routed_views(isolated=isolated)
    root = sequence(filing)
    first = replace(
        root.blocks[0], work=tuple(replace(work, evaluation=True) if isinstance(work, Call) else work for work in root.blocks[0].work)
    )
    with pytest.raises(Rejected, match="named synchronous view modes"):
        Contract().accept(replace(filing, root=replace(root, blocks=(first, *root.blocks[1:]))))


@pytest.mark.parametrize("changed_aliases", (False, True))
def test_isolated_copy_preserves_alias_paths_not_just_distinct_member_count(changed_aliases):
    entered = []

    class Aliased(nn.Module):
        def __init__(self, changed=False):
            super().__init__()
            self.x = nn.Parameter(torch.tensor(1.0))
            self.y = nn.Parameter(torch.tensor(1.0)) if changed else self.x
            self.z = self.x if changed else nn.Parameter(torch.tensor(1.0))

        def __deepcopy__(self, memo):
            return Aliased(changed_aliases)

    def loss(context):
        entered.append(True)
        model = context.models["model"]
        return model.get_parameter("x") + 2 * model.get_parameter("y") + 3 * model.get_parameter("z")

    filing = Run(
        Sequence(
            "aliased-source",
            (
                Block(
                    "derive",
                    (
                        Retain("retain", ("model",), ("U",), "old", isolated=True),
                        Call("loss", loss, outputs=(Output("loss", torch.Tensor),), uses=(Use("model"),), retained="old"),
                        Differentiate("gradient", "loss", ("U",), "gradient", retained="old"),
                    ),
                ),
                Block("finish", (Release("release", "old"), Advance("step", "gradient", ("U",), after=("release",)))),
            ),
        ),
        (Participant("model", Aliased, lambda model: None),),
        (Unit("U", (("model", "x"), ("model", "y"), ("model", "z")), learning_rate=0.1),),
    )
    ready = prepare(Contract().accept(filing)).publish()
    assert ready.state.image is not None
    if changed_aliases:
        with pytest.raises(WorkFailed, match="alias/member correspondence"):
            run(Engine(ready).run())
        assert not entered and ready.state.image.units["U"].steps == 0
    else:
        run(Engine(ready).run())
        model = ready.state.image.bindings["model"].model
        assert model.get_parameter("x") is model.get_parameter("y")
        assert len(ready.state.image.units["U"].parameters) == 2
        assert float(model.get_parameter("x").detach()) == pytest.approx(0.7)
        assert float(model.get_parameter("z").detach()) == pytest.approx(0.7)
    assert not ready.state.readers and not ready.state.active_retentions


@pytest.mark.parametrize("copied_offset", (0.0, 10.0))
def test_isolated_copy_checks_registered_nonpersistent_buffers(copied_offset):
    entered = []

    class Offset(Scale):
        def __init__(self, weight=1.0, offset=0.0):
            super().__init__(weight)
            self.register_buffer("offset", torch.tensor(offset), persistent=False)

        def __deepcopy__(self, memo):
            copied = Offset(float(self.get_parameter("weight").detach()), copied_offset)
            copied.requires_grad_(self.get_parameter("weight").requires_grad)
            return copied

        def forward(self, value):
            entered.append(True)
            return super().forward(value) + self.get_buffer("offset")

    filing = routed_views(isolated=True)
    filing = replace(filing, participants=(Participant("frozen", Offset, lambda model: None), *filing.participants[1:]))
    ready = prepare(Contract().accept(filing)).publish()
    assert ready.state.image is not None
    assert "offset" not in ready.state.image.bindings["frozen"].model.state_dict()
    if copied_offset:
        with pytest.raises(WorkFailed, match="registered source state"):
            run(Engine(ready).run())
        assert not entered and all(unit.steps == 0 for unit in ready.state.image.units.values())
    else:
        run(Engine(ready).run())
        assert entered
        assert float(ready.state.image.bindings["b"].model.get_parameter("weight").detach()) == pytest.approx(2.6)
    assert not ready.state.readers and not ready.state.active_retentions


@pytest.mark.parametrize("changed_mode", (False, True))
def test_isolated_copy_preserves_registered_module_modes(changed_mode):
    class Modes(Scale):
        def __init__(self):
            super().__init__(1.0)
            self.child = nn.Identity().eval()

        def __deepcopy__(self, memo):
            copied = Modes().requires_grad_(self.get_parameter("weight").requires_grad)
            if changed_mode:
                copied.child.train()
            return copied

        def forward(self, value):
            return super().forward(value) * (2 if self.child.training else 1)

    filing = routed_views(isolated=True)
    filing = replace(filing, participants=(Participant("frozen", Modes, lambda model: None), *filing.participants[1:]))
    ready = prepare(Contract().accept(filing)).publish()
    assert ready.state.image is not None
    if changed_mode:
        with pytest.raises(WorkFailed, match="registered module state"):
            run(Engine(ready).run())
        assert all(unit.steps == 0 for unit in ready.state.image.units.values())
    else:
        run(Engine(ready).run())
        assert float(ready.state.image.bindings["b"].model.get_parameter("weight").detach()) == pytest.approx(2.6)
    assert not ready.state.readers and not ready.state.active_retentions


def test_no_gradient_view_cuts_passthrough_outputs_without_mutating_the_input():
    observed = []

    def loss(context):
        value = context.models["a"].get_parameter("weight")
        target = context.views["target"](value)
        observed.append((target is value, target.requires_grad, value.requires_grad))
        return target * value

    filing = Run(
        Block(
            "passthrough",
            (
                Call("loss", loss, outputs=(Output("loss", torch.Tensor),), uses=(Use("a"), Use("f", view="target"))),
                Differentiate("gradient", "loss", ("A",), "gradient"),
                Advance("step", "gradient", ("A",)),
            ),
        ),
        (Participant("a", lambda: Scale(2.0), lambda model: None), Participant("f", nn.Identity, lambda model: None)),
        (Unit("A", (("a", "weight"),), learning_rate=0.1),),
        views=(View("target", "f", gradients=False),),
    )
    ready = run(execute(filing))
    assert ready.state.image is not None
    assert observed == [(False, False, True)]
    assert float(ready.state.image.bindings["a"].model.get_parameter("weight").detach()) == pytest.approx(1.8)


def test_no_gradient_view_cuts_nested_tensor_outputs_and_restores_modes():
    class Structured(nn.Module):
        def __init__(self):
            super().__init__()
            self.child = nn.Identity().eval()

        def forward(self, value):
            return {"value": value, "other": [value * 2, (value, 4, None)]}

    model = Structured()
    target = PreparedView(View("target", "model", gradients=False, evaluation=True), model)
    value = torch.tensor(2.0, requires_grad=True)
    result = target(value)
    tensors = (result["value"], result["other"][0], result["other"][1][0])
    assert all(not tensor.requires_grad for tensor in tensors)
    assert torch.equal(result["value"], value) and result["value"] is not value
    # A gradient cut is not an isolated-storage claim.
    assert result["value"].untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
    assert result["other"][1][1:] == (4, None)
    assert value.requires_grad and torch.is_grad_enabled()
    assert model.training and not model.child.training


@pytest.mark.parametrize("shape", ("opaque", "tensor-key"))
def test_no_gradient_view_rejects_outputs_it_cannot_cut(shape):
    class Hidden:
        def __init__(self, value):
            self.value = value

    class Unsupported(nn.Module):
        def forward(self, value):
            return Hidden(value) if shape == "opaque" else {value: "hidden in key"}

    model = Unsupported()
    model.child = nn.Identity().eval()
    value = torch.tensor(2.0, requires_grad=True)
    target = PreparedView(View("target", "model", gradients=False, evaluation=True), model)
    with pytest.raises(NotReady, match="unsupported output"):
        target(value)
    assert model.training and not model.child.training and value.requires_grad
    assert torch.is_grad_enabled()
    # No new restriction on differentiable views of selected Python results.
    PreparedView(View("conduit", "model"), model)(value)


@pytest.mark.parametrize("raw_use", (False, True))
@pytest.mark.parametrize("fail", (False, True))
def test_call_evaluation_reaches_named_views_and_restores_mixed_modes(raw_use, fail):
    observed = []

    class Switch(nn.Module):
        def __init__(self):
            super().__init__()
            self.child = nn.Identity().eval()
            self.evaluations = 0

        def eval(self):
            self.evaluations += 1
            return super().eval()

        def forward(self, value):
            return value if self.training else -value

    model = Switch()

    def selected(context):
        assert bool(context.models) is raw_use
        observed.append(float(context.views["named"](torch.tensor(2.0))))
        if fail:
            raise ValueError("evaluation failed")

    uses = (Use("f", view="named"), Use("f", view="another")) + ((Use("f"),) if raw_use else ())
    filing = Run(
        Block("evaluate", (Call("evaluate", selected, uses=uses, evaluation=True),)),
        (Participant("f", lambda: model, lambda model: None),),
        views=(View("named", "f"), View("another", "f")),
    )
    ready = prepare(Contract().accept(filing)).publish()
    if fail:
        with pytest.raises(WorkFailed, match="evaluation failed"):
            run(Engine(ready).run())
        assert ready.state.unusable == {"f"}
    else:
        run(Engine(ready).run())
    assert observed == [-2.0] and model.evaluations == 1
    assert model.training and not model.child.training
    assert not ready.state.readers and not ready.state.writers


def test_unsupported_no_gradient_result_is_post_invocation_failure_with_possible_effects():
    class Hidden:
        def __init__(self, value):
            self.value = value

    class Changed(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("calls", torch.tensor(0), persistent=False)

        def forward(self, value):
            self.get_buffer("calls").add_(1)
            return Hidden(value)

    model = Changed()
    filing = Run(
        Block(
            "selected-work",
            (Call("call", lambda context: context.views["target"](torch.tensor(2.0)), uses=(Use("f", "write", "target"),)),),
        ),
        (Participant("f", lambda: model, lambda model: None),),
        views=(View("target", "f", gradients=False, evaluation=True),),
    )
    ready = prepare(Contract().accept(filing)).publish()
    with pytest.raises(WorkFailed) as failure:
        run(Engine(ready).run())
    assert isinstance(failure.value.cause, NotReady)
    assert "unsupported output" in str(failure.value.cause)
    assert int(model.get_buffer("calls")) == 1 and model.training
    assert ready.state.unusable == {"f"} and not ready.state.writers
    assert not failure.value.implementation_returned
    assert ready.state.stamp("f").numerical_epoch == 0  # No returned-effect certificate, not proof of no mutation.


@pytest.mark.parametrize("returned", (False, True))
def test_named_view_call_preserves_primary_and_all_evaluation_cleanup_failures(returned):
    primary, cleanup = ValueError("selected failure"), RuntimeError("restore failed")
    later_cleanup = RuntimeError("second restore failed")

    class Fragile(nn.Module):
        def __init__(self, failure):
            super().__init__()
            self.failure = failure

        def __setattr__(self, name, value):
            if name == "training" and value is True and getattr(self, "armed", False):
                raise self.failure
            super().__setattr__(name, value)

    model = Scale(1.0)
    model.bad = Fragile(cleanup)
    model.also_bad = Fragile(later_cleanup)
    model.good = nn.Identity()

    def selected(context):
        context.views["named"].model.bad.armed = True
        context.views["named"].model.also_bad.armed = True
        if not returned:
            raise primary

    filing = Run(
        Block("work", (Call("evaluate", selected, uses=(Use("model", view="named"),), evaluation=True),)),
        (Participant("model", lambda: model, lambda model: None),),
        views=(View("named", "model"),),
    )
    ready = prepare(Contract().accept(filing)).publish()
    with pytest.raises(WorkFailed) as failure:
        run(Engine(ready).run())
    assert failure.value.cause is (cleanup if returned else primary)
    assert failure.value.cleanup_failures == (cleanup, later_cleanup)
    assert failure.value.implementation_returned is returned
    assert model.training and model.good.training and not model.bad.training and not model.also_bad.training
    assert ready.state.unusable == {"model"} and not ready.state.writers
