"""Module-level feasibility tests, not a production view/AMP implementation.

The private-structure experiment assumes registered state and ordinary Python
behavior can be safely copied and buffers are read-only. Negative cases show
why it cannot transparently adapt arbitrary modules. Existing RunState/View
interfaces and the uncached worker default are deliberately untouched.
"""

from __future__ import annotations

import copy
import threading
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass

import pytest
import torch
from torch import nn
from torch.func import functional_call

from .autocast_cache_probe import cast_nodes, gradient


@dataclass
class BoundModule:
    module: nn.Module
    parameters: dict[str, torch.Tensor]
    buffers: dict[str, torch.Tensor]
    thread: int
    active: bool = True

    def __call__(self, *args, **kwargs):
        if not self.active or threading.get_ident() != self.thread:
            raise RuntimeError("Binding is outside its synchronous use")
        return functional_call(self.module, (self.parameters, self.buffers), args, kwargs, strict=True, tie_weights=True)


class PrivateModule[M: nn.Module]:
    """One experimental private structure, requiring external source protection.

    cast_names is selected implementation knowledge, not a generic AMP policy.
    Copying registered tensors is avoided; copying arbitrary module state is NOT
    certified. One structure has one active user, rather than concurrent swaps.
    """

    def __init__(self, source: M, cast_names: frozenset[str], *, evaluation: bool = False):
        modules = tuple(source.named_modules(remove_duplicate=False))
        if len({id(module) for _, module in modules}) != len(modules):
            raise ValueError("Shared submodule aliases need a different binding realization")
        self.source = source
        self.sources = dict(source.named_parameters(remove_duplicate=False))
        self.buffers = dict(source.named_buffers(remove_duplicate=False))
        if cast_names - self.sources.keys():
            raise ValueError("Unknown cast source")
        self.cast_names = cast_names
        by_identity = {}
        for name, parameter in self.sources.items():
            cast = name in cast_names
            if id(parameter) in by_identity and by_identity[id(parameter)] != cast:
                raise ValueError("Tied parameter has conflicting precision selections")
            by_identity[id(parameter)] = cast
        memo = {id(tensor): tensor for tensor in (*self.sources.values(), *self.buffers.values())}
        self.module = copy.deepcopy(source, memo)
        if evaluation:
            self.module.eval()
        self.busy = threading.Lock()

    @contextmanager
    def region(self, dtype: torch.dtype) -> Generator[BoundModule]:
        if not self.busy.acquire(blocking=False):
            raise RuntimeError("This private structure already has an active use")
        binding = None
        try:
            current = {name: id(value) for name, value in self.source.named_parameters(remove_duplicate=False)}
            current_buffers = {name: id(value) for name, value in self.source.named_buffers(remove_duplicate=False)}
            if current != {name: id(value) for name, value in self.sources.items()} or current_buffers != {
                name: id(value) for name, value in self.buffers.items()
            }:
                raise RuntimeError("Source bindings changed; rebuild the private structure")
            converted, parameters = {}, {}
            for name, parameter in self.sources.items():
                if id(parameter) not in converted:
                    converted[id(parameter)] = parameter.to(dtype) if name in self.cast_names else parameter
                parameters[name] = converted[id(parameter)]
            binding = BoundModule(self.module, parameters, dict(self.buffers), threading.get_ident())
            # Preserve Torch's op-specific AMP rules. Already-converted selected
            # parameters bypass its global weight cache; other casts are uncached.
            with torch.autocast("cpu", dtype=dtype, cache_enabled=False):
                yield binding
        finally:
            if binding is not None:
                binding.active = False
            self.busy.release()


class TiedBlock(nn.Module):
    gain: torch.Tensor

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 4, bias=False)
        self.alias = nn.Linear(4, 4, bias=False)
        self.alias.weight = self.layer.weight
        self.tail = nn.Linear(4, 4, bias=False)
        self.register_buffer("gain", torch.tensor(0.5))
        with torch.no_grad():
            self.layer.weight.fill_(0.125)
            self.tail.weight.fill_(0.25)

    def forward(self, value, repeats=3, activate=True, gate=None):
        assert self.layer.weight is self.alias.weight
        for index in range(repeats):
            value = (self.layer if index % 2 == 0 else self.alias)(value)
            if index == 0 and gate is not None:
                gate(self)
            if activate:
                value = torch.tanh(value)
        return self.tail(value) * self.gain


@pytest.mark.parametrize("first_dtype", (torch.bfloat16, torch.float16))
def test_private_module_views_preserve_source_ties_gradients_and_dynamic_calls(first_dtype):
    source = TiedBlock()
    original = dict(source.named_parameters(remove_duplicate=False))
    names = frozenset(original)
    views = (PrivateModule(source, names), PrivateModule(source, names, evaluation=True))
    assert not ({id(module) for module in views[0].module.modules()} & {id(module) for module in views[1].module.modules()})
    assert all(view.module is not source and view.module.layer.weight is source.layer.weight for view in views)
    second_dtype = torch.float16 if first_dtype == torch.bfloat16 else torch.bfloat16
    barrier = threading.Barrier(2)
    inputs = torch.ones(2, 4)
    references = {}
    # Independent native-AMP baseline, before any concurrent execution.
    for dtype in (first_dtype, second_dtype):
        with torch.autocast("cpu", dtype=dtype, cache_enabled=True):
            values = (source(inputs, repeats=3), source(inputs, repeats=2, activate=False))
        references[dtype] = (values, torch.autograd.grad(tuple(value.float().sum() for value in values), tuple(source.parameters())))

    def execute(view, dtype):
        def gate(module):
            assert module is view.module
            assert module.layer.weight.dtype == dtype
            assert module.layer.weight is module.alias.weight
            assert source.training and not views[1].module.training
            assert all(parameter is dict(source.named_parameters(remove_duplicate=False))[name] for name, parameter in original.items())
            assert all(parameter.dtype == torch.float32 for parameter in source.parameters())
            barrier.wait(timeout=10)

        with view.region(dtype) as bound:
            values = (bound(inputs, repeats=3, gate=gate), bound(inputs, repeats=2, activate=False))
            assert cast_nodes(values) == 2  # one cast per distinct source, reused across both forwards
        return values, threading.get_ident()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = tuple(pool.submit(execute, view, dtype) for view, dtype in zip(views, (first_dtype, second_dtype), strict=True))
        results = tuple(future.result(timeout=15) for future in futures)
    assert results[0][1] != results[1][1]
    for (values, _), dtype in zip(results, (first_dtype, second_dtype), strict=True):
        reference, expected_gradients = references[dtype]
        assert all(torch.equal(value, expected) for value, expected in zip(values, reference, strict=True))
        gradients = torch.autograd.grad(tuple(value.float().sum() for value in values), tuple(source.parameters()))
        assert all(torch.equal(value, expected) for value, expected in zip(gradients, expected_gradients, strict=True))
    assert all(parameter is dict(source.named_parameters(remove_duplicate=False))[name] for name, parameter in original.items())
    assert source.training and source.gain.item() == 0.5


def test_shared_module_functional_call_is_not_parameter_table_isolation():
    linear = nn.Linear(4, 4, bias=False)
    source = nn.Sequential(linear)
    parameter = linear.weight
    first_entered, second_entered, first_done = threading.Event(), threading.Event(), threading.Event()
    role = threading.local()

    def gate(module, args):
        if role.first:
            first_entered.set()
            assert second_entered.wait(10)
        else:
            second_entered.set()
            assert first_done.wait(10)

    hook = source.register_forward_pre_hook(gate)

    def first():
        role.first = True
        try:
            return functional_call(source, {"0.weight": parameter.to(torch.bfloat16)}, (torch.ones(2, 4, dtype=torch.bfloat16),))
        finally:
            first_done.set()

    def second():
        role.first = False
        assert first_entered.wait(10)
        return functional_call(source, {"0.weight": parameter.to(torch.float16)}, (torch.ones(2, 4, dtype=torch.float16),))

    try:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="shared-module") as pool:
            futures = (pool.submit(first), pool.submit(second))
            for future in futures:
                with pytest.raises(RuntimeError, match="same dtype"):
                    future.result(timeout=15)
        assert linear.weight is not parameter and linear.weight.dtype == torch.bfloat16
    finally:
        # The deliberately broken disposable fixture must not retain corruption.
        hook.remove()
        linear.weight = parameter


class NormBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 4, bias=False)
        self.norm = nn.RMSNorm(4, eps=1e-5)
        self.tail = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.eye(4))
            self.tail.weight.copy_(torch.eye(4))
            self.norm.weight.copy_(torch.tensor([1.001, 1.003, 0.501, 2.001]))

    def forward(self, value):
        return self.tail(self.norm(self.head(value)))


def test_parameter_view_does_not_automatically_reproduce_amp_precision_policy():
    source = NormBlock()
    inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    selected = PrivateModule(source, frozenset(("head.weight", "tail.weight")))
    blanket = PrivateModule(source, frozenset(dict(source.named_parameters())))
    with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=True):
        expected = source(inputs)
    expected_gradients = torch.autograd.grad(expected.float().sum(), tuple(source.parameters()))
    with selected.region(torch.bfloat16) as bound:
        actual = bound(inputs)
    assert torch.equal(actual, expected)
    gradients = torch.autograd.grad(actual.float().sum(), tuple(source.parameters()))
    assert all(torch.equal(value, expected) for value, expected in zip(gradients, expected_gradients, strict=True))
    with blanket.region(torch.bfloat16) as bound:
        wrong = bound(inputs)
    assert not torch.equal(wrong, expected), "blanket casting must expose the changed algorithm"


def test_private_view_refreshes_values_but_rejects_changed_parameter_bindings():
    source = nn.Linear(4, 4, bias=False)
    view = PrivateModule(source, frozenset(("weight",)))
    inputs = torch.ones(2, 4)
    with view.region(torch.bfloat16) as bound:
        before = bound(inputs)
    before_gradient = gradient((before,), source.weight)
    with torch.no_grad():
        source.weight.mul_(2)
    with view.region(torch.bfloat16) as bound:
        after = bound(inputs)
    assert torch.equal(after, before * 2)
    after_gradient = gradient((after,), source.weight)
    # Linear's weight gradient depends on its input, not its weight value.
    assert torch.equal(before_gradient, torch.full_like(source.weight, 2))
    assert torch.equal(after_gradient, before_gradient)
    source.weight = nn.Parameter(source.weight.detach().clone())
    with pytest.raises(RuntimeError, match="Source bindings changed"), view.region(torch.bfloat16):
        pass


def test_private_structure_rejects_replaced_buffer_before_execution():
    source = nn.BatchNorm1d(4)
    view = PrivateModule(source, frozenset())
    source.running_mean = torch.ones(4)
    with pytest.raises(RuntimeError, match="Source bindings changed"), view.region(torch.bfloat16):
        pass


def test_one_private_structure_rejects_overlapping_or_expired_binding():
    source = nn.Linear(4, 4, bias=False)
    view = PrivateModule(source, frozenset(("weight",)))
    with view.region(torch.bfloat16) as bound:
        with pytest.raises(RuntimeError, match="already has an active use"), view.region(torch.float16):
            pass
        with ThreadPoolExecutor(max_workers=1) as pool, pytest.raises(RuntimeError, match="outside its synchronous use"):
            pool.submit(bound, torch.ones(2, 4)).result(timeout=10)
    with pytest.raises(RuntimeError, match="outside its synchronous use"):
        bound(torch.ones(2, 4))


def test_tied_parameter_cannot_have_conflicting_view_precision():
    with pytest.raises(ValueError, match="Tied parameter"):
        PrivateModule(TiedBlock(), frozenset(("layer.weight",)))


def test_failure_restores_private_structure_and_allows_a_fresh_use():
    source = TiedBlock()
    view = PrivateModule(source, frozenset(dict(source.named_parameters(remove_duplicate=False))))

    def fails(_):
        raise ValueError("selected forward failed")

    with pytest.raises(ValueError, match="selected forward failed"), view.region(torch.bfloat16) as bound:
        bound(torch.ones(2, 4), gate=fails)
    assert view.module.layer.weight is source.layer.weight
    with view.region(torch.bfloat16) as bound:
        assert torch.isfinite(bound(torch.ones(2, 4))).all()


def test_private_structure_alone_does_not_isolate_mutable_buffers():
    source = nn.BatchNorm1d(4)
    assert source.num_batches_tracked is not None and source.running_mean is not None
    view = PrivateModule(source, frozenset())
    assert view.module.running_mean is source.running_mean
    with view.region(torch.bfloat16) as bound:
        bound(torch.full((2, 4), 2.0))
    assert source.num_batches_tracked.item() == 1
    assert not torch.equal(source.running_mean, torch.zeros(4))
    # This is negative evidence: independent module dictionaries do not grant
    # buffer mutation authority or create independent logical running stats.


def test_shared_submodule_aliases_need_a_different_binding_realization():
    source = TiedBlock()
    source.alias = source.layer
    names = frozenset(dict(source.named_parameters(remove_duplicate=False)))
    with pytest.raises(ValueError, match="Shared submodule aliases"):
        PrivateModule(source, names)
    # Pin the reason for this experimental rejection: two paths to one physical
    # slot are swapped/restored twice by the current functional_call machinery.
    parameters = dict(source.named_parameters(remove_duplicate=False))
    private = copy.deepcopy(source, {id(value): value for value in parameters.values()})
    casts = {id(value): value.to(torch.bfloat16) for value in parameters.values()}
    supplied = {name: casts[id(value)] for name, value in parameters.items()}
    functional_call(private, (supplied, dict(private.named_buffers())), (torch.ones(2, 4, dtype=torch.bfloat16),), strict=True)
    assert private.layer.weight is not source.layer.weight
    assert source.layer.weight.dtype == torch.float32


def test_deepcopy_and_to_severs_the_gradient_route_to_original_parameters():
    source = nn.Linear(4, 4, bias=False)
    replica = copy.deepcopy(source).to(torch.bfloat16)
    value = replica(torch.ones(2, 4, dtype=torch.bfloat16))
    assert torch.autograd.grad(value.float().sum(), tuple(source.parameters()), allow_unused=True) == (None,)


def test_one_parameter_can_need_both_low_precision_and_original_fp32_uses():
    class MixedUse(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(4, 4, bias=False)

        def forward(self, value):
            return self.projection(value), self.projection.weight.float().square().sum()

    source = MixedUse()
    with torch.no_grad():
        source.projection.weight.fill_(1.001)
    view = PrivateModule(source, frozenset(("projection.weight",)))
    inputs = torch.ones(2, 4)
    with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=True):
        expected, original_penalty = source(inputs)
    original_gradient = gradient((original_penalty,), source.projection.weight)
    with view.region(torch.bfloat16) as bound:
        actual, changed_penalty = bound(inputs)
    assert torch.equal(actual, expected)
    assert not torch.equal(original_penalty, changed_penalty)
    assert not torch.equal(original_gradient, gradient((changed_penalty,), source.projection.weight))
    # A per-parameter dtype choice cannot preserve both uses by itself. AMP's
    # existing per-operation decisions retain the master-weight use correctly.
