"""Test-only probes for standard-profile advancement policy questions.

These do not implement Trainer optimization, distributed synchronization, or
contract acceptance. They test what an accepted policy would have to preserve.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch

from library.training.execution import Action, Operation, OptimizationInput, Value, compile_run


@pytest.mark.training
@pytest.mark.unit
def test_alternating_units_keep_independent_accumulation_windows():
    batch = Value("batch")
    generator_loss = Value("generator_loss")
    discriminator_loss = Value("discriminator_loss")
    generator = torch.nn.Parameter(torch.tensor(1.0))
    discriminator = torch.nn.Parameter(torch.tensor(2.0))
    prepared = compile_run(
        (
            Action(
                "generator_action",
                inputs=(batch,),
                operations=(
                    Operation(
                        "generator_objective",
                        lambda item: (generator * discriminator.detach() * item).square(),
                        (batch,),
                        (generator_loss,),
                    ),
                ),
                optimization=(OptimizationInput("generator", generator_loss),),
            ),
            Action(
                "discriminator_action",
                inputs=(batch,),
                operations=(
                    Operation(
                        "discriminator_objective",
                        lambda item: (discriminator * generator.detach() * item - 3).square(),
                        (batch,),
                        (discriminator_loss,),
                    ),
                ),
                optimization=(OptimizationInput("discriminator", discriminator_loss),),
            ),
        ),
        due=lambda coordinates: ("generator_action",) if coordinates["turn"] % 2 == 0 else ("discriminator_action",),
    )
    parameters = {"generator": generator, "discriminator": discriminator}
    optimizers = {unit: torch.optim.SGD((parameter,), lr=0.1) for unit, parameter in parameters.items()}
    accepted_sources = {"generator": generator_loss, "discriminator": discriminator_loss}
    window_size = {"generator": 2, "discriminator": 1}
    pending = {"generator": 0, "discriminator": 0}
    advances = {"generator": 0, "discriminator": 0}
    events: list[str] = []

    def consume(turn: int) -> None:
        action = prepared.due_actions({"turn": turn})[0]
        result = action.execute({batch: torch.tensor(1.0)})
        assert len(result.optimization) == 1
        offered = result.optimization[0]
        unit = offered.unit
        assert offered.source == accepted_sources[unit]
        assert isinstance(offered.value, torch.Tensor)
        if pending[unit] == 0:
            optimizers[unit].zero_grad(set_to_none=True)
            events.append(f"open:{unit}")
        offered.value.backward()
        pending[unit] += 1
        events.append(f"contribute:{unit}")
        if pending[unit] == window_size[unit]:
            torch.nn.utils.clip_grad_norm_((parameters[unit],), 100.0)
            optimizers[unit].step()
            optimizers[unit].zero_grad(set_to_none=True)
            pending[unit] = 0
            advances[unit] += 1
            events.append(f"advance:{unit}")

    consume(0)
    first_generator_gradient = generator.grad.detach().clone()
    assert advances == {"generator": 0, "discriminator": 0}

    consume(1)
    assert generator.grad is not None
    torch.testing.assert_close(generator.grad, first_generator_gradient)
    assert advances == {"generator": 0, "discriminator": 1}

    consume(2)
    assert advances == {"generator": 1, "discriminator": 1}
    assert generator.grad is None
    assert events == [
        "open:generator",
        "contribute:generator",
        "open:discriminator",
        "contribute:discriminator",
        "advance:discriminator",
        "contribute:generator",
        "advance:generator",
    ]


@pytest.mark.training
@pytest.mark.unit
def test_distinct_sources_need_explicit_gradient_routing_and_backend_support():
    batch = Value("batch")
    first_loss = Value("first_loss")
    second_loss = Value("second_loss")
    first = torch.nn.Parameter(torch.tensor(1.0))
    second = torch.nn.Parameter(torch.tensor(2.0))

    def objectives(_item: object) -> tuple[torch.Tensor, torch.Tensor]:
        shared = first * second
        return (shared - 1).square(), (shared + 1).square()

    action = Action(
        "two_sources",
        inputs=(batch,),
        operations=(Operation("objectives", objectives, (batch,), (first_loss, second_loss)),),
        optimization=(OptimizationInput("first", first_loss), OptimizationInput("second", second_loss)),
    )
    result = compile_run((action,), due=lambda _: ("two_sources",)).actions["two_sources"].execute({batch: None})
    offered = {item.unit: item for item in result.optimization}
    accepted_sources = {"first": first_loss, "second": second_loss}
    parameters = {"first": first, "second": second}

    def route_gradients(*, backend_supports_isolation: bool) -> Mapping[str, torch.Tensor]:
        if not backend_supports_isolation:
            raise ValueError("backend cannot realize accepted per-unit gradient routing")
        if {unit: item.source for unit, item in offered.items()} != accepted_sources:
            raise ValueError("optimization sources differ from accepted policy")
        first_value = offered["first"].value
        second_value = offered["second"].value
        assert isinstance(first_value, torch.Tensor)
        assert isinstance(second_value, torch.Tensor)
        return {
            "first": torch.autograd.grad(first_value, parameters["first"], retain_graph=True)[0],
            "second": torch.autograd.grad(second_value, parameters["second"])[0],
        }

    with pytest.raises(ValueError, match="backend cannot realize"):
        route_gradients(backend_supports_isolation=False)
    assert first.grad is None and second.grad is None

    first_value = offered["first"].value
    second_value = offered["second"].value
    assert isinstance(first_value, torch.Tensor)
    assert isinstance(second_value, torch.Tensor)
    summed_gradients = torch.autograd.grad(first_value + second_value, (first, second), retain_graph=True)
    torch.testing.assert_close(summed_gradients[0], torch.tensor(16.0))
    torch.testing.assert_close(summed_gradients[1], torch.tensor(8.0))

    gradients = route_gradients(backend_supports_isolation=True)
    torch.testing.assert_close(gradients["first"], torch.tensor(4.0))
    torch.testing.assert_close(gradients["second"], torch.tensor(6.0))

    # Backward on the sum would yield 16 and 8, violating the accepted routing.
    for unit, parameter in parameters.items():
        parameter.grad = gradients[unit]
        torch.optim.SGD((parameter,), lr=0.1).step()
    torch.testing.assert_close(first.detach(), torch.tensor(0.6))
    torch.testing.assert_close(second.detach(), torch.tensor(1.4))
