"""Equivalent optimizer behavior and genuine default-GIL-off execution."""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
import textwrap

import pytest
import torch

from .optimizer import EagerSGD


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("lr", (0.0, 0.125))
@pytest.mark.parametrize("set_to_none", (False, True))
def test_eager_optimizer_matches_standard_sgd(dtype, lr, set_to_none):
    parameters = [torch.nn.Parameter(torch.tensor([2.0, -3.0], dtype=dtype)) for _ in range(2)]
    references = [torch.nn.Parameter(parameter.detach().clone()) for parameter in parameters]
    actual = EagerSGD(parameters, lr=lr)
    # Deliberately independent wrapper/oracle: its imports may enable the GIL
    # in THIS process. The separate integration process below must not do so.
    expected = torch.optim.SGD(references, lr=lr, foreach=False, fused=False)
    for step in range(3):
        for index, (parameter, reference) in enumerate(zip(parameters, references, strict=True)):
            gradient = torch.tensor([step + 1.0, index - 2.0], dtype=dtype) if step != index else None
            parameter.grad = gradient
            reference.grad = gradient.clone() if gradient is not None else None
        actual.step()
        expected.step()
        for parameter, reference in zip(parameters, references, strict=True):
            torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
        actual.zero_grad(set_to_none=set_to_none)
        expected.zero_grad(set_to_none=set_to_none)
        for parameter, reference in zip(parameters, references, strict=True):
            if reference.grad is None:
                assert parameter.grad is None
            else:
                torch.testing.assert_close(parameter.grad, reference.grad, rtol=0, atol=0)
    snapshot = actual.state_dict()
    assert snapshot == {"state": {}, "param_groups": [{"lr": lr, "params": [0, 1]}]}
    snapshot.clear()
    assert actual.state_dict()["state"] == {}  # Snapshot is not live optimizer state.


@pytest.mark.skipif(not sysconfig.get_config_var("Py_GIL_DISABLED"), reason="Requires a free-threaded Python build")
def test_candidate_examples_and_worker_calls_keep_the_gil_disabled():
    script = textwrap.dedent("""\
        import asyncio
        import importlib
        import sys
        import sysconfig

        assert sysconfig.get_config_var('Py_GIL_DISABLED') == 1
        assert not sys._is_gil_enabled()
        base = 'docs_design.models-strategy-trainer.execution_candidate.'
        examples = importlib.import_module(base + 'examples')
        graph = importlib.import_module(base + 'graph')
        runtime = importlib.import_module(base + 'prepare')

        async def check():
            await examples.main()  # All nine connected arrangements.
            for lowering in ('calls', 'inline'):
                ready = runtime.prepare(
                    graph.Contract().accept(examples.joint()),
                    lowering=lowering,
                    worker_calls=('objective',),
                    worker_limit=2,
                ).publish()
                await runtime.Engine(ready).run()
                assert ready.state.completed
                assert all(unit.steps == 1 for unit in ready.state.image.units.values())
                assert any(kind == 'worker_completed' for kind, _ in ready.state.trace)
                assert not sys._is_gil_enabled()

        asyncio.run(check())
        assert not sys._is_gil_enabled()
        assert 'triton' not in sys.modules and 'torch._dynamo' not in sys.modules
        print('GIL remains disabled; Triton and Dynamo unimported')
    """)
    environment = os.environ.copy()
    environment.pop("PYTHON_GIL", None)  # Never override an extension's safety declaration.
    result = subprocess.run([sys.executable, "-c", script], env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "GIL remains disabled" in result.stdout
