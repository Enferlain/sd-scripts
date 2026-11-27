"""Device settings a run applies (D15): deterministic algorithms on the CUDA adapter, each case in a fresh process."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]

pytestmark = [pytest.mark.requires_gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")]


def fresh(script: str) -> str:
    """Runs the script in a fresh process without a cuBLAS workspace setting and returns what it printed."""
    env = {name: value for name, value in os.environ.items() if name != "CUBLAS_WORKSPACE_CONFIG"} | {"PYTHONPATH": str(ROOT)}
    done = subprocess.run([sys.executable, "-c", textwrap.dedent(script)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-3000:]
    return done.stdout


def test_the_deterministic_setting_applies_for_the_process_and_provides_cublas_s_workspace() -> None:
    out = fresh(
        """
        import os, torch
        from library.iro.devices.cuda import Cuda

        device = Cuda(0, deterministic=True)
        print(os.environ["CUBLAS_WORKSPACE_CONFIG"], torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark)
        print(device.settings)
        """
    )
    assert out.splitlines() == [":4096:8 True False", "{'deterministic': True, 'cudnn_benchmark': False, 'cublas_workspace': ':4096:8'}"]


def test_the_deterministic_setting_is_refused_once_cuda_has_started_without_the_workspace() -> None:
    out = fresh(
        """
        import torch
        from library.iro.devices.cuda import Cuda

        torch.cuda.init()
        try:
            Cuda(0, deterministic=True)
        except RuntimeError as refused:
            print(refused)
        """
    )
    assert "deterministic cuBLAS needs CUBLAS_WORKSPACE_CONFIG before CUDA starts" in out


def test_an_adapter_without_the_setting_leaves_the_process_as_it_is() -> None:
    out = fresh(
        """
        import os, torch
        from library.iro.devices.cuda import Cuda

        device = Cuda(0)
        print("CUBLAS_WORKSPACE_CONFIG" in os.environ, torch.are_deterministic_algorithms_enabled(), device.settings)
        """
    )
    assert out.strip() == "False False {'deterministic': False, 'cudnn_benchmark': False, 'cublas_workspace': None}"


def test_the_setting_is_the_process_s_so_every_adapter_reports_what_the_process_runs_with() -> None:
    """PyTorch's switch is process-wide: an adapter made without the setting after one applied it runs deterministic too, and says so."""
    out = fresh(
        """
        import torch
        from library.iro.devices.cuda import Cuda

        first = Cuda(0, deterministic=True)
        second = Cuda(torch.cuda.device_count() - 1)
        print(second.settings, first.settings)
        """
    )
    on = "{'deterministic': True, 'cudnn_benchmark': False, 'cublas_workspace': ':4096:8'}"
    assert out.strip() == f"{on} {on}"


def test_a_workspace_setting_deterministic_cublas_cannot_use_is_refused() -> None:
    out = fresh(
        """
        import os
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"
        from library.iro.devices.cuda import Cuda
        try:
            Cuda(0, deterministic=True)
        except RuntimeError as refused:
            print(refused)
        """
    )
    assert "CUBLAS_WORKSPACE_CONFIG is ':0:0', and deterministic cuBLAS needs :4096:8 or :16:8" in out
