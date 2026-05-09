from types import SimpleNamespace

import torch
from torch import nn

from library.adapters.shared.reporting import build_adapter_component_report_rows
from library.adapters.shared.trainables import AdapterTrainableParameterRef


def test_build_adapter_component_report_rows_tracks_model_and_adapter_modules():
    adapter = SimpleNamespace()
    param_a = nn.Parameter(torch.ones(4))
    param_b = nn.Parameter(torch.ones(6))
    param_c = nn.Parameter(torch.ones(3), requires_grad=False)

    adapter.describe_trainable_parameter_refs = lambda: [
        AdapterTrainableParameterRef(
            param=param_a,
            name="lora_unet.block1.up",
            algorithm="lora",
            component="unet",
            component_key="denoiser",
            target_path="unet.block1",
            adapter_module_path="lora_unet.block1.lora_down",
        ),
        AdapterTrainableParameterRef(
            param=param_b,
            name="lora_unet.block1.down",
            algorithm="lora",
            component="unet",
            component_key="denoiser",
            target_path="unet.block1",
            adapter_module_path="lora_unet.block1.lora_up",
        ),
        AdapterTrainableParameterRef(
            param=param_c,
            name="lora_clip_l.block1.up",
            algorithm="lora",
            component="clip_l",
            component_key="text_encoder1",
            target_path="clip_l.block1",
            adapter_module_path="lora_clip_l.block1.lora_down",
        ),
    ]

    rows = build_adapter_component_report_rows(adapter)

    assert [(row.label, row.component_key) for row in rows] == [("clip_l", "text_encoder1"), ("unet", "denoiser")]
    assert rows[1].modules_total == 1
    assert rows[1].adapter_modules_total == 2
    assert rows[1].params_trainable == 10


def test_build_adapter_component_report_rows_requires_adapter_module_path():
    adapter = SimpleNamespace()
    adapter.describe_trainable_parameter_refs = lambda: [
        AdapterTrainableParameterRef(
            param=nn.Parameter(torch.ones(1)),
            name="broken.weight",
            algorithm="lora",
            component="unet",
            component_key="denoiser",
            target_path="unet.block1",
        )
    ]

    try:
        build_adapter_component_report_rows(adapter)
    except TypeError as exc:
        assert "adapter_module_path" in str(exc)
    else:
        raise AssertionError("Expected missing adapter_module_path to fail")
