import torch
from torch import nn

from library.models import NamedParameterComponentNames, build_named_components, build_selector_name
from library.models.parameter_dump import (
    derive_parameter_dump_identifier,
    format_component_module_dump,
    format_component_state_dump,
    format_named_parameter_dump,
)


class DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 3)
        self.register_buffer("scale", torch.ones(1))


class DummyDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Linear(3, 2)


class DummyVae(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Linear(2, 2)


def test_build_named_components_uses_explicit_component_names():
    components = build_named_components(
        component_names=NamedParameterComponentNames(
            text_encoder_names=("clip_l", "clip_g", "t5xxl"),
            vae_name="vae",
            denoiser_name="mmdit",
        ),
        text_encoders=[DummyEncoder(), DummyEncoder(), DummyEncoder()],
        vae=DummyVae(),
        denoiser=DummyDenoiser(),
    )

    assert [name for name, _ in components] == ["clip_l", "clip_g", "t5xxl", "vae", "mmdit"]


def test_format_named_parameter_dump_groups_one_line_per_parameter():
    encoder = DummyEncoder()
    for param in encoder.parameters():
        param.requires_grad_(False)

    rendered = format_named_parameter_dump(
        identifier="sdxl-sgm",
        components=[("clip_l", encoder)],
    )

    assert rendered.startswith("identifier: sdxl-sgm\ncomponents:\n  clip_l:\n")
    assert f"    {build_selector_name('clip_l', 'proj.bias')}: {{shape: [3], dtype: float32, requires_grad: false}}\n" in rendered
    assert (
        f"    {build_selector_name('clip_l', 'proj.weight')}: {{shape: [3, 4], dtype: float32, requires_grad: false}}\n"
        in rendered
    )
    assert "scale" not in rendered


def test_format_named_parameter_dump_can_filter_to_trainable_only():
    module = DummyEncoder()
    module.proj.weight.requires_grad_(True)
    module.proj.bias.requires_grad_(False)

    rendered = format_named_parameter_dump(
        identifier=derive_parameter_dump_identifier("sd3", "medium"),
        components=[("clip_l", module)],
        trainable_only=True,
    )

    assert rendered == (
        "identifier: sd3-medium\n"
        "components:\n"
        "  clip_l:\n"
        "    clip_l.proj.weight: {shape: [3, 4], dtype: float32, requires_grad: true}\n"
    )


def test_format_component_state_dump_includes_parameters_and_buffers():
    module = DummyEncoder()
    module.proj.weight.requires_grad_(False)

    rendered = format_component_state_dump(
        identifier="sdxl-sgm",
        components=[("clip_l", module)],
    )

    assert rendered.startswith("identifier: sdxl-sgm\ncomponents:\n  clip_l:\n")
    assert "    parameters:\n" in rendered
    assert "      clip_l.proj.weight: {kind: parameter, shape: [3, 4], dtype: float32, requires_grad: false}\n" in rendered
    assert "    buffers:\n" in rendered
    assert "      clip_l.scale: {kind: buffer, shape: [1], dtype: float32, requires_grad: false, persistent: true}\n" in rendered


def test_format_component_module_dump_includes_named_modules():
    rendered = format_component_module_dump(
        identifier="sd15",
        components=[("clip_l", DummyEncoder())],
    )

    assert rendered == (
        "identifier: sd15\n"
        "components:\n"
        "  clip_l:\n"
        "    modules:\n"
        "      <root>: {type: DummyEncoder}\n"
        "      proj: {type: Linear}\n"
    )
