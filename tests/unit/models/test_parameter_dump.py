from torch import nn

from library.models.parameter_dump import (
    derive_parameter_dump_identifier,
    format_named_parameter_dump,
    resolve_named_parameter_components,
)


class DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 3)


class DummyDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Linear(3, 2)


class DummyVae(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Linear(2, 2)


def test_resolve_named_parameter_components_uses_explicit_sd3_names():
    components = resolve_named_parameter_components(
        model_type="sd3",
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
    assert "    proj.bias: {shape: [3], dtype: float32, requires_grad: false}\n" in rendered
    assert "    proj.weight: {shape: [3, 4], dtype: float32, requires_grad: false}\n" in rendered


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
        "    proj.weight: {shape: [3, 4], dtype: float32, requires_grad: true}\n"
    )
