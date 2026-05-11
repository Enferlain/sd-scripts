import argparse
from types import SimpleNamespace

from torch import nn

import library.models as model_metadata
from library.models import LoadedModelComponentSpec, NamedParameterComponentNames
from tools.model_management import dump_named_parameters


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


class CrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(4, 4)
        self.to_k = nn.Linear(4, 4)
        self.to_out = nn.ModuleList([nn.Linear(4, 4)])


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 4), nn.SiLU(), nn.Identity())


class BasicTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = CrossAttention()
        self.ff = FeedForward()
        self.norm = nn.LayerNorm(4)


class Transformer2DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_in = nn.Linear(4, 4)
        self.blocks = nn.ModuleList([BasicTransformerBlock()])


def test_build_runtime_cfg_sets_minimal_strategy_fields():
    args = SimpleNamespace(
        model_type="sd3",
        model_path="tests/assets/model.safetensors",
        vae=None,
        clip_l="clip_l.safetensors",
        clip_g="clip_g.safetensors",
        t5xxl="t5xxl.safetensors",
        disable_mmap=True,
        dtype="bf16",
    )

    cfg = dump_named_parameters.build_runtime_cfg(args)

    assert cfg.model.model_type == "sd3"
    assert cfg.model.pretrained_model_name_or_path == "tests/assets/model.safetensors"
    assert cfg.model.clip_l == "clip_l.safetensors"
    assert cfg.data.caching.disable_mmap_load_safetensors is True
    assert cfg.performance.precision.mixed_precision == "bf16"
    assert cfg.objective.prediction == "epsilon"


def test_filter_components_keeps_requested_order_from_input_components():
    components = [("clip_l", DummyEncoder()), ("vae", DummyVae()), ("mmdit", DummyDenoiser())]

    filtered = dump_named_parameters.filter_components(components, ["mmdit", "clip_l"])

    assert [name for name, _ in filtered] == ["clip_l", "mmdit"]


def test_resolve_component_names_reads_existing_package_metadata(monkeypatch):
    fake_package = SimpleNamespace(
        LOADED_MODEL_COMPONENT_SPECS=(
            LoadedModelComponentSpec(key="text_encoder1", public_name="clip_l", roles=("text_encoder",)),
            LoadedModelComponentSpec(key="text_encoder2", public_name="clip_g", roles=("text_encoder",)),
            LoadedModelComponentSpec(key="vae", public_name="vae", roles=("vae",)),
            LoadedModelComponentSpec(key="denoiser", public_name="unet", roles=("denoiser",)),
        )
    )
    monkeypatch.setattr(model_metadata.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(model_metadata.importlib, "import_module", lambda _: fake_package)

    component_names = model_metadata.resolve_component_names("sdxl")

    assert component_names == NamedParameterComponentNames(
        text_encoder_names=("clip_l", "clip_g"),
        vae_name="vae",
        denoiser_name="unet",
    )


def test_validate_model_type_accepts_supported_types(monkeypatch):
    parser = argparse.ArgumentParser()
    monkeypatch.setattr(dump_named_parameters, "supported_model_types", lambda: ("sdxl", "sd3"))

    dump_named_parameters.validate_model_type(parser, "sd3")


def test_format_component_summary_dump_builds_component_composite_child_type_cheat_sheet():
    rendered = dump_named_parameters.format_component_summary_dump(
        identifier="sdxl",
        components=[("unet", Transformer2DModel())],
    )

    assert rendered == (
        "identifier: sdxl\n"
        "components:\n"
        "  unet:\n"
        "    Transformer2DModel:\n"
        "      - Linear\n"
        "      - BasicTransformerBlock\n"
        "    BasicTransformerBlock:\n"
        "      - CrossAttention\n"
        "      - FeedForward\n"
        "      - LayerNorm\n"
        "    CrossAttention:\n"
        "      - Linear\n"
        "    FeedForward:\n"
        "      - Linear\n"
        "      - SiLU\n"
    )


def test_load_components_uses_strategy_loading_path(monkeypatch):
    args = SimpleNamespace(
        model_type="sd3",
        model_path="tests/assets/model.safetensors",
        vae=None,
        clip_l=None,
        clip_g=None,
        t5xxl=None,
        device="cpu",
        dtype="float32",
        disable_mmap=False,
        identifier=None,
        component=[],
        trainable_only=False,
        view="parameters",
        output=None,
    )
    expected_text_encoders = [DummyEncoder(), DummyEncoder(), DummyEncoder()]
    expected_vae = DummyVae()
    expected_denoiser = DummyDenoiser()

    class FakeStrategy:
        def load_target_model(self, cfg, weight_dtype, accelerator):
            assert cfg.model.model_type == "sd3"
            assert str(accelerator.device) == "cpu"
            return "medium", expected_text_encoders, expected_vae, expected_denoiser

    monkeypatch.setattr(dump_named_parameters, "build_strategy", lambda cfg: FakeStrategy())
    monkeypatch.setattr(
        dump_named_parameters,
        "resolve_component_names",
        lambda model_type: NamedParameterComponentNames(
            text_encoder_names=("clip_l", "clip_g", "t5xxl"),
            vae_name="vae",
            denoiser_name="mmdit",
        ),
    )

    identifier, components = dump_named_parameters.load_components(args)

    assert identifier == "sd3-medium"
    assert [name for name, _ in components] == ["clip_l", "clip_g", "t5xxl", "vae", "mmdit"]
