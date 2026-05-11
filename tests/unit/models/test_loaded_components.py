from types import SimpleNamespace

import library.models.components as component_metadata

from library.models import LoadedModelComponentSpec, resolve_component_names, resolve_component_specs


def test_resolve_component_specs_returns_sd_component_order_and_identity():
    component_specs = resolve_component_specs("sd")

    assert component_specs == (
        LoadedModelComponentSpec(key="text_encoder1", public_name="clip_l", roles=("text_encoder",)),
        LoadedModelComponentSpec(key="vae", public_name="vae", roles=("vae",)),
        LoadedModelComponentSpec(key="denoiser", public_name="unet", roles=("denoiser",)),
    )


def test_resolve_component_specs_returns_sdxl_component_order_and_identity():
    component_specs = resolve_component_specs("sdxl")

    assert component_specs == (
        LoadedModelComponentSpec(key="text_encoder1", public_name="clip_l", roles=("text_encoder",)),
        LoadedModelComponentSpec(key="text_encoder2", public_name="clip_g", roles=("text_encoder",)),
        LoadedModelComponentSpec(key="vae", public_name="vae", roles=("vae",)),
        LoadedModelComponentSpec(key="denoiser", public_name="unet", roles=("denoiser",)),
    )


def test_resolve_component_specs_returns_sd3_component_order_and_identity():
    component_specs = resolve_component_specs("sd3")

    assert component_specs == (
        LoadedModelComponentSpec(key="text_encoder1", public_name="clip_l", roles=("text_encoder",)),
        LoadedModelComponentSpec(key="text_encoder2", public_name="clip_g", roles=("text_encoder",)),
        LoadedModelComponentSpec(key="text_encoder3", public_name="t5xxl", roles=("text_encoder",)),
        LoadedModelComponentSpec(key="vae", public_name="vae", roles=("vae",)),
        LoadedModelComponentSpec(key="denoiser", public_name="mmdit", roles=("denoiser",)),
    )


def test_resolve_component_names_derives_legacy_names_from_declared_specs():
    component_names = resolve_component_names("sd3")

    assert component_names is not None
    assert component_names.text_encoder_names == ("clip_l", "clip_g", "t5xxl")
    assert component_names.vae_name == "vae"
    assert component_names.denoiser_name == "mmdit"


def test_resolve_component_specs_returns_none_for_unknown_model_type():
    assert resolve_component_specs("definitely_not_a_model_family") is None


def test_resolve_component_specs_returns_none_for_malformed_package_metadata(monkeypatch):
    fake_package = SimpleNamespace(
        LOADED_MODEL_COMPONENT_SPECS=(
            LoadedModelComponentSpec(key="text_encoder1", public_name="clip_l", roles=("text_encoder",)),
            "not-a-component-spec",
        )
    )
    monkeypatch.setattr(component_metadata.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(component_metadata.importlib, "import_module", lambda _: fake_package)

    assert resolve_component_specs("sdxl") is None
