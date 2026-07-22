"""Characterization tests for the active model-loading result contract."""

import importlib
import sys

from types import SimpleNamespace
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
import torch

from library.constants import MODEL_VERSION_SDXL_BASE_V1_0


def _strategy_type(monkeypatch, module_name: str, class_name: str):
    """Import a loading strategy without initializing optional CUDA RamTorch."""
    ramtorch = ModuleType("ramtorch")
    helpers = ModuleType("ramtorch.helpers")
    helpers.replace_linear_with_ramtorch = None
    ramtorch.helpers = helpers
    monkeypatch.setitem(sys.modules, "ramtorch", ramtorch)
    monkeypatch.setitem(sys.modules, "ramtorch.helpers", helpers)
    return getattr(importlib.import_module(module_name), class_name)


def _loading_cfg(model_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(model_type=model_type),
        objective=SimpleNamespace(prediction=object()),
        data=SimpleNamespace(caching=object()),
        performance=SimpleNamespace(
            memory=SimpleNamespace(use_ramtorch=False),
            attention=SimpleNamespace(
                mem_eff_attn=False,
                xformers=False,
                sdpa=True,
            ),
            precision=object(),
        ),
    )


def _component_surface(components) -> list[tuple[str, str, bool]]:
    return [(component.key, component.public_name, component.module is not None) for component in components]


@pytest.mark.unit
def test_sd_loading_returns_current_family_declared_component_surface(monkeypatch) -> None:
    strategy_type = _strategy_type(
        monkeypatch,
        "library.strategies.sd.loading",
        "SdModelLoadingStrategy",
    )
    cfg = _loading_cfg("sd15")
    accelerator = SimpleNamespace(device=torch.device("cpu"))
    text_encoder = object()
    vae = MagicMock()
    denoiser = object()

    with (
        patch(
            "library.strategies.sd.loading.load_target_model",
            return_value=(text_encoder, vae, denoiser, object()),
        ) as load_model,
        patch("library.strategies.sd.loading.replace_unet_modules"),
        patch(
            "library.strategies.sd.loading.resolve_ddpm_prediction_type",
            return_value="epsilon",
        ),
        patch(
            "library.strategies.sd.loading.library.models.sd.conversion.get_model_version_str_for_sd1_sd2",
            return_value="sd_v1",
        ),
    ):
        result = strategy_type().load_target_model(
            cfg,
            torch.float16,
            accelerator,
        )

    model_version, components = result
    assert isinstance(result, tuple)
    assert isinstance(components, tuple)
    assert model_version == "sd_v1"
    assert _component_surface(components) == [
        ("text_encoder1", "clip_l", True),
        ("vae", "vae", True),
        ("denoiser", "unet", True),
    ]
    assert [component.module for component in components] == [
        text_encoder,
        vae,
        denoiser,
    ]
    load_model.assert_called_once_with(
        cfg.model,
        cfg.performance.memory,
        torch.float16,
        accelerator,
    )


@pytest.mark.unit
def test_sdxl_loading_returns_current_surface_and_retains_loader_side_state(
    monkeypatch,
) -> None:
    strategy_type = _strategy_type(
        monkeypatch,
        "library.strategies.sdxl.loading",
        "SdxlModelLoadingStrategy",
    )
    cfg = _loading_cfg("sdxl")
    accelerator = SimpleNamespace(device=torch.device("cpu"))
    text_encoder1 = object()
    text_encoder2 = object()
    vae = MagicMock()
    denoiser = object()
    checkpoint_info = object()
    logit_scale = object()
    strategy = strategy_type()

    with (
        patch(
            "library.strategies.sdxl.loading.load_sdxl_target_model",
            return_value=(
                True,
                text_encoder1,
                text_encoder2,
                vae,
                denoiser,
                logit_scale,
                checkpoint_info,
            ),
        ) as load_model,
        patch("library.strategies.sdxl.loading.replace_unet_modules"),
    ):
        result = strategy.load_target_model(cfg, torch.float16, accelerator)

    model_version, components = result
    assert isinstance(result, tuple)
    assert isinstance(components, tuple)
    assert model_version == MODEL_VERSION_SDXL_BASE_V1_0
    assert _component_surface(components) == [
        ("text_encoder1", "clip_l", True),
        ("text_encoder2", "clip_g", True),
        ("vae", "vae", True),
        ("denoiser", "unet", True),
    ]
    assert [component.module for component in components] == [
        text_encoder1,
        text_encoder2,
        vae,
        denoiser,
    ]
    assert strategy.load_stable_diffusion_format is True
    assert strategy.logit_scale is logit_scale
    assert strategy.ckpt_info is checkpoint_info
    load_model.assert_called_once_with(
        cfg.model,
        cfg.performance.memory,
        cfg.data.caching,
        cfg.performance.precision,
        accelerator,
        MODEL_VERSION_SDXL_BASE_V1_0,
        torch.float16,
    )


@pytest.mark.unit
def test_sd3_loading_preserves_declared_absent_and_deferred_components(
    monkeypatch,
) -> None:
    strategy_type = _strategy_type(
        monkeypatch,
        "library.strategies.sd3.loading",
        "Sd3ModelLoadingStrategy",
    )
    cfg = _loading_cfg("sd3")
    accelerator = SimpleNamespace(device=torch.device("cpu"))
    text_encoder1 = object()
    text_encoder2 = object()
    vae = object()
    strategy = strategy_type()

    with patch(
        "library.strategies.sd3.loading.load_sd3_target_model",
        return_value=(
            "sd3_medium",
            [text_encoder1, text_encoder2, None],
            vae,
            None,
        ),
    ) as load_model:
        result = strategy.load_target_model(cfg, torch.float16, accelerator)

    model_version, components = result
    assert isinstance(result, tuple)
    assert isinstance(components, tuple)
    assert model_version == "sd3_medium"
    assert _component_surface(components) == [
        ("text_encoder1", "clip_l", True),
        ("text_encoder2", "clip_g", True),
        ("text_encoder3", "t5xxl", False),
        ("vae", "vae", True),
        ("denoiser", "mmdit", False),
    ]
    assert [component.module for component in components] == [
        text_encoder1,
        text_encoder2,
        None,
        vae,
        None,
    ]
    assert strategy._model_version == "sd3_medium"
    load_model.assert_called_once_with(
        cfg.model,
        cfg.performance.memory,
        cfg.data.caching,
        cfg.performance.precision,
        accelerator,
        torch.float16,
        resolutions=None,
    )
