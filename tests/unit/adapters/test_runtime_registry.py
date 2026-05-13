import sys
import tempfile
import types
from pathlib import Path

import pytest
from safetensors import safe_open
import torch
from safetensors.torch import load_file, save_file

from library.adapters import (
    AdapterExportLoadRequest,
    AdapterExportSaveRequest,
    AdapterBuildContext,
    AdapterBuildRequest,
    AdapterModelContext as RuntimeAdapterModelContext,
    AdapterRuntimeSpec,
    LoadedAdapterRuntime,
    build_component_module_targets as runtime_build_component_module_targets,
    build_component_root_targets as runtime_build_component_root_targets,
    build_adapter_for_legacy_module,
    build_adapter_from_weights_for_legacy_module,
    get_adapter_method,
    get_adapter_method_for_legacy_module,
    load_adapter_export,
    list_adapter_methods,
    save_adapter_export,
)
from library.adapters.runtime import AdapterMergeRequest
from library.models import build_loaded_components
from library.strategies.base.context import (
    DenoiserContext,
    StrategyContext,
    StrategyPhase,
    TrainingContext,
    publish_strategy_context,
)


def _build_sdxl_loaded_components(*, text_encoders=None, vae=None, denoiser=None):
    if text_encoders is None:
        text_encoder_modules = []
    elif isinstance(text_encoders, list):
        text_encoder_modules = list(text_encoders)
    else:
        text_encoder_modules = [text_encoders]
    text_encoder_modules.extend([None] * max(0, 2 - len(text_encoder_modules)))

    return build_loaded_components(
        "sdxl",
        {
            "text_encoder1": text_encoder_modules[0],
            "text_encoder2": text_encoder_modules[1],
            "vae": vae,
            "denoiser": denoiser,
        },
    )


def AdapterModelContext(*, loaded_components=None, vae=None, text_encoder=None, denoiser=None):
    if loaded_components is None:
        loaded_components = _build_sdxl_loaded_components(text_encoders=text_encoder, vae=vae, denoiser=denoiser)
    return RuntimeAdapterModelContext(loaded_components=tuple(loaded_components))


def build_component_root_targets(*, model_type, loaded_components=None, text_encoders=None, vae=None, denoiser=None, **kwargs):
    if loaded_components is None:
        loaded_components = _build_sdxl_loaded_components(text_encoders=text_encoders, vae=vae, denoiser=denoiser)
    return runtime_build_component_root_targets(model_type=model_type, loaded_components=loaded_components, **kwargs)


def build_component_module_targets(*, model_type, loaded_components=None, text_encoders=None, vae=None, denoiser=None, **kwargs):
    if loaded_components is None:
        loaded_components = _build_sdxl_loaded_components(text_encoders=text_encoders, vae=vae, denoiser=denoiser)
    return runtime_build_component_module_targets(model_type=model_type, loaded_components=loaded_components, **kwargs)


class TestAdapterRegistry:
    def test_builds_component_root_targets_with_public_component_names(self):
        clip_l = object()
        clip_g = object()
        unet = object()

        resolved_targets = build_component_root_targets(
            model_type="sdxl",
            text_encoders=[clip_l, clip_g],
            vae=None,
            denoiser=unet,
            include_text_encoders=[True, False],
            include_denoiser=True,
        )

        assert [target.component for target in resolved_targets.targets] == ["clip_l", "unet"]
        assert resolved_targets.targets[0].component_key == "text_encoder1"
        assert resolved_targets.targets[1].component_key == "denoiser"
        assert resolved_targets.targets[0].metadata["component_key"] == "text_encoder1"
        assert resolved_targets.targets[1].metadata["component_key"] == "denoiser"

    def test_builds_component_module_targets_with_component_qualified_paths(self):
        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4)
                self.norm = torch.nn.LayerNorm(4)

        class DummyDenoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(4, 4)
                self.conv = torch.nn.Conv2d(4, 4, kernel_size=1)

        resolved_targets = build_component_module_targets(
            model_type="sdxl",
            text_encoders=[DummyTextEncoder(), None],
            vae=None,
            denoiser=DummyDenoiser(),
            include_text_encoders=[True, False],
            include_denoiser=True,
        )

        assert [target.component for target in resolved_targets.targets] == ["clip_l", "clip_l", "unet", "unet"]
        assert [target.component_key for target in resolved_targets.targets] == [
            "text_encoder1",
            "text_encoder1",
            "denoiser",
            "denoiser",
        ]
        assert [target.path for target in resolved_targets.targets] == [
            "clip_l.proj",
            "clip_l.norm",
            "unet.to_q",
            "unet.conv",
        ]
        assert [target.local_path for target in resolved_targets.targets] == [
            "proj",
            "norm",
            "to_q",
            "conv",
        ]

    def test_registered_runtime_filters_text_encoder_context_to_resolved_targets(self, monkeypatch):
        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4)
                self.norm = torch.nn.LayerNorm(4)

        class DummyDenoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(4, 4)
                self.conv = torch.nn.Conv2d(4, 4, kernel_size=3, padding=1)

        clip_l = DummyTextEncoder()
        clip_g = DummyTextEncoder()
        unet = DummyDenoiser()

        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="lora",
                settings={"adapter_rank": 8, "adapter_alpha": 16.0, "conv_dim": 4, "conv_alpha": 8.0, "neuron_dropout": 0.1},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae="vae", text_encoder=[clip_l, clip_g], denoiser=unet),
            ),
            resolved_targets=build_component_module_targets(
                model_type="sdxl",
                text_encoders=[clip_l, clip_g],
                vae=None,
                denoiser=unet,
                include_text_encoders=[False, True],
                include_denoiser=True,
            ),
        )

        adapter = build_adapter_for_legacy_module("library.adapters.lora", request)
        refs = adapter.describe_trainable_parameter_refs()

        assert adapter.adapter_resolved_targets is request.resolved_targets
        assert [module.lora_name for module in adapter.lora_modules] == [
            "lora_te2_proj",
            "lora_unet_to_q",
            "lora_unet_conv",
        ]
        assert [ref.target_path for ref in refs] == [
            "clip_g.proj",
            "clip_g.proj",
            "unet.to_q",
            "unet.to_q",
            "unet.conv",
            "unet.conv",
        ]
        assert all(ref.source_target_ref is not None for ref in refs)

    def test_registered_runtime_filters_weight_build_text_encoder_context(self, monkeypatch):
        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4, bias=False)

        clip_l = DummyTextEncoder()
        clip_g = DummyTextEncoder()
        unet = torch.nn.Linear(4, 4, bias=False)
        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="lora",
                settings={"adapter_rank": 2, "adapter_alpha": 4.0},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae="vae", text_encoder=[clip_l, clip_g], denoiser=unet),
                for_inference=False,
            ),
            resolved_targets=build_component_module_targets(
                model_type="sdxl",
                text_encoders=[clip_l, clip_g],
                vae=None,
                denoiser=unet,
                include_text_encoders=[False, True],
                include_denoiser=True,
            ),
        )

        adapter = build_adapter_for_legacy_module("library.adapters.lora", request)
        with torch.no_grad():
            for module in adapter.lora_modules:
                module.lora_down.weight.fill_(0.25)
                module.lora_up.weight.fill_(0.125)

        with tempfile.TemporaryDirectory() as temp_dir:
            weights_path = Path(temp_dir) / "adapter.safetensors"
            adapter.save_weights(str(weights_path), dtype=None, metadata=None)
            loaded_runtime = build_adapter_from_weights_for_legacy_module(
                "library.adapters.lora",
                request,
                str(weights_path),
            )

        assert loaded_runtime.adapter.adapter_resolved_targets is request.resolved_targets
        loaded_modules = loaded_runtime.adapter.lora_modules
        assert [module.lora_name for module in loaded_modules] == ["lora_te2_proj", "lora_unet"]
        assert torch.allclose(loaded_modules[0].lora_down.weight, torch.full_like(loaded_modules[0].lora_down.weight, 0.25))
        assert torch.allclose(loaded_modules[0].lora_up.weight, torch.full_like(loaded_modules[0].lora_up.weight, 0.125))

    def test_registered_runtime_exposes_repo_owned_trainable_refs(self, monkeypatch):
        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(2, 2, bias=False)

        class DummyDenoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(2, 2, bias=False)

        clip_l = DummyTextEncoder()
        unet = DummyDenoiser()
        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="lora",
                settings={"adapter_rank": 8, "adapter_alpha": 16.0},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[clip_l], denoiser=unet),
            ),
            resolved_targets=build_component_module_targets(
                model_type="sd15",
                text_encoders=[clip_l],
                vae=None,
                denoiser=unet,
                include_text_encoders=[True],
                include_denoiser=True,
            ),
        )

        adapter = build_adapter_for_legacy_module("library.adapters.lora", request)
        refs = adapter.describe_trainable_parameter_refs()

        assert [ref.component for ref in refs] == ["clip_l", "clip_l", "unet", "unet"]
        assert [ref.name for ref in refs] == [
            "lora_te_proj.lora_down.weight",
            "lora_te_proj.lora_up.weight",
            "lora_unet_to_q.lora_down.weight",
            "lora_unet_to_q.lora_up.weight",
        ]
        assert refs[2].component_key == "denoiser"
        assert refs[0].target_path == "clip_l.proj"
        assert all(ref.source_target_ref is not None for ref in refs)

    def test_registered_vera_runtime_exposes_repo_owned_trainable_refs(self):
        from library.adapters.methods.peft.vera.module import VeraModule

        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4, bias=False)
                self.norm = torch.nn.LayerNorm(4)

        class DummyDenoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(4, 4, bias=False)

        text_encoder = DummyTextEncoder()
        denoiser = DummyDenoiser()
        resolved_targets = build_component_module_targets(
            model_type="sdxl",
            text_encoders=[text_encoder, None],
            vae=None,
            denoiser=denoiser,
            include_text_encoders=[True, False],
            include_denoiser=True,
        )
        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="vera",
                settings={"adapter_rank": 4, "dropout": 0.1, "projection_prng_key": 7, "save_projection": True},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=denoiser),
            ),
            resolved_targets=resolved_targets,
        )

        adapter = build_adapter_for_legacy_module("library.adapters.vera", request)
        refs = adapter.describe_trainable_parameter_refs()

        assert adapter.adapter_resolved_targets is resolved_targets
        assert all(isinstance(module, VeraModule) for module in adapter.vera_modules)
        assert {module.__class__.__module__ for module in adapter.vera_modules} == {"library.adapters.methods.peft.vera.module"}
        assert {module.adapter_target.path for module in adapter.vera_modules} == {"clip_l.proj", "unet.to_q"}
        assert {ref.component for ref in refs} == {"clip_l", "unet"}
        assert {ref.component_key for ref in refs} == {"text_encoder1", "denoiser"}
        assert {ref.target_path for ref in refs} == {"clip_l.proj", "unet.to_q"}
        assert {ref.source_target_ref.selector for ref in refs if ref.source_target_ref is not None} == {"clip_l.proj", "unet.to_q"}
        assert {ref.source_target_ref.module_type for ref in refs if ref.source_target_ref is not None} == {"Linear"}
        assert {ref.name.split(".")[-1] for ref in refs} == {"vera_lambda_b", "vera_lambda_d"}
        assert all("norm" not in ref.target_path for ref in refs)

    def test_lists_builtin_adapter_types(self):
        registrations = list_adapter_methods()

        assert [registration.name for registration in registrations] == [
            "abba",
            "boft",
            "dylora",
            "glora",
            "ia3",
            "loha",
            "locon",
            "lokr",
            "lora",
            "oft",
            "tlora",
            "vera",
        ]

    def test_resolves_repo_owned_abba_adapter_type(self):
        registration = get_adapter_method("abba")

        assert registration.legacy_module_path == "library.adapters.abba"
        assert registration.runtime_module_path == "library.adapters.methods.peft.abba.runtime"
        assert registration.config_binding is not None
        assert registration.config_binding.config_key == "abba"
        assert registration.config_binding.runtime_settings_builder is not None

    def test_resolves_repo_owned_adapter_type(self):
        registration = get_adapter_method("loha")

        assert registration.legacy_module_path == "library.adapters.loha"
        assert registration.runtime_module_path == "library.adapters.methods.peft.loha.runtime"
        assert registration.config_binding is not None
        assert registration.config_binding.config_key == "loha"
        assert registration.config_binding.runtime_settings_builder is not None

    def test_resolves_repo_owned_boft_adapter_type(self):
        registration = get_adapter_method("boft")

        assert registration.legacy_module_path == "library.adapters.boft"
        assert registration.runtime_module_path == "library.adapters.methods.peft.boft.runtime"
        assert registration.config_binding is not None
        assert registration.config_binding.config_key == "boft"
        assert registration.config_binding.runtime_settings_builder is not None

    def test_resolves_repo_owned_dylora_adapter_type(self):
        registration = get_adapter_method("dylora")

        assert registration.legacy_module_path == "library.adapters.dylora"
        assert registration.runtime_module_path == "library.adapters.methods.peft.dylora.runtime"
        assert registration.config_binding is not None
        assert registration.config_binding.config_key == "dylora"
        assert registration.config_binding.runtime_settings_builder is not None

    def test_resolves_repo_owned_glora_adapter_type(self):
        registration = get_adapter_method("glora")

        assert registration.legacy_module_path == "library.adapters.glora"
        assert registration.runtime_module_path == "library.adapters.methods.peft.glora.runtime"
        assert registration.config_binding is not None
        assert registration.config_binding.config_key == "glora"
        assert registration.config_binding.runtime_settings_builder is not None

    def test_resolves_repo_owned_ia3_adapter_type(self):
        registration = get_adapter_method("ia3")

        assert registration.legacy_module_path == "library.adapters.ia3"
        assert registration.runtime_module_path == "library.adapters.methods.peft.ia3.runtime"
        assert registration.config_binding is not None
        assert registration.config_binding.config_key == "ia3"
        assert registration.config_binding.runtime_settings_builder is not None

    def test_resolves_repo_owned_tlora_adapter_type(self):
        registration = get_adapter_method("tlora")

        assert registration.legacy_module_path == "library.adapters.tlora"
        assert registration.runtime_module_path == "library.adapters.methods.peft.tlora.runtime"
        assert registration.config_binding is not None
        assert registration.config_binding.config_key == "tlora"
        assert registration.config_binding.runtime_settings_builder is not None

    def test_resolves_repo_owned_vera_adapter_type(self):
        registration = get_adapter_method("vera")

        assert registration.legacy_module_path == "library.adapters.vera"
        assert registration.runtime_module_path == "library.adapters.methods.peft.vera.runtime"
        assert registration.config_binding is not None
        assert registration.config_binding.config_key == "vera"
        assert registration.config_binding.runtime_settings_builder is not None

    def test_resolves_repo_owned_lokr_adapter_type(self):
        registration = get_adapter_method("lokr")

        assert registration.legacy_module_path == "library.adapters.lokr"
        assert registration.runtime_module_path == "library.adapters.methods.peft.lokr.runtime"
        assert registration.config_binding is not None
        assert registration.config_binding.config_key == "lokr"
        assert registration.config_binding.runtime_settings_builder is not None

    def test_resolves_repo_owned_locon_adapter_type(self):
        registration = get_adapter_method("locon")

        assert registration.legacy_module_path == "library.adapters.locon"
        assert registration.runtime_module_path == "library.adapters.methods.peft.locon.runtime"
        assert registration.config_binding is not None
        assert registration.config_binding.config_key == "locon"
        assert registration.config_binding.runtime_settings_builder is not None

    def test_resolves_repo_owned_oft_adapter_type(self):
        registration = get_adapter_method("oft")

        assert registration.legacy_module_path == "library.adapters.oft"
        assert registration.runtime_module_path == "library.adapters.methods.peft.oft.runtime"
        assert registration.config_binding is not None
        assert registration.config_binding.config_key == "oft"
        assert registration.config_binding.runtime_settings_builder is not None

    def test_registration_owns_method_config_translation(self):
        from library.adapters.methods.peft.loha.config import PeftLohaConfig

        registration = get_adapter_method("loha")
        assert registration.config_binding is not None
        settings = registration.config_binding.runtime_settings_builder(
            PeftLohaConfig(
                rank=16,
                alpha=32.0,
                init_mode="zero_delta_he",
                rank_dropout=0.2,
                use_tucker=True,
            )
        )

        assert registration.config_binding.config_key == "loha"
        assert settings["adapter_rank"] == 16
        assert settings["adapter_alpha"] == 32.0
        assert settings["init_mode"] == "zero_delta_he"
        assert settings["rank_dropout"] == 0.2
        assert settings["use_tucker"] is True

    def test_lokr_registration_owns_method_config_translation(self):
        from library.adapters.methods.peft.lokr.config import PeftLokrConfig

        registration = get_adapter_method("lokr")
        assert registration.config_binding is not None
        settings = registration.config_binding.runtime_settings_builder(
            PeftLokrConfig(
                rank=16,
                alpha=32.0,
                init_mode="zero_delta_he",
                rank_dropout=0.2,
                use_tucker=True,
                decompose_both=True,
                factor=8,
            )
        )

        assert registration.config_binding.config_key == "lokr"
        assert settings["adapter_rank"] == 16
        assert settings["adapter_alpha"] == 32.0
        assert settings["init_mode"] == "zero_delta_he"
        assert settings["rank_dropout"] == 0.2
        assert settings["use_tucker"] is True
        assert settings["decompose_both"] is True
        assert settings["factor"] == 8

    def test_locon_registration_owns_method_config_translation(self):
        from library.adapters.methods.peft.locon.config import PeftLoconConfig

        registration = get_adapter_method("locon")
        assert registration.config_binding is not None
        settings = registration.config_binding.runtime_settings_builder(
            PeftLoconConfig(
                rank=16,
                alpha=32.0,
                dropout=0.15,
                init_mode="zero_delta_he",
                rank_dropout=0.2,
                use_tucker=True,
                orthogonalize=True,
            )
        )

        assert registration.config_binding.config_key == "locon"
        assert settings["adapter_rank"] == 16
        assert settings["adapter_alpha"] == 32.0
        assert settings["dropout"] == 0.15
        assert settings["init_mode"] == "zero_delta_he"
        assert settings["rank_dropout"] == 0.2
        assert settings["use_tucker"] is True
        assert settings["orthogonalize"] is True

    def test_oft_registration_owns_method_config_translation(self):
        from library.adapters.methods.peft.oft.config import PeftOftConfig

        registration = get_adapter_method("oft")
        assert registration.config_binding is not None
        settings = registration.config_binding.runtime_settings_builder(
            PeftOftConfig(
                factor=8,
                constraint=0.25,
                rescaled=True,
                dropout=0.15,
                bypass_mode=True,
            )
        )

        assert registration.config_binding.config_key == "oft"
        assert settings["factor"] == 8
        assert settings["constraint"] == 0.25
        assert settings["rescaled"] is True
        assert settings["dropout"] == 0.15
        assert settings["bypass_mode"] is True

    def test_boft_registration_owns_method_config_translation(self):
        from library.adapters.methods.peft.boft.config import PeftBoftConfig

        registration = get_adapter_method("boft")
        assert registration.config_binding is not None
        settings = registration.config_binding.runtime_settings_builder(
            PeftBoftConfig(
                factor=8,
                constraint=0.25,
                num_stages=2,
                rescaled=True,
                dropout=0.15,
                bypass_mode=True,
            )
        )

        assert registration.config_binding.config_key == "boft"
        assert settings["factor"] == 8
        assert settings["constraint"] == 0.25
        assert settings["num_stages"] == 2
        assert settings["rescaled"] is True
        assert settings["dropout"] == 0.15
        assert settings["bypass_mode"] is True

    def test_dylora_registration_owns_method_config_translation(self):
        from library.adapters.methods.peft.dylora.config import PeftDyloraConfig

        registration = get_adapter_method("dylora")
        assert registration.config_binding is not None
        settings = registration.config_binding.runtime_settings_builder(
            PeftDyloraConfig(
                rank=16,
                alpha=32.0,
                block_size=4,
                module_dropout=0.2,
                bypass_mode=True,
            )
        )

        assert registration.config_binding.config_key == "dylora"
        assert settings["adapter_rank"] == 16
        assert settings["adapter_alpha"] == 32.0
        assert settings["block_size"] == 4
        assert settings["module_dropout"] == 0.2
        assert settings["bypass_mode"] is True

    def test_glora_registration_owns_method_config_translation(self):
        from library.adapters.methods.peft.glora.config import PeftGloraConfig

        registration = get_adapter_method("glora")
        assert registration.config_binding is not None
        settings = registration.config_binding.runtime_settings_builder(
            PeftGloraConfig(
                rank=16,
                alpha=32.0,
                dropout=0.15,
                rank_dropout=0.2,
                use_tucker=True,
                orthogonalize=True,
                bypass_mode=True,
            )
        )

        assert registration.config_binding.config_key == "glora"
        assert settings["adapter_rank"] == 16
        assert settings["adapter_alpha"] == 32.0
        assert settings["dropout"] == 0.15
        assert settings["rank_dropout"] == 0.2
        assert settings["use_tucker"] is True
        assert settings["orthogonalize"] is True
        assert settings["bypass_mode"] is True

    def test_ia3_registration_owns_method_config_translation(self):
        from library.adapters.methods.peft.ia3.config import PeftIa3Config

        registration = get_adapter_method("ia3")
        assert registration.config_binding is not None
        settings = registration.config_binding.runtime_settings_builder(
            PeftIa3Config(
                train_on_input=True,
                module_dropout=0.2,
                bypass_mode=True,
            )
        )

        assert registration.config_binding.config_key == "ia3"
        assert settings["train_on_input"] is True
        assert settings["module_dropout"] == 0.2
        assert settings["bypass_mode"] is True

    def test_abba_registration_owns_method_config_translation(self):
        from library.adapters.methods.peft.abba.config import PeftAbbaConfig

        registration = get_adapter_method("abba")
        assert registration.config_binding is not None
        settings = registration.config_binding.runtime_settings_builder(
            PeftAbbaConfig(
                rank=8,
                alpha=32.0,
                dropout=0.15,
                rank_dropout=0.2,
                module_dropout=0.3,
                use_scalar=True,
                weight_decompose=True,
            )
        )

        assert registration.config_binding.config_key == "abba"
        assert settings["adapter_rank"] == 8
        assert settings["adapter_alpha"] == 32.0
        assert settings["dropout"] == 0.15
        assert settings["rank_dropout"] == 0.2
        assert settings["module_dropout"] == 0.3
        assert settings["use_scalar"] is True
        assert settings["weight_decompose"] is True

    def test_tlora_registration_owns_method_config_translation(self):
        from library.adapters.methods.peft.tlora.config import PeftTloraConfig

        registration = get_adapter_method("tlora")
        assert registration.config_binding is not None
        settings = registration.config_binding.runtime_settings_builder(
            PeftTloraConfig(
                rank=8,
                alpha=8.0,
                dropout=0.15,
                module_dropout=0.3,
                use_scalar=True,
                sig_type="middle",
                use_data_init=False,
                min_rank=2,
                mask_alpha=1.5,
            )
        )

        assert registration.config_binding.config_key == "tlora"
        assert settings["adapter_rank"] == 8
        assert settings["adapter_alpha"] == 8.0
        assert settings["dropout"] == 0.15
        assert settings["module_dropout"] == 0.3
        assert settings["use_scalar"] is True
        assert settings["sig_type"] == "middle"
        assert settings["use_data_init"] is False
        assert settings["mask_min_rank"] == 2
        assert settings["mask_alpha"] == 1.5

    def test_lora_registration_owns_method_config_translation(self):
        from library.adapters.methods.peft.lora.config import PeftLoraConfig

        registration = get_adapter_method("lora")
        assert registration.config_binding is not None
        settings = registration.config_binding.runtime_settings_builder(
            PeftLoraConfig(
                rank=16,
                alpha=32.0,
                dropout=0.15,
                conv_rank=8,
                conv_alpha=12.0,
                rank_dropout=0.2,
                module_dropout=0.1,
            )
        )

        assert registration.config_binding.config_key == "lora"
        assert settings["adapter_rank"] == 16
        assert settings["adapter_alpha"] == 32.0
        assert settings["neuron_dropout"] == 0.15
        assert settings["conv_dim"] == 8
        assert settings["conv_alpha"] == 12.0
        assert settings["rank_dropout"] == 0.2
        assert settings["module_dropout"] == 0.1

    def test_vera_registration_owns_method_config_translation(self):
        from library.adapters.methods.peft.vera.config import PeftVeraConfig

        registration = get_adapter_method("vera")
        assert registration.config_binding is not None
        settings = registration.config_binding.runtime_settings_builder(
            PeftVeraConfig(
                rank=16,
                dropout=0.15,
                d_initial=0.25,
                projection_prng_key=1234,
            )
        )

        assert registration.config_binding.config_key == "vera"
        assert settings["adapter_rank"] == 16
        assert settings["dropout"] == 0.15
        assert settings["d_initial"] == 0.25
        assert settings["projection_prng_key"] == 1234
        assert settings["save_projection"] is True

    def test_loha_translation_requires_explicit_rank(self):
        from library.adapters.methods.peft.loha.config import PeftLohaConfig

        registration = get_adapter_method("loha")
        assert registration.config_binding is not None

        with pytest.raises(ValueError, match="adapter\\.peft\\.loha\\.rank must be set"):
            registration.config_binding.runtime_settings_builder(PeftLohaConfig())

    def test_loha_translation_rejects_plain_dropout(self):
        from library.adapters.methods.peft.loha.config import PeftLohaConfig

        registration = get_adapter_method("loha")
        assert registration.config_binding is not None

        with pytest.raises(ValueError, match="adapter\\.peft\\.loha\\.dropout is not supported"):
            registration.config_binding.runtime_settings_builder(PeftLohaConfig(rank=8, dropout=0.1))

    def test_loha_translation_rejects_unknown_init_mode(self):
        from library.adapters.methods.peft.loha.config import PeftLohaConfig

        registration = get_adapter_method("loha")
        assert registration.config_binding is not None

        with pytest.raises(ValueError, match="adapter\\.peft\\.loha\\.init_mode must be one of"):
            registration.config_binding.runtime_settings_builder(PeftLohaConfig(rank=8, init_mode="mystery_mode"))  # type: ignore[arg-type]

    def test_lokr_translation_requires_explicit_rank(self):
        from library.adapters.methods.peft.lokr.config import PeftLokrConfig

        registration = get_adapter_method("lokr")
        assert registration.config_binding is not None

        with pytest.raises(ValueError, match="adapter\\.peft\\.lokr\\.rank must be set"):
            registration.config_binding.runtime_settings_builder(PeftLokrConfig())

    def test_lokr_translation_rejects_plain_dropout(self):
        from library.adapters.methods.peft.lokr.config import PeftLokrConfig

        registration = get_adapter_method("lokr")
        assert registration.config_binding is not None

        with pytest.raises(ValueError, match="adapter\\.peft\\.lokr\\.dropout is not supported"):
            registration.config_binding.runtime_settings_builder(PeftLokrConfig(rank=8, dropout=0.1))

    def test_locon_translation_requires_explicit_rank(self):
        from library.adapters.methods.peft.locon.config import PeftLoconConfig

        registration = get_adapter_method("locon")
        assert registration.config_binding is not None

        with pytest.raises(ValueError, match="adapter\\.peft\\.locon\\.rank must be set"):
            registration.config_binding.runtime_settings_builder(PeftLoconConfig())

    def test_locon_translation_rejects_unknown_init_mode(self):
        from library.adapters.methods.peft.locon.config import PeftLoconConfig

        registration = get_adapter_method("locon")
        assert registration.config_binding is not None

        with pytest.raises(ValueError, match="adapter\\.peft\\.locon\\.init_mode must be one of"):
            registration.config_binding.runtime_settings_builder(PeftLoconConfig(rank=8, init_mode="mystery_mode"))  # type: ignore[arg-type]

    def test_oft_translation_requires_explicit_factor(self):
        from library.adapters.methods.peft.oft.config import PeftOftConfig

        registration = get_adapter_method("oft")
        assert registration.config_binding is not None

        with pytest.raises(ValueError, match="adapter\\.peft\\.oft\\.factor must be set"):
            registration.config_binding.runtime_settings_builder(PeftOftConfig())

    def test_oft_translation_rejects_out_of_range_dropout(self):
        from library.adapters.methods.peft.oft.config import PeftOftConfig

        registration = get_adapter_method("oft")
        assert registration.config_binding is not None

        with pytest.raises(ValueError, match="adapter\\.peft\\.oft\\.dropout must be between 0.0 and 1.0 inclusive"):
            registration.config_binding.runtime_settings_builder(PeftOftConfig(factor=4, dropout=1.5))

    def test_boft_translation_requires_explicit_factor(self):
        from library.adapters.methods.peft.boft.config import PeftBoftConfig

        registration = get_adapter_method("boft")
        assert registration.config_binding is not None

        with pytest.raises(ValueError, match="adapter\\.peft\\.boft\\.factor must be set"):
            registration.config_binding.runtime_settings_builder(PeftBoftConfig())

    def test_boft_translation_rejects_out_of_range_dropout(self):
        from library.adapters.methods.peft.boft.config import PeftBoftConfig

        registration = get_adapter_method("boft")
        assert registration.config_binding is not None

        with pytest.raises(ValueError, match="adapter\\.peft\\.boft\\.dropout must be between 0.0 and 1.0 inclusive"):
            registration.config_binding.runtime_settings_builder(PeftBoftConfig(factor=4, dropout=1.5))

    def test_boft_translation_rejects_non_positive_num_stages(self):
        from library.adapters.methods.peft.boft.config import PeftBoftConfig

        registration = get_adapter_method("boft")
        assert registration.config_binding is not None

        with pytest.raises(ValueError, match="adapter\\.peft\\.boft\\.num_stages must be a positive integer when set"):
            registration.config_binding.runtime_settings_builder(PeftBoftConfig(factor=4, num_stages=0))

    def test_dylora_translation_requires_explicit_rank(self):
        from library.adapters.methods.peft.dylora.config import PeftDyloraConfig

        registration = get_adapter_method("dylora")
        assert registration.config_binding is not None

        with pytest.raises(ValueError, match="adapter\\.peft\\.dylora\\.rank must be set"):
            registration.config_binding.runtime_settings_builder(PeftDyloraConfig())

    def test_dylora_translation_rejects_block_size_that_does_not_divide_rank(self):
        from library.adapters.methods.peft.dylora.config import PeftDyloraConfig

        registration = get_adapter_method("dylora")
        assert registration.config_binding is not None

        with pytest.raises(ValueError, match="adapter\\.peft\\.dylora\\.block_size must divide adapter\\.peft\\.dylora\\.rank exactly"):
            registration.config_binding.runtime_settings_builder(PeftDyloraConfig(rank=8, block_size=3))

    def test_glora_translation_requires_explicit_rank(self):
        from library.adapters.methods.peft.glora.config import PeftGloraConfig

        registration = get_adapter_method("glora")
        assert registration.config_binding is not None

        with pytest.raises(ValueError, match="adapter\\.peft\\.glora\\.rank must be set"):
            registration.config_binding.runtime_settings_builder(PeftGloraConfig())

    def test_glora_translation_rejects_out_of_range_dropout(self):
        from library.adapters.methods.peft.glora.config import PeftGloraConfig

        registration = get_adapter_method("glora")
        assert registration.config_binding is not None

        with pytest.raises(ValueError, match="adapter\\.peft\\.glora\\.dropout must be between 0.0 and 1.0 inclusive"):
            registration.config_binding.runtime_settings_builder(PeftGloraConfig(rank=8, dropout=1.5))

    def test_ia3_translation_rejects_out_of_range_module_dropout(self):
        from library.adapters.methods.peft.ia3.config import PeftIa3Config

        registration = get_adapter_method("ia3")
        assert registration.config_binding is not None

        with pytest.raises(ValueError, match="adapter\\.peft\\.ia3\\.module_dropout must be between 0.0 and 1.0 inclusive"):
            registration.config_binding.runtime_settings_builder(PeftIa3Config(module_dropout=1.5))

    def test_abba_translation_requires_rank_of_at_least_two(self):
        from library.adapters.methods.peft.abba.config import PeftAbbaConfig

        registration = get_adapter_method("abba")
        assert registration.config_binding is not None

        with pytest.raises(ValueError, match="adapter\\.peft\\.abba\\.rank must be set to an integer greater than or equal to 2"):
            registration.config_binding.runtime_settings_builder(PeftAbbaConfig())
        with pytest.raises(ValueError, match="adapter\\.peft\\.abba\\.rank must be greater than or equal to 2 when set"):
            registration.config_binding.runtime_settings_builder(PeftAbbaConfig(rank=1))

    def test_abba_translation_rejects_bypass_mode_with_weight_decompose(self):
        from library.adapters.methods.peft.abba.config import PeftAbbaConfig

        registration = get_adapter_method("abba")
        assert registration.config_binding is not None

        with pytest.raises(
            ValueError,
            match="adapter\\.peft\\.abba\\.bypass_mode cannot be enabled when adapter\\.peft\\.abba\\.weight_decompose is true",
        ):
            registration.config_binding.runtime_settings_builder(
                PeftAbbaConfig(rank=4, weight_decompose=True, bypass_mode=True)
            )

    def test_abba_translation_rejects_out_of_range_dropout(self):
        from library.adapters.methods.peft.abba.config import PeftAbbaConfig

        registration = get_adapter_method("abba")
        assert registration.config_binding is not None

        with pytest.raises(ValueError, match="adapter\\.peft\\.abba\\.dropout must be between 0.0 and 1.0 inclusive"):
            registration.config_binding.runtime_settings_builder(PeftAbbaConfig(rank=4, dropout=1.5))

    def test_tlora_translation_requires_positive_rank(self):
        from library.adapters.methods.peft.tlora.config import PeftTloraConfig

        registration = get_adapter_method("tlora")
        assert registration.config_binding is not None

        with pytest.raises(ValueError, match="adapter\\.peft\\.tlora\\.rank must be set to a positive integer"):
            registration.config_binding.runtime_settings_builder(PeftTloraConfig())
        with pytest.raises(ValueError, match="adapter\\.peft\\.tlora\\.rank must be a positive integer when set"):
            registration.config_binding.runtime_settings_builder(PeftTloraConfig(rank=0))

    def test_tlora_translation_rejects_invalid_mask_bounds(self):
        from library.adapters.methods.peft.tlora.config import PeftTloraConfig

        registration = get_adapter_method("tlora")
        assert registration.config_binding is not None

        with pytest.raises(ValueError, match="adapter\\.peft\\.tlora\\.min_rank cannot exceed adapter\\.peft\\.tlora\\.rank"):
            registration.config_binding.runtime_settings_builder(PeftTloraConfig(rank=4, min_rank=5))

    def test_tlora_translation_rejects_out_of_range_dropout(self):
        from library.adapters.methods.peft.tlora.config import PeftTloraConfig

        registration = get_adapter_method("tlora")
        assert registration.config_binding is not None

        with pytest.raises(ValueError, match="adapter\\.peft\\.tlora\\.dropout must be between 0.0 and 1.0 inclusive"):
            registration.config_binding.runtime_settings_builder(PeftTloraConfig(rank=4, dropout=1.5))

    def test_vera_translation_requires_positive_rank(self):
        from library.adapters.methods.peft.vera.config import PeftVeraConfig

        registration = get_adapter_method("vera")
        assert registration.config_binding is not None

        with pytest.raises(ValueError, match="adapter\\.peft\\.vera\\.rank must be set to a positive integer"):
            registration.config_binding.runtime_settings_builder(PeftVeraConfig())
        with pytest.raises(ValueError, match="adapter\\.peft\\.vera\\.rank must be a positive integer when set"):
            registration.config_binding.runtime_settings_builder(PeftVeraConfig(rank=0))

    def test_vera_translation_rejects_out_of_range_dropout(self):
        from library.adapters.methods.peft.vera.config import PeftVeraConfig

        registration = get_adapter_method("vera")
        assert registration.config_binding is not None

        with pytest.raises(ValueError, match="adapter\\.peft\\.vera\\.dropout must be between 0.0 and 1.0 inclusive"):
            registration.config_binding.runtime_settings_builder(PeftVeraConfig(rank=8, dropout=1.5))

    def test_vera_translation_allows_unsaved_shared_projections(self):
        from library.adapters.methods.peft.vera.config import PeftVeraConfig

        registration = get_adapter_method("vera")
        assert registration.config_binding is not None

        settings = registration.config_binding.runtime_settings_builder(PeftVeraConfig(rank=8, save_projection=False))
        assert settings["save_projection"] is False

    def test_resolves_abba_legacy_module_path(self):
        registration = get_adapter_method_for_legacy_module("library.adapters.abba")

        assert registration.name == "abba"

    def test_resolves_legacy_module_path(self):
        registration = get_adapter_method_for_legacy_module("library.adapters.oft")

        assert registration.name == "oft"

    def test_resolves_boft_legacy_module_path(self):
        registration = get_adapter_method_for_legacy_module("library.adapters.boft")

        assert registration.name == "boft"

    def test_resolves_dylora_legacy_module_path(self):
        registration = get_adapter_method_for_legacy_module("library.adapters.dylora")

        assert registration.name == "dylora"

    def test_resolves_glora_legacy_module_path(self):
        registration = get_adapter_method_for_legacy_module("library.adapters.glora")

        assert registration.name == "glora"

    def test_resolves_ia3_legacy_module_path(self):
        registration = get_adapter_method_for_legacy_module("library.adapters.ia3")

        assert registration.name == "ia3"

    def test_resolves_tlora_legacy_module_path(self):
        registration = get_adapter_method_for_legacy_module("library.adapters.tlora")

        assert registration.name == "tlora"

    def test_resolves_vera_legacy_module_path(self):
        registration = get_adapter_method_for_legacy_module("library.adapters.vera")

        assert registration.name == "vera"

    def test_build_adapter_for_legacy_module_uses_registered_wrapper(self, monkeypatch):
        captured = {}
        runtime_module = types.SimpleNamespace()

        def fake_create_adapter(request):
            captured["request"] = request
            return "adapter-runtime"

        runtime_module.create_adapter = fake_create_adapter
        monkeypatch.setitem(sys.modules, "library.adapters.methods.peft.lora.runtime", runtime_module)

        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(adapter_type="lora"),
            context=AdapterBuildContext(model=AdapterModelContext(vae=None, text_encoder=[], denoiser=None)),
            resolved_targets=build_component_root_targets(
                model_type="sdxl",
                text_encoders=[],
                vae=None,
                denoiser=None,
            ),
        )
        result = build_adapter_for_legacy_module("library.adapters.lora", request)

        assert result == "adapter-runtime"
        assert captured["request"] is request

    def test_build_adapter_from_weights_for_legacy_module_uses_registered_wrapper(self, monkeypatch):
        captured = {}
        runtime_module = types.SimpleNamespace()

        def fake_create_adapter_from_weights(request, weights_path):
            captured["request"] = request
            captured["weights_path"] = weights_path
            return "adapter-from-weights", None

        runtime_module.create_adapter_from_weights = fake_create_adapter_from_weights
        monkeypatch.setitem(sys.modules, "library.adapters.methods.peft.dylora.runtime", runtime_module)

        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(adapter_type="dylora"),
            context=AdapterBuildContext(model=AdapterModelContext(vae=None, text_encoder=[], denoiser=None)),
            resolved_targets=build_component_root_targets(
                model_type="sdxl",
                text_encoders=[],
                vae=None,
                denoiser=None,
            ),
        )
        result = build_adapter_from_weights_for_legacy_module("library.adapters.dylora", request, "weights.safetensors")

        assert result.adapter == "adapter-from-weights"
        assert result.state is None
        assert captured["request"] is request
        assert captured["weights_path"] == "weights.safetensors"

    def test_build_adapter_from_weights_for_legacy_module_rejects_invalid_wrapper_result(self, monkeypatch):
        runtime_module = types.SimpleNamespace()
        runtime_module.create_adapter_from_weights = lambda request, weights_path: "invalid-shape"
        monkeypatch.setitem(sys.modules, "library.adapters.methods.peft.dylora.runtime", runtime_module)

        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(adapter_type="dylora"),
            context=AdapterBuildContext(model=AdapterModelContext(vae=None, text_encoder=[], denoiser=None)),
            resolved_targets=build_component_root_targets(
                model_type="sdxl",
                text_encoders=[],
                vae=None,
                denoiser=None,
            ),
        )

        with pytest.raises(TypeError, match="two-item"):
            build_adapter_from_weights_for_legacy_module("library.adapters.dylora", request, "weights.safetensors")

    def test_loaded_adapter_runtime_uses_repo_owned_merge_request(self):
        captured = {}

        loaded_runtime = LoadedAdapterRuntime(
            adapter="merge-adapter",
            state={"weights": "state"},
            _merge_into_impl=lambda request: captured.update(
                {
                    "model": request.model,
                    "resolved_targets": request.resolved_targets,
                    "dtype": request.dtype,
                    "device": request.device,
                }
            ),
        )
        resolved_targets = build_component_root_targets(
            model_type="sdxl",
            text_encoders=[],
            vae=None,
            denoiser="denoiser",
            include_denoiser=True,
        )

        loaded_runtime.merge_into(
            AdapterMergeRequest(
                model=AdapterModelContext(vae=None, text_encoder=["text-encoder"], denoiser="denoiser"),
                resolved_targets=resolved_targets,
                dtype="fp16",
                device="cpu",
            )
        )

        assert captured["model"].module_or_modules_by_role("text_encoder") == "text-encoder"
        assert captured["model"].require_module_by_role("denoiser") == "denoiser"
        assert captured["resolved_targets"] is resolved_targets
        assert captured["dtype"] == "fp16"
        assert captured["device"] == "cpu"

    def test_registered_vera_runtime_round_trips_export_and_merge(self, tmp_path):
        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4, bias=False)

        class DummyDenoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(4, 4, bias=False)

        text_encoder = DummyTextEncoder()
        denoiser = DummyDenoiser()
        resolved_targets = build_component_module_targets(
            model_type="sdxl",
            text_encoders=[text_encoder, None],
            vae=None,
            denoiser=denoiser,
            include_text_encoders=[True, False],
            include_denoiser=True,
        )
        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="vera",
                settings={"adapter_rank": 4, "dropout": 0.1, "projection_prng_key": 7, "save_projection": True},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=denoiser),
            ),
            resolved_targets=resolved_targets,
        )

        adapter = build_adapter_for_legacy_module("library.adapters.vera", request)
        for parameter in adapter.parameters():
            parameter.data.fill_(0.25)

        export_path = tmp_path / "vera.safetensors"
        save_adapter_export(
            adapter,
            AdapterExportSaveRequest(file=str(export_path), dtype=torch.float32, metadata={"format": "test"}),
        )

        fresh_adapter = build_adapter_for_legacy_module("library.adapters.vera", request)
        load_info = load_adapter_export(fresh_adapter, AdapterExportLoadRequest(file=str(export_path)))
        assert load_info == {}

        original_weight = denoiser.to_q.weight.detach().clone()
        loaded_runtime = build_adapter_from_weights_for_legacy_module("library.adapters.vera", request, str(export_path))
        assert isinstance(loaded_runtime, LoadedAdapterRuntime)
        loaded_runtime.merge_into(
            AdapterMergeRequest(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=denoiser),
                resolved_targets=resolved_targets,
                dtype=torch.float32,
                device="cpu",
            )
        )

        assert not torch.allclose(denoiser.to_q.weight, original_weight)

    def test_registered_vera_runtime_round_trips_without_saved_projections(self, tmp_path):
        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4, bias=False)

        class DummyDenoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(4, 4, bias=False)

        text_encoder = DummyTextEncoder()
        denoiser = DummyDenoiser()
        resolved_targets = build_component_module_targets(
            model_type="sdxl",
            text_encoders=[text_encoder, None],
            vae=None,
            denoiser=denoiser,
            include_text_encoders=[True, False],
            include_denoiser=True,
        )
        save_request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="vera",
                settings={"adapter_rank": 4, "dropout": 0.1, "projection_prng_key": 7, "save_projection": False},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=denoiser),
            ),
            resolved_targets=resolved_targets,
        )
        load_request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="vera",
                settings={"adapter_rank": 4, "dropout": 0.1, "projection_prng_key": 123, "save_projection": False},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=denoiser),
            ),
            resolved_targets=resolved_targets,
        )
        weights_request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="vera",
                settings={"adapter_rank": 9, "dropout": 0.1, "projection_prng_key": 123, "save_projection": False},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=denoiser),
            ),
            resolved_targets=resolved_targets,
        )

        adapter = build_adapter_for_legacy_module("library.adapters.vera", save_request)
        for parameter in adapter.parameters():
            parameter.data.fill_(0.25)
        source_vera_A = adapter.shared_bank.vera_A.detach().clone()
        source_vera_B = adapter.shared_bank.vera_B.detach().clone()

        export_path = tmp_path / "vera-unsaved.safetensors"
        save_adapter_export(
            adapter,
            AdapterExportSaveRequest(file=str(export_path), dtype=torch.float32, metadata={"format": "test"}),
        )

        exported_state = load_file(str(export_path))
        assert "vera_shared.vera_A" not in exported_state
        assert "vera_shared.vera_B" not in exported_state
        with safe_open(str(export_path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        assert metadata["sd_scripts_vera.save_projection"] == "false"
        assert metadata["sd_scripts_vera.projection_prng_key"] == "7"

        fresh_adapter = build_adapter_for_legacy_module("library.adapters.vera", load_request)
        assert fresh_adapter.shared_bank.projection_prng_key == 123
        load_info = load_adapter_export(fresh_adapter, AdapterExportLoadRequest(file=str(export_path)))
        assert load_info == {}
        assert fresh_adapter.shared_bank.projection_prng_key == 7
        assert torch.allclose(fresh_adapter.shared_bank.vera_A, source_vera_A)
        assert torch.allclose(fresh_adapter.shared_bank.vera_B, source_vera_B)

        original_weight = denoiser.to_q.weight.detach().clone()
        loaded_runtime = build_adapter_from_weights_for_legacy_module("library.adapters.vera", weights_request, str(export_path))
        assert isinstance(loaded_runtime, LoadedAdapterRuntime)
        assert loaded_runtime.adapter.shared_bank.projection_prng_key == 7
        assert loaded_runtime.adapter.shared_bank.rank == 4
        loaded_runtime.merge_into(
            AdapterMergeRequest(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=denoiser),
                resolved_targets=resolved_targets,
                dtype=torch.float32,
                device="cpu",
            )
        )

        assert not torch.allclose(denoiser.to_q.weight, original_weight)

    def test_builds_component_module_targets_with_transformers_conv1d_when_available(self):
        transformers_pytorch_utils = pytest.importorskip("transformers.pytorch_utils")
        Conv1D = transformers_pytorch_utils.Conv1D

        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = Conv1D(4, 4)

        resolved_targets = build_component_module_targets(
            model_type="sd15",
            text_encoders=[DummyTextEncoder()],
            vae=None,
            denoiser=None,
            include_text_encoders=[True],
            include_denoiser=False,
        )

        assert [target.path for target in resolved_targets.targets] == ["clip_l.proj"]
        assert [target.module_type for target in resolved_targets.targets] == ["Conv1D"]

    def test_registered_loha_runtime_exposes_repo_owned_trainable_refs(self):
        from library.adapters.methods.peft.loha.module import LohaModule

        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4, bias=False)
                self.norm = torch.nn.LayerNorm(4)

        class DummyDenoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(4, 4, bias=False)

        text_encoder = DummyTextEncoder()
        denoiser = DummyDenoiser()
        resolved_targets = build_component_module_targets(
            model_type="sdxl",
            text_encoders=[text_encoder, None],
            vae=None,
            denoiser=denoiser,
            include_text_encoders=[True, False],
            include_denoiser=True,
        )
        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="loha",
                settings={"adapter_rank": 4, "adapter_alpha": 8.0},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=denoiser),
            ),
            resolved_targets=resolved_targets,
        )

        adapter = build_adapter_for_legacy_module("library.adapters.loha", request)
        refs = adapter.describe_trainable_parameter_refs()

        assert adapter.adapter_resolved_targets is resolved_targets
        assert all(isinstance(module, LohaModule) for module in adapter.loha_modules)
        assert {module.__class__.__module__ for module in adapter.loha_modules} == {"library.adapters.methods.peft.loha.module"}
        assert {module.adapter_target.path for module in adapter.loha_modules} == {"clip_l.proj", "unet.to_q"}
        assert {ref.component for ref in refs} == {"clip_l", "unet"}
        assert {ref.component_key for ref in refs} == {"text_encoder1", "denoiser"}
        assert {ref.target_path for ref in refs} == {"clip_l.proj", "unet.to_q"}
        assert {ref.source_target_ref.selector for ref in refs if ref.source_target_ref is not None} == {"clip_l.proj", "unet.to_q"}
        assert {ref.source_target_ref.module_type for ref in refs if ref.source_target_ref is not None} == {"Linear"}
        assert all("norm" not in ref.target_path for ref in refs)

    def test_registered_loha_runtime_round_trips_export_and_merge(self, tmp_path):
        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4, bias=False)

        class DummyDenoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(4, 4, bias=False)

        text_encoder = DummyTextEncoder()
        denoiser = DummyDenoiser()
        resolved_targets = build_component_module_targets(
            model_type="sdxl",
            text_encoders=[text_encoder, None],
            vae=None,
            denoiser=denoiser,
            include_text_encoders=[True, False],
            include_denoiser=True,
        )
        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="loha",
                settings={"adapter_rank": 4, "adapter_alpha": 8.0},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=denoiser),
            ),
            resolved_targets=resolved_targets,
        )

        adapter = build_adapter_for_legacy_module("library.adapters.loha", request)
        for parameter in adapter.parameters():
            parameter.data.fill_(0.25)

        export_path = tmp_path / "loha.safetensors"
        save_adapter_export(
            adapter,
            AdapterExportSaveRequest(file=str(export_path), dtype=torch.float32, metadata={"format": "test"}),
        )

        fresh_adapter = build_adapter_for_legacy_module("library.adapters.loha", request)
        load_info = load_adapter_export(fresh_adapter, AdapterExportLoadRequest(file=str(export_path)))
        assert load_info == {}

        original_weight = denoiser.to_q.weight.detach().clone()
        loaded_runtime = build_adapter_from_weights_for_legacy_module("library.adapters.loha", request, str(export_path))
        assert isinstance(loaded_runtime, LoadedAdapterRuntime)
        loaded_runtime.merge_into(
            AdapterMergeRequest(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=denoiser),
                resolved_targets=resolved_targets,
                dtype=torch.float32,
                device="cpu",
            )
        )

        assert not torch.allclose(denoiser.to_q.weight, original_weight)

    def test_registered_loha_runtime_reports_partial_checkpoint_as_missing(self, tmp_path):
        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4, bias=False)

        class DummyDenoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(4, 4, bias=False)

        text_encoder = DummyTextEncoder()
        denoiser = DummyDenoiser()
        resolved_targets = build_component_module_targets(
            model_type="sdxl",
            text_encoders=[text_encoder, None],
            vae=None,
            denoiser=denoiser,
            include_text_encoders=[True, False],
            include_denoiser=True,
        )
        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="loha",
                settings={"adapter_rank": 4, "adapter_alpha": 8.0},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=denoiser),
            ),
            resolved_targets=resolved_targets,
        )

        adapter = build_adapter_for_legacy_module("library.adapters.loha", request)
        export_path = tmp_path / "loha_partial.safetensors"
        save_adapter_export(
            adapter,
            AdapterExportSaveRequest(file=str(export_path), dtype=torch.float32, metadata={"format": "test"}),
        )
        state_dict = dict(load_file(str(export_path)))
        state_dict.pop("loha_clip_l_proj.alpha")
        partial_path = tmp_path / "loha_partial_missing_alpha.safetensors"
        save_file(state_dict, str(partial_path), {"format": "test"})

        fresh_adapter = build_adapter_for_legacy_module("library.adapters.loha", request)
        load_info = load_adapter_export(fresh_adapter, AdapterExportLoadRequest(file=str(partial_path)))

        assert load_info == {"missing keys": ["loha_clip_l_proj"]}

    def test_registered_loha_runtime_rejects_plain_dropout_settings(self):
        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4, bias=False)

        class DummyDenoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(4, 4, bias=False)

        text_encoder = DummyTextEncoder()
        denoiser = DummyDenoiser()
        resolved_targets = build_component_module_targets(
            model_type="sdxl",
            text_encoders=[text_encoder, None],
            vae=None,
            denoiser=denoiser,
            include_text_encoders=[True, False],
            include_denoiser=True,
        )
        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="loha",
                settings={"adapter_rank": 4, "adapter_alpha": 8.0, "dropout": 0.1},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=denoiser),
            ),
            resolved_targets=resolved_targets,
        )

        with pytest.raises(ValueError, match="LoHa plain dropout is disabled"):
            build_adapter_for_legacy_module("library.adapters.loha", request)

    def test_registered_lokr_runtime_exposes_repo_owned_trainable_refs(self):
        from library.adapters.methods.peft.lokr.module import LokrModule

        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4, bias=False)
                self.norm = torch.nn.LayerNorm(4)

        class DummyDenoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(4, 4, bias=False)

        text_encoder = DummyTextEncoder()
        denoiser = DummyDenoiser()
        resolved_targets = build_component_module_targets(
            model_type="sdxl",
            text_encoders=[text_encoder, None],
            vae=None,
            denoiser=denoiser,
            include_text_encoders=[True, False],
            include_denoiser=True,
        )
        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="lokr",
                settings={"adapter_rank": 4, "adapter_alpha": 8.0},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=denoiser),
            ),
            resolved_targets=resolved_targets,
        )

        adapter = build_adapter_for_legacy_module("library.adapters.lokr", request)
        refs = adapter.describe_trainable_parameter_refs()

        assert adapter.adapter_resolved_targets is resolved_targets
        assert all(isinstance(module, LokrModule) for module in adapter.lokr_modules)
        assert {module.__class__.__module__ for module in adapter.lokr_modules} == {"library.adapters.methods.peft.lokr.module"}
        assert {module.adapter_target.path for module in adapter.lokr_modules} == {"clip_l.proj", "unet.to_q"}
        assert {ref.component for ref in refs} == {"clip_l", "unet"}
        assert {ref.component_key for ref in refs} == {"text_encoder1", "denoiser"}
        assert {ref.target_path for ref in refs} == {"clip_l.proj", "unet.to_q"}
        assert {ref.source_target_ref.selector for ref in refs if ref.source_target_ref is not None} == {"clip_l.proj", "unet.to_q"}
        assert {ref.source_target_ref.module_type for ref in refs if ref.source_target_ref is not None} == {"Linear"}
        assert all("norm" not in ref.target_path for ref in refs)

    def test_registered_lokr_runtime_round_trips_export_and_merge(self, tmp_path):
        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4, bias=False)

        class DummyDenoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(4, 4, bias=False)

        text_encoder = DummyTextEncoder()
        denoiser = DummyDenoiser()
        resolved_targets = build_component_module_targets(
            model_type="sdxl",
            text_encoders=[text_encoder, None],
            vae=None,
            denoiser=denoiser,
            include_text_encoders=[True, False],
            include_denoiser=True,
        )
        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="lokr",
                settings={"adapter_rank": 4, "adapter_alpha": 8.0},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=denoiser),
            ),
            resolved_targets=resolved_targets,
        )

        adapter = build_adapter_for_legacy_module("library.adapters.lokr", request)
        for parameter in adapter.parameters():
            parameter.data.fill_(0.25)

        export_path = tmp_path / "lokr.safetensors"
        save_adapter_export(
            adapter,
            AdapterExportSaveRequest(file=str(export_path), dtype=torch.float32, metadata={"format": "test"}),
        )

        fresh_adapter = build_adapter_for_legacy_module("library.adapters.lokr", request)
        load_info = load_adapter_export(fresh_adapter, AdapterExportLoadRequest(file=str(export_path)))
        assert load_info == {}

        original_weight = denoiser.to_q.weight.detach().clone()
        loaded_runtime = build_adapter_from_weights_for_legacy_module("library.adapters.lokr", request, str(export_path))
        assert isinstance(loaded_runtime, LoadedAdapterRuntime)
        loaded_runtime.merge_into(
            AdapterMergeRequest(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=denoiser),
                resolved_targets=resolved_targets,
                dtype=torch.float32,
                device="cpu",
            )
        )

        assert not torch.allclose(denoiser.to_q.weight, original_weight)

    def test_registered_lokr_runtime_reports_partial_checkpoint_as_missing(self, tmp_path):
        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4, bias=False)

        text_encoder = DummyTextEncoder()
        resolved_targets = build_component_module_targets(
            model_type="sdxl",
            text_encoders=[text_encoder, None],
            vae=None,
            denoiser=None,
            include_text_encoders=[True, False],
            include_denoiser=False,
        )
        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="lokr",
                settings={"adapter_rank": 4, "adapter_alpha": 8.0},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=None),
            ),
            resolved_targets=resolved_targets,
        )

        adapter = build_adapter_for_legacy_module("library.adapters.lokr", request)
        export_path = tmp_path / "lokr_partial.safetensors"
        save_adapter_export(
            adapter,
            AdapterExportSaveRequest(file=str(export_path), dtype=torch.float32, metadata={"format": "test"}),
        )
        state_dict = dict(load_file(str(export_path)))
        state_dict.pop("lokr_clip_l_proj.alpha")
        partial_path = tmp_path / "lokr_partial_missing_alpha.safetensors"
        save_file(state_dict, str(partial_path), {"format": "test"})

        fresh_adapter = build_adapter_for_legacy_module("library.adapters.lokr", request)
        load_info = load_adapter_export(fresh_adapter, AdapterExportLoadRequest(file=str(partial_path)))

        assert load_info == {"missing keys": ["lokr_clip_l_proj"]}

    def test_registered_lokr_runtime_rejects_plain_dropout_settings(self):
        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4, bias=False)

        text_encoder = DummyTextEncoder()
        resolved_targets = build_component_module_targets(
            model_type="sdxl",
            text_encoders=[text_encoder, None],
            vae=None,
            denoiser=None,
            include_text_encoders=[True, False],
            include_denoiser=False,
        )
        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="lokr",
                settings={"adapter_rank": 4, "adapter_alpha": 8.0, "dropout": 0.1},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=None),
            ),
            resolved_targets=resolved_targets,
        )

        with pytest.raises(ValueError, match="LoKr plain dropout is disabled"):
            build_adapter_for_legacy_module("library.adapters.lokr", request)

    def test_registered_locon_runtime_exposes_repo_owned_trainable_refs(self):
        from library.adapters.methods.peft.locon.module import LoconModule

        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4, bias=False)
                self.norm = torch.nn.LayerNorm(4)

        class DummyDenoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(4, 4, bias=False)

        text_encoder = DummyTextEncoder()
        denoiser = DummyDenoiser()
        resolved_targets = build_component_module_targets(
            model_type="sdxl",
            text_encoders=[text_encoder, None],
            vae=None,
            denoiser=denoiser,
            include_text_encoders=[True, False],
            include_denoiser=True,
        )
        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="locon",
                settings={"adapter_rank": 4, "adapter_alpha": 8.0},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=denoiser),
            ),
            resolved_targets=resolved_targets,
        )

        adapter = build_adapter_for_legacy_module("library.adapters.locon", request)
        refs = adapter.describe_trainable_parameter_refs()

        assert adapter.adapter_resolved_targets is resolved_targets
        assert all(isinstance(module, LoconModule) for module in adapter.locon_modules)
        assert {module.__class__.__module__ for module in adapter.locon_modules} == {"library.adapters.methods.peft.locon.module"}
        assert {module.adapter_target.path for module in adapter.locon_modules} == {"clip_l.proj", "unet.to_q"}
        assert {ref.component for ref in refs} == {"clip_l", "unet"}
        assert {ref.component_key for ref in refs} == {"text_encoder1", "denoiser"}
        assert {ref.target_path for ref in refs} == {"clip_l.proj", "unet.to_q"}
        assert {ref.source_target_ref.selector for ref in refs if ref.source_target_ref is not None} == {"clip_l.proj", "unet.to_q"}
        assert {ref.source_target_ref.module_type for ref in refs if ref.source_target_ref is not None} == {"Linear"}
        assert all("norm" not in ref.target_path for ref in refs)

    def test_registered_locon_runtime_round_trips_export_and_merge(self, tmp_path):
        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4, bias=False)

        class DummyDenoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(4, 4, bias=False)

        text_encoder = DummyTextEncoder()
        denoiser = DummyDenoiser()
        resolved_targets = build_component_module_targets(
            model_type="sdxl",
            text_encoders=[text_encoder, None],
            vae=None,
            denoiser=denoiser,
            include_text_encoders=[True, False],
            include_denoiser=True,
        )
        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="locon",
                settings={"adapter_rank": 4, "adapter_alpha": 8.0, "dropout": 0.1},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=denoiser),
            ),
            resolved_targets=resolved_targets,
        )

        adapter = build_adapter_for_legacy_module("library.adapters.locon", request)
        for parameter in adapter.parameters():
            parameter.data.fill_(0.25)

        export_path = tmp_path / "locon.safetensors"
        save_adapter_export(
            adapter,
            AdapterExportSaveRequest(file=str(export_path), dtype=torch.float32, metadata={"format": "test"}),
        )

        fresh_adapter = build_adapter_for_legacy_module("library.adapters.locon", request)
        load_info = load_adapter_export(fresh_adapter, AdapterExportLoadRequest(file=str(export_path)))
        assert load_info == {}

        original_weight = denoiser.to_q.weight.detach().clone()
        loaded_runtime = build_adapter_from_weights_for_legacy_module("library.adapters.locon", request, str(export_path))
        assert isinstance(loaded_runtime, LoadedAdapterRuntime)
        loaded_runtime.merge_into(
            AdapterMergeRequest(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=denoiser),
                resolved_targets=resolved_targets,
                dtype=torch.float32,
                device="cpu",
            )
        )

        assert not torch.allclose(denoiser.to_q.weight, original_weight)

    def test_registered_locon_runtime_reports_partial_checkpoint_as_missing(self, tmp_path):
        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4, bias=False)

        text_encoder = DummyTextEncoder()
        resolved_targets = build_component_module_targets(
            model_type="sdxl",
            text_encoders=[text_encoder, None],
            vae=None,
            denoiser=None,
            include_text_encoders=[True, False],
            include_denoiser=False,
        )
        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="locon",
                settings={"adapter_rank": 4, "adapter_alpha": 8.0},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=None),
            ),
            resolved_targets=resolved_targets,
        )

        adapter = build_adapter_for_legacy_module("library.adapters.locon", request)
        export_path = tmp_path / "locon_partial.safetensors"
        save_adapter_export(
            adapter,
            AdapterExportSaveRequest(file=str(export_path), dtype=torch.float32, metadata={"format": "test"}),
        )
        state_dict = dict(load_file(str(export_path)))
        state_dict.pop("locon_clip_l_proj.alpha")
        partial_path = tmp_path / "locon_partial_missing_alpha.safetensors"
        save_file(state_dict, str(partial_path), {"format": "test"})

        fresh_adapter = build_adapter_for_legacy_module("library.adapters.locon", request)
        load_info = load_adapter_export(fresh_adapter, AdapterExportLoadRequest(file=str(partial_path)))

        assert load_info == {"missing keys": ["locon_clip_l_proj"]}

    def test_registered_oft_runtime_exposes_repo_owned_trainable_refs(self):
        from library.adapters.methods.peft.oft.module import OftModule

        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4, bias=False)
                self.norm = torch.nn.LayerNorm(4)

        class DummyDenoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(4, 4, bias=False)

        text_encoder = DummyTextEncoder()
        denoiser = DummyDenoiser()
        resolved_targets = build_component_module_targets(
            model_type="sdxl",
            text_encoders=[text_encoder, None],
            vae=None,
            denoiser=denoiser,
            include_text_encoders=[True, False],
            include_denoiser=True,
        )
        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="oft",
                settings={"factor": 4, "constraint": 0.25},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=denoiser),
            ),
            resolved_targets=resolved_targets,
        )

        adapter = build_adapter_for_legacy_module("library.adapters.oft", request)
        refs = adapter.describe_trainable_parameter_refs()

        assert adapter.adapter_resolved_targets is resolved_targets
        assert all(isinstance(module, OftModule) for module in adapter.oft_modules)
        assert {module.__class__.__module__ for module in adapter.oft_modules} == {"library.adapters.methods.peft.oft.module"}
        assert {module.adapter_target.path for module in adapter.oft_modules} == {"clip_l.proj", "unet.to_q"}
        assert {ref.component for ref in refs} == {"clip_l", "unet"}
        assert {ref.component_key for ref in refs} == {"text_encoder1", "denoiser"}
        assert {ref.target_path for ref in refs} == {"clip_l.proj", "unet.to_q"}
        assert {ref.source_target_ref.selector for ref in refs if ref.source_target_ref is not None} == {"clip_l.proj", "unet.to_q"}
        assert {ref.source_target_ref.module_type for ref in refs if ref.source_target_ref is not None} == {"Linear"}
        assert all("norm" not in ref.target_path for ref in refs)

    def test_registered_oft_runtime_round_trips_export_and_merge(self, tmp_path):
        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4, bias=False)

        class DummyDenoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(4, 4, bias=True)

        text_encoder = DummyTextEncoder()
        denoiser = DummyDenoiser()
        resolved_targets = build_component_module_targets(
            model_type="sdxl",
            text_encoders=[text_encoder, None],
            vae=None,
            denoiser=denoiser,
            include_text_encoders=[True, False],
            include_denoiser=True,
        )
        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="oft",
                settings={"factor": 4, "constraint": 0.25, "rescaled": True},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=denoiser),
            ),
            resolved_targets=resolved_targets,
        )

        adapter = build_adapter_for_legacy_module("library.adapters.oft", request)
        for parameter in adapter.parameters():
            parameter.data.fill_(0.25)

        export_path = tmp_path / "oft.safetensors"
        save_adapter_export(
            adapter,
            AdapterExportSaveRequest(file=str(export_path), dtype=torch.float32, metadata={"format": "test"}),
        )

        fresh_adapter = build_adapter_for_legacy_module("library.adapters.oft", request)
        load_info = load_adapter_export(fresh_adapter, AdapterExportLoadRequest(file=str(export_path)))
        assert load_info == {}

        original_weight = denoiser.to_q.weight.detach().clone()
        original_bias = denoiser.to_q.bias.detach().clone()
        loaded_runtime = build_adapter_from_weights_for_legacy_module("library.adapters.oft", request, str(export_path))
        assert isinstance(loaded_runtime, LoadedAdapterRuntime)
        loaded_runtime.merge_into(
            AdapterMergeRequest(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=denoiser),
                resolved_targets=resolved_targets,
                dtype=torch.float32,
                device="cpu",
            )
        )

        assert not torch.allclose(denoiser.to_q.weight, original_weight)
        assert not torch.allclose(denoiser.to_q.bias, original_bias)

    def test_registered_oft_runtime_reports_partial_checkpoint_as_missing(self, tmp_path):
        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4, bias=False)

        text_encoder = DummyTextEncoder()
        resolved_targets = build_component_module_targets(
            model_type="sdxl",
            text_encoders=[text_encoder, None],
            vae=None,
            denoiser=None,
            include_text_encoders=[True, False],
            include_denoiser=False,
        )
        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="oft",
                settings={"factor": 4, "constraint": 0.25},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=None),
            ),
            resolved_targets=resolved_targets,
        )

        adapter = build_adapter_for_legacy_module("library.adapters.oft", request)
        export_path = tmp_path / "oft_partial.safetensors"
        save_adapter_export(
            adapter,
            AdapterExportSaveRequest(file=str(export_path), dtype=torch.float32, metadata={"format": "test"}),
        )
        state_dict = dict(load_file(str(export_path)))
        state_dict.pop("oft_clip_l_proj.alpha")
        partial_path = tmp_path / "oft_partial_missing_alpha.safetensors"
        save_file(state_dict, str(partial_path), {"format": "test"})

        fresh_adapter = build_adapter_for_legacy_module("library.adapters.oft", request)
        load_info = load_adapter_export(fresh_adapter, AdapterExportLoadRequest(file=str(partial_path)))

        assert load_info == {"missing keys": ["oft_clip_l_proj"]}

    def test_registered_boft_runtime_exposes_repo_owned_trainable_refs(self):
        from library.adapters.methods.peft.boft.module import BoftModule

        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4, bias=False)
                self.norm = torch.nn.LayerNorm(4)

        class DummyDenoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(4, 4, bias=False)

        text_encoder = DummyTextEncoder()
        denoiser = DummyDenoiser()
        resolved_targets = build_component_module_targets(
            model_type="sdxl",
            text_encoders=[text_encoder, None],
            vae=None,
            denoiser=denoiser,
            include_text_encoders=[True, False],
            include_denoiser=True,
        )
        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="boft",
                settings={"factor": 2, "constraint": 0.25},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=denoiser),
            ),
            resolved_targets=resolved_targets,
        )

        adapter = build_adapter_for_legacy_module("library.adapters.boft", request)
        refs = adapter.describe_trainable_parameter_refs()

        assert adapter.adapter_resolved_targets is resolved_targets
        assert all(isinstance(module, BoftModule) for module in adapter.boft_modules)
        assert {module.__class__.__module__ for module in adapter.boft_modules} == {"library.adapters.methods.peft.boft.module"}
        assert {module.adapter_target.path for module in adapter.boft_modules} == {"clip_l.proj", "unet.to_q"}
        assert {ref.component for ref in refs} == {"clip_l", "unet"}
        assert {ref.component_key for ref in refs} == {"text_encoder1", "denoiser"}
        assert {ref.target_path for ref in refs} == {"clip_l.proj", "unet.to_q"}
        assert {ref.source_target_ref.selector for ref in refs if ref.source_target_ref is not None} == {"clip_l.proj", "unet.to_q"}
        assert {ref.source_target_ref.module_type for ref in refs if ref.source_target_ref is not None} == {"Linear"}
        assert all("norm" not in ref.target_path for ref in refs)

    def test_registered_boft_runtime_round_trips_export_and_merge(self, tmp_path):
        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(8, 8, bias=False)

        class DummyDenoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(8, 8, bias=False)

        text_encoder = DummyTextEncoder()
        denoiser = DummyDenoiser()
        resolved_targets = build_component_module_targets(
            model_type="sdxl",
            text_encoders=[text_encoder, None],
            vae=None,
            denoiser=denoiser,
            include_text_encoders=[True, False],
            include_denoiser=True,
        )
        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="boft",
                settings={"factor": 2, "constraint": 0.25, "num_stages": 1, "rescaled": True},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=denoiser),
            ),
            resolved_targets=resolved_targets,
        )

        adapter = build_adapter_for_legacy_module("library.adapters.boft", request)
        assert all(module.boft_m == 1 for module in adapter.boft_modules)
        for parameter in adapter.parameters():
            parameter.data.fill_(0.25)

        export_path = tmp_path / "boft.safetensors"
        save_adapter_export(
            adapter,
            AdapterExportSaveRequest(file=str(export_path), dtype=torch.float32, metadata={"format": "test"}),
        )

        fresh_adapter = build_adapter_for_legacy_module("library.adapters.boft", request)
        load_info = load_adapter_export(fresh_adapter, AdapterExportLoadRequest(file=str(export_path)))
        assert load_info == {}

        original_weight = denoiser.to_q.weight.detach().clone()
        loaded_runtime = build_adapter_from_weights_for_legacy_module("library.adapters.boft", request, str(export_path))
        assert isinstance(loaded_runtime, LoadedAdapterRuntime)
        loaded_runtime.merge_into(
            AdapterMergeRequest(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=denoiser),
                resolved_targets=resolved_targets,
                dtype=torch.float32,
                device="cpu",
            )
        )

        assert not torch.allclose(denoiser.to_q.weight, original_weight)

    def test_registered_boft_runtime_reports_partial_checkpoint_as_missing(self, tmp_path):
        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4, bias=False)

        text_encoder = DummyTextEncoder()
        resolved_targets = build_component_module_targets(
            model_type="sdxl",
            text_encoders=[text_encoder, None],
            vae=None,
            denoiser=None,
            include_text_encoders=[True, False],
            include_denoiser=False,
        )
        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="boft",
                settings={"factor": 2, "constraint": 0.25},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=None),
            ),
            resolved_targets=resolved_targets,
        )

        adapter = build_adapter_for_legacy_module("library.adapters.boft", request)
        export_path = tmp_path / "boft_partial.safetensors"
        save_adapter_export(
            adapter,
            AdapterExportSaveRequest(file=str(export_path), dtype=torch.float32, metadata={"format": "test"}),
        )
        state_dict = dict(load_file(str(export_path)))
        state_dict.pop("boft_clip_l_proj.alpha")
        partial_path = tmp_path / "boft_partial_missing_alpha.safetensors"
        save_file(state_dict, str(partial_path), {"format": "test"})

        fresh_adapter = build_adapter_for_legacy_module("library.adapters.boft", request)
        load_info = load_adapter_export(fresh_adapter, AdapterExportLoadRequest(file=str(partial_path)))

        assert load_info == {"missing keys": ["boft_clip_l_proj"]}

    def test_registered_ia3_runtime_exposes_repo_owned_trainable_refs(self):
        from library.adapters.methods.peft.ia3.module import Ia3Module

        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4, bias=False)
                self.norm = torch.nn.LayerNorm(4)

        class DummyDenoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(4, 6, bias=True)

        text_encoder = DummyTextEncoder()
        denoiser = DummyDenoiser()
        resolved_targets = build_component_module_targets(
            model_type="sdxl",
            text_encoders=[text_encoder, None],
            vae=None,
            denoiser=denoiser,
            include_text_encoders=[True, False],
            include_denoiser=True,
        )
        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="ia3",
                settings={"train_on_input": True},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=denoiser),
            ),
            resolved_targets=resolved_targets,
        )

        adapter = build_adapter_for_legacy_module("library.adapters.ia3", request)
        refs = adapter.describe_trainable_parameter_refs()

        assert adapter.adapter_resolved_targets is resolved_targets
        assert all(isinstance(module, Ia3Module) for module in adapter.ia3_modules)
        assert {module.__class__.__module__ for module in adapter.ia3_modules} == {"library.adapters.methods.peft.ia3.module"}
        assert {module.adapter_target.path for module in adapter.ia3_modules} == {"clip_l.proj", "unet.to_q"}
        assert {ref.component for ref in refs} == {"clip_l", "unet"}
        assert {ref.component_key for ref in refs} == {"text_encoder1", "denoiser"}
        assert {ref.target_path for ref in refs} == {"clip_l.proj", "unet.to_q"}
        assert {ref.name for ref in refs} == {"ia3_clip_l_proj.weight", "ia3_unet_to_q.weight"}
        assert {ref.source_target_ref.selector for ref in refs if ref.source_target_ref is not None} == {"clip_l.proj", "unet.to_q"}
        assert all("norm" not in ref.target_path for ref in refs)

    def test_registered_ia3_runtime_auto_selects_axis_per_target(self):
        class DummyMlp(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc2 = torch.nn.Linear(4, 4, bias=False)

        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.k_proj = torch.nn.Linear(4, 4, bias=False)
                self.v_proj = torch.nn.Linear(4, 4, bias=False)
                self.mlp = DummyMlp()

        class DummyFf(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity(), torch.nn.Linear(4, 4, bias=False)])

        class DummyDenoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_k = torch.nn.Linear(4, 4, bias=False)
                self.ff = DummyFf()

        text_encoder = DummyTextEncoder()
        denoiser = DummyDenoiser()
        resolved_targets = build_component_module_targets(
            model_type="sdxl",
            text_encoders=[text_encoder, None],
            vae=None,
            denoiser=denoiser,
            include_text_encoders=[True, False],
            include_denoiser=True,
        )
        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="ia3",
                settings={},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=denoiser),
            ),
            resolved_targets=resolved_targets,
        )

        adapter = build_adapter_for_legacy_module("library.adapters.ia3", request)
        train_on_input_by_path = {module.adapter_target.path: module.train_on_input for module in adapter.ia3_modules}

        assert train_on_input_by_path["clip_l.k_proj"] is False
        assert train_on_input_by_path["clip_l.v_proj"] is False
        assert train_on_input_by_path["clip_l.mlp.fc2"] is True
        assert train_on_input_by_path["unet.to_k"] is False
        assert train_on_input_by_path["unet.ff.net.2"] is True

    def test_registered_ia3_runtime_allows_explicit_global_axis_override(self):
        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.k_proj = torch.nn.Linear(4, 4, bias=False)
                self.mlp = torch.nn.Module()
                self.mlp.fc2 = torch.nn.Linear(4, 4, bias=False)

        text_encoder = DummyTextEncoder()
        resolved_targets = build_component_module_targets(
            model_type="sdxl",
            text_encoders=[text_encoder, None],
            vae=None,
            denoiser=None,
            include_text_encoders=[True, False],
            include_denoiser=False,
        )
        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="ia3",
                settings={"train_on_input": True},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=None),
            ),
            resolved_targets=resolved_targets,
        )

        adapter = build_adapter_for_legacy_module("library.adapters.ia3", request)

        assert all(module.train_on_input is True for module in adapter.ia3_modules)

    def test_registered_ia3_runtime_round_trips_export_and_merge(self, tmp_path):
        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4, bias=False)

        class DummyDenoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(4, 4, bias=True)

        text_encoder = DummyTextEncoder()
        denoiser = DummyDenoiser()
        resolved_targets = build_component_module_targets(
            model_type="sdxl",
            text_encoders=[text_encoder, None],
            vae=None,
            denoiser=denoiser,
            include_text_encoders=[True, False],
            include_denoiser=True,
        )
        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="ia3",
                settings={"train_on_input": False, "bypass_mode": True},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=denoiser),
            ),
            resolved_targets=resolved_targets,
        )

        adapter = build_adapter_for_legacy_module("library.adapters.ia3", request)
        for parameter in adapter.parameters():
            parameter.data.fill_(0.25)

        export_path = tmp_path / "ia3.safetensors"
        save_adapter_export(
            adapter,
            AdapterExportSaveRequest(file=str(export_path), dtype=torch.float32, metadata={"format": "test"}),
        )

        fresh_adapter = build_adapter_for_legacy_module("library.adapters.ia3", request)
        load_info = load_adapter_export(fresh_adapter, AdapterExportLoadRequest(file=str(export_path)))
        assert load_info == {}

        original_weight = denoiser.to_q.weight.detach().clone()
        original_bias = denoiser.to_q.bias.detach().clone()
        loaded_runtime = build_adapter_from_weights_for_legacy_module("library.adapters.ia3", request, str(export_path))
        assert isinstance(loaded_runtime, LoadedAdapterRuntime)
        loaded_runtime.merge_into(
            AdapterMergeRequest(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=denoiser),
                resolved_targets=resolved_targets,
                dtype=torch.float32,
                device="cpu",
            )
        )

        assert not torch.allclose(denoiser.to_q.weight, original_weight)
        assert not torch.allclose(denoiser.to_q.bias, original_bias)

    def test_registered_ia3_runtime_reports_partial_checkpoint_as_missing(self, tmp_path):
        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4, bias=False)

        text_encoder = DummyTextEncoder()
        resolved_targets = build_component_module_targets(
            model_type="sdxl",
            text_encoders=[text_encoder, None],
            vae=None,
            denoiser=None,
            include_text_encoders=[True, False],
            include_denoiser=False,
        )
        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="ia3",
                settings={"train_on_input": True},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=None),
            ),
            resolved_targets=resolved_targets,
        )

        adapter = build_adapter_for_legacy_module("library.adapters.ia3", request)
        export_path = tmp_path / "ia3_partial.safetensors"
        save_adapter_export(
            adapter,
            AdapterExportSaveRequest(file=str(export_path), dtype=torch.float32, metadata={"format": "test"}),
        )
        state_dict = dict(load_file(str(export_path)))
        state_dict.pop("ia3_clip_l_proj.on_input")
        partial_path = tmp_path / "ia3_partial_missing_on_input.safetensors"
        save_file(state_dict, str(partial_path), {"format": "test"})

        fresh_adapter = build_adapter_for_legacy_module("library.adapters.ia3", request)
        load_info = load_adapter_export(fresh_adapter, AdapterExportLoadRequest(file=str(partial_path)))

        assert load_info == {"missing keys": ["ia3_clip_l_proj"]}

    def test_registered_abba_runtime_exposes_repo_owned_trainable_refs(self):
        from library.adapters.methods.peft.abba.module import AbbaModule

        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4, bias=False)
                self.norm = torch.nn.LayerNorm(4)

        class DummyDenoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(4, 6, bias=True)

        text_encoder = DummyTextEncoder()
        denoiser = DummyDenoiser()
        resolved_targets = build_component_module_targets(
            model_type="sdxl",
            text_encoders=[text_encoder, None],
            vae=None,
            denoiser=denoiser,
            include_text_encoders=[True, False],
            include_denoiser=True,
        )
        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="abba",
                settings={"adapter_rank": 4, "use_scalar": True},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=denoiser),
            ),
            resolved_targets=resolved_targets,
        )

        adapter = build_adapter_for_legacy_module("library.adapters.abba", request)
        refs = adapter.describe_trainable_parameter_refs()

        assert adapter.adapter_resolved_targets is resolved_targets
        assert all(isinstance(module, AbbaModule) for module in adapter.abba_modules)
        assert {module.__class__.__module__ for module in adapter.abba_modules} == {"library.adapters.methods.peft.abba.module"}
        assert {module.adapter_target.path for module in adapter.abba_modules} == {"clip_l.proj", "unet.to_q"}
        assert {ref.component for ref in refs} == {"clip_l", "unet"}
        assert {ref.component_key for ref in refs} == {"text_encoder1", "denoiser"}
        assert {ref.target_path for ref in refs} == {"clip_l.proj", "unet.to_q"}
        assert any(ref.name.endswith("scalar") for ref in refs)
        assert {ref.source_target_ref.selector for ref in refs if ref.source_target_ref is not None} == {"clip_l.proj", "unet.to_q"}
        assert all("norm" not in ref.target_path for ref in refs)

    def test_registered_abba_runtime_round_trips_export_and_merge(self, tmp_path):
        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4, bias=False)

        class DummyDenoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(4, 4, bias=True)

        text_encoder = DummyTextEncoder()
        denoiser = DummyDenoiser()
        resolved_targets = build_component_module_targets(
            model_type="sdxl",
            text_encoders=[text_encoder, None],
            vae=None,
            denoiser=denoiser,
            include_text_encoders=[True, False],
            include_denoiser=True,
        )
        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="abba",
                settings={"adapter_rank": 4, "bypass_mode": True, "use_scalar": True},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=denoiser),
            ),
            resolved_targets=resolved_targets,
        )

        adapter = build_adapter_for_legacy_module("library.adapters.abba", request)
        for parameter in adapter.parameters():
            parameter.data.fill_(0.25)

        export_path = tmp_path / "abba.safetensors"
        save_adapter_export(
            adapter,
            AdapterExportSaveRequest(file=str(export_path), dtype=torch.float32, metadata={"format": "test"}),
        )

        fresh_adapter = build_adapter_for_legacy_module("library.adapters.abba", request)
        load_info = load_adapter_export(fresh_adapter, AdapterExportLoadRequest(file=str(export_path)))
        assert load_info == {}

        original_weight = denoiser.to_q.weight.detach().clone()
        original_bias = denoiser.to_q.bias.detach().clone()
        loaded_runtime = build_adapter_from_weights_for_legacy_module("library.adapters.abba", request, str(export_path))
        assert isinstance(loaded_runtime, LoadedAdapterRuntime)
        loaded_runtime.merge_into(
            AdapterMergeRequest(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=denoiser),
                resolved_targets=resolved_targets,
                dtype=torch.float32,
                device="cpu",
            )
        )

        assert not torch.allclose(denoiser.to_q.weight, original_weight)
        assert torch.allclose(denoiser.to_q.bias, original_bias)

    def test_registered_abba_runtime_reports_partial_checkpoint_as_missing(self, tmp_path):
        class DummyTextEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4, bias=False)

        text_encoder = DummyTextEncoder()
        resolved_targets = build_component_module_targets(
            model_type="sdxl",
            text_encoders=[text_encoder, None],
            vae=None,
            denoiser=None,
            include_text_encoders=[True, False],
            include_denoiser=False,
        )
        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="abba",
                settings={"adapter_rank": 4},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=None),
            ),
            resolved_targets=resolved_targets,
        )

        adapter = build_adapter_for_legacy_module("library.adapters.abba", request)
        export_path = tmp_path / "abba_partial.safetensors"
        save_adapter_export(
            adapter,
            AdapterExportSaveRequest(file=str(export_path), dtype=torch.float32, metadata={"format": "test"}),
        )
        state_dict = dict(load_file(str(export_path)))
        state_dict.pop("abba_clip_l_proj.alpha")
        partial_path = tmp_path / "abba_partial_missing_alpha.safetensors"
        save_file(state_dict, str(partial_path), {"format": "test"})

        fresh_adapter = build_adapter_for_legacy_module("library.adapters.abba", request)
        load_info = load_adapter_export(fresh_adapter, AdapterExportLoadRequest(file=str(partial_path)))

        assert load_info == {"missing keys": ["abba_clip_l_proj"]}

    def test_registered_tlora_runtime_exposes_repo_owned_trainable_refs(self):
        from library.adapters.methods.peft.tlora.module import TloraModule

        class DummyDenoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(4, 4, bias=False)

        denoiser = DummyDenoiser()
        resolved_targets = build_component_module_targets(
            model_type="sdxl",
            text_encoders=[None, None],
            vae=None,
            denoiser=denoiser,
            include_text_encoders=[False, False],
            include_denoiser=True,
        )
        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="tlora",
                settings={"adapter_rank": 4, "mask_min_rank": 2, "mask_alpha": 1.0},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[None, None], denoiser=denoiser),
            ),
            resolved_targets=resolved_targets,
        )

        adapter = build_adapter_for_legacy_module("library.adapters.tlora", request)
        refs = adapter.describe_trainable_parameter_refs()

        assert all(isinstance(module, TloraModule) for module in adapter.tlora_modules)
        assert {ref.target_path for ref in refs} == {"unet.to_q"}

    def test_registered_tlora_runtime_round_trips_export_and_merge(self, tmp_path):
        class DummyDenoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(4, 4, bias=True)

        denoiser = DummyDenoiser()
        resolved_targets = build_component_module_targets(
            model_type="sdxl",
            text_encoders=[None, None],
            vae=None,
            denoiser=denoiser,
            include_text_encoders=[False, False],
            include_denoiser=True,
        )
        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="tlora",
                settings={"adapter_rank": 4, "adapter_alpha": 4.0, "use_scalar": True},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[None, None], denoiser=denoiser),
            ),
            resolved_targets=resolved_targets,
        )

        adapter = build_adapter_for_legacy_module("library.adapters.tlora", request)
        for parameter in adapter.parameters():
            parameter.data.fill_(0.25)

        export_path = tmp_path / "tlora.safetensors"
        save_adapter_export(
            adapter,
            AdapterExportSaveRequest(file=str(export_path), dtype=torch.float32, metadata={"format": "test"}),
        )

        fresh_adapter = build_adapter_for_legacy_module("library.adapters.tlora", request)
        load_info = load_adapter_export(fresh_adapter, AdapterExportLoadRequest(file=str(export_path)))
        assert load_info == {}

        original_weight = denoiser.to_q.weight.detach().clone()
        original_bias = denoiser.to_q.bias.detach().clone()
        loaded_runtime = build_adapter_from_weights_for_legacy_module("library.adapters.tlora", request, str(export_path))
        assert isinstance(loaded_runtime, LoadedAdapterRuntime)
        loaded_runtime.merge_into(
            AdapterMergeRequest(
                model=AdapterModelContext(vae=None, text_encoder=[None, None], denoiser=denoiser),
                resolved_targets=resolved_targets,
                dtype=torch.float32,
                device="cpu",
            )
        )

        assert not torch.allclose(denoiser.to_q.weight, original_weight)
        assert torch.allclose(denoiser.to_q.bias, original_bias)

    def test_registered_tlora_runtime_derives_timestep_masks_from_strategy_context(self):
        class DummyDenoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(4, 4, bias=False)

        denoiser = DummyDenoiser()
        resolved_targets = build_component_module_targets(
            model_type="sdxl",
            text_encoders=[None, None],
            vae=None,
            denoiser=denoiser,
            include_text_encoders=[False, False],
            include_denoiser=True,
        )
        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="tlora",
                settings={"adapter_rank": 4, "adapter_alpha": 4.0, "mask_min_rank": 2, "mask_alpha": 1.0},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[None, None], denoiser=denoiser),
            ),
            resolved_targets=resolved_targets,
        )

        adapter = build_adapter_for_legacy_module("library.adapters.tlora", request)
        adapter.apply_to(None, denoiser, False, True)
        for parameter in adapter.parameters():
            parameter.data.fill_(0.25)

        inputs = torch.randn(2, 4)
        baseline = denoiser.to_q(inputs)

        low_rank_context = StrategyContext(
            phase=StrategyPhase.TRAIN,
            model_family="sdxl",
            training=TrainingContext(global_step=10, is_train=True),
            denoiser=DenoiserContext(timesteps=torch.tensor([1000, 1000], dtype=torch.long), batch_size=2),
        )
        full_rank_context = StrategyContext(
            phase=StrategyPhase.TRAIN,
            model_family="sdxl",
            training=TrainingContext(global_step=10, is_train=True),
            denoiser=DenoiserContext(timesteps=torch.tensor([0, 0], dtype=torch.long), batch_size=2),
        )

        with publish_strategy_context(low_rank_context):
            low_rank_output = denoiser.to_q(inputs)
        with publish_strategy_context(full_rank_context):
            full_rank_output = denoiser.to_q(inputs)

        assert torch.allclose(baseline, full_rank_output)
        assert not torch.allclose(low_rank_output, full_rank_output)
        assert all(module._active_timestep_mask is None for module in adapter.tlora_modules)
