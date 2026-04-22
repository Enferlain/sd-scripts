import sys
import types

import pytest
import torch

from library.adapters import (
    AdapterExportLoadRequest,
    AdapterExportSaveRequest,
    AdapterBuildContext,
    AdapterBuildRequest,
    AdapterModelContext,
    AdapterRuntimeSpec,
    LoadedAdapterRuntime,
    build_component_module_targets,
    build_component_root_targets,
    build_adapter_for_legacy_module,
    build_adapter_from_weights_for_legacy_module,
    get_adapter_method,
    get_adapter_method_for_legacy_module,
    load_adapter_export,
    list_adapter_methods,
    save_adapter_export,
)
from library.adapters.runtime import AdapterMergeRequest


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

    def test_registered_runtime_preserves_full_model_context_and_attaches_resolved_targets(self, monkeypatch):
        from library.adapters.methods.peft.lora import runtime as lora_runtime

        clip_l = object()
        clip_g = object()
        unet = object()
        captured = {}

        def fake_legacy_create_adapter(multiplier, adapter_rank, adapter_alpha, vae, text_encoder, denoiser, neuron_dropout=None, **kwargs):
            captured["multiplier"] = multiplier
            captured["adapter_rank"] = adapter_rank
            captured["adapter_alpha"] = adapter_alpha
            captured["vae"] = vae
            captured["text_encoder"] = text_encoder
            captured["denoiser"] = denoiser
            captured["neuron_dropout"] = neuron_dropout
            captured["kwargs"] = kwargs
            return types.SimpleNamespace()

        monkeypatch.setattr(lora_runtime.legacy_lora, "create_adapter", fake_legacy_create_adapter)
        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="lora",
                settings={"adapter_rank": 8, "adapter_alpha": 16.0, "neuron_dropout": 0.1, "dropout": 0.1},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae="vae", text_encoder=[clip_l, clip_g], denoiser=unet),
            ),
            resolved_targets=build_component_root_targets(
                model_type="sdxl",
                text_encoders=[clip_l, clip_g],
                vae=None,
                denoiser=unet,
                include_text_encoders=[False, True],
                include_denoiser=True,
            ),
        )

        adapter = build_adapter_for_legacy_module("library.adapters.lora", request)

        assert captured["vae"] == "vae"
        assert captured["text_encoder"] == [clip_l, clip_g]
        assert captured["denoiser"] is unet
        assert captured["kwargs"]["dropout"] == 0.1
        assert adapter.adapter_resolved_targets is request.resolved_targets

    def test_registered_runtime_exposes_repo_owned_trainable_refs(self, monkeypatch):
        from library.adapters.methods.peft.lora import runtime as lora_runtime

        class FakeLoraModule(torch.nn.Module):
            def __init__(self, lora_name: str):
                super().__init__()
                self.lora_name = lora_name
                self.lora_down = torch.nn.Linear(2, 2, bias=False)
                self.lora_up = torch.nn.Linear(2, 2, bias=False)

        class FakeAdapter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.text_encoder_loras = [FakeLoraModule("lora_te1_text_model_attn_proj")]
                self.unet_loras = [FakeLoraModule("lora_unet_input_blocks_1_attn_proj")]
                self.block_lr = False
                self.loraplus_lr_ratio = 2.0
                self.loraplus_unet_lr_ratio = None
                self.loraplus_text_encoder_lr_ratio = None

        monkeypatch.setattr(lora_runtime.legacy_lora, "create_adapter", lambda *args, **kwargs: FakeAdapter())
        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(
                adapter_type="lora",
                settings={"adapter_rank": 8, "adapter_alpha": 16.0},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[object(), object()], denoiser=object()),
            ),
            resolved_targets=build_component_root_targets(
                model_type="sdxl",
                text_encoders=[object(), object()],
                vae=None,
                denoiser=object(),
                include_text_encoders=[True, False],
                include_denoiser=True,
            ),
        )

        adapter = build_adapter_for_legacy_module("library.adapters.lora", request)
        refs = adapter.describe_trainable_parameter_refs()

        assert [ref.component for ref in refs] == ["clip_l", "clip_l", "unet", "unet"]
        assert [ref.name for ref in refs] == [
            "lora_te1_text_model_attn_proj.lora_down.weight",
            "lora_te1_text_model_attn_proj.lora_up.weight",
            "lora_unet_input_blocks_1_attn_proj.lora_down.weight",
            "lora_unet_input_blocks_1_attn_proj.lora_up.weight",
        ]
        assert refs[2].component_key == "denoiser"

    def test_lists_builtin_adapter_types(self):
        registrations = list_adapter_methods()

        assert [registration.name for registration in registrations] == ["loha", "lora", "dylora_deprecated", "oft_deprecated"]

    def test_resolves_repo_owned_adapter_type(self):
        registration = get_adapter_method("loha")

        assert registration.legacy_module_path == "library.adapters.loha"
        assert registration.runtime_module_path == "library.adapters.methods.peft.loha.runtime"

    def test_resolves_legacy_module_path(self):
        registration = get_adapter_method_for_legacy_module("library.adapters.oft")

        assert registration.name == "oft_deprecated"

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
        monkeypatch.setitem(sys.modules, "library.adapters.methods.peft.dylora_deprecated.runtime", runtime_module)

        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(adapter_type="dylora_deprecated"),
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
        monkeypatch.setitem(sys.modules, "library.adapters.methods.peft.dylora_deprecated.runtime", runtime_module)

        request = AdapterBuildRequest(
            adapter=AdapterRuntimeSpec(adapter_type="dylora_deprecated"),
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

        assert captured["model"].text_encoder == ["text-encoder"]
        assert captured["model"].denoiser == "denoiser"
        assert captured["resolved_targets"] is resolved_targets
        assert captured["dtype"] == "fp16"
        assert captured["device"] == "cpu"

    def test_registered_loha_runtime_exposes_repo_owned_trainable_refs(self):
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
                settings={"adapter_rank": 4, "adapter_alpha": 8.0, "dropout": 0.0},
            ),
            context=AdapterBuildContext(
                model=AdapterModelContext(vae=None, text_encoder=[text_encoder, None], denoiser=denoiser),
            ),
            resolved_targets=resolved_targets,
        )

        adapter = build_adapter_for_legacy_module("library.adapters.loha", request)
        refs = adapter.describe_trainable_parameter_refs()

        assert adapter.adapter_resolved_targets is resolved_targets
        assert {module.adapter_target.path for module in adapter.loha_modules} == {"clip_l.proj", "unet.to_q"}
        assert {ref.component for ref in refs} == {"clip_l", "unet"}
        assert {ref.component_key for ref in refs} == {"text_encoder1", "denoiser"}
        assert {ref.target_path for ref in refs} == {"clip_l.proj", "unet.to_q"}
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
                settings={"adapter_rank": 4, "adapter_alpha": 8.0, "dropout": 0.0},
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
