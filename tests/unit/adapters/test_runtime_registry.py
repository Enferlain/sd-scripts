import sys
import types

from library.adapters import (
    AdapterBuildContext,
    AdapterBuildRequest,
    AdapterModelContext,
    AdapterRuntimeSpec,
    build_component_root_targets,
    build_adapter_for_legacy_module,
    build_adapter_from_weights_for_legacy_module,
    get_adapter_method,
    get_adapter_method_for_legacy_module,
    list_adapter_methods,
)


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
        assert resolved_targets.targets[0].metadata["component_key"] == "text_encoder1"
        assert resolved_targets.targets[1].metadata["component_key"] == "denoiser"

    def test_registered_runtime_preserves_full_model_context_and_attaches_resolved_targets(self, monkeypatch):
        from library.adapters.methods.lora import runtime as lora_runtime

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

    def test_lists_builtin_adapter_types(self):
        registrations = list_adapter_methods()

        assert [registration.name for registration in registrations] == ["lora", "dylora", "oft"]

    def test_resolves_repo_owned_adapter_type(self):
        registration = get_adapter_method("lora")

        assert registration.legacy_module_path == "library.adapters.lora"
        assert registration.runtime_module_path == "library.adapters.methods.lora.runtime"

    def test_resolves_legacy_module_path(self):
        registration = get_adapter_method_for_legacy_module("library.adapters.oft")

        assert registration.name == "oft"

    def test_build_adapter_for_legacy_module_uses_registered_wrapper(self, monkeypatch):
        captured = {}
        runtime_module = types.SimpleNamespace()

        def fake_create_adapter(request):
            captured["request"] = request
            return "adapter-runtime"

        runtime_module.create_adapter = fake_create_adapter
        monkeypatch.setitem(sys.modules, "library.adapters.methods.lora.runtime", runtime_module)

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
            return "adapter-from-weights"

        runtime_module.create_adapter_from_weights = fake_create_adapter_from_weights
        monkeypatch.setitem(sys.modules, "library.adapters.methods.dylora.runtime", runtime_module)

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

        assert result == "adapter-from-weights"
        assert captured["request"] is request
        assert captured["weights_path"] == "weights.safetensors"
