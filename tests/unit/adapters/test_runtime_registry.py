import sys
import types

from library.adapters import (
    build_adapter_for_legacy_module,
    build_adapter_from_weights_for_legacy_module,
    get_adapter_method,
    get_adapter_method_for_legacy_module,
    list_adapter_methods,
)


class TestAdapterRegistry:
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

        def fake_create_adapter(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return "adapter-runtime"

        runtime_module.create_adapter = fake_create_adapter
        monkeypatch.setitem(sys.modules, "library.adapters.methods.lora.runtime", runtime_module)

        result = build_adapter_for_legacy_module("library.adapters.lora", 1.0, 8, 8.0)

        assert result == "adapter-runtime"
        assert captured["args"] == (1.0, 8, 8.0)

    def test_build_adapter_from_weights_for_legacy_module_uses_registered_wrapper(self, monkeypatch):
        captured = {}
        runtime_module = types.SimpleNamespace()

        def fake_create_adapter_from_weights(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return "adapter-from-weights"

        runtime_module.create_adapter_from_weights = fake_create_adapter_from_weights
        monkeypatch.setitem(sys.modules, "library.adapters.methods.dylora.runtime", runtime_module)

        result = build_adapter_from_weights_for_legacy_module("library.adapters.dylora", 1.0, "weights.safetensors")

        assert result == "adapter-from-weights"
        assert captured["args"] == (1.0, "weights.safetensors")
