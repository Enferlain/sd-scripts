from __future__ import annotations

from typing import Any

from torch import nn

from library.adapters.runtime import AdapterBuildRequest, AdapterMergeRequest, LoadedAdapterRuntime
from library.adapters.shared import AdapterTrainableParameterRef, attach_trainable_parameter_provider

from .module import Ia3Config, Ia3Module, SUPPORTED_MODULE_TYPES
from .state_dict import load_ia3_state_dict, save_ia3_state_dict


_IA3_INPUT_AXIS_SUFFIXES = ("mlp.fc2", "ff.net.2")
_IA3_OUTPUT_AXIS_SUFFIXES = ("k_proj", "v_proj", "to_k", "to_v")


def _build_ia3_target_name(target) -> str:
    return f"ia3_{target.path}".replace(".", "_")


def _iter_supported_targets(resolved_targets) -> list:
    return [target for target in resolved_targets.targets if isinstance(target.module, SUPPORTED_MODULE_TYPES)]


def _resolve_train_on_input(target, explicit_value: bool | None) -> bool:
    if explicit_value is not None:
        return explicit_value

    metadata_value = target.metadata.get("ia3_train_on_input", target.metadata.get("train_on_input"))
    if metadata_value is not None:
        return bool(metadata_value)

    selector_candidates = (target.local_path, target.path)
    for selector in selector_candidates:
        if any(selector.endswith(suffix) for suffix in _IA3_INPUT_AXIS_SUFFIXES):
            return True
        if any(selector.endswith(suffix) for suffix in _IA3_OUTPUT_AXIS_SUFFIXES):
            return False
    return False


class Ia3AdapterRuntime(nn.Module):
    """Repo-owned runtime for IA3 modules."""

    def __init__(self, modules: list[Ia3Module]):
        super().__init__()
        self._module_names: list[str] = []
        self.adapter_resolved_targets: Any = None
        for module in modules:
            self.add_module(module.lora_name, module)
            self._module_names.append(module.lora_name)

    @property
    def ia3_modules(self) -> list[Ia3Module]:
        return [getattr(self, module_name) for module_name in self._module_names]

    def apply_to(self, *_args) -> None:
        for module in self.ia3_modules:
            module.apply_to()

    def prepare_grad_etc(self, *_args) -> None:
        self.requires_grad_(True)

    def enable_gradient_checkpointing(self) -> None:
        def mark_grad_ckpt(module: nn.Module) -> None:
            module.grad_ckpt = True  # type: ignore[unresolved-attribute]

        self.apply(mark_grad_ckpt)

    def on_epoch_start(self, *_args) -> None:
        self.train()

    def get_trainable_params(self, *_args) -> list[nn.Parameter]:
        return list(self.parameters())

    def apply_max_norm_regularization(self, max_norm_value, device):
        keys_scaled = 0
        norms = []
        for module in self.ia3_modules:
            scaled, norm = module.apply_max_norm(max_norm_value, device)
            if scaled is None:
                continue
            norms.append(norm)
            keys_scaled += int(bool(scaled))

        if not norms:
            return 0, 0.0, 0.0
        return keys_scaled, sum(norms) / len(norms), max(norms)

    def load_weights(self, file: str):
        state_dict = load_ia3_state_dict(file)
        missing_keys: list[str] = []
        for module in self.ia3_modules:
            if not module.algo_check(state_dict, module.lora_name):
                missing_keys.append(module.lora_name)
                continue
            required_keys = {key: f"{module.lora_name}.{key}" for key in module.required_export_weight_keys}
            if any(full_key not in state_dict for full_key in required_keys.values()):
                missing_keys.append(module.lora_name)
                continue
            weights = {key: state_dict[full_key] for key, full_key in required_keys.items()}
            module.load_export_state_dict(weights)

        if missing_keys:
            return {"missing keys": missing_keys}
        return {}

    def save_weights(self, file: str, dtype, metadata: dict[str, str] | None):
        save_ia3_state_dict(self.ia3_modules, file, dtype=dtype, metadata=metadata)


def _attach_trainable_ref_provider(adapter: Ia3AdapterRuntime, request: AdapterBuildRequest) -> Ia3AdapterRuntime:
    def describe_trainable_parameter_refs() -> list[AdapterTrainableParameterRef]:
        refs: list[AdapterTrainableParameterRef] = []
        for module in adapter.ia3_modules:
            target = module.adapter_target
            if target is None:
                raise ValueError(f"IA3 module {module.lora_name!r} is missing adapter target provenance.")
            refs.append(
                AdapterTrainableParameterRef(
                    param=module.weight,
                    name=f"{module.lora_name}.weight",
                    algorithm=request.adapter.adapter_type,
                    component=target.component,
                    component_key=target.component_key,
                    target_path=target.path,
                    source_target_ref=target.target_ref,
                )
            )
        return refs

    attach_trainable_parameter_provider(adapter, describe_trainable_parameter_refs)
    return adapter


def _create_ia3_module_from_target(target, request: AdapterBuildRequest) -> Ia3Module:
    settings = dict(request.adapter.settings)
    explicit_train_on_input = settings.pop("train_on_input", None)
    config = Ia3Config(
        multiplier=request.context.multiplier,
        train_on_input=_resolve_train_on_input(target, explicit_train_on_input),
        module_dropout=float(settings.pop("module_dropout", 0.0)),
        bypass_mode=settings.pop("bypass_mode", None),
    )
    if settings:
        unknown = ", ".join(sorted(settings))
        raise ValueError(f"IA3 runtime received unsupported settings: {unknown}.")

    module = Ia3Module.from_target_module(_build_ia3_target_name(target), target.module, config=config)
    module.adapter_target = target
    return module


def create_adapter(request: AdapterBuildRequest):
    """Create a repo-owned IA3 runtime from resolved adapter targets."""

    supported_targets = _iter_supported_targets(request.resolved_targets)
    if not supported_targets:
        raise ValueError("IA3 adapter runtime requires at least one supported resolved target module")

    adapter = Ia3AdapterRuntime([_create_ia3_module_from_target(target, request) for target in supported_targets])
    adapter.adapter_resolved_targets = request.resolved_targets
    return _attach_trainable_ref_provider(adapter, request)


def create_adapter_from_weights(request: AdapterBuildRequest, weights_path: str):
    """Create a loaded repo-owned IA3 runtime from saved weights."""

    weights_sd = load_ia3_state_dict(weights_path)
    modules: list[Ia3Module] = []
    for target in _iter_supported_targets(request.resolved_targets):
        lora_name = _build_ia3_target_name(target)
        if not Ia3Module.algo_check(weights_sd, lora_name):
            continue
        weight, on_input = Ia3Module.extract_state_dict(weights_sd, lora_name)
        if weight is None:
            continue
        module = Ia3Module.make_module_from_state_dict(
            lora_name,
            target.module,
            weight,
            on_input,
        )
        module.multiplier = request.context.multiplier
        module.adapter_target = target
        modules.append(module)

    if not modules:
        raise ValueError(f"No IA3 weights in '{weights_path}' matched the resolved adapter targets")

    adapter = Ia3AdapterRuntime(modules)
    adapter.adapter_resolved_targets = request.resolved_targets
    adapter = _attach_trainable_ref_provider(adapter, request)

    def _merge_into_impl(merge_request: AdapterMergeRequest) -> None:
        for module in adapter.ia3_modules:
            module.merge_to(multiplier=request.context.multiplier)

    return LoadedAdapterRuntime(adapter=adapter, state=weights_sd, _merge_into_impl=_merge_into_impl)
