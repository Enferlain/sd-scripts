from __future__ import annotations

from typing import Any

from torch import nn

from library.adapters.runtime import AdapterBuildRequest, AdapterMergeRequest, LoadedAdapterRuntime
from library.adapters.shared import AdapterTrainableParameterRef, attach_trainable_parameter_provider, build_named_parameter_refs

from .module import GloraConfig, GloraModule, SUPPORTED_MODULE_TYPES
from .state_dict import load_glora_state_dict, save_glora_state_dict


def _build_glora_target_name(target) -> str:
    return f"glora_{target.path}".replace(".", "_")


def _iter_supported_targets(resolved_targets) -> list:
    return [target for target in resolved_targets.targets if isinstance(target.module, SUPPORTED_MODULE_TYPES)]


class GloraAdapterRuntime(nn.Module):
    """Repo-owned runtime for GLoRA modules."""

    def __init__(self, modules: list[GloraModule]):
        super().__init__()
        self._module_names: list[str] = []
        self.adapter_resolved_targets: Any = None
        for module in modules:
            self.add_module(module.lora_name, module)
            self._module_names.append(module.lora_name)

    @property
    def glora_modules(self) -> list[GloraModule]:
        return [getattr(self, module_name) for module_name in self._module_names]

    def apply_to(self, *_args) -> None:
        for module in self.glora_modules:
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
        for module in self.glora_modules:
            scaled, norm = module.apply_max_norm(max_norm_value, device)
            if scaled is None:
                continue
            norms.append(norm)
            keys_scaled += scaled

        if not norms:
            return 0, 0.0, 0.0
        return keys_scaled, sum(norms) / len(norms), max(norms)

    def load_weights(self, file: str):
        state_dict = load_glora_state_dict(file)
        missing_keys: list[str] = []
        for module in self.glora_modules:
            if not module.algo_check(state_dict, module.lora_name):
                missing_keys.append(module.lora_name)
                continue
            required_keys = {key: f"{module.lora_name}.{key}" for key in module.required_export_weight_keys}
            if any(full_key not in state_dict for full_key in required_keys.values()):
                missing_keys.append(module.lora_name)
                continue
            weights = {key: state_dict[full_key] for key, full_key in required_keys.items()}
            for optional_key in module.export_weight_keys:
                if optional_key in weights:
                    continue
                full_key = f"{module.lora_name}.{optional_key}"
                if full_key in state_dict:
                    weights[optional_key] = state_dict[full_key]
            module.load_export_state_dict(weights)

        if missing_keys:
            return {"missing keys": missing_keys}
        return {}

    def save_weights(self, file: str, dtype, metadata: dict[str, str] | None):
        save_glora_state_dict(self.glora_modules, file, dtype=dtype, metadata=metadata)


def _attach_trainable_ref_provider(adapter: GloraAdapterRuntime, request: AdapterBuildRequest) -> GloraAdapterRuntime:
    def describe_trainable_parameter_refs() -> list[AdapterTrainableParameterRef]:
        refs: list[AdapterTrainableParameterRef] = []
        for module in adapter.glora_modules:
            target = module.adapter_target
            if target is None:
                raise ValueError(f"GLoRA module {module.lora_name!r} is missing adapter target provenance.")
            refs.extend(
                build_named_parameter_refs(
                    module_name=module.lora_name,
                    module=module,
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


def _create_glora_module_from_target(target, request: AdapterBuildRequest) -> GloraModule:
    settings = dict(request.adapter.settings)
    adapter_rank = settings.pop("adapter_rank", None)
    adapter_alpha = settings.pop("adapter_alpha", None)
    if adapter_rank is None:
        raise ValueError("GLoRA runtime settings must include adapter_rank; set adapter.peft.glora.rank explicitly.")

    config = GloraConfig(
        multiplier=request.context.multiplier,
        lora_dim=int(adapter_rank),
        alpha=adapter_alpha,
        dropout=float(settings.pop("dropout", 0.0)),
        rank_dropout=float(settings.pop("rank_dropout", 0.0)),
        module_dropout=float(settings.pop("module_dropout", 0.0)),
        use_tucker=bool(settings.pop("use_tucker", False)),
        use_scalar=bool(settings.pop("use_scalar", False)),
        rank_dropout_scale=bool(settings.pop("rank_dropout_scale", False)),
        bypass_mode=settings.pop("bypass_mode", None),
        rs_lora=bool(settings.pop("rs_lora", False)),
        orthogonalize=bool(settings.pop("orthogonalize", False)),
    )
    if settings:
        unknown = ", ".join(sorted(settings))
        raise ValueError(f"GLoRA runtime received unsupported settings: {unknown}.")

    module = GloraModule.from_target_module(_build_glora_target_name(target), target.module, config=config)
    module.adapter_target = target
    return module


def create_adapter(request: AdapterBuildRequest):
    """Create a repo-owned GLoRA runtime from resolved adapter targets."""

    supported_targets = _iter_supported_targets(request.resolved_targets)
    if not supported_targets:
        raise ValueError("GLoRA adapter runtime requires at least one supported resolved target module")

    adapter = GloraAdapterRuntime([_create_glora_module_from_target(target, request) for target in supported_targets])
    adapter.adapter_resolved_targets = request.resolved_targets
    return _attach_trainable_ref_provider(adapter, request)


def create_adapter_from_weights(request: AdapterBuildRequest, weights_path: str):
    """Create a loaded repo-owned GLoRA runtime from saved weights."""

    weights_sd = load_glora_state_dict(weights_path)
    modules: list[GloraModule] = []
    for target in _iter_supported_targets(request.resolved_targets):
        lora_name = _build_glora_target_name(target)
        if not GloraModule.algo_check(weights_sd, lora_name):
            continue
        a1, a2, b1, b2, bm, alpha = GloraModule.extract_state_dict(weights_sd, lora_name)
        if any(param is None for param in (a1, a2, b1, b2, alpha)):
            continue
        assert a1 is not None
        assert a2 is not None
        assert b1 is not None
        assert b2 is not None
        assert alpha is not None
        module = GloraModule.make_module_from_state_dict(
            lora_name,
            target.module,
            a1,
            a2,
            b1,
            b2,
            bm,
            alpha,
        )
        module.multiplier = request.context.multiplier
        module.adapter_target = target
        modules.append(module)

    if not modules:
        raise ValueError(f"No GLoRA weights in '{weights_path}' matched the resolved adapter targets")

    adapter = GloraAdapterRuntime(modules)
    adapter.adapter_resolved_targets = request.resolved_targets
    adapter = _attach_trainable_ref_provider(adapter, request)

    def _merge_into_impl(merge_request: AdapterMergeRequest) -> None:
        for module in adapter.glora_modules:
            module.merge_to(multiplier=request.context.multiplier)

    return LoadedAdapterRuntime(adapter=adapter, state=weights_sd, _merge_into_impl=_merge_into_impl)
