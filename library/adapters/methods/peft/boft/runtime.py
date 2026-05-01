from __future__ import annotations

from typing import Any

import torch
from torch import nn

from library.adapters.runtime import AdapterBuildRequest, AdapterMergeRequest, LoadedAdapterRuntime
from library.adapters.shared import AdapterTrainableParameterRef, attach_trainable_parameter_provider

from .module import BoftConfig, BoftModule, SUPPORTED_MODULE_TYPES
from .state_dict import load_boft_state_dict, save_boft_state_dict


def _build_boft_target_name(target) -> str:
    return f"boft_{target.path}".replace(".", "_")


def _iter_supported_targets(resolved_targets) -> list:
    return [target for target in resolved_targets.targets if isinstance(target.module, SUPPORTED_MODULE_TYPES)]


class BoftAdapterRuntime(nn.Module):
    """Repo-owned runtime for absorbed BOFT modules."""

    def __init__(self, modules: list[BoftModule]):
        super().__init__()
        self._module_names: list[str] = []
        self.adapter_resolved_targets: Any = None
        for module in modules:
            self.add_module(module.lora_name, module)
            self._module_names.append(module.lora_name)

    @property
    def boft_modules(self) -> list[BoftModule]:
        return [getattr(self, module_name) for module_name in self._module_names]

    def apply_to(self, *_args) -> None:
        for module in self.boft_modules:
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
        for module in self.boft_modules:
            scaled, norm = module.apply_max_norm(max_norm_value, device)
            if scaled is None:
                continue
            norms.append(norm)
            keys_scaled += scaled

        if not norms:
            return 0, 0.0, 0.0
        return keys_scaled, sum(norms) / len(norms), max(norms)

    def load_weights(self, file: str):
        state_dict = load_boft_state_dict(file)
        missing_keys: list[str] = []
        for module in self.boft_modules:
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
        save_boft_state_dict(self.boft_modules, file, dtype=dtype, metadata=metadata)


def _attach_trainable_ref_provider(adapter: BoftAdapterRuntime, request: AdapterBuildRequest) -> BoftAdapterRuntime:
    def describe_trainable_parameter_refs() -> list[AdapterTrainableParameterRef]:
        refs: list[AdapterTrainableParameterRef] = []
        for module in adapter.boft_modules:
            target = module.adapter_target
            if target is None:
                raise ValueError(f"BOFT module {module.lora_name!r} is missing adapter target provenance.")
            for param_name, param in module.named_parameters():
                refs.append(
                    AdapterTrainableParameterRef(
                        param=param,
                        name=f"{module.lora_name}.{param_name}",
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


def _create_boft_module_from_target(target, request: AdapterBuildRequest) -> BoftModule:
    settings = dict(request.adapter.settings)
    factor = settings.pop("factor", None)
    if factor is None:
        factor = settings.pop("block_size", None)
    if factor is None:
        factor = settings.pop("adapter_rank", None)
    constraint = settings.pop("constraint", None)
    if constraint is None:
        constraint = settings.pop("adapter_alpha", 0.0)
    if factor is None:
        raise ValueError("BOFT runtime settings must include factor; set adapter.peft.boft.factor explicitly.")

    config = BoftConfig(
        multiplier=request.context.multiplier,
        factor=int(factor),
        constraint=float(constraint),
        num_stages=settings.pop("num_stages", None),
        rescaled=bool(settings.pop("rescaled", False)),
        dropout=float(settings.pop("dropout", 0.0)),
        module_dropout=float(settings.pop("module_dropout", 0.0)),
        bypass_mode=settings.pop("bypass_mode", None),
    )
    if settings:
        unknown = ", ".join(sorted(settings))
        raise ValueError(f"BOFT runtime received unsupported settings: {unknown}.")

    module = BoftModule.from_target_module(_build_boft_target_name(target), target.module, config=config)
    module.adapter_target = target
    return module


def create_adapter(request: AdapterBuildRequest):
    """Create a repo-owned BOFT runtime from resolved adapter targets."""

    supported_targets = _iter_supported_targets(request.resolved_targets)
    if not supported_targets:
        raise ValueError("BOFT adapter runtime requires at least one supported resolved target module")

    adapter = BoftAdapterRuntime([_create_boft_module_from_target(target, request) for target in supported_targets])
    adapter.adapter_resolved_targets = request.resolved_targets
    return _attach_trainable_ref_provider(adapter, request)


def create_adapter_from_weights(request: AdapterBuildRequest, weights_path: str):
    """Create a loaded repo-owned BOFT runtime from saved weights."""

    weights_sd = load_boft_state_dict(weights_path)
    modules: list[BoftModule] = []
    for target in _iter_supported_targets(request.resolved_targets):
        lora_name = _build_boft_target_name(target)
        if not BoftModule.algo_check(weights_sd, lora_name):
            continue
        oft_blocks, rescale, alpha = BoftModule.extract_state_dict(weights_sd, lora_name)
        if oft_blocks is None or alpha is None:
            continue
        with torch.no_grad():
            module = BoftModule.make_module_from_state_dict(
                lora_name,
                target.module,
                oft_blocks,
                rescale,
                alpha,
            )
        module.multiplier = request.context.multiplier
        module.adapter_target = target
        modules.append(module)

    if not modules:
        raise ValueError(f"No BOFT weights in '{weights_path}' matched the resolved adapter targets")

    adapter = BoftAdapterRuntime(modules)
    adapter.adapter_resolved_targets = request.resolved_targets
    adapter = _attach_trainable_ref_provider(adapter, request)

    def _merge_into_impl(merge_request: AdapterMergeRequest) -> None:
        for module in adapter.boft_modules:
            module.merge_to(multiplier=request.context.multiplier)

    return LoadedAdapterRuntime(adapter=adapter, state=weights_sd, _merge_into_impl=_merge_into_impl)
