from __future__ import annotations

from typing import Any

import torch
from torch import nn

from library.adapters.runtime import AdapterBuildRequest, AdapterMergeRequest, LoadedAdapterRuntime
from library.adapters.shared import AdapterTrainableParameterRef, attach_trainable_parameter_provider

from .module import LokrConfig, LokrModule, SUPPORTED_MODULE_TYPES
from .state_dict import load_lokr_state_dict, save_lokr_state_dict


def _build_lokr_target_name(target) -> str:
    return f"lokr_{target.path}".replace(".", "_")


def _iter_supported_targets(resolved_targets) -> list:
    return [target for target in resolved_targets.targets if isinstance(target.module, SUPPORTED_MODULE_TYPES)]


class LokrAdapterRuntime(nn.Module):
    """Repo-owned runtime for absorbed LoKr modules."""

    def __init__(self, modules: list[LokrModule]):
        super().__init__()
        self._module_names: list[str] = []
        self.adapter_resolved_targets: Any = None
        for module in modules:
            self.add_module(module.lora_name, module)
            self._module_names.append(module.lora_name)

    @property
    def lokr_modules(self) -> list[LokrModule]:
        return [getattr(self, module_name) for module_name in self._module_names]

    def apply_to(self, *_args) -> None:
        for module in self.lokr_modules:
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
        for module in self.lokr_modules:
            scaled, norm = module.apply_max_norm(max_norm_value, device)
            if scaled is None:
                continue
            norms.append(norm)
            keys_scaled += scaled

        if not norms:
            return 0, 0.0, 0.0
        return keys_scaled, sum(norms) / len(norms), max(norms)

    def load_weights(self, file: str):
        state_dict = load_lokr_state_dict(file)
        missing_keys: list[str] = []
        for module in self.lokr_modules:
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
        save_lokr_state_dict(self.lokr_modules, file, dtype=dtype, metadata=metadata)


def _attach_trainable_ref_provider(adapter: LokrAdapterRuntime, request: AdapterBuildRequest) -> LokrAdapterRuntime:
    def describe_trainable_parameter_refs() -> list[AdapterTrainableParameterRef]:
        refs: list[AdapterTrainableParameterRef] = []
        for module in adapter.lokr_modules:
            target = module.adapter_target
            if target is None:
                raise ValueError(f"LoKr module {module.lora_name!r} is missing adapter target provenance.")
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


def _create_lokr_module_from_target(target, request: AdapterBuildRequest) -> LokrModule:
    settings = dict(request.adapter.settings)
    adapter_rank = settings.pop("adapter_rank", None)
    adapter_alpha = settings.pop("adapter_alpha", None)
    neuron_dropout = settings.pop("neuron_dropout", None)
    dropout = settings.pop("dropout", None)
    if adapter_rank is None:
        raise ValueError("LoKr runtime settings must include adapter_rank; set adapter.peft.lokr.rank explicitly.")
    if neuron_dropout is not None or dropout is not None:
        raise ValueError("LoKr plain dropout is disabled; use rank_dropout or module_dropout instead.")
    config = LokrConfig(
        multiplier=request.context.multiplier,
        lora_dim=adapter_rank,
        alpha=adapter_alpha,
        dropout=0.0,
        **settings,
    )

    module = LokrModule.from_target_module(_build_lokr_target_name(target), target.module, config=config)
    module.adapter_target = target
    return module


def create_adapter(request: AdapterBuildRequest):
    """Create a repo-owned LoKr runtime from resolved adapter targets."""

    supported_targets = _iter_supported_targets(request.resolved_targets)
    if not supported_targets:
        raise ValueError("LoKr adapter runtime requires at least one supported resolved target module")

    adapter = LokrAdapterRuntime([_create_lokr_module_from_target(target, request) for target in supported_targets])
    adapter.adapter_resolved_targets = request.resolved_targets
    return _attach_trainable_ref_provider(adapter, request)


def create_adapter_from_weights(request: AdapterBuildRequest, weights_path: str):
    """Create a loaded repo-owned LoKr runtime from saved weights."""

    weights_sd = load_lokr_state_dict(weights_path)
    modules: list[LokrModule] = []
    for target in _iter_supported_targets(request.resolved_targets):
        lora_name = _build_lokr_target_name(target)
        if not LokrModule.algo_check(weights_sd, lora_name):
            continue
        (
            lokr_w1,
            lokr_w1_a,
            lokr_w1_b,
            lokr_w2,
            lokr_w2_a,
            lokr_w2_b,
            lokr_t2,
            alpha,
            dora_scale,
        ) = LokrModule.extract_state_dict(weights_sd, lora_name)
        if alpha is None:
            continue
        with torch.no_grad():
            module = LokrModule.make_module_from_state_dict(
                lora_name,
                target.module,
                lokr_w1,
                lokr_w1_a,
                lokr_w1_b,
                lokr_w2,
                lokr_w2_a,
                lokr_w2_b,
                lokr_t2,
                alpha,
                dora_scale,
            )
        module.multiplier = request.context.multiplier
        module.adapter_target = target
        modules.append(module)

    if not modules:
        raise ValueError(f"No LoKr weights in '{weights_path}' matched the resolved adapter targets")

    adapter = LokrAdapterRuntime(modules)
    adapter.adapter_resolved_targets = request.resolved_targets
    adapter = _attach_trainable_ref_provider(adapter, request)

    def _merge_into_impl(merge_request: AdapterMergeRequest) -> None:
        for module in adapter.lokr_modules:
            module.merge_to(multiplier=request.context.multiplier)

    return LoadedAdapterRuntime(adapter=adapter, state=weights_sd, _merge_into_impl=_merge_into_impl)
