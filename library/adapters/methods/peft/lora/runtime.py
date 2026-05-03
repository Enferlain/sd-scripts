from __future__ import annotations

from typing import Any

from torch import nn

from library.adapters.runtime import AdapterBuildRequest, AdapterMergeRequest, LoadedAdapterRuntime
from library.adapters.shared import AdapterTrainableParameterRef, attach_trainable_parameter_provider

from .module import LoraConfig, LoraModule, SUPPORTED_MODULE_TYPES
from .state_dict import load_lora_state_dict, save_lora_state_dict


def _is_pointwise_conv(module: nn.Module) -> bool:
    return isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) and all(size == 1 for size in module.kernel_size)


def _build_lora_target_name(target, request: AdapterBuildRequest) -> str:
    if target.component_key == "denoiser":
        prefix = "lora_unet"
    elif target.component_key == "text_encoder1":
        text_encoder = request.context.model.text_encoder
        if isinstance(text_encoder, list) and len(text_encoder) > 1:
            prefix = "lora_te1"
        else:
            prefix = "lora_te"
    elif target.component_key == "text_encoder2":
        prefix = "lora_te2"
    else:
        prefix = f"lora_{target.component_key}"
    local_path = target.local_path or ""
    if not local_path:
        return prefix
    return f"{prefix}_{local_path.replace('.', '_')}"


def _iter_supported_targets(resolved_targets) -> list:
    return [target for target in resolved_targets.targets if isinstance(target.module, SUPPORTED_MODULE_TYPES)]


def _resolve_base_rank(settings: dict[str, object]) -> int:
    rank = settings.pop("adapter_rank", None)
    if rank is None:
        return 4
    return int(rank)


def _resolve_base_alpha(settings: dict[str, object]) -> float:
    alpha = settings.pop("adapter_alpha", None)
    if alpha is None:
        return 1.0
    return float(alpha)


def _resolve_module_rank_alpha(target, settings: dict[str, object]) -> tuple[int | None, float | None]:
    base_rank = _resolve_base_rank(settings)
    base_alpha = _resolve_base_alpha(settings)
    conv_rank = settings.pop("conv_dim", None)
    conv_alpha = settings.pop("conv_alpha", None)
    if isinstance(target.module, (nn.Linear,)) or _is_pointwise_conv(target.module):
        return base_rank, base_alpha
    if conv_rank is None:
        return None, None
    return int(conv_rank), float(conv_alpha) if conv_alpha is not None else 1.0


class LoraAdapterRuntime(nn.Module):
    """Repo-owned runtime for LoRA modules."""

    def __init__(self, modules: list[LoraModule]):
        super().__init__()
        self._module_names: list[str] = []
        self.adapter_resolved_targets: Any = None
        for module in modules:
            self.add_module(module.lora_name, module)
            self._module_names.append(module.lora_name)

    @property
    def lora_modules(self) -> list[LoraModule]:
        return [getattr(self, module_name) for module_name in self._module_names]

    def apply_to(self, *_args) -> None:
        for module in self.lora_modules:
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

    def load_weights(self, file: str):
        state_dict = load_lora_state_dict(file)
        missing_keys: list[str] = []
        for module in self.lora_modules:
            if not module.algo_check(state_dict, module.lora_name):
                missing_keys.append(module.lora_name)
                continue
            required_keys = {key: f"{module.lora_name}.{key}" for key in module.required_export_weight_keys}
            if any(full_key not in state_dict for full_key in required_keys.values()):
                missing_keys.append(module.lora_name)
                continue
            weights = {key: state_dict[full_key] for key, full_key in required_keys.items()}
            alpha_key = f"{module.lora_name}.alpha"
            if alpha_key in state_dict:
                weights["alpha"] = state_dict[alpha_key]
            module.load_export_state_dict(weights)

        if missing_keys:
            return {"missing keys": missing_keys}
        return {}

    def save_weights(self, file: str, dtype, metadata: dict[str, str] | None):
        save_lora_state_dict(self.lora_modules, file, dtype=dtype, metadata=metadata)


def _attach_trainable_ref_provider(adapter: LoraAdapterRuntime, request: AdapterBuildRequest) -> LoraAdapterRuntime:
    def describe_trainable_parameter_refs() -> list[AdapterTrainableParameterRef]:
        refs: list[AdapterTrainableParameterRef] = []
        for module in adapter.lora_modules:
            target = module.adapter_target
            if target is None:
                raise ValueError(f"LoRA module {module.lora_name!r} is missing adapter target provenance.")
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


def _create_lora_module_from_target(target, request: AdapterBuildRequest) -> LoraModule | None:
    settings = dict(request.adapter.settings)
    module_rank, module_alpha = _resolve_module_rank_alpha(target, settings)
    if module_rank is None or module_alpha is None:
        return None

    dropout = float(settings.pop("neuron_dropout", settings.pop("dropout", 0.0)) or 0.0)
    rank_dropout = float(settings.pop("rank_dropout", 0.0) or 0.0)
    module_dropout = float(settings.pop("module_dropout", 0.0) or 0.0)
    if settings:
        unknown = ", ".join(sorted(settings))
        raise ValueError(f"LoRA runtime received unsupported settings: {unknown}.")

    config = LoraConfig(
        multiplier=request.context.multiplier,
        lora_dim=module_rank,
        alpha=module_alpha,
        dropout=dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
    )
    module = LoraModule.from_target_module(_build_lora_target_name(target, request), target.module, config=config)
    module.adapter_target = target
    return module


def create_adapter(request: AdapterBuildRequest):
    """Create a repo-owned LoRA runtime from resolved adapter targets."""

    supported_targets = _iter_supported_targets(request.resolved_targets)
    if not supported_targets:
        raise ValueError("LoRA adapter runtime requires at least one supported resolved target module")

    modules = [module for target in supported_targets if (module := _create_lora_module_from_target(target, request)) is not None]
    if not modules:
        raise ValueError("LoRA adapter runtime resolved no target modules with active LoRA ranks")

    adapter = LoraAdapterRuntime(modules)
    adapter.adapter_resolved_targets = request.resolved_targets
    return _attach_trainable_ref_provider(adapter, request)


def create_adapter_from_weights(request: AdapterBuildRequest, weights_path: str):
    """Create a loaded repo-owned LoRA runtime from saved weights."""

    weights_sd = load_lora_state_dict(weights_path)
    modules: list[LoraModule] = []
    for target in _iter_supported_targets(request.resolved_targets):
        lora_name = _build_lora_target_name(target, request)
        if not LoraModule.algo_check(weights_sd, lora_name):
            continue
        alpha, up, down = LoraModule.extract_state_dict(weights_sd, lora_name)
        if up is None or down is None:
            continue
        module = LoraModule.make_module_from_state_dict(
            lora_name,
            target.module,
            up,
            down,
            alpha,
        )
        module.multiplier = request.context.multiplier
        module.adapter_target = target
        modules.append(module)

    if not modules:
        raise ValueError(f"No LoRA weights in '{weights_path}' matched the resolved adapter targets")

    adapter = LoraAdapterRuntime(modules)
    adapter.adapter_resolved_targets = request.resolved_targets
    adapter = _attach_trainable_ref_provider(adapter, request)

    def _merge_into_impl(merge_request: AdapterMergeRequest) -> None:
        for module in adapter.lora_modules:
            module.merge_to(multiplier=request.context.multiplier)

    return LoadedAdapterRuntime(adapter=adapter, state=weights_sd, _merge_into_impl=_merge_into_impl)
