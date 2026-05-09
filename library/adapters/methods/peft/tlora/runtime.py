from __future__ import annotations

from types import MethodType
from typing import Any

from torch import nn

from library.adapters.runtime import AdapterBuildRequest, AdapterMergeRequest, LoadedAdapterRuntime
from library.adapters.shared import AdapterTrainableParameterRef, attach_trainable_parameter_provider, build_named_parameter_refs
from library.strategies.base.context import current_strategy_context

from .module import SUPPORTED_MODULE_TYPES, TloraConfig, TloraModule, compute_timestep_mask_batch
from .state_dict import load_tlora_state_dict, save_tlora_state_dict


def _build_tlora_target_name(target) -> str:
    return f"tlora_{target.path}".replace(".", "_")


def _iter_supported_targets(resolved_targets) -> list:
    return [target for target in resolved_targets.targets if isinstance(target.module, SUPPORTED_MODULE_TYPES)]


class TloraAdapterRuntime(nn.Module):
    """Repo-owned runtime for TLora modules."""

    def __init__(
        self,
        modules: list[TloraModule],
        *,
        mask_min_rank: int,
        mask_alpha: float,
        mask_max_timestep: int,
    ):
        super().__init__()
        self._module_names: list[str] = []
        self.adapter_resolved_targets: Any = None
        self.mask_min_rank = mask_min_rank
        self.mask_alpha = mask_alpha
        self.mask_max_timestep = mask_max_timestep
        for module in modules:
            self.add_module(module.lora_name, module)
            self._module_names.append(module.lora_name)

    @property
    def tlora_modules(self) -> list[TloraModule]:
        return [getattr(self, module_name) for module_name in self._module_names]

    def apply_to(self, *_args) -> None:
        for module in self.tlora_modules:
            self._wrap_module_forward(module)
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
        for module in self.tlora_modules:
            scaled, norm = module.apply_max_norm(max_norm_value, device)
            if scaled is None:
                continue
            norms.append(norm)
            keys_scaled += int(bool(scaled))

        if not norms:
            return 0, 0.0, 0.0
        return keys_scaled, sum(norms) / len(norms), max(norms)

    def load_weights(self, file: str):
        state_dict = load_tlora_state_dict(file)
        missing_keys: list[str] = []
        for module in self.tlora_modules:
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
        save_tlora_state_dict(self.tlora_modules, file, dtype=dtype, metadata=metadata)

    def _resolve_timestep_mask(self, module: TloraModule):
        context = current_strategy_context()
        if context is None or context.denoiser is None or context.denoiser.timesteps is None:
            return None
        return compute_timestep_mask_batch(
            context.denoiser.timesteps,
            max_timestep=self.mask_max_timestep,
            max_rank=module.lora_dim,
            min_rank=self.mask_min_rank,
            alpha=self.mask_alpha,
        )

    def _wrap_module_forward(self, module: TloraModule) -> None:
        if getattr(module, "_strategy_context_wrapped", False):
            return
        original_forward = module.forward

        def forward_with_strategy_context(tlora_module: TloraModule, x, *args, **kwargs):
            mask = self._resolve_timestep_mask(tlora_module)
            tlora_module.set_timestep_mask(mask)
            try:
                return original_forward(x, *args, **kwargs)
            finally:
                tlora_module.clear_timestep_mask()

        module.forward = MethodType(forward_with_strategy_context, module)
        module._strategy_context_wrapped = True


def _attach_trainable_ref_provider(adapter: TloraAdapterRuntime, request: AdapterBuildRequest) -> TloraAdapterRuntime:
    def describe_trainable_parameter_refs() -> list[AdapterTrainableParameterRef]:
        refs: list[AdapterTrainableParameterRef] = []
        for module in adapter.tlora_modules:
            target = module.adapter_target
            if target is None:
                raise ValueError(f"TLora module {module.lora_name!r} is missing adapter target provenance.")
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


def _extract_mask_settings(settings: dict[str, Any]) -> tuple[int, float, int]:
    mask_min_rank = int(settings.pop("mask_min_rank", 1))
    mask_alpha = float(settings.pop("mask_alpha", 1.0))
    mask_max_timestep = int(settings.pop("mask_max_timestep", 1000))
    return mask_min_rank, mask_alpha, mask_max_timestep


def _create_tlora_module_from_target(target, request: AdapterBuildRequest) -> TloraModule:
    settings = dict(request.adapter.settings)
    _extract_mask_settings(settings)
    adapter_rank = settings.pop("adapter_rank", None)
    adapter_alpha = settings.pop("adapter_alpha", None)
    if adapter_rank is None:
        raise ValueError("TLora runtime settings must include adapter_rank; set adapter.peft.tlora.rank explicitly.")

    config = TloraConfig(
        multiplier=request.context.multiplier,
        lora_dim=int(adapter_rank),
        alpha=adapter_alpha,
        dropout=float(settings.pop("dropout", 0.0)),
        module_dropout=float(settings.pop("module_dropout", 0.0)),
        use_scalar=bool(settings.pop("use_scalar", False)),
        bypass_mode=settings.pop("bypass_mode", None),
        sig_type=settings.pop("sig_type", "principal"),
        use_data_init=bool(settings.pop("use_data_init", True)),
    )
    if settings:
        unknown = ", ".join(sorted(settings))
        raise ValueError(f"TLora runtime received unsupported settings: {unknown}.")

    module = TloraModule.from_target_module(_build_tlora_target_name(target), target.module, config=config)
    module.adapter_target = target
    return module


def create_adapter(request: AdapterBuildRequest):
    """Create a repo-owned TLora runtime from resolved adapter targets."""

    supported_targets = _iter_supported_targets(request.resolved_targets)
    if not supported_targets:
        raise ValueError("TLora adapter runtime requires at least one supported resolved target module")

    mask_min_rank, mask_alpha, mask_max_timestep = _extract_mask_settings(dict(request.adapter.settings))
    adapter = TloraAdapterRuntime(
        [_create_tlora_module_from_target(target, request) for target in supported_targets],
        mask_min_rank=mask_min_rank,
        mask_alpha=mask_alpha,
        mask_max_timestep=mask_max_timestep,
    )
    adapter.adapter_resolved_targets = request.resolved_targets
    return _attach_trainable_ref_provider(adapter, request)


def create_adapter_from_weights(request: AdapterBuildRequest, weights_path: str):
    """Create a loaded repo-owned TLora runtime from saved weights."""

    weights_sd = load_tlora_state_dict(weights_path)
    modules: list[TloraModule] = []
    for target in _iter_supported_targets(request.resolved_targets):
        lora_name = _build_tlora_target_name(target)
        if not TloraModule.algo_check(weights_sd, lora_name):
            continue
        q_weight, p_weight, lambda_weight, alpha, base_q, base_p, base_lambda = TloraModule.extract_state_dict(weights_sd, lora_name)
        if any(param is None for param in (q_weight, p_weight, lambda_weight, alpha)):
            continue
        assert q_weight is not None
        assert p_weight is not None
        assert lambda_weight is not None
        assert alpha is not None
        module = TloraModule.make_module_from_state_dict(
            lora_name,
            target.module,
            q_weight,
            p_weight,
            lambda_weight,
            alpha,
            base_q,
            base_p,
            base_lambda,
        )
        module.multiplier = request.context.multiplier
        module.adapter_target = target
        modules.append(module)

    if not modules:
        raise ValueError(f"No TLora weights in '{weights_path}' matched the resolved adapter targets")

    mask_min_rank, mask_alpha, mask_max_timestep = _extract_mask_settings(dict(request.adapter.settings))
    adapter = TloraAdapterRuntime(
        modules,
        mask_min_rank=mask_min_rank,
        mask_alpha=mask_alpha,
        mask_max_timestep=mask_max_timestep,
    )
    adapter.adapter_resolved_targets = request.resolved_targets
    adapter = _attach_trainable_ref_provider(adapter, request)

    def _merge_into_impl(merge_request: AdapterMergeRequest) -> None:
        del merge_request
        for module in adapter.tlora_modules:
            module.merge_to(multiplier=request.context.multiplier)

    return LoadedAdapterRuntime(adapter=adapter, state=weights_sd, _merge_into_impl=_merge_into_impl)
