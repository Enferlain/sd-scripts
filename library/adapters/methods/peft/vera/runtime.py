from __future__ import annotations

from typing import Any

from torch import nn

from library.adapters.runtime import AdapterBuildRequest, AdapterMergeRequest, LoadedAdapterRuntime
from library.adapters.shared import AdapterTrainableParameterRef, attach_trainable_parameter_provider, build_named_parameter_refs

from .module import SUPPORTED_MODULE_TYPES, VeraConfig, VeraModule, VeraSharedProjectionBank, get_target_feature_shape
from .state_dict import (
    extract_vera_shared_weights,
    load_vera_metadata,
    load_vera_state_dict,
    resolve_projection_metadata,
    save_vera_state_dict,
)


def _build_vera_target_name(target) -> str:
    return f"vera_{target.path}".replace(".", "_")


def _iter_supported_targets(resolved_targets) -> list:
    return [target for target in resolved_targets.targets if isinstance(target.module, SUPPORTED_MODULE_TYPES)]


def _resolve_shared_shape(targets: list) -> tuple[int, int]:
    target_shapes = [get_target_feature_shape(target.module) for target in targets]
    return max(out_features for out_features, _in_features in target_shapes), max(
        in_features for _out_features, in_features in target_shapes
    )


def _resolve_projection_restore_settings(
    *,
    metadata: dict[str, str] | None,
    fallback_save_projection: bool,
    fallback_projection_prng_key: int,
) -> tuple[bool, int]:
    # Artifact metadata wins over the active request settings so reloaded VeRA
    # checkpoints reconstruct the same shared projections they were exported
    # with, even if the caller's current config drifted later.
    metadata_save_projection, metadata_projection_prng_key = resolve_projection_metadata(metadata)
    save_projection = fallback_save_projection if metadata_save_projection is None else metadata_save_projection
    projection_prng_key = (
        fallback_projection_prng_key if metadata_projection_prng_key is None else metadata_projection_prng_key
    )
    return save_projection, projection_prng_key


def _infer_rank_from_weights(state_dict: dict[str, Any]) -> int | None:
    ranks = {
        int(value.shape[0])
        for key, value in state_dict.items()
        if key.endswith(".vera_lambda_d") and getattr(value, "ndim", None) == 1
    }
    if not ranks:
        return None
    if len(ranks) != 1:
        raise ValueError(f"VeRA weights contain inconsistent lambda_d ranks: {sorted(ranks)}.")
    return next(iter(ranks))


class VeraAdapterRuntime(nn.Module):
    """Repo-owned runtime for VeRA modules."""

    def __init__(self, shared_bank: VeraSharedProjectionBank, modules: list[VeraModule]):
        super().__init__()
        self.shared_bank = shared_bank
        self._module_names: list[str] = []
        self.adapter_resolved_targets: Any = None
        for module in modules:
            self.add_module(module.lora_name, module)
            self._module_names.append(module.lora_name)

    @property
    def vera_modules(self) -> list[VeraModule]:
        return [getattr(self, module_name) for module_name in self._module_names]

    def apply_to(self, *_args) -> None:
        for module in self.vera_modules:
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
        state_dict = load_vera_state_dict(file)
        metadata = load_vera_metadata(file)
        shared_A, shared_B = extract_vera_shared_weights(state_dict)
        save_projection, projection_prng_key = _resolve_projection_restore_settings(
            metadata=metadata,
            fallback_save_projection=self.shared_bank.save_projection,
            fallback_projection_prng_key=self.shared_bank.projection_prng_key,
        )
        if shared_A is None or shared_B is None:
            if save_projection:
                raise ValueError(
                    "VeRA checkpoint is missing shared projection weights, but the artifact metadata marks "
                    "save_projection=true."
                )
            self.shared_bank.reset_from_prng_key(projection_prng_key)
        else:
            self.shared_bank.load_export_state_dict({"vera_A": shared_A, "vera_B": shared_B})

        missing_keys: list[str] = []
        for module in self.vera_modules:
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
        save_vera_state_dict(self.shared_bank, self.vera_modules, file, dtype=dtype, metadata=metadata)


def _attach_trainable_ref_provider(adapter: VeraAdapterRuntime, request: AdapterBuildRequest) -> VeraAdapterRuntime:
    def describe_trainable_parameter_refs() -> list[AdapterTrainableParameterRef]:
        refs: list[AdapterTrainableParameterRef] = []
        for module in adapter.vera_modules:
            target = module.adapter_target
            if target is None:
                raise ValueError(f"VeRA module {module.lora_name!r} is missing adapter target provenance.")
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


def _materialize_runtime_config(request: AdapterBuildRequest) -> VeraConfig:
    settings = dict(request.adapter.settings)
    adapter_rank = settings.pop("adapter_rank", None)
    if adapter_rank is None:
        raise ValueError("VeRA runtime settings must include adapter_rank; set adapter.peft.vera.rank explicitly.")
    config = VeraConfig(
        multiplier=request.context.multiplier,
        rank=int(adapter_rank),
        dropout=float(settings.pop("dropout", 0.0) or 0.0),
        d_initial=float(settings.pop("d_initial", 0.1)),
        init_weights=bool(settings.pop("init_weights", True)),
    )
    save_projection = bool(settings.pop("save_projection", True))
    projection_prng_key = int(settings.pop("projection_prng_key", 0))
    if settings:
        unknown = ", ".join(sorted(settings))
        raise ValueError(f"VeRA runtime received unsupported settings: {unknown}.")
    config_kwargs = config.as_kwargs()
    config_kwargs["save_projection"] = save_projection
    config_kwargs["projection_prng_key"] = projection_prng_key
    return VeraConfig(**{key: value for key, value in config_kwargs.items() if key in {"multiplier", "rank", "dropout", "d_initial", "init_weights"}})


def _resolve_shared_bank(request: AdapterBuildRequest, targets: list) -> VeraSharedProjectionBank:
    settings = dict(request.adapter.settings)
    adapter_rank = settings.pop("adapter_rank", None)
    if adapter_rank is None:
        raise ValueError("VeRA runtime settings must include adapter_rank; set adapter.peft.vera.rank explicitly.")
    max_out_features, max_in_features = _resolve_shared_shape(targets)
    return VeraSharedProjectionBank(
        rank=int(adapter_rank),
        max_in_features=max_in_features,
        max_out_features=max_out_features,
        projection_prng_key=int(settings.get("projection_prng_key", 0)),
        save_projection=bool(settings.get("save_projection", True)),
    )


def _create_vera_module_from_target(target, shared_bank: VeraSharedProjectionBank, config: VeraConfig) -> VeraModule:
    module = VeraModule.from_target_module(
        _build_vera_target_name(target),
        target.module,
        shared_bank=shared_bank,
        config=config,
    )
    module.adapter_target = target
    return module


def create_adapter(request: AdapterBuildRequest):
    """Create a repo-owned VeRA runtime from resolved adapter targets."""

    supported_targets = _iter_supported_targets(request.resolved_targets)
    if not supported_targets:
        raise ValueError("VeRA adapter runtime requires at least one supported resolved target Linear module")

    shared_bank = _resolve_shared_bank(request, supported_targets)
    config = _materialize_runtime_config(request)
    adapter = VeraAdapterRuntime(
        shared_bank,
        [_create_vera_module_from_target(target, shared_bank, config) for target in supported_targets],
    )
    adapter.adapter_resolved_targets = request.resolved_targets
    return _attach_trainable_ref_provider(adapter, request)


def create_adapter_from_weights(request: AdapterBuildRequest, weights_path: str):
    """Create a loaded repo-owned VeRA runtime from saved weights."""

    weights_sd = load_vera_state_dict(weights_path)
    metadata = load_vera_metadata(weights_path)
    supported_targets = _iter_supported_targets(request.resolved_targets)
    if not supported_targets:
        raise ValueError("VeRA adapter runtime requires at least one supported resolved target Linear module")
    shared_A, shared_B = extract_vera_shared_weights(weights_sd)
    fallback_save_projection = bool(request.adapter.settings.get("save_projection", True))
    fallback_projection_prng_key = int(request.adapter.settings.get("projection_prng_key", 0))
    save_projection, projection_prng_key = _resolve_projection_restore_settings(
        metadata=metadata,
        fallback_save_projection=fallback_save_projection,
        fallback_projection_prng_key=fallback_projection_prng_key,
    )

    if shared_A is None or shared_B is None:
        if save_projection:
            raise ValueError(
                f"VeRA weights in '{weights_path}' are missing shared projection tensors while save_projection=true."
            )
        max_out_features, max_in_features = _resolve_shared_shape(supported_targets)
        inferred_rank = _infer_rank_from_weights(weights_sd)
        if inferred_rank is None:
            raise ValueError(f"VeRA weights in '{weights_path}' do not include any lambda_d tensors to infer the rank.")
        shared_bank = VeraSharedProjectionBank(
            rank=inferred_rank,
            max_in_features=max_in_features,
            max_out_features=max_out_features,
            projection_prng_key=projection_prng_key,
            save_projection=False,
        )
    else:
        shared_bank = VeraSharedProjectionBank.from_state_dict(
            shared_A,
            shared_B,
            save_projection=save_projection,
        )

    modules: list[VeraModule] = []
    for target in supported_targets:
        lora_name = _build_vera_target_name(target)
        if not VeraModule.algo_check(weights_sd, lora_name):
            continue
        lambda_b, lambda_d = VeraModule.extract_state_dict(weights_sd, lora_name)
        if lambda_b is None or lambda_d is None:
            continue
        module = VeraModule.make_module_from_state_dict(
            lora_name,
            target.module,
            shared_bank,
            lambda_b,
            lambda_d,
        )
        module.multiplier = request.context.multiplier
        module.adapter_target = target
        modules.append(module)

    if not modules:
        raise ValueError(f"No VeRA weights in '{weights_path}' matched the resolved adapter targets")

    adapter = VeraAdapterRuntime(shared_bank, modules)
    adapter.adapter_resolved_targets = request.resolved_targets
    adapter = _attach_trainable_ref_provider(adapter, request)

    def _merge_into_impl(merge_request: AdapterMergeRequest) -> None:
        for module in adapter.vera_modules:
            module.merge_to(multiplier=request.context.multiplier)

    return LoadedAdapterRuntime(adapter=adapter, state=weights_sd, _merge_into_impl=_merge_into_impl)
