from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from dataclasses import MISSING, fields
from typing import Any

from library.adapters.registry import get_adapter_method, get_adapter_method_for_legacy_module, list_adapter_methods
from library.adapters.runtime.context import AdapterRuntimeSpec
from library.adapters.types import AdapterMethodRegistration
from library.config.dataclasses.peft import PeftConfig
from library.optimization.arguments import parse_key_value_args


def _resolve_legacy_module_registration(module_or_name: str) -> AdapterMethodRegistration:
    try:
        return get_adapter_method_for_legacy_module(module_or_name)
    except KeyError:
        return get_adapter_method(module_or_name)


def _field_default(field_info) -> Any:
    if field_info.default is not MISSING:
        return field_info.default
    if field_info.default_factory is not MISSING:
        return field_info.default_factory()
    return MISSING


def _get_nondefault_config_values(config_obj: Any, *, config_type: type | None) -> dict[str, Any]:
    if config_obj is None or config_type is None:
        return {}

    active_values: dict[str, Any] = {}
    for field_info in fields(config_type):
        default_value = _field_default(field_info)
        try:
            current_value = getattr(config_obj, field_info.name)
        except Exception:
            current_value = default_value
        if default_value is MISSING or current_value != default_value:
            active_values[field_info.name] = current_value
    return active_values


def _has_config_key(config_obj: Any, key: str) -> bool:
    if isinstance(config_obj, Mapping):
        return key in config_obj
    with suppress(Exception):
        if key in config_obj:
            return True
    with suppress(Exception):
        if key in vars(config_obj):
            return True
    return False


def _get_config_value(config_obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(config_obj, Mapping):
        return config_obj.get(key, default)
    try:
        return getattr(config_obj, key)
    except Exception:
        return default


def _materialize_method_config(config_obj: Any, *, config_type: type | None) -> Any:
    """Return a typed method config populated with dataclass defaults."""

    if config_obj is None or config_type is None:
        return config_obj
    if isinstance(config_obj, config_type):
        return config_obj

    field_values: dict[str, Any] = {}
    for field_info in fields(config_type):
        default_value = _field_default(field_info)
        current_value = _get_config_value(config_obj, field_info.name, default_value)
        if current_value is MISSING:
            continue
        field_values[field_info.name] = current_value
    return config_type(**field_values)


def _registered_method_config_keys() -> tuple[str, ...]:
    """Return config branch keys owned by registered PEFT methods."""

    return tuple(
        registration.config_binding.config_key for registration in list_adapter_methods() if registration.config_binding is not None
    )


def _method_branch_examples() -> str:
    """Return a human-facing example list derived from registered PEFT method keys."""

    examples = [f"adapter.peft.{config_key}" for config_key in _registered_method_config_keys()]
    if len(examples) <= 1:
        return examples[0] if examples else "adapter.peft.<method>"
    return f"{', '.join(examples[:-1])}, or {examples[-1]}"


def get_adapter_peft_config(cfg_or_peft_config: Any) -> PeftConfig | None:
    """Return the active PEFT family config from either root or family config."""

    if cfg_or_peft_config is None:
        return None

    if _has_config_key(cfg_or_peft_config, "adapter"):
        adapter_config = _get_config_value(cfg_or_peft_config, "adapter")
        if adapter_config is not None and _has_config_key(adapter_config, "peft"):
            peft_config = _get_config_value(adapter_config, "peft")
            if peft_config is not None:
                return peft_config

    if _has_config_key(cfg_or_peft_config, "peft"):
        peft_config = _get_config_value(cfg_or_peft_config, "peft")
        if peft_config is not None:
            return peft_config

    peft_family_keys = ("continue_from", "adapter_module", *_registered_method_config_keys())
    if any(_has_config_key(cfg_or_peft_config, attr) for attr in peft_family_keys):
        return cfg_or_peft_config
    return None


def get_active_method_branch_names(peft_config: PeftConfig) -> list[str]:
    """Return method names whose config branch is present."""

    active_methods: list[str] = []
    for registration in list_adapter_methods():
        config_binding = registration.config_binding
        if config_binding is None:
            continue
        if getattr(peft_config, config_binding.config_key, None) is not None:
            active_methods.append(registration.name)
    return active_methods


def resolve_adapter_method_registration(peft_config: PeftConfig) -> AdapterMethodRegistration:
    """Resolve the active PEFT adapter method from branch presence."""

    registration: AdapterMethodRegistration | None = None
    active_methods = get_active_method_branch_names(peft_config)

    configured_method = getattr(peft_config, "method", None)
    if isinstance(configured_method, str) and configured_method.strip():
        registration = get_adapter_method(configured_method)
        if not active_methods:
            raise ValueError(
                f"Legacy peft.method={registration.name!r} is set, but no method branch is configured. "
                f"Add adapter.peft.{registration.name} or remove the legacy method field."
            )
        if registration.name not in active_methods:
            raise ValueError(
                f"Configured legacy peft.method={registration.name!r} does not match active method branch: "
                f"{', '.join(sorted(active_methods))}."
            )
    elif len(active_methods) == 1:
        registration = get_adapter_method(active_methods[0])
    elif len(active_methods) > 1:
        raise ValueError(f"adapter.peft must configure exactly one method branch, got: {', '.join(sorted(active_methods))}.")

    legacy_module = getattr(peft_config, "adapter_module", None)
    if isinstance(legacy_module, str) and legacy_module.strip():
        legacy_registration = _resolve_legacy_module_registration(legacy_module)
        if registration is not None and legacy_registration.name != registration.name:
            raise ValueError(
                f"Configured peft.method={registration.name!r} does not match legacy adapter_module={legacy_module!r} "
                f"resolved as {legacy_registration.name!r}."
            )
        registration = legacy_registration

    if registration is None:
        raise ValueError(f"adapter.peft must configure exactly one method branch, such as {_method_branch_examples()}.")

    return registration


def get_method_config(peft_config: PeftConfig, method_name: str | None = None) -> tuple[AdapterMethodRegistration, Any]:
    """Return the active method registration plus its method-local config object."""

    registration = resolve_adapter_method_registration(peft_config) if method_name is None else get_adapter_method(method_name)
    config_binding = registration.config_binding
    if config_binding is None:
        return registration, None
    return registration, getattr(peft_config, config_binding.config_key, None)


def get_nondefault_method_config_values(peft_config: PeftConfig, method_name: str | None = None) -> dict[str, Any]:
    """Return non-default values for the requested method-local config subtree."""

    registration, method_config = get_method_config(peft_config, method_name)
    config_binding = registration.config_binding
    if config_binding is None:
        return {}
    return _get_nondefault_config_values(method_config, config_type=config_binding.config_type)


def get_inactive_method_config_values(peft_config: PeftConfig, active_method: str | None = None) -> dict[str, dict[str, Any]]:
    """Return non-default values from method-local config subtrees that are not active."""

    selected_method = active_method or resolve_adapter_method_registration(peft_config).name
    inactive_values: dict[str, dict[str, Any]] = {}
    for registration in list_adapter_methods():
        config_binding = registration.config_binding
        if registration.name == selected_method or config_binding is None:
            continue
        values = _get_nondefault_config_values(
            getattr(peft_config, config_binding.config_key, None),
            config_type=config_binding.config_type,
        )
        if values:
            inactive_values[registration.name] = values
    return inactive_values


def parse_legacy_adapter_args(peft_config: PeftConfig) -> dict[str, Any]:
    """Parse compatibility-era adapter args into a normalized dict."""

    return parse_key_value_args(getattr(peft_config, "adapter_args", None))


def build_adapter_runtime_spec(peft_config: PeftConfig) -> AdapterRuntimeSpec:
    """Build a repo-owned runtime spec from PEFT-family config resolution."""

    registration, method_config = get_method_config(peft_config)
    config_binding = registration.config_binding
    if config_binding is None or method_config is None:
        settings: dict[str, Any] = {}
    else:
        typed_method_config = _materialize_method_config(method_config, config_type=config_binding.config_type)
        settings = config_binding.runtime_settings_builder(typed_method_config)
    legacy_args = parse_legacy_adapter_args(peft_config)
    if legacy_args:
        raise ValueError(
            "peft.adapter_args is no longer part of the forward adapter config surface. "
            f"Move these settings into peft.{registration.name}: {', '.join(sorted(legacy_args))}."
        )
    return AdapterRuntimeSpec(adapter_type=registration.name, settings=settings)
