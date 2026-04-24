from __future__ import annotations

from dataclasses import MISSING, fields
from typing import Any

from library.config.dataclasses.peft import PeftConfig
from library.optimization.arguments import parse_key_value_args

from .registry import get_adapter_method, get_adapter_method_for_legacy_module, list_adapter_methods
from .runtime.context import AdapterRuntimeSpec
from .types import AdapterMethodRegistration


def _resolve_legacy_module_registration(module_or_name: str) -> AdapterMethodRegistration:
    try:
        return get_adapter_method_for_legacy_module(module_or_name)
    except KeyError:
        return get_adapter_method(module_or_name)


def _field_default(field_info) -> Any:
    if field_info.default is not MISSING:
        return field_info.default
    if field_info.default_factory is not MISSING:  # type: ignore[attr-defined]
        return field_info.default_factory()  # type: ignore[misc]
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


def resolve_adapter_method_registration(peft_config: PeftConfig) -> AdapterMethodRegistration:
    """Resolve the active adapter method, honoring the thin legacy shim when needed."""

    registration: AdapterMethodRegistration | None = None

    configured_method = getattr(peft_config, "method", None)
    if isinstance(configured_method, str) and configured_method.strip():
        registration = get_adapter_method(configured_method)

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
        raise ValueError("peft.method is required.")

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
    """Build a repo-owned runtime spec from method-owned config translation."""

    registration, method_config = get_method_config(peft_config)
    config_binding = registration.config_binding
    if config_binding is None or method_config is None:
        settings: dict[str, Any] = {}
    else:
        settings = config_binding.runtime_settings_builder(method_config)
    legacy_args = parse_legacy_adapter_args(peft_config)
    if legacy_args:
        raise ValueError(
            "peft.adapter_args is no longer part of the forward adapter config surface. "
            f"Move these settings into peft.{registration.name}: {', '.join(sorted(legacy_args))}."
        )
    return AdapterRuntimeSpec(adapter_type=registration.name, settings=settings)
