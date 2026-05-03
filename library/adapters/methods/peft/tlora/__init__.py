"""Repo-owned TLora adapter runtime."""

from library.adapters.types import AdapterMethodRegistration

from .config import CONFIG_BINDING


REGISTRATION = AdapterMethodRegistration(
    name="tlora",
    legacy_module_path="library.adapters.tlora",
    runtime_module_path="library.adapters.methods.peft.tlora.runtime",
    config_binding=CONFIG_BINDING,
)

__all__ = ["REGISTRATION"]
