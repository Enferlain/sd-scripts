"""Repo-owned ABBA adapter runtime."""

from library.adapters.types import AdapterMethodRegistration

from .config import CONFIG_BINDING


REGISTRATION = AdapterMethodRegistration(
    name="abba",
    legacy_module_path="library.adapters.abba",
    runtime_module_path="library.adapters.methods.peft.abba.runtime",
    config_binding=CONFIG_BINDING,
)

__all__ = ["REGISTRATION"]
