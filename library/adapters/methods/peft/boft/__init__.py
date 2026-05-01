"""Repo-owned absorbed BOFT adapter runtime."""

from library.adapters.types import AdapterMethodRegistration

from .config import CONFIG_BINDING


REGISTRATION = AdapterMethodRegistration(
    name="boft",
    legacy_module_path="library.adapters.boft",
    runtime_module_path="library.adapters.methods.peft.boft.runtime",
    config_binding=CONFIG_BINDING,
)

__all__ = ["REGISTRATION"]
