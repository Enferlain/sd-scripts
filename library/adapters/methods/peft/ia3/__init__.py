"""Repo-owned IA3 adapter runtime."""

from library.adapters.types import AdapterMethodRegistration

from .config import CONFIG_BINDING


REGISTRATION = AdapterMethodRegistration(
    name="ia3",
    legacy_module_path="library.adapters.ia3",
    runtime_module_path="library.adapters.methods.peft.ia3.runtime",
    config_binding=CONFIG_BINDING,
)

__all__ = ["REGISTRATION"]
