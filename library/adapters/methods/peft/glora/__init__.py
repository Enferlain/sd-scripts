"""Repo-owned GLoRA adapter runtime."""

from library.adapters.types import AdapterMethodRegistration

from .config import CONFIG_BINDING


REGISTRATION = AdapterMethodRegistration(
    name="glora",
    legacy_module_path="library.adapters.glora",
    runtime_module_path="library.adapters.methods.peft.glora.runtime",
    config_binding=CONFIG_BINDING,
)

__all__ = ["REGISTRATION"]
