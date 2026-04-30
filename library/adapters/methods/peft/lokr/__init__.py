"""Repo-owned absorbed LoKr adapter runtime."""

from library.adapters.types import AdapterMethodRegistration

from .config import CONFIG_BINDING


REGISTRATION = AdapterMethodRegistration(
    name="lokr",
    legacy_module_path="library.adapters.lokr",
    runtime_module_path="library.adapters.methods.peft.lokr.runtime",
    config_binding=CONFIG_BINDING,
)

__all__ = ["REGISTRATION"]
