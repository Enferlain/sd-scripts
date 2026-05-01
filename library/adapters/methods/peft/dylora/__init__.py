"""Repo-owned absorbed DyLoRA adapter runtime."""

from library.adapters.types import AdapterMethodRegistration

from .config import CONFIG_BINDING


REGISTRATION = AdapterMethodRegistration(
    name="dylora",
    legacy_module_path="library.adapters.dylora",
    runtime_module_path="library.adapters.methods.peft.dylora.runtime",
    config_binding=CONFIG_BINDING,
)

__all__ = ["REGISTRATION"]
