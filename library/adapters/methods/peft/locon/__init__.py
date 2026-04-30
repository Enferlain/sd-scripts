"""Repo-owned absorbed LoCon adapter runtime."""

from library.adapters.types import AdapterMethodRegistration

from .config import CONFIG_BINDING


REGISTRATION = AdapterMethodRegistration(
    name="locon",
    legacy_module_path="library.adapters.locon",
    runtime_module_path="library.adapters.methods.peft.locon.runtime",
    config_binding=CONFIG_BINDING,
)

__all__ = ["REGISTRATION"]
