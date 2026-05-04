"""Repo-owned VeRA adapter method package."""

from library.adapters.types import AdapterMethodRegistration

from .config import CONFIG_BINDING


REGISTRATION = AdapterMethodRegistration(
    name="vera",
    legacy_module_path="library.adapters.vera",
    runtime_module_path="library.adapters.methods.peft.vera.runtime",
    config_binding=CONFIG_BINDING,
)

__all__ = ["REGISTRATION"]
