"""Repo-owned absorbed OFT adapter runtime."""

from library.adapters.types import AdapterMethodRegistration

from .config import CONFIG_BINDING


REGISTRATION = AdapterMethodRegistration(
    name="oft",
    legacy_module_path="library.adapters.oft",
    runtime_module_path="library.adapters.methods.peft.oft.runtime",
    config_binding=CONFIG_BINDING,
)

__all__ = ["REGISTRATION"]
