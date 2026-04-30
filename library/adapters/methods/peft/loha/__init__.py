"""Repo-owned absorbed LoHa adapter runtime."""

from library.adapters.types import AdapterMethodRegistration

from .config import CONFIG_BINDING


REGISTRATION = AdapterMethodRegistration(
    name="loha",
    # This remains the compatibility-era selection string for now even
    # though the repo-owned runtime lives under adapters.methods.
    legacy_module_path="library.adapters.loha",
    runtime_module_path="library.adapters.methods.peft.loha.runtime",
    config_binding=CONFIG_BINDING,
)

__all__ = ["REGISTRATION"]
