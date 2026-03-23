"""
Hydra ConfigStore schema registration.

This module provides explicit registration for the shared root config schema.
Scripts should call the shared register function they need.
Tests can call register_all() to register all active schemas at once.

Pattern follows Hydra's documented "library registers its configs" approach.
"""

from hydra.core.config_store import ConfigStore


def register_run():
    """Register the shared run config schema."""
    from library.config.dataclasses.run import RunConfig

    cs = ConfigStore.instance()
    cs.store(name="run_schema", node=RunConfig)


def register_all():
    """Register all active shared config schemas. Use in tests or tools that need all configs."""
    register_run()
