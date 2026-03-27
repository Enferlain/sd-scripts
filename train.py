"""Canonical root entrypoint for the active config-driven training launcher."""

import hydra

from library.config.dataclasses.run import RunConfig
from library.training.launcher import run_training


def train(cfg: RunConfig) -> None:
    """Run training through the shared config-driven launcher."""
    run_training(cfg)


@hydra.main(version_base=None, config_path="configs", config_name=None)
def main(cfg: RunConfig) -> None:
    """Main entrypoint for the active training launcher."""
    train(cfg)

if __name__ == "__main__":
    from library.config.schemas import register_run

    register_run()
    main()
