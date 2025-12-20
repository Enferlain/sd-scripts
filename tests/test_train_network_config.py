import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import pytest
import os

def test_train_network_config_loading():
    GlobalHydra.instance().clear()
    # Use absolute path to be safe in this environment
    abs_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs"))

    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(config_name="train_network")
        assert cfg is not None

        # Check if key sections exist
        assert "network" in cfg
        assert "buckets" in cfg
        assert "optimizer" in cfg
        assert "dataset" in cfg
        assert "training" in cfg

        # Check defaults
        assert cfg.network.network_alpha == 1.0
        assert cfg.buckets.min_bucket_reso == 256
        assert cfg.training.train_batch_size == 1
        assert cfg.optimizer.learning_rate == 2.0e-6
