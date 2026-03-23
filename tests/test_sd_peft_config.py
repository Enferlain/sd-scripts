import hydra
from hydra.core.global_hydra import GlobalHydra
import os


def test_sd_peft_config_loading():
    GlobalHydra.instance().clear()
    # Use absolute path for config
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs"))

    with hydra.initialize_config_dir(version_base=None, config_dir=config_path):
        cfg = hydra.compose(config_name="presets/sd_peft")
        assert cfg is not None

        # Check if key sections exist
        assert "peft" in cfg
        assert "data" in cfg  # bucketing is at cfg.data.bucketing
        assert "optimizer" in cfg
        assert "model" in cfg
        assert "training" in cfg

        # Check defaults
        assert cfg.peft.adapter_alpha == 1.0
        assert cfg.data.bucketing.min_bucket_reso == 256
        assert cfg.training.train_batch_size == 1
        assert cfg.optimizer.learning_rates.base == 2.0e-6
