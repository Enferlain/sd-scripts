import unittest
from unittest.mock import MagicMock
import torch
from library.timestep.timestep_utils import init_timestep_sampler
from library.config.dataclasses.timestep import TimestepConfig, TemperedAdaptiveConfig, GaussianMidSNRConfig, SNRWindowedConfig, MixAdaptiveConfig

class TestTimestepSamplers(unittest.TestCase):
    def setUp(self):
        self.noise_scheduler = MagicMock()
        self.noise_scheduler.config.num_train_timesteps = 1000
        self.noise_scheduler.alphas_cumprod = torch.linspace(0.99, 0.01, 1000)
        self.accelerator = MagicMock()
        self.accelerator.print = MagicMock()

    def test_init_log_snr_uniform_sampler(self):
        config = TimestepConfig(timestep_sampling="log_snr_uniform")
        sampler = init_timestep_sampler(config, self.noise_scheduler, self.accelerator)
        self.assertIsNotNone(sampler)
        self.assertEqual(config.timestep_sampling, "mix_adaptive")
        self.accelerator.print.assert_called_with("Initializing LogSNRUniformSampler.")

    def test_init_tempered_adaptive_sampler(self):
        # Test that entropy_floor (config) is correctly passed as entropy_floor_ratio (sampler arg)
        ta_config = TemperedAdaptiveConfig(entropy_floor=0.5)
        config = TimestepConfig(timestep_sampling="tempered_adaptive", tempered_adaptive=ta_config)

        sampler = init_timestep_sampler(config, self.noise_scheduler, self.accelerator)

        self.assertIsNotNone(sampler)
        self.assertEqual(sampler.entropy_floor_ratio, 0.5)
        self.assertEqual(config.timestep_sampling, "mix_adaptive")
        self.accelerator.print.assert_called_with("Initializing TemperedAdaptiveSampler.")

    def test_init_gaussian_mid_snr_sampler(self):
        gc_config = GaussianMidSNRConfig(entropy_floor=0.4)
        config = TimestepConfig(timestep_sampling="gaussian_mid_snr", gaussian_mid_snr=gc_config)

        sampler = init_timestep_sampler(config, self.noise_scheduler, self.accelerator)

        self.assertIsNotNone(sampler)
        self.assertEqual(sampler.entropy_floor_ratio, 0.4)
        self.assertEqual(config.timestep_sampling, "mix_adaptive")
        self.accelerator.print.assert_called_with("Initializing GaussianMidSNRSampler.")

    def test_init_snr_windowed_sampler(self):
        sc_config = SNRWindowedConfig(entropy_floor=0.3)
        config = TimestepConfig(timestep_sampling="snr_windowed", snr_windowed=sc_config)

        sampler = init_timestep_sampler(config, self.noise_scheduler, self.accelerator)

        self.assertIsNotNone(sampler)
        self.assertEqual(sampler.entropy_floor_ratio, 0.3)
        self.assertEqual(config.timestep_sampling, "mix_adaptive")
        self.accelerator.print.assert_called_with("Initializing SNRWindowedSampler.")

    def test_init_mix_adaptive_sampler(self):
        mc_config = MixAdaptiveConfig()
        config = TimestepConfig(timestep_sampling="mix_adaptive", mix_adaptive=mc_config)
        sampler = init_timestep_sampler(config, self.noise_scheduler, self.accelerator)
        self.assertIsNotNone(sampler)
        self.accelerator.print.assert_called_with("Initializing LossAwareTimestepSampler.")

    def test_init_uniform_sampler(self):
        config = TimestepConfig(timestep_sampling="uniform")
        sampler = init_timestep_sampler(config, self.noise_scheduler, self.accelerator)
        self.assertIsNone(sampler)
        self.assertEqual(config.timestep_sampling, "uniform")

    def test_init_sigma_sampler(self):
        config = TimestepConfig(timestep_sampling="sigma")
        sampler = init_timestep_sampler(config, self.noise_scheduler, self.accelerator)
        self.assertIsNone(sampler)
        self.assertEqual(config.timestep_sampling, "uniform")
