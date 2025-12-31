"""
Timestep sampling utilities.

Factory functions for initializing timestep samplers based on configuration.
"""

import ast
import logging

from library.config.dataclasses.timestep import TimestepConfig
from library.timestep.samplers.loss_aware_sampler import LossAwareTimestepSampler
from library.timestep.samplers.log_snr_sampler import LogSNRUniformSampler
from library.timestep.samplers.tempered_adaptive_sampler import TemperedAdaptiveSampler
from library.timestep.samplers.gaussian_mid_snr_sampler import GaussianMidSNRAdaptiveSampler
from library.timestep.samplers.snr_windowed_loss_aware_sampler import SNRWindowedLossAwareSampler

logger = logging.getLogger(__name__)


def parse_dynamic_timestep_schedule(timestep_config: TimestepConfig, noise_scheduler, accelerator):
    """
    Parse dynamic timestep schedule from config.
    
    Args:
        timestep_config: Training configuration
        noise_scheduler: Diffusers noise scheduler
        accelerator: HuggingFace Accelerator (for printing)
        
    Returns:
        Tuple of (schedule_list, current_min_timestep, current_max_timestep)
        schedule_list is None if no dynamic schedule is configured.
    """
    # Parse the schedule from the config string
    dynamic_timestep_schedule = ast.literal_eval(
        timestep_config.dynamic_timestep_schedule) if timestep_config.dynamic_timestep_schedule else None
    if dynamic_timestep_schedule:
        # Sort the schedule by step number to be safe
        dynamic_timestep_schedule.sort(key=lambda x: x[0])
        accelerator.print(f"Using dynamic timestep schedule: {dynamic_timestep_schedule}")

    # Initialize the current range with the defaults
    current_min_timestep = 0 if timestep_config.min_timestep is None else timestep_config.min_timestep
    current_max_timestep = noise_scheduler.config.num_train_timesteps if timestep_config.max_timestep is None else timestep_config.max_timestep

    return dynamic_timestep_schedule, current_min_timestep, current_max_timestep


def init_timestep_sampler(timestep_config: TimestepConfig, noise_scheduler, accelerator):
    """
    Initialize the appropriate timestep sampler based on config.
    
    Returns the sampler instance and potentially modifies cfg.timestep.timestep_sampling.
    
    Args:
        timestep_config: Training configuration
        noise_scheduler: Diffusers noise scheduler
        accelerator: HuggingFace Accelerator
        
    Returns:
        Timestep sampler instance or None for uniform/shift sampling
    """
    la_sampler = None
    
    if not timestep_config.timestep_sampling:
        return None
    
    sampling_type = timestep_config.timestep_sampling
    
    if sampling_type == "log_snr_uniform":
        accelerator.print("Initializing LogSNRUniformSampler.")
        la_sampler = LogSNRUniformSampler(noise_scheduler, noise_scheduler.config.num_train_timesteps)
        timestep_config.timestep_sampling = "mix_adaptive"
        
    elif sampling_type == "tempered_adaptive":
        accelerator.print("Initializing TemperedAdaptiveSampler.")
        tc = timestep_config.tempered_adaptive
        la_sampler = TemperedAdaptiveSampler(
            noise_scheduler,
            num_bins=tc.bins,
            ema_beta=tc.ema_beta,
            temperature=tc.temperature,
            prior_weight=tc.prior_weight,
            min_prob=tc.min_prob,
            warmup_steps=tc.warmup_steps,
            prior_bias=tc.prior_bias,
            entropy_floor=tc.entropy_floor,  # TODO: Unexpected argument
        )
        timestep_config.timestep_sampling = "mix_adaptive"
        
    elif sampling_type == "gaussian_mid_snr":
        accelerator.print("Initializing GaussianMidSNRSampler.")
        gc = timestep_config.gaussian_mid_snr
        la_sampler = GaussianMidSNRAdaptiveSampler(
            noise_scheduler,
            num_bins=gc.bins,
            ema_beta=gc.ema_beta,
            temperature=gc.temperature,
            min_prob=gc.min_prob,
            entropy_floor=gc.entropy_floor,  # TODO: Unexpected argument
            prior_mu=gc.prior_mu,
            prior_sigma=gc.prior_sigma,
            prior_weight=gc.prior_weight,
            warmup_steps=gc.warmup_steps,
        )
        timestep_config.timestep_sampling = "mix_adaptive"
        
    elif sampling_type == "snr_windowed":
        accelerator.print("Initializing SNRWindowedSampler.")
        sc = timestep_config.snr_windowed
        la_sampler = SNRWindowedLossAwareSampler(
            noise_scheduler,
            num_bins=sc.bins,
            ema_beta=sc.ema_beta,
            temperature=sc.temperature,
            min_prob=sc.min_prob,
            entropy_floor=sc.entropy_floor,  # TODO: Unexpected argument
            center_mu=sc.center_mu,
            half_width=sc.half_width,
            widen_to=sc.widen_to,
            total_widen_steps=sc.max_train_steps,
            cap_max_t=sc.cap_max_t,
        )
        timestep_config.timestep_sampling = "mix_adaptive"
        
    elif sampling_type == "mix_adaptive":
        accelerator.print("Initializing LossAwareTimestepSampler.")
        mc = timestep_config.mix_adaptive
        la_sampler = LossAwareTimestepSampler(
            num_train_timesteps=noise_scheduler.config.num_train_timesteps,
            num_bins=mc.bins,
            ema_beta=mc.ema_beta,
            small_t_frac=mc.small_t_frac,
            small_t_cap=mc.small_t_cap,
            start_p=mc.start_p,
            end_p=mc.end_p,
            anneal=mc.anneal,
            fixed_p=mc.fixed_p,
        )
        
    elif sampling_type in ("sigma", "uniform"):
        la_sampler = None
        timestep_config.timestep_sampling = "uniform"
        if sampling_type == "sigma":
            logger.warning("sigma sampling is not supported yet, using uniform sampling")
            
    elif sampling_type == "shift":
        la_sampler = None
        # shift sampling is handled in get_noise_noisy_latents_and_timesteps
    
    return la_sampler
