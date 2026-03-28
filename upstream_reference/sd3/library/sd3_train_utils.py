"""
SD3 upstream donor status for ``sd3_train_utils.py``.

Ported out of this donor file:
- ``save_models`` -> ``library/models/sd3/conversion.py``
- ``get_all_sigmas`` -> ``library/strategies/sd3/sampling.py``
- ``max_denoise`` -> ``library/strategies/sd3/sampling.py``
- ``do_sample`` -> ``library/strategies/sd3/sampling.py`` (as ``_do_sample``)
- ``sample_images`` -> ``library/strategies/sd3/sampling.py``
- ``sample_image_inference`` -> ``library/strategies/sd3/sampling.py``
- ``compute_density_for_timestep_sampling`` -> ``library/strategies/sd3/diffusion.py``
- ``compute_loss_weighting_for_sd3`` -> ``library/strategies/sd3/diffusion.py``
- ``get_noisy_model_input_and_timesteps`` -> ``library/strategies/sd3/diffusion.py``

Still unported from this donor file:
- ``save_sd3_model_on_train_end``
- ``save_sd3_model_on_epoch_end_or_stepwise``
- ``add_sd3_training_arguments``
- ``verify_sdxl_training_args`` (upstream naming retained there)
- ``FlowMatchEulerDiscreteSchedulerOutput``
- ``FlowMatchEulerDiscreteScheduler``
- ``get_sigmas``

This file is now an inventory of remaining work rather than a frozen full copy.
"""
