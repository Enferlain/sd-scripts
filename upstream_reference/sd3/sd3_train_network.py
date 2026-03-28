"""
SD3 upstream donor status for ``sd3_train_network.py``.

Ported out of this donor file:
- ``load_target_model`` -> ``library/models/sd3/loader.py`` and ``library/strategies/sd3/loading.py``
- ``get_tokenize_strategy`` -> ``library/strategies/sd3/training.py``
- ``get_tokenizers`` -> ``library/strategies/sd3/training.py``
- ``get_latents_caching_strategy`` -> ``library/strategies/sd3/caching.py``
- ``get_text_encoding_strategy`` -> ``library/strategies/sd3/training.py``
- ``sample_images`` -> ``library/strategies/sd3/sampling.py``
- ``encode_images_to_latents`` -> ``library/strategies/sd3/diffusion.py``
- ``shift_scale_latents`` -> ``library/strategies/sd3/diffusion.py``
- ``get_noise_pred_and_target`` -> ``library/strategies/sd3/diffusion.py``
- ``update_metadata`` -> ``library/strategies/sd3/checkpointing.py``
- ``prepare_text_encoder_grad_ckpt_workaround`` -> ``library/strategies/sd3/model_preparation.py``
- SD3 composition wiring -> ``library/strategies/sd3/training.py``

Still unported or only partially handled from this donor file:
- ``assert_extra_args``
- ``post_process_network`` (only partially reflected in ``post_process_trainable``)
- ``get_models_for_text_encoding`` (current port is simplified)
- ``get_text_encoders_train_flags`` (current port still uses shared LR logic)
- ``get_text_encoder_outputs_caching_strategy``
- ``cache_text_encoder_outputs_if_needed``
- ``get_noise_scheduler`` (training path still uses repo-global scheduler setup)
- ``post_process_loss`` (explicitly not ported as-is)
- ``get_sai_model_spec`` (covered indirectly via metadata helpers, not as a direct strategy method)
- ``is_text_encoder_not_needed_for_training``
- ``prepare_text_encoder_fp8`` T5 branch
- ``on_step_start``
- ``on_validation_step_end``
- ``prepare_unet_with_accelerator``
- ``setup_parser``

This file is now an inventory of remaining work rather than a frozen full copy.
"""
