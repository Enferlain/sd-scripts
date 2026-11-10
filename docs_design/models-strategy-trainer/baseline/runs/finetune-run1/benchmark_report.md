# Benchmark Report

**Date:** 2026-11-10 22:48:34
**Status:** succeeded
**Config:** baseline_sdxl_finetune
**Output Name:** test_peft_resource_sampled
**Total Time:** 81.18s

## Summary

| Key | Value |
|-----|-------|
| Status | succeeded |
| Config | baseline_sdxl_finetune |
| Output Name | test_peft_resource_sampled |
| Total Time | 81.18s |

## Runtime

| Key | Value |
|-----|-------|
| Mode | FineTuneMode |
| Strategy | SdxlTrainingStrategy |
| Optimizer | bitsandbytes.optim.adamw.AdamW8bit |
| Global Step | 50 |
| Num Epochs | 1 |
| Avg Train Step Time | 1.26s |
| Session ID | 860755131 |
| Output Dir | D:/Projects/sd-scripts/outputs/rmst-baseline/finetune-run1 |

## System

| Key | Value |
|-----|-------|
| GPU | NVIDIA GeForce RTX 3090, 24576 MiB |
| Python | 3.13.5 |
| PyTorch | 2.11.0+cu130 |

## Key Configuration

| YAML Key | Value |
|----------|-------|
| `hydra.config_name` | baseline_sdxl_finetune |
| `mode` | finetune |
| `model.model_type` | sdxl |
| `output.saving.output_dir` | D:/Projects/sd-scripts/outputs/rmst-baseline/finetune-run1 |
| `output.saving.output_name` | test_peft_resource_sampled |
| `training.train_batch_size` | 2 |
| `training.gradient_accumulation_steps` | 1 |
| `training.max_train_steps` | 50 |
| `data.preprocessing.resolution` | 1024,1024 |
| `data.caching.cache_latents` | true |
| `data.caching.cache_latents_to_disk` | true |
| `data.caching.cache_text_encoder_outputs` | true |
| `data.caching.cache_text_encoder_outputs_to_disk` | true |
| `data.caching.vae_batch_size` | 2 |
| `data.caching.te_batch_size` | 8 |
| `data.loader.num_workers` | 0 |
| `data.loader.prefetch_factor` | 2 |
| `data.loader.persistent_workers` | true |
| `data.loader.pin_memory` | true |
| `performance.precision.mixed_precision` | bf16 |
| `performance.precision.no_half_vae` | true |
| `performance.memory.gradient_checkpointing` | true |
| `performance.memory.offload_text_encoders` | false |
| `performance.attention.xformers` | true |
| `performance.deepspeed.deepspeed` | false |
| `objective.prediction` | v_prediction |
| `optimizer.optimizer_type` | AdamW8bit |
| `optimizer.learning_rates.denoiser` | 1e-05 |
| `optimizer.learning_rates.text_encoders` | [0.0, 0.0] |
| `output.logging.benchmark_report.enabled` | true |
| `output.logging.resource_monitor.enabled` | true |
| `output.logging.resource_monitor.mode` | sampled |
| `output.logging.resource_monitor.output_jsonl` | resource_monitor.jsonl |
| `output.logging.logging_dir` | D:/Projects/sd-scripts/outputs/rmst-baseline/finetune-run1/logs |
| `hydra.task_overrides` | ["output.saving.output_dir=D:/Projects/sd-scripts/outputs/rmst-baseline/finetune-run1", "output.logging.logging_dir=D:/Projects/sd-scripts/outputs/rmst-baseline/finetune-run1/logs"] |

## Speed Results

| Phase | Final Speed |
|-------|-------------|
| Latent Caching | N/A |
| TE Caching | N/A |
| Training | 1.26s / it |

## Runtime Trace

| Milestone | Time |
|-----------|------|
| Launch -> Progress Bar | 10.27s |
| Launch -> First Step Started | 10.30s |
| Launch -> First Synced Step | 12.64s |
| Launch -> Run Ended | 82.73s |

| Slowest Phase | Total Time |
|---------------|------------|
| `training.epoch.0` | 62.88s |
| `checkpoint.save` | 9.50s |
| `training.prep.accelerator` | 1.23s |
| `training.prep.shared_precision` | 1.21s |
| `cache.latents` | 0.40s |

## Trace-only Phases

| Phase | Duration | Note |
|-------|----------|------|
| `startup.accelerator` | 0.09s | Trace-only span with no matching resource-monitor phase row. |

## Startup Estimates

Optimizer state memory is estimated from optimizer family and trainable parameter size.

| Component | Modules Trainable / Total | Params Trainable / Total | Param MB | Grad MB Est | Optimizer State MB Est |
|-----------|---------------------------|---------------------------|----------|-------------|------------------------|
| `clip_l` | 0 / 99 | 0 / 123,060,480 | 234.7 | 0.0 | 0.0 |
| `clip_g` | 0 / 260 | 0 / 694,659,840 | 1325.0 | 0.0 | 0.0 |
| `vae` | 0 / 124 | 0 / 83,653,863 | 319.1 | 0.0 | 0.0 |
| `unet` | 1,050 / 1,050 | 2,567,463,684 / 2,567,463,684 | 4897.0 | 4897.0 | 9794.1 |

## Session Summary

| Metric | Value |
|--------|-------|
| Duration | 80.64s |
| GPU Allocated | 10080 MB |
| GPU Reserved | 16678 MB |
| GPU Peak Allocated | 10136 MB |
| GPU Used | N/A |
| GPU Used Peak | 227 MB |
| CPU RSS | 18147 MB |
| CPU VMS | 38587 MB |
| Resource Events | 178 |
| Resource JSONL | D:\Projects\sd-scripts\outputs\rmst-baseline\finetune-run1\resource_monitor.jsonl |

## Run Resource Profiles

### run_resource_summary

| Field | Value |
|-------|-------|
| Identifier | `860755131:profile:run_resource_summary:run_resource_summary_v1` |
| Derivation Version | `run_resource_summary_v1` |
| Generated At | 1.79435e+09 |
| Source Records | 1190 |

| Value | Amount |
|-------|--------|
| CPU RSS Delta | 17266 MB |
| CPU RSS End | 18147 MB |
| CPU RSS Start | 881 MB |
| CPU VMS Delta | 35851 MB |
| CPU VMS End | 38587 MB |
| CPU VMS Start | 2736 MB |
| GPU Allocated Peak by Device | cuda:0: 15743 MB |
| GPU Allocated Peak | 15743 MB |
| GPU Reserved Peak by Device | cuda:0: 16678 MB |
| GPU Reserved Peak | 16678 MB |
| GPU Used Peak by Device | cuda:0: 227 MB |
| GPU Used Peak | 227 MB |
| Gradient Memory Estimate | 4897 MB |
| Loaded Parameter Memory | 6776 MB |
| Observation Frames | 178 |
| Observation Measurements | 1002 |
| Optimizer State Memory Estimate | 9794 MB |
| Phases | 15 |
| Phase Names | ["startup.dataset_manifest", "cache.latents", "cache.text_encoder", "training.prep.trainables", "training.prep.shared_precision", "training.prep.trainable_precision", "training.prep.optimizer_groups", "training.prep.lr_scheduler", "training.prep.accelerator", "training.prep.gradient_checkpointing", "startup.summary", "startup.metadata", "startup.runtime", "training.epoch.0", "checkpoint.save"] |
| Session Duration | 80.64s |
| Structural Facts | 10 |
| Trainable Parameter Memory | 4897 MB |

## Run Resource Accounting

| Resource | Quantity | Owner | Basis | Scope | Source Records |
|----------|----------|-------|-------|-------|----------------|
| parameter_memory | 234.719 MiB | `model_component:text_encoder1` | structural | run_startup_structure | 1 |
| trainable_parameter_memory | 0 MiB | `model_component:text_encoder1` | structural | run_startup_structure | 1 |
| parameter_memory | 1324.96 MiB | `model_component:text_encoder2` | structural | run_startup_structure | 1 |
| trainable_parameter_memory | 0 MiB | `model_component:text_encoder2` | structural | run_startup_structure | 1 |
| parameter_memory | 319.114 MiB | `model_component:vae` | structural | run_startup_structure | 1 |
| trainable_parameter_memory | 0 MiB | `model_component:vae` | structural | run_startup_structure | 1 |
| parameter_memory | 4897.05 MiB | `model_component:denoiser` | structural | run_startup_structure | 1 |
| trainable_parameter_memory | 4897.05 MiB | `model_component:denoiser` | structural | run_startup_structure | 1 |
| gradient_memory | 4897.05 MiB | `training_state:gradients` | estimated_structural | startup_estimate | 1 |
| optimizer_state_memory | 9794.1 MiB | `training_state:optimizer_state` | estimated_structural | startup_estimate | 1 |

## Unresolved Resource Accounting Gaps

| Resource | Quantity | Scope | Reason | Basis | Source Records |
|----------|----------|-------|--------|-------|----------------|
| gpu_memory | 15743.1 MiB | run:gpu_allocated_peak | no_accepted_accounting_for_profile_value | profile_value_minus_accounted | 1 |
| gpu_memory | 16678 MiB | run:gpu_reserved_peak | no_accepted_accounting_for_profile_value | profile_value_minus_accounted | 1 |
| gpu_memory | 227 MiB | run:gpu_used_peak | no_accepted_accounting_for_profile_value | profile_value_minus_accounted | 1 |

## Phase Resource Summary

| Phase | Duration | GPU Allocated | GPU Reserved | GPU Peak Allocated | GPU Used Peak | CPU RSS | CPU VMS |
|-------|----------|---------------|--------------|--------------------|---------------|---------|---------|
| `startup.dataset_manifest` | 0.07s | 0 MB -> 0 MB | 0 MB -> 0 MB | 0 MB | N/A | 881 MB -> 881 MB | 2736 MB -> 2736 MB |
| `cache.latents` | 0.39s | 321 MB -> 0 MB | 356 MB -> 0 MB | 321 MB | 227 MB | 20694 MB -> 20878 MB | 25820 MB -> 25904 MB |
| `cache.text_encoder` | 0.14s | 3184 MB -> 3184 MB | 3276 MB -> 3276 MB | 3184 MB | N/A | 19786 MB -> 19786 MB | 27662 MB -> 27662 MB |
| `training.prep.trainables` | 0.04s | 0 MB -> 0 MB | 0 MB -> 0 MB | 0 MB | N/A | 20902 MB -> 20904 MB | 25940 MB -> 25941 MB |
| `training.prep.shared_precision` | 1.21s | 0 MB -> 0 MB | 0 MB -> 0 MB | 0 MB | 227 MB | 20904 MB -> 17254 MB | 25941 MB -> 21138 MB |
| `training.prep.trainable_precision` | 0.23s | 0 MB -> 0 MB | 0 MB -> 0 MB | 0 MB | N/A | 17254 MB -> 17633 MB | 21138 MB -> 21506 MB |
| `training.prep.optimizer_groups` | 0.04s | 0 MB -> 0 MB | 0 MB -> 0 MB | 0 MB | N/A | 17633 MB -> 17633 MB | 21506 MB -> 21507 MB |
| `training.prep.lr_scheduler` | 0.01s | 0 MB -> 0 MB | 0 MB -> 0 MB | 0 MB | N/A | 17633 MB -> 17633 MB | 21507 MB -> 21507 MB |
| `training.prep.accelerator` | 1.23s | 0 MB -> 4990 MB | 0 MB -> 5168 MB | 4990 MB | 227 MB | 17633 MB -> 14748 MB | 21507 MB -> 23323 MB |
| `training.prep.gradient_checkpointing` | 0.01s | 4990 MB -> 4990 MB | 5168 MB -> 5168 MB | 4990 MB | N/A | 14748 MB -> 14748 MB | 23323 MB -> 23323 MB |
| `startup.summary` | 0.04s | 4990 MB -> 4990 MB | 5168 MB -> 5168 MB | 4990 MB | 227 MB | 14756 MB -> 14756 MB | 23331 MB -> 23331 MB |
| `startup.metadata` | 0.06s | 4990 MB -> 4990 MB | 5168 MB -> 5168 MB | 4990 MB | N/A | 14756 MB -> 14756 MB | 23331 MB -> 23331 MB |
| `startup.runtime` | 0.31s | 4990 MB -> 4990 MB | 5168 MB -> 5168 MB | 4990 MB | N/A | 14756 MB -> 14773 MB | 23331 MB -> 23347 MB |
| `training.epoch.0` | 62.87s | 4990 MB -> 10080 MB | 5168 MB -> 16678 MB | 15743 MB | 227 MB | 14774 MB -> 15732 MB | 23347 MB -> 35898 MB |
| `checkpoint.save` | 9.50s | 10080 MB -> 10080 MB | 16678 MB -> 16678 MB | 10136 MB | 227 MB | 15732 MB -> 18147 MB | 35898 MB -> 38587 MB |

## Resource Interpretation

These summaries interpret observed start/end/peak counters only. They do not claim component-level ownership or causality.

| Scope | Highlights |
|-------|------------|
| Session | GPU allocated ended 10080 MB higher than it started.; GPU reserved ended 16678 MB higher than it started.; CPU RSS ended 17266 MB higher than it started.; CPU VMS ended 35851 MB higher than it started.; GPU peak allocated exceeded both start and end snapshots, indicating a transient in-phase allocator peak.; GPU reserved grew more than GPU allocated, so allocator capacity expanded beyond the end-of-window live allocation level. |
| `cache.latents` | GPU allocated ended 321 MB lower than it started.; GPU reserved ended 356 MB lower than it started.; CPU RSS ended 183 MB higher than it started.; CPU VMS ended 84 MB higher than it started. |
| `training.prep.trainables` | CPU RSS ended 2 MB higher than it started.; CPU VMS ended 1 MB higher than it started. |
| `training.prep.shared_precision` | CPU RSS ended 3650 MB lower than it started.; CPU VMS ended 4803 MB lower than it started.; Sampled GPU used peaked above the allocator snapshots; this shows higher visible device usage during the window, not component-level ownership. |
| `training.prep.trainable_precision` | CPU RSS ended 379 MB higher than it started.; CPU VMS ended 368 MB higher than it started. |
| `training.prep.accelerator` | GPU allocated ended 4990 MB higher than it started.; GPU reserved ended 5168 MB higher than it started.; CPU RSS ended 2886 MB lower than it started.; CPU VMS ended 1816 MB higher than it started.; GPU reserved grew more than GPU allocated, so allocator capacity expanded beyond the end-of-window live allocation level. |
| `startup.runtime` | CPU RSS ended 17 MB higher than it started.; CPU VMS ended 16 MB higher than it started. |
| `training.epoch.0` | GPU allocated ended 5090 MB higher than it started.; GPU reserved ended 11510 MB higher than it started.; CPU RSS ended 958 MB higher than it started.; CPU VMS ended 12551 MB higher than it started.; GPU peak allocated exceeded both start and end snapshots, indicating a transient in-phase allocator peak.; GPU reserved grew more than GPU allocated, so allocator capacity expanded beyond the end-of-window live allocation level. |
| `checkpoint.save` | CPU RSS ended 2415 MB higher than it started.; CPU VMS ended 2689 MB higher than it started.; GPU peak allocated exceeded both start and end snapshots, indicating a transient in-phase allocator peak. |

## Per-Device GPU Session Peaks

| Device | GPU Used Peak |
|--------|---------------|
| `0` | 227 MB |

## Per-Device GPU Phase Details

| Phase | GPU Used Peak by Device | GPU Peak Allocated by Device |
|-------|-------------------------|------------------------------|
| `startup.dataset_manifest` | N/A | 0: 0 MB |
| `cache.latents` | 0: 227 MB | 0: 321 MB |
| `cache.text_encoder` | N/A | 0: 3184 MB |
| `training.prep.trainables` | N/A | 0: 0 MB |
| `training.prep.shared_precision` | 0: 227 MB | 0: 0 MB |
| `training.prep.trainable_precision` | N/A | 0: 0 MB |
| `training.prep.optimizer_groups` | N/A | 0: 0 MB |
| `training.prep.lr_scheduler` | N/A | 0: 0 MB |
| `training.prep.accelerator` | 0: 227 MB | 0: 4990 MB |
| `training.prep.gradient_checkpointing` | N/A | 0: 4990 MB |
| `startup.summary` | 0: 227 MB | 0: 4990 MB |
| `startup.metadata` | N/A | 0: 4990 MB |
| `startup.runtime` | N/A | 0: 4990 MB |
| `training.epoch.0` | 0: 227 MB | 0: 15743 MB |
| `checkpoint.save` | 0: 227 MB | 0: 10136 MB |

## Full Composed Config

The full resolved Hydra config is included here so the report is self-contained.

```yaml
'{''mode'': ''finetune'', ''training'': {''train_batch_size'': 2, ''max_token_length'':
  225, ''max_train_steps'': 50, ''max_train_epochs'': None, ''seed'': 42, ''gradient_accumulation_steps'':
  1, ''clip_skip'': None, ''dry_run'': False, ''initial_epoch'': None, ''initial_step'':
  None, ''skip_until_initial_step'': False}, ''optimizer'': {''optimizer_type'': ''AdamW8bit'',
  ''use_8bit_adam'': False, ''use_lion_optimizer'': False, ''learning_rates'': {''base'':
  None, ''denoiser'': 1e-05, ''text_encoders'': [0.0, 0.0], ''groups_file'': None,
  ''groups'': []}, ''scheduler'': {''lr_scheduler'': ''constant'', ''lr_scheduler_type'':
  '''', ''lr_scheduler_args'': [], ''lr_warmup_steps'': 0, ''lr_decay_steps'': 0,
  ''lr_scheduler_num_cycles'': 1, ''lr_scheduler_power'': 1.0, ''lr_scheduler_timescale'':
  None, ''lr_scheduler_min_lr_ratio'': None}, ''max_grad_norm'': 1.0, ''optimizer_args'':
  [''weight_decay=0.01'', ''betas=0.9, 0.999''], ''fused_backward_pass'': False, ''optimizer_schedulefree_wrapper'':
  False, ''schedulefree_wrapper_args'': None, ''fused_optimizer_groups'': None}, ''data'':
  {''source'': {''train_data_dir'': ''D:/Projects/sd-scripts/tests/assets/training_images'',
  ''reg_data_dir'': None, ''val_data_dir'': None, ''dataset_config'': None, ''in_json'':
  None, ''dataset_class'': None, ''dataset_repeats'': 1, ''subsets'': []}, ''preprocessing'':
  {''resolution'': ''1024,1024'', ''resize_interpolation'': None, ''alpha_mask'':
  False, ''flip_aug'': False, ''color_aug'': False, ''random_crop'': False, ''random_crop_padding_percent'':
  0.05, ''face_crop_aug_range'': None, ''debug_dataset'': False}, ''caption'': {''caption_extension'':
  ''.txt'', ''caption_extention'': None, ''caption_separator'': '','', ''shuffle_caption'':
  False, ''keep_tokens'': 0, ''keep_tokens_separator'': ''|||'', ''secondary_separator'':
  None, ''enable_wildcard'': False, ''caption_prefix'': None, ''caption_suffix'':
  None, ''caption_dropout_rate'': 0.0, ''caption_dropout_every_n_epochs'': 0, ''caption_tag_dropout_rate'':
  0.0, ''weighted_captions'': False, ''token_warmup_min'': 1, ''token_warmup_step'':
  0.0}, ''bucketing'': {''enable_bucket'': True, ''min_bucket_reso'': 256, ''max_bucket_reso'':
  1024, ''bucket_reso_steps'': 64, ''bucket_no_upscale'': False}, ''caching'': {''cache_dir'':
  ''D:/Projects/sd-scripts/benchmark_cache'', ''tokenizer_cache_dir'': None, ''cache_latents'':
  True, ''cache_latents_to_disk'': True, ''cache_text_encoder_outputs'': True, ''cache_text_encoder_outputs_to_disk'':
  True, ''disable_mmap_load_safetensors'': False, ''vae_batch_size'': 2, ''te_batch_size'':
  8, ''num_workers'': 4, ''skip_cache_check'': False, ''cache_tokens_per_epoch'':
  False}, ''loader'': {''num_workers'': 0, ''persistent_workers'': True, ''prefetch_factor'':
  2, ''pin_memory'': True}}, ''model'': {''model_type'': ''sdxl'', ''pretrained_model_name_or_path'':
  ''D:/stable-diffusion-webui-reForge/models/Stable-diffusion/21862-seele_pop3_v-pred.safetensors'',
  ''vae'': None, ''vae_conv2d_padding_mode'': ''zeros''}, ''objective'': {''path'':
  ''ddpm'', ''prediction'': ''v_prediction''}, ''adapter'': None, ''textual_inversion'':
  None, ''output'': {''saving'': {''output_dir'': ''D:/Projects/sd-scripts/outputs/rmst-baseline/finetune-run1'',
  ''output_name'': ''test_peft_resource_sampled'', ''save_precision'': ''fp16'', ''save_every_n_epochs'':
  999999, ''save_every_n_steps'': 999999, ''save_n_epoch_ratio'': None, ''save_last_n_epochs'':
  None, ''save_last_n_epochs_state'': None, ''save_last_n_steps'': None, ''save_last_n_steps_state'':
  None, ''save_state'': False, ''save_state_on_train_end'': False, ''resume'': None,
  ''save_model_as'': ''safetensors'', ''use_safetensors'': False, ''no_metadata'':
  False, ''hash_algorithm'': ''sha256''}, ''logging'': {''logging_dir'': ''D:/Projects/sd-scripts/outputs/rmst-baseline/finetune-run1/logs'',
  ''log_with'': None, ''log_prefix'': None, ''log_tracker_name'': None, ''wandb_run_name'':
  None, ''log_tracker_config'': {}, ''wandb_api_key'': None, ''log_config'': False,
  ''console_log_level'': None, ''console_log_file'': None, ''console_log_simple'':
  False, ''log_every_n_steps'': 1, ''log_timestep_distribution_every_n_steps'': None,
  ''live_plot_port'': None, ''benchmark_report'': {''enabled'': True, ''output_dir'':
  None, ''filename_prefix'': ''benchmark_report'', ''include_full_config'': True},
  ''resource_monitor'': {''enabled'': True, ''mode'': ''sampled'', ''log_every_n_steps'':
  0, ''sample_interval_sec'': 0.5, ''rank_scope'': ''main'', ''device_scope'': ''local'',
  ''output_jsonl'': ''resource_monitor.jsonl'', ''jsonl_flush_mode'': ''auto'', ''jsonl_flush_every_n_events'':
  50, ''queue_maxsize'': 1024, ''drop_policy'': ''drop_oldest'', ''max_collection_ms'':
  0.0, ''phase_summary'': ''default'', ''component_breakdown'': True, ''deep_window_steps'':
  0, ''deep_window_seconds'': 0.0}}, ''huggingface'': {''huggingface_repo_id'': None,
  ''huggingface_repo_type'': None, ''huggingface_path_in_repo'': None, ''huggingface_token'':
  None, ''huggingface_repo_visibility'': None, ''save_state_to_huggingface'': False,
  ''resume_from_huggingface'': False, ''async_upload'': False}, ''sampling'': {''sample_every_n_steps'':
  None, ''sample_at_first'': False, ''sample_every_n_epochs'': None, ''sample_prompt'':
  None, ''sample_prompt_file'': None, ''sample_negative_prompt'': None, ''sample_width'':
  None, ''sample_height'': None, ''sample_steps'': None, ''sample_cfg_scale'': None,
  ''sample_seed'': None, ''sample_sampler'': ''ddim'', ''sample_flow_shift'': None,
  ''sample_vae_dtype'': ''fp16''}, ''metadata'': {''training_comment'': None, ''metadata_title'':
  None, ''metadata_author'': None, ''metadata_description'': None, ''metadata_license'':
  None, ''metadata_tags'': None, ''metadata_usage_hint'': None, ''metadata_thumbnail'':
  None, ''metadata_merged_from'': None, ''metadata_trigger_phrase'': None, ''metadata_preprocessor'':
  None, ''metadata_is_negative_embedding'': None}}, ''performance'': {''precision'':
  {''mixed_precision'': ''bf16'', ''full_fp16'': False, ''full_bf16'': False, ''fp8_base'':
  False, ''fp8_base_unet'': False, ''no_half_vae'': True, ''disable_cuda_reduced_precision_operations'':
  False, ''enable_cuda_reduced_precision_operations'': False}, ''memory'': {''gradient_checkpointing'':
  True, ''cpu_offload_checkpointing'': False, ''lowram'': False, ''highvram'': False,
  ''use_ramtorch'': False, ''direct_ramtorch'': False, ''offload_text_encoders'':
  False}, ''attention'': {''mem_eff_attn'': False, ''xformers'': True, ''sdpa'': False,
  ''diffusers_xformers'': False}, ''compilation'': {''torch_compile'': False, ''dynamo_backend'':
  ''inductor''}, ''distributed'': {''ddp_timeout'': None, ''ddp_gradient_as_bucket_view'':
  False, ''ddp_static_graph'': False}, ''deepspeed'': {''deepspeed'': False, ''zero_stage'':
  2, ''offload_optimizer_device'': None, ''offload_optimizer_nvme_path'': None, ''offload_param_device'':
  None, ''offload_param_nvme_path'': None, ''zero3_init_flag'': False, ''zero3_save_16bit_model'':
  False, ''fp16_master_weights_and_gradients'': False}}, ''loss'': {''loss_type'':
  ''l2'', ''loss_scale'': 1.0, ''loss_multiplier'': None, ''prior_loss_weight'': 1.0,
  ''v_parameterization'': True, ''huber'': {''huber_schedule'': ''snr'', ''huber_c'':
  0.1, ''huber_scale'': 1.0}, ''snr'': {''min_snr_gamma'': 5.0, ''scale_v_pred_loss_like_noise_pred'':
  False, ''v_pred_like_loss'': None, ''debiased_estimation_loss'': False}, ''masked'':
  {''masked_loss'': False, ''conditioning_data_dir'': None}, ''regularization'': {''noise_offset'':
  None, ''noise_offset_random_strength'': False, ''multires_noise_iterations'': None,
  ''multires_noise_discount'': 0.3, ''ip_noise_gamma'': None, ''ip_noise_gamma_random_strength'':
  False, ''adaptive_noise_scale'': None, ''zero_terminal_snr'': True}, ''edm2'': {''enabled'':
  False, ''laplace_timestep_sampling'': False, ''num_channels'': 128, ''initial_weights'':
  None, ''optimizer'': {''type'': ''torch.optim.AdamW'', ''lr'': 0.02, ''args'': "{''weight_decay'':
  0, ''betas'': (0.9,0.999)}", ''use_scheduler'': False, ''warmup_percent'': 0.1,
  ''constant_percent'': 0.1, ''decay_scaling'': 1.0}, ''importance'': {''enabled'':
  False, ''max_weight'': 10.0, ''min_snr_gamma'': 1.0, ''safety_override'': False},
  ''visualization'': {''enabled'': False, ''every_n_steps'': 20, ''output_dir'': None,
  ''y_limit'': None, ''y_scale'': ''linear''}}}, ''timestep'': {''timestep_sampling'':
  ''uniform'', ''min_timestep'': 0, ''max_timestep'': 1000, ''dynamic_timestep_schedule'':
  None, ''training_shift'': 1.0, ''rf_loss_weighting_scheme'': ''uniform'', ''logit_mean'':
  0.0, ''logit_std'': 1.0, ''cosine_shape_scale'': 1.29, ''adaptive_log_snr'': {''bins'':
  32, ''ema_beta'': 0.9, ''temperature'': 0.5, ''prior_weight'': 0.25, ''min_prob'':
  0.0001, ''warmup_steps'': 2000, ''entropy_floor'': 0.7, ''uniform_mix_when_low_entropy'':
  0.1}}, ''validation'': {''validation_split'': 0.0, ''validation_seed'': None, ''run_at_start'':
  False, ''run_at_end'': False, ''validate_every_n_steps'': None, ''validate_every_n_epochs'':
  None, ''max_validation_steps'': None, ''validation_timesteps'': ''[50, 350, 500,
  650, 950]''}}'
```
