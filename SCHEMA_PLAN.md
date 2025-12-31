# Dataclass Field Mapping to Schema 1

Total: **~200 fields** across 17 dataclasses

## 🔜 Remaining

| Task                          | Notes                                                                                         |
| ----------------------------- | --------------------------------------------------------------------------------------------- |
| ~~`model` restructuring~~     | ✅ Done: Added `model_type` field, removed `v2`                                               |
| ~~`training` restructuring~~  | ✅ Validation separated to top-level `ValidationConfig`                                       |
| ~~`optimizer` restructuring~~ | ✅ Done: `scheduler` subcategory nested (alongside `learning_rates`)                          |
| ~~`data` restructuring~~      | ✅ Done: Split into 5 nested sub-configs (source, preprocessing, caption, bucketing, caching) |
| ~~`timestep` restructuring~~  | ✅ Done: Per-sampler nested configs (mix_adaptive, tempered_adaptive, etc.)                   |

## 🔮 Future Design Goals

### Self-Documenting Configs

Add explicit fields so configs are identifiable at a glance:

- **`model.model_type`**: `sd15 | sd2 | sdxl | flux` - which base model architecture
- **`training.method`**: `peft | finetune | textual_inversion` - what training approach

Example:

```yaml
model:
  model_type: sdxl
  pretrained_model_name_or_path: "stabilityai/sdxl-base-1.0"

training:
  method: peft
  max_train_epochs: 10
```

This enables:

1. Configs are self-documenting (no need to infer from filename)
2. Potential unified entry point script that dispatches based on these fields
3. Better validation (e.g., `peft:` section only valid when `training.method: peft`)

### Other Considerations

- Might need to consider renaming redundant settings after the nested restructuring (e.g., `huber.huber_schedule` → `huber.schedule`)

---

## Execution Strategy

### What Makes This Harder Than a Typical Rename

| Layer                             | Tool Support                  |
| --------------------------------- | ----------------------------- |
| Python dataclass field names      | ✅ PyCharm "Rename Symbol"    |
| YAML config keys                  | ❌ Manual / script            |
| Config access in code (`cfg.X.Y`) | ⚠️ PyCharm partial (if typed) |
| Documentation / comments          | ❌ Manual / grep              |

### Actual Approach: Direct Migration (Breaking Changes)

> **Note**: We initially considered using `@property` aliases for backward compatibility, but this doesn't work well with Hydra/OmegaConf since YAML keys must match actual dataclass field names.

**What we did instead:**

1. **Remove old fields directly** from dataclasses (`unet_lr`, `text_encoder_lr`, `block_lr`, etc.)
2. **Add new structure** (`LearningRatesConfig` in `OptimizerConfig`)
3. **Update YAML configs** to use new keys
4. **Find/replace in code** to use new paths (`cfg.peft.*`, `cfg.optimizer.learning_rates.*`)
5. **Update tests** for new config structure
6. **Run tests** to verify nothing breaks

**Trade-off**: Breaking change for users, but cleaner codebase without compatibility layers.

### Tooling Options

| Tool                    | Use For                     |
| ----------------------- | --------------------------- |
| PyCharm "Rename Symbol" | Python class/field renames  |
| `grep -r "old_name"`    | Find all references         |
| Custom Python script    | YAML key transformation     |
| `libcst`                | AST-safe Python refactoring |
| Tests                   | Verify nothing breaks       |

### Practical Script for YAML Migration

```python
# Example: Transform old YAML keys to new structure
import yaml

def migrate_config(old_config: dict) -> dict:
    new_config = {}

    # Move peft.unet_lr → optimizer.learning_rates.unet
    if 'peft' in old_config and 'unet_lr' in old_config['peft']:
        new_config.setdefault('optimizer', {})
        new_config['optimizer'].setdefault('learning_rates', {})
        new_config['optimizer']['learning_rates']['unet'] = old_config['peft']['unet_lr']

    # ... etc for all mappings
    return new_config
```

### Recommendation

Start with **one script** (`sd_peft.py`) as a prototype:

1. Create new dataclass structure for just PEFT mode
2. Add compatibility layer
3. Migrate that script
4. Validate with tests
5. Then expand to other scripts

---

## Open Questions / Future Work

### PEFT Field Naming

Current `network_*` fields in `NetworkConfig`:

- `network_dim` → keep as `dim` (drop prefix in peft context)
- `network_alpha` → keep as `alpha`
- `adapter_module` → keep as `module`
- `network_args` → keep as `args`
- `network_dropout` → `neuron_dropout` (clarity vs rank/module dropout)

**Decision**: Drop `network_` prefix, but use specific names like `neuron_dropout` where generic names lose context.

### Undocumented kwargs in lora.py

`create_adapter()` accepts many kwargs **not exposed in config**:

| kwarg                            | Type  | Purpose                   |
| -------------------------------- | ----- | ------------------------- |
| `conv_dim`                       | int   | Conv layer LoRA rank      |
| `conv_alpha`                     | float | Conv layer alpha          |
| `block_dims`                     | str   | Per-block rank string     |
| `block_alphas`                   | str   | Per-block alpha string    |
| `conv_block_dims`                | str   | Per-block conv rank       |
| `conv_block_alphas`              | str   | Per-block conv alpha      |
| `rank_dropout`                   | float | Dropout on rank           |
| `module_dropout`                 | float | Dropout on whole module   |
| `loraplus_lr_ratio`              | float | LoRA+ learning rate ratio |
| `loraplus_unet_lr_ratio`         | float | LoRA+ UNet LR ratio       |
| `loraplus_text_encoder_lr_ratio` | float | LoRA+ TE LR ratio         |

**TODO**: Should these be explicit config fields instead of `network_args: ["conv_dim=8"]`?

### LyCORIS Vendoring

**Current**: LyCORIS installed as package (`pip install lycoris-lora`)

**Consider**: Vendoring LyCORIS into `library/vendor/lycoris/`

| Pros                   | Cons                     |
| ---------------------- | ------------------------ |
| No external dependency | Maintenance burden       |
| Can patch/customize    | Larger repo              |
| Version stability      | May fall behind upstream |

**Decision**: TBD - add to roadmap for evaluation.

---

## Summary by Current Config

| Current Dataclass      | Fields | Maps To (Schema 1)                            |
| ---------------------- | ------ | --------------------------------------------- |
| `TrainingConfig`       | 21     | `training`                                    |
| `DatasetConfig`        | 37     | `data.source`, `data.processing`              |
| `BucketsConfig`        | 5      | `data.batching`                               |
| `ModelConfig`          | 5      | `model`                                       |
| `SavingConfig`         | 16     | `output.saving`                               |
| `LoggingConfig`        | 13     | `output.logging`                              |
| `PerformanceConfig`    | 24     | `performance.precision`, `performance.memory` |
| `LossConfig`           | 33     | `loss` (new category)                         |
| `TimestepConfig`       | 28     | `timestep` (new category)                     |
| `RegularizationConfig` | 8      | `regularization` (merge into `loss`?)         |
| `MetadataConfig`       | 11     | `output.metadata`                             |
| `SamplingConfig`       | 5      | `output.sampling`                             |
| `MaskedLossConfig`     | 2      | `loss.masked`                                 |
| `HuggingFaceConfig`    | 8      | `output.huggingface`                          |
| `DeepSpeedConfig`      | 9      | `performance.deepspeed`                       |
| `NetworkConfig`        | 14     | `mode.peft`                                   |
| `SDXLConfig`           | 8      | Split (see below)                             |

---

## Detailed Field Mapping

### `model` (from ModelConfig) 🔜

```yaml
model:
  path: pretrained_model_name_or_path
  vae: vae
  vae_padding_mode: vae_conv2d_padding_mode
  tokenizer_cache_dir: tokenizer_cache_dir
  v2: v2 # SD2 flag
```

**Notes**: Clean 1:1 mapping.

---

### `peft` (from NetworkConfig) ✅

```yaml
mode:
  type: peft
  peft:
    module: module
    dim: dim
    alpha: alpha
    dropout: neuron_dropout
    weights: weights
    args: args
    dim_from_weights: dim_from_weights
    scale_weight_norms: scale_weight_norms
    base_weights: base_weights
    base_weights_multiplier: base_weights_multiplier
    training_comment: training_comment
    orthograd_targets: orthograd_targets
```

**Removed** (replaced by LR = 0):

- `network_train_unet_only`
- `network_train_text_encoder_only`

**Moved to `optimizer.learning_rates`**:

- `unet_lr` → `optimizer.learning_rates.unet`
- `text_encoder_lr` → `optimizer.learning_rates.text_encoders`

---

### `optimizer` (from OptimizerConfig + LR fields) 🔜

```yaml
optimizer:
  type: optimizer_type
  use_8bit_adam: use_8bit_adam
  use_lion: use_lion_optimizer
  max_grad_norm: max_grad_norm
  args: optimizer_args
  fused_backward_pass: fused_backward_pass
  schedulefree_wrapper: optimizer_schedulefree_wrapper
  schedulefree_wrapper_args: schedulefree_wrapper_args

  learning_rates:
    base: learning_rate # from OptimizerConfig
    unet: unet_lr # from NetworkConfig
    text_encoders: text_encoder_lr # from NetworkConfig (list for multi-TE)
    # Consolidated from SDXLConfig learning_rate_te1/te2
    blocks: block_lr # from SDXLConfig

  scheduler:
    type: lr_scheduler
    custom_module: lr_scheduler_type
    args: lr_scheduler_args
    warmup_steps: lr_warmup_steps
    decay_steps: lr_decay_steps
    num_cycles: lr_scheduler_num_cycles
    power: lr_scheduler_power
    timescale: lr_scheduler_timescale
    min_lr_ratio: lr_scheduler_min_lr_ratio
```

---

### `training` (from TrainingConfig) 🔜

```yaml
training:
  seed: seed
  max_steps: max_train_steps
  max_epochs: max_train_epochs
  gradient_accumulation_steps: gradient_accumulation_steps
  batch_size: train_batch_size # could also go to data.batching
  max_token_length: max_token_length
  clip_skip: clip_skip
  dry_run: dry_run
  initial_epoch: initial_epoch
  initial_step: initial_step
  skip_until_initial_step: skip_until_initial_step

  validation:
    every_n_steps: validate_every_n_steps
    every_n_epochs: validate_every_n_epochs
    max_steps: max_validation_steps
    timesteps: validation_timesteps
```

**Removed** (meta/internal):

- `config_file`, `output_config`

**Moved to `data.loader`**:

- `max_data_loader_n_workers` → `data.loader.max_workers`
- `persistent_data_loader_workers` → `data.loader.persistent_workers`

---

### `data.source` (from DatasetConfig) 🔜

```yaml
data:
  source:
    train_dir: train_data_dir
    reg_dir: reg_data_dir
    config: dataset_config
    json: in_json
    class: dataset_class
    repeats: dataset_repeats
    subsets: subsets
```

---

### `data.processing` (from DatasetConfig) 🔜

```yaml
data:
  processing:
    resolution: resolution
    flip_aug: flip_aug
    color_aug: color_aug
    random_crop: random_crop
    face_crop_aug_range: face_crop_aug_range
    alpha_mask: alpha_mask
    resize_interpolation: resize_interpolation
    cache_latents: cache_latents
    cache_latents_to_disk: cache_latents_to_disk
    vae_batch_size: vae_batch_size
    skip_cache_check: skip_cache_check
    cache_info: cache_info

  captions:
    extension: caption_extension
    separator: caption_separator
    shuffle: shuffle_caption
    keep_tokens: keep_tokens
    keep_tokens_separator: keep_tokens_separator
    secondary_separator: secondary_separator
    wildcard: enable_wildcard
    prefix: caption_prefix
    suffix: caption_suffix
    dropout_rate: caption_dropout_rate
    dropout_every_n_epochs: caption_dropout_every_n_epochs
    tag_dropout_rate: caption_tag_dropout_rate
    weighted: weighted_captions
    token_warmup_min: token_warmup_min
    token_warmup_step: token_warmup_step

  validation:
    split: validation_split
    seed: validation_seed
```

---

### `data.batching` (from BucketsConfig + partial DatasetConfig) 🔜

```yaml
data:
  batching:
    bucket_enabled: enable_bucket
    bucket_min_reso: min_bucket_reso
    bucket_max_reso: max_bucket_reso
    bucket_reso_steps: bucket_reso_steps
    bucket_no_upscale: bucket_no_upscale
    debug_dataset: debug_dataset # from DatasetConfig
```

---

### `output.saving` (from SavingConfig) ✅

```yaml
output:
  saving:
    dir: output_dir
    name: output_name
    precision: save_precision
    format: save_model_as
    safetensors: use_safetensors
    no_metadata: no_metadata
    every_n_epochs: save_every_n_epochs
    every_n_steps: save_every_n_steps
    n_epoch_ratio: save_n_epoch_ratio
    last_n_epochs: save_last_n_epochs
    last_n_epochs_state: save_last_n_epochs_state
    last_n_steps: save_last_n_steps
    last_n_steps_state: save_last_n_steps_state
    state: save_state
    state_on_train_end: save_state_on_train_end
    resume: resume
```

---

### `output.logging` (from LoggingConfig) ✅

```yaml
output:
  logging:
    dir: logging_dir
    with: log_with
    prefix: log_prefix
    tracker_name: log_tracker_name
    tracker_config: log_tracker_config
    wandb_run_name: wandb_run_name
    wandb_api_key: wandb_api_key
    log_config: log_config
    console_level: console_log_level
    console_file: console_log_file
    console_simple: console_log_simple
    timestep_dist_every_n_steps: log_timestep_distribution_every_n_steps
    live_plot_port: live_plot_port
```

---

### `output.huggingface` (from HuggingFaceConfig) ✅

```yaml
output:
  huggingface:
    repo_id: huggingface_repo_id
    repo_type: huggingface_repo_type
    path_in_repo: huggingface_path_in_repo
    token: huggingface_token
    visibility: huggingface_repo_visibility
    save_state: save_state_to_huggingface
    resume: resume_from_huggingface
    async: async_upload
```

---

### `output.sampling` (from SamplingConfig) ✅

```yaml
output:
  sampling:
    every_n_steps: sample_every_n_steps
    every_n_epochs: sample_every_n_epochs
    at_first: sample_at_first
    prompts: sample_prompts
    sampler: sample_sampler
```

---

### `output.metadata` (from MetadataConfig) ✅

```yaml
output:
  metadata:
    title: metadata_title
    author: metadata_author
    description: metadata_description
    license: metadata_license
    tags: metadata_tags
    usage_hint: metadata_usage_hint
    thumbnail: metadata_thumbnail
    merged_from: metadata_merged_from
    trigger_phrase: metadata_trigger_phrase
    preprocessor: metadata_preprocessor
    is_negative_embedding: metadata_is_negative_embedding
```

---

### `performance.precision` (from PerformanceConfig) ✅

```yaml
performance:
  precision:
    mixed: mixed_precision
    full_fp16: full_fp16
    full_bf16: full_bf16
    fp8_base: fp8_base
    fp8_base_unet: fp8_base_unet
    no_half_vae: no_half_vae
    disable_cuda_reduced_precision: disable_cuda_reduced_precision_operations
    enable_cuda_reduced_precision: enable_cuda_reduced_precision_operations
```

---

### `performance.memory` (from PerformanceConfig) ✅

```yaml
performance:
  memory:
    gradient_checkpointing: gradient_checkpointing
    cpu_offload_checkpointing: cpu_offload_checkpointing
    lowram: lowram
    highvram: highvram
    ramtorch: use_ramtorch
    direct_ramtorch: direct_ramtorch

  attention:
    mem_eff: mem_eff_attn
    xformers: xformers
    sdpa: sdpa
    diffusers_xformers: diffusers_xformers

  compilation:
    torch_compile: torch_compile
    dynamo_backend: dynamo_backend

  distributed:
    ddp_timeout: ddp_timeout
    ddp_gradient_as_bucket_view: ddp_gradient_as_bucket_view
    ddp_static_graph: ddp_static_graph
```

---

### `performance.deepspeed` (from DeepSpeedConfig) ✅

```yaml
performance:
  deepspeed:
    enabled: deepspeed
    zero_stage: zero_stage
    offload_optimizer_device: offload_optimizer_device
    offload_optimizer_nvme_path: offload_optimizer_nvme_path
    offload_param_device: offload_param_device
    offload_param_nvme_path: offload_param_nvme_path
    zero3_init: zero3_init_flag
    zero3_save_16bit: zero3_save_16bit_model
    fp16_master_weights: fp16_master_weights_and_gradients
```

---

### `loss` (from LossConfig + MaskedLossConfig + RegularizationConfig) ✅

```yaml
loss:
  type: loss_type
  scale: loss_scale
  multiplier: loss_multiplier
  prior_weight: prior_loss_weight
  v_parameterization: v_parameterization

  huber:
    schedule: huber_schedule
    c: huber_c
    scale: huber_scale

  snr:
    min_gamma: min_snr_gamma
    scale_v_pred: scale_v_pred_loss_like_noise_pred
    v_pred_like: v_pred_like_loss
    debiased_estimation: debiased_estimation_loss

  masked:
    enabled: masked_loss
    conditioning_dir: conditioning_data_dir

  regularization:
    noise_offset: noise_offset
    noise_offset_random_strength: noise_offset_random_strength
    multires_noise_iterations: multires_noise_iterations
    multires_noise_discount: multires_noise_discount
    ip_noise_gamma: ip_noise_gamma
    ip_noise_gamma_random_strength: ip_noise_gamma_random_strength
    adaptive_noise_scale: adaptive_noise_scale
    zero_terminal_snr: zero_terminal_snr

  edm2:
    enabled: edm2_loss_weighting
    laplace: edm2_loss_weighting_laplace
    optimizer: edm2_loss_weighting_optimizer
    optimizer_lr: edm2_loss_weighting_optimizer_lr
    optimizer_args: edm2_loss_weighting_optimizer_args
    # ... (remaining 14 EDM2 fields)
```

---

### `timestep` (from TimestepConfig) 🔜

```yaml
timestep:
  min: min_timestep
  max: max_timestep
  dynamic_schedule: dynamic_timestep_schedule
  sampling: timestep_sampling
  sigmoid_scale: sigmoid_scale
  discrete_flow_shift: discrete_flow_shift

  mix_adaptive:
    start_p: mix_adaptive_start_p
    end_p: mix_adaptive_end_p
    fixed_p: mix_adaptive_fixed_p
    # ... (remaining 18 mix_adaptive fields)
```

---

### SDXLConfig Dissolution ✅

| SDXLConfig Field                     | New Location                                | Rationale         |
| ------------------------------------ | ------------------------------------------- | ----------------- |
| `cache_text_encoder_outputs`         | `performance.memory`                        | Generic feature   |
| `cache_text_encoder_outputs_to_disk` | `performance.memory`                        | Generic feature   |
| `disable_mmap_load_safetensors`      | `performance.memory`                        | Generic feature   |
| `learning_rate_te1`                  | `optimizer.learning_rates.text_encoders[0]` | Consolidated LRs  |
| `learning_rate_te2`                  | `optimizer.learning_rates.text_encoders[1]` | Consolidated LRs  |
| `train_text_encoder`                 | Implicit (LR > 0)                           | Redundant with LR |
| `block_lr`                           | `optimizer.learning_rates.blocks`           | Consolidated LRs  |
| `fused_optimizer_groups`             | `optimizer.fused_groups`                    | Optimizer setting |

**Result**: `SDXLConfig` becomes empty and can be removed.

---

## Schema 1 Top-Level Structure

```
model:           # 5 fields
mode:            # varies by type (peft: 12, finetune: TBD, ti: TBD)
optimizer:       # ~20 fields (with nested learning_rates + scheduler)
training:        # ~15 fields (core loop settings)
data:            # ~50 fields (source + processing + captions + batching)
output:          # ~45 fields (saving + logging + huggingface + sampling + metadata)
performance:     # ~30 fields (precision + memory + attention + deepspeed)
loss:            # ~45 fields (core + huber + snr + masked + regularization + edm2)
timestep:        # ~28 fields (basic + mix_adaptive)
```

**Total: ~230 fields** (some consolidation/removal expected)
