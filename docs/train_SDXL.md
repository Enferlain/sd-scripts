# SDXL Training Guide

This document provides a brief explanation of the training scripts for SDXL.

## Training Scripts for SDXL

### `sdxl_train.py`
This script is used for SDXL fine-tuning. Its usage is similar to `fine_tune.py`, but it also supports DreamBooth datasets.

**Key Options:**
- `--full_bf16`: Enables full bfloat16 training, including gradients. This can help reduce GPU memory usage but may be unstable. Use at your own risk.
- `--block_lr`: Sets different learning rates for each of the 23 blocks in the U-Net. The values should be comma-separated (e.g., `--block_lr 1e-3,1e-3,...,1e-3`). The blocks are indexed as follows:
  - `0`: time/label embed
  - `1-9`: input blocks 0-8
  - `10-12`: mid blocks 0-2
  - `13-21`: output blocks 0-8
  - `22`: out
- `--cache_text_encoder_outputs` and `--cache_text_encoder_outputs_to_disk`: Cache the outputs of the text encoders to reduce GPU memory usage. This option cannot be used with caption shuffling or dropout.
- `--no_half_vae`: Disables the half-precision (mixed-precision) VAE to prevent potential NaN issues.
- `--min_timestep` and `--max_timestep`: Sets the minimum and maximum timesteps for training the U-Net (default: 0 and 1000).

### `sdxl_train_network.py`
This script is used for LoRA training for SDXL. Its usage is similar to `train_network.py`.

### `sdxl_train_textual_inversion.py`
This script is used for Textual Inversion training for SDXL. Its usage is similar to `train_textual_inversion.py`.
- `--cache_text_encoder_outputs` is not supported.
- Captions can be provided in two ways:
  1.  **With captions**: All captions must include the token string, which will be replaced with multiple tokens.
  2.  **With templates**: Use `--use_object_template` or `--use_style_template` to generate captions from a template, ignoring existing captions.

## Utility Scripts for SDXL

### `tools/cache_latents.py`
This script caches latents to disk in advance. The options are similar to `sdxl_train.py`.
- **Usage**: `accelerate launch --num_cpu_threads_per_process 1 tools/cache_latents.py ...`

### `tools/cache_text_encoder_outputs.py`
This script caches text encoder outputs to disk in advance. The options are similar to `sdxl_train.py`.

### `sdxl_gen_img.py`
This script generates images with SDXL, supporting LoRA, Textual Inversion, and ControlNet-LLLite.

## Tips for SDXL Training

- The default resolution for SDXL is 1024x1024.
- **Fine-tuning on a 24GB GPU**:
  - Train the U-Net only.
  - Use gradient checkpointing.
  - Use `--cache_text_encoder_outputs` and cache latents.
  - Use the Adafactor optimizer. AdamW 8-bit may not work.
- **LoRA training on an 8-10GB GPU**:
  - Train the U-Net only (`--network_train_unet_only` is recommended).
  - Use gradient checkpointing.
  - Use `--cache_text_encoder_outputs` and cache latents.
  - Use an 8-bit optimizer or Adafactor.
  - Use a lower network dimension (e.g., 4-8 for 8GB GPU).
- Set `--bucket_reso_steps` to 32 (default is 64). Smaller values are not supported for SDXL.
- PyTorch 2 may use slightly less GPU memory than PyTorch 1.

### Example Optimizer Settings (Adafactor)
```toml
optimizer_type = "adafactor"
optimizer_args = [ "scale_parameter=False", "relative_step=False", "warmup_init=False" ]
lr_scheduler = "constant_with_warmup"
lr_warmup_steps = 100
learning_rate = 4e-7 # SDXL original learning rate
```

## Textual Inversion Embeddings Format
```python
from safetensors.torch import save_file

state_dict = {"clip_g": embs_for_text_encoder_1280, "clip_l": embs_for_text_encoder_768}
save_file(state_dict, file)
```
