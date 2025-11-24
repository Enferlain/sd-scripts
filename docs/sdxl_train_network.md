# SDXL LoRA Training Guide

This document explains how to train a LoRA (Low-Rank Adaptation) model for SDXL using the `sdxl_train_network.py` script.

## Introduction

`sdxl_train_network.py` is a script for training additional networks like LoRA for SDXL models. The basic usage is similar to `train_network.py`, but with some SDXL-specific settings. This guide focuses on the main differences and SDXL-specific configurations.

**Prerequisites:**
- You have cloned the `sd-scripts` repository and set up the Python environment.
- Your training dataset is ready.
- You are familiar with the basic usage of `train_network.py`.

## Running the Training

To start training, run `sdxl_train_network.py` from your terminal. Here is a basic command-line example:

```bash
accelerate launch --num_cpu_threads_per_process 1 sdxl_train_network.py \
 --pretrained_model_name_or_path="<SDXL base model path>" \
 --dataset_config="my_sdxl_dataset_config.toml" \
 --output_dir="<output directory for training results>" \
 --output_name="my_sdxl_lora" \
 --save_model_as=safetensors \
 --network_module=networks.lora \
 --network_dim=32 \
 --network_alpha=16 \
 --learning_rate=1e-4 \
 --unet_lr=1e-4 \
 --text_encoder_lr1=1e-5 \
 --text_encoder_lr2=1e-5 \
 --optimizer_type="AdamW8bit" \
 --lr_scheduler="constant" \
 --max_train_epochs=10 \
 --save_every_n_epochs=1 \
 --mixed_precision="bf16" \
 --gradient_checkpointing \
 --cache_text_encoder_outputs \
 --cache_latents
```

### Key Differences from `train_network.py`
- The script to execute is `sdxl_train_network.py`.
- `--pretrained_model_name_or_path` should point to an SDXL base model.
- `--text_encoder_lr` is split into `--text_encoder_lr1` and `--text_encoder_lr2` for the two text encoders in SDXL.
- `--mixed_precision` is recommended to be `bf16` or `fp16`.
- `--cache_text_encoder_outputs` and `--cache_latents` are recommended to reduce VRAM usage.

## Main Command-Line Arguments

### Model-Related
- `--pretrained_model_name_or_path="<model path>"` **[Required]**
  - Specifies the SDXL model to be used as the base for training. This can be a Hugging Face Hub model ID, a local Diffusers format model directory, or a path to a `.safetensors` file.

### Training Parameters
- `--learning_rate=1e-4`
  - The overall learning rate. This is the default if `unet_lr`, `text_encoder_lr1`, and `text_encoder_lr2` are not specified.
- `--unet_lr=1e-4`
  - The learning rate for LoRA modules in the U-Net.
- `--text_encoder_lr1=1e-5`
  - The learning rate for LoRA modules in Text Encoder 1 (OpenCLIP ViT-G/14). A smaller value than the U-Net is recommended.
- `--text_encoder_lr2=1e-5`
  - The learning rate for LoRA modules in Text Encoder 2 (CLIP ViT-L/14). A smaller value than the U-Net is recommended.
- `--mixed_precision="bf16"`
  - The mixed precision training setting. For SDXL, `bf16` or `fp16` is recommended to reduce VRAM usage and improve training speed.
- `--gradient_checkpointing`
  - Recommended for SDXL due to its high memory consumption.
- `--cache_latents`
  - Caches VAE outputs in memory (or on disk with `--cache_latents_to_disk`). This reduces VRAM usage and speeds up training by skipping VAE computation. Image augmentations are disabled with this option.
- `--cache_text_encoder_outputs`
  - Caches Text Encoder outputs in memory (or on disk with `--cache_text_encoder_outputs_to_disk`). This reduces VRAM usage and speeds up training by skipping Text Encoder computation. Caption augmentations are disabled with this option.
  - **Note:** When using this option, LoRA modules for the Text Encoder cannot be trained (`--network_train_unet_only` must be specified).
- `--no_half_vae`
  - Runs the VAE in `float32` even when using mixed precision. This is recommended when using `fp16` as the SDXL VAE can be unstable in `float16`.
- `--fused_backward_pass`
  - Fuses the gradient computation and optimizer steps to reduce VRAM usage. Currently only supports the `Adafactor` optimizer.

## Using the Trained Model

When training is complete, the LoRA model file will be saved in the directory specified by `output_dir`. This file can be used with any GUI tool that supports SDXL.
