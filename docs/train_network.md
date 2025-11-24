# LoRA Training Guide: `train_network.py`

This document explains the basic procedures for training LoRA (Low-Rank Adaptation) models using the `train_network.py` script.

## Introduction

`train_network.py` is a script for training additional networks like LoRA on Stable Diffusion models (v1.x, v2.x). It allows for efficient fine-tuning of the original model to reproduce specific characters or art styles.

**Prerequisites:**
- The `sd-scripts` repository has been cloned and the Python environment is set up.
- The training dataset has been prepared.

## Preparation

Before starting, you will need the following:
1.  **Training script:** `train_network.py`
2.  **Dataset definition file (.toml):** A TOML file describing the configuration of your training dataset, including image directories, repetition counts, captions, and resolution bucketing.

## Running the Training

Training is initiated by running `train_network.py` from the terminal with various command-line arguments.

Here is a basic command-line example:

```bash
accelerate launch --num_cpu_threads_per_process 1 train_network.py \
 --pretrained_model_name_or_path="<path to Stable Diffusion model>" \
 --dataset_config="my_dataset_config.toml" \
 --output_dir="<output directory for training results>" \
 --output_name="my_lora" \
 --save_model_as=safetensors \
 --network_module=networks.lora \
 --network_dim=16 \
 --network_alpha=1 \
 --learning_rate=1e-4 \
 --optimizer_type="AdamW8bit" \
 --lr_scheduler="constant" \
 --sdpa \
 --max_train_epochs=10 \
 --save_every_n_epochs=1 \
 --mixed_precision="fp16" \
 --gradient_checkpointing
```

## Main Command-Line Arguments

### Model-Related
- `--pretrained_model_name_or_path="<path to model>"` **[Required]**
  - Specifies the base Stable Diffusion model. This can be a local `.ckpt` or `.safetensors` file, a Diffusers format directory, or a Hugging Face Hub model ID.
- `--v2`
  - Specify this if the base model is Stable Diffusion v2.x.
- `--v_parameterization`
  - Specify this when training with a v-prediction model (e.g., v2.x 768px models).

### Dataset-Related
- `--dataset_config="<path to configuration file>"`
  - Specifies the path to the `.toml` file for your dataset configuration.

### Output and Save-Related
- `--output_dir="<output directory>"` **[Required]**
  - The directory where trained LoRA models, sample images, and logs will be saved.
- `--output_name="<output filename>"` **[Required]**
  - The filename for the trained LoRA model (without the extension).
- `--save_model_as="safetensors"`
  - The format for saving the model. Options are `safetensors` (recommended), `ckpt`, or `pt`.
- `--save_every_n_epochs=1`
  - Saves the model every specified number of epochs.
- `--save_every_n_steps=1000`
  - Saves the model every specified number of steps.

### LoRA Parameters
- `--network_module=networks.lora` **[Required]**
  - The type of network to train. For LoRA, use `networks.lora`.
- `--network_dim=16` **[Required]**
  - The rank (dimension) of the LoRA. Higher values increase expressiveness but also file size. Common values are between 4 and 128.
- `--network_alpha=1`
  - The alpha value for LoRA, which relates to learning rate scaling. It is often set to half of `network_dim`.

### Training Parameters
- `--learning_rate=1e-4`
  - The learning rate. For LoRA, higher values (e.g., `1e-4` to `1e-3`) are often used.
- `--unet_lr=1e-4`
  - A separate learning rate for the U-Net part of the LoRA.
- `--text_encoder_lr=1e-5`
  - A separate learning rate for the Text Encoder part of the LoRA. A smaller value than the U-Net is recommended.
- `--optimizer_type="AdamW8bit"`
  - The optimizer to use. Options include `AdamW8bit`, `AdamW`, `Lion`, `DAdaptation`, and `Adafactor`.
- `--lr_scheduler="constant"`
  - The learning rate scheduler. Options include `constant`, `cosine`, `linear`, `constant_with_warmup`, and `cosine_with_restarts`.
- `--max_train_epochs=12`
  - The number of training epochs.
- `--sdpa`
  - Uses Scaled Dot-Product Attention to reduce memory usage and improve training speed.
- `--mixed_precision="fp16"`
  - The mixed precision training setting. Options are `no`, `fp16`, and `bf16`.
- `--gradient_checkpointing`
  - Enables Gradient Checkpointing to significantly reduce memory usage at the cost of a slight decrease in training speed.
- `--clip_skip=1`
  - The number of layers to skip from the end of the Text Encoder. `1` means no skip.

## Using the Trained Model

Once training is complete, the LoRA model file will be saved in the directory specified by `output_dir`. This file can be used with GUI tools that support LoRA.
