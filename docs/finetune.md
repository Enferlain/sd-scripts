# Fine-tuning Guide

This document explains how to perform fine-tuning on various model architectures using the `*_train.py` scripts.

### Difference between Fine-tuning and LoRA tuning

This repository supports two methods for additional model training: **Fine-tuning** and **LoRA (Low-Rank Adaptation)**. Each method has distinct features and advantages.

**Fine-tuning** is a method that retrains all (or most) of the weights of a pre-trained model.
- **Pros**: It can improve the overall expressive power of the model and is suitable for learning styles or concepts that differ significantly from the original model.
- **Cons**:
    - It requires a large amount of VRAM and computational cost.
    - The saved file size is large (same as the original model).
    - It is prone to "overfitting," where the model loses the diversity of the original model if over-trained.
- **Corresponding scripts**: Scripts named `*_train.py`, such as `fine_tune.py` and `sdxl_train.py`.

**LoRA tuning** is a method that freezes the model's weights and only trains a small additional network called an "adapter."
- **Pros**:
    - It allows for fast training with low VRAM and computational cost.
    - It is considered resistant to overfitting because it trains fewer weights.
    - The saved file (LoRA network) is very small, ranging from tens to hundreds of MB, making it easy to manage.
    - Multiple LoRAs can be used in combination.
- **Cons**: Since it does not train the entire model, it may not achieve changes as significant as fine-tuning.
- **Corresponding scripts**: Scripts named `*_train_network.py`, such as `train_network.py` and `sdxl_train_network.py`.

| Feature | Fine-tuning | LoRA tuning |
|:---|:---|:---|
| **Training Target** | All model weights | Additional network (adapter) only |
| **VRAM/Compute Cost**| High | Low |
| **Training Time** | Long | Short |
| **File Size** | Large (several GB) | Small (few MB to hundreds of MB) |
| **Overfitting Risk** | High | Low |
| **Suitable Use Case** | Major style changes, concept learning | Adding specific characters or styles |

Generally, it is recommended to start with **LoRA tuning** if you want to add a specific character or style. **Fine-tuning** is a valid option for more fundamental style changes or aiming for a high-quality model.

---

### Fine-tuning for each architecture

Fine-tuning updates the entire weights of the model, so it has different options and considerations than LoRA tuning. This section describes the fine-tuning scripts for major architectures.

The basic command structure is common to all architectures.

```bash
accelerate launch --mixed_precision bf16 {script_name}.py \
  --pretrained_model_name_or_path <path_to_model> \
  --dataset_config <path_to_config.toml> \
  --output_dir <output_directory> \
  --output_name <model_output_name> \
  --save_model_as safetensors \
  --max_train_steps 10000 \
  --learning_rate 1e-5 \
  --optimizer_type AdamW8bit
```

#### SD/SDW (`fine_tune.py`)

Performs fine-tuning for SD1.x/2.x and SD WebUI models. It is possible to train both the U-Net and the Text Encoder.

**Key Options:**

- `--train_text_encoder`: Includes the weights of the Text Encoder in the training. Effective for significant style changes or strongly learning specific concepts.
- `--learning_rate_te`: Sets a specific learning rate for the Text Encoder. If not provided, it will use the same learning rate as the U-Net.

**Command Example:**

```bash
accelerate launch --mixed_precision bf16 fine_tune.py \
  --pretrained_model_name_or_path "sd-v1-5-pruned-emaonly.safetensors" \
  --dataset_config "dataset_config.toml" \
  --output_dir "output" \
  --output_name "sd15_finetuned" \
  --train_text_encoder \
  --learning_rate 1e-5 \
  --learning_rate_te 5e-6
```

#### SDXL (`sdxl_train.py`)

Performs fine-tuning for SDXL models. It is possible to train both the U-Net and the Text Encoders.

**Key Options:**

- `--train_text_encoder`: Includes the weights of the Text Encoders (CLIP ViT-L and OpenCLIP ViT-bigG) in the training. Effective for significant style changes or strongly learning specific concepts.
- `--learning_rate_te1`, `--learning_rate_te2`: Set individual learning rates for each Text Encoder.
- `--block_lr`: Divides the U-Net into 23 blocks and sets a different learning rate for each block. This allows for advanced adjustments, such as strengthening or weakening the learning of specific layers. (Not available in LoRA tuning).

**Command Example:**

```bash
accelerate launch --mixed_precision bf16 sdxl_train.py \
  --pretrained_model_name_or_path "sd_xl_base_1.0.safetensors" \
  --dataset_config "dataset_config.toml" \
  --output_dir "output" \
  --output_name "sdxl_finetuned" \
  --train_text_encoder \
  --learning_rate 1e-5 \
  --learning_rate_te1 5e-6 \
  --learning_rate_te2 2e-6
```

---

### Differences between Fine-tuning and LoRA tuning per architecture

| Architecture | Key Features/Options Specific to Fine-tuning | Main Differences from LoRA tuning |
|:---|:---|:---|
| **SD/SDW** | `--train_text_encoder` | Only fine-tuning can train the entire Text Encoder. LoRA only trains the adapter parts. |
| **SDXL** | `--block_lr` | Only fine-tuning allows for granular control over the learning rate for each U-Net block. |
