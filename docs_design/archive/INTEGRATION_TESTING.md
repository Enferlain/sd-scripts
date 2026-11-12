# Integration Testing Checklist

Manual integration testing for the new trainer (`Trainer`) and data pipeline.

## Status Legend

- `[ ]` Not tested
- `[x]` Tested and working
- `[!]` Issue found (see notes)

---

## Core Training

| Status | Feature                    | How to Verify                                         |
| ------ | -------------------------- | ----------------------------------------------------- |
| [x]    | Basic training runs        | Loss values appear in progress bar                    |
| [ ]    | Loss decreases over time   | Run 50+ steps, verify `avr_loss` trending down        |
| [ ]    | Loss matches legacy script | Compare loss curves with `sd_peft.py` (legacy)        |
| [x]    | Output file created        | Check `output_dir/` for `.safetensors` after training |

---

## Checkpointing

| Status | Feature                 | Config Key               | How to Verify                           |
| ------ | ----------------------- | ------------------------ | --------------------------------------- |
| [x]    | Save every N steps      | `save_every_n_steps: 10` | Files created at step 10, 20, 30...     |
| [ ]    | Save every N epochs     | `save_every_n_epochs: 1` | Files created at epoch end              |
| [x]    | Save state              | `save_state: true`       | `output_dir/` contains accelerate state |
| [ ]    | Keep only N checkpoints | `save_n_epoch_ratio: 2`  | Old checkpoints deleted                 |
| [ ]    | No metadata             | `no_metadata: true`      | Checkpoint has minimal metadata         |

---

## Resume Training

| Status | Feature                      | How to Verify                                    |
| ------ | ---------------------------- | ------------------------------------------------ |
| [ ]    | Resume from final checkpoint | Run, stop, run again - `global_step` continues   |
| [x]    | Resume optimizer state       | Loss curve is continuous (no spike after resume) |
| [x]    | Resume LR scheduler          | Learning rate continues from where it left off   |
| [x]    | Resume epoch correctly       | Epoch counter continues correctly                |

---

## Sampling

| Status | Feature                | Config Key                    | How to Verify                 |
| ------ | ---------------------- | ----------------------------- | ----------------------------- |
| [ ]    | Sample at first        | `sample_at_first: true`       | Image generated before step 1 |
| [x]    | Sample every N steps   | `sample_every_n_steps: 10`    | Images at step 10, 20, 30...  |
| [ ]    | Sample every N epochs  | `sample_every_n_epochs: 1`    | Images at epoch end           |
| [x]    | Sample prompts file    | `sample_prompts: prompts.txt` | Uses prompts from file        |
| [x]    | Sample output location | Check `output_dir/sample/`    | Images saved correctly        |

---

## Validation

| Status | Feature                | Config Key                   | How to Verify                       |
| ------ | ---------------------- | ---------------------------- | ----------------------------------- |
| [ ]    | Validation split       | `validation_split: 0.1`      | "Validating..." appears in logs     |
| [ ]    | Validate every N steps | `validate_every_n_steps: 10` | Val loss reported at step 10, 20... |
| [ ]    | Separate val directory | `val_data_dir: /path`        | Uses separate images for validation |
| [ ]    | Val loss logged        | -                            | `val_loss` appears in tracker logs  |
| [ ]    | Val loss stable        | -                            | Val loss comparable to train loss   |

---

## Text Encoder Training

| Status | Feature                   | Config Key                     | How to Verify                   |
| ------ | ------------------------- | ------------------------------ | ------------------------------- |
| [ ]    | Train text encoder        | `text_encoders: 1e-5`          | TE LR appears in log output     |
| [ ]    | Freeze text encoder       | `text_encoders: 0`             | TE not in optimizer params      |
| [ ]    | Per-TE learning rates     | `text_encoders: [1e-5, 1e-6]`  | SDXL: different LRs for each TE |
| [ ]    | TE gradient checkpointing | `gradient_checkpointing: true` | Lower VRAM with TE training     |

---

## Memory Optimization

| Status | Feature                | Config Key                         | How to Verify                        |
| ------ | ---------------------- | ---------------------------------- | ------------------------------------ |
| [ ]    | Gradient checkpointing | `gradient_checkpointing: true`     | Lower VRAM usage                     |
| [ ]    | Offload text encoders  | `offload_text_encoders: true`      | TEs move to CPU during training      |
| [x]    | Cache latents          | `cache_latents: true`              | Latent cache created before training |
| [x]    | Cache TE outputs       | `cache_text_encoder_outputs: true` | TE cache created                     |
| [ ]    | No cache (live encode) | `cache_latents: false`             | Encodes images each step             |

---

## Advanced Features

| Status | Feature               | Config Key                       | How to Verify                |
| ------ | --------------------- | -------------------------------- | ---------------------------- |
| [ ]    | DeepSpeed             | `deepspeed: true`                | DeepSpeed logs appear        |
| [ ]    | FP8 training          | `fp8_base: true`                 | Lower VRAM, no NaN loss      |
| [ ]    | Multi-GPU             | `accelerate launch`              | Uses multiple GPUs           |
| [ ]    | Gradient accumulation | `gradient_accumulation_steps: 2` | Steps take longer, same VRAM |
| [ ]    | Mixed precision       | `mixed_precision: fp16`          | Default, verify no issues    |
| [ ]    | BF16 precision        | `mixed_precision: bf16`          | Works on Ampere+ GPUs        |

---

## Logging & Tracking

| Status | Feature                    | Config Key                                    | How to Verify           |
| ------ | -------------------------- | --------------------------------------------- | ----------------------- |
| [ ]    | WandB logging              | `log_with: wandb`                             | Metrics appear in WandB |
| [ ]    | TensorBoard                | `log_with: tensorboard`                       | Logs in `logging_dir/`  |
| [ ]    | Log tracker name           | `log_tracker_name: custom`                    | Custom project name     |
| [ ]    | Timestep distribution plot | `log_timestep_distribution_every_n_steps: 50` | Plot saved              |

---

## Notes

Record any issues or observations here:

```
Date:
Issue:
Resolution:
```
