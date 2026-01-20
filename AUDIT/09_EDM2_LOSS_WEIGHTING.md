# Audit Report: EDM2 Loss Weighting

## Status: PASS

## Findings

### 1. Model, Optimizer, and Scheduler Creation/Usage
**Question:** Are `edm2_model`, `edm2_optimizer`, `edm2_lr_scheduler` created and used?

- **Creation:** Verified. `edm2_model`, `edm2_optimizer`, and `edm2_lr_scheduler` are correctly initialized in `PeftTrainer._log_training_info` using `library.losses.edm2_loss_utils.prepare_edm2_loss_weighting`.
- **Usage:** Verified. In `library/training/phases/training_loop.py`, the EDM2 model is passed to `strategies.process_batch` for loss scaling, and the optimizer/scheduler are stepped correctly when `cfg.loss.edm2.edm2_loss_weighting` is enabled. `determine_grad_sync_context` also correctly includes the `edm2_model` for distributed training synchronization.
- **Reference Comparison:** The implementation matches the logic in the pre-refactor `scripts/sdxl_peft copy.py`.

### 2. Checkpoint Saving
**Question:** Is checkpoint saving for EDM2 weights implemented?

- **Implementation:** Verified.
  - **Step-wise:** Implemented in `run_training_loop` via `trainer.save_checkpoint` with the `_edm2_loss_weights` suffix.
  - **Epoch-end:** Implemented in `run_training_loop` via `trainer.save_checkpoint` with the `_edm2_loss_weights` suffix.
  - **Final:** Implemented in `PeftTrainer._finalize_training` via `trainer.save_checkpoint`.
  - **Cleanup:** Old EDM2 checkpoints are correctly removed using `trainer.remove_checkpoint`.
- **Reference Comparison:** The implementation matches the logic in the pre-refactor `scripts/sdxl_peft copy.py`.

### 3. Dynamic Timestep Schedule Integration
**Question:** Is dynamic timestep schedule integration functional?

- **Dynamic Range (Min/Max):** Verified. `PeftTrainer` parses the `dynamic_timestep_schedule` config. The training loop (`run_training_loop`) correctly checks the schedule at the start of each step and updates `trainer._current_min_timestep` and `trainer._current_max_timestep`. These values are passed to `strategies.process_batch` and subsequently to `get_noise_noisy_latents_and_timesteps`, ensuring the timestep sampling range is dynamically adjusted.
- **EDM2 Laplace Weights:**
  - `library/training/diffusion.py` contains logic to sample timesteps using `noise_scheduler.edm2_laplace_weights` if the attribute exists.
  - **Investigation:** A comprehensive search confirms that `edm2_laplace_weights` is **not** set on the noise scheduler anywhere in the current codebase, nor was it present in the reference `scripts/sdxl_peft copy.py`.
  - **Conclusion:** The specific "Laplace weights" feature appears to be dormant code (possibly from a future or separate feature branch), but this is **not a regression** caused by the refactor. The "Dynamic Timestep Schedule" (shifting min/max) is fully functional.

## Summary
The EDM2 Loss Weighting functionality has been successfully ported to the new modular architecture. The logic for model creation, training, and checkpointing is preserved and functions as expected.
