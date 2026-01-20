# Audit Report: Strategy Pattern Boundaries

**Topic:** Strategy Pattern Boundaries in PeftTrainer Refactoring
**Date:** 2026-01-20
**Scope:** `PeftTrainer`, `TrainingStrategy`, `SdxlTrainingStrategy`, `run_training_loop`

## 1. Executive Summary

The refactoring successfully separates orchestration (Trainer) from model specifics (Strategy). However, boundaries are blurred in `calculate_val_loss`, where the Strategy owns the validation *loop*, which is an orchestration concern. Additionally, `calculate_val_loss` is effectively used by the Trainer but is missing from the `TrainingStrategy` abstract base class (ABC), creating a type safety gap.

## 2. Question: What is the clear separation between `PeftTrainer` and `TrainingStrategy`?

**Current State Analysis:**

*   **Trainer (`PeftTrainer`):** Owns the training loop lifecycle, accelerator management, state synchronization, logging, and checkpointing. It decides *when* things happen.
*   **Strategy (`TrainingStrategy`):** Owns model loading, tokenization, encoding, caching, and the mathematical implementation of a training step (`process_batch`). It decides *how* things happen for a specific architecture.

This high-level separation is generally sound, with specific exceptions noted below.

## 3. Detailed Boundary Analysis

### A. `strategies.sample_images()`

**Question:** Ownership? Called from trainer.

**Findings:**
*   **Current Owner:** Strategy (`SdxlTrainingStrategy` implements it).
*   **Verdict:** **Correct**.
*   **Rationale:** While the Trainer decides *when* to sample (e.g., every epoch), the *how* is deeply model-dependent. `SdxlTrainingStrategy` needs to instantiate a specific pipeline class (e.g., `SdxlStableDiffusionLongPromptWeightingPipeline`). If this were in the Trainer, the Trainer would need to import every possible model pipeline, breaking the generic design. The Strategy acts as the factory for the generation pipeline.

### B. `strategies.calculate_val_loss()`

**Question:** Ownership? Called from trainer.

**Findings:**
*   **Current Owner:** Strategy (`SdxlTrainingStrategy` implements it).
*   **Verdict:** **Mixed / Incorrect Boundary**.
*   **Analysis:**
    *   The `calculate_val_loss` method in `SdxlTrainingStrategy` currently owns the **entire validation loop**: it iterates over the dataloader, manages the tqdm progress bar, handles RNG state switching, and updates the `val_loss_recorder`.
    *   **Violation:** Iterating over a dataloader is an orchestration task (Trainer). The Strategy should only define how to compute loss for a *single* validation batch.
*   **Technical Debt:** The `calculate_val_loss` method is **missing from the `TrainingStrategy` ABC** in `library/strategies/base/training.py`. `PeftTrainer` calls it dynamically, relying on the concrete implementation, which bypasses type safety.

**Recommendation:**
1.  **Refactor:** Move the validation loop, RNG switching, and recorder updates into `PeftTrainer` (or a `ValidationPhase` module).
2.  **Interface:** Add `process_val_batch` to the `TrainingStrategy` ABC.
3.  **Result:** Trainer loops over validation data -> calls `strategies.process_val_batch(batch)` -> Strategy calculates loss -> Trainer aggregates.

### C. `strategies.get_noise_scheduler()`

**Question:** Called from trainer - should this be in setup?

**Findings:**
*   **Current Location:** Called in `PeftTrainer._log_training_info()`, effectively part of the setup sequence.
*   **Ownership:** **Strategy** (Correct).
    *   **Rationale:** Future models (e.g., Flux, SD3) use Flow Matching instead of DDPM. The Strategy must define what "scheduler" or "flow" means for that model. Keeping it in the Strategy ensures the Trainer doesn't hardcode DDPM logic.
*   **Timing:** The method `_log_training_info` is misnamed. It currently acts as "Final Initialization" (initializing noise schedulers, plotters, and trackers).
    *   **Verdict:** Functionally fine, but semantically confusing. Ideally, `get_noise_scheduler` should be called in `setup()` or a dedicated `prepare_scheduler()` phase, and `_log_training_info` should only log.

## 4. Other Findings

*   **Missing Interface Definitions:** The `TrainingStrategy` ABC is missing explicit definitions for methods the Trainer relies on, including `process_batch` and `calculate_val_loss`. This makes the contract between Trainer and Strategy implicit rather than explicit.
*   **Orchestration Leak:** `SdxlTrainingStrategy.calculate_val_loss` handles `tqdm` (UI concern) and RNG state management (Training consistency concern), which are clearly Trainer responsibilities.

## 5. Action Plan Recommendations

1.  **Immediate:** Add `process_batch` and `calculate_val_loss` (or `process_val_batch`) to `library/strategies/base/training.py` to fix the interface gap.
2.  **Refactor:** Extract the validation loop from `SdxlTrainingStrategy.calculate_val_loss` into `PeftTrainer` (or `phases/validation.py`). Reduce the strategy method to `process_val_batch`.
3.  **Cleanup:** Rename `_log_training_info` to `_finalize_setup` or move the initialization logic (scheduler, trackers) to `setup()`/`prepare_scheduler()` to make the lifecycle clearer.
