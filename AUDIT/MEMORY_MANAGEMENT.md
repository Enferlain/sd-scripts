# Audit Report: Memory Management

**Date:** 2026-01-20
**Scope:** `scripts/sdxl_peft.py` refactor (Trainer/Phases)
**Reference:** `scripts/sdxl_peft copy.py` (Legacy)

## Executive Summary

The refactor has improved the stability of the training loop by removing the aggressive deletion of Text Encoders, which previously caused crashes during validation and sampling. However, a regression was identified regarding the explicit `offload_text_encoders` configuration, which is currently ignored.

## Detailed Findings

### 1. Model Cleanup (Text Encoder Deletion)
**Status:** ✅ **Intentional Behavior Change**

*   **Legacy Behavior:** The script aggressively deleted Text Encoder instances if `is_text_encoder_not_needed_for_training(cfg)` returned true.
*   **Refactored Behavior:** The Text Encoders are retained but moved to CPU.
*   **Analysis:** The legacy behavior was a "footgun." Deleting the Text Encoders caused runtime crashes if the user enabled features that require them, specifically:
    *   Validation Loop (`calculate_val_loss`)
    *   Sample Image Generation (`sample_images`)
*   **Conclusion:** The new behavior is safer and prevents valid configurations from crashing. The slight increase in system RAM usage is an acceptable trade-off for stability.

### 2. Text Encoder Offloading
**Status:** ❌ **Regression**

*   **Issue:** The logic to handle `cfg.performance.memory.offload_text_encoders` is missing from the new pipeline.
*   **Legacy Code:**
    ```python
    elif cfg.performance.memory.offload_text_encoders:
        logger.info("Offloading text encoders to CPU (on-the-fly encoding enabled)")
        for t_enc in text_encoders:
            t_enc.to("cpu")
        clean_memory_on_device(accelerator.device)
    ```
*   **Current State:** This block is absent in `library/training/phases/caching.py` (specifically in `run_te_caching`).
*   **Impact:** Users relying on this setting for on-the-fly encoding with low VRAM may experience Out-Of-Memory (OOM) errors, as the Text Encoders might remain on the GPU or not be explicitly cleared.
*   **Recommendation:** Restore this logic in `library/training/phases/caching.py` immediately after the caching block.

### 3. Memory Cleaning (`clean_memory_on_device`)
**Status:** ⚠️ **Mostly Correct (Affected by Regression)**

*   **Analysis:** `clean_memory_on_device()` is correctly called:
    *   After Latent Caching (`phases/caching.py`)
    *   After TE Caching (`phases/caching.py`)
    *   Before the main training loop starts (`peft_trainer.py`)
*   **Missing Call:** The legacy script had a specific `clean_memory_on_device()` call inside the `offload_text_encoders` block. Since that block is missing (see item #2), this specific memory cleanup is also missing.

### 4. Unwrap Sequence
**Status:** ✅ **Correct**

*   **Concern:** Ensure adapter is unwrapped before `accelerator.end_training()`.
*   **Verification:**
    *   **Legacy:**
        ```python
        if is_main_process:
            adapter = accelerator.unwrap_model(adapter)
        accelerator.end_training()
        ```
    *   **Refactor (`peft_trainer.py`):**
        ```python
        def _finalize_training(self) -> None:
            # ...
            if self.is_main_process:
                self.adapter = self.accelerator.unwrap_model(self.adapter)
            self.accelerator.end_training()
            # ...
        ```
*   **Conclusion:** The sequence is preserved correctly.

## Summary of Actions Required

1.  **Fix Regression:** Implement the missing `offload_text_encoders` logic in `library/training/phases/caching.py`.
