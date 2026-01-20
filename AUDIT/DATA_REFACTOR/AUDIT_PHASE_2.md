# Audit: Batch Format Compatibility

**Status:** Completed
**Date:** 2026-01-05
**Auditor:** Jules

## Executive Summary

The audit of the `library/strategies/` directory and related components reveals that while SDXL integration (`peft_strategy_sdxl.py`) is fully compatible with the new data pipeline's batch format, legacy SD components (`peft_strategy_sd.py`) remain incompatible. Additionally, a potential risk exists with `conditional_loss` silently falling back to unmasked loss when expected masks are missing.

## Findings

### 1. SD Strategy Incompatibility (`peft_strategy_sd.py`)

**Severity:** ❌ Incompatible
**Location:** `library/strategies/peft_strategy_sd.py`

The SD PEFT strategy has not been updated to consume the new batch format produced by `TrainingDataset`.

-   **Input IDs:** The code expects `batch["input_ids_list"]` (legacy list of tensors), whereas the new pipeline provides `batch["input_ids"]` (a dictionary, e.g., `{'clip_l': ...}`).
    ```python
    # Current code in peft_strategy_sd.py
    input_ids = [ids.to(accelerator.device) for ids in batch["input_ids_list"]]
    ```
-   **Text Encoder Outputs:** The code expects `batch["text_encoder_outputs_list"]`, whereas the new pipeline provides `batch["text_encoder_outputs"]` (a dictionary).
    ```python
    # Current code in peft_strategy_sd.py
    text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
    ```

**Recommendation:** Update `peft_strategy_sd.py` to handle the new dictionary-based `input_ids` and `text_encoder_outputs` keys, similar to the implementation in `peft_strategy_sdxl.py`.

### 2. Silent Fallback in `conditional_loss` / `apply_masked_loss`

**Severity:** ⚠️ Potential Issue
**Location:** `library/losses/loss_weighting.py`

The `apply_masked_loss` function is designed to return the original loss unmodified if masks are not found in the batch.

```python
# Current implementation
if "conditioning_images" in batch:
    # ... logic ...
elif "alpha_masks" in batch and batch["alpha_masks"] is not None:
    # ... logic ...
else:
    return loss  # Silent return
```

While this prevents runtime crashes ("safe"), it creates a risk where a user intends to use masked loss (setting `cfg.loss.masked=True`) but, due to a dataloader or configuration error (e.g., missing alpha channels), the training proceeds without masking and without alerting the user.

**Recommendation:** Implement a warning check in the calling strategy or within `apply_masked_loss` itself. If `cfg.loss.masked` is asserted but no masks are found in the batch, a warning should be logged.

### 3. Legacy Key Access (`original_sizes_hw`, `crop_top_lefts`)

**Severity:** ⚠️ Mixed Usage
**Location:** Various files

-   **SDXL Strategy:** `peft_strategy_sdxl.py` correctly uses the new wrapper `_extract_conditioning_tensors` to access these values from `batch["conditionings"]`. It does not access flat keys directly.
-   **Legacy Datasets:** `library/data/_deprecated/dataset.py` and `controlnet_dataset.py` still populate and access the flat keys `original_sizes_hw` and `crop_top_lefts`.
-   **Impact:** As long as the new `TrainingDataset` is used with the updated `peft_strategy_sdxl.py`, there is no issue. However, if any legacy components (like old callbacks or loggers) expect these flat keys to exist in the batch, they will fail because `TrainingDataset` does not populate them at the root level of the batch dictionary.

**Recommendation:** Ensure all active callbacks or auxiliary scripts are updated to look for `batch["conditionings"]` for SDXL metadata, or ensure the `TrainingDataset` provides backward compatibility if necessary (though strictly migrating to the new format is preferred).

## Conclusion

The SDXL path is cleared for integration testing with the new data pipeline. The SD path requires significant refactoring of `peft_strategy_sd.py` before it can support the new batch format. The silent failure mode for masked loss should be addressed to improve user experience and debuggability.
