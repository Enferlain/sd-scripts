# Data Pipeline Audit Report

## Executive Summary

The audit of the new data pipeline (`library/data/pipeline/`) against the legacy implementation (`library/data/_deprecated/dataset.py`) has identified **two significant findings**:

1.  **Reproducibility Issue**: The shuffle algorithm does not automatically vary by epoch when a fixed seed is provided, leading to identical batches across epochs.
2.  **Missing Feature**: Textual Inversion string replacement support is missing from the new caption processor.

All other audited features (secondary separators, token warmup, wildcards, general dropout/shuffle) are correctly ported.

---

## Detailed Findings

### 1. Shuffle Reproducibility (Phase 3: Epoch Preparation)

**Status:** ⚠️ **Requires Fix**

*   **Legacy Behavior**: Automatically adds `epoch` to `seed` before shuffling.
    ```python
    random.seed(self.seed + self.current_epoch)
    ```
*   **Current Behavior**: Uses `seed` if provided, effectively ignoring `epoch` for initialization.
    ```python
    seed = seed if seed is not None else epoch
    rng = random.Random(seed)
    ```
*   **Impact**: If a user provides a fixed seed (common for reproducibility), every epoch will have the exact same batch composition and order. This reduces training effectiveness as the model sees the same sequence repeatedly.
*   **Recommendation**: Modify `prepare_epoch` to mix the seed:
    ```python
    seed = (seed if seed is not None else 0) + epoch
    ```

### 2. Textual Inversion Support (Caption Processing)

**Status:** ❌ **Missing Feature**

*   **Legacy Behavior**: Supports a `replacements` dictionary to swap strings in captions (used for Textual Inversion triggers).
    ```python
    for str_from, str_to in self.replacements.items():
        caption = caption.replace(str_from, str_to)
    ```
*   **Current Behavior**: `CaptionConfig` and `process_caption` have no mechanism for string replacements.
*   **Impact**: Textual Inversion training that relies on replacing placeholder strings with learnable tokens will fail or require external caption modification.

### 3. Caption Augmentation Features

**Status:** ✅ **Verified**

The following features have been verified as correctly ported:

*   **Secondary Separator**: `secondary_separator` is correctly replaced with `caption_separator`.
*   **Token Warmup**: `int()` based calculation is functionally equivalent to legacy `math.floor()`.
    *   Legacy: `math.floor(...)`
    *   New: `int(...)`
*   **Wildcards**: `{a|b}` parsing and multiline handling logic matches.
*   **Dropout & Shuffle**: Tag shuffling, caption dropout, and tag dropout logic structure is consistent.
*   **Prefix/Suffix**: Applied correctly before other processing.

### 4. Codebase Reference

*   **Legacy Reference**: `library/data/_deprecated/dataset.py`
*   **New Implementation**:
    *   `library/data/pipeline/epoch_preparation.py` (Shuffle logic)
    *   `library/data/pipeline/caption_processor.py` (Caption logic)

## Next Steps

1.  **Fix Shuffle**: Update `prepare_epoch` to `seed = (seed if seed is not None else 0) + epoch`.
2.  **Implement Replacements**: Add `replacements: dict[str, str]` to `CaptionConfig` and implement the replacement loop in `process_caption`.
