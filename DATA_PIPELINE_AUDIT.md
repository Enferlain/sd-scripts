# Data Pipeline Audit Report

**Date:** 2026-01-04
**Scope:** Bucketing logic, edge case handling, and batch formation isolation.

## Executive Summary

The new data pipeline implementation in `library/data/pipeline/` has been audited against the legacy `library/data/_deprecated/` codebase. The audit confirms that the core bucketing logic and resolution calculations are functionally identical to the legacy system. Batch formation logic ensures strict isolation between buckets.

One known edge case (zero-dimension buckets for tiny images) has been reproduced in the new system, preserving legacy behavior.

## Detailed Findings

### 1. Bucket Resolution Calculation

**Status:** ✅ **Verified Parity**

The bucket resolution generation and image-to-bucket assignment logic in `dataset_scanner.py` was compared against the legacy `BucketManager`.

*   **Standard Mode (`no_upscale=False`)**:
    *   The new `select_bucket` implementation produces identical bucket resolutions and resized image dimensions as the legacy code.
    *   Aspect ratio matching and "nearest bucket" selection logic are preserved.
    *   Upscaling/downscaling calculations are identical.

*   **No Upscale Mode (`no_upscale=True`)**:
    *   The logic for creating custom buckets for images smaller than `max_area` is preserved.
    *   The handling of images larger than `max_area` (downscaling to fit) is identical.

### 2. Edge Case Handling

**Status:** ⚠️ **Legacy Behavior Preserved (including bugs)**

The audit specifically tested edge cases to ensure consistent behavior:

*   **Small Images (`< min_size`)**:
    *   **Behavior**: When `no_upscale=True`, images smaller than the minimum bucket size but larger than the step size are assigned to custom small buckets (e.g., `128x128` for a `100x100` image with step 64).
    *   **Assessment**: This matches legacy behavior and is considered correct for users who explicitly request no upscaling.

*   **Large Images (`> max_size`)**:
    *   **Behavior**: Images larger than the maximum resolution are downscaled to fit within `max_area` while maintaining aspect ratio.
    *   **Assessment**: Correctly handled in both standard and `no_upscale` modes.

*   **Tiny Images (Zero-Dimension Bug)**:
    *   **Finding**: Images smaller than `bucket_reso_steps` (e.g., `30x30` with `step=64`) result in a bucket resolution of `(0, 0)` in both legacy and new implementations.
    *   **Cause**: `bucket_width = resized_size[0] - resized_size[0] % self.reso_steps`. If size < step, result is 0.
    *   **Recommendation**: This is a low-priority issue inherited from legacy. It should be addressed in a future update by clamping the minimum bucket size to `bucket_reso_steps`.

### 3. Batch Formation Isolation

**Status:** ✅ **Verified Correctness**

The batch formation logic in `library/data/pipeline/epoch_preparation.py` (`prepare_epoch`) was analyzed for cross-bucket leakage.

*   **Process**:
    1.  All samples are grouped into a dictionary keyed by bucket resolution (`bucket_keys`).
    2.  The code iterates strictly over these keys.
    3.  Batches are constructed *inside* this loop, using only samples from the current bucket list.
    4.  Completed `BatchInfo` objects (which contain the bucket resolution) are added to a global list.
    5.  Shuffling happens *after* batches are formed, rearranging the order of batches but never mixing their contents.

*   **Conclusion**: It is algorithmically impossible for a single batch to contain images from different buckets, as batches are finalized before any cross-bucket aggregation occurs.

## Audit Methodology

1.  **Code Review**: Manual inspection of `dataset_scanner.py` and `epoch_preparation.py`.
2.  **Automated Verification**: A script (`tools/audit_verification.py`) was created to run identical inputs through both legacy and new `select_bucket` functions.
3.  **Test Cases**:
    *   Standard resolutions (1024x1024, 512x512)
    *   Aspect ratio variations (Portrait, Landscape)
    *   Edge cases (Tiny, Small, Large, irregular aspect ratios)

## Conclusion

The new data pipeline implementation faithfully reproduces the bucketing strategy of the legacy system, including its edge case behaviors. The batch formation logic is robust and correctly enforces bucket isolation.
