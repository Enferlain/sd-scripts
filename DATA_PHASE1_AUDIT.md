# Data Pipeline Audit: Phase 1 (Scanning & Manifest)

This document details the findings of an audit comparing the new data pipeline (`library/data/pipeline/`) against the legacy system (`library/data/_deprecated/`).

**Audit Date:** 2026-01-04
**Scope:** Phase 1 (Dataset Scanning & Manifest)

---

## 1. Image Format Support

**Question:** Are we handling all image formats the legacy system supports?

**Finding:** **NO**

The new system is missing dynamic support for `.avif` and relies on a generic library for `.jxl` which may lack the specialized robustness of the legacy implementation.

### Details:
*   **Legacy System (`library/constants.py`):**
    *   Base support: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`
    *   **Dynamic AVIF:** Checks for `pillow_avif` and adds `.avif` if available.
    *   **Specialized JXL:** Checks for `jxlpy` or `pillow_jxl` and uses a dedicated `library.utils.jpeg_xl_util.get_jxl_size` function to parse JXL headers/streams directly.
*   **New System (`dataset_scanner.py`):**
    *   Hardcoded list: `.png`, `.jpg`, `.jpeg`, `.webp`, `.jxl`, `.bmp`, `.gif`, `.tiff`, `.tif`
    *   **Missing AVIF:** No support for `.avif` files.
    *   **Generic JXL:** Relies on `imagesize` library or PIL fallback. It does not utilize the robust `jpeg_xl_util` from the legacy codebase.
    *   **Additions:** Adds `.gif`, `.tiff`, `.tif` which were not explicitly in the legacy constant list.

**Recommendation:**
*   Port the dynamic `pillow_avif` check to `dataset_scanner.py`.
*   Import and use `library.utils.jpeg_xl_util.get_jxl_size` for JXL files to ensure consistent behavior with legacy.

---

## 2. Missing Caption Handling

**Question:** Do we correctly handle images with no caption file?

**Finding:** **PARTIALLY (Behavior Change)**

The new system is significantly stricter by default than the legacy system.

### Details:
*   **Legacy System:**
    *   If a caption file is missing, it attempts to use `class_tokens`.
    *   If both are missing, it logs a **warning** and proceeds with an empty caption.
    *   It does not stop execution.
*   **New System (`dataset_scanner.py`):**
    *   `scan_directory` defaults to `require_caption=True`.
    *   If a caption is missing (and image is not regularization), it **raises a `ValueError`** and halts execution.
    *   Users must explicitly set `require_caption=False` to replicate legacy "warning" behavior.

**Recommendation:**
*   Confirm if strict enforcement is the intended design goal for v2.
*   If backward compatibility is priority, consider changing the default to `False` or ensuring the calling script exposes this configuration option clearly.

---

## 3. Bucket Resolution Algorithm

**Question:** Is the bucket resolution algorithm identical to legacy?

**Finding:** **YES**

The bucket resolution and selection logic has been faithfully ported.

### Details:
*   **Resolution Generation:** `make_bucket_resolutions` in `dataset_scanner.py` uses identical logic (square bucket + aspect ratio buckets) to the legacy `BucketManager`.
*   **Bucket Selection:** `select_bucket` in `dataset_scanner.py` implements the same Aspect Ratio matching logic and "no upscale" fallback logic (calculating resized dimensions and rounding to steps) as the legacy implementation.
*   **Comparison:** A logic verification confirms that for a given input resolution and target config, both systems will select the same bucket and resize dimensions.

---

## 4. Metadata Preservation

**Question:** Are we preserving all metadata needed for training (loss weights, alpha masks, etc.)?

**Finding:** **PARTIALLY (Alpha Mask Gap)**

While `CacheEntry` supports `is_reg` (loss weights) and `has_alpha_mask`, the **scanner currently fails to populate the alpha mask status**.

### Details:
*   **Alpha Masks (Gap Identified):**
    *   **Legacy:** Controlled via `alpha_mask` boolean in the subset configuration (passed to `DreamBoothDataset`). If True, images are loaded as RGBA.
    *   **New:** `scan_directory` in `dataset_scanner.py` **does not accept an `alpha_mask` argument**, nor does it check if images have an alpha channel (RGBA) during scanning.
    *   **Result:** The generated manifest will default to `has_alpha_mask=False` for all images. Training scripts relying on the manifest will fail to load alpha masks, breaking functionality for transparent images.
*   **Loss Weights:** Preserved. `is_reg` flag is correctly propagated.
*   **Crop Coordinates:** Handled via `.safetensors` metadata header (architectural change).
*   **Flipped Status:** Preserved in `CacheEntry`.

**Recommendation:**
*   Update `scan_directory` to accept an `alpha_mask` argument (to force enable for a folder).
*   Alternatively, update `scan_directory` to automatically detect alpha channels (e.g., `img.mode == 'RGBA'`) during the size check, although this requires full file opening which might be slower.

---

**Conclusion:** The new data structures support the metadata, but the **scanner implementation** is incomplete regarding Alpha Masks.