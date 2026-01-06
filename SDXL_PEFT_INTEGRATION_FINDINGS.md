# SDXL PEFT Integration Findings & Revised Plan

**Date:** 2024-05-22
**Status:** Ready for Implementation

## Overview

This document outlines the confirmed path for integrating the new data pipeline into `scripts/sdxl_peft.py`. It incorporates the original integration plan and refinements derived from architectural review and requirement clarification.

## Key Architecture Decisions

### 1. Metadata Generation (`library/training/training_metadata.py`)
**Decision:** Refactor `create_training_metadata` to support `DatasetManifest`.
- The current function relies heavily on the legacy `DatasetGroup` interface.
- **Action:** Update `create_training_metadata` to accept `Union[DatasetGroup, DatasetManifest]`.
- **Implementation:** Implement a branch within the function to extract statistics (image counts, bucket info, repeats) directly from the `DatasetManifest` structure when passed, while maintaining backward compatibility for other scripts.

### 2. Text Encoder Caching
**Decision:** Fully migrate to `SdxlTextEncoderPipelineStrategy`.
- The legacy method `cache_text_encoder_outputs_if_needed` on the PEFT strategy calls methods on the legacy dataset object, which do not exist in the new pipeline.
- **Action:** Instantiate `SdxlTextEncoderPipelineStrategy` and use the `CachingEngine` to generate text encoder caches.
- **Implementation:**
  ```python
  if cfg.data.caching.cache_text_encoder_outputs:
      te_strategy = SdxlTextEncoderPipelineStrategy(
          max_token_length=cfg.training.max_token_length,
          dtype="fp16", # or derived from config
      )
      te_caching_engine = CachingEngine(strategy=te_strategy, ...)
      te_caching_engine.cache_dataset(
          manifest=train_manifest,
          model=(text_encoders[0], text_encoders[1], tokenizers[0], tokenizers[1]),
          accelerator=accelerator,
          cache_dir=cfg.data.caching.cache_dir,
      )
  ```

### 3. Validation Data Configuration
**Decision:** Add explicit `val_data_dir` support and dual-mode validation.
- The `DataConfig` currently lacks an explicit validation directory field.
- **Action:**
  1.  Update `SourceConfig` in `library/config/dataclasses/data.py` to include `val_data_dir: str | None`.
  2.  Update `create_manifest_from_config` to accept `validation_split` and `validation_seed`.
  3.  Update `create_manifest_from_config` logic:
      -   **Priority 1:** If `val_data_dir` is present, scan it and mark entries as `split="val"`.
      -   **Priority 2:** If `validation_split > 0`, perform a split on the training data.

## Revised Implementation Steps

### Phase A: Core Library Updates

1.  **Config Schema (`library/config/dataclasses/data.py`):**
    -   Add `val_data_dir` to `SourceConfig`.

2.  **Dataset Scanner (`library/data/pipeline/dataset_scanner.py`):**
    -   Update `create_manifest_from_config` signature to accept `validation_split` and `validation_seed`.
    -   Implement logic to scan `val_data_dir` if provided.
    -   Pass split parameters to `scan_directory` for training data if `validation_split` is used.

3.  **Metadata Utility (`library/training/training_metadata.py`):**
    -   Refactor to handle `DatasetManifest`.
    -   Map manifest entries/buckets to required metadata fields (`ss_num_train_images`, `ss_bucket_info`, etc.).

### Phase B: Script Integration (`scripts/sdxl_peft.py`)

1.  **Imports & Setup:**
    -   Remove `prepare_datasets`.
    -   Import pipeline components: `DatasetManifest`, `CachingEngine`, `SdxlLatentsPipelineStrategy`, `SdxlTextEncoderPipelineStrategy`, `create_training_dataloader`, `prepare_epoch`.

2.  **Manifest Creation:**
    -   Call `create_manifest_from_config` passing `cfg.data` and validation settings from `cfg.validation`.
    -   Separate manifest into `train_manifest` and `val_manifest` (or handle single manifest with splits).

3.  **Caching (Latents & Text Encoders):**
    -   **Latents:** Initialize `SdxlLatentsPipelineStrategy` and run `CachingEngine`.
    -   **Text Encoders:** If enabled, initialize `SdxlTextEncoderPipelineStrategy` and run `CachingEngine`.

4.  **Data Loaders:**
    -   **Validation:** Create persistent `val_dataloader` using `prepare_validation_epoch`.
    -   **Training:** Initialize `train_dataloader` as `None` (created per-epoch).

5.  **Training Loop:**
    -   **Start of Epoch:**
        -   Generate `EpochManifest` using `prepare_epoch` (handles shuffling, warmup).
        -   Create `TrainingDataset` and `DataLoader` for the current epoch.
    -   **Metadata:** Pass `DatasetManifest` to `create_training_metadata`.
    -   **Cleanup:** Remove legacy `dataset_group` references and manual `accelerator.skip_first_batches` (handled by manifest if needed, though simple resume usually just steps forward).

### Phase C: Verification

1.  **Smoke Test:** Run with a small dataset to verify manifest creation, caching, and batch generation.
2.  **Metadata Check:** Verify output `metadata` in the model file contains correct image counts and bucket info.
3.  **Validation Check:** Ensure validation loop runs correctly with `val_data_dir` or split.
