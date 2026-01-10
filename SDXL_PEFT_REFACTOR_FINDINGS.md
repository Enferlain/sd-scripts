# SDXL PEFT Refactoring Findings

## Introduction

This document details findings from an investigation into `scripts/sdxl_peft.py`. The goal is to identify opportunities for organization, modernization, and reducing redundancies by moving code into appropriate library modules.

## Organization Opportunities

### 1. Dataset Manifest Setup
The logic for creating dataset manifests is currently embedded in the training script. This includes:
- Checking `val_data_dir` vs `validation_split`.
- Calling `create_manifest_from_config` or `get_or_create_manifest`.
- Manually filtering train entries when a validation split is used.

**Recommendation:**
Create a dedicated function (e.g., `prepare_dataset_manifests` in `library/data/setup.py`) that encapsulates this logic and returns the `train_manifest` and `val_manifest`.

### 2. Latent & Text Encoder Caching
The script currently orchestrates the caching process for both latents and text encoder outputs.
- **Latents:** The caching loop using `CachingEngine` is explicit in the script.
- **Text Encoders (Disk):** Similar explicit usage of `CachingEngine`.
- **Text Encoders (Memory):** A verbose block (Lines 264-306) handles in-memory TE caching, exposing low-level details like tokenization, `get_hidden_states_sdxl`, and entry manipulation.

**Recommendation:**
- Encapsulate the high-level caching orchestration in `library/data/caching_utils.py` (e.g., `prepare_latents_cache`, `prepare_text_encoder_cache`).
- Move the in-memory TE caching logic into `SdxlPeftStrategy` or a utility function to hide implementation details.

### 3. PEFT Adapter Initialization
Lines 325-376 handle importing the adapter module, loading base weights (merging them if necessary), resolving kwargs, and creating the adapter instance. This low-level setup logic clutters the training script.

**Recommendation:**
Refactor this into `library/adapters/setup.py` (or `factory.py`), creating a function like `setup_peft_adapter(cfg, vae, text_encoder, unet)` that handles the details and returns the prepared adapter.

### 4. Checkpointing Helpers
The `train` function contains nested definitions for `save_model` and `remove_model` (Lines 528-566). These functions capture local variables (`cfg`, `metadata`, etc.) but are essentially generic saving logic specific to PEFT (saving only adapter weights).

**Recommendation:**
Move these to `library/training/checkpointing.py` or `sdxl_checkpointing.py`. They can be implemented as standalone functions accepting the necessary config and state objects, or returned by a closure factory if strict context capture is needed (though explicit arguments are cleaner).

## Redundancies

### 1. Saving Logic (Step vs. Epoch)
The script duplicates the logic for saving models and states:
- **Step-based:** Lines 777-811 handles saving every N steps.
- **Epoch-based:** Lines 890-918 handles saving every N epochs.

Both blocks perform:
- Checking conditions.
- Constructing filenames.
- Calling `save_model`.
- Saving EDM2 weights (if enabled).
- Saving state (if enabled).
- Removing old models/states.

**Recommendation:**
Unify this logic into a single `manage_checkpointing` function in `library/training/checkpointing.py` that handles both step and epoch triggers, or at least share the core "save and cleanup" sequence.

### 2. Validation and Sampling Calls
`strategies.calculate_val_loss` and `strategies.sample_images` are called in two distinct places:
1.  **Before Training:** For initial checks (`sample_at_first`).
2.  **During Training:** Inside the loop.

The wrapping logic (setting the adapter to eval/train mode, handling optimizers, updating recorders) is repeated.

**Recommendation:**
Create a wrapper function `validate_and_sample(...)` that handles the mode switching and calls the strategy methods.

### 3. Text Encoder Device Management
There are multiple blocks handling the movement of text encoders:
- Moving to GPU for caching.
- Moving back to CPU.
- Offloading to CPU if not caching.
- Deleting if not needed.

This logic is scattered and could be centralized in a `manage_text_encoder_device` helper or within the strategy.

## Summary of Plan

1.  **Refactor Adapter Setup**: Extract adapter creation/merging logic to `library/adapters/setup.py`.
2.  **Move Save/Remove Logic**: Move nested `save_model`/`remove_model` to `library/training/checkpointing.py`.
3.  **Encapsulate Dataset Setup**: Create `prepare_dataset_manifests` in `library/data/setup.py`.
4.  **Consolidate Validation/Sampling**: Create a wrapper for the validation/sampling sequence.
5.  **Refactor Caching Logic**: Move verbose caching blocks (especially in-memory TE caching) to library functions.
