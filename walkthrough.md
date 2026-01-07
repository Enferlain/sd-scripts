# Data Pipeline Integration Walkthrough

## Summary

Successfully integrated the new high-performance data pipeline into `scripts/sdxl_peft.py`, replacing legacy data loading mechanisms with the new manifest-based approach.

## Changes Made

### Pre-Integration Updates

| File                                                                                          | Change                                                                                                |
| --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| [data.py](file:///d:/Projects/sd-scripts/library/config/dataclasses/data.py)                  | Added `val_data_dir` to `SourceConfig`, `cache_dir` to `CachingConfig`                                |
| [dataset_scanner.py](file:///d:/Projects/sd-scripts/library/data/pipeline/dataset_scanner.py) | Updated `create_manifest_from_config` with validation support; added `compute_tag_frequency()` helper |
| [training_metadata.py](file:///d:/Projects/sd-scripts/library/training/training_metadata.py)  | Refactored to accept `DatasetManifest` instead of `DatasetGroup`                                      |

### sdxl_peft.py Integration

render_diffs(file:///d:/Projects/sd-scripts/scripts/sdxl_peft.py)

#### Phase A: Imports

- Replaced `prepare_datasets` import with new pipeline imports (`CachingEngine`, `create_manifest_from_config`, `prepare_epoch`, etc.)
- Added SDXL caching strategy imports

#### Phase B: Manifest Creation

- Replaced `prepare_datasets()` call with `create_manifest_from_config()`
- Added validation manifest creation (supports both `val_data_dir` and `validation_split`)

#### Phase C: Latent Caching

- Replaced `train_dataset_group.new_cache_latents()` with `CachingEngine` + `SdxlLatentsPipelineStrategy`

#### Phase D: Text Encoder Caching

- Replaced legacy TE caching with `CachingEngine` + `SdxlTextEncoderPipelineStrategy`

#### Phase E: DataLoader Creation

- Removed upfront `train_dataloader` creation
- Created `val_dataloader` once using `prepare_validation_epoch()` + `create_training_dataloader()`

#### Phase F: Statistics & Metadata

- Updated stats printing to use manifest properties (`num_train_images`, `num_reg_images`)
- Updated `create_training_metadata()` call to use new signature with manifests

#### Phase G: Epoch Loop

- Added per-epoch DataLoader creation inside training loop
- Uses `prepare_epoch()` for fresh shuffling each epoch
- Uses `CaptionConfig` for caption augmentation per-epoch

#### Phase H: Cleanup

- Removed `accelerator.prepare()` calls for dataloaders (new pipeline handles sharding internally)
- Removed legacy `del train_dataset_group` statements

## Remaining Work

> [!NOTE]
> The following items may need attention before testing:

1. **Smoke test** - Run a short training to verify batch composition
2. **Type hint fixes** - Some pre-existing `ty` warnings remain (not related to this integration)
3. **Unused variables** - `adapter_has_multiplier` and `val_logs` are pre-existing issues

## Validation Checklist

- [ ] Run `uv run scripts/sdxl_peft.py --help` to verify imports
- [ ] Test with a small dataset (5-10 images)
- [ ] Verify latent caching creates `.safetensors` cache files
- [ ] Check epoch shuffling differs between epochs
- [ ] Test with `validation_split > 0` to verify split logic
