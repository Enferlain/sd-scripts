# Data Pipeline: Test Plan & Open Questions

## Unit Tests Required

### 1. Dataset Scanner (`test_dataset_scanner.py`)

| Test                                   | What It Should Verify                                     |
| -------------------------------------- | --------------------------------------------------------- |
| `test_scan_directory_basic`            | Finds images, reads captions, returns `ScannedImage` list |
| `test_scan_directory_class_tokens`     | Uses `class_tokens` as fallback when no caption file      |
| `test_scan_directory_is_reg`           | Sets `is_reg=True` correctly, doesn't require captions    |
| `test_scan_directory_validation_split` | Deterministically splits train/val with seed              |
| `test_scan_metadata_file`              | Parses JSON metadata, resolves image paths                |
| `test_create_manifest`                 | Assigns images to buckets, generates correct manifest     |
| `test_create_manifest_from_config`     | Handles train_data_dir, reg_data_dir, in_json, subsets    |
| `test_bucket_resolution_generation`    | Matches legacy bucket resolution output                   |
| `test_bucket_selection`                | Selects correct bucket for various aspect ratios          |

### 2. Epoch Preparation (`test_epoch_preparation.py`)

| Test                           | What It Should Verify                           |
| ------------------------------ | ----------------------------------------------- |
| `test_prepare_epoch_shuffle`   | Different seed+epoch produces different order   |
| `test_prepare_epoch_warmup`    | Largest buckets first                           |
| `test_prepare_epoch_repeats`   | `num_repeats` expands images correctly          |
| `test_caption_processing`      | Shuffle, dropout, prefix/suffix applied         |
| `test_tokenize_epoch_manifest` | Saves tokens to safetensors with correct shapes |
| `test_load_epoch_tokens`       | Loads tokens, validates manifest hash           |

### 3. Dataloader (`test_pipeline_dataloader.py`)

| Test                              | What It Should Verify                                 |
| --------------------------------- | ----------------------------------------------------- |
| `test_training_dataset_iteration` | Yields batches in correct order                       |
| `test_batch_format`               | Contains `latents`, `conditionings`, `captions`, etc. |
| `test_flip_aug`                   | Randomly selects flipped latents                      |
| `test_prior_loss_weight`          | `is_reg=True` uses configured weight                  |
| `test_streaming_tokens`           | `get_slice()` loads per-batch                         |
| `test_on_the_fly_tokenization`    | Works when `tokens_path=None`                         |
| `test_distributed_sharding`       | Rank/world_size splits batches correctly              |

### 4. Caching Engine (`test_pipeline_caching.py`) - Existing

| Test                                    | Status    |
| --------------------------------------- | --------- | --------------------------------------------- |
| `test_batch_entries_by_bucket`          | ✅ Exists |
| `test_split_for_single_gpu`             | ✅ Exists |
| `test_split_for_multi_gpu`              | ✅ Exists |
| `test_cache_dataset_creates_files`      | ✅ Exists |
| `test_skip_existing_caches`             | ✅ Exists |
| `test_all_cached_returns_early`         | ✅ Exists |
| `test_cache_invalidation_bucket_change` | ❌ Needed | Verify re-caching when bucket settings change |

---

## Integration Tests

### 1. End-to-End Pipeline Test (`test_pipeline_integration.py`)

**Purpose:** Test full flow without training (mock VAE/TE)

```python
def test_full_pipeline_flow():
    # 1. Create temp directory with test images + captions
    # 2. scan_directory() → ScannedImage list
    # 3. create_manifest() → DatasetManifest
    # 4. CachingEngine.cache_dataset() with mock strategy
    # 5. prepare_epoch() → EpochManifest
    # 6. create_training_dataloader() → DataLoader
    # 7. Iterate 5 batches, verify batch format
    assert batch["latents"].shape == (batch_size, 4, H, W)
    assert batch["conditionings"] is not None
    assert batch["captions"] is list
```

### 2. Multi-GPU Simulation Test (Existing: `AUDIT/test_pipeline_multigpu.py`)

Move to `tests/unit/` or `tests/integration/`

### 3. Bucket Parity Test (Existing: `AUDIT/bucket_audit_verification.py`)

Move to `tests/unit/` - verifies bucket resolution generation matches legacy

---

## Open Questions & Audit Categories

### AUDIT 1: Training Loop Integration

| Question                                                   | Status  | Notes                                         |
| ---------------------------------------------------------- | ------- | --------------------------------------------- |
| How to handle `current_epoch` / `current_step` References? | ❓ Open | Legacy uses `Namespace` objects passed around |
| How to integrate with `accelerator.prepare()`?             | ❓ Open | DataLoader needs to be prepared per-epoch     |
| Does `skip_first_batches` work with IterableDataset?       | ❓ Open | Needs testing for resume                      |
| How to handle `train_dataset_group.set_max_train_steps()`? | ❓ Open | For token warmup calculation                  |

### AUDIT 2: Batch Format Compatibility

| Question                                                      | Status          | Notes                                                              |
| ------------------------------------------------------------- | --------------- | ------------------------------------------------------------------ |
| Are all strategy methods compatible with new batch?           | ✅ Done         | `peft_strategy_sdxl._get_text_cond` updated                        |
| Is `peft_strategy_sd.py` compatible?                          | ❌ Incompatible | Expects `input_ids_list`, `text_encoder_outputs_list`              |
| Does `conditional_loss` expect specific batch keys?           | ⚠️ Warning      | `apply_masked_loss` silently falls back if `alpha_masks` missing   |
| Are `original_sizes_hw` / `crop_top_lefts` accessed directly? | ⚠️ Partial      | SDXL Strategy is safe; legacy datasets/loggers may access flat keys |

### AUDIT 3: Validation Dataset

| Question                                            | Status    | Notes                        |
| --------------------------------------------------- | --------- | ---------------------------- |
| How is validation dataloader currently created?     | ❓ Open   | Same `DatasetGroup` pattern  |
| Should validation use `prepare_validation_epoch()`? | ✅ Exists | Function ready, not wired    |
| Does validation need per-epoch manifest?            | ❓ Open   | Probably fixed, not shuffled |

### AUDIT 4: Resume & Checkpointing

| Question                                            | Status  | Notes                                |
| --------------------------------------------------- | ------- | ------------------------------------ |
| Does epoch manifest need to be saved in checkpoint? | ❓ Open | For exact resume                     |
| How to resume mid-epoch with new dataloader?        | ❓ Open | Skip batches or regenerate manifest? |
| Token file handling on resume?                      | ❓ Open | Need to regenerate or save?          |

### AUDIT 5: Memory & Performance

| Question                                    | Status      | Notes                             |
| ------------------------------------------- | ----------- | --------------------------------- |
| Per-epoch dataloader creation overhead?     | ❓ Open     | Needs benchmarking                |
| Is `prepare_epoch()` fast enough?           | ✅ Designed | Caption processing + shuffle only |
| ThreadPool per-batch vs persistent workers? | ⚠️ TODO     | In DATA_PIPELINE_IMPL.md          |

### AUDIT 6: Cache Invalidation

**Context:** Legacy behavior only re-buckets changed images but can cause torch stack errors if cached latent res doesn't match new bucket. We are safer - `is_cache_valid()` checks `bucket_reso` mismatch.

| Question                                              | Status      | Notes                                       |
| ----------------------------------------------------- | ----------- | ------------------------------------------- |
| Does changing bucket_reso_steps invalidate caches?    | ✅ Designed | `is_cache_valid()` checks bucket_reso       |
| Config-hash namespacing for separate cache dirs?      | ⚠️ ROADMAP  | Would fully prevent cross-config issues     |
| What happens if user changes resolution mid-training? | ❓ Open     | Re-caches affected images, not full dataset |

---

## Next Steps

1. **Create unit tests** for scanner and dataloader (tests above)
2. **Move AUDIT test files** to `tests/unit/` or `tests/integration/`
3. **Create training loop audit doc** answering integration questions
4. **Benchmark `prepare_epoch()`** for various dataset sizes
5. **Implement parallel pipeline** with config flag in `sdxl_peft.py`
