# Data Pipeline Audit: Phase 6 - Cache Invalidation

**Date:** October 26, 2023
**Auditor:** Jules (AI Agent)
**Scope:** `library/data/pipeline/`, `library/strategies/`

## Executive Summary

This audit evaluates the robustness of the cache invalidation logic in the new data pipeline. Specifically, it verifies whether changes in training configuration (resolution, bucket steps) correctly trigger re-caching of latents to prevent dimension mismatches during training.

The new pipeline implements a strict validity check (`is_cache_valid`) that compares stored metadata with current configuration requirements, ensuring that stale caches are automatically identified and regenerated.

## Detailed Findings

| Question | Status | Implementation Details |
| :--- | :--- | :--- |
| **Does changing `bucket_reso_steps` invalidate caches?** | ✅ **Designed** | **Yes.** The system explicitly validates that the cached latent resolution matches the expected bucket resolution derived from the current config. |
| **Config-hash namespacing for separate cache dirs?** | ⚠️ **Roadmap** | **Not Implemented.** Caches are stored directly in the user-specified `cache_dir`. There is no automatic segregation by config hash. |
| **What happens if user changes resolution mid-training?** | ✅ **Verified** | **Selective Re-caching.** Only images whose bucket assignments change (resulting in a resolution mismatch) are re-cached. Valid images are preserved. |

## Code Analysis

### 1. Bucket Resolution Validation

**File:** `library/strategies/sd_caching.py` / `library/strategies/sdxl_caching.py`
**Method:** `is_cache_valid`

The caching strategy enforces strict resolution checks. When `bucket_reso_steps` or base resolution changes, the `entry.bucket_reso` calculated during Phase 1 (Manifest Generation) changes. The strategy compares this new requirement against the cached file's properties.

```python
# library/strategies/sdxl_caching.py

def is_cache_valid(self, path: Path, entry: CacheEntry, ...):
    with safe_open(str(path), framework="pt") as f:
        # ... existence checks ...

        # 1. Tensor Shape Check
        latents = f.get_tensor("latents")
        expected_h = entry.bucket_reso[1] // 8
        expected_w = entry.bucket_reso[0] // 8

        # If bucket_reso_steps changed, expected dimensions change, causing mismatch here
        if latents.shape != (4, expected_h, expected_w):
            return False

        # 2. Metadata Explicit Check
        metadata = f.metadata()
        if metadata:
            stored_bucket = metadata.get("bucket_reso", "")
            expected_bucket = f"{entry.bucket_reso[0]},{entry.bucket_reso[1]}"
            if stored_bucket and stored_bucket != expected_bucket:
                return False

    return True
```

This prevents the "torch stack error" seen in legacy systems where a dataloader might try to stack tensors of different sizes because it loaded a stale cache file.

### 2. Selective Re-caching (Mid-Training Changes)

**File:** `library/data/pipeline/caching_engine.py`
**Method:** `_get_entries_to_cache`

When the user restarts training with new settings, the `CachingEngine` iterates through the manifest. It relies on `is_cache_valid` to decide whether to reuse or regenerate.

```python
# library/data/pipeline/caching_engine.py

def _get_entries_to_cache(self, ...):
    entries = []
    for entry in manifest.entries.values():
        if skip_existing:
            cache_path = self.strategy.get_cache_path(entry, cache_dir)
            if cache_path.exists():
                # Full validation: check cache contents
                if self.strategy.is_cache_valid(cache_path, entry, ...):
                    # Cache is valid, reuse it
                    entry.latent_cache_path = str(cache_path)
                    continue
                else:
                    logger.debug(f"Cache invalid for {entry.id}, will re-cache")
        # Cache missing or invalid, add to processing list
        entries.append(entry)
    return entries
```

This logic ensures that if a user changes resolution from `512` to `768`, only the images that need new bucket resolutions are re-processed. Images that might still map to the same resolution (rare, but possible depending on bucketing logic) would be skipped, but critically, any image that *needs* a new size is guaranteed to be updated.

### 3. Namespace Isolation (Config Hashing)

**Status:** Not Implemented

The current implementation relies on the user to manage `cache_dir`. If a user toggles between two very different configurations (e.g., SD1.5 vs SDXL) pointing to the same directory, the files will overwrite each other (if IDs collide) or cause constant re-caching due to validation failures.

**Recommendation for Future:**
Implement an automatic subdirectory based on a hash of critical config parameters (resolution, flip, alpha mask, model version).

```python
# Proposed logic
config_hash = stable_hash(resolution, bucket_steps, model_version)
actual_cache_dir = user_cache_dir / config_hash
```

## Legacy Comparison

**Legacy File:** `library/data/_deprecated/caching.py`

The legacy system's validation was less robust. While `is_disk_cached_latents_is_expected` did check shapes, the monolithic structure made it difficult to decouple "checking" from "loading", often leading to race conditions or silent failures if the bucket manager's state drifted from the disk state.

```python
# Legacy check
if npz["latents"].shape[1:3] != expected_latents_size:
    return False
```

The new pipeline improves on this by:
1.  **Metadata headers:** Explicitly storing `bucket_reso` in `.safetensors` header allows fast validation without loading the tensor body.
2.  **Decoupling:** `is_cache_valid` is a pure boolean check separate from the loading logic, allowing for a "dry run" or "scan" phase (Phase 1/2 boundary) before heavy compute begins.

## Verification

The audit claims have been validated through the following means:

1.  **Unit Tests:**
    *   `tests/unit/data/test_pipeline_caching.py`: Tests the `CachingEngine` orchestration logic, including the `_get_entries_to_cache` filtering based on validity.
    *   `tests/unit/strategies/test_strategies_sd.py`: Verifies `SdLatentsPipelineStrategy.is_cache_valid` logic for SD1.5/2.
    *   `tests/unit/strategies/test_strategies_sdxl.py`: Verifies `SdxlLatentsPipelineStrategy.is_cache_valid` logic for SDXL, including metadata parsing.

2.  **Manual Verification:**
    *   Simulated configuration changes (bucket steps change) in local environment confirmed that `is_cache_valid` returns `False` for mismatched entries.

3.  **Source:**
    *   These features were introduced and consolidated in the data pipeline refactor PRs associated with Phase 2 (Caching Pipeline) and Phase 6 (Cache Invalidation).

## Conclusion

The new data pipeline successfully addresses the core cache invalidation issues. By embedding target resolution metadata in the cache files and enforcing a strict check against the current manifest's expectations, it guarantees data integrity during training, even when configuration parameters change.
