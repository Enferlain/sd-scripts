# Data Pipeline Memory Efficiency Audit

**Date:** 2026-01-04
**Auditor:** Jules (AI Agent)
**Scope:** `library/data/pipeline/` vs Legacy `library/data/_deprecated/`

## Executive Summary

The new data pipeline architecture represents a significant improvement in memory predictability and efficiency compared to the legacy system. By separating scanning, caching, and training into distinct phases, the runtime memory footprint is now deterministic.

**Key Findings:**
1.  **Memory Architecture**: Moved from "load everything upfront" to "stream-from-disk".
2.  **Critical Bug**: The `streaming_tokens=True` mode is currently broken (logic gap), meaning token loading currently requires ~120MB RAM for 100k images (loading all upfront). This is acceptable for target hardware but breaks the "streaming" promise for massive datasets.
3.  **Leaks**: No significant memory leaks identified. Resource management (file handles, threads) is handled correctly.

---

## 1. Audit Questions & Answers

### Q1: Are we loading too much upfront vs streaming?

**Verdict: Mostly efficient, with one exception.**

*   **Images/Latents:** **Efficient.** The system strictly streams latents from disk (`.safetensors`) only when needed for the current batch. It does *not* load the entire dataset's latents into RAM, which was a risk in some legacy configurations.
*   **Manifests:** **Acceptable.** The `DatasetManifest` and `EpochManifest` are loaded into RAM. For a 100k image dataset, this consumes ~50-100MB, which is negligible on 32GB systems.
*   **Tokens:** **Upfront (Current Limitation).** The `TrainingDataset` currently loads **all** tokenized captions into memory at initialization (`self._load_all_tokens()`).
    *   *Impact:* For 100k images (2 encoders, 77 tokens, int64), this is ~120MB.
    *   *Status:* The code contains a `streaming_tokens` flag, but the implementation is incomplete (see Critical Issues below).

### Q2: Is pin_memory actually helping or wasting RAM?

**Verdict: Helpful and recommended.**

*   **Mechanism:** `TrainingDataset` loads tensors from `.safetensors` (disk) -> CPU RAM. `DataLoader(pin_memory=True)` then copies these tensors to page-locked (pinned) RAM before transfer to GPU.
*   **Analysis:** While this involves an extra copy in system RAM, it enables **asynchronous GPU transfer**. Without pinning, `batch.to('cuda')` blocks the CPU.
*   **Cost:** The memory overhead is limited to `prefetch_factor * num_workers` batches. For a batch size of 4 at 1024x1024, this is roughly `4 * 4 * 16MB ≈ 256MB` of pinned memory.
*   **Conclusion:** The throughput gain (non-blocking transfer) significantly outweighs the modest RAM cost.

### Q3: Do we have memory leaks in cache loading?

**Verdict: No leaks found.**

*   **Caching Engine:** `CachingEngine` correctly manages resources.
    *   Images are opened, processed, and explicitly closed (`img.close()`).
    *   `ThreadPoolExecutor` is used as a context manager (or created/destroyed per batch), ensuring worker threads don't accumulate.
    *   Tensors created during inference are local to the loop and collected by Python's GC.
*   **Validation:** Static analysis shows proper scope management. No global lists accumulate tensor references.

---

## 2. Critical Issues

### 1. Broken Streaming Token Loading
**Severity: High (Functionality)** / Low (Memory impact for typical datasets)

The `TrainingDataset` class has a logic gap when `streaming_tokens=True` is used.

*   **File:** `library/data/pipeline/dataloader.py`
*   **Issue:** In `_load_batch`, there is no implementation for reading tokens from disk on-the-fly when `self._tokens` is None.
*   **Result:** Batches returned in streaming mode lack the `input_ids` key, causing training to fail.
*   **Code Gap:**
    ```python
    # library/data/pipeline/dataloader.py

    # Current Logic
    if self._tokens is not None and batch_size > 0:
        # Load from memory dict
        batch["input_ids"] = ...
    elif batch_info.input_ids:
        # Load from legacy batch info
        ...
    # MISSING: else branch for streaming from self.tokens_path
    ```

---

## 3. Comparison with Legacy System

| Feature | Legacy (`dataset.py`) | New Pipeline | Improvement |
| :--- | :--- | :--- | :--- |
| **Object Model** | `ImageInfo` "God Object" holding paths, latents, captions, and state. Hard to serialize. | `CacheEntry` (static metadata) + `BatchInfo` (runtime instructions). | **High**: separation of static data vs runtime state. |
| **Bucketing** | Dynamic/On-the-fly. Buckets cleared/refilled during training. | Pre-computed in Phase 1. Buckets are static logic concepts. | **High**: Deterministic batches, zero training-time overhead. |
| **Memory** | Unpredictable. High-res images could clump, causing OOMs. Latents often kept in RAM. | **Deterministic**. Batches ordered by resolution (warmup). Latents never held in RAM. | **High**: "Warmup largest first" strategy prevents late-training OOMs. |
| **Tokens** | Processed text on-the-fly (tokenizer overhead every epoch). | Pre-tokenized to `.safetensors`. | **Medium**: Saves CPU cycles; memory usage similar. |

---

## 4. Recommendations

1.  **Prioritize Fixing Streaming Tokens:** This is the most critical actionable item. Implement the missing branch in `TrainingDataset._load_batch` to read tokens from disk on-the-fly.
    *   *Implementation Guidance:* `safe_open` from the `safetensors` library does not support efficient random access slicing on all backends.
    *   *Suggested Approach:* Use `numpy.memmap`. Since safetensors headers contain the offset and length of each tensor, you can map the file as a numpy array and slice it directly without loading the entire file. This is zero-copy and OS-managed.
    *   *Alternative:* If `numpy.memmap` is too complex to integrate with the existing safetensors file structure, consider keeping the `safe_open` context alive (if thread-safe) or opening/closing per batch (though this incurs overhead).

2.  **Optimize ThreadPool Usage:** In `CachingEngine._cache_batch`, a new `ThreadPoolExecutor` is created for *every batch*.
    *   *Recommendation:* Move the executor to `CachingEngine.__init__` to reduce thread creation overhead, though this is a CPU optimization, not memory.

3.  **JSON vs Binary Manifests:** For datasets >1M images, `DatasetManifest` (JSON) parsing will become a bottleneck (GBs of RAM).
    *   *Future Work:* Consider using SQLite or Parquet for the manifest if scaling to millions of images.

4.  **Pinning Optimization:** Ensure `prefetch_factor` is not set excessively high (default 2 is fine). High prefetch factors multiply the pinned memory requirement.

---

## Conclusion
The new pipeline is architecturally sound and memory-safe. The only identified functional regression is the unimplemented streaming token mode, which currently forces an "all-upfront" load for tokens (manageable for <500k images).
