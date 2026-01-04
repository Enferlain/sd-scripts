# Data Pipeline Audit Report

## 1. I/O Performance

**Question:** Is `safetensors` `safe_open()` fast enough for per-batch loading?

**Finding:** **Yes, absolutely.**

A synthetic benchmark was run to compare eager loading (`load_file`) vs. lazy loading (`safe_open` + `get_tensor`) on a generated dataset of 1000 files (simulating SDXL latents).

| Metric | Result | Target |
| :--- | :--- | :--- |
| **Throughput (safe_open)** | **~901 batches/sec** | **125 batches/sec** |
| Throughput (load_file) | ~866 batches/sec | - |
| Batch Latency | ~1.11 ms | ~8.00 ms |

The measured throughput is approximately **7x higher** than the target requirement. The I/O overhead for loading latents via `safetensors` is negligible compared to the expected GPU forward pass time.

## 2. Memory Mapping

**Question:** Should we use memory-mapped files for latent caches?

**Finding:** **Yes, we should, but the current implementation uses eager loading.**

The audit of `library/strategies/pipeline_sdxl.py` reveals:

```python
def load_cache(self, path: Path) -> dict[str, torch.Tensor]:
    return load_file(str(path))
```

`load_file` performs a full copy of the file content into RAM. To leverage the memory efficiency and speed benefits of memory mapping (especially for larger SDXL latents or when system RAM is constrained), the implementation should switch to `safe_open`.

**Recommendation:**
Update the strategies to use `safe_open` in the `load_cache` method. This allows reading specific tensors (like just "latents" and not "latents_flipped" if not needed) and relies on the OS page cache rather than allocating Python/Torch memory for the whole file.

## 3. File Handle Management

**Question:** Are we closing file handles properly?

**Finding:** **Yes, the code follows best practices.**

A scan of the codebase shows consistent use of context managers (`with` statements) for file operations:

*   **`safe_open`**: Used correctly with context managers in `is_cache_valid()` and token loading methods.
    ```python
    with safe_open(str(path), framework="pt") as f:
        keys = set(f.keys())
    ```
*   **`Image.open`**: Used correctly with context managers in `dataset_scanner.py`.
*   **Standard I/O**: `open()` calls in `manifest.py` use `with` blocks.
*   **`load_file` / `save_file`**: These are atomic function calls from the `safetensors` library that handle file opening and closing internally. No leaks were found related to these calls.

**Conclusion:**
There are no obvious file handle leaks in the reviewed code.

---

## Summary of Action Items

1.  **Refactor `load_cache`**: Modify `SdxlLatentsPipelineStrategy` and other strategies to use `safe_open` instead of `load_file` for the primary data loading path. This aligns the implementation with the goal of using memory-mapped, lazy loading.
