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

### Methodology

The benchmark was conducted using a synthetic script (`benchmark_safetensors.py`) with the following parameters:

*   **Dataset**: 1000 generated `.safetensors` files.
*   **Content**: Random fp16 tensors of shape `(4, 128, 128)` to simulate SDXL latents (corresponding to 1024x1024 images encoded by VAE).
*   **Batch Size**: 4.
*   **Environment**: Standard SSD storage, single process.
*   **Metric**: Total time to load all 1000 files in batches, converted to batches/second.

The benchmark compared two approaches:
1.  **Eager**: `safetensors.torch.load_file(path)` which reads the entire file into memory.
2.  **Lazy**: `with safe_open(path) as f: f.get_tensor("latents")` which uses memory mapping.

## 2. Memory Mapping

**Question:** Should we use memory-mapped files for latent caches?

**Finding:** **Yes, we should, but the current implementation uses eager loading.**

The audit of `library/strategies/pipeline_sdxl.py` reveals:

```python
def load_cache(self, path: Path) -> dict[str, torch.Tensor]:
    return load_file(str(path))
```

`load_file` performs a full copy of the file content into RAM. To leverage the memory efficiency and speed benefits of memory mapping (especially for larger SDXL latents or when system RAM is constrained), the implementation should switch to `safe_open`.

### Quantification of Benefits

Switching to `safe_open` offers specific memory advantages:

*   **Zero-Copy**: `safe_open` uses memory mapping (mmap). Tensors are effectively views into the file on disk (cached by the OS), avoiding a user-space memory allocation and copy.
*   **Selective Loading**: Latent cache files may contain auxiliary data like `latents_flipped` or `alpha_mask`. `load_file` loads *everything*. `safe_open` allows loading only the required `"latents"` tensor, saving RAM.
    *   *Example*: If a cache file is 100MB but only 50MB is needed for the current training step (no flip augmentation), `load_file` wastes 50MB RAM. `safe_open` wastes 0MB.
*   **Impact**: For a batch size of 4 with SDXL latents (~32KB each), the absolute difference is small. However, with multiple dataloader workers (e.g., 8 workers) and prefetching, the reduced memory pressure helps avoid OOMs in tight environments.

**Proposed Implementation Pattern:**

```python
# library/strategies/pipeline_sdxl.py

def load_cache(self, path: Path) -> dict[str, torch.Tensor]:
    # Use safe_open for lazy, memory-mapped loading
    with safe_open(str(path), framework="pt") as f:
        # Only load what we need
        latents = f.get_tensor("latents")
        return {"latents": latents}
```

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

1.  **Refactor `load_cache`**: Modify the following strategy classes to use `safe_open` instead of `load_file`.
    *   **Priority**: High (Low Effort).
    *   **Affected Classes**:
        *   `SdxlLatentsPipelineStrategy` in `library/strategies/pipeline_sdxl.py`
        *   `SdLatentsPipelineStrategy` in `library/strategies/pipeline_sd.py`
    *   **Goal**: Align implementation with the memory-mapped, lazy loading architecture.
