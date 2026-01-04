# Data Pipeline Audit - Phase 2: Caching

**Date:** 2024-01-04
**Scope:** `library/strategies/` and `library/data/pipeline/` (Caching Phase)

## 1. Latent Cache Compatibility
**Question:** Are latent cache files compatible with the legacy format?

*   **Finding:** ❌ **No.** The new pipeline generates and reads `.safetensors` files exclusively.
*   **Analysis:**
    *   The `load_cache` abstraction (which would check extensions for `.npz` vs `.safetensors`) described in `DATA_PIPELINE_PLAN.md` was **not implemented**.
    *   The code strictly uses `safetensors.torch.load_file`.
    *   Legacy `.npz` caches (NumPy format) are not supported.
*   **Status:** **Design Decision.** The requirement to load legacy `.npz` files has been intentionally dropped in favor of performance (memory mapping, GIL-free loading) and simplicity. Legacy caches must be regenerated.

## 2. VAE Dtype Handling
**Question:** Do we handle VAE dtype correctly (fp16/bf16/fp32)?

*   **Finding:** ✅ **Yes.**
*   **Analysis:**
    *   Strategies (`SdLatentsPipelineStrategy`, `SdxlLatentsPipelineStrategy`) accept a `dtype` argument (`fp16`, `bf16`, `fp32`).
    *   Latent generation happens in the VAE's native dtype (preserving inference precision).
    *   The resulting tensors are explicitly cast to the requested storage `dtype` before saving:
        ```python
        latents = latents.to(dtype=self._torch_dtype).cpu()
        ```
    *   This correctly decouples the computation precision from the storage precision.

## 3. TE Cache Formats (SD vs SDXL vs SD3)
**Question:** Are TE cache formats correct for SD vs SDXL vs SD3?

*   **Finding:** ⚠️ **Mixed / As Expected**
*   **Analysis:**
    *   **SDXL:** ✅ **Correct.** `SdxlTextEncoderPipelineStrategy` saves `hidden_state1` (CLIP-L), `hidden_state2` (CLIP-G), and `pool2` (Pooled CLIP-G). This matches the requirements for SDXL training.
    *   **SD 1.5/2.0:** ℹ️ **Deferred.** No `SdTextEncoderPipelineStrategy` exists. This is a known low-priority item as SD1.5 TE caching offers minimal performance gains.
    *   **SD3:** ℹ️ **Out of Scope.** No implementation exists, matching the current project scope (SDXL focus).

## 4. Caption Hash Stability
**Question:** Do we handle the "caption hash" for TE cache invalidation?

*   **Finding:** ❌ **Critical Bug**
*   **Analysis:**
    *   The `SdxlTextEncoderPipelineStrategy` (line 217) calculates the caption hash using Python's built-in `hash()` function:
        ```python
        "caption_hash": str(hash(entry.caption) & 0xFFFFFFFF)
        ```
    *   **Issue:** Python's `hash()` function is **randomized** per process (salted) unless `PYTHONHASHSEED` is fixed. This means the same caption will generate different hashes in different runs.
    *   **Impact:** Cache validation will fail reliably across different training sessions or processes, causing unnecessary re-caching of text encoder outputs.
    *   **Requirement:** The code must use the deterministic `stable_string_hash()` function from `library.utils.hash_utils` (using blake2b), as specified in `DATA_PIPELINE_IMPL.md`.

## Summary of Actions Required
*   **Fix Bug:** Replace `hash()` with `stable_string_hash()` in `library/strategies/pipeline_sdxl.py`.
*   **Documentation:** Update documentation to reflect that legacy `.npz` cache support is dropped.
