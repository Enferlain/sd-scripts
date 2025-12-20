# Project Roadmap & Future Ideas

## Testing Suite

**Current Status (2025-12-20):**

- ✅ Testing infrastructure complete (pytest, fixtures, coverage)
- ✅ 115+ unit tests passing (28 config + 26 optimizer + 23 checkpointing + 17 diffusion + 21 noise_utils)

**Next Steps:**

1. **Expand Core Module Tests** - Target 70% coverage on training modules
   - ~~`library/training/model_prep.py` - Model preparation and wrapping~~ ✅ Complete
   - ~~`library/training/diffusion.py` - Diffusion utilities~~ ✅ Complete
   - ~~`library/training/noise_utils.py` - Noise generation~~ ✅ Complete
2. **Data Module Tests** - Critical for ensuring data pipeline correctness
   - ~~`library/data/dataset.py` - Dataset loading and bucketing~~ ✅ Complete (Unit tests only)
   - ~~`library/data/data_structures.py` - Data structures and batching~~ ✅ Complete
   - `library/data/image_utils.py` - Image preprocessing
   - ~~`library/utils/common_utils.py`~~ ✅ Complete (Tested `str_to_dtype`, `size`, `GradualLatent`)
   - ~~`library/utils/safetensors_utils.py`~~ ✅ Complete (Tested I/O, metadata, large tensors)
   - ~~`library/losses/loss.py`~~ ✅ Complete (Tested stable losses, fixed bugs in SmoothL1)
   - `library/losses/loss_weighting.py` - SNR weighting logic
   - `library/timestep_samplers/` - Sampler initialization and step logic
3. **Integration Tests** - Validate full workflows
   - **Data Loading**: `dataset.py` caching methods (`cache_latents`, `cache_text_encoder_outputs`) and image loading (requires filesystem/GPU mocks)
   - End-to-end config → training setup
   - Checkpoint save/load cycles
   - Multi-GPU scenarios (requires_gpu marker)
4. **CI/CD Setup** - Automate testing

   - GitHub Actions workflow for pytest
   - Coverage reporting and tracking
   - Pre-commit hooks for running tests

5. **Documentation** - Testing best practices
   - Update `DEVELOPMENT_GUIDE.md` with testing patterns
   - Document fixture usage and test organization
   - Add testing examples for contributors

---

## Configuration Refactoring

- [ ] **Consolidate Learning Rate Configurations**: Unify the handling of learning rates across different training modes (LoRA vs Fine-tune) and models (SD1.5 vs SDXL). Currently, there is a mix of `text_encoder_lr` (List/Any in NetworkConfig) and `learning_rate_te1/te2` (floats in SDXLConfig).
- [ ] **Type Safety**: Improve type definitions for `text_encoder_lr` to avoid `Any` when possible, perhaps by using custom validators or strict union handling if OmegaConf improves.
- Will need to strip BASE level code from sd_peft and sd_textual_inversion and sd_finetune. Currently it's base (everything imports) AND sd1/2 combined.

## Testability Improvements (Future)

- **Split [prepare_accelerator](cci:1://file:///d:/Projects/sd-scripts/library/training/trainer_utils.py:10:0-90:22)** - Separate config computation from side effects
- **Explicit step 0 validation** - Clarify [calculate_val_loss_check](cci:1://file:///d:/Projects/sd-scripts/library/training/trainer_utils.py:118:0-138:15) behavior at step 0

### Modules Requiring Heavier Mocking

These modules have substantial side effects requiring mocked Accelerate/tokenizers/VAE:

- `trainer_utils.py` - `init_trackers()`, `determine_grad_sync_context()` - need Accelerator mocks
- `caching.py` - `cache_batch_latents()`, `cache_batch_text_encoder_outputs()` - need VAE/encoder mocks
- `prompt_utils.py` - `get_prompts_with_weights()`, `get_weighted_text_embeddings()` - need tokenizer mocks
- `dataset.py` - `cache_latents()`, `register_image()`, `__getitem__` - requires filesystem and VAE interaction mocks
- `data_structures.py` - `BucketManager.make_buckets()`, `AugHelper.color_aug()` - depends on model_util and OpenCV/randomness
- ~~`safetensors_utils.py` - `mem_eff_save_file`, `load_safetensors` - requires temporary file creation/cleanup~~ ✅ Complete
- ~~`common_utils.py` - `swap_weight_devices` - requires CUDA context/mocks~~ ✅ Complete
