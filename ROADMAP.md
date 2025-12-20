# Project Roadmap & Future Ideas

## Testing Suite

**Current Status (2025-12-20):**

- ✅ Testing infrastructure complete (pytest, fixtures, coverage)
- ✅ 115+ unit tests passing (28 config + 26 optimizer + 23 checkpointing + 17 diffusion + 21 noise_utils)

**Next Steps:**

1. **Expand Core Module Tests** - Target 70% coverage on training modules
   - `library/training/model_prep.py` - Model preparation and wrapping
   - ~~`library/training/diffusion.py` - Diffusion utilities~~ ✅ Complete
   - ~~`library/training/noise_utils.py` - Noise generation~~ ✅ Complete
2. **Data Module Tests** - Critical for ensuring data pipeline correctness
   - `library/data/dataset.py` - Dataset loading and bucketing (high priority)
   - `library/data/data_structures.py` - Data structures and batching
   - `library/data/image_utils.py` - Image preprocessing
3. **Integration Tests** - Validate full workflows
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
