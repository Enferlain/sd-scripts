# TODO Resolutions

Investigated TODO comments with recommended resolutions.

**Complexity Scale**: 1 (trivial) → 5 (major refactor)

---

## Training Scripts

### `sd_peft.py:282` - Remove `hasattr` checks for adapter methods ⭐ C:3

**Current State**: Uses `hasattr` to check if adapter has optional methods before calling them.

**Problem**: Different adapters (LoRA, LyCORIS) may not implement all methods consistently.

**Resolution**: Create ABC `AdapterBase` with default no-op methods. All adapters inherit and override as needed.

```python
# library/adapters/adapter_base.py
class AdapterBase(ABC):
    def prepare_adapter(self, cfg) -> None:
        """Called before training. Override for custom prep."""
        pass

    def apply_max_norm_regularization(self, scale, device) -> tuple:
        """Override to enable scale_weight_norms."""
        raise NotImplementedError("This adapter doesn't support weight norm scaling")

    def set_multiplier(self, multiplier: float) -> None:
        """Override to support dynamic multiplier."""
        pass
```

**Why ABC over Protocol**: Can provide default implementations. Protocol would require every adapter to implement everything or use hasattr anyway.

---

### `sd_peft.py:335,541` / `peft_common.py:399` - Why only text_encoder_lr? ⭐ C:2 - ✅ Addressed

**Current State**: `prepare_optimizer` returns only `text_encoder_lr`, not `unet_lr`.

**Context**: This is used in metadata recording. Looking at `create_training_metadata`, it records `ss_text_encoder_lr` but UNet LR comes from `cfg.optimizer.learning_rates.unet`.

**Resolution**: Either:

1. Return both LRs from `prepare_optimizer` for consistency
2. Remove return entirely - get LRs from config in metadata function

**Recommendation**: Option 2 - metadata should read from config directly. Remove `text_encoder_lr` return, update `create_training_metadata` to use `cfg.optimizer.learning_rates`.

---

### `sd_peft.py:614` / `sdxl_peft.py:611` - Automate TE deletion ⭐ C:2 - 🔶 To be addressed after/as part of dataset refactors

**Current State**: Manual check `is_text_encoder_not_needed_for_training()` before deleting text encoder.

**Proposed**: Automate after SDXL sample prompt cache is implemented.

**Resolution**: This is a future enhancement, deferred until sample prompt caching is implemented for SDXL.

**Status**: DEFERRED

---

### `sd_finetune.py:479` - epoch variable reference ⭐ C:1 - ✅ Addressed

**Current State**: Warning about `epoch` potentially referenced before assignment.

**Resolution**: Investigate control flow. Likely false positive from linter, but verify `epoch` is always set before this line.

---

### `sdxl_textual_inversion.py:65` - noisy_latents dtype ⭐ C:1 - ✅ Addressed

**Current State**: `noisy_latents = noisy_latents.to(weight_dtype)  # TODO check why noisy_latents is not weight_dtype`

**Resolution**: Investigate `get_noise_noisy_latents_and_timesteps` return dtype. Either fix at source or document why cast is required.

---

## Training Utilities

### `optimizer.py:423` - Expected float got None ⭐ C:2 - 🔶 Type mismatch due to adafactor relative_step managing lr

**Current State**: `optimizer_config.learning_rates.base = None  # TODO: expected float got none?`

**Resolution**: This is inside schedulefree wrapper logic. Verify if setting to None is intentional (to force use of schedulefree internal LR) or a bug.

---

### `optimizer.py:498,607,608,669` - Unresolved attribute warnings ⭐ C:1 - ✅ Addressed

**Current State**: IDE warnings about `.split()`, `.train()`, `.eval()`, `.base_optimizer` attributes.

**Resolution**: These are valid methods on the optimizer objects at runtime. Add type hints or `# type: ignore` comments to silence linter. Low priority.

---

### `optimizer.py:685,701` - Decay steps UI support ⭐ C:2 - ✅ Addressed

**Current State**: Comments about adding UI inputs for decay steps, temp fix for warmup.

**Resolution**: Track as feature request. Requires Hydra config additions and potentially UI changes. Not blocking.

---

### `checkpointing.py:23,181` / `sdxl_checkpointing.py:16` - Old comments ⭐ C:1 - ✅ Addressed

**Current State**: Stale comments about TrainingConfig bug (already fixed), Hydra issue.

**Resolution**:

- Line 23/16: Delete outdated comment (bug is fixed)
- Line 181: Investigate `resume_from_huggingface` Hydra integration

---

### `trainer_utils.py:185` - Ivnestigation TODO ⭐ C:1 - 🔶 Pending deeper investigation for why this was considered

**Current State**: `# TODO` to investigate sync gradients for full_bf16 training.

**Resolution**: Delete or investigate surrounding code to determine intent.

---

## Strategies

### `strategy_base.py:14,160,465` - ImageInfo circular import ⭐ C:3 - 🔶 To be addressed after/as part of dataset refactors

**Current State**: Multiple comments about circular import with `ImageInfo`.

**Resolution**: Move `ImageInfo` to a separate module (e.g., `library/data/image_info.py`). Update all imports.

---

### `strategy_base.py:209` - Batch input support ⭐ C:2 - 🔶 To be addressed after/as part of dataset refactors

**Current State**: Single image encoding, comment asks for batch support.

**Resolution**: Modify encoding functions to accept batches. Verify all callers can handle batch output.

---

### `strategy_base.py:369` - Commonize utilities ⭐ C:2 - 🔶 To be addressed after/as part of dataset refactors

**Current State**: Utility functions scattered, should be in base class.

**Resolution**: Audit strategy classes for common patterns, move to base.

---

### `peft_strategy_base.py:151,153` - is_sdxl redundancy ⭐ C:2 - ✅ Addressed

**Current State**: `is_sdxl: bool = False` exists alongside `model_type` in config.

**Resolution**: Remove `is_sdxl` field, derive from `cfg.model.model_type == "sdxl"` where needed.

---

## Config

### `config_validation.py:101` - block_lr 23-value requirement ⭐ C:2 - 🔶 To be addressed as part of block/layer granular training support

**Current State**: Hardcoded validation for 23 block LRs.

**Resolution**: Make dynamic based on model architecture, or accept shorter lists with fallback to base LR.

---

### `config_util.py:458` - Arbitrary dataset comment ⭐ C:1 - 🔶 To be addressed after/as part of dataset refactors

**Current State**: Unclear comment about subset config.

**Resolution**: Clarify or remove comment.

---

## Models

### `original_unet.py:668` / `sdxl_original_unet.py:473` - Hypernetworks ⭐ C:4 - 🔶

**Current State**: No hypernetwork support.

**Resolution**: Large feature. Either implement or mark as WONTFIX.

**Status**: DEFERRED (feature request), not priority/old tech

---

### `model_util.py:1252` - Memory consumption ⭐ C:2 - 🔶

**Current State**: Comment notes high memory usage.

**Resolution**: Profile and optimize, or add memory cleanup.

---

## Utilities

### `torch_utils.py:24` - prepare_dtype dual concerns ⭐ C:2 - 🔶 Needs split for cleaner code

**Current State**: Function handles both saving and training dtype concerns.

**Resolution**: Split into `prepare_training_dtype()` and `prepare_save_dtype()` for clarity.

---

### `sai_model_spec.py:220` - Naming question ⭐ C:1

**Current State**: Name `sai_model_spec` but handles multiple model types.

**Resolution**: Historical naming from SAI (Stability AI). Add docstring clarifying it's a general model spec utility.

---

## Priority Order

1. **High** (blocking or causing confusion):

   - `strategy_base.py` ImageInfo circular import (C:3)
   - `sd_peft.py:282` adapter ABC (C:3)

2. **Medium** (cleanup, maintainability):

   - `optimizer.py` LR questions (C:2)
   - `peft_strategy_base.py` is_sdxl cleanup (C:2)
   - `config_validation.py` block_lr flexibility (C:2)

3. **Low** (comments, minor cleanup):

   - Stale comments, empty TODOs (C:1)
   - Type hint warnings (C:1)

4. **Deferred**:
   - Hypernetwork support
   - TE auto-deletion with prompt cache
   - Decay steps UI
