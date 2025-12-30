# Library Audit Report

## 1. Executive Summary

This audit assesses the current state of the `library/` folder, focusing on code organization, modularity, and adherence to the "General Script -> Model Specific Script" schema.

**Major Findings:**
- `peft_common.py` is an amalgamation that should be fully dissolved into specific modules.
- `common_utils.py` serves as a "junk drawer" containing logging, image processing, torch utilities, and scheduler extensions. These should be moved to their respective domains.
- `trainer_utils.py` mixes logging setup, accelerator preparation, and validation logic.
- `model_metadata.py` and `checkpointing.py` share some logic (hashing) that should be consolidated.
- `optimizer.py` is large but relatively cohesive; however, scheduler logic could be extracted.

## 2. Refactoring Recommendations

### 2.1 Dismantling `peft_common.py`

This file is a collection of disparate functions. It should be emptied and deleted.

| Function | Current Location | Proposed Location | Rationale |
| :--- | :--- | :--- | :--- |
| `prepare_datasets` | `peft_common.py` | `library/data/dataset_setup.py` | Handles dataset group/collator creation. |
| `calculate_initial_step` | `peft_common.py` | `library/training/training_loop.py` | Core training loop calculation. |
| `register_adapter_state_hooks` | `peft_common.py` | `library/training/checkpointing.py` | Specific to saving/loading logic. |
| `generate_step_logs` | `peft_common.py` | `library/logging/step_logging.py` | Pure logging generation logic. |
| `step_logging` | `peft_common.py` | `library/logging/step_logging.py` | Logging action. |
| `epoch_logging` | `peft_common.py` | `library/logging/step_logging.py` | Logging action. |
| `accelerator_logging` | `peft_common.py` | `library/logging/step_logging.py` | Logging action. |
| `create_training_metadata` | `peft_common.py` | `library/utils/model_metadata.py` | Generates metadata for saved models. |
| `resolve_adapter_kwargs` | `peft_common.py` | `library/adapters/lora_utils.py` | PEFT/LoRA specific utility. |

### 2.2 Dismantling `common_utils.py`

This file violates the single-responsibility principle.

| Function/Class | Proposed Location | Rationale |
| :--- | :--- | :--- |
| `setup_logging` | `library/logging/logging_setup.py` | App logging configuration. |
| `swap_weight_devices` | `library/utils/torch_utils.py` | Generic PyTorch tensor operation. |
| `weighs_to_device` | `library/utils/torch_utils.py` | Generic PyTorch tensor operation. |
| `str_to_dtype` | `library/utils/torch_utils.py` | Generic PyTorch type conversion. |
| `resize_image`, `pil_resize` | `library/data/image_utils.py` | Image processing. |
| `validate_interpolation_fn` | `library/data/image_utils.py` | Image processing validation. |
| `GradualLatent` | `library/timestep/samplers/gradual_latent.py` | Specific sampling strategy. |
| `EulerAncestral...GL` | `library/timestep/samplers/euler_gl.py` | Specific scheduler implementation. |

### 2.3 Refactoring `trainer_utils.py`

This file contains mixed concerns (Setup, Logging, Training Loop).

| Function | Proposed Location | Rationale |
| :--- | :--- | :--- |
| `prepare_accelerator` | `library/training/training_setup.py` | Environment/Accelerator setup. |
| `init_trackers` | `library/logging/logging_setup.py` | Tracker initialization. |
| `calculate_val_loss_check` | `library/training/validation.py` | Logic to determine if validation runs. |
| `append_lr_to_logs*` | `library/logging/step_logging.py` | Logging helper. |
| `determine_grad_sync_context` | `library/training/training_loop.py` | Training loop execution details. |

### 2.4 Code Duplication & Consistency

**Safetensors Hashing:**
- `library/utils/model_metadata.py` has `precalculate_safetensors_hashes` (placeholder/NotImplemented).
- `library/training/checkpointing.py` has `precalculate_safetensors_hashes` (implemented).
- **Recommendation:** Move the implementation to `library/utils/safetensors_utils.py` (which exists) or `library/utils/model_metadata.py` and have `checkpointing.py` import it.

**Scheduler Logic:**
- `optimizer.py` contains `get_scheduler_fix` which is quite large.
- **Recommendation:** Extract to `library/training/schedulers.py` to keep `optimizer.py` focused on optimizers.

**Config Utils:**
- `library/config/config_util.py` contains `Blueprint` classes which are effectively Data Transfer Objects (DTOs) for Dataset creation.
- **Recommendation:** These seem fine here as they bridge Config -> Dataset, but moving `BlueprintGenerator` to `library/data/blueprint.py` could clarify `config_util`'s purpose if it becomes too cluttered.

## 3. Proposed File Structure Changes

```
library/
  data/
    dataset_setup.py      <-- NEW (from peft_common.py)
    image_utils.py        <-- UPDATE (absorb common_utils.py image ops)
  logging/
    logging_setup.py      <-- NEW (setup_logging, init_trackers) - Renamed from setup.py
    step_logging.py       <-- NEW (logging functions from peft_common, trainer_utils)
  training/
    training_setup.py     <-- NEW (prepare_accelerator) - Renamed from setup.py
    training_loop.py      <-- NEW (step calculation, grad sync)
    validation.py         <-- NEW (validation checks)
    checkpointing.py      <-- UPDATE (absorb register_adapter_state_hooks)
  utils/
    torch_utils.py        <-- UPDATE (absorb common_utils.py torch ops)
    model_metadata.py     <-- UPDATE (absorb create_training_metadata)
  adapters/
    lora_utils.py         <-- UPDATE (absorb resolve_adapter_kwargs)
```

## 4. Next Steps

1.  Create the new modules defined in Section 3.
2.  Move functions one by one, ensuring imports are updated in all consumers (scripts).
3.  Delete `peft_common.py` and `common_utils.py` once empty.
4.  Standardize imports in scripts to use the new locations.
