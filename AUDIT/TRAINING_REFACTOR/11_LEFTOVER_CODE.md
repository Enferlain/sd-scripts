# Audit Report: Leftover Code / Dead Code

**Topic:** 11. Leftover Code / Dead Code
**Date:** 2026-01-18
**Status:** Audit Completed

## 1. Executive Summary

The refactor of `sdxl_peft.py` into a modular `PeftTrainer` class and phase functions in `library/training/phases/` has been largely successful. However, several artifacts from the original monolithic script remain. These include unused imports, a copy-paste error involving `sys.path`, and an unused dataclass.

## 2. Critical Findings

### 2.1. Incorrect `sys.path.append` in `model_prep.py`

**Location:** `library/training/phases/model_prep.py:46`

```python
sys.path.append(os.path.dirname(__file__))  # TODO: Maybe leftover from different location adapters, idk investigate
```

**Issue:**
This line appends the directory of the current file (`library/training/phases/`) to `sys.path`.
In the original script (`scripts/sdxl_peft.py`), this appended `scripts/`, which allowed importing modules from the script's directory.
In its new location, it adds a library subdirectory to the path, which is not a standard practice and serves no clear purpose for finding user-provided adapter modules.

**Recommendation:**
Remove this line entirely. Adapter modules should be discoverable via standard installation or the user's `PYTHONPATH`.

## 3. Dead Code & Unused Definitions

### 3.1. Unused `StepOutput` Dataclass

**Location:** `library/training/trainers/peft_trainer.py:34-45`

```python
@dataclass
class StepOutput:
    """Output from a single training step..."""
    loss: float
    timesteps: Tensor
    did_sync: bool
    metrics: dict = field(default_factory=dict)
```

**Issue:**
This dataclass is defined but never instantiated or returned by any method. The original plan in `TRAINER_PLAN.md` called for `train_step()` to return this object, but the implemented `run_training_loop` keeps the logic inline and does not use this abstraction.

**Recommendation:**
Remove the `StepOutput` dataclass definition.

### 3.2. Unused Import in `model_prep.py`

**Location:** `library/training/phases/model_prep.py:17`

```python
from torch import nn
```

**Issue:**
`nn` is imported but not used in this module.

**Recommendation:**
Remove the unused import.

## 4. Maintenance Items (TODOs)

The following TODO comments were found in the refactored code and should be addressed:

| File | Location | Comment | Action |
|------|----------|---------|--------|
| `library/training/phases/model_prep.py` | Line 40 | `text_encoder = trainer._text_encoder # TODO: Original reference for adapter API` | Verify if `_text_encoder` is still needed for compatibility or if `text_encoders` list is sufficient. |
| `library/training/phases/model_prep.py` | Line 46 | `# TODO: Maybe leftover from different location adapters, idk investigate` | Remove the line (see Critical Findings 2.1). |

## 5. Other Observations

*   **`PeftTrainer._emit`**: This method is currently a no-op placeholder for future event callbacks. It is unused but intentional as part of the architectural design for future extensibility.
*   **`SimpleNamespace` Usage**: The `PeftTrainer` uses `_current_epoch_state` and `_current_step_state` (SimpleNamespaces) to share mutable integers across methods. This is an unusual pattern but appears to be a deliberate design choice to maintain reference semantics with `accelerator.state`.
