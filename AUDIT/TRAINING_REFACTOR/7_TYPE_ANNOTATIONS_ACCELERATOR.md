# Audit Report: Type Annotations (Accelerator | None)

## 1. Problem Description

In `library/training/trainers/peft_trainer.py`, the `self.accelerator` attribute is initialized as `None` in `__init__` and assigned a concrete `Accelerator` object later in `setup()`.

```python
class PeftTrainer:
    def __init__(self, ...):
        # ...
        self.accelerator: Accelerator | None = None
```

However, the rest of the codebase (and the `PeftTrainer` methods themselves) assume `self.accelerator` is always available after setup. This leads to numerous static analysis warnings (e.g., `Attribute 'device' may be missing on object of type 'Accelerator | None'`) and forces developers to either:
1.  Add runtime checks (`if self.accelerator:`) which clutter the code.
2.  Use `type: ignore` comments.
3.  Live with false positive warnings (current state).

## 2. Constraints

The `Accelerator` cannot be initialized in `__init__` due to **side effects**. The `prepare_accelerator()` function (called in `setup()`):
-   Initializes CUDA and distributed processes.
-   Sets environment variables.
-   Configures logging.
-   May spawn subprocesses (DDP).

Initializing it in `__init__` would make the `PeftTrainer` class difficult to test, inspect, or import without triggering global state changes or process forking. Therefore, the "construction" (init) must remain separate from "initialization" (setup).

## 3. Options Analysis

### Option 1: Add runtime assert before using accelerator
Explicitly check `if self.accelerator is None: raise ...` before usage.
*   **Pros:** Safe.
*   **Cons:** Extremely verbose if done at every call site.

### Option 2: Use type: ignore comments
*   **Pros:** Easy.
*   **Cons:** Hides genuine bugs; brittle refactoring.

### Option 3: Change to `Accelerator` with initializer (REJECTED)
Move logic to `__init__`.
*   **Status:** Rejected due to side effects described in constraints.

### Option 4: Keep as-is (Accelerator | None)
*   **Pros:** Accurate to the code state.
*   **Cons:** Noisy linting, poor IDE completion.

### Option 5: Property with Assertion (RECOMMENDED)
Use a private nullable backing field (`_accelerator`) and a public non-nullable property (`accelerator`) that asserts initialization.

```python
class PeftTrainer:
    def __init__(self, ...):
        self._accelerator: Accelerator | None = None

    @property
    def accelerator(self) -> Accelerator:
        if self._accelerator is None:
             raise RuntimeError("Accelerator not initialized. Call setup() first.")
        return self._accelerator
```

*   **Pros:**
    *   **Type Safety:** The public interface claims `Accelerator` (non-null), satisfying linters and type checkers.
    *   **Runtime Safety:** Prevents accidental usage before setup with a clear error message.
    *   **Clean Code:** Call sites (`self.accelerator.device`) remain clean without checks.
    *   **Architecture Preserved:** Keeps side effects in `setup()`.

## 4. Recommendation

Implement **Option 5**. This provides the best balance of type safety, developer experience, and architectural correctness.

**Plan:**
1.  Rename `self.accelerator` to `self._accelerator` in `PeftTrainer`.
2.  Add the `accelerator` property with the null check.
3.  Update `setup()` to assign to `self._accelerator`.
