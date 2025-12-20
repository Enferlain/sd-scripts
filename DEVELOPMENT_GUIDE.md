# Development Guide & Architectural Vision

## 1. Project Vision

**Goal:** Modernize `sd-scripts` from a collection of monolithic, ad-hoc scripts into a robust, modular, and maintainable library for training image generation models.

**Philosophy:**

- **Centralized Configuration:** Move from per-script `argparse` definitions to a global, type-safe system using **Hydra** and **Dataclasses**.
- **Separation of Concerns:** Break down multi-thousand line scripts into focused, reusable library modules.
- **Explicit over Implicit:** Functions should declare exactly what configuration they need. Avoid passing opaque `args` objects or global state.
- **Production Quality:** Adopt best practices from high-end repositories (typing, structured configs, potential shipping as a package).

## 2. Architectural Principles

### A. Configuration (The Source of Truth)

- **Location:** `library/config/dataclasses/`
- **Format:** Python `dataclasses` are the schema. YAML files in `configs/` are the user interface.
- **Rule:** Every configurable parameter must belong to a strictly typed dataclass.
- **Rule:** Do not duplicate config definitions across scripts. Import them from the central library.

### B. The Library (`library/`)

This is the core of the project. It should contain the "building blocks" of training.

- **Design Pattern:** Config Injection.
  - ❌ **Bad (Legacy):**
    ```python
    def setup_optimizer(args, model):
        # args is a massive unrelated object
        lr = args.learning_rate
    ```
  - ✅ **Good (Modern):**
    ```python
    def setup_optimizer(config: OptimizerConfig, model):
        # config contains ONLY optimizer settings
        lr = config.learning_rate
    ```
- **Modularity:** Modules should be loosely coupled. An optimizer module shouldn't need to know about dataset details.

### C. Scripts (`scripts/`)

Scripts are thin entry points (Consumers) of the library.

- **Responsibility:**
  1.  Initialize Hydra (`@hydra.main`).
  2.  Instantiate configuration objects.
  3.  Orchestrate calls to library functions.
  4.  Run the main loop.
- **No Business Logic:** Complex logic (like "how to save a model" or "how to build a dataset") belongs in `library/`, not in the script script.

## 3. Migration Strategy

We are transitioning from a "Script-First" to a "Library-First" architecture.

1.  **Refactor Library Modules (Bottom-Up):**

    - Identify a "concern" (e.g., Checkpointing, Model Loading).
    - Refactor its functions to accept `Config` objects instead of `args`.
    - _Note:_ This might temporarily break legacy scripts. We accept this cost or handle it via temporary wrappers, but the _new_ code must be pure.

2.  **Migrate Scripts (Top-Down):**

    - Once the underlying library modules are ready, rewrite the main script (e.g., `train_network.py`) to use Hydra.
    - Remove `argparse` entirely from the script.

3.  **Eliminate Adapters:**
    - `ArgsAdapter` is a temporary bridge. It is **technical debt**.
    - The goal is `0` usage of `ArgsAdapter`.

## 4. Coding Standards

- **Typing:** Use Python type hints (`typing`) everywhere.
- **Imports:** Absolute imports preferred (e.g., `from library.training import optimizer`).
- **Docstrings:** Document the _config_ expected by functions.
- **No Argparse:** Do not import `argparse` in `library/` modules.

## 5. Workflow for Contributors

1.  **Adding a Feature:**
    - Add key/value to the relevant Dataclass (or create a new one).
    - Update the logic in `library/`.
    - Expose it in the main script.
2.  **Refactoring:**
    - If you see `args` being passed, refactor it to a detailed Config object.

## 6. Testing Strategy

**Goal:** Ensure reliability by testing components in isolation.

- **Framework:** We use `pytest`.
- **Requirement:** Every new or refactored library module MUST have corresponding unit tests in `tests/`.
- **What to Test:**
  - **Unit Tests:** Verify that `library/training/optimizer.py` correctly creates an optimizer given a specific `OptimizerConfig`.
  - **Config Tests:** Verify that `dataclasses` correctly default values and handle missing keys.
- **How to Run:**
  ```bash
  pytest tests/
  ```
