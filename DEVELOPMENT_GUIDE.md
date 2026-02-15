# Development Guide & Architectural Vision

## 1. Project Vision

**Goal:** Modernize `sd-scripts` from a collection of monolithic, ad-hoc scripts into a robust, modular, and maintainable library for training image generation models.

**Philosophy:**

- **No backwards compatibility:** The focus is on refactoring, not to keep legacy working. Only keep for as long as references are needed for the active work.
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
    def setup_optimizer(config_name: OptimizerConfig, model):
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

## 3. Coding Standards

- **Typing:** Use Python type hints (`typing`) everywhere. Prefer modern syntax (`X | None` over `Optional[X]`, `list[int]` over `List[int]`).
- **Imports:** Absolute imports preferred (e.g., `from library.training import optimizer`). Lazy imports should be avoided unless it brings proven performance benefits. Circular imports should be fixed, not worked around.
- **Docstrings:** Document the _config_ expected by functions.
- **No Argparse:** Do not import `argparse` in `library/` modules.
- **Linting:** Use `ruff check .` and `ruff format .` before committing. Configuration is in `pyproject.toml`.
- **Type Checking:** Use `uvx ty check` for fast type checking. Configuration is in `pyproject.toml` under `[tool.ty]`.
- **Searching:** Use `rg` (ripgrep) for fast, gitignore-aware searching throughout the codebase.
- **Import Order:** isort is disabled; use `tools/fix_imports.py` for custom ordering if needed.
- **Naming Convention:** Folders use **plural** names (`adapters/`, `strategies/`, `models/`), files use **singular** names (`lora.py`, `strategy_base.py`, `model_util.py`) unless it houses various utilities.

## 4. Config Design Principles

### A. Schema Enforces Validity

Each script's root config (e.g., `SDPeftConfig`, `SDXLFineTuneConfig`) defines what's valid for that mode. Hydra validates YAML against the dataclass schema at load time.

- **No `*SpecificConfig` pattern needed** - If `SDPeftConfig` doesn't have a `fine_tune:` field, Hydra errors if someone tries to use it.
- **Mode-specific options** belong in the root config or a shared sub-config, not nested specific configs.

### B. Config Grouping

| Setting Type                                                           | Location            |
| ---------------------------------------------------------------------- | ------------------- |
| Performance/memory (xformers, gradient_checkpointing, mixed_precision) | `PerformanceConfig` |
| Learning rates (optimizer LR, scheduler)                               | `OptimizerConfig`   |
| Model-specific (in the future)                                         | `ModelnameConfig`   |
| Adapter/LoRA settings                                                  | `PeftConfig`        |

### C. Code Style in Scripts and Strategies

```python
# ✅ Good - use cfg.X.Y directly
def train(cfg: SDPeftConfig):
    if cfg.training.max_train_epochs:
        ...
    if cfg.performance.memory.gradient_checkpointing:
        ...

# ❌ Bad - don't create aliases
def train(cfg: SDPeftConfig):
    training_config = cfg.training  # Unnecessary
    perf_config = cfg.performance   # Unnecessary
```

- **Use `cfg`** as the config variable name (not `config`)
- **Access sub-configs directly** - `cfg.training.X`, not `training_config.X`

### D. Config Passing Patterns

Different layers of the codebase use different patterns for passing configuration:

| Layer                   | Pattern      | Example                                   | Reason                                   |
| ----------------------- | ------------ | ----------------------------------------- | ---------------------------------------- |
| **Scripts** (`train()`) | `cfg.*`      | `cfg.training.max_train_epochs`           | Has full typed root config               |
| **Strategies**          | `cfg.*`      | `cfg.data.caching.cache_latents`          | Receives full config, model-specific     |
| **Library utilities**   | Typed params | `func(precision_config: PrecisionConfig)` | Modular, testable, explicit dependencies |

**Key Rule: Pass the smallest container that has what the function needs.**

```python
# ✅ Good - Pass exactly what's needed (can be at different depths)
def prepare_dtype(precision_config: PrecisionConfig, saving_config: SavingConfig):
    if precision_config.mixed_precision == "fp16":  # Direct access
        ...

# ❌ Bad - Passing more than needed
def prepare_dtype(performance_config: PerformanceConfig, ...):
    if performance_config.precision.mixed_precision == "fp16":  # Unnecessary nesting
        ...

# ❌ Bad - Opaque untyped cfg
def prepare_dtype(cfg):  # What type? What does it need?
    if cfg.performance.precision.mixed_precision:
        ...
```

**Call sites show the full config path:**

```python
# In script (uses cfg.* pattern)
def train(cfg: SDPeftConfig):
    # When calling library functions, pass the smallest container needed:
    prepare_dtype(cfg.performance.precision, cfg.output.saving)  # Different depths OK
    init_trackers(accelerator, cfg.output.logging, "my_project")
```

## 6. Workflow for Contributors

1.  **Adding a Feature:**
    - Add key/value to the relevant Dataclass (or create a new one).
    - Update the logic in `library/`.
    - Expose it in the main script.
2.  **Refactoring:**
    - If you see `args` being passed, refactor it to a detailed Config object.

## 7. Testing Strategy

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
