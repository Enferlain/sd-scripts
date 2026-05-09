# Development Guide & Architectural Vision

## 1. Project Vision

**Goal:** Evolve `sd-scripts` into a robust, modular, and maintainable framework for training diffusion models.

**Philosophy:**

- **No backwards compatibility:** The focus is on refactoring, not to keep legacy working. Only keep for as long as references are needed for the active work.
- **Centralized Configuration:** The active training path uses a shared, type-safe config system built on **Hydra** and **Dataclasses**.
- **Separation of Concerns:** Break down multi-thousand line scripts into focused, reusable library modules.
- **Explicit over Implicit:** Functions should declare exactly what configuration they need. Orchestration-level code may use `cfg`; otherwise descriptive typed config parameter names are required.
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
  - ❌ **Bad (Legacy broad container pattern):**
    ```python
    def setup_optimizer(args, model):
        # args is a massive unrelated object
        lr = args.learning_rate
    ```
  - ✅ **Good (Modern):**
    ```python
    def setup_optimizer(optimizer_config: OptimizerConfig, model):
        # config contains ONLY optimizer settings
        lr = optimizer_config.learning_rates.base
    ```
- **Modularity:** Modules should be loosely coupled. An optimizer module shouldn't need to know about dataset details.

### C. Launchers (`train.py`, `scripts/`)

The active launcher surface should stay thin and orchestration-focused.

- **Responsibility:**
  1. Initialize Hydra (`@hydra.main`).
  2. Compose and validate the shared config.
  3. Build the active strategy and training mode.
  4. Hand off to the shared `Trainer`.
- **No Business Logic:** Complex logic (like "how to save a model" or "how to build a dataset") belongs in `library/`, not in the launcher.
- **Current shape:** `train.py` is the canonical launcher for active adapter / fine-tune runs. The current adapter family is PEFT under `adapter.peft`. Dedicated scripts that remain under `scripts/` are either transitional helpers or still-unmigrated paths such as textual inversion.

### D. Observability And Logging

- **Location:** `library/logging/`
- **Rule:** Training/runtime code owns *when* facts happen. Logging/observability code owns *how* human-facing console output, tracker metrics, startup summaries, reports, and resource-monitor integrations are rendered or routed.
- **Ownership buckets:**
  - `console.py` owns canonical user-facing lifecycle output
  - `metrics.py` owns tracker/metrics routing plus backend sink seams
  - `summaries.py` owns structured startup diagnostics and related summary models
  - `reports.py` owns benchmark/run report composition
  - `resource_monitor.py` remains its own runtime/resource subsystem
- **Console rule:** Use the repo-owned logging/console path for canonical training UX. Do not grow new long-term `accelerator.print(...)` formatting paths for startup or lifecycle summaries.
- **Naming rule:** Internal normalized component keys (`denoiser`, `text_encoder1`) are for shared code paths; human-facing diagnostics should prefer the public model-family labels carried by repo-owned provenance.

## 3. Coding Standards

- **Typing:** Use Python type hints (`typing`) everywhere. Prefer modern syntax (`X | None` over `Optional[X]`, `list[int]` over `List[int]`).
- **Imports:** Absolute imports preferred (e.g., `from library.training import optimizer`). Lazy imports should be avoided unless it brings proven performance benefits. Circular imports should be fixed, not worked around.
- **Use this rule:**
  - Top-level imports by default.
  - Lazy imports only for strict optional dependencies or proven startup/memory bottlenecks.
  - No lazy imports as a cycle workaround; fix module boundaries instead.
- **Docstrings:** Document the _config_ expected by functions.
- **Linting:** Use `ruff check .` and `ruff format .` before committing. Configuration is in `pyproject.toml`.
- **Type Checking:** Use `uv run ty check`. In WSL use `uv run ty check --python ./.venv-wsl/bin/python`. Configuration is in `pyproject.toml` under `[tool.ty]`.
- **Searching:** Use `rg` (ripgrep) for fast, gitignore-aware searching throughout the codebase.
- **Import Order:** isort is disabled; use `tools/fix_imports.py` for custom ordering if needed.
- **Naming Convention:** Folders use **plural** names (`adapters/`, `strategies/`, `models/`) where it makes sense. Files are usually named for the concern they own (`tokenization.py`, `checkpointing.py`, `trainer.py`) rather than by one universal singular/plural rule.
- **WSL + Windows filesystem note:** When this repo is accessed from WSL but stored on the Windows filesystem, Python and `pytest` can become noticeably slow and sometimes appear to hang silently. Prefer bounded commands such as `timeout 180 ./.venv-wsl/bin/python -m pytest ...` when testing from WSL, and if a command stays silent for too long, run it directly on the user side and report the result back rather than waiting indefinitely.

## 4. Config Design Principles

### A. Schema Enforces Validity

`RunConfig` is the shared root schema for the active launcher. Hydra validates composed YAML against the dataclass schema at load time, and presets layer mode/model defaults on top of the shared nested sections.

- **Nested ownership matters** - each concern should have one typed home (`optimizer`, `data`, `objective`, `output`, etc.).
- **Mode-specific options** belong in their dedicated sub-configs (`adapter`, `textual_inversion`) and are gated by launcher/validation rules.
- **Hard rule** - reusable library helpers should not accept broad root configs or generic `args`; they should take the narrowest typed config objects they actually use.
- **Naming rule** - orchestration-level code such as entrypoints, strategies, trainers, and phases may use `cfg`. Otherwise use descriptive typed names such as `optimizer_config`, `saving_config`, or `run_config`.

### B. Config Grouping

| Setting Type                                            | Location             |
| ------------------------------------------------------- | -------------------- |
| Training loop / runtime cadence                         | `TrainingConfig`     |
| Learning rates, optimizer, scheduler                    | `OptimizerConfig`    |
| Dataset source, captions, bucketing, caching, loader    | `DataConfig`         |
| Model family and checkpoint inputs                      | `ModelConfig`        |
| Objective path / prediction semantics                   | `ObjectiveConfig`    |
| Output, logging, sampling, metadata, publishing         | `OutputConfig`       |
| Precision, memory, attention, distributed, deepspeed    | `PerformanceConfig`  |
| Loss shaping / EDM2                                     | `LossConfig`         |
| Timestep sampling                                       | `TimestepConfig`     |
| Validation policy                                       | `ValidationConfig`   |
| Mode-specific adapter settings                          | `AdapterConfig`      |
| Mode-specific textual inversion settings                | `TextualInversionConfig` |

### C. Code Style in Orchestration Code

```python
# ✅ Good - orchestration-level code may use cfg
def train(cfg: RunConfig):
    if cfg.training.max_train_epochs:
        ...
    if cfg.performance.memory.gradient_checkpointing:
        ...

# ✅ Good - obvious short aliases are fine when reused heavily
def train(cfg: RunConfig):
    te_lr = cfg.optimizer.learning_rates.text_encoders
    ...

# ❌ Bad - don't alias every sub-config by default
def train(cfg: RunConfig):
    training_config = cfg.training
    perf_config = cfg.performance
```

- **`cfg` is fine in orchestration-level code** such as entrypoints, strategies, trainers, and phases
- **Prefer direct access for simple reads** - `cfg.training.X`, not `training_config.X`
- **Obvious short aliases are fine** when they materially improve readability and are reused a lot (`te_lr`, similar repeated values)
- **Do not create convenience aliases by default** just to shorten repeated `cfg.output.*` or `cfg.training.*` access

Outside orchestration-level code, use descriptive typed parameter names:

```python
# ✅ Good - helper uses descriptive typed config names
def setup_outputs(saving_config: SavingConfig, logging_config: LoggingConfig):
    ...

# ❌ Bad - helper uses cfg outside orchestration code
def setup_outputs(cfg: OutputConfig):
    ...
```

### D. Config Passing Patterns

Different layers of the codebase use different patterns for passing configuration:

| Layer                          | Pattern      | Example                                   | Reason                                        |
| ------------------------------ | ------------ | ----------------------------------------- | --------------------------------------------- |
| **Scripts** (`train()`)        | `cfg.*`      | `cfg.training.max_train_epochs`           | Has full typed root config                    |
| **Strategies**                 | `cfg.*`      | `cfg.data.caching.cache_latents`          | Receives full config, model-specific          |
| **Training phases / runners**  | `cfg.*`      | `cfg.validation.validate_every_n_steps`   | Shared orchestration layer coordinating flows |
| **Lower-level reusable helpers** | Typed params | `func(precision_config: PrecisionConfig)` | Modular, testable, explicit dependencies      |

**Key Rule: Outside orchestration layers, pass the smallest container that has what the function needs.**

```python
# ✅ Good - Pass exactly what's needed (can be at different depths)
def prepare_dtype(precision_config: PrecisionConfig, saving_config: SavingConfig):
    if precision_config.mixed_precision == "fp16":  # Direct access
        ...

# ❌ Bad - Passing more than needed
def prepare_dtype(performance_config: PerformanceConfig, ...):
    if performance_config.precision.mixed_precision == "fp16":  # Unnecessary nesting
        ...

# ❌ Bad - Reusable helper taking a broad root config
def prepare_dtype(cfg):  # What type? What does it need?
    if cfg.performance.precision.mixed_precision:
        ...
```

`cfg` is appropriate in strategies, trainers, phases, entrypoints, and similar orchestration layers because those layers legitimately coordinate multiple concerns. Lower-level helpers should stay explicit and accept the narrowest typed config objects they need, even if that makes signatures more verbose.

**Call sites should still show the full config path clearly:**

```python
# In launcher / orchestration code
def train(cfg: RunConfig):
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
    - If you see legacy broad-config passing, refactor it toward explicit typed config objects with ownership that matches the layer.

## 7. Testing Strategy

**Goal:** Ensure reliability by testing components in isolation.

- **Framework:** We use `pytest`.
- **Requirement:** Every new or refactored library module MUST have corresponding unit tests in `tests/`.
- **What to Test:**
  - **Unit Tests:** Verify focused library behavior such as optimizer/scheduler construction, config validation, sampling helpers, or strategy facets.
  - **Config Tests:** Verify that dataclasses compose correctly and validation catches invalid combinations.
  - **Integration / Smoke Tests:** Cover representative end-to-end launcher or trainer flows where wiring matters.
- **How to Run:**
  ```bash
  uv run pytest tests/unit/ -v
  uv run pytest tests/integration/ -v

  # WSL + Windows filesystem fallback
  timeout 180 ./.venv-wsl/bin/python -m pytest tests/unit/ -v --tb=short
  ```
