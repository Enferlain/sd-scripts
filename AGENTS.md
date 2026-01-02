# Agent Instructions for sd-scripts

This document provides essential context for AI agents working on this repository.

## Environment

**Python Virtual Environment:**

```powershell
d:\Projects\sd-scripts\venv\Scripts\python.exe
```

Always use the venv Python for running scripts, tests, and imports.

## Running Tests

```powershell
# All tests
d:\Projects\sd-scripts\venv\Scripts\python.exe -m pytest tests/unit/ -v

# Specific test file
d:\Projects\sd-scripts\venv\Scripts\python.exe -m pytest tests/unit/test_training_checkpointing.py -v

# With short traceback
d:\Projects\sd-scripts\venv\Scripts\python.exe -m pytest tests/unit/ -v --tb=short
```

**Test markers:** `unit`, `integration`, `training`, `config`, `slow`, `requires_gpu`

## Linting and Formatting

```powershell
# Check for lint errors
ruff check .

# Fix auto-fixable errors
ruff check . --fix

# Format code
ruff format .

# Check specific rules
ruff check . --select F401  # unused imports
```

**Note:** isort (I) is disabled. Use custom import sorter instead:

```powershell
python tools/fix_imports.py <file_path>
```

## Type Checking

```powershell
# Run ty type checker (configured in pyproject.toml)
uvx ty check

# Check specific directory
uvx ty check library/

# Check single file
uvx ty check library/training/checkpointing.py
```

**Note:** ty is configured to exclude legacy directories (`tools/`, `data_processing/`, `upscaling/`) and downgrade noisy rules to warnings for gradual adoption.

## Important Files

| File                   | Purpose                                       |
| ---------------------- | --------------------------------------------- |
| `ROADMAP.md`           | Tracks refactoring progress and future plans  |
| `CHANGELOG.md`         | Document all notable changes here             |
| `DEVELOPMENT_GUIDE.md` | Architectural principles and coding standards |
| `pyproject.toml`       | Project config: ruff, pytest, coverage        |

Always check DEVELOPMENT_GUIDE.md, DEVELOPMENT_GUIDE.md, and the top of CHANGELOG.md to refresh your memory of the latest work and the current state of the project.

## Project Structure

```
├── AGENTS.md              # This file - agent instructions
├── CHANGELOG.md           # Notable changes log (update when completing work)
├── DEVELOPMENT_GUIDE.md   # Architectural principles and coding standards
├── LICENSE.md             # Apache 2.0 license
├── README.md              # User-facing documentation
├── ROADMAP.md             # Refactoring progress and future plans
├── configs/               # YAML configuration files for Hydra
├── docs/                  # Additional documentation
├── library/               # Core library modules
│   ├── config/            # Hydra dataclasses (source of truth for config)
│   ├── data/              # Dataset loading, bucketing, caching
│   ├── losses/            # Loss functions and weighting utilities
│   ├── models/            # Model definitions (UNet, VAE, text encoders)
│   ├── networks/          # LoRA/network implementations
│   ├── performance/       # DeepSpeed and memory, various optimizations
│   ├── optimizers/        # Custom optimizer implementations
│   ├── pipelines/         # Inference pipelines (LPW, etc.)
│   ├── strategies/        # Model-specific training strategies
│   ├── timestep_samplers/ # Timestep sampling strategies
│   ├── training/          # Checkpointing, model prep, sample generation
│   ├── utils/             # General utilities (hashing, device, torch)
│   └── vendor/            # Third-party vendored code
├── scripts/               # Thin Hydra entry points (training scripts)
├── tests/                 # Unit and integration tests
└── tools/                 # Standalone utilities (still use argparse)
```

## Module Naming Convention

The codebase follows a **generic + model-specific** pattern:

| Generic (shared)       | SD1.5/2-specific          | SDXL-specific               |
| ---------------------- | ------------------------- | --------------------------- |
| `checkpointing.py`     | `sd_checkpointing.py`     | `sdxl_checkpointing.py`     |
| `model_prep.py`        | `sd_model_prep.py`        | `sdxl_model_prep.py`        |
| `sample_generation.py` | `sd_sample_generation.py` | `sdxl_sample_generation.py` |

- **Generic modules** contain utilities used by both SD and SDXL
- **`sd_*` modules** contain SD1.5/2-specific wrappers
- **`sdxl_*` modules** contain SDXL-specific wrappers

## Configuration System

- **Dataclasses** in `library/config/dataclasses/` define the schema
- **YAML files** in `configs/` are user-facing configuration
- Scripts use `@hydra.main()` decorator

**Key principle:** Functions should accept specific `Config` objects, not generic `args`:

```python
# ✅ Good
def save_model(saving_config: SavingConfig, ...):

# ❌ Bad (legacy pattern)
def save_model(args, ...):
```

## Common Tasks

### Verifying imports work

```powershell
d:\Projects\sd-scripts\venv\Scripts\python.exe -c "from library.training.checkpointing import model_hash; print('OK')"
```

### Running a specific script (example)

```powershell
d:\Projects\sd-scripts\venv\Scripts\python.exe scripts/sd_finetune.py --config-name=sd_finetune
```

### Checking for import errors in a module

```powershell
d:\Projects\sd-scripts\venv\Scripts\python.exe -c "from scripts.sd_finetune import train; print('OK')"
```

## Documentation Updates

When completing refactoring work:

1. Update `CHANGELOG.md` with a summary of changes
2. Update `ROADMAP.md` to mark completed items
3. Run relevant tests to verify changes

## Known Quirks

- Some optimizer tests have pre-existing failures (SGD betas issue) - unrelated to most work
- `tools/` contains standalone utilities that still use `argparse` (external compatibility)
- `v2` flag distinguishes SD2.x from SD1.x; both handled by `sd_*` modules (not SDXL)
