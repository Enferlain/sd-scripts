# Agent Instructions for sd-scripts

This document provides essential context for AI agents working on this repository.

## Environment

**Python Virtual Environment:**

```powershell
uv run python
```

Using `rg` (ripgrep) is highly recommended for searching the codebase. It is significantly faster than the default tools and respects `.gitignore` by default.

```powershell
rg "search_term"
```

## Standard Agent Tools

You have access to specialized tools via MCP (Model Context Protocol). Use them to enhance your workflow:

### Code Review (`review-mcp`)

**Tool:** `mcp_review_with_context`

Use this tool **after completing a significant chunk of work** (e.g., refactoring a module, implementing a feature) but **before** asking the user to verify. It acts as a senior engineer peer review.

- **When to use:** After implementing changes, before final user handoff.
- **How to use:** Provide the `diff_target` (usually 'HEAD') and a `task_description`.
- **Benefit:** Catches architectural issues, typos, and best-practice violations that linters miss.

### Web Search & Research

**Tools:** `search_web`, `webSearchPrime`, `browser_subagent`

- **`search_web` / `webSearchPrime`:** Use for quick fact-checking, library documentation lookup, or error message researching.
- **`browser_subagent`:** Use for deep dives, navigating complex documentation sets, or when you need to "see" a page or interact with a UI.

### Web Reading (`web-reader`)

**Tool:** `webReader`
Use to extract full content from a specific URL found during search (e.g., a specific documentation page).

## Running Tests

```powershell
# All tests
uv run pytest tests/unit/ -v

# Specific test file
uv run pytest tests/unit/test_training_checkpointing.py -v

# With short traceback
uv run pytest tests/unit/ -v --tb=short
```

**Test markers:** `unit`, `integration`, `training`, `config`, `slow`, `requires_gpu`

## Linting and Formatting

```powershell
# Check for lint errors
uv run ruff check .

# Fix auto-fixable errors
uv run ruff check . --fix

# Format code
uv run ruff format .

# Check specific rules
uv run ruff check . --select F401  # unused imports
```

**Note:** isort (I) is disabled. Use custom import sorter instead:

```powershell
uv run python tools/fix_imports.py <file_path>
```

## Type Checking

```powershell
# Run ty type checker (configured in pyproject.toml)
uv run ty check

# Check specific directory
uv run ty check library/

# Check single file
uv run ty check library/training/checkpointing.py
```

**Note:** ty is configured to exclude legacy directories (`tools/`, `data_processing/`, `upscaling/`) and downgrade noisy rules to warnings for gradual adoption.

**Environment differences:** In WSL needs to use `uv run ty check --python ./.venv-wsl/bin/python`


## Important Files

| File                   | Purpose                                       |
| ---------------------- | --------------------------------------------- |
| `AGENTS.md`            | Agent instructions                            |
| `ROADMAP.md`           | Tracks refactoring progress and future plans  |
| `CHANGELOG.md`         | Document all notable changes here             |
| `DEVELOPMENT_GUIDE.md` | Architectural principles and coding standards |
| `pyproject.toml`       | Project config: ruff, ty, pytest, coverage    |

Always check AGENTS.md, DEVELOPMENT_GUIDE.md, and the top of CHANGELOG.md to refresh your memory of the latest work and the current state of the project.

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
│   ├── adapters/          # LoRA and network adapter implementations
│   ├── config/            # Hydra dataclasses (source of truth for config)
│   ├── data/              # Data pipeline: scanning, caching, bucketing, dataloaders
│   ├── logging/           # Training logging and plotting utilities
│   ├── losses/            # Loss functions and weighting utilities
│   ├── models/            # Model definitions (per-model subfolders)
│   │   ├── sd/            # SD1.5/2 UNet, conversion, loader, VAE
│   │   └── sdxl/          # SDXL UNet, conversion, loader, text encoder, control net
│   ├── optimizers/        # Custom optimizer implementations
│   ├── performance/       # Memory optimization, gradient checkpointing
│   ├── pipelines/         # Inference pipelines (LPW, etc.)
│   ├── strategies/        # Model-specific strategies (per-model subfolders)
│   │   ├── base/          # Shared strategy contracts: TrainingStrategy, TokenizationStrategy, TextEncodingStrategy, CachingStrategy
│   │   ├── sd/            # SD1.5/2: training, tokenization, encoding, caching
│   │   └── sdxl/          # SDXL: training, tokenization, encoding, caching
│   ├── timesteps/         # Timestep sampling strategies
│   ├── training/          # Checkpointing, sample generation, trainer utilities
│   ├── utils/             # General utilities (hashing, device, torch)
│   └── vendor/            # Third-party vendored code
├── scripts/               # Thin Hydra entry points (training scripts)
├── tests/                 # Unit and integration tests, test assets
└── tools/                 # Standalone utilities (still use argparse)
```

## Module Naming Convention

### Models (`library/models/`)

| Folder  | Contents                                                                     |
| ------- | ---------------------------------------------------------------------------- |
| `sd/`   | `unet.py`, `conversion.py`, `loader.py`, `vae.py`                            |
| `sdxl/` | `unet.py`, `conversion.py`, `loader.py`, `text_encoder.py`, `control_net.py` |

Shared utilities: `conversion_utils.py`, `runtime_utils.py` (at root level)

### Strategies (`library/strategies/`)

| Folder  | Contents                                                            |
| ------- | ------------------------------------------------------------------- |
| `base/` | ABCs and shared strategy contracts in `contracts.py`                |
| `sd/`   | `SdTrainingStrategy`, `SdTokenizeStrategy`, etc.                    |
| `sdxl/` | `SdxlTrainingStrategy`, `SdxlTokenizeStrategy`, etc.                |

### Training (`library/training/`)

| File                   | Purpose                                     |
| ---------------------- | ------------------------------------------- |
| `checkpointing.py`     | Generic checkpoint utilities                |
| `sample_generation.py` | `sample_images_common()` for all models     |
| `trainer_utils.py`     | Training loop utilities                     |
| `sd_*.py`, `sdxl_*.py` | Legacy wrappers (kept for finetune scripts) |

## Configuration System

- **Dataclasses** in `library/config/dataclasses/` define the schema
- **YAML files** in `configs/` are user-facing configuration
- Scripts use `@hydra.main()` decorator

**Key principle:** Functions should accept specific `Config` objects, not generic `args`:

```python
# ✅ Good
def save_model(saving_config: SavingConfig, ...):

# ❌ Bad (_deprecated pattern)
def save_model(args, ...):
```

## Common Tasks

### Verifying imports work

```powershell
uv run python -c "from library.training.checkpointing import model_hash; print('OK')"
```

### Running a specific script (example)

```powershell
uv run python scripts/sd_finetune.py --config-name=sd_finetune
```

### Checking for import errors in a module

```powershell
uv run python -c "from scripts.sd_finetune import train; print('OK')"
```

## Documentation Updates

When completing refactoring work:

1. Update `CHANGELOG.md` with a summary of changes
2. Update `ROADMAP.md` to mark completed items
3. Run relevant tests to verify changes

## Known Quirks

- `tools/` contains standalone utilities that still use `argparse` (external compatibility)
- `v2` flag distinguishes SD2.x from SD1.x; both handled by `sd_*` modules (not SDXL)
