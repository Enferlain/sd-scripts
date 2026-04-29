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
├── train.py               # Canonical Hydra launcher for active training runs
├── AGENTS.md              # This file - agent instructions
├── CHANGELOG.md           # Notable changes log
├── DEVELOPMENT_GUIDE.md   # Architectural principles and coding standards
├── ROADMAP.md             # Active follow-up work and future ideas
├── configs/               # Hydra YAML composition
│   ├── _defaults/         # Nested config-group defaults for shared schema sections
│   ├── presets/           # Representative runnable mode/model presets
│   ├── examples/          # Example overrides layered on presets
│   ├── tests/             # Test/smoke configs
│   └── benchmarks/        # Benchmark-oriented configs
├── docs/                  # User-facing documentation
├── library/               # Core library modules
│   ├── adapters/          # LoRA and network adapter implementations
│   ├── config/            # Shared dataclasses, schema registration, validation
│   ├── data/              # Data pipeline: manifests, caching, bucketing, loaders
│   ├── logging/           # Training logging, plotting, resource monitoring
│   ├── losses/            # Losses and loss-modifier helpers
│   ├── models/            # Model-family loaders, conversion, runtime helpers
│   ├── objectives/        # DDPM / RF objective ownership and runtime assembly
│   ├── optimization/      # Optimizers, schedulers, wrappers, factories
│   ├── performance/       # Memory and distributed/runtime helpers
│   ├── pipelines/         # Inference and sampling backends
│   ├── strategies/        # Shared contracts plus family-specific strategy facets
│   ├── timesteps/         # Timestep runtime and sampler logic
│   ├── training/          # Trainer, phases, modes, checkpointing, sampling
│   ├── utils/             # General utilities (hashing, device, torch, logging)
│   └── vendor/            # Third-party vendored code
├── scripts/               # Transitional helpers and non-root entrypoints
│   ├── sdxl_textual_inversion.py
│   └── _deprecated/       # Legacy launcher references kept during migration
├── tests/                 # Unit and integration tests, test assets
└── tools/                 # Standalone utilities (still use argparse)
```

## Module Naming Convention

### Models (`library/models/`)

Representative families:

| Folder  | Typical contents                                                                 |
| ------- | -------------------------------------------------------------------------------- |
| `sd/`   | SD1.5/2 loaders, UNet/VAE helpers, tokenizer/bootstrap utilities                |
| `sdxl/` | SDXL loaders, UNet/VAE/text-encoder helpers, control-net support                |
| `sd3/`  | SD3 family-specific loading and runtime helpers                                 |

Shared utilities live alongside the family folders when they are genuinely cross-family (`conversion_utils.py`, `runtime_utils.py`, etc.).

### Strategies (`library/strategies/`)

| Folder   | Contents                                                                  |
| -------- | ------------------------------------------------------------------------- |
| `base/`  | Required contracts and optional feature seams                              |
| `shared/`| Cross-family strategy-owned helpers (for example shared CLIP behavior)    |
| `sd/`    | SD family strategy facets and assembly                                     |
| `sdxl/`  | SDXL family strategy facets and assembly                                   |

Additional family folders may exist; use the above as the shape of the architecture, not an exhaustive catalog.

### Training (`library/training/`)

| Path                         | Purpose                                                  |
| ---------------------------- | -------------------------------------------------------- |
| `runners/trainer.py`         | Shared trainer orchestration                             |
| `modes/`                     | Training-mode plugins (`AdapterMode`, `FineTuneMode`)       |
| `phases/`                    | Shared training phases (caching, model prep, loop, etc.) |
| `checkpointing.py`           | Generic checkpoint utilities                             |
| `sample_generation.py`       | Shared sample generation orchestration                   |
| `_deprecated/`               | Legacy wrappers retained for unmigrated paths            |

## Configuration System

- **Dataclasses** in `library/config/dataclasses/` define the schema
- **YAML files** in `configs/` compose the user-facing config surface
- `configs/_defaults/` defines nested shared sections; `configs/presets/` selects a runnable mode/model baseline
- The active launcher is `train.py`, which uses `@hydra.main()` with an explicit `--config-name`
- Textual inversion still has a dedicated entrypoint while that runtime path is being migrated

**Key principle:** Reusable helpers should accept the narrowest specific typed config objects they need, and use descriptive typed parameter names. Orchestration-level code may use `cfg`:

```python
# ✅ Good - reusable helper takes the exact typed config it needs
def save_model(saving_config: SavingConfig, ...):

# ✅ Also good - orchestration-level code may use `cfg`
def train(cfg: RunConfig):
    ...

# ❌ Bad - generic container
def save_model(args, ...):
```

## Common Tasks

### Verifying imports work

```powershell
uv run python -c "from train import train; print('OK')"
```

### Running the active launcher (example)

```powershell
uv run python train.py --config-name=presets/sd_finetune
```

### Checking for import errors in a module

```powershell
uv run python -c "from library.training.runners.trainer import Trainer; print('OK')"
```

## Documentation Updates

When completing refactoring work:

1. Update `CHANGELOG.md` with a summary of changes
2. Update `ROADMAP.md` to mark completed items
3. Run relevant tests to verify changes

## Known Quirks

- `tools/` contains standalone utilities that still use `argparse` (external compatibility)
- `train.py` is the active launcher for PEFT and fine-tune flows; textual inversion still uses its dedicated script for now

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## OpenSpec Changes

This project also uses **OpenSpec** for spec-driven changes. Long-lived accepted
requirements live under `openspec/specs/`. Active change proposals, designs,
and task checklists live under `openspec/changes/` when a change is in flight.

### When to use OpenSpec

- Use OpenSpec for multi-step changes that benefit from explicit proposal,
  design, spec, and task artifacts, especially architecture work, behavior
  changes, and work that may span multiple sessions.
- OpenSpec is optional for small bug fixes, narrow refactors, doc-only edits,
  or other self-contained work that is already clear.

### Relationship to `bd`

- `bd` remains the source of truth for issue tracking, prioritization,
  dependencies, claiming, and completion status.
- OpenSpec complements `bd` by defining the change itself: what is changing,
  why, how it should work, and which implementation steps belong to it.
- OpenSpec artifacts such as `openspec/changes/<name>/tasks.md` are allowed as
  part of a spec-driven change. They do **not** replace `bd`, and should not
  be used as the repo-wide task tracker.
- Recommended flow for larger work:
  1. Create or claim the `bd` issue.
  2. Create or update the related OpenSpec change when spec/design/task
     artifacts would help.
  3. Implement against the OpenSpec artifacts.
  4. Archive the OpenSpec change when complete.
  5. Close the `bd` issue after implementation and verification are done.

### Useful OpenSpec Commands

```bash
openspec list --json
openspec new change "<name>"
openspec status --change "<name>" --json
```

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
