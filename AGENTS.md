# Agent Instructions for sd-scripts

This document provides essential context for AI agents working on this repository.

## Environment

**Python Virtual Environment:**

```powershell
uv run python
```

**WSL note:** when the repo lives on the Windows filesystem and tests are run from WSL, Python and `pytest` can look hung or stay silent for a long time even on small commands. Prefer bounded runs so a stalled command cleans itself up:

```powershell
timeout 180 ./.venv-wsl/bin/python -m pytest tests/unit/adapters/test_lora_module.py -q
timeout 60 ./.venv-wsl/bin/python -c "from train import train; print('OK')"
```

If a command appears stuck, ask the user to run it locally and report the result rather than repeatedly polling a silent WSL process.

## Review Policy

Significant code changes and every OpenSpec implementation milestone require `review-mcp` after the relevant quality gates and before handoff. Follow the global review workflow for invocation, waiting, findings, and follow-up review.

## Running Tests

```powershell
# All tests
uv run pytest tests/unit/ -v

# Specific test file
uv run pytest tests/unit/test_training_checkpointing.py -v

# With short traceback
uv run pytest tests/unit/ -v --tb=short

# WSL + Windows filesystem fallback (bounded run)
timeout 180 ./.venv-wsl/bin/python -m pytest tests/unit/ -v --tb=short
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

```plaintext
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

<!-- codebase-memory-mcp:start -->
## Codebase Memory

This repository uses Codebase Memory for structural code exploration.
Use the installed `codebase-memory` skill for architecture, symbol discovery,
call/dependency tracing, impact analysis, and related structural queries.
Prefer its graph workflow before broad grep/glob/file exploration.
<!-- codebase-memory-mcp:end -->

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:1105d646 -->
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

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/core-concepts/sync-concepts.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->

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

### OpenSpec Milestones And Handoffs

- Treat a numbered section in `openspec/changes/<name>/tasks.md` as the normal implementation milestone. Complete, verify, review, and hand off that section before starting the next one.
- Adjacent sections may be grouped only when they are genuinely small or cannot be reviewed meaningfully in isolation. State the grouped scope before implementation.
- For each milestone, run the relevant quality gates, request `review-mcp`, wait for its result, address actionable feedback, and then provide the handoff.
- Do not describe a milestone as reviewed or ready for handoff while its review is still running. If review explicitly fails or reaches its configured timeout, report that outcome rather than substituting an unreviewed success claim.

### Useful OpenSpec Commands

```bash
openspec list --json
openspec new change "<name>"
openspec status --change "<name>" --json
```

## Session Completion

**When ending a work session**, you should leave the work in a reviewable state and provide a clear handoff.

**RECOMMENDED WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Complete review** - For significant or OpenSpec milestone work, request `review-mcp`, wait for the result, address actionable findings, and rerun affected checks
5. **Hand off** - Summarize the completed scope, verification, review outcome, and any remaining risks or follow-up work

<!-- BEGIN BEADS CODEX SETUP: generated by bd setup codex -->
## Beads Issue Tracker

Use Beads (`bd`) for durable task tracking in repositories that include it. Use the `beads` skill at `.agents/skills/beads/SKILL.md` (project install) or `~/.agents/skills/beads/SKILL.md` (global install) for Beads workflow guidance, then use the `bd` CLI for issue operations.

### Quick Reference

```bash
bd ready                # Find available work
bd show <id>            # View issue details
bd update <id> --claim  # Claim work
bd close <id>           # Complete work
bd prime                # Refresh Beads context
```

### Rules

- Use `bd` for all task tracking; do not create markdown TODO lists.
- Run `bd prime` when Beads context is missing or stale. Codex 0.129.0+ can load Beads context automatically through native hooks; use `/hooks` to inspect or toggle them.
- Keep persistent project memory in Beads via `bd remember`; do not create ad hoc memory files.

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/core-concepts/sync-concepts.md for details and anti-patterns.
<!-- END BEADS CODEX SETUP -->

<!-- bd-doctor-divergence: ok -->
