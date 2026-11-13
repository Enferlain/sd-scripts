# Project Instructions for AI Agents

## Subagents

- Use Sonnet Medium agents for reviewing sections.

## Environment

**Python Virtual Environment:**

```bash
uv run python
```

## Running Tests

```bash
# All tests
uv run pytest tests/unit/ -v

# Specific test file
uv run pytest tests/unit/iro/test_text.py -v

# With short traceback
uv run pytest tests/unit/ -v --tb=short
```

`uv run` uses `.venv-wsl` (set through `UV_PROJECT_ENVIRONMENT`). Today's
trainer does not run in this environment; it runs only from the pinned Windows
checkout (see `docs_design/models-strategy-trainer/baseline/README.md`).

**Test markers:** `unit`, `integration`, `training`, `config`, `slow`, `requires_gpu`

## Linting and Formatting

```bash
# Check for lint errors
uv run ruff check .

# Fix auto-fixable errors
uv run ruff check . --fix

# Format code
uv run ruff format .

# Check specific rules
uv run ruff check . --select F401  # unused imports
```

**Note:** isort (I) is disabled. Use custom import sorter (needs work so paused) instead:

```bash
uv run python tools/fix_imports.py <file_path>
```

## Type Checking

```bash
# Run ty type checker (configured in pyproject.toml)
uv run ty check

# Check specific directory
uv run ty check library/iro/

# Check single file
uv run ty check library/iro/text.py
```

## Documentation

- `CHANGELOG.md` should be updated with landed progress whenever it's handed off.
- `ROADMAP.md` is the larger-horizon board, not a progress log; never record completed work there.
- `DEVELOPMENT_GUIDE.md` holds coding standards and the workflow. The [rework design](openspec/changes/rework-model-strategy-trainer/design.md) governs new code; guide sections marked *Today's system* describe code being replaced.

## Building the New System

The rework ([design](openspec/changes/rework-model-strategy-trainer/design.md), D21) builds the new system new, in place. Today's code is read, and may be run for comparison, only to check D21's coverage rows; it never decides how new code is shaped.

- D21's module map is the boundary. Code it does not list is not part of the new system, and nothing in the new system depends on it.
- New code never imports the code it replaces, research code under `docs_design/`, or a `_deprecated/` folder. Once an area's coverage rows are checked, its replaced code moves to its package's `_deprecated/` folder.
- Today's code is never edited to follow the new rules. When new work breaks one of today's tests, that test moves to `tests/_deprecated/` instead of being fixed.
- Dependencies point down:
  - Launch may import everything below it. `library/config` belongs to launch, and nothing below launch imports it.
  - Strategies may import training's words, the run state, the IR system, the libraries and base.
  - Training may import the run state, the IR system, the libraries, observability and base, never strategies or config.
  - Libraries may import the run state, the IR system, observability fact types, base and lower libraries, never training, strategies, launch or config. From the bottom: artifacts, performance, models, conditioning, losses, timesteps, objectives, data, optimization, adapters, pipelines, evaluation.
  - The run state (`library/runstate`) may import only the IR system, observability fact types and base, never the libraries or Torch.
  - The IR system (`library/iro`) may import only xDSL, observability fact types and base. Torch loads only in its device adapters.
  - Observability imports only base, and base imports nothing in `library`.
- `tests/unit/test_dependency_rules.py` checks these rules over the new system's modules. A section that adds a package adds it to `NEW_SYSTEM` there.
- Tests build accepted runs with the test-only builder in `tests/unit/iro/accepted_runs.py`. No production module imports it.

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
