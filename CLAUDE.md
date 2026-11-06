# Project Instructions for AI Agents

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
uv run pytest tests/unit/test_training_checkpointing.py -v

# With short traceback
uv run pytest tests/unit/ -v --tb=short

# WSL + Windows filesystem fallback (bounded run)
timeout 180 ./.venv-wsl/bin/python -m pytest tests/unit/ -v --tb=short
```

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
uv run ty check library/

# Check single file
uv run ty check library/training/checkpointing.py
```

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
