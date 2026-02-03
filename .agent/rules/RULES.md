---
trigger: always_on
---

# sd-scripts Critical Rules

## Python Environment

Always use the venv Python for all commands:

```
d:\Projects\sd-scripts\venv\Scripts\python.exe
```

## Session Start

Read these files to understand current project state:

- `AGENTS.md` — Full agent instructions
- `DEVELOPMENT_GUIDE.md` — Repo guidelines
- `CHANGELOG.md` (top section) — Recent changes
- `ROADMAP.md` — TODOs and architecture

## Testing

```powershell
.\venv\Scripts\python.exe -m pytest tests/unit/ -q --tb=short
```

## Linting

```powershell
ruff check library/ scripts/
```

```powershell
uvx ty check
```

## Key Architecture

- Strategies in `library/strategies/{base,sd,sdxl}/`
- Models in `library/models/{sd,sdxl}/`
- Training scripts in `scripts/` are thin Hydra entry points
- Config dataclasses in `library/config/dataclasses/` are source of truth

## After Completing Work

1. Update `CHANGELOG.md`
2. Run unit tests
3. Update `ROADMAP.md` if applicable