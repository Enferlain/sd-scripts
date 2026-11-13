# Guardrails

Shared operating rules for every skill in this repo. Follow them **unless the
user explicitly says otherwise.** Individual skills link here rather than
repeating the list.

## Prior work

Before starting a port, migration, or other substantial change, find the
project's **upstream** and check whether the work already exists or has been
attempted — don't reinvent a branch someone already wrote.

- **Locate the upstream repo.** `git remote -v`, plus the `[project.urls]` /
  `Home-page` / `Source` fields in `pyproject.toml` / `setup.cfg` / `setup.py`.
  If the checkout is a fork, find the parent (`gh repo view --json parent`); the
  real discussion is usually upstream.
- **Search issues *and* pull requests — open, merged, *and* closed** — for the
  change you're about to make (see each skill's step 1 for the terms that fit it).
  `gh issue list --search "<terms>"`, `gh pr list --search "<terms>" --state all`,
  or the forge's web search. Do the same on GitLab / other forges.
- **Read what you find.** An **open** PR you can build on or continue; a
  **merged** one that already did it (stop, or port only the remaining delta); a
  **closed/abandoned** one whose review comments tell you the blockers and why it
  stalled — often the most valuable of the three.
- **Report before proceeding.** Link the relevant issues/PRs and say whether
  you're continuing existing work, redoing it, or starting fresh.

## Environment

- Work in a **project-local `.venv`**, preferably created with **`uv venv`**.
- Prefer **uv** for everything: `uv venv`, `uv pip …`, `uv build`, `uvx`. Run
  one-off tools with **`uvx`** (not `pipx`).
- **Never** install into global or user site-packages — no `sudo pip`, no
  `pip install --user`, no touching the system interpreter.
- If uv isn't available, still use a `.venv` (`python -m venv`); never install
  globally.
- **Favor `uv build`** over `python -m build`, and `uv pip …` over `pip …`.
  `python -m build` / `pip …` are the equivalent fallbacks when uv isn't
  available — use them 1:1, inside the venv.

## Changes

- **Get existing code under VCS with a clean tree before you modify it.** If a
  project holds code that isn't committed (not a git repo, or a dirty working tree),
  stop and **ask the user how to proceed** — to commit/stash outstanding changes, or
  to initialize a repo and commit first — rather than doing it on their behalf. This
  keeps your edits a reviewable, revertible diff and ensures nothing pre-existing is
  lost. Never modify code you can't diff against a committed baseline.
- **Don't delete existing content** (code, docs, comments, config) unless the
  task requires it and the user is aware.
- **Don't commit or push without the user's explicit OK**, and **never commit
  directly to `main`** — branch first.
- Leave changes in the working tree for review rather than finalizing them.

## Footprint

- Be mindful of **disk and CPU/memory**. Some dependencies are large — a torch +
  CUDA install runs to several GB.
- **Ask the user for consent before installing heavy packages or starting a heavy
  compile** (e.g. torch/CUDA, a full C++/CUDA build, cibuildwheel matrices).
  Estimate the cost first (download size, disk, build time) and let them decide.
- Install only what the task needs; don't pull heavy extras or duplicate wheels.
- Reuse an existing venv/cache when it's already set up rather than recreating it.
- **Prefer venv-local wheels over a system toolchain.** For CUDA builds you don't
  need a system toolkit install: `nvcc` and the CUDA libraries ship as PyPI wheels
  (`nvidia-cuda-nvcc-cu12`, `nvidia-cuda-runtime-cu12`, `nvidia-cuda-cccl-cu12`,
  `nvidia-cuda-nvrtc-cu12`, `nvidia-cublas-cu12`, … and a `-cu11` line — the same
  ones torch's CUDA wheels pull in). `uv pip install nvidia-cuda-nvcc-cu12` puts
  nvcc in the venv with no sudo. Match the CUDA major/minor to the torch you build
  against; still ask before the download (hundreds of MB).
- **CPU-only torch is the lightweight default for testing and much porting.**
  `uv pip install torch --index-url https://download.pytorch.org/whl/cpu` is a few
  hundred MB (vs multi-GB for the CUDA build) and needs no nvcc/GPU. Use it for
  everything device-independent — header compiles, the CPU build, test suites,
  cross-version checks — and only pull the CUDA torch + nvcc wheels when you must
  build `.cu` device code or run GPU tests.

## Style

- **Match the project's existing conventions** for formatting, naming, comments,
  and docstrings — read the surrounding code first.
- Keep comments and docstrings **terse and to the point**. Explain *why*, not
  *what*; don't restate what the code already makes obvious.
- **Never add unnecessary comments.** No comment that restates the line it sits
  on, narrates the edit ("added for migration", "changed default"), or leaves
  TODO/placeholder chatter. This applies to generated config too — a
  `pyproject.toml`, `CMakeLists.txt`, or `meson.build` should read as a clean
  hand-written file, not an annotated diff. Add a comment only when it records a
  non-obvious *why* the reader would otherwise miss.
- **Stay DRY.** When a port repeats the same edit across many files — a swap
  wrapper, a stream/device helper, a check/dispatch macro — factor it into **one
  shared definition** instead of copy-pasting. For C/C++, put common helpers in a
  **shared header** the translation units include; don't inline the same snippet
  into every `.cpp`/`.cu`. The ported code deserves the same DRY discipline as the
  original.
