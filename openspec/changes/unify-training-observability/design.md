## Context

The repo already has meaningful observability-related pieces, but they are organized by history more than by an explicit architecture:

- `library/logging/step_logging.py` owns tracker emission and tracker initialization.
- `library/training/trainer_utils.py` owns startup diagnostics formatting.
- `library/logging/run_report.py` owns end-of-run markdown/JSON reporting.
- `library/logging/resource_monitor.py` owns runtime resource sampling, phase summaries, and JSONL event output.
- trainer, modes, and phases still emit human-facing console output directly through a mix of `logger.*` and `accelerator.print`.

This is workable today, but it creates three problems:

1. the same underlying training facts are reformatted separately for console, reports, and tracker output
2. new observability work tends to get attached to whatever nearby file already prints something
3. a future repo-owned dashboard/experiment system would have to reverse-engineer training-layer behavior instead of consuming stable structured signals

The adapter diagnostics discussion exposed the same pressure point in a small way: the console startup block is only a few lines, but the missing abstraction is not "more pretty printing", it is a stable source of structured observability data that multiple sinks can share.

Today the effective ownership split is:

- human console lifecycle output lives across trainer, phases, and helper modules
- tracker metrics live in `library/logging/step_logging.py`
- startup diagnostics live in `library/training/trainer_utils.py`
- report payload building lives in `library/logging/run_report.py`
- resource observability lives in `library/logging/resource_monitor.py`

That means the problem is not just code location; it is that production of facts and ownership of sinks/formatting are still partially mixed.

## Goals / Non-Goals

**Goals:**

- Define repo-owned observability concepts for training-emitted signals.
- Separate signal production from formatting and sink/back-end routing.
- Keep startup diagnostics, tracker metrics, reports, and future dashboard sinks aligned on shared data shapes where practical.
- Make adapter diagnostics derive from repo-owned adapter provenance rather than method-specific display logic.
- Preserve the current visible categories of output while improving ownership boundaries.

**Non-Goals:**

- Replacing the current progress bar implementation in this change.
- Replacing Accelerate tracker integration immediately.
- Folding resource monitoring into generic logging utilities until the contracts are clear.
- Building the future dashboard UI or its Svelte frontend in this change.
- Reorganizing every existing `logger.info` call before the new observability seams prove useful.

## Decisions

### 1. Treat observability as a first-class architecture concern, not just a package move

We should not "solve" the problem by only moving files under `library/logging/`. The real need is a repo-owned contract between training/runtime code that knows facts and observability code that knows how to emit, persist, and present them.

Why this direction:

- It creates stable inputs for future custom dashboard/back-end work.
- It avoids growing more ad hoc formatting logic in trainer/mode/phase files.
- It lets console, tracker, and report outputs share structured facts without forcing one sink to define the others.

Alternative considered:

- Keep current ownership and only patch the current adapter breakdown.
  - Rejected because it solves one symptom without giving future observability work a better home.

### 1A. The target package direction is explicit

The long-term package direction should be a real observability/logging package with distinct sub-concerns, not a grab-bag of unrelated output helpers:

```text
library/logging/
  console.py        # rank-aware human-facing lifecycle output
  metrics.py        # per-step scalar emission + sink routing
  summaries.py      # startup diagnostics rows, optimizer summaries, trainability breakdowns
  reports.py        # end-of-run markdown/json report generation
  resource_monitor.py
  sinks/
    accelerate.py   # Accelerate tracker backend adapter
    local_jsonl.py  # optional local sink(s) as needed
```

The exact filenames can vary, but these concerns should exist as distinct ownership buckets.

Why this direction:

- It matches the actual categories of emitted training signals.
- It gives future custom dashboard/back-end work a predictable home.
- It keeps "move every print into logging" from becoming the implicit design.

### 1B. Module names should follow concern-oriented repo nouns

For the first-pass observability package layout, the module names should be:

- `console.py`
- `metrics.py`
- `summaries.py`
- `reports.py`
- `resource_monitor.py`

Why this direction:

- These names describe ownership buckets rather than specific one-off helpers.
- `metrics`, `summaries`, and `reports` are naturally plural because they own families of emitted values/builders rather than one singular object.
- `console` works best as singular because it represents one output surface, not a collection of unrelated console implementations.
- `resource_monitor` should keep its established name because it is already a concrete subsystem, not a newly named concern bucket.

This means we should prefer:

- plural module names for categories that own multiple emitted record types or builders
- singular module names for a single output surface or already-established subsystem

The purpose of this rule is to keep the package readable and prevent churn from arbitrary singular/plural mixing once implementation starts.

### 1C. Preserve the current logging stack and its source-rich console output

This change should keep the repo on stdlib `logging` with Rich-backed console handling rather than replacing the logging stack with a different framework. The timestamped, level-aware, source-rich output is already valuable and should remain part of the canonical console experience.

Why this direction:

- The current runtime logger output already provides good operator-facing context such as timestamp, level, and exact `file:line`.
- The architecture problem is ownership and consistency, not that the repo lacks a capable logging library.
- Replacing the stack would create churn that is unrelated to the observability-boundary work.

This means:

- no `loguru`-style logging-stack migration is part of this change
- Rich-backed console handling remains the expected default console path
- human-facing repo output should preserve source-aware logging affordances where they already exist and make sense

### 1D. Early config/setup logs and runtime logs are one console surface

Pre-trainer config/setup logging and runtime/trainer logging should be treated as one user-facing console surface even though they are initialized in different phases today.

Why this direction:

- Users experience early config validation, startup, and runtime messages as one continuous training console UX.
- The current split between Hydra-era formatting, repo runtime logging, and direct `accelerator.print(...)` is an implementation artifact, not a desirable product boundary.
- Unifying the surface does not require one identical call path for every producer, but it does require compatible presentation and shared ownership rules.

This means:

- repo-owned logging setup should become the canonical console style as early as practical in the launcher flow
- early config/setup warnings are in scope for console-style alignment with runtime logs
- the observability change should avoid creating a separate visual language for pre-trainer warnings versus runtime messages

### 2. Keep resource monitoring as a distinct observability sub-concern

`resource_monitor` should remain distinct from startup diagnostics and generic tracker metrics. It is about runtime memory/time sampling, phase summaries, JSONL events, and benchmark/resource reporting.

Why this direction:

- Resource sampling has different performance and configuration constraints from ordinary logging.
- The repo already has a meaningful config-driven monitor system.
- Future observability architecture should consume its outputs through stable interfaces, not collapse it into generic print helpers.

Alternative considered:

- Merge resource monitoring into the same immediate implementation slice as startup diagnostics.
  - Rejected because the ownership pressure is different and would inflate the first change unnecessarily.

### 3. Use structured observability rows/events as the main shared seam

For startup diagnostics and similar summaries, the repo should produce structured rows/events first, then let observability-owned code render them for console, reports, or future custom sinks.

Why this direction:

- It keeps presentation concerns out of trainer/mode logic.
- It makes report reuse natural instead of duplicative.
- It gives future dashboard systems structured data instead of forcing them to scrape console-oriented strings.

Alternative considered:

- Continue passing only raw modules/components into each sink and letting each sink derive its own summary.
  - Rejected because it duplicates logic and drifts naming/output behavior over time.

### 3A. Introduce a minimal repo-owned observer/sink seam

The observability package should expose a small repo-owned seam between training producers and concrete sinks/backends. The point is not to hide all behavior behind a god-object, but to stabilize how metrics, summaries, console messages, and artifacts are routed. `TrainingObserver` is the preferred name for this seam because it covers more than telemetry-style scalar emission.

Representative shape:

```python
class TrainingObserver(Protocol):
    def log_metrics(self, step: int, metrics: dict[str, float]) -> None: ...
    def log_console(self, message: str, *, level: str = "info") -> None: ...
    def log_startup_summary(self, summary: TrainingStartupSummary) -> None: ...
    def log_artifact(self, path: str, *, kind: str) -> None: ...
```

Representative structured models:

```python
@dataclass(frozen=True)
class DiagnosticRow:
    label: str
    component_key: str
    modules_trainable: int
    modules_total: int
    params_trainable: int
    params_total: int


@dataclass(frozen=True)
class TrainingStartupSummary:
    runtime_rows: list[tuple[str, str]]
    component_rows: list[DiagnosticRow]
    optimizer_rows: list[tuple[str, int, float | None]]
```

Why this direction:

- It gives the repo a stable backend boundary for current Accelerate trackers and future repo-owned sinks.
- It makes console/report/dashboard consumers read the same structured data.
- It keeps training phases/modes responsible for deciding when facts happen, without making them own sink routing.

Alternative considered:

- Let sink boundaries emerge implicitly from existing helper signatures.
  - Rejected because that continues the current fragmentation and does not give future custom backends a clear contract.

### 3B. Keep the backend sink narrower than the repo-facing observer

`TrainingObserver` is the repo-facing seam for observability coordination, but the swappable backend sink should stay narrower than that. The first-pass backend sink only needs to cover run lifecycle and metric persistence, with artifact logging as an optional extension when a concrete backend needs it.

Representative first-pass backend sink:

```python
class MetricsSink(Protocol):
    def start_run(self, run_name: str, config: dict[str, object]) -> None: ...
    def log_metrics(
        self,
        metrics: dict[str, float],
        *,
        step: int,
        epoch: int | None = None,
    ) -> None: ...
    def finish_run(self) -> None: ...
```

Optional extension:

```python
    def log_artifact(self, path: str, *, kind: str) -> None: ...
```

This means:

- backend sinks are responsible for adapting repo-owned metric/run facts to Accelerate-backed trackers or future repo-owned experiment backends
- backend sinks are not required to own console output, startup-summary rendering, or resource-monitor runtime behavior
- `TrainingObserver` may coordinate multiple observability concerns, but concrete backend swap points should remain as small as possible

Why this direction:

- The current tracker path is already thin: repo-owned code shapes configs and metrics, while backend-specific behavior mainly concerns initialization and how step/epoch data is passed to trackers.
- It keeps the backend interface stable without forcing every observability concern through the same sink abstraction prematurely.
- It prevents the sink layer from becoming a second oversized orchestration seam.

### 4. Adapter diagnostics should derive from repo-owned adapter provenance

Adapter diagnostics should use repo-owned adapter provenance such as resolved targets and trainable refs, rather than requiring each adapter method/runtime to invent its own reporting behavior before the repo can show a useful breakdown.

Why this direction:

- The adapter layer already owns component/public-label/internal-key provenance.
- It prevents shared training observability code from branching on method-local runtime shapes.
- It supports future adapter breadth better than per-method custom console logic.

Alternative considered:

- Require every adapter runtime to implement bespoke diagnostics methods first.
  - Rejected as the default path because it adds per-method duplication and delays useful observability improvements.

### 4A. User-facing diagnostics should prefer public component names, not normalized internal keys

Normalized component names such as `denoiser` and `text_encoder1` exist so training, optimization, and adapter code can share stable internal keys without hardcoding model-family-specific names everywhere. They are not the intended default labels for human-facing diagnostics.

For user-facing observability output:

- public component labels from repo-owned provenance should be the default display labels
- normalized internal keys should remain available in the structured summary model for grouping, joins, and sink/backend logic
- when both labels are useful, the display may include both, but the internal normalized key should not replace the public label as the default human-facing name

Why this direction:

- It matches the original purpose of normalized keys as an internal compatibility mechanism.
- It preserves model-family-specific vocabulary in the places users actually read.
- It still allows shared observability code to stay generic internally.

### 5. Prefer one coherent observability refactor over artificial staging

This change should land as one coherent observability refactor for the affected concerns instead of being decomposed into temporary mini-phases just because the work touches multiple files. The main implementation focus should still be startup diagnostics, tracker/report alignment, and adapter provenance-driven diagnostics, but those should move directly to their intended ownership boundaries rather than through transitional stopgap layers.

Why this direction:

- The current observability code is uneven, not uniformly broken: `step_logging.py`, `run_report.py`, and `resource_monitor.py` already own meaningful sub-concerns and should be realigned around shared contracts instead of being rewritten piecemeal.
- The current pressure is concentrated in a few high-value seams, especially startup diagnostics, adapter reporting, and report reuse, rather than every lifecycle log call being equally urgent.
- Artificial staging would create duplicate paths and temporary abstractions for concerns whose intended long-term ownership is already clear.
- A coherent refactor better matches the repo's preference for explicit final ownership boundaries over transitional compatibility-era structure.

Alternative considered:

- Large one-shot package reorganization that rewrites all lifecycle logging at the same time.
  - Rejected because it would mix clear observability-boundary work with broad wording/location churn and make it harder to evaluate whether the new contracts themselves are good.

### 6. Keep training-side ownership explicit

The new architecture should follow this boundary:

```text
training code emits structured events / metrics / summaries
logging package owns sinks, formatting, buffering, routing, persistence
```

And explicitly not:

```text
training code stops knowing anything and logging package becomes a giant orchestration owner
```

That means:

- phases still decide when events happen
- modes still decide what trainable things exist
- strategies still decide family-specific facts
- logging owns how those facts are formatted, routed, and persisted

Why this direction:

- It matches the repo's separation-of-concerns rules.
- It avoids turning observability into a hidden orchestration layer.
- It gives future dashboard/backend work a stable consumer side without weakening production ownership elsewhere.

### 6A. Observability-owned console output should cover canonical training UX, not every local log

This change should centralize the console output that forms the repo's canonical training UX, while leaving incidental subsystem-local logging in place unless it needs shared rank-aware or progress-safe behavior.

Observability-owned console helpers should cover:

- startup summaries and structured training diagnostics
- canonical lifecycle announcements that appear in most runs
- progress-bar-adjacent messages that need rank-aware and progress-safe emission
- other repeated human-facing summaries that are likely to matter to more than one sink

Subsystem-local logging may remain local for:

- helper- or phase-specific implementation details
- one-off informational messages that are not part of the run's public narrative
- warnings/errors whose ownership is naturally local to one subsystem
- third-party or library-originated logs

Why this direction:

- It centralizes the output users actually experience as the training run's public narrative.
- It avoids turning observability helpers into a required wrapper for every local `logger.info`.
- It keeps the boundary practical enough that developers will keep using it consistently.

### 6B. Preferred console presentation style is sectioned startup blocks plus tagged lifecycle lines

The preferred human-facing console presentation for repo-owned training output is:

- sectioned startup blocks for dense run configuration and diagnostics
- short tagged lifecycle lines for standalone status messages

Representative startup block shape:

```text
training run
  mode: AdapterMode
  strategy: SdxlTrainingStrategy
  precision: bf16

dataset
  train images: 550
  validation images: 0
  reg images: 0

schedule
  batches/epoch: 275
  epochs: 1
  batch/device: 2
  grad accum: 1
  effective batch: 2
  total steps: 50

components
  unet   | modules 1,588/1,588 | params 98,825,472/98,825,472   | 100.0%
  clip_l | modules     0/  122 | params         0/3,145,728     | frozen
  clip_g | modules     0/  104 | params         0/2,621,440     | frozen
  total  | modules 1,588/1,814 | params 98,825,472/104,592,640 | 94.5%

optimizer
  AdamW8bit
  - unet: lr=1e-05, params=98,825,472
```

Representative lifecycle line shape:

```text
[startup] training initialized
[epoch] prepared epoch 1: 281 batches, 550 images
[resource] startup parameter memory estimates
[checkpoint] saving checkpoint at step 500
```

Why this direction:

- It keeps dense startup information readable without forcing every line through plain logger event formatting.
- It gives standalone lifecycle messages a compact, consistent identity that works well around progress bars.
- It matches the repo's preference for terminal-friendly plain text over overly decorative console widgets as the default presentation.

This means:

- large startup summaries should render as structured blocks owned by `summaries.py` / `console.py`, not as ad hoc `accelerator.print(...)` calls
- simple event/status lines should remain logger-like messages routed through the observability console layer
- richer Rich tables/panels remain optional implementation details, not the required default presentation

### 6C. The first pass should move directly into the target observability modules

The first implementation pass should use the target `library/logging/console.py`, `metrics.py`, `summaries.py`, and `reports.py` modules directly rather than introducing intermediate stopgap modules.

Representative first-pass ownership moves:

- startup diagnostics rendering and formatting from `library/training/trainer_utils.py` to `library/logging/summaries.py`
- canonical main-process console helpers from scattered trainer/helper usage into `library/logging/console.py`
- tracker/run metric routing from `library/logging/step_logging.py` into `library/logging/metrics.py` with backend adapters under `library/logging/sinks/`
- benchmark/run report composition from `library/logging/run_report.py` into `library/logging/reports.py`
- `library/logging/resource_monitor.py` remains in place as its own subsystem and integrates through the new observability seams where needed

Why this direction:

- The target module names already reflect the intended ownership buckets, so adding temporary layers would only create cleanup work.
- It makes the refactor easier to review against the final architecture instead of against a transitional structure.
- It narrows implementation ambiguity by saying where the first meaningful extractions should land.

## Risks / Trade-offs

- **[Risk]** The change becomes a package move without real architectural improvement.  
  **Mitigation:** Define structured observability contracts and explicit producer/sink boundaries before moving much code.

- **[Risk]** The scope drifts into rewriting all lifecycle logging instead of fixing the highest-value ownership boundaries.  
  **Mitigation:** Keep the refactor coherent, but limit it to the observability seams whose final ownership is already clear: startup diagnostics, tracker/report alignment, adapter provenance-driven diagnostics, and explicit resource-monitor boundaries.

- **[Risk]** Adapter diagnostics become coupled to method-local runtime details again.  
  **Mitigation:** Derive default adapter diagnostics from repo-owned provenance and only allow runtime-specific extensions when justified.

- **[Risk]** Resource monitoring gets conflated with unrelated console/report concerns.  
  **Mitigation:** Keep `resource_monitor` as a distinct sub-concern with its own config/runtime behavior.

- **[Risk]** Future dashboard goals bias the current design toward UI-driven abstractions too early.  
  **Mitigation:** Keep this change focused on backend observability contracts and sink routing, not frontend/dashboard implementation details.

## Migration Plan

1. Define the new `training-observability` capability and the structured summary/event expectations.
2. Keep stdlib `logging` plus Rich-backed console handling as the canonical console stack, and align early config/setup logging with that surface as early as practical.
3. Introduce the shared models plus `TrainingObserver` / backend sink seams directly in `library/logging/`.
4. Move startup diagnostics ownership behind structured summary rows and sectioned startup-block rendering in `library/logging/summaries.py` and `library/logging/console.py`.
5. Align tracker emission behind `library/logging/metrics.py` and backend adapters while keeping current Accelerate-backed behavior working.
6. Rework adapter diagnostics to derive from repo-owned adapter provenance and feed the shared summary model.
7. Move report composition toward `library/logging/reports.py` and reuse the shared startup/report summary data.
8. Keep resource monitoring as a distinct subsystem, but align its integration points with the broader observability contracts.
9. Leave unrelated wording-only lifecycle log cleanup outside this change unless it is directly required by the new ownership boundaries.

## Open Questions
None at this time.
