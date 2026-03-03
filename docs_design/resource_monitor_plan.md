# Resource Monitoring Plan (2026-02-23)

## Objective

Add detailed, production-usable resource monitoring (GPU/CPU/memory/component footprint) with minimal impact on training flow and runtime performance. Controllable via config.

## Current State (Verified)

1. Resource tracking exists as debug/benchmark utility in `library/utils/resource_tracker.py`.
2. It is wired ad hoc via `BENCHMARK_RESOURCES` environment checks in:
   - `library/training/phases/caching.py`
   - `library/training/phases/training_loop.py`
3. Current tracker is good for coarse block timing/memory summaries, but is not integrated as a first-class, configurable monitoring system.
4. Per-step resource tracking is not structured; there is no typed config schema controlling frequency/mode/rank scope.

## Non-Invasive Design Goals

1. Keep core training math untouched.
2. Add monitoring through phase/step hooks, not inline instrumentation everywhere.
3. Default overhead near-zero when disabled.
4. Keep outputs useful both in console and trackers.
5. Work in single GPU, DDP, and DeepSpeed setups.

## Proposed Architecture

## 1. Central service: `ResourceMonitor`

Create `library/logging/resource_monitor.py` with one monitor instance owned by `Trainer`.

### Public API

1. `start_session()`
2. `end_session()`
3. `phase_start(name: str)`
4. `phase_end(name: str)`
5. `step_end(global_step: int, epoch: int)`
6. `emit_startup_component_memory(...)`

This preserves clean phase code: phases call monitor hooks only.

## 2. Monitoring modes

Use one typed mode switch:

1. `off`
2. `basic`
3. `sampled`
4. `deep`

### Mode behavior

1. `off`: no-op monitor.
2. `basic`:
   - stage/session summaries only
   - low-cost `torch.cuda.memory_*` + process RSS
   - no subprocess polling
3. `sampled`:
   - includes periodic background sampling for true device memory
   - interval-based, not per-step forced sync
4. `deep`:
   - short profiling windows only (opt-in)
   - memory timeline/snapshot artifacts for debugging specific issues

## 3. Configuration schema (typed)

Extend `LoggingConfig` with:

1. `resource_monitor_enabled: bool = False`
2. `resource_monitor_mode: str = "off"`  # off|basic|sampled|deep
3. `resource_monitor_log_every_n_steps: int = 0`  # 0 disables step resource logs
4. `resource_monitor_sample_interval_sec: float = 1.0`
5. `resource_monitor_rank_scope: str = "main"`  # main|all
6. `resource_monitor_device_scope: str = "local"`  # local|all_visible
7. `resource_monitor_output_jsonl: str | None = None`  # optional structured stream
8. `resource_monitor_jsonl_flush_mode: str = "auto"`  # auto|line|batch
9. `resource_monitor_jsonl_flush_every_n_events: int = 50`
10. `resource_monitor_queue_maxsize: int = 1024`
11. `resource_monitor_drop_policy: str = "drop_oldest"`  # drop_oldest|drop_newest|block
12. `resource_monitor_max_collection_ms: float = 0.0`  # 0 disables budget enforcement
13. `resource_monitor_phase_summary: bool = True`
14. `resource_monitor_component_breakdown: bool = True`
15. `resource_monitor_deep_window_steps: int = 0`  # 0 disables deep windows
16. `resource_monitor_deep_window_seconds: float = 0.0`  # 0 disables time-based deep windows

Default behavior must preserve current training speed and log volume.

## 3.1 Config validation rules

Add explicit validation/normalization in `config_validation.py`:

1. `resource_monitor_mode` must be one of `off|basic|sampled|deep`.
2. `resource_monitor_log_every_n_steps >= 0` (`0` means disabled).
3. `resource_monitor_sample_interval_sec > 0`.
4. `resource_monitor_rank_scope` must be one of `main|all`.
5. `resource_monitor_device_scope` must be one of `local|all_visible`.
6. `resource_monitor_jsonl_flush_mode` must be one of `auto|line|batch`.
7. `resource_monitor_jsonl_flush_every_n_events >= 1`.
8. `resource_monitor_queue_maxsize >= 1`.
9. `resource_monitor_drop_policy` must be one of `drop_oldest|drop_newest|block`.
10. `resource_monitor_max_collection_ms >= 0`.
11. `resource_monitor_deep_window_steps >= 0`.
12. `resource_monitor_deep_window_seconds >= 0`.
13. If `resource_monitor_enabled=False`, force effective mode to `off` regardless of mode string.
14. If mode is `off`, ignore sampling/jsonl/deep-window-specific options without error.

## 3.2 Execution semantics (explicit contract)

To avoid hidden behavior differences, implementation should follow these rules:

1. Monitor object never raises into training flow; internal errors are downgraded to warning logs.
2. `off` mode uses a strict no-op implementation (`NoOpResourceMonitor`) with near-zero overhead.
3. `step_end()` is called only when `accelerator.sync_gradients` is `True` (optimization steps only).
4. All memory values use explicit units (MB for human logs, bytes or MB in JSONL with field suffix).
5. Phase names are stable strings to keep logs/dashboards comparable across runs.
6. Monitor timing uses:
   - wall-clock durations (`time.perf_counter`) for end-to-end phase/session time
   - optional CUDA event timing only in deep mode windows for GPU-kernel timing
7. Per-phase peak memory requires `torch.cuda.reset_peak_memory_stats()` at `phase_start()` when CUDA is available.

## 3.3 Sampler thread contract

`sampled`/`deep` mode sampler behavior must be explicit:

1. Sampler runs as a daemon thread and must never block process exit.
2. Sampler writes events to a bounded queue (`resource_monitor_queue_maxsize`).
3. Queue pressure behavior is controlled by `resource_monitor_drop_policy`.
4. Sampler exceptions are captured and surfaced once through monitor warnings; monitor continues in degraded mode.
5. Main training thread never waits on sampler queue in hot path.

## 4. Integration points (minimal changes)

## Trainer lifecycle

1. In `Trainer.setup()`:
   - instantiate monitor from config
   - `monitor.start_session()`
2. In `_log_training_info()`:
   - emit one startup component/optimizer memory estimate block
3. In `_finalize_training()`:
   - `monitor.end_session()`

## Caching phases

In `library/training/phases/caching.py`:

1. Wrap latent caching with `phase_start("latent_caching")` / `phase_end("latent_caching")`
2. Wrap TE caching similarly.
3. Remove direct env-var checks once monitor mode is adopted.

## Training loop

In `library/training/phases/training_loop.py`:

1. `phase_start("training_epoch_<n>")` at epoch start, `phase_end(...)` at epoch end.
2. `step_end(global_step, epoch)` at optimization steps only (already gated by `accelerator.sync_gradients`).

## Metric Model

## Always-on cheap metrics (basic mode)

1. GPU allocated/reserved/peak (`torch.cuda.memory_allocated`, `memory_reserved`, `max_memory_allocated`).
2. CPU process RSS (psutil).
3. Duration per phase and session (wall-clock).
4. Throughput:
   - `steps_per_sec` from optimization-step deltas
   - optional `samples_per_sec` when effective batch size is known

## Sampled metrics (sampled/deep mode)

1. Device-level memory used (prefer NVML; fallback to existing approach if unavailable).
2. Peak tracking via background sampler thread at configured interval.
3. For short/fast steps, complement sampler with cheap per-step snapshots in `step_end()`:
   - `memory_allocated`
   - `memory_reserved`
4. Document interval caveat: if sample interval exceeds step duration, transient peaks can be missed.

## Deep mode diagnostics

Deep mode adds optional high-detail counters for short windows:

1. `torch.cuda.memory_stats()` counters (allocation retries, OOM-related counters, fragmentation signals).
2. Optional CUDA-event timing per deep window for GPU-side duration.
3. Deep windows must be explicitly bounded by `resource_monitor_deep_window_steps` or `resource_monitor_deep_window_seconds`.

## Startup component memory accounting (estimates)

Emit one table after model/optimizer prep:

1. Per component:
   - parameter bytes (dtype-aware)
   - trainable parameter bytes
2. Gradient bytes estimate:
   - ~trainable param bytes (or precision-adjusted)
3. Optimizer state bytes estimate by optimizer family:
   - Adam-like: ~2 states/param (+master weights if applicable)
   - SGD-like: momentum optional
4. Total estimated model+grad+optimizer memory footprint.

For DeepSpeed/ZeRO: annotate that effective per-rank footprint depends on stage partitioning.

## Output Strategy

## Human-readable logs

1. Startup summary block: component estimates + monitor mode/rank scope.
2. Phase summaries: duration + start/end/peak memory.
3. Optional periodic step resource line every `resource_monitor_log_every_n_steps`.
4. Include reserved-vs-allocated note in startup output:
   - allocated = active tensor memory tracked by allocator
   - reserved = allocator-managed pool (includes cached/free blocks)

## Structured outputs

Optional JSONL stream with events:

1. `session_start`, `phase_start`, `phase_end`, `step_sample`, `session_end`
2. Useful for post-run graphing and regression checks.

### JSONL event schema (fixed keys)

Use a stable event contract so downstream tooling does not break:

1. `ts` (unix seconds, float)
2. `event` (string)
3. `rank` (int)
4. `world_size` (int)
5. `mode` (string)
6. `device_scope` (string)
7. `global_step` (int | null)
8. `epoch` (int | null)
9. `phase` (string | null)
10. `duration_ms` (float | null)
11. `gpu_allocated_mb` (float | null)
12. `gpu_reserved_mb` (float | null)
13. `gpu_peak_allocated_mb` (float | null)
14. `gpu_used_mb` (float | null, sampled/deep when available)
15. `cpu_rss_mb` (float | null)
16. `steps_per_sec` (float | null)
17. `samples_per_sec` (float | null)
18. `dropped_samples` (int | null)
19. `collection_ms` (float | null)

If `resource_monitor_output_jsonl` is relative, resolve it under `cfg.output.saving.output_dir`.
Writes should append line-by-line with explicit flush behavior:

1. `auto`: line-flush for `sampled`/`deep`, batch flush for `basic`.
2. `line`: flush every event.
3. `batch`: flush every `resource_monitor_jsonl_flush_every_n_events` events.
4. Always flush at `phase_end()` and `end_session()`.

## Performance Guardrails

1. No mandatory `torch.cuda.synchronize()` inside per-step path.
2. Any high-cost collection in background sampler (sampled/deep mode only).
3. Keep default config effectively no-op (`off`).
4. Ensure rank filtering defaults to main process to avoid log storms.
5. Enforce optional collection budget using `resource_monitor_max_collection_ms` in sampled/deep mode.

### Performance budget targets

Use benchmark runs to verify practical ceilings:

1. `off`: indistinguishable from baseline (target <0.2% runtime delta).
2. `basic`: low overhead (target <1% runtime delta).
3. `sampled`: overhead bounded primarily by `resource_monitor_sample_interval_sec`.
4. `deep`: explicitly debug-oriented; higher overhead is acceptable but should be scoped to short windows.

## Dependency and fallback policy

1. CPU metrics use existing `psutil` dependency.
2. Device-used memory in sampled/deep mode should prefer NVML when available.
3. NVML is optional: if unavailable, continue with reduced metrics and log one warning on startup (not every step).
4. Unsupported backends/platforms should degrade gracefully, not disable training.
5. NVML polling should stay in sampler thread; main thread should use cheap allocator counters at configured cadence.

## Migration Plan

### Phase 0: Introduce monitor skeleton

1. Add `ResourceMonitor` class with no-op behavior and basic phase/session tracking.
2. Wire it into trainer setup/finalize and caching/training phase hooks.
3. Remove direct `BENCHMARK_RESOURCES` checks from phase code as part of monitor wiring.

### Phase 1: Config-driven replacement

1. Add logging config fields.
2. Switch phase code from env-var checks to monitor mode.
3. Use only typed config to control monitor behavior and output.

### Phase 2: Component/optimizer footprint block

1. Add startup estimation block.
2. Integrate with existing diagnostics output to avoid duplicate sections.

### Phase 3: Sampled + JSONL modes

1. Add sampler thread mode.
2. Add optional JSONL artifact.
3. Add targeted tests for sampler lifecycle:
   - daemon behavior
   - bounded queue/drop policy
   - exception surfacing
4. Add targeted tests for event schema and flush modes.

## Validation Checklist

1. Disabled mode adds no measurable overhead in short benchmark run.
2. Basic mode emits phase/session summaries for:
   - latent caching
   - TE caching
   - training epochs
   - full training
3. Sampled mode records higher-fidelity peak memory than start/end-only snapshots.
4. Multi-GPU: logs remain rank-scoped according to config.
5. DeepSpeed runs include explicit partitioning caveat in startup summary.
6. Per-phase peaks reset correctly and do not leak previous-phase maxima.
7. Throughput fields (`steps_per_sec`) are present and stable.
8. Sampled/deep mode reports sampler failures/dropped samples without breaking training.

## Testing Plan

1. Unit tests:
   - monitor no-op behavior
   - phase duration/memory aggregation
   - config parsing and mode gating
2. Integration tests:
   - one short PEFT run (`basic`)
   - one short fine-tune run (`basic`)
   - optional sampled smoke test (skip when NVML unavailable)

## Out of Scope

1. Full profiler UI integration.
2. Continuous per-kernel profiling during all training steps.
3. Replacing existing progress bar or tracker systems.
