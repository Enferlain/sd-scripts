# Prioritized TODO List

This document consolidates all TODOs from `ROADMAP.md` and `DATA_PIPELINE_CURRENT.md`, organized by priority and complexity with dependency analysis.

**Generated:** 2026-01-09

---

## Priority Legend

| Priority      | Meaning                                |
| ------------- | -------------------------------------- |
| 🔴 **P0**     | Blocking / Prerequisite for other work |
| 🟠 **P1**     | High value, should do soon             |
| 🟡 **P2**     | Important but not blocking             |
| 🟢 **P3**     | Nice to have / Low priority            |
| 🔵 **Future** | Long-term / Research phase             |

## Complexity Legend

| Complexity | Effort             |
| ---------- | ------------------ |
| ◯          | Trivial (< 1 hour) |
| ◐          | Small (1-4 hours)  |
| ●          | Medium (1-2 days)  |
| ⬤          | Large (1+ week)    |

---

## Tier 1: Blocking / Prerequisites

These should be completed first as other work depends on them.

| #   | Task                                    | Priority | Complexity | Blocking?           | Notes                                                                                                                                   |
| --- | --------------------------------------- | -------- | ---------- | ------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| 1.1 | **Benchmark new vs legacy performance** | 🔴 P0    | ◐ Small    | Yes                 | Must validate before removing deprecated code or migrating SD script. If performance regresses, need to investigate before any cleanup. |
| 1.2 | **Remove `library/data/_deprecated/`**  | 🔴 P0    | ◐ Small    | Blocked by 1.1, 1.3 | Can only remove once benchmarked AND SD script migrated.                                                                                |
| 1.3 | **Replace DataLoader in `sd_peft.py`**  | 🔴 P0    | ● Medium   | Blocked by 1.1      | Currently uses legacy `prepare_datasets()`. Migrate to new pipeline like `sdxl_peft.py`.                                                |

**Rationale:** These three are tightly coupled:

1. Benchmark first to ensure new pipeline is at least as good
2. Then migrate SD script
3. Then remove deprecated code

---

## Tier 2: Testing & Validation Gaps

Important to build confidence in the new pipeline before expanding features.

| #   | Task                                | Priority | Complexity | Notes                                               |
| --- | ----------------------------------- | -------- | ---------- | --------------------------------------------------- |
| 2.1 | **Multi-GPU sharding testing**      | 🟠 P1    | ◐ Small    | Requires multi-GPU setup. Manual test with 2+ GPUs. |
| 2.2 | **Validation loss testing**         | 🟠 P1    | ◐ Small    | Test `val_data_dir` path end-to-end.                |
| 2.3 | **Regularization testing**          | 🟠 P1    | ◐ Small    | Test `reg_data_dir` with `prior_loss_weight`.       |
| 2.4 | **Saving/resuming state testing**   | 🟠 P1    | ◐ Small    | Test checkpoint save + resume at mid-epoch.         |
| 2.5 | **Subsets testing**                 | 🟡 P2    | ◐ Small    | Lower priority, less common use case.               |
| 2.6 | **prefetch, bf16/fp16 paths**       | 🟡 P2    | ◐ Small    | Test with different precision configs.              |
| 2.7 | **Streaming tokens vs RAM loading** | 🟡 P2    | ◐ Small    | Needs GPU testing for memory/latency tradeoff.      |
| 2.8 | **`pin_memory=True` effectiveness** | 🟡 P2    | ◯ Trivial  | Requires CUDA device to test.                       |

---

## Tier 3: Data Pipeline Enhancements

Functional improvements for the new pipeline.

| #   | Task                                        | Priority | Complexity | Notes                                                                        |
| --- | ------------------------------------------- | -------- | ---------- | ---------------------------------------------------------------------------- |
| 3.1 | **Fast Skip: Add `start_batch_index`**      | 🟠 P1    | ◐ Small    | O(1) resume. Add param to `TrainingDataset.__init__()`, slice in `__iter__`. |
| 3.2 | **Token File Reuse**                        | 🟡 P2    | ◐ Small    | Check existing file + validate `manifest_hash` before regenerating.          |
| 3.3 | **Confirm selective re-caching works**      | 🟡 P2    | ◯ Trivial  | TODO note says "# TODO CONFIRM IF TRUE". Quick verification.                 |
| 3.4 | **Alpha mask encoding/saving**              | 🟡 P2    | ● Medium   | Detection works; need to implement actual mask saving/loading in caching.    |
| 3.5 | **ThreadPool per batch - CPU optimization** | 🟢 P3    | ◐ Small    | Performance optimization, not blocking.                                      |

---

## Tier 4: General Code Quality

Technical debt and cleanup items.

| #    | Task                                          | Priority | Complexity | Notes                                                                    |
| ---- | --------------------------------------------- | -------- | ---------- | ------------------------------------------------------------------------ |
| 4.1  | **Timestep sampling proper reimplementation** | 🟠 P1    | ● Medium   | Currently "hacked into training scripts". Should be cleaner abstraction. |
| 4.2  | **Config Validation Edge Cases**              | 🟡 P2    | ◐ Small    | Test `prepare_config()` and `validate_config()` for dataset conflicts.   |
| 4.3  | **Work on validation system**                 | 🟡 P2    | ● Medium   | System for catching invalid configs. May need to be post-testing.        |
| 4.4  | **Split `prepare_accelerator`**               | 🟡 P2    | ◐ Small    | Separate config computation from side effects for testability.           |
| 4.5  | **Explicit step 0 validation**                | 🟡 P2    | ◯ Trivial  | Add explicit early return for `calculate_val_loss_check` at step 0.      |
| 4.6  | **Pre-commit hooks for tests**                | 🟢 P3    | ◯ Trivial  | CI/CD enhancement.                                                       |
| 4.7  | **`edm2_loss_utils.py` Config Cleanup**       | 🟢 P3    | ◐ Small    | Extract to `Edm2LossConfig`. Not critical component.                     |
| 4.8  | **`training_plots.py` cleanup**               | 🟢 P3    | ◯ Trivial  | Functions access multiple sub-configs.                                   |
| 4.9  | **Consolidate `init_ipex()` calls**           | 🟢 P3    | ◯ Trivial  | May be useless post Torch 2.6.0. Investigate before removing.            |
| 4.10 | **Clean integration for `live_plotter`**      | 🟢 P3    | ◐ Small    | Possible rework with dedicated logging setup.                            |

---

## Tier 5: Future Ideas / Long-Term

Research items and architecture changes. Not blocking any current work.

### Performance & Scalability

| #   | Task                                    | Priority  | Complexity | Notes                                                                                  |
| --- | --------------------------------------- | --------- | ---------- | -------------------------------------------------------------------------------------- |
| 5.1 | **Fix zero-dimension bucket edge case** | 🟢 P3     | ◯ Trivial  | Images smaller than `bucket_reso_steps`.                                               |
| 5.2 | **Config-hash cache namespace**         | 🟢 P3     | ◐ Small    | Auto-segregate caches by config hash.                                                  |
| 5.3 | **Large-scale manifest optimization**   | 🔵 Future | ⬤ Large    | Binary format, incremental updates, lazy loading, sharding, databases. 100k+ datasets. |
| 5.4 | **FP8 for TE output storage**           | 🔵 Future | ◐ Small    | Needs tests for quality impact.                                                        |

### Large Scale Caching Architecture (Design Phase)

For 1M+ image datasets. Not implemented yet.

| #    | Task                                                | Priority  | Complexity | Notes |
| ---- | --------------------------------------------------- | --------- | ---------- | ----- |
| 5.5  | Design config-hash computation for latent namespace | 🔵 Future | ◐ Small    |       |
| 5.6  | Design caption-hash computation for TE dedup        | 🔵 Future | ◐ Small    |       |
| 5.7  | Implement `ShardedCacheStore`                       | 🔵 Future | ⬤ Large    |       |
| 5.8  | Implement shard-level `get_slice`                   | 🔵 Future | ● Medium   |       |
| 5.9  | Add `cache_backend` config option                   | 🔵 Future | ◐ Small    |       |
| 5.10 | Migration utility: per-image → sharded              | 🔵 Future | ● Medium   |       |

### Architecture & Features

| #    | Task                                   | Priority  | Complexity | Notes                                                  |
| ---- | -------------------------------------- | --------- | ---------- | ------------------------------------------------------ |
| 5.11 | **Per-Model Directory Structure**      | 🔵 Future | ⬤ Large    | Reorganize `library/models/` to per-model directories. |
| 5.12 | **Sample Generation Config Defaults**  | 🔵 Future | ◐ Small    | Global defaults in `SamplingConfig`.                   |
| 5.13 | **YAML support for `sample_prompts`**  | 🔵 Future | ◐ Small    | Currently only .txt, .toml, .json.                     |
| 5.14 | **Support for Feather**                | 🔵 Future | ● Medium   | FP8 emulation for older GPUs (inference).              |
| 5.15 | **Investigate 2022-2023 backend code** | 🔵 Future | ● Medium   | `sd_original_unet.py` may have outdated workarounds.   |
| 5.16 | **TE Caching + Caption Augmentations** | 🔵 Future | ● Medium   | Per-epoch TE caching or embedding-level augmentations. |

---

## Tier 6: Critical Performance - Caching Pipeline

Benchmarks show current caching at **3-4 it/s** vs planned **100+ it/s**. This is a 25x+ performance gap.

### Current vs Planned Architecture

```
CURRENT (Sequential - 3-4 it/s):
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────────┐
│  ThreadPool  │   │  SEQUENTIAL  │   │  SEQUENTIAL  │   │ SEQUENTIAL │
│  LOAD IMAGE  │──▶│ RESIZE/CROP  │──▶│ VAE ENCODE   │──▶│ SAVE DISK  │
│  (Parallel)  │   │ (Main thread)│   │ (Main thread)│   │(Main thread)│
└──────────────┘   └──────────────┘   └──────────────┘   └────────────┘
      ✓                  ✗                  ✗                  ✗

PLANNED (Pipelined - 100+ it/s):
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────────┐
│  Thread 1    │   │  Thread 2    │   │  Thread 3    │   │  Thread 4  │
│  LOAD IMAGE  │──▶│ RESIZE (GPU) │──▶│ VAE ENCODE   │──▶│ SAVE DISK  │
│  (Disk I/O)  │   │ (torchvision)│   │ (GPU)        │   │ (Async)    │
└──────────────┘   └──────────────┘   └──────────────┘   └────────────┘
      │                  │                  │                  │
      ▼                  ▼                  ▼                  ▼
  Queue(16)          Queue(16)          Queue(8)           Output
```

### Implementation Tasks

| #   | Task                                           | Priority | Complexity | Notes                                                                                                    |
| --- | ---------------------------------------------- | -------- | ---------- | -------------------------------------------------------------------------------------------------------- |
| 6.1 | **Async disk writes**                          | 🟠 P1    | ◐ Small    | Use `ThreadPoolExecutor` to save safetensors in background while GPU encodes next batch. ~10-20% speedup |
| 6.2 | **GPU-accelerated resize**                     | 🟠 P1    | ◐ Small    | Replace PIL resize with `torchvision.transforms.functional.resize`. ~20-30% speedup                      |
| 6.3 | **Pipeline queue architecture**                | 🟠 P1    | ● Medium   | Use `queue.Queue` with bounded size to connect stages. Each stage runs in its own thread.                |
| 6.4 | **Batch accumulator for VAE stage**            | 🟡 P2    | ◐ Small    | Collect same-resolution images until batch full, then encode together.                                   |
| 6.5 | **Progress bar on encoder (bottleneck) stage** | 🟢 P3    | ◯ Trivial  | Move tqdm to the actual bottleneck for accurate ETA.                                                     |

### Bottleneck Analysis (550 images, RTX 3090)

| Stage      | Current Time | Bottleneck | Fix                       |
| ---------- | ------------ | ---------- | ------------------------- |
| Load image | ~20ms        | Disk I/O   | Already parallelized ✓    |
| Resize     | ~15ms        | CPU (PIL)  | Move to GPU (torchvision) |
| VAE encode | ~10ms        | GPU        | Already fast, batch helps |
| Save disk  | ~15ms        | Disk I/O   | Async write in background |
| **Total**  | ~60ms/img    | Sequential | Pipeline overlap all      |

With pipelining: **Limited only by slowest stage (~20ms) = 50 it/s**
With GPU resize: Resize drops to ~2ms, load becomes bottleneck = **50-100 it/s**

---

## Recommended Execution Order

Based on dependencies and blocking relationships:

### Phase A: Validation & Migration (Do First)

```
1. Benchmark new vs legacy (1.1)
   ↓
2. Replace DataLoader in sd_peft.py (1.3)
   ↓
3. Remove library/data/_deprecated/ (1.2)
```

### Phase B: Testing Gaps (Parallel with A or After)

```
4. Multi-GPU sharding (2.1)
5. Validation loss (2.2)
6. Regularization (2.3)
7. Save/resume state (2.4)
```

### Phase C: Enhancements (After B)

```
8. Fast Skip (3.1)
9. Alpha mask (3.4)
10. Token File Reuse (3.2)
```

### Phase D: Code Quality (Ongoing)

```
11. Timestep sampling (4.1)
12. Config validation (4.2, 4.3)
13. Split prepare_accelerator (4.4)
```

---

## Quick Wins (Can Do Anytime)

These are fast, independent tasks:

- [ ] Explicit step 0 validation (4.5) - ◯ Trivial
- [ ] Confirm selective re-caching (3.3) - ◯ Trivial
- [ ] Fix zero-dimension bucket edge case (5.1) - ◯ Trivial
- [ ] `pin_memory=True` test (2.8) - ◯ Trivial
- [ ] Pre-commit hooks (4.6) - ◯ Trivial
