# PEFT and Fine-Tune Runners: Findings and Suggested Approach

Date: 2026-02-16
Owner: Agent notes for follow-up contributors

## Purpose

Document current state of `library/strategies`, `library/training/phases`, and runner architecture, then propose a practical migration path so PEFT and full fine-tune can share one clear training system.

## Scope Investigated

- `library/strategies/base`, `library/strategies/sd`, `library/strategies/sdxl`
- `library/training/runners`, `library/training/phases`, checkpoint helpers
- `scripts/sdxl_peft.py`, `scripts/sdxl_finetune.py`, `scripts/sd_peft.py`, `scripts/sd_finetune.py`
- `ROADMAP.md` section "Training Mode Extensibility"

## Architecture Snapshot

- `scripts/sdxl_peft.py` is the only thin script currently using the extracted runner/phases architecture.
- `scripts/sdxl_finetune.py` and `scripts/sd_finetune.py` are still legacy monolithic training scripts.
- The extracted phase pipeline exists, but is still tightly coupled to PEFT and in multiple places to SDXL.

## Key Findings

1. Roadmap diagnosis is correct and still active.
- `ROADMAP.md:97` states phase functions are typed to `PeftTrainer` and fine-tune needs the same architecture.
- `ROADMAP.md:112` to `ROADMAP.md:115` estimates shareability by phase and is directionally accurate.

2. `PeftTrainer` and phases are PEFT-centric by design.
- Adapter-first state is baked into runner fields and save path: `library/training/runners/peft_trainer.py:95`, `library/training/runners/peft_trainer.py:362`.
- Runner lifecycle is phase-based and good, but mode-neutral naming has not happened yet: `library/training/runners/peft_trainer.py:182`.

3. "Shared" caching phase contains SDXL hardcoding.
- SDXL strategy classes are imported directly in phase code: `library/training/phases/caching.py:18`.
- Latent and TE caching use SDXL concrete strategies directly: `library/training/phases/caching.py:62`, `library/training/phases/caching.py:135`.
- This blocks clean reuse for SD and future model families.

4. Optimizer phase is adapter-specific.
- Optimizer setup calls adapter-aware utility with `cfg.peft` and `trainer.adapter`: `library/training/phases/optimizer.py:52`.
- `accelerator.prepare` wraps `trainer.adapter` as main training model: `library/training/phases/optimizer.py:178`.
- Gradient clipping and grad prep are adapter API dependent: `library/training/phases/optimizer.py:214`.

5. Training loop still assumes PEFT behavior.
- Per-epoch callback is adapter method: `library/training/phases/training_loop.py:71`.
- Adapter-specific scale norm regularization is in loop: `library/training/phases/training_loop.py:219`.
- Checkpoint save paths currently pass unwrapped adapter model objects: `library/training/phases/training_loop.py:288`.

6. Tokenization branch in training loop is SDXL-specific.
- Per-epoch token caching imports SDXL tokenizer helper directly: `library/training/phases/training_loop.py:98`.
- This should move behind strategy/mode hooks.

7. SD strategy data contract is behind the new dataloader.
- New loader emits `input_ids` and `text_encoder_outputs`: `library/data/dataloader.py:288`, `library/data/dataloader.py:304`.
- SD strategy still reads legacy keys `input_ids_list` and `text_encoder_outputs_list`: `library/strategies/sd/training.py:465`, `library/strategies/sd/training.py:477`.
- This matches roadmap TODO: `ROADMAP.md:82`.

8. Strategy boundary cleanup is partially done, not complete.
- `TrainingStrategy` now includes abstract `process_batch` and `calculate_val_loss`: `library/strategies/base/training.py:598`, `library/strategies/base/training.py:652`.
- Validation loop ownership is still inside strategies (`sd` and `sdxl`): `library/strategies/sd/training.py:644`, `library/strategies/sdxl/training.py:902`.

9. `runners` package name implies multiple modes, but only PEFT runner exists.
- `library/training/runners/__init__.py:2` advertises "PEFT, finetune, etc."
- Export list includes only `PeftTrainer`: `library/training/runners/__init__.py:6`.

10. There are temporary debug logs in SDXL strategy path.
- `_get_text_cond` includes explicit debug lines marked "remove": `library/strategies/sdxl/training.py:557`, `library/strategies/sdxl/training.py:592`, `library/strategies/sdxl/training.py:604`.

## Suggested Direction

Use one runner lifecycle with a small mode plugin layer.

Reasoning:
- Keeps current phase investment.
- Avoids duplicating orchestration into separate large trainer classes.
- Keeps model-family differences in strategy and mode differences in mode hooks.

Recommended interpretation of roadmap option:
- Follow Option A (`TrainingMode`) from `ROADMAP.md:119`, but keep the mode API minimal and explicit.

## Minimal Target Design

1. Keep one orchestrator.
- Rename `PeftTrainer` to neutral `Trainer` only after mode abstractions are in place.

2. Introduce mode protocol with only divergent concerns.
- Proposed methods:
  - `prepare_trainables(trainer)` for mode-specific model prep.
  - `build_optimizer(trainer)` for param groups and optimizer creation.
  - `prepare_with_accelerator(trainer)` for wrapping target models.
  - `on_epoch_start(trainer)` for mode-specific callbacks.
  - `save_checkpoint(trainer, ...)` for mode-specific checkpoint format.
  - `finalize_and_save(trainer)` for end-of-training save behavior.

3. Keep strategy axis for model family math and conditioning.
- `strategies/*/training.py` stays responsible for batch processing math and model-family specifics.

4. Move SDXL hardcoding out of phases.
- Caching strategy creation, tokenization function, and TE cache behavior should come from strategy or mode hooks.

## Phased Migration Plan

Phase 0: Stabilize current PEFT core before adding new abstractions.
- Make `caching.py` call strategy-provided caching strategy factories.
- Remove SDXL-only tokenization import from `training_loop.py`.
- Align SD strategy with new dataloader keys.

Phase 1: Add mode interface while preserving behavior.
- Introduce `PeftMode` implementing current behavior.
- Replace direct adapter calls in phases with mode calls.
- Keep tests green with no functional changes.

Phase 2: Implement `FineTuneMode` for SDXL first.
- Reuse existing phase flow.
- Switch trainables from adapter params to unet/te params.
- Use existing full-model checkpoint helpers (`library/training/sdxl_checkpointing.py`).

Phase 3: Migrate `scripts/sdxl_finetune.py` to thin entrypoint.
- Match shape of `scripts/sdxl_peft.py`.
- Keep config schema as `SDXLFineTuneConfig`.

Phase 4: Migrate SD modes.
- Port `scripts/sd_finetune.py` and optionally `scripts/sd_peft.py` to the same runner.
- Remove remaining legacy key assumptions.

Phase 5: Validation boundary cleanup.
- Move validation loop orchestration from strategies into trainer phase.
- Keep strategy method for per-batch val computation.

## Acceptance Criteria

- `scripts/sdxl_peft.py` and `scripts/sdxl_finetune.py` both use same runner lifecycle.
- No direct SDXL strategy class references inside generic phase files.
- SD and SDXL strategies both consume new batch contract keys from `library/data/dataloader.py`.
- Checkpoint behavior remains correct for both PEFT adapters and full-model fine-tune.
- Existing phase unit and training-loop integration tests pass.

## Risks and Watchouts

- Resume and checkpoint hooks currently assume adapter-type filtering.
  - See `library/training/checkpointing.py:534`.
- Gradient checkpointing order is preserved from legacy and may need distributed verification.
  - Context in `ROADMAP.md` and audit references.
- Over-abstraction risk.
  - Keep mode API small and concrete.

## Suggested Next Tasks for Agents

1. Remove SDXL hardcoding in `library/training/phases/caching.py`.
2. Update `SdTrainingStrategy` to read `input_ids` and `text_encoder_outputs` from new dataloader format.
3. Introduce mode interface plus `PeftMode` pass-through implementation.
4. Write focused tests around mode hook integration and checkpoint behavior.
5. Migrate `scripts/sdxl_finetune.py` to thin runner entrypoint.

## Notes

- This document is intentionally implementation-oriented for agent handoff.
- Use `ROADMAP.md` as the canonical task tracker and mirror completed milestones there.
