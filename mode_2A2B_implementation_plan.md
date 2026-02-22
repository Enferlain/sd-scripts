# Phase 2: Adapter-Neutral Shared Flow + FineTuneMode (Hard Cut)

## Background

Phase 1 extracted PEFT-specific behavior into `PeftMode`, but shared phases and runner code still carry adapter-specific assumptions.  
Phase 2 will be a hard migration with no compatibility shims, aliases, or dual APIs.

## Hard-Cut Rules

1. No `PeftTrainer` alias after rename.
2. No `all_reduce_adapter` alias after rename.
3. No temporary `*_new.py` scripts kept long-term; migrate canonical script paths directly.

---

## Phase 2A: Neutralize Shared Flow + Rename Runner

**Goal:** make shared phases mode-agnostic and rename the runner in the same pass.

### Audit Results — Remaining adapter-specific spots

| Location | What it does | Target |
| --- | --- | --- |
| `library/training/phases/training_loop.py:167` | `_on_step_start_for_adapter(...)` callback | Replace with `trainer.mode.on_step_start(trainer)` |
| `library/training/phases/training_loop.py:173,180,263` | passes `trainer.adapter` into strategy methods | Use `trainer.trainable_model` |
| `library/training/phases/training_loop.py:203` | `strategies.all_reduce_adapter(...)` | Rename call to `all_reduce_trainable(...)` |
| `library/training/phases/training_loop.py:205` | direct adapter params for grad clip | Use `trainer.mode.get_trainable_params(trainer)` |
| `library/training/phases/training_loop.py:239,332,436,492` | eval/train on unwrapped adapter | Use `trainer.mode.set_eval(trainer)` / `trainer.mode.set_train(trainer)` |
| `library/training/phases/training_loop.py:286,447` | checkpoint save target assumes adapter | Use `accelerator.unwrap_model(trainer.trainable_model)` |
| `library/training/runners/peft_trainer.py:181` | `_on_step_start_for_adapter` state | Remove; mode owns step-start behavior |
| `library/training/runners/peft_trainer.py:545,592` | eval/train on adapter in startup sampling | Use mode hooks |
| `library/training/runners/peft_trainer.py:613,626-628` | finalize unwrap/assert/save assumes adapter | Use `trainable_model` as primary target |

### Proposed Changes

#### [MOVE/RENAME] `library/training/runners/peft_trainer.py` -> `library/training/runners/trainer.py`

- Rename class `PeftTrainer` -> `Trainer`.
- Update all imports and type references across:
  - `library/training/phases/*.py`
  - `library/training/modes/*.py`
  - `scripts/*.py`
  - `tests/**`
- Update `library/training/runners/__init__.py` to export only `Trainer`.

#### [MODIFY] `library/training/modes/base.py`

Add mode hooks:

```python
def on_step_start(self, trainer: Trainer) -> None:
    """Mode-specific callback before each step."""

def get_trainable_params(self, trainer: Trainer) -> list:
    """Return parameters for gradient clipping."""

def set_eval(self, trainer: Trainer) -> None:
    """Switch primary trainable module(s) to eval mode."""

def set_train(self, trainer: Trainer) -> None:
    """Switch primary trainable module(s) to train mode."""
```

Keep current checkpoint contract with `target_model`; do not add a separate save-target hook.

#### [MODIFY] `library/training/modes/peft_mode.py`

Implement new hooks:

```python
def on_step_start(self, trainer):
    unwrapped = trainer.accelerator.unwrap_model(trainer.adapter)
    if hasattr(unwrapped, "on_step_start"):
        unwrapped.on_step_start(trainer._text_encoder, trainer.unet)

def get_trainable_params(self, trainer):
    return trainer.accelerator.unwrap_model(trainer.adapter).get_trainable_params()

def set_eval(self, trainer):
    trainer.accelerator.unwrap_model(trainer.adapter).eval()

def set_train(self, trainer):
    trainer.accelerator.unwrap_model(trainer.adapter).train()
```

#### [MODIFY] `library/training/runners/trainer.py`

- Add `trainable_model` property:
  - For current PEFT path returns `self.adapter`.
  - Future modes may set/override via mode-owned state.
- Remove `_on_step_start_for_adapter`.
- Update `_maybe_sample_at_start` to call mode `set_eval`/`set_train`.
- Update `_finalize_training` to use `self.trainable_model` for standard final save target.
- Keep EDM2 save path explicit by passing `_edm2_model` to `save_checkpoint(..., target_model=...)` via existing API flow.
- Widen SDXL-specific cached strategy type hints to neutral types (`Any`).

#### [MODIFY] `library/training/phases/training_loop.py`

Replace adapter-specific calls with mode/trainable calls:

```diff
-trainer._on_step_start_for_adapter(trainer._text_encoder, trainer.unet)
+trainer.mode.on_step_start(trainer)

-strategies.on_step_start(..., trainer.adapter, ...)
+strategies.on_step_start(..., trainer.trainable_model, ...)

-strategies.process_batch(..., trainer.adapter, ...)
+strategies.process_batch(..., trainer.trainable_model, ...)

-strategies.all_reduce_adapter(accelerator, trainer.adapter)
+strategies.all_reduce_trainable(accelerator, trainer.trainable_model)

-params_to_clip = accelerator.unwrap_model(trainer.adapter).get_trainable_params()
+params_to_clip = trainer.mode.get_trainable_params(trainer)

-accelerator.unwrap_model(trainer.adapter).eval()
+trainer.mode.set_eval(trainer)

-accelerator.unwrap_model(trainer.adapter).train()
+trainer.mode.set_train(trainer)

-trainer.save_checkpoint(ckpt_name, accelerator.unwrap_model(trainer.adapter), ...)
+trainer.save_checkpoint(ckpt_name, accelerator.unwrap_model(trainer.trainable_model), ...)
```

#### [MODIFY] `library/strategies/base/training.py` and implementations

- Rename abstract method to `all_reduce_trainable`.
- Update all concrete strategy classes in the same PR.
- Remove `all_reduce_adapter` entirely (no compatibility alias).

#### Test updates (Phase 2A)

- Rename runner imports/usages from `PeftTrainer` to `Trainer`.
- Update fixtures in `tests/unit/training/phases/conftest.py`:
  - provide `trainable_model`
  - provide `mode.on_step_start`, `mode.set_eval`, `mode.set_train`, `mode.get_trainable_params`
- Update training loop tests for `all_reduce_trainable`.

---

## Phase 2B: Implement `FineTuneMode` (SDXL)

**Goal:** add full-model SDXL fine-tune mode on top of the neutralized shared runner/phases.

### Proposed Changes

#### [NEW] `library/training/modes/finetune_mode.py`

Implement `TrainingMode` hooks for full-model training:

| Hook | FineTune behavior |
| --- | --- |
| `prepare_trainables` | unfreeze UNet and optional text encoders |
| `configure_trainable_precision` | cast full trainable modules to configured dtype |
| `build_optimizer_params` | build optimizer from UNet + optional TE params |
| `prepare_with_accelerator` | wrap UNet/TE trainables and optimizer/scheduler |
| `setup_gradient_training` | gradient checkpointing for UNet/TE |
| `register_state_hooks` | full-model state hook registration |
| `on_epoch_start` | fine-tune-specific epoch start behavior |
| `on_step_start` | no-op or mode-specific pre-step action |
| `on_step_end` | mode-specific post-step action (often empty) |
| `get_trainable_params` | return UNet/TE params for clipping |
| `set_eval` / `set_train` | switch full trainable modules mode |
| `save_checkpoint` | full-model save path using existing SDXL checkpoint helpers |

Required mode contract details for `FineTuneMode`:

- `prepare_with_accelerator` must set both:
  - `trainer._grad_sync_handle` (object passed to `accelerator.accumulate(...)`)
  - `trainer._primary_trainable` (semantic trainable model returned by `trainer.trainable_model`)
- `register_state_hooks` must explicitly define full-model checkpoint state behavior:
  - either register full-model save/load hooks for resume parity
  - or be a documented no-op if `save_checkpoint` fully owns persistence in this mode

#### [MODIFY] `scripts/sdxl_finetune.py` (direct migration)

Convert existing script to thin entrypoint:

```python
mode = FineTuneMode()
trainer = Trainer(cfg, strategies, mode)
trainer.train()
```

#### [MODIFY] `scripts/sdxl_peft.py`

Use renamed runner:

```python
mode = PeftMode()
trainer = Trainer(cfg, strategies, mode)
trainer.train()
```

#### Tests (Phase 2B)

- Unit tests for `FineTuneMode`.
- Integration test proving shared phase lifecycle works with both `PeftMode` and `FineTuneMode`.
- Checkpoint-content tests:
  - PEFT mode saves adapter payload.
  - FineTune mode saves full-model payload.

---

## Verification Plan

### Phase 2A checks

- `rg -n "PeftTrainer" library scripts tests` returns zero hits.
- `rg -n "all_reduce_adapter" library scripts tests` returns zero hits.
- `rg -n "trainer\\.adapter" library/training/phases/training_loop.py` returns zero hits.
- `uv run ruff check` passes on touched files.
- Relevant unit tests pass.

### Phase 2B checks

- `FineTuneMode` tests pass.
- `scripts/sdxl_finetune.py` imports and constructs `Trainer(..., mode=FineTuneMode())` without errors.
- Both modes produce expected checkpoint artifacts in tests.
- `FineTuneMode.prepare_with_accelerator()` sets `_grad_sync_handle` and `_primary_trainable` in tests.
- `register_state_hooks` behavior is covered by tests (hook-based resume or explicit no-op policy).
