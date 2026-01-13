# Phase 3: Training Script Modularization

## Goal

Extract the monolithic `train()` function (~1200 lines) into reusable phase modules.

---

## Current Structure Analysis

The `train()` function in `sdxl_peft.py` (lines 108-1334) has these logical phases:

| Phase                   | Lines    | Responsibility                                         |
| ----------------------- | -------- | ------------------------------------------------------ |
| **Setup**               | 108-220  | Config validation, accelerator, tokenizers, strategies |
| **Manifests**           | 154-220  | Create/load train/val dataset manifests                |
| **Caching**             | 220-400  | Latent caching, TE caching (disk or memory)            |
| **Model Prep**          | 400-600  | Load UNet, VAE, TEs, adapter, precision settings       |
| **Optimizer/Scheduler** | 500-640  | Optimizer, LR scheduler, step calculation              |
| **Training Loop**       | 900-1300 | Epoch loop, batch processing, checkpointing            |

---

## Proposed Target Structure

```
library/training/phases/
├── __init__.py
├── context.py        # TrainingContext dataclass
├── setup.py          # Initial setup, accelerator, strategies
├── caching.py        # Latent + TE caching orchestration
├── model_prep.py     # Model loading & preparation
├── optimizer_prep.py # Optimizer & scheduler setup
└── loop.py           # Main training loop logic
```

---

## TrainingContext Design

A dataclass that carries state between phases:

```python
@dataclass
class TrainingContext:
    # Config
    cfg: SDXLPeftConfig
    strategies: TrainingStrategy

    # Accelerator & Devices
    accelerator: Accelerator
    device: torch.device
    weight_dtype: torch.dtype

    # Data
    train_manifest: DatasetManifest
    val_manifest: DatasetManifest | None
    tokenizers: list[Any]

    # Models (set during model_prep)
    unet: nn.Module | None = None
    vae: nn.Module | None = None
    text_encoders: list[nn.Module] = field(default_factory=list)
    adapter: nn.Module | None = None

    # Training state
    optimizer: Any = None
    lr_scheduler: Any = None
    global_step: int = 0
    epoch: int = 0
```

---

## Phase Functions

Each phase takes and returns `TrainingContext`:

```python
# library/training/phases/setup.py
def run_setup_phase(cfg: SDXLPeftConfig, strategies: TrainingStrategy) -> TrainingContext:
    """Create accelerator, load tokenizers, prepare strategies."""
    ...

# library/training/phases/caching.py
def run_caching_phase(ctx: TrainingContext) -> TrainingContext:
    """Cache latents and text encoder outputs."""
    ...

# library/training/phases/model_prep.py
def run_model_prep_phase(ctx: TrainingContext) -> TrainingContext:
    """Load and prepare UNet, VAE, text encoders, adapter."""
    ...

# library/training/phases/optimizer_prep.py
def run_optimizer_phase(ctx: TrainingContext) -> TrainingContext:
    """Create optimizer, scheduler, calculate steps."""
    ...

# library/training/phases/loop.py
def run_training_loop(ctx: TrainingContext) -> None:
    """Execute the main training loop."""
    ...
```

---

## Refactored train() Function

After modularization:

```python
def train(cfg: SDXLPeftConfig, strategies: SdxlTrainingStrategy):
    # Phase 1: Setup
    ctx = run_setup_phase(cfg, strategies)

    # Phase 2: Caching
    ctx = run_caching_phase(ctx)

    # Phase 3: Model Preparation
    ctx = run_model_prep_phase(ctx)

    # Phase 4: Optimizer Setup
    ctx = run_optimizer_phase(ctx)

    # Phase 5: Training Loop
    run_training_loop(ctx)
```

---

## Migration Strategy

### Option A: Incremental (Safer)

1. Create `TrainingContext` dataclass
2. Extract one phase at a time, test after each
3. Keep original code commented out for reference
4. Delete original code after all phases work

### Option B: Big Bang (Faster)

1. Create all phase files with extracted code
2. Wire up new `train()` function
3. Run integration test to verify

---

## Complexity Warnings

⚠️ **High Complexity:**

- The training loop has many nested closures (`save_model`, `tokenize_fn`, `on_step_start`)
- State is mutated extensively (`global_step`, `current_epoch`, loss recorders)
- Many conditional branches (TE caching modes, resume logic, validation)
- DeepSpeed and multi-GPU handling adds complexity

⚠️ **Recommended Approach:**
Start with **extracting just caching phase** since it's the most self-contained. The training loop itself has too many interdependencies to extract cleanly without significant refactoring.

---

## Questions for User

1. **Scope**: Extract all phases or just the cleanest ones (caching)?
2. **SD Support**: Should phases work for both SD and SDXL, or just SDXL first?
3. **Testing**: Run full training smoke test after each phase extraction?
