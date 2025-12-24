# PEFT Training Architecture Refactoring Plan (Status: CONSIDERED COMPLETED AS OF DEC 24 2025 ✅)

## Goal

Refactor `sd_peft.py` (2,326 lines, god class) into a modular, strategy-based architecture that:

- Follows the [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) principles
- Enables easy addition of new models (Flux, Lumina, Wan)
- Makes both `sd_peft.py` and `sdxl_peft.py` thin orchestrators like `*_finetune.py`

---

## Feasibility Analysis

### ✅ Why This Is Feasible

1. **Pattern already exists**: Strategy pattern proven in `library/strategies/` for tokenization/encoding
2. **Reference implementation**: `sd_finetune.py` (488 lines) shows the target pattern
3. **Incremental approach**: Can migrate one piece at a time without breaking existing functionality
4. **Shared logic is real**: SD and SDXL peft training loops are ~90% identical

### ⚠️ Risks & Mitigations

| Risk            | Mitigation                                                      |
| --------------- | --------------------------------------------------------------- |
| Regression bugs | Existing test suite (897 tests) + careful incremental migration |
| Feature parity  | Keep old code until new code passes same training runs          |
| External users  | Clean break - users can use forks or old commits if needed      |

---

## Proposed Architecture

### New File Structure

```
library/
├── strategies/
│   ├── strategy_base.py              # EXISTING (unchanged)
│   ├── strategy_sd.py                # EXISTING (unchanged)
│   ├── strategy_sdxl.py              # EXISTING (unchanged)
│   │
│   ├── peft_strategy_base.py         # NEW: Abstract interfaces
│   │   ├── class ModelLoadingStrategy
│   │   ├── class TextConditioningStrategy
│   │   ├── class UNetCallingStrategy
│   │   └── class NetworkCheckpointingStrategy
│   │
│   ├── peft_strategy_sd.py           # NEW: SD1.5/2 implementations
│   └── peft_strategy_sdxl.py         # NEW: SDXL implementations
│
├── training/
│   ├── peft_common.py                # NEW: Shared utilities
│   │   ├── setup_network()
│   │   ├── prepare_training_state()
│   │   └── generate_step_logs()
│   │
│   └── peft_trainer.py               # NEW: Generic training function
│       └── train_peft(config, strategies)

scripts/
├── sd_peft.py                        # MODIFIED: Thin wrapper (~400 lines)
└── sdxl_peft.py                      # MODIFIED: Thin wrapper (~300 lines)
```

### Strategy Interfaces

```python
# library/strategies/peft_strategy_base.py

class ModelLoadingStrategy(ABC):
    @abstractmethod
    def load_model(self, config, accelerator, weight_dtype) -> Tuple[List[TextEncoder], VAE, UNet]:
        """Load model components for this architecture."""
        pass

class TextConditioningStrategy(ABC):
    @abstractmethod
    def get_text_cond(self, batch, tokenizers, text_encoders, accelerator, weight_dtype):
        """Get text conditioning from batch."""
        pass

class UNetCallingStrategy(ABC):
    @abstractmethod
    def call_unet(self, unet, noisy_latents, timesteps, text_conds, batch, weight_dtype):
        """Call UNet with architecture-specific signature."""
        pass

class NetworkCheckpointingStrategy(ABC):
    @abstractmethod
    def save_network(self, network, config, epoch, step):
        """Save network checkpoint with architecture-specific metadata."""
        pass
```

### Target Script Structure

```python
# scripts/sd_peft.py (target: ~400 lines)

@hydra.main(config_path="../configs", config_name="sd_peft")
def main(config: SDPeftConfig):
    prepare_config(config)
    validate_config(config)

    # Set up strategies
    strategies = PeftStrategies(
        model_loading=SdModelLoadingStrategy(),
        text_conditioning=SdTextConditioningStrategy(),
        unet_calling=SdUNetCallingStrategy(),
        checkpointing=SdNetworkCheckpointingStrategy(),
    )

    # Call generic training function
    train_peft(config, strategies)
```

---

## Migration Phases

### Phase 1: Extract Utilities (Low Risk)

Extract pure functions from `SDPeftTrainer` that don't need model-specific behavior:

| Function                                    | Target Location                   |
| ------------------------------------------- | --------------------------------- |
| `generate_step_logs()`                      | `library/training/peft_common.py` |
| `step_logging()`, `epoch_logging()`         | `library/training/peft_common.py` |
| `save_timestep_distribution_plot()`         | `library/training/peft_common.py` |
| `switch_rng_state()`, `restore_rng_state()` | `library/training/peft_common.py` |

### Phase 2: Create Strategy Interfaces

Create abstract base classes in `library/strategies/peft_strategy_base.py`

### Phase 3: Implement SD Strategies

Extract from current `SDPeftTrainer`:

- `load_target_model()` → `SdModelLoadingStrategy`
- `get_text_cond()` → `SdTextConditioningStrategy`
- `call_unet()` → `SdUNetCallingStrategy`
- Checkpoint logic → `SdNetworkCheckpointingStrategy`

### Phase 4: Implement SDXL Strategies

Extract from current `SDXLPeftTrainer` overrides

### Phase 5: Create Generic Training Function

Extract the core loop from `SDPeftTrainer.train()` → `library/training/peft_trainer.py`

### Phase 6: Rewrite Scripts as Thin Wrappers

Convert `sd_peft.py` and `sdxl_peft.py` to use new architecture

### Phase 7: Remove Old Classes

Clean break - remove `SDPeftTrainer` inheritance pattern entirely

---

## Verification Plan

### Automated Tests

- Existing 897 unit tests should continue to pass
- Add new unit tests for each strategy class
- Add integration test: training 1 epoch produces valid network file

### Manual Verification

1. Train SD1.5 LoRA with new architecture, compare output to old
2. Train SDXL LoRA with new architecture, compare output to old
3. Verify checkpoint loading works across old/new

> [!NOTE]
> Manual verification should be done by the user since it requires actual model files and GPU access.

---

## Decisions Made

| Question                | Decision                                                                            |
| ----------------------- | ----------------------------------------------------------------------------------- |
| Backwards compatibility | **Clean break** - no deprecated wrapper needed. Users can use forks or old commits. |
| Naming convention       | **`peft_strategy_*.py`** - consistent with PEFT terminology                         |
| Migration priority      | **SD first** - it's the base training scheme, SDXL is already more isolated         |
