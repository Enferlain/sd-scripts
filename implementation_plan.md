# SDXL PEFT Data Pipeline Integration Plan

**Goal:** Replace legacy data loading with new pipeline in `scripts/sdxl_peft.py`.

---

## Finalized Design Decisions

| Question          | Decision                                                                |
| ----------------- | ----------------------------------------------------------------------- |
| Metadata          | Accept `DatasetManifest` directly; add `compute_tag_frequency()` helper |
| TE Caching        | Fully migrate to `SdxlTextEncoderPipelineStrategy`                      |
| Validation Config | Add `val_data_dir` to `SourceConfig`                                    |
| Validation Split  | Support both `val_data_dir` (priority) and `validation_split`           |

---

## Pre-Integration Changes

### 1. Add `val_data_dir` to Config

**File:** `library/config/dataclasses/data.py`

```python
@dataclass
class SourceConfig:
    train_data_dir: str | None = ...
    reg_data_dir: str | None = ...
    val_data_dir: str | None = field(default=None, metadata={"help": "directory for validation images"})
    ...
```

### 2. Add `cache_dir` to CachingConfig (if missing)

```python
@dataclass
class CachingConfig:
    cache_dir: str | None = field(default=None, metadata={"help": "directory for cache files"})
    ...
```

### 3. Update `create_manifest_from_config`

**File:** `library/data/pipeline/dataset_scanner.py`

```python
def create_manifest_from_config(
    data_config: DataConfig,
    validation: bool = False,  # NEW
    validation_split: float = 0.0,  # NEW
    validation_seed: int | None = None,  # NEW
    ...
) -> DatasetManifest:
    if validation:
        # Read from val_data_dir only
        if data_config.source.val_data_dir:
            scanned = scan_directory(data_config.source.val_data_dir, ...)
        else:
            raise ValueError("val_data_dir not configured")
    else:
        # Existing logic, but pass validation_split to scan_directory
        ...
```

### 4. Add Tag Frequency Helper

**File:** `library/data/pipeline/dataclasses.py` (or new utils file)

```python
def compute_tag_frequency(manifest: DatasetManifest, separator: str = ",") -> dict[str, dict[str, int]]:
    freq: dict[str, dict[str, int]] = {}
    for entry in manifest.entries.values():
        tags = [t.strip() for t in entry.caption.split(separator) if t.strip()]
        dir_name = Path(entry.image_path).parent.name
        if dir_name not in freq:
            freq[dir_name] = {}
        for tag in tags:
            freq[dir_name][tag] = freq[dir_name].get(tag, 0) + 1
    return freq
```

### 5. Refactor `create_training_metadata`

**File:** `library/training/training_metadata.py`

```python
def create_training_metadata(
    cfg,
    manifest: DatasetManifest,  # Changed from train_dataset_group
    val_manifest: DatasetManifest | None,  # Changed from val_dataset_group
    num_train_epochs: int,
    ...
) -> tuple:
    # Compute stats from manifest
    num_train = sum(e.num_repeats for e in manifest.entries.values() if not e.is_reg)
    num_reg = sum(e.num_repeats for e in manifest.entries.values() if e.is_reg)
    num_val = sum(e.num_repeats for e in val_manifest.entries.values()) if val_manifest else 0
    tag_freq = compute_tag_frequency(manifest, cfg.data.caption.caption_separator)
    ...
```

---

## Script Integration Phases

### Phase A: Imports

```diff
-from library.data._deprecated.dataset_setup import prepare_datasets
+from library.data.pipeline import (
+    DatasetManifest, EpochManifest, CaptionConfig,
+    CachingEngine, create_training_dataloader,
+    prepare_epoch, prepare_validation_epoch,
+)
+from library.data.pipeline.dataset_scanner import create_manifest_from_config
+from library.strategies.sdxl_caching import (
+    SdxlLatentsPipelineStrategy, SdxlTextEncoderPipelineStrategy,
+)
```

### Phase B: Manifest Creation (line ~118)

```python
train_manifest = create_manifest_from_config(
    data_config=cfg.data,
    validation_split=cfg.validation.validation_split,
    validation_seed=cfg.validation.validation_seed,
    latent_dtype="fp16" if not cfg.performance.precision.no_half_vae else "fp32",
)
val_manifest = None
if cfg.data.source.val_data_dir:
    val_manifest = create_manifest_from_config(cfg.data, validation=True)
elif cfg.validation.validation_split > 0:
    # Filter train_manifest entries where split="val"
    val_manifest = ...  # Extract val entries
```

### Phase C: Latent Caching (line ~150)

```python
latent_strategy = SdxlLatentsPipelineStrategy(
    flip_aug=cfg.data.preprocessing.flip_aug,
    dtype="fp16" if not cfg.performance.precision.no_half_vae else "fp32",
)
caching_engine = CachingEngine(strategy=latent_strategy, batch_size=cfg.data.caching.vae_batch_size)
train_manifest = caching_engine.cache_dataset(
    manifest=train_manifest, model=vae, accelerator=accelerator,
    cache_dir=cfg.data.caching.cache_dir,
    flip_aug=cfg.data.preprocessing.flip_aug,
)
```

### Phase D: TE Caching (line ~169)

```python
if cfg.sdxl.cache_text_encoder_outputs:
    te_strategy = SdxlTextEncoderPipelineStrategy(
        max_token_length=cfg.training.max_token_length,
    )
    te_engine = CachingEngine(strategy=te_strategy, batch_size=cfg.data.caching.vae_batch_size)
    train_manifest = te_engine.cache_dataset(
        manifest=train_manifest,
        model=(text_encoders[0], text_encoders[1]),  # Both encoders
        accelerator=accelerator,
        cache_dir=cfg.data.caching.cache_dir,
    )
```

### Phase E-G: DataLoader & Epoch Loop

DataLoader created per-epoch inside the training loop:

```python
for epoch in range(epoch_to_start, num_train_epochs):
    caption_config = CaptionConfig(
        shuffle_caption=cfg.data.caption.shuffle_caption,
        caption_dropout_rate=cfg.data.caption.caption_dropout_rate,
        ...
    )
    epoch_manifest = prepare_epoch(
        manifest=train_manifest, epoch=epoch, seed=cfg.training.seed,
        batch_size=cfg.training.train_batch_size,
        caption_config=caption_config,
    )
    train_dataloader = create_training_dataloader(
        dataset_manifest=train_manifest,
        epoch_manifest=epoch_manifest,
        latent_strategy=latent_strategy,
        te_strategy=te_strategy if cfg.sdxl.cache_text_encoder_outputs else None,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    for step, batch in enumerate(train_dataloader):
        ...
```

---

## Key Warnings

> [!CAUTION] > **Do NOT call `accelerator.prepare(train_dataloader)`** - the new pipeline handles sharding via `rank`/`world_size`.

> [!NOTE]
> DataLoader is now created **per-epoch** to enable caption augmentation and shuffle variation.
