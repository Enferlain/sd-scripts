# Data Pipeline Architecture Plan

This document outlines a proposed redesign of the data loading and caching system to address significant performance bottlenecks identified in the current implementation.

## Problem Statement

The current data loading architecture has severe performance issues:

| Metric                | Current     | Target         |
| --------------------- | ----------- | -------------- |
| Latent caching speed  | ~4 it/s     | 100+ it/s      |
| Training data loading | ~4 it/s     | 200+ batches/s |
| Memory predictability | Random OOMs | Predetermined  |

### Root Causes

1. **Monolithic Architecture**: The `dataset.py` file (~1300 lines) handles discovery, bucketing, caching, and training data loading all in one place. These responsibilities are intertwined in ways that prevent optimization.

2. **Serial Processing During Caching**: The `new_cache_latents()` method processes batches sequentially:

   - Wait for all images in batch to load
   - Then send to VAE
   - Then save to disk
   - Then move to next batch

   This leaves the GPU idle during I/O operations.

3. **DataLoader Misuse**: The DataLoader is configured with `batch_size=1`, with each `__getitem__` call returning a custom "batch". This means:

   - No true parallel data loading
   - Heavy processing inside `__getitem__` (caption processing, latent loading, collation)
   - Multiprocessing overhead without the benefits

4. **Runtime Bucketing**: Bucket assignments and batch organization happen during training iteration, adding per-step overhead.

5. **Unpredictable Memory**: Bucket order after shuffling is random, meaning a large-resolution batch might come first (causing OOM) or later (wasting allocated memory).

---

## Proposed Architecture

### Core Principle: Separation of Concerns

Split the current monolithic system into four distinct phases:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  PHASE 1        │     │  PHASE 2        │     │  PHASE 3        │     │  PHASE 4        │
│  Dataset        │────▶│  Latent/TE      │────▶│  Epoch          │────▶│  Training       │
│  Preparation    │     │  Caching        │     │  Preparation    │     │  Data Loading   │
│  (runs once)    │     │  (runs once)    │     │  (per epoch)    │     │  (per step)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

Each phase has a single responsibility, clear inputs, and clear outputs.

---

## Phase 1: Dataset Preparation

**Purpose**: Scan directories, read captions, compute bucket assignments, generate manifest.

**When it runs**: Once per dataset, or when dataset changes.

**Input**:

- Image directories
- Caption files
- Configuration (resolution, bucket settings)

**Output**: A single `dataset.json` manifest file:

```json
{
  "version": "2.0",
  "created_at": "2024-01-15T10:30:00Z",
  "config": {
    "base_resolution": [1024, 1024],
    "bucket_reso_steps": 64,
    "min_bucket_reso": 512,
    "max_bucket_reso": 1536
  },
  "images": [
    {
      "id": "img_00001",
      "path": "/data/images/photo001.png",
      "size": [1920, 1080],
      "bucket_reso": [1024, 576],
      "resized_size": [1024, 576],
      "caption": "a photograph of a sunset over mountains",
      "tags": ["photograph", "sunset", "mountains", "landscape"],
      "num_repeats": 1,
      "latent_cache": "/cache/latents/img_00001.safetensors",
      "te_cache": "/cache/te/img_00001.safetensors"
    }
  ],
  "buckets": {
    "1024x576": {
      "resolution": [1024, 576],
      "count": 1500,
      "recommended_batch_size": 4
    },
    "768x768": {
      "resolution": [768, 768],
      "count": 3000,
      "recommended_batch_size": 6
    }
  },
  "statistics": {
    "total_images": 50000,
    "total_with_repeats": 75000,
    "bucket_count": 12,
    "mean_aspect_ratio_error": 0.023
  }
}
```

### Justification

- **Single source of truth**: All dataset information in one human-readable file
- **Debuggable**: Can inspect/edit manifest without running code
- **Fast training startup**: No directory scanning, caption reading, or bucket computation at training time
- **Portable**: Manifest can be shared, versioned, filtered with external tools
- **Pre-computed batch sizes**: Memory-safe batch sizes determined upfront based on resolution

### Implementation Notes

- Use `imagesize` library for fast resolution detection (no full image load)
- Parallelize with ThreadPoolExecutor for I/O-bound caption reading
- Store relative paths in manifest, resolve at load time for portability

---

## Phase 2: Caching Pipeline

**Purpose**: Encode images to latents (VAE) and captions to embeddings (text encoders), save to disk.

**When it runs**: Once per dataset, or when cache is invalidated.

**Input**:

- Dataset manifest from Phase 1
- VAE model
- Text encoder model(s)

**Output**:

- Per-image `.safetensors` files with latents
- Per-image `.safetensors` files with text encoder outputs

### Why `.safetensors` over `.npz`?

| Aspect           | `.npz` (NumPy)                     | `.safetensors` (HuggingFace) |
| ---------------- | ---------------------------------- | ---------------------------- |
| **Loading**      | Loads entire file into RAM         | Memory-mapped (lazy load)    |
| **Speed**        | Decompress + copy                  | Direct memory access         |
| **GPU transfer** | RAM → copy → GPU                   | Near-direct to GPU           |
| **Safety**       | Pickle-based (code execution risk) | No code execution            |
| **Metadata**     | Limited                            | Built-in JSON metadata       |

### Format Abstraction

The cache I/O should be abstracted behind a format-agnostic interface. Default to `.safetensors` for new caches, but design the code to easily support `.npz` for users with existing caches:

```python
@dataclass
class LatentCacheEntry:
    latents: torch.Tensor
    latents_flipped: Optional[torch.Tensor]
    original_size: Tuple[int, int]
    crop_ltrb: Tuple[int, int, int, int]

def save_cache(path: str, entry: LatentCacheEntry):
    """Save to format based on file extension"""
    if path.endswith(".safetensors"):
        _save_safetensors(path, entry)
    elif path.endswith(".npz"):
        _save_npz(path, entry)
    else:
        raise ValueError(f"Unknown format: {path}")

def load_cache(path: str) -> LatentCacheEntry:
    """Load from any supported format (auto-detect by extension)"""
    if path.endswith(".safetensors"):
        return _load_safetensors(path)
    elif path.endswith(".npz"):
        return _load_npz(path)
    else:
        raise ValueError(f"Unknown format: {path}")
```

This approach:

- Defaults to safetensors for better performance on new caches
- Can read existing `.npz` caches without migration
- Training code never knows or cares about the underlying format
- Easy to add format support in one place

### Pipeline Architecture

The key innovation is a **4-stage asynchronous pipeline**:

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────────┐
│  Thread 1    │   │  Thread 2    │   │  Thread 3    │   │  Thread 4  │
│  LOAD IMAGE  │──▶│ RESIZE/CROP  │──▶│ VAE ENCODE   │──▶│ SAVE DISK  │
│  (Disk I/O)  │   │ (CPU)        │   │ (GPU)        │   │ (Disk I/O) │
└──────────────┘   └──────────────┘   └──────────────┘   └────────────┘
      │                  │                  │                  │
      ▼                  ▼                  ▼                  ▼
  Queue(16)          Queue(16)          Queue(8)           Output
```

### Why 4 Threads?

Each stage has a different bottleneck:

| Stage  | Bottleneck | Typical Time | Notes                       |
| ------ | ---------- | ------------ | --------------------------- |
| Load   | Disk I/O   | 5-50ms       | SSD vs HDD matters          |
| Resize | CPU        | 2-10ms       | Can be parallelized further |
| Encode | GPU        | 5-15ms       | Batched for efficiency      |
| Save   | Disk I/O   | 2-20ms       | Shouldn't block GPU         |

With pipelining:

- GPU is never waiting for disk I/O
- Disk loading and saving happen simultaneously
- Total throughput limited only by slowest stage

### Justification

- **Current**: 4 it/s because GPU waits for I/O
- **Pipelined**: 100+ it/s because GPU runs continuously
- **Multi-GPU**: Each GPU gets its own encoder thread, loaders/savers are shared

### Batching Within Pipeline

The VAE encoding stage should batch images:

- Collect 4-8 images of same resolution
- Encode as single batch
- Improves GPU utilization

Resolution changes force a batch flush (can't batch mixed resolutions).

### Implementation Notes

- Use `queue.Queue` with `maxsize` to bound memory usage
- Graceful shutdown with sentinel values
- Progress bar on the encoder stage (the bottleneck)
- Resume support: check if cache file exists before adding to pipeline

---

## Phase 3: Epoch Preparation

**Purpose**: Generate shuffled batch order for one epoch, applying all per-epoch randomness.

**When it runs**: Once at the start of each epoch.

**Input**:

- Dataset manifest
- Epoch number (for deterministic seeding)
- Training config (caption dropout, shuffle, etc.)

**Output**: `epoch_N_manifest.json`:

```json
{
  "epoch": 5,
  "seed": 42,
  "total_batches": 5000,
  "batches": [
    {
      "batch_id": 0,
      "bucket_reso": [768, 1024],
      "image_ids": ["img_00042", "img_01337", "img_00888"],
      "processed_captions": [
        "a cat sitting on a windowsill",
        "",  // caption dropout applied
        "dog, outdoor, happy"  // shuffled tags
      ],
      "input_ids_1": [[101, 345, ...], ...],
      "input_ids_2": [[201, 567, ...], ...]
    }
  ]
}
```

### What Gets Computed Here

1. **Shuffle within buckets**: Images in each bucket shuffled with epoch-specific seed
2. **Caption processing**:
   - Caption dropout (some captions → empty string)
   - Tag shuffle (comma-separated tags randomized)
   - Wildcard resolution (`{cat|dog}` → pick one)
   - Prefix/suffix application
   - Token warmup (limit tags based on step, precomputed)
3. **Tokenization**: Run tokenizer on processed captions, store input IDs
4. **Batch assembly**: Group images into batches respecting bucket boundaries
5. **Batch shuffle**: Interleave batches from different buckets

### Justification

- **Deterministic**: Same epoch + seed = same batch order (reproducibility)
- **Fast training**: No caption processing during training loop
- **Feature preservation**: All current caption features supported
- **Inspectable**: Can examine exact batches that will be trained

### Memory Considerations

For 100k images with avg 50 tokens each:

- Input IDs: ~20MB (stored as int16)
- Captions: ~10MB (strings)
- Total manifest: ~50MB

This is loaded once at epoch start, not per-batch.

### Implementation Notes

- Support for validation split (separate manifest or flag per batch)
- Token warmup requires knowing total steps, passed as parameter
- Caption dropout rate applied stochastically per caption

---

## Phase 4: Training Data Loading

**Purpose**: Feed pre-prepared batches to GPU as fast as possible.

**When it runs**: Every training step.

**Input**:

- Epoch manifest from Phase 3
- Latent cache directory

**Output**: PyTorch tensors ready for model forward pass

### Design: Simple IterableDataset

```python
from safetensors import safe_open

class EpochDataset(IterableDataset):
    def __init__(self, epoch_manifest_path, latent_cache_dir):
        with open(epoch_manifest_path) as f:
            self.manifest = json.load(f)
        self.cache_dir = latent_cache_dir

    def __iter__(self):
        for batch in self.manifest['batches']:
            # Load latents using safetensors
            latents = []
            for img_id in batch['image_ids']:
                path = f"{self.cache_dir}/{img_id}.safetensors"
                with safe_open(path, framework="pt", device="cpu") as f:
                    latents.append(f.get_tensor("latents"))

            yield {
                'latents': torch.stack(latents),
                'input_ids': torch.tensor(batch['input_ids_1']),
                # ... other fields
            }
```

### DataLoader Configuration

```python
train_dataloader = DataLoader(
    dataset,
    batch_size=None,        # Dataset yields complete batches
    num_workers=4,          # Parallel file loading
    pin_memory=True,        # Fast GPU transfer
    prefetch_factor=2,      # Keep 2 batches ready
    persistent_workers=True # Don't respawn workers
)
```

- **Minimal `__getitem__` work**: Just `safe_open()` + `get_tensor()`
- **True parallelism**: `num_workers` actually helps because work is I/O-bound
- **No Python GIL contention**: safetensors releases GIL during load
- **Predictable memory**: Batch sizes predetermined, no surprises

### Latent Cache Format

Use `.safetensors` for memory-mapped loading and fast GPU transfer. Each file contains:

```python
# Tensors (accessed via safe_open)
{
    'latents': torch.Tensor,           # (4, H/8, W/8) float16
    'latents_flipped': torch.Tensor,   # Optional, for flip augmentation
}

# Metadata (JSON, accessed via f.metadata())
{
    'original_size': [1024, 768],      # For SDXL conditioning
    'crop_ltrb': [0, 0, 1024, 768],    # For SDXL conditioning
    'bucket_reso': [1024, 768],
}
```

Saving example:

```python
from safetensors.torch import save_file

save_file(
    {'latents': latent_tensor, 'latents_flipped': flipped_tensor},
    path,
    metadata={'original_size': '1024,768', 'crop_ltrb': '0,0,1024,768'}
)
```

### Performance Target

With 4 workers and SSD storage:

- File load: ~1ms per image
- Batch assembly: ~0.5ms
- GPU transfer: ~0.5ms

For batch size 4: ~8ms per batch = **125 batches/second**

This is faster than the model forward pass, so data loading is no longer the bottleneck.

---

## Memory Management

### Resolution-Aware Batch Sizing

Batch sizes are computed during Phase 1 (manifest generation):

```python
def compute_safe_batch_size(resolution, vram_budget_gb=20):
    """
    Estimate batch size that fits in VRAM.

    Memory usage scales with resolution:
    - Latents: O(resolution)
    - Attention: O(resolution²) - this is the killer
    - Gradients: O(resolution)
    """
    base_resolution = (512, 512)
    base_batch_size = 8

    pixels = resolution[0] * resolution[1]
    base_pixels = base_resolution[0] * base_resolution[1]

    scale = pixels / base_pixels
    attention_scale = scale ** 1.5  # Attention is superlinear

    return max(1, int(base_batch_size / attention_scale))
```

Example batch sizes for 20GB VRAM:

| Resolution | Batch Size |
| ---------- | ---------- |
| 512×512    | 8          |
| 768×768    | 4          |
| 1024×1024  | 2          |
| 512×1536   | 3          |

### Warmup Strategy

Process largest-resolution batches first in each epoch to establish CUDA memory allocation:

```python
# In Phase 3 (epoch preparation):
batches.sort(key=lambda b: -b['resolution'][0] * b['resolution'][1])
warmup_batches = batches[:10]  # Largest resolutions first
remaining = batches[10:]
random.shuffle(remaining)
final_order = warmup_batches + remaining
```

This ensures CUDA allocates for worst-case upfront, preventing random OOMs later in training.

---

## File Structure

```
sd-scripts/
├── tools/
│   ├── prepare_dataset.py      # Phase 1: Generate manifest
│   ├── cache_latents.py        # Phase 2: Pipeline caching (new implementation)
│   └── prepare_epoch.py        # Phase 3: Generate epoch batches
│
├── library/
│   └── data/
│       ├── manifest.py         # Manifest dataclass and I/O
│       ├── caching/
│       │   ├── pipeline.py     # 4-stage async pipeline
│       │   ├── stages.py       # Individual stage implementations
│       │   └── strategies/     # VAE/TE caching strategies per model
│       ├── loading/
│       │   ├── epoch_dataset.py    # IterableDataset implementation
│       │   └── prefetcher.py       # Optional advanced prefetching
│       └── legacy/             # Old implementation for migration period
│           ├── dataset.py
│           └── ...
│
├── configs/
│   └── data/
│       ├── default_caching.yaml
│       └── default_loading.yaml
```

### Implementation Notes

1. **Config Dataclass Location**: Consider adding `library/config/dataclasses/data_pipeline.py` for Hydra integration of caching/loading configs. This keeps config schema separate from implementation.

2. **Legacy Folder Naming**: Rather than `library/data/legacy/`, consider `library/data/_deprecated/` or keeping old files with `_v1` suffix. The underscore convention signals "don't import this directly."

3. **Tokenizer Validation**: When storing `input_ids` in epoch manifests (Phase 3), consider storing a tokenizer hash for validation. This catches issues if someone changes tokenizers between runs.

4. **CacheEntry vs ImageInfo**: We create a fresh `CacheEntry` dataclass (in `library/data/pipeline/dataclasses.py`) rather than evolving `ImageInfo`. The legacy class mixes static metadata with runtime state (tensors in memory), making it hard to serialize. `CacheEntry` stores **paths** to cache files, not tensors - tensors are loaded on-demand. Once the new pipeline is complete, the old data pipeline (including `ImageInfo`) moves to a legacy folder.

---

## Migration Path

### Phase 1: Caching Pipeline (Week 1)

**Scope**: Rewrite `cache_latents` with pipelined architecture.

**Backwards compatible**: No - outputs `.safetensors` format (migration tool provided).

**Immediate benefit**: 25x speedup on cache generation.

### Phase 2: Manifest System (Week 2)

**Scope**: Implement `prepare_dataset.py` and manifest dataclass.

**Backwards compatible**: Yes - can generate manifest from existing directories.

**Benefit**: Fast training startup, debuggable dataset configuration.

### Phase 3: Epoch Preparation (Week 3)

**Scope**: Implement `prepare_epoch.py` and epoch manifest.

**Backwards compatible**: No - training scripts need modification.

**Benefit**: All caption processing moved out of training loop.

### Phase 4: New DataLoader (Week 4)

**Scope**: Replace current Dataset/DataLoader with EpochDataset.

**Backwards compatible**: No - requires epoch manifest.

**Benefit**: 50x speedup on training data loading.

### Deprecation

After Phase 4 is stable (2-4 weeks of testing):

- Move old `dataset.py` to `library/data/legacy/`
- Add deprecation warnings
- Remove in next major version

---

## Open Questions

1. **Pre-batched files vs per-image files**: Should we save entire batches as single files? Pros: fewer file operations. Cons: less flexible reshuffling.

a: Starting with per-image. Flexible and not much overhead.

2. **Text encoder caching granularity**: Cache per-image or per-unique-caption? Captions may repeat across images.

a: Per-image for simplicity. Caption repetition is mainly reg images (limited overlap). Deduplication can be a future optimization if profiling shows significant repetition.

3. **Validation split handling**: Separate manifest or flag field? Need to ensure consistency across epochs.

a: Flag per image in main manifest (`"split": "train"` or `"val"`). Phase 3 generates separate epoch manifests for each. Simple and keeps validation consistent across epochs.

4. **Multi-GPU coordination**: How do workers distribute load? Current approach uses process index modulo.

a: Keep modulo approach for caching (simple, works). For training, let accelerator's DataLoader sharding handle distribution automatically.

5. **Streaming/partial datasets**: Support for datasets that don't fit on disk? Out of scope for initial implementation.

a: Out of scope for v1. Focus on local-disk performance first. Architecture supports future extension (manifest could reference remote URLs).

---

## Strategy Integration

The new pipeline engine lives in `library/data/` but delegates model-specific behavior to existing strategies in `library/strategies/`:

- **Generic engine** (`library/data/pipeline/`): Handles I/O, batching, multi-GPU coordination
- **Model-specific strategies** (`library/strategies/`): Provide VAE encoding, file formats, scale factors

The engine calls strategy methods as plugins:

```python
latents_strategy = peft_strategy.get_latents_caching_strategy(cfg)
engine = CachingEngine(latents_strategy)
engine.cache_dataset(dataset, vae, accelerator)
```

This preserves the current pattern where strategies define "what/how" while the new engine provides a faster "infrastructure".

---

## Success Metrics

| Metric                | Current                  | Target                | Measurement                         |
| --------------------- | ------------------------ | --------------------- | ----------------------------------- |
| Latent cache speed    | 4 it/s                   | 100 it/s              | Time to cache 10k images            |
| Training throughput   | Limited by data          | Limited by GPU        | GPU utilization during training     |
| Epoch start time      | ~30s                     | <2s                   | Time from epoch start to first step |
| Memory predictability | Random OOMs              | Zero OOMs             | Run 100 epochs without crash        |
| Code complexity       | 1300 lines in dataset.py | <300 lines per module | Line counts                         |

---

## References

- Original issue discussion: "DataLoader is 4it/s, manual caching is 300it/s"
- Current implementation: `library/data/dataset.py`
- Pipeline architecture inspiration: Producer-consumer patterns, ETL pipelines
