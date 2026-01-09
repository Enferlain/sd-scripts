# SDXL PEFT Data Pipeline Audit

## 1. Overview
This document audits the current state of the SDXL PEFT data training pipeline, focusing on the flow from `scripts/sdxl_peft.py` through `library/strategies/` and `library/data/`. The pipeline adopts a "Pre-computed Epoch" design, where data is organized into static manifests and caching is heavily utilized to maximize training throughput.

## 2. Process Flow

### Phase A: Initialization
*   **Script**: `scripts/sdxl_peft.py`
*   **Action**: Initializes `SdxlPeftStrategy` and loads the configuration.
*   **Key Concept**: The script uses a strategy pattern (`SdxlPeftStrategy`) to delegate model-specific logic (loading, encoding, processing) while sharing generic training loop infrastructure.

### Phase B: Manifest Creation
*   **Script**: `scripts/sdxl_peft.py` -> `library/data/manifest.py` (implied)
*   **Action**: Calls `create_manifest_from_config` or `get_or_create_manifest`.
*   **Function**:
  *   Scans the dataset directory.
  *   **Bucketing**: Calculates `bucket_reso` for each image based on aspect ratios and config constraints (Phase 1 scanning). This resolution is stored in the manifest entry and persists throughout training.
  *   **Persistence**: If `get_or_create_manifest` is used, the manifest (including bucketing decisions) is saved to disk to allow quick resumption.
  *   **Validation Split**: A separate `val_manifest` is created or filtered from the main manifest.

### Phase C: Latent Caching
*   **Script**: `scripts/sdxl_peft.py` -> `library/strategies/sdxl_caching.py` -> `library/data/caching_engine.py`
*   **Strategy**: `SdxlLatentsPipelineStrategy`
*   **Flow**:
  1.  **VAE Loading**: VAE is moved to GPU.
  2.  **Engine Execution**: `CachingEngine.cache_dataset` is called.
  3.  **Parallel Loading**: Images are loaded in parallel using `ThreadPoolExecutor` (`num_workers`).
  4.  **Preprocessing**: Images are resized/normalized.
  5.  **Batching**: Images are grouped by `bucket_reso` to allow efficient VAE batching.
  6.  **Encoding**: `SdxlLatentsPipelineStrategy.encode_batch` encodes images to latents.
  7.  **SDXL Conditioning**: The strategy calculates and stores **Micro-Conditioning** metadata for each image:
      *   `original_size`: The original image dimensions.
      *   `crop_ltrb`: Left/Top/Right/Bottom crop coordinates (calculated via `get_crop_ltrb` to center-crop the image into the bucket resolution).
      *   `bucket_reso`: The target resolution (used as `target_size`).
  8.  **Saving**: Latents and metadata are saved to `.safetensors` files (one per image).
  9.  **Cleanup**: VAE is moved to CPU/cleared to free VRAM.

### SDXL Micro-Conditioning Metadata Format
Each cached image includes the following SDXL-specific metadata:
- `original_size`: Tuple (width, height) of the original input image
- `crop_ltrb`: Tuple (left, top, right, bottom) for center-crop coordinates
- `target_size`: Resolution tuple matching `bucket_reso` (e.g., (1024, 1024))

Example: An image of 2048×1024 bucketed to 1024×1024 would have:
- `original_size`: (2048, 1024)
- `crop_ltrb`: (512, 0, 1536, 1024)
- `target_size`: (1024, 1024)

### Phase D: Text Encoder Caching (Optional)
*   **Script**: `scripts/sdxl_peft.py` -> `library/strategies/sdxl_caching.py`
*   **Strategy**: `SdxlTextEncoderPipelineStrategy`
*   **Condition**: Enabled via `cfg.data.caching.cache_text_encoder_outputs`.
*   **Flow**:
  *   **Disk-Based**: Uses `CachingEngine` to encode captions with both CLIP-L and CLIP-G. Saves `hidden_state1`, `hidden_state2`, and `pool2` to `.safetensors`.
  *   **Memory-Based**: (Fallback) Computes outputs and stores them in-memory (`entry.te_outputs`) if disk caching is disabled but caching is requested.
  *   **On-the-fly**: If caching is disabled entirely, text encoding happens per-step in the training loop.

### Phase E: DataLoader Creation
*   **Validation**: `prepare_validation_epoch` creates a deterministic, sorted `EpochManifest`. `create_training_dataloader` creates a single loader reused across epochs.
*   **Training**: Created **per-epoch** inside the training loop (Phase G).

### Phase G: Epoch Preparation (The Training Loop)
*   **Script**: `scripts/sdxl_peft.py` -> `library/data/epoch_preparation.py`
*   **Action**: `prepare_epoch` is called at the start of every epoch.
*   **Flow**:
  1.  **Grouping**: Entries are grouped by `bucket_reso`.
  2.  **Shuffling**: Entries within each bucket are shuffled using a seed derived from `seed + epoch`.
  3.  **Batching**: `BatchInfo` objects are created by chunking the shuffled lists into `batch_size`.
  4.  **Caption Processing**: Captions are processed (wildcards, shuffle, dropout) and stored in `BatchInfo`.
  5.  **Warmup Ordering**: Largest resolution batches are optionally placed first (`warmup_largest_first=True`) to initialize CUDA memory allocators safely.
  6.  **Tokenization (Optional)**: `tokenize_epoch_manifest` can pre-tokenize all captions for the epoch into a single `.safetensors` file, enabling high-speed streaming during training.

### Phase H: Data Loading & Training
*   **Script**: `library/data/dataloader.py` (`TrainingDataset`)
*   **Flow**:
  1.  **Iteration**: `TrainingDataset` iterates through the pre-computed `EpochManifest`.
  2.  **Sharding**:
      *   **Distributed**: Batches are filtered via `idx % world_size == rank`.
      *   **Workers**: Batches are filtered via `idx // world_size % num_workers == worker_id`.
  3.  **Batch Loading**:
      *   Latents are loaded from `.safetensors` cache.
      *   **Flip Augmentation**: 50% chance to load `latents_flipped` if enabled.
      *   **Tokens**: Loaded from the pre-tokenized file (streaming or full load) OR text encoder outputs are loaded from cache.
      *   **Conditioning**: SDXL `original_size`, `crop_ltrb`, `target_size` are retrieved from cache metadata.
  4.  **Yielding**: Batches are yielded on CPU. `pin_memory=True` (in DataLoader) speeds up transfer to GPU.

## 3. Key Mechanisms

### Caching
*   **Format**: `.safetensors` (fast, zero-copy).
*   **Granularity**: Per-image.
*   **Validation**: `is_cache_valid` checks for existence, tensor shape matches (bucket consistency), and required keys (latents, flip, alpha).
*   **Invalidation**: Changing `bucket_reso` triggers re-caching because the latent shape must match the bucket.

### Bucketing
*   **Static**: Buckets are assigned during Manifest Creation (Phase 1).
*   **Dynamic Batching**: While buckets are static per image, the *batching* of those images is dynamic per epoch (Phase G) to allow shuffling.
*   **SDXL Support**: The pipeline explicitly handles SDXL's requirement for fixed bucket resolutions by storing `crop_ltrb` metadata, ensuring the model knows about the cropping applied to fit the bucket.

### Batching
*   **Pre-computed**: Unlike standard PyTorch `Sampler`, batching is fully resolved in `prepare_epoch` before the `DataLoader` starts.
*   **Efficiency**: This removes the overhead of bucket search/grouping from the training step loop.

### Saving & Loading
*   **Manifests**: Dataset manifests are persistent (json/toml).
*   **Cache**: Latent/TE caches are persistent.
*   **Tokens**: Epoch token files are ephemeral (created per epoch, deleted after).
*   **Checkpoints**: Standard model saving.

## 4. Performance Factors

### `num_workers`
*   **Caching**: Directly controls parallelism in `CachingEngine` (using `ThreadPoolExecutor`). Higher is better for fast initial caching (up to CPU core count).
*   **Training**: Controls `DataLoader` subprocesses.
    *   Since `TrainingDataset` does simple file reads (latents/tokens), high worker counts are less critical than in augmentation-heavy pipelines.
    *   *Recommendation*: 4-8 workers usually suffice to keep the GPU fed.

### `prefetch_factor`
*   **Effect**: Controls how many batches each worker loads in advance.
*   **Impact**: Helps smooth out I/O latency. Since batches are pre-computed, prefetching is highly effective.
*   **Warning**: Too high `prefetch_factor` with large `num_workers` can consume significant system RAM (storing loaded latent tensors).

### `persistent_workers`
*   **Effect**: Keeps DataLoader workers alive between epochs.
*   **Benefit**: Saves startup overhead (process creation) at the start of each epoch.
*   **Trade-off**: Holds memory between epochs. Since `TrainingDataset` is lightweight, this is generally recommended (`True`).

### `pin_memory`
*   **Effect**: Allocates page-locked memory for batches on CPU.
*   **Benefit**: Accelerates `batch.to(device)` transfer. Crucial for keeping high-performance GPUs (3090/4090/A100) utilized.

## 5. SDXL Specifics
*   **Micro-Conditioning**: The pipeline is fully SDXL-aware. It captures and passes `original_size`, `crop_top_left` (from `crop_ltrb`), and `target_size` (from `bucket_reso`) to the model during training.
*   **Dual Text Encoders**: Strategies handle dual CLIP-L/CLIP-G encoding, chunking (75+ tokens), and pooling (`pool2`).
