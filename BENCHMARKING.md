# Benchmarking & Profiling Guide

This guide provides instructions for third-party reviewers to profile and benchmark the training pipeline, with focus on resource utilization and performance comparisons.

## Quick Start

```powershell
# Run smoke test with basic timing
uv run scripts/sdxl_peft.py --config-name=smoke_test

# Run with Python profiling (dumps to profile.prof for later analysis)
uv run python -m cProfile -o profile.prof scripts/sdxl_peft.py --config-name=smoke_test
```

---

## 1. Resource Monitoring

### GPU Utilization

**Option A: nvidia-smi (simple)**

```powershell
# Real-time monitoring (1 second interval) - run in separate terminal
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv -l 1 | Tee-Object -FilePath gpu_log.csv

# Or save silently:
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv -l 1 > gpu_log.csv
```

**Option B: nvitop (recommended)**

```powershell
pip install nvitop
nvitop  # Interactive monitoring
# Note: --log flag may not work on Windows; use nvidia-smi for CSV logging
```

**Option C: PyTorch built-in**

```python
import torch
# Add to training loop:
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### CPU & RAM Monitoring

**Option A: psutil (Python)**

```python
import psutil
process = psutil.Process()
print(f"RAM: {process.memory_info().rss / 1e9:.2f} GB")
print(f"CPU: {psutil.cpu_percent()}%")
```

**Option B: Resource Monitor (Windows)**

1. Open Task Manager → Performance → Open Resource Monitor
2. Monitor: CPU %, Memory (Working Set), Disk I/O

### Disk I/O

```powershell
# Windows: Use Resource Monitor → Disk tab
# Or via PowerShell:
Get-Counter '\PhysicalDisk(_Total)\Disk Bytes/sec' -Continuous
```

---

## 2. Training Loop Profiling

### PyTorch Profiler (Recommended)

Add this wrapper to the training loop in `sdxl_peft.py`:

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step, batch in enumerate(train_dataloader):
        with record_function("forward_pass"):
            loss, _, _, _ = strategies.process_batch(...)
        with record_function("backward_pass"):
            accelerator.backward(loss)
        with record_function("optimizer_step"):
            optimizer.step()
        prof.step()
        if step >= 20:
            break

# View results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

View in TensorBoard:

```powershell
tensorboard --logdir=./profiler_logs
```

### cProfile (Simple Python Profiling)

```powershell
# Run profiling and save to file
uv run python -m cProfile -s cumtime -o profile.prof scripts/sdxl_peft.py --config-name=smoke_test

# View top functions afterward
uv run python -c "import pstats; p = pstats.Stats('profile.prof'); p.sort_stats('cumtime').print_stats(50)"
```

---

## 3. Data Pipeline Benchmarks

### DataLoader Throughput

```python
import time
from library.data import create_training_dataloader

# Measure batch loading speed
dataloader = create_training_dataloader(...)
start = time.perf_counter()
for i, batch in enumerate(dataloader):
    if i >= 100:
        break
elapsed = time.perf_counter() - start
print(f"Throughput: {100 / elapsed:.1f} batches/sec")
```

### Epoch Preparation Timing

```python
import time
from library.data import prepare_epoch

start = time.perf_counter()
epoch_manifest = prepare_epoch(manifest, seed=42, epoch=0, ...)
print(f"prepare_epoch: {time.perf_counter() - start:.3f}s for {len(epoch_manifest.batches)} batches")
```

### Cache Loading Speed

```python
from library.strategies.sdxl_caching import SdxlLatentsPipelineStrategy

strategy = SdxlLatentsPipelineStrategy(...)
cache_path = Path("cache/image_id_1024x1024_sdxl_latents.safetensors")

start = time.perf_counter()
for _ in range(100):
    cache_data = strategy.load_cache(cache_path, device="cpu")
print(f"Cache load: {(time.perf_counter() - start) / 100 * 1000:.2f}ms per file")
```

### Manifest Creation Timing

```python
import time
from library.data import create_manifest_from_config
from library.config.dataclasses.data import DataConfig

# Measure full manifest creation (scanning + bucketing)
start = time.perf_counter()
manifest = create_manifest_from_config(data_config, cache_dir="./cache")
elapsed = time.perf_counter() - start
print(f"Manifest creation: {elapsed:.2f}s for {manifest.image_count} images, {len(manifest.buckets)} buckets")
```

### VAE Caching Benchmark

```python
import time
from library.data import CachingEngine
from library.strategies.sdxl_caching import SdxlLatentsPipelineStrategy

# Measure full dataset caching time
strategy = SdxlLatentsPipelineStrategy(vae=vae, device=device, dtype=dtype)
engine = CachingEngine(strategy, accelerator, skip_existing=True)

start = time.perf_counter()
engine.cache_all(manifest)
elapsed = time.perf_counter() - start
print(f"VAE caching: {elapsed:.2f}s ({manifest.image_count / elapsed:.1f} images/sec)")
```

---

## 4. Key Metrics to Report

### Training Performance

| Metric           | How to Measure                      | Target          |
| ---------------- | ----------------------------------- | --------------- |
| Steps/second     | `1 / (time per step)`               | >1.0 for SDXL   |
| GPU Utilization  | nvidia-smi                          | >90%            |
| GPU Memory Peak  | `torch.cuda.max_memory_allocated()` | <24GB for SDXL  |
| Batch Throughput | DataLoader timing                   | >50 batches/sec |

### Data Pipeline Performance

| Metric                 | How to Measure                   | Target             |
| ---------------------- | -------------------------------- | ------------------ |
| `prepare_epoch()` time | Direct timing                    | <2s for 10k images |
| First batch latency    | Time to first `next(dataloader)` | <1s                |
| Cache load time        | Per-file timing                  | <5ms per file      |
| Token file load        | `load_epoch_tokens()` timing     | <1s for 10k images |

### Memory Footprint

| Component            | How to Measure           | Notes                  |
| -------------------- | ------------------------ | ---------------------- |
| Manifest (RAM)       | `sys.getsizeof()` + deep | ~2KB per entry         |
| Token file (disk)    | File size                | ~1KB per image         |
| Latent cache (disk)  | File size                | ~1MB per 1024x1024     |
| Epoch manifest (RAM) | Object size              | Regenerated each epoch |

---

## 5. Comparison Checklist

When comparing new pipeline vs legacy:

- [ ] Same dataset (images, captions, repeats)
- [ ] Same model configuration
- [ ] Same batch size and gradient accumulation
- [ ] Same number of workers
- [ ] Same seed for reproducibility
- [ ] Fresh cache (delete existing before each run)
- [ ] Same precision (bf16/fp16/fp32)
- [ ] Same gradient checkpointing setting

### New Pipeline (SDXL)

```powershell
# Clear cache first
Remove-Item -Recurse -Force ./cache -ErrorAction SilentlyContinue

# Run with timing
$start = Get-Date
uv run scripts/sdxl_peft.py --config-name=benchmark_test
$elapsed = (Get-Date) - $start
Write-Host "Total time: $($elapsed.TotalSeconds)s"
```

### Legacy Baseline (SD 1.x - still using old pipeline)

```powershell
# SD script still uses legacy DatasetGroup
$start = Get-Date
uv run scripts/sd_peft.py --config-name=sd_benchmark_test
$elapsed = (Get-Date) - $start
Write-Host "Total time: $($elapsed.TotalSeconds)s"
```

> **Note:** `sd_peft.py` still uses the legacy `prepare_datasets()` from `library/data/_deprecated/`.
> After migrating SD to the new pipeline, this comparison becomes SDXL new vs SDXL old.

### Metrics to Compare

| Metric                    | How to Measure                                               |
| ------------------------- | ------------------------------------------------------------ |
| **Startup time**          | Time from script start to first training step                |
| **Per-step time**         | Average time per optimization step (from tqdm)               |
| **Memory usage**          | Peak GPU (`torch.cuda.max_memory_allocated()`) and RAM usage |
| **Cache generation time** | Time to cache full dataset (first run only)                  |
| **Epoch transition time** | Time between epochs (visible in logs)                        |
| **First batch latency**   | Time to get first batch from DataLoader                      |

---

## 6. Automated Test Suite

Run existing benchmarks:

```powershell
# Unit benchmarks (mocked I/O)
uv run pytest tests/unit/data/test_pipeline_benchmark.py -v

# Integration smoke tests
uv run pytest tests/integration/test_sdxl_peft_smoke.py -v
```

---

## 7. Reporting Template

Use this template for benchmark reports:

```markdown
## Benchmark Report: [Description]

**Environment:**

- GPU: [Model, VRAM]
- CPU: [Model, Cores]
- RAM: [Size]
- Python: [Version]
- PyTorch: [Version]
- CUDA: [Version]

**Dataset:**

- Images: [Count]
- Resolution: [Size]
- Repeats: [Value]

**Configuration:**

- Batch size: [Value]
- Gradient accumulation: [Value]
- Workers: [Value]
- Precision: [bf16/fp16/fp32]

**Results:**

| Metric          | Value  | Notes |
| --------------- | ------ | ----- |
| Startup time    | X.Xs   |       |
| Steps/second    | X.X    |       |
| GPU utilization | X%     |       |
| GPU memory peak | X.X GB |       |
| RAM usage       | X.X GB |       |

**Profiler Highlights:**

- Top 3 CUDA operations: ...
- Bottlenecks identified: ...
```

---

## 8. Known Issues / Windows Notes

- **DataLoader Worker Spawn**: Windows uses `spawn` not `fork`, causing ~8s delay per worker initialization. Set `num_workers=0` for faster startup during profiling.
- **Console Redirects Warning**: `torch.distributed.elastic` logs warnings about redirects on Windows - these are harmless.
- **Memory Fragmentation**: On long runs, use `torch.cuda.empty_cache()` periodically.

---

## 9. Benchmark Results (SDXL New Pipeline)

**Environment:**

- **Script:** `sdxl_peft.py` (New Pipeline)
- **GPU:** RTX 3090 (24GB)
- **Dataset:** 550 Real Images (Tests/Assets)
- **Config:**
  - Batch Size: 2
  - Gradient Accumulation: 1
  - Gradient Checkpointing: True
  - Precision: bf16 (mixed) + no_half_vae
  - Optimizer: AdamW8bit
  - LoRA Rank: 32 / Alpha: 16

### Common Configuration

- `cache_latents_to_disk`: True
- `cache_text_encoder_outputs_to_disk`: True
- `vae_batch_size`: 2
- `caching num_workers`: 4
- `loader num_workers`: 0 (Training)
- `persistent_workers`: False
- `prefetch_factor`: 2
- `pin_memory`: True
- `offload_text_encoders`: False
- `xformers`: True

### Phase 1: Before VRAM Fix (Memory Fragmentation Bug)

| Phase              | Metric   | Value      | Peak/Range     | Notes                             |
| :----------------- | :------- | :--------- | :------------- | :-------------------------------- |
| **Latent Caching** | Speed    | 3.68 it/s  | -              | I/O Bound (30MB PNGs)             |
|                    | GPU Mem  | -          | 6.3 - 13.4 GB  | Memory accumulated across buckets |
|                    | Phys Mem | -          | 50.6 GB        |                                   |
|                    | Virt Mem | -          | 76.7 - 84.7 GB | Swapping to shared memory         |
| **TE Caching**     | Speed    | 34.51 it/s | -              | Fast (Disk Write Bound)           |
|                    | GPU Mem  | -          | 4.8 GB         |                                   |
| **Training**       | Speed    | 1.66 s/it  | 0.60 it/s      |                                   |
|                    | GPU Mem  | -          | 9.6 GB         |                                   |

### Phase 2: After VRAM Fix (torch.cuda.empty_cache between buckets)

| Phase              | Metric  | Value      | Peak/Range   | Notes                                         |
| :----------------- | :------ | :--------- | :----------- | :-------------------------------------------- |
| **Latent Caching** | Speed   | 3.22 it/s  | 2:50 total   | Slightly slower (empty_cache overhead)        |
|                    | GPU Mem | -          | 4.9 - 7.5 GB | **62% reduction** - bounded by largest bucket |
| **TE Caching**     | Speed   | 29.32 it/s | 0:18 total   | Fast                                          |
|                    | GPU Mem | -          | ~4 GB        |                                               |
| **Training**       | Speed   | 1.91 s/it  | 0.52 it/s    |                                               |
|                    | GPU Mem | -          | ~9 GB        | Comfortable fit                               |

**Total Time:** 319.66s (5.3 min)

### Key Findings

1. **VRAM fix reduced peak memory by 62%** (13.4GB → 7.5GB peak)
2. **Latent caching still I/O bound** at 3.22 it/s vs target 100+ it/s
3. **cProfile overhead ~30%** - profiled runs are slower (2.53 s/it vs 1.91 s/it)
4. **Primary bottleneck:** `.cpu()` calls (90s) - async disk writes would help
