# Benchmarking & Profiling Guide

This guide provides instructions for third-party reviewers to profile and benchmark the training pipeline, with focus on resource utilization and performance comparisons.

## Quick Start

```bash
# Run smoke test with basic timing
python scripts/sdxl_peft.py --config-name=smoke_test

# Run with Python profiling
python -m cProfile -o profile.prof scripts/sdxl_peft.py --config-name=smoke_test
```

---

## 1. Resource Monitoring

### GPU Utilization

**Option A: nvidia-smi (simple)**

```bash
# Real-time monitoring (1 second interval)
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv -l 1 > gpu_log.csv
```

**Option B: nvitop (recommended)**

```bash
pip install nvitop
nvitop --log gpu_metrics.csv
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

```bash
# Windows: Use Resource Monitor → Disk tab
# Linux: iostat -x 1
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

```bash
tensorboard --logdir=./profiler_logs
```

### cProfile (Simple Python Profiling)

```bash
python -m cProfile -s cumtime scripts/sdxl_peft.py --config-name=smoke_test 2>&1 | head -100
```

---

## 3. Data Pipeline Benchmarks

### DataLoader Throughput

```python
import time
from library.data.pipeline import create_training_dataloader, TrainingDataset

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
from library.data.pipeline import prepare_epoch
import time

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

### Legacy Baseline

```bash
# Run legacy script for comparison
python sdxl_train_network.py --config_file=legacy_config.toml
```

### Metrics to Compare

1. **Startup time** - Time from script start to first training step
2. **Per-step time** - Average time per optimization step
3. **Memory usage** - Peak GPU and RAM usage
4. **Cache generation time** - Time to cache full dataset
5. **Epoch transition time** - Time between epochs

---

## 6. Automated Test Suite

Run existing benchmarks:

```bash
# Unit benchmarks (mocked I/O)
pytest tests/unit/data/test_pipeline_benchmark.py -v

# Integration smoke tests
pytest tests/integration/test_sdxl_peft_smoke.py -v
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

## 9. Benchmark Results

### RTX 3090 · SDXL LoRA · 1024×1024 · Rank 32 · bf16 · xformers

**Environment:**

- GPU: NVIDIA RTX 3090 (24 GB VRAM)
- Dataset: 50 images × 1 repeat
- Resolution: 1024×1024
- LoRA Rank: 32, Alpha: 16
- Precision: bf16 mixed
- Optimizer: AdamW8bit

| #   | Batch | GA  | GC  | VRAM    | Virt Mem | s/step | img/s | Status      |
| --- | ----- | --- | --- | ------- | -------- | ------ | ----- | ----------- |
| 1   | 1     | 1   | ✓   | 12.5 GB | 76 GB    | 2.65   | 0.38  | ✅          |
| 2   | 2     | 1   | ✓   | 13.4 GB | 76 GB    | 2.33   | 0.86  | ✅ **Best** |
| 3   | 1     | 1   | ✗   | 21.4 GB | 89 GB    | 1.79   | 0.56  | ✅          |
| 4   | 2     | 1   | ✗   | ~29 GB  | -        | -      | -     | ❌ OOM      |
| 5   | 1     | 8   | ✗   | 21.6 GB | 88.5 GB  | 11.62  | 0.69  | ✅          |

_GA = Gradient Accumulation, GC = Gradient Checkpointing_

**Key Findings:**

- **Optimal for 24GB GPU:** Batch 2 + Gradient Checkpointing gives best throughput (0.86 img/s)
- **GC trade-off:** Disabling GC saves ~32% time per step but costs +8.9 GB VRAM
- **GA overhead:** Gradient accumulation 8 has ~17% lower throughput than batch 2 (0.69 vs 0.86 img/s)
