# Offload + On-the-Fly TE Notes

Date: 2026-02-24  
Scope: Discussion recap after "what does offloading do in our repo?" focused on workarounds/ideas for practical training.

## Confirmed Current Behavior

1. `offload_text_encoders` keeps text encoders on CPU between uses to reduce VRAM.
2. Offload is only valid with on-the-fly TE encoding (not TE output caching).
3. TE training with offload is currently blocked by config validation.
4. Frozen TEs + offload + no TE cache means per-step CPU TE encoding, then transfer embeddings to GPU.

## Key Constraints We Identified

1. TE caching and on-the-fly augmentation are not equivalent:
   - TE caching is fast but does not preserve per-step caption augmentation dynamics.
   - On-the-fly preserves augmentation, but CPU TE encode can bottleneck GPU training.
2. On single-GPU systems, CPU is the only practical overlap engine for this path.
3. Multi-embedding caches can become too large for disk/RAM on big datasets.

## Feasible Workarounds / Ideas

## 1) Async queued TE encoding (preferred first path)

1. Keep epoch-manifest-driven sample order.
2. CPU producer pre-encodes upcoming batches.
3. GPU consumer dequeues embeddings and runs UNet step.
4. Use pinned memory queue for faster host-to-device transfer.

Notes:
1. This is conceptually compatible with current epoch manifest flow.
2. It reduces GPU idle time but cannot eliminate it if CPU TE throughput is still too slow.

## 2) CPU microbatch encode > train batch

1. Encode TE on CPU in larger chunks (e.g., 8-16 samples).
2. Split and feed GPU training batches (e.g., size 1-2).
3. Improves CPU-side throughput and amortizes per-call overhead.

Tradeoffs:
1. More queue memory.
2. More bookkeeping and boundary handling.

## 3) Epoch-ahead pre-encoding pipeline

1. Train current epoch while CPU pre-encodes next epoch's manifest.
2. Works only if the augmentation/caption path is deterministic from epoch manifest data.
3. Cache key should include augmentation-relevant state, not only image id.

## 4) Hybrid cache strategy

1. Start with on-the-fly encoding.
2. Populate a bounded cache (LRU/partial) in background.
3. Use cache hits when available, fall back to queue otherwise.

Goal:
1. Avoid full dataset embedding storage while still reducing repeated TE compute.

## 5) Embedding-space perturbation (optional research path)

1. Use one cached embedding and apply controlled perturbations (small noise/mixup/dropout).
2. Storage-friendly, but generally weaker than true text-level augmentation + re-encode.
3. Best treated as optional regularization, not a replacement for all augmentation.

## Practical Takeaways

1. For augmentation-heavy jobs, on-the-fly TE conditioning is the required semantics.
2. The most actionable optimization is an async TE prefetch queue integrated with epoch manifests.
3. Initial success metric should be throughput gain (e.g., step time reduction) with no data-order/conditioning mismatch.

## Throughput Reality Check

1. Queue overlap helps, but cannot beat the slower stage:
   - Effective step time is approximately `max(unet_step_time, te_prep_time)`.
2. A single-step-ahead queue may be enough when UNet fwd+bwd dominates.
3. If queue under-runs (GPU waits), increase TE prep throughput:
   - larger TE microbatches
   - deeper queue
   - lower TE workload (e.g., token length)

## Device/Execution Notes

1. TE encoding batch means model-level batch forward, not merely Python threads.
2. CPU TE prep can be viable for SDXL frozen-TE workflows, but hardware dependent.
3. Running TE prep on a second GPU is also valid and can further reduce stall risk.

## Validation Checklist (for planning + PR acceptance)

1. Log queue occupancy over time (`avg`, `p95`, `underflow count`).
2. Compare baseline vs async-prefetch:
   - step time
   - samples/sec
   - GPU utilization
3. Verify deterministic behavior against epoch manifest ordering.
4. Verify no conditioning mismatch at epoch boundaries.
5. Confirm fallback path works when queue worker fails or lags.

## Suggested Next Planning Artifact

Create an implementation plan for `async_te_prefetch` including:

1. API surface and ownership (trainer vs strategy vs dataloader).
2. Queue design (depth, pinned memory, timeout/fallback behavior).
3. Epoch boundary + distributed synchronization behavior.
4. Benchmarks and acceptance criteria.
