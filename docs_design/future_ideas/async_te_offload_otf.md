## Research Notes: Async Text-Encoder Prefetch for Offloaded On-the-Fly Conditioning

### Summary

Some training workflows need text-encoder conditioning to be computed **on the fly** rather than loaded from a fixed TE-output cache.

This is especially important when caption behavior changes per epoch or per step, such as:

* caption shuffling
* caption dropout
* tag dropout
* dynamic prompt weighting
* other augmentation-heavy caption policies

However, on-the-fly TE encoding can become expensive, especially when text encoders are offloaded to CPU to save VRAM.

The goal of this research direction is:

> preserve true on-the-fly text-conditioning semantics while reducing GPU idle time through asynchronous TE prefetching.

This is a narrower and more practical subtopic of the broader asynchronous data-pipeline work.

---

## Current Behavior

The current offload path has a specific meaning:

* `offload_text_encoders` keeps text encoders on CPU between uses to reduce VRAM.
* Offload is only valid with on-the-fly TE encoding, not TE-output caching.
* TE training with offload is currently blocked by config validation.
* Frozen TEs + offload + no TE cache means text conditioning is encoded on CPU, then transferred to the GPU for the denoiser step. 

This saves VRAM, but it can shift the bottleneck to CPU text encoding and host-to-device transfer.

---

## Why This Matters

There is an important tradeoff between TE caching and on-the-fly TE encoding.

### TE caching

Benefits:

* fastest during training
* avoids repeated TE computation
* simple runtime path

Costs:

* locks in caption augmentation results
* cannot naturally preserve per-step or per-epoch caption variation
* can become very large if many caption variants are cached

### On-the-fly TE encoding

Benefits:

* preserves true caption augmentation behavior
* supports deterministic per-epoch caption changes
* avoids storing many TE-output variants

Costs:

* expensive if TE runs every step
* CPU TE encoding can bottleneck GPU training
* offloaded TE path requires host-to-device transfer of embeddings

So for augmentation-heavy jobs, on-the-fly conditioning may be the correct semantic path, but it needs better execution overlap. 

---

## Key Architectural Idea

The epoch manifest already defines the upcoming sample and caption order.

That means the system can precompute text embeddings for upcoming batches **before** the GPU training loop needs them.

Instead of:

```text
training loop asks for batch
  -> encode text
  -> transfer embeddings
  -> run denoiser step
```

use:

```text
CPU TE producer
  -> encode upcoming captions
  -> place embeddings in pinned-memory queue
  -> GPU training loop dequeues ready embeddings
  -> run denoiser step
```

The text encoder still runs on the true augmented captions, but its cost is overlapped with GPU training where possible.

This is the core idea behind `async_te_prefetch`. 

---

## Relationship to Other Future Data-Layer Notes

### Relationship to async data pipeline

This is a concrete sub-pipeline inside the broader async data-layer direction.

The larger async pipeline concerns:

* image loading
* decoding
* resizing
* caching
* registry readiness
* training consumption

This note focuses specifically on:

* text conditioning preparation
* frozen/offloaded text encoders
* CPU/GPU overlap for TE outputs

### Relationship to cached TE-output augmentation

The cached-TE augmentation topic asks:

> can we avoid re-encoding by manipulating cached embeddings?

This note asks:

> can we keep real re-encoding semantics but hide some of the cost through async prefetch?

Those are complementary paths.

Cached TE augmentation is more storage/runtime efficient but approximate.
Async TE prefetch is more semantically faithful but depends on CPU throughput and queueing.

### Relationship to large-scale caching

Hybrid bounded TE caches and epoch-ahead pre-encoding connect this topic to the larger cache architecture.

Instead of caching all possible TE variants, the system may eventually support:

* partial TE caches
* LRU TE caches
* epoch-local TE caches
* next-epoch TE prefetch buffers
* shard-aware TE output storage

---

## Proposed First Path: Async Queued TE Encoding

The most actionable first implementation is an async producer/consumer queue.

### Producer side

A CPU worker:

* reads upcoming epoch-manifest entries
* resolves captions and augmentation state
* batches captions for TE encoding
* runs frozen text encoders on CPU
* places encoded outputs into a queue

### Consumer side

The training loop:

* requests the next batch
* dequeues precomputed TE outputs
* transfers embeddings to GPU if needed
* runs the denoiser/training step

### Queue behavior

The queue should likely support:

* bounded max depth
* pinned-memory tensors where useful
* timeout/fallback behavior
* graceful shutdown
* epoch-boundary signaling
* error propagation from worker to trainer

This is conceptually compatible with the current epoch-manifest flow and can reduce GPU idle time if CPU TE prep keeps up. 

---

## CPU Microbatching

CPU TE encoding should likely use a larger batch size than the GPU training batch.

Example:

* CPU encodes 8–16 samples at once
* GPU consumes training batches of 1–2 samples
* queue stores split outputs for upcoming train batches

This can improve CPU-side TE throughput by amortizing model-forward overhead.

### Benefits

* better CPU TE utilization
* fewer small TE calls
* smoother queue filling

### Costs

* more queue memory
* more bookkeeping
* more boundary handling around epoch end / distributed shards

This is one of the main knobs for making async prefetch viable. 

---

## Epoch-Ahead Pre-Encoding

Another possible strategy is to pre-encode epoch `N+1` while training epoch `N`.

This can work if the epoch manifest fully determines caption augmentation.

Requirements:

* caption augmentation must be deterministic from epoch manifest data
* cache/prep key must include augmentation-relevant state
* epoch boundaries must be handled carefully

This is less dynamic than a step-ahead queue, but may provide smoother throughput if CPU encoding is slower than desired. 

---

## Hybrid Bounded TE Cache

A hybrid path could use a bounded TE-output cache rather than a full dataset cache.

Possible behavior:

* start with on-the-fly encoding
* populate a small cache in the background
* reuse cached TE outputs when repeated captions/augmentation states recur
* fall back to async queue when cache misses

This may reduce repeated CPU TE computation without requiring full multi-variant TE storage.

Potential cache scopes:

* per-epoch
* per-run
* LRU memory cache
* small disk-backed cache
* caption-hash + augmentation-state keyed cache

The goal is to avoid all-or-nothing TE caching. 

---

## Embedding-Space Perturbation as an Optional Research Path

If true on-the-fly TE encoding is still too expensive, another path is to use one cached embedding and apply controlled perturbations:

* small noise
* token dropout
* span dropout
* mixup
* per-token reweighting

This is storage-friendly, but weaker than true text-level augmentation plus re-encoding. It should be treated as optional regularization, not a full semantic replacement for on-the-fly TE. 

---

## Throughput Reality Check

Queue overlap helps, but it cannot beat the slowest stage.

A useful approximation is:

```text
effective_step_time ≈ max(unet_step_time, te_prep_time)
```

If denoiser training is slower than TE prep, the queue can hide most TE cost.

If TE prep is slower than denoiser training, the GPU will eventually wait for the queue.

Possible responses to queue underflow:

* increase CPU TE microbatch size
* increase queue depth
* simplify or shorten TE workload
* reduce token length if valid
* pre-encode farther ahead
* use a second GPU for TE prep if available

The async queue smooths stalls and improves overlap, but it does not eliminate the underlying throughput limit. 

---

## Device and Execution Notes

Important device assumptions:

* TE encoding batch means an actual model-level batch forward, not just Python threading.
* CPU TE prep may be viable for SDXL frozen-TE workflows, but it is hardware-dependent.
* Running TE prep on a second GPU is also valid and may reduce stall risk.
* On single-GPU systems, CPU is usually the only practical overlap engine for this path.  

The implementation should avoid moving text encoders back and forth constantly in a way that destroys the benefit of offload.

---

## Determinism and Correctness Requirements

Because this path interacts with epoch manifests and caption augmentation, correctness matters as much as throughput.

The implementation must ensure:

* the queued embeddings match the exact batch/sample order consumed by training
* caption augmentation state matches the epoch manifest
* distributed ranks consume the correct shard/order
* epoch boundaries do not leak stale embeddings
* queue worker failure does not silently corrupt conditioning
* fallback paths preserve correctness

The initial success metric should be throughput gain with no data-order or conditioning mismatch. 

---

## Observability Requirements

This feature needs explicit runtime observability.

Useful metrics:

* queue occupancy average
* queue occupancy p95
* queue underflow count
* queue wait time
* TE encode throughput
* TE microbatch size
* host-to-device transfer time
* GPU training step time
* samples/sec
* GPU utilization

The validation checklist should compare baseline vs async-prefetch on:

* step time
* samples/sec
* GPU utilization
* deterministic behavior
* epoch-boundary correctness
* fallback behavior when worker lags or fails. 

---

## API and Ownership Questions

A planning artifact for `async_te_prefetch` should answer:

### Ownership

Where does this live?

Possible owners:

* trainer phase
* strategy conditioning layer
* dataloader / dataset layer
* new conditioning prefetch service

The best answer is likely a small runtime service that sits between epoch manifest/batch planning and strategy-specific conditioning resolution.

### Strategy involvement

Strategies may need to define:

* how captions become TE inputs
* what TE output payload shape is expected
* how pooled outputs are represented
* how SD vs SDXL conditioning differs

### Trainer involvement

Trainer likely owns:

* lifecycle start/stop
* epoch boundary handling
* distributed synchronization
* fallback and error handling
* metrics logging

### Data layer involvement

Data layer likely owns:

* epoch manifest iteration
* sample ordering
* caption augmentation state
* batch grouping

---

## Open Research Questions

* Can CPU TE encoding keep up with typical denoiser training steps?
* What queue depth is enough to smooth stalls without wasting memory?
* What CPU microbatch size gives the best throughput?
* Should TE prefetch operate batch-by-batch, epoch-ahead, or both?
* How should TE outputs be keyed if a bounded hybrid cache is used?
* Can pinned memory materially reduce transfer overhead for this path?
* How should distributed ranks coordinate TE prefetch and epoch boundaries?
* Should SDXL dual-encoder outputs be prefetched together or separately?
* What fallback behavior is safest if the queue under-runs?
* Is second-GPU TE preparation worth supporting as a first-class mode?

---

## Recommended Implementation Path

### Phase 1 — Benchmark baseline

Measure:

* on-the-fly CPU TE encode time
* denoiser train step time
* host-to-device transfer time
* baseline samples/sec

Before implementing queues, confirm whether overlap is likely to help.

### Phase 2 — Minimal async queue

Implement:

* single producer worker
* bounded queue
* epoch-manifest order
* CPU TE microbatching
* deterministic dequeue
* basic underflow logging

Goal:

* prove throughput improvement without changing semantics.

### Phase 3 — Pinned-memory and tuning

Add:

* pinned-memory queue option
* configurable queue depth
* configurable TE microbatch size
* queue occupancy metrics

Goal:

* tune overlap behavior.

### Phase 4 — Epoch-ahead / hybrid cache experiments

Add optional:

* next-epoch pre-encoding
* bounded TE cache
* cache keys including augmentation state

Goal:

* reduce repeated TE compute without full TE-output storage.

### Phase 5 — Advanced device placement

Explore:

* second-GPU TE prefetch
* CPU/GPU mixed TE prep
* deeper integration with broader async data pipeline

---

## Recommendation

Treat this as a practical near-term optimization for augmentation-heavy jobs where:

* text encoders are frozen
* TE outputs are not cached
* caption augmentation must remain semantically real
* VRAM pressure makes TE offload attractive

The first concrete feature should be:

> **`async_te_prefetch`: CPU-side TE encoding queue driven by epoch-manifest order, with pinned-memory transfer, queue metrics, and strict correctness checks.**

This is more immediately actionable than embedding-space research paths, while still fitting into the broader async data-pipeline and TE-cache design.

---

## Short Roadmap Version

* [ ] Research **async TE prefetch** for offloaded frozen text encoders
* [ ] Preserve on-the-fly caption augmentation semantics without full TE caching
* [ ] Add CPU producer / GPU consumer queue for upcoming TE outputs
* [ ] Use epoch manifest order as the deterministic source of truth
* [ ] Support CPU TE microbatching larger than GPU train batch size
* [ ] Investigate pinned-memory queue for faster host-to-device transfer
* [ ] Log queue occupancy, queue wait time, and underflow count
* [ ] Compare baseline vs async-prefetch step time, samples/sec, and GPU utilization
* [ ] Verify no conditioning/order mismatch at epoch boundaries
* [ ] Explore epoch-ahead pre-encoding and bounded hybrid TE cache later
* [ ] Treat embedding-space perturbation as optional research, not a replacement for true TE encoding
