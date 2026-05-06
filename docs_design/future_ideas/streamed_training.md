## Research Notes: Training on Online / Streaming Data

### Summary

Training on online data means replacing the assumption of a fully prebuilt, static dataset with a pipeline where samples can be discovered, prepared, and made train-ready while the run is already in progress.

This can mean either:

* **streaming from remote storage** where the dataset is finite but not fully local/materialized up front, or
* **truly online / continuously arriving data** where new samples appear during training from uploads, generators, crawlers, active learning loops, etc.

For this repo, the most realistic and useful direction is probably **incremental dataset preparation**, not “pure live streaming with no persistent dataset structure at all.”

---

## Why This Matters

Potential benefits:

* train without requiring a full up-front preprocessing pass
* support remote or very large datasets
* support continuously arriving samples
* enable background caching / preparation while training continues
* open the door to more dynamic captioning, filtering, and synthetic-data workflows

Main challenge:

* the trainer cannot stall waiting on data preparation
* correctness and reproducibility become harder if sample state is not tracked explicitly
* “epoch” semantics become less clear once the dataset is no longer fixed

---

## Key Architectural Observation

The likely goal should **not** be “remove manifests entirely.”

A better direction is:

> move from a fully prebuilt static manifest
> to an **incrementally maintained sample registry** with explicit readiness/cache state

That preserves useful properties like:

* reproducibility
* cache identity
* deduplication
* bucket assignment
* dataset analytics
* future run-history integration

while still allowing online preparation and discovery.

---

## Likely Runtime Requirements

Online/streaming training usually involves concurrency because the pipeline contains multiple bottlenecks:

* remote fetch latency
* disk I/O
* image decode
* resize / bucketing
* caption/token transforms
* latent generation
* TE output generation
* training step execution

If all of those happen synchronously in the trainer loop, GPU utilization will suffer badly.

Likely building blocks:

* producer / consumer queues
* bounded buffers
* worker processes for CPU-heavy preparation
* background threads for coordination / I/O
* optional background cache writers
* backpressure handling so memory/disk usage stays bounded

---

## Useful Mental Model

### Cold path

Work that can happen ahead of consumption or in the background:

* sample discovery
* remote fetch
* validation
* metadata extraction
* dimension inspection
* bucket assignment
* latent caching
* TE output caching
* filtering / dedup
* registry updates

### Hot path

Work that must stay close to training time:

* selecting ready samples
* lightweight caption mutation / augmentation
* batching / collation
* moving tensors to device
* final trainer consumption

**Design goal:** keep the hot path as thin and predictable as possible.

---

## Recommended Future Architecture

### 1. `SampleSource`

Abstract source of candidate samples.

Possible source types:

* local folders
* static manifest input
* shard streams
* remote object store
* live ingest queue
* synthetic generator

Responsibilities:

* enumerate or yield candidate samples
* provide source metadata / identifiers
* not own preparation state

### 2. `SampleRegistry`

Persistent record of known sample state.

Tracks per-sample information such as:

* stable sample ID
* source URI / origin
* content hash
* image metadata
* captions / text metadata
* bucket assignment
* cache readiness
* failure / retry state
* timestamps / versioning

This would act as the online equivalent of a manifest foundation.

### 3. `PreparationPipeline`

Background preparation pipeline that converts raw sample records into train-ready samples.

Possible stages:

* fetch / materialize
* validate
* decode / normalize
* inspect dimensions
* bucket assignment
* latent cache generation
* TE cache generation
* metadata enrichment

Should be able to advance a sample through explicit states.

### 4. `ReadyPool` / `BatchProvider`

Consumes only samples that are considered ready for the current training mode.

Responsibilities:

* choose samples from the ready pool
* enforce batching constraints
* preserve bucket compatibility
* handle balancing / replay policy
* keep trainer-facing API stable

---

## Proposed Sample Lifecycle

A sample should probably move through explicit states such as:

* `discovered`
* `fetched`
* `validated`
* `bucketed`
* `latent_cached`
* `te_cached`
* `ready`
* `failed`
* `quarantined`

Not every sample needs every cache stage, but explicit state transitions will help avoid hidden behavior differences.

---

## Queue / Concurrency Notes

### Likely concurrent areas

**Good worker-process candidates**

* image decoding
* resize / preprocessing
* metadata extraction
* caption processing
* tokenization
* latent preparation
* TE output generation

**Good thread/background-task candidates**

* remote prefetch
* local cache writes
* queue monitoring
* registry flushing
* asynchronous cleanup / bookkeeping

### Why queues matter

Queues provide:

* **decoupling** between fetch/prep and training
* **smoothing** across temporary stalls
* **backpressure** so upstream work pauses when downstream is saturated

Backpressure is important. Without it, an online system can easily grow unbounded in RAM, disk, or queued work.

---

## Diffusion-Specific Considerations

For this repo, the important diffusion-specific prep steps likely include:

* dimension inspection
* bucket assignment
* latent caching
* TE output caching
* caption mutation / weighting interactions
* regularization image handling
* readiness rules based on current training mode

A likely per-sample flow would be:

```text
incoming sample
  -> inspect metadata and dimensions
  -> assign bucket
  -> register persistent sample state
  -> optionally create latent cache
  -> optionally create TE cache
  -> mark sample ready
  -> ready pool / batch selection
  -> training step
```

This suggests that the existing manifest/cache direction should evolve toward a **persistent registry + cache-state system**, rather than being discarded.

---

## Epoch / Sampling Semantics Follow-Up

A static dataset gives clear epoch semantics. Online data does not.

Future design will likely need a deliberate policy for one of these modes:

* **step-based training only**
* **snapshot epochs** over the currently known ready set
* **windowed epochs**
* **replay-buffer sampling**
* **age-aware or source-aware balancing**

This needs explicit thought because many trainer assumptions depend on dataset size and fixed ordering.

---

## Important Problems to Solve

### 1. Fairness / recency bias

New samples may dominate if sampling is purely arrival-order-based.

Possible solutions:

* replay buffer
* age-aware weighting
* per-source balancing
* class/source quotas

### 2. Duplicate control

Online sources often resend or regenerate near-duplicates.

Possible solutions:

* content hashes
* persistent source IDs
* dedup windows
* optional future similarity-based checks

### 3. Partial readiness

Some samples may be known but not fully prepared.

Need:

* explicit readiness states
* trainer only consumes `ready` samples
* batcher must ignore partial / failed states

### 4. Failure isolation

Bad samples should not destabilize the run.

Need:

* retry policies
* per-sample failure tracking
* quarantine / dead-letter handling

### 5. Backpressure

Without bounded queues and readiness controls, background prep can outrun the system.

Need:

* bounded queue sizes
* pause/slow discovery when buffers are full
* separate hot-path and cold-path priorities

---

## Recommended First Implementation Direction

Best first step is likely:

> **incremental registry + background preparation**, not full live-streaming chaos

Proposed behavior:

1. New samples may be discovered during training.
2. They are registered persistently.
3. Background workers validate / inspect / bucket / cache them.
4. Only samples marked `ready` enter the batch provider.
5. Training continues from the current ready pool without directly blocking on prep.

This would preserve the current training architecture better than trying to make the trainer itself handle arbitrary live preparation synchronously.

---

## Suggested Future Scope Split

### Phase 1 — Incremental local/remote preparation

* support lazy sample discovery
* add persistent sample registry
* add explicit sample state model
* background prepare into ready pool
* trainer consumes ready samples only

### Phase 2 — Streaming / remote source support

* remote object source abstraction
* bounded prefetch queues
* local cache materialization policy
* failure/retry and dedup rules

### Phase 3 — Truly online / continuously arriving data

* live ingest queue
* dynamic balancing policy
* replay buffer / age-aware sampling
* step-based or snapshot-epoch semantics

### Phase 4 — Advanced online data-flow experimentation

* on-the-fly caption mutation pipelines
* online TE/latent caching policy
* async CPU encoding or hybrid hot/cold conditioning paths
* integration with future run-history / telemetry systems

---

## Open Research Questions

* What should be the persistent identity of a sample in the online path?
* How much of the current manifest structure should become a registry schema?
* Which cache states are required before a sample is considered `ready`?
* Should training remain epoch-oriented, or become explicitly step-oriented in online mode?
* How should new samples be mixed with old ones to avoid recency bias?
* How should bucket planning behave when the ready set is constantly changing?
* How much preparation can safely happen on the hot path before GPU utilization suffers?
* Should online data be an alternate `SampleSource`, or a broader data-pipeline mode?

---

## Recommendation

Treat this as an evolution of the current data pipeline toward:

> **persistent sample registry + readiness states + background preparation + trainer consumption from a ready pool**

rather than as “no dataset structure, just stream samples directly into training.”

That direction seems much more compatible with:

* your manifest/caching architecture
* future cache invalidation rules
* future run telemetry / analytics
* reproducibility and debugging
* diffusion-specific bucket/caching constraints

---

## Short roadmap version

* [ ] Research **online / streaming data training** architecture
* [ ] Investigate **persistent sample registry** as an evolution of current manifest design
* [ ] Design **sample lifecycle states** (`discovered` → `ready` / `failed`)
* [ ] Explore **background preparation pipeline** with bounded queues and backpressure
* [ ] Keep trainer consumption limited to a **ready pool**
* [ ] Decide future **epoch / step semantics** for non-static datasets
* [ ] Explore remote-source and live-ingest support after incremental preparation path is stable
