## Research Notes: Large-Scale Caching Architecture

### Summary

The current per-image caching approach is simple and works well for small to medium datasets, but it is likely to hit major scaling limits once dataset size moves into the **100k–1M+ image range**.

The main issue is not only raw cache size. At large scale, the system starts paying heavily for:

* too many filesystem objects
* too many file opens
* filesystem metadata overhead
* manifest / registry growth
* random small I/O instead of larger sequential reads

For this repo, the likely long-term direction should be:

> move from **per-image cache files + large JSON manifest**
> to **separate large-scale storage layers**:
>
> * a **persistent sample registry**
> * a **sharded latent cache**
> * a **sharded TE-output cache**

---

## Why This Matters

The current per-image layout is operationally convenient:

* simple mental model
* easy incremental updates
* easy debugging
* easy manual inspection

But at very large scale, this design likely becomes increasingly expensive even if total byte size is still manageable.

Likely bottlenecks include:

* filesystem overhead from very large file counts
* per-file open/close latency
* directory scaling issues
* random-access read patterns during training
* excessively large manifest JSON files
* slow startup / scan / lookup behavior

This means the problem is not just “cache storage is big.”
It is also that the current storage shape becomes increasingly inefficient.

---

## Key Architectural Observation

The **manifest / registry problem** and the **tensor-cache problem** should probably be treated as separate storage concerns.

A single giant JSON file is not an ideal long-term solution for sample metadata at 1M+ scale, and the best storage shape for metadata is not necessarily the best shape for latent or TE tensors.

So future design should likely split into:

### 1. Sample Registry / Manifest Store

Stores canonical metadata and cache-state information.

### 2. Latent Cache Store

Stores image latents in a sharded form.

### 3. TE Output Cache Store

Stores text-encoder outputs in a separately sharded and independently invalidated form.

This separation aligns with your current design instincts around:

* config-hash namespacing
* independent invalidation
* keeping the training-facing API stable

---

## Current Architecture

### Per-image cache layout

```text id="qka2r1"
cache/
├── img_001_sdxl_latents.safetensors
├── img_001_sdxl_te.safetensors
└── ...
```

### Strengths

* simple to understand
* easy to update incrementally
* easy to inspect manually
* failure scope is small and local to each sample

### Weaknesses

* file count grows linearly with dataset size
* random I/O dominates at large scale
* directory/file lookup overhead becomes significant
* difficult to scale cleanly on Windows/NTFS
* manifest JSON can become very large and expensive to load/update

---

## Proposed Direction

The long-term direction should likely be:

> **sharded cache payloads + explicit persistent registry**

Instead of:

* one cache file per image
* one large JSON manifest

use:

* a persistent registry for sample metadata/state
* larger shard payloads for latents
* larger shard payloads for TE outputs
* explicit shard indexing
* namespace-based invalidation instead of in-place mutation

---

## Storage Backend Notes

A big part of the design question is **which storage technology should own which problem**.

### Manifest / Registry storage options

#### JSON

Useful for:

* export
* debugging
* human inspection
* compatibility during transition

Not ideal as the canonical large-scale store because:

* full-file rewrite/update patterns get ugly
* point lookups are weak
* indexes are external/manual
* memory cost grows badly with giant manifests

#### SQLite

Probably the strongest default option for the **canonical registry**.

Good for:

* incremental updates
* point lookups
* sample-state tracking
* dedup keys
* bucket lookup
* shard membership mapping
* local-first workflows
* easy shipping inside the repo/tooling

This is the option I’d currently favor for the **manifest / registry side**.

#### Parquet

Best viewed as:

* analytics/export format
* efficient bulk scan format
* future stats/reporting companion

Good for:

* compact columnar storage
* filtered scans
* large exports
* future run/data analytics

Less ideal as the primary mutable registry if sample state changes frequently.

#### SQLite + Parquet hybrid

Probably the best long-term combination.

Use:

* **SQLite** as canonical mutable registry
* **Parquet** for exports, scans, analytics, offline summaries

This gives you a stable operational store without giving up future analytical flexibility.

---

### Tensor cache storage options

#### Per-image safetensors

Current approach.

Good for:

* simplicity
* debugging
* isolated failures

Bad for:

* file count
* filesystem pressure
* small random reads at scale

#### Sharded safetensors

Strong candidate if you want to stay close to current tensor tooling.

Good for:

* fewer files
* larger sequential reads
* lower open/close overhead
* continuity with existing tensor handling

Caution:

* shard-local lookup should still be explicit
* you do not want to parse the whole shard just to find one sample

#### Custom shard payload + explicit index

This is probably the most flexible design direction.

Shape:

* one shard payload file
* one shard index file or DB
* registry points to shard membership and row/offset

Good for:

* controlled lookup behavior
* easier evolution of the format
* explicit read-path optimization
* clean separation of payload vs metadata

This is the option I’d personally lean toward for the **latent/TE cache side**.

#### Zarr

Interesting research option, but not an automatic win for this exact problem.

Why it is attractive:

* chunked-array abstraction
* natural slicing/chunk-based reads
* good conceptual fit for large tensor storage

Why I would be cautious:

* plain filesystem-backed chunk stores can recreate the **too many files** problem if chunk count gets large
* may introduce more abstraction/integration surface than you want initially
* helpful if you want chunk semantics, but not necessarily the simplest first implementation for NTFS/file-count pain

So: **worth researching, not my first implementation choice** here.

#### LMDB / key-value store approach

Also a reasonable direction in theory.

Potential benefits:

* avoids file-per-sample explosion
* transactional-ish behavior
* single-store ergonomics

Tradeoffs:

* less transparent/debuggable than shard files
* more “database-owned blob store” feel
* stronger architectural commitment

This is viable, but I would still prefer **registry + explicit shard payloads** before jumping to a pure KV-backed cache store.

---

## Recommended Storage Split

### 1. Sample Registry / Manifest Store

This should become the canonical store for sample metadata and cache-state tracking.

Responsibilities:

* stable sample identity
* source/path/hash metadata
* image dimensions
* bucket assignment
* cache readiness state
* latent shard membership
* TE shard membership
* failure/retry state
* timestamps / versioning

This is effectively the large-scale evolution of the current manifest concept.

### Recommendation

Use **SQLite** as the canonical registry backend.
Optionally support **Parquet export** for analytics and large offline scans.

---

### 2. Latent Cache Store

This should store precomputed latents in a sharded structure keyed by latent-cache configuration.

Responsibilities:

* grouped storage of many samples per shard
* fewer file opens
* more sequential reads
* shard-local indexing for sample lookup
* namespace-based invalidation

### Recommendation

Use a **sharded payload + explicit index** design.
Sharded safetensors is plausible; custom payload + index may be even better long-term.

---

### 3. TE Output Cache Store

This should store text-encoder outputs independently from latents and keyed by caption identity plus TE config.

Responsibilities:

* deduplicate repeated captions
* independently invalidate from latent cache
* reduce repeated TE storage
* support multiple tokenizer / encoder configurations

### Recommendation

Use the same broad storage pattern as latents, but key entries by **caption hash + TE config hash**, not image identity.

---

## Registry Notes

A giant JSON manifest is likely not the right long-term store at very large scale.

A better direction is a **persistent registry** that supports:

* incremental updates
* point lookups
* shard membership tracking
* filtering by bucket/state/hash
* future analytics/export

Conceptually, the registry would replace “manifest as one big blob” with “manifest as a maintained state store.”

### Useful registry fields

A registry likely needs to track things such as:

* `sample_id`
* `source_uri` or source path
* `content_hash`
* `caption_hash`
* `width`
* `height`
* `bucket_key`
* `is_reg`
* `latent_namespace`
* `latent_shard_id`
* `latent_row`
* `te_namespace`
* `te_shard_id`
* `te_row`
* `state`
* `error_code`
* `created_at`
* `updated_at`

This would make the registry the canonical answer to:

* what samples exist
* what bucket they belong to
* what cache state they are in
* where their cached tensors live

---

## Sharded Cache Design

### Latent Cache

A likely future layout:

```text id="0pqye5"
cache/
├── latents/
│   └── {latent_config_hash}/
│       ├── bucket_1024x1024/
│       │   ├── shard_0000/
│       │   │   ├── payload
│       │   │   └── index
│       │   └── shard_0001/
│       └── bucket_768x1024/
│           └── shard_0000/
```

The key idea is:

* keep payload storage sharded
* keep shard membership/index explicit
* do not require opening the payload just to learn where a sample lives

### TE Output Cache

A likely future layout:

```text id="9l7wyt"
cache/
├── te_outputs/
│   └── {te_config_hash}/
│       ├── shard_0000/
│       │   ├── payload
│       │   └── index
│       └── shard_0001/
```

TE outputs should likely be keyed by **caption hash**, not image ID, so repeated captions can share the same cache entry.

---

## Design Principles

### 1. Config-Hash Namespacing

Changing cache-relevant config should create a new namespace instead of mutating existing cache in place.

Benefits:

* simpler invalidation
* easier crash recovery
* lower risk of partial corruption
* easier debugging of old/new cache generations

Likely latent namespace inputs:

* base resolution
* bucket steps
* no upscale / resize policy
* latent dtype
* model family / latent contract version

Likely TE namespace inputs:

* tokenizer identity/version
* text encoder identity/version
* max token length
* clip skip
* prompt-weighting / mutation policy version
* TE output contract version

### 2. Independent Invalidation

Latent and TE cache should invalidate independently.

This matters because:

* bucket-setting changes should not force TE rebuild
* caption/TE changes should not force latent rebuild
* cache rebuild cost should stay scoped to the changed concern

### 3. Caption-Hash Deduplication for TE

TE outputs should likely be keyed by normalized caption identity plus TE config, not by image identity.

Benefits:

* repeated captions share one cache entry
* storage growth is lower when captions repeat
* clearer identity model for TE caching

### 4. Keep Training-Facing API Stable

The training loop should ideally not care whether the backend is:

* per-image
* sharded
* future hybrid backend

The backend swap should happen behind shared cache-store interfaces.

---

## Shard Design Notes

### Separate payload from index

Each shard should probably have:

* a payload file containing the actual tensor data
* an index describing what is inside and where

The index should answer:

* which sample or caption hashes are present
* what row / offset corresponds to each entry
* what dtype / shape metadata applies
* shard-local stats/checksums/version info

This avoids needing to parse the full payload to answer simple membership questions.

### Prefer target shard size over target image count

It is tempting to define shard size in “images per shard,” but that may not be the best long-term tuning unit.

A better rule is likely:

* tune shards by **target payload size**
* let image count vary with latent size / bucket size / dtype

That makes the storage layout more portable across:

* different resolutions
* different model families
* different latent shapes
* different cache dtypes

### Append-only shard generations are safer

Future shards should probably prefer:

* append-only writes
* generation-based replacement
* explicit cleanup/compaction later

rather than:

* in-place mutation of existing shards

Append-only behavior aligns naturally with namespace-based invalidation and reduces correctness risks.

---

## Tokens and Token Caching

The current idea of keeping tokenization **on-the-fly by default** still seems sensible.

Reasons:

* tokenization cost is relatively small
* token caching creates additional invalidation complexity
* prompt weighting / caption mutation can easily destabilize token cache identity
* keeping tokens uncached reduces coupling between data pipeline and conditioning policy

So token caching likely makes more sense as:

* optional
* workflow-specific
* not the default large-scale path

---

## Why the Registry Matters as Much as the Cache

It is easy to focus on tensor storage and overlook the metadata side, but at very large scale the metadata structure becomes just as important.

The registry needs to support:

* incremental updates
* dedup checks
* shard lookup
* state transitions
* error tracking
* possible future analytics and run-history integration

If the system keeps a huge JSON manifest as its primary truth source, that becomes a likely scaling bottleneck even before tensor I/O is fully optimized.

So a large-scale cache design should probably be read as:

> **registry redesign + cache redesign**, not just “bundle more tensors into fewer files”

---

## Likely Runtime Flow

A likely future batch-read flow would look more like:

```text id="xghm3d"
batch selection
  -> registry lookup for selected sample IDs
  -> group selected entries by shard
  -> read only required rows/slices from each shard
  -> collate tensors into batch
  -> train
```

This is very different from:

* one file open per image
* one cache read per sample

and is where much of the performance gain should come from.

---

## Important Problems to Solve

### 1. Registry scale

Need a canonical registry that can grow incrementally without becoming a giant text blob.

### 2. Shard membership lookup

Need fast mapping from sample/caption hash to shard + row/offset.

### 3. Incremental writes

Need a clear policy for how partially filled shards are written and finalized.

### 4. Cleanup / compaction

Old namespaces and stale shard generations will accumulate unless cleanup rules exist.

### 5. Debuggability

Sharded systems are more efficient but less trivial to inspect manually than file-per-image layouts.

### 6. Crash recovery

Need clear guarantees for what happens when a write is interrupted mid-shard or mid-index update.

### 7. Batch locality

Need to decide how strongly sampling and batching should align with shard boundaries to reduce scatter-gather overhead.

---

## Recommended First Implementation Direction

The best first step is probably **not** “fully replace everything at once.”

A better path is:

### Phase 1

* introduce a `CacheStore` abstraction
* keep current per-image store as baseline implementation
* add explicit namespace model
* add persistent registry concept

### Phase 2

* move canonical manifest state into a real registry backend
* use **SQLite** as the default registry
* keep JSON export/debug path if useful
* optionally support Parquet export for large scans/analytics
* make batch lookup go through registry instead of giant manifest blob

### Phase 3

* implement `ShardedCacheStore` for latents
* shard by namespace + bucket
* add shard-local index
* start with explicit shard payload + index design rather than exotic backends

### Phase 4

* implement TE sharding with caption-hash dedup
* validate independent invalidation and dedup correctness

### Phase 5

* add migration / compaction / cleanup utilities
* add threshold-based recommendations and filesystem-aware defaults
* optionally evaluate **Zarr** as a future alternative if chunked-array semantics become especially attractive
* optionally evaluate KV-store approaches only if shard-file approaches prove too limiting

---

## Open Research Questions

* What should be the canonical registry backend at very large scale?
* Is **SQLite + Parquet export** enough, or is a heavier registry eventually needed?
* What is the best shard payload format for latent/TE data in this repo?
* Should initial implementation use sharded safetensors or a custom payload format?
* Is **Zarr** actually beneficial here, or would it just recreate chunk/file-count issues in another form?
* Should shard lookup live entirely in the registry, or partly in per-shard index files?
* How should partially written shards be handled safely?
* How much batch sampling should be shard-aware to reduce read amplification?
* What cleanup and compaction policy should exist for old namespaces?
* How should the large-scale cache design interact with future online/streaming data support?

---

## Recommendation

Treat large-scale caching as a **two-part storage redesign**:

> **persistent registry for sample metadata/state**
> plus
> **sharded tensor cache for latents and TE outputs**

rather than as a narrower “replace many small files with fewer big files” project.

And more concretely:

> **SQLite should likely be the default canonical registry backend**
> while the tensor cache should likely move toward
> **explicit shard payloads + explicit indexes**.

**Parquet** is likely useful for export/analytics.
**Zarr** is worth research, but not an automatic first choice for NTFS/file-count pain.
A pure **KV-store / LMDB-like** direction is possible, but probably not the most natural first step for this repo.

That broader framing better captures the real scaling issues:

* file-count pressure
* manifest growth
* invalidation correctness
* shard lookup
* future online-data compatibility
* future run-history / analytics integration

---

## Short roadmap version

* [ ] Research **large-scale cache architecture** for 100k–1M+ image datasets
* [ ] Split design into **registry storage** and **tensor-cache storage**
* [ ] Replace giant manifest JSON with a **persistent sample registry**
* [ ] Default registry direction: **SQLite**
* [ ] Support **Parquet export** for analytics / bulk scans
* [ ] Add **config-hash namespacing** for latent and TE cache independently
* [ ] Design **caption-hash TE deduplication**
* [ ] Implement **sharded latent cache** by namespace + bucket
* [ ] Implement explicit **shard index** for sample/caption lookup
* [ ] Prefer **append-only shard generations** over in-place mutation
* [ ] Tune shards by **target byte size**, not only image count
* [ ] Evaluate **Zarr** only as a later chunked-array research option
* [ ] Evaluate KV-store approaches only if shard-file design proves insufficient
* [ ] Add migration / cleanup / compaction utilities
* [ ] Keep training-facing cache API unchanged across backends

---

Here’s exactly what’s missing from Text 2:

### 1. The Concrete Shard Size Targets
In Text 1, there was a specific recommendation for the "sweet spot" for shard sizing:
> *"Aim for 256 MB to 1 GB shards is usually a more portable design knob than 10k images."*

Text 2 mentions tuning by byte size, but it **completely drops those specific numbers**. If you’re setting your `MAX_SHARD_SIZE` constant later, Text 1 gives you a better starting point than the vague "target byte size" in Text 2.

### 2. The Mutation "Pros vs. Cons" Breakdown
Text 1 had a dedicated section on the "Biggest hidden concern"—how to handle updates. It explicitly broke down the trade-offs:
* **Append-only:** Simpler correctness, easier crash recovery.
* **Mutable shards:** Fewer leftovers, but more failure modes and fragmentation.

Text 2 just says "Prefer append-only," but it **omits the reasoning** why you might occasionally consider mutable shards (and why they are dangerous). 

### 3. Deep-Dive Niche Specs (Zarr v3)
Text 1 mentioned a few specific technical details about **Zarr v3**, like its custom store API and **AsyncIO support**. Text 2 keeps the Zarr discussion much more high-level and doesn't mention the asynchronous benefits, which might matter if you’re planning to fetch data from a network-attached storage or a slow HDD.

### 4. The "Direct Validation" of your Plan
Text 1 was much better at telling you exactly which parts of *your* original ideas were winners. It specifically called out:
* **Config-hash namespacing** as a "strongest part."
* **Caption-hash dedup** as "one of the best parts of the design."
