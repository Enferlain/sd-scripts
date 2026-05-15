# Metadata System Research Inputs

Date: 2026-05-14

This note collects metadata-system implications from future-plan research notes.
It is intentionally selective: it records the parts that the metadata backbone
would need to shoulder, not the full design of each future feature.

Companion notes:

- `docs_design/metadata_system_inventory.md`
- `docs_design/metadata_system_coverage.md`

## Source: `docs_design/future_ideas/async_data.md`

The async-data note reframes the future data layer from global phase barriers to
sample-level state progression. The metadata system would carry much of the
state, identity, compatibility, and observability needed to make that safe.

### 1. Persistent Sample State

The note's core requirement is persistent sample state. This is directly a
metadata-system responsibility.

Metadata should cover:

- Stable sample identity.
- Source path, URI, or content identity.
- Image/media dimensions and inspection facts.
- Bucket assignment.
- Caption and tag metadata.
- Cache namespace membership.
- Shard membership or row mapping.
- Timestamps and versioning.
- Failure, retry, skipped, deferred, or quarantined state.

Why it matters:

- Async work cannot rely on an in-memory manifest as the only source of truth.
- Training needs to know which samples are known, structurally assigned, and
  ready.
- Later streaming and large-scale cache work depends on this same state model.

### 2. Sample Lifecycle And Readiness

The note proposes moving from dataset-level phase completion to per-sample
lifecycle state.

Metadata should cover lifecycle states such as:

- Discovered.
- Inspected.
- Bucketed.
- Latent pending.
- Latent ready.
- Text-encoder pending.
- Text-encoder ready.
- Ready for training.
- Failed.
- Quarantined.

Metadata should also distinguish mode-specific readiness, for example:

- Ready for latent-only training.
- Ready for latent plus text-encoder cached training.
- Ready for token-file epoch use.
- Ready for future mode-specific payload requirements.

Why it matters:

- Readiness becomes the gate for training consumption.
- Partial readiness must be explicit instead of inferred from file existence.
- The system needs to explain why a sample is not yet trainable.

### 3. Conceptual Phase Preservation

The async note keeps the current conceptual responsibilities:

- Scan and inspection.
- Cache and preprocessing.
- Epoch or view preparation.
- Training consumption.

Metadata should preserve the output facts from each responsibility without
requiring those responsibilities to remain hard global barriers.

Why it matters:

- The metadata backbone should support phase ownership while allowing runtime
  overlap.
- Producers should record facts as they become available.
- Consumers should query readiness and compatibility rather than assuming the
  previous global phase completed.

### 4. Ready Pool And Epoch Snapshot Semantics

The note recommends starting with snapshot-per-epoch over the ready pool.

Metadata should cover:

- Which samples are in the ready pool.
- Which ready pool snapshot an epoch/view used.
- Snapshot creation time and source readiness criteria.
- Sample membership in each training view.
- Window or live-ready semantics if added later.
- Exposure accounting across snapshots.

Why it matters:

- Epoch semantics become a metadata question once readiness changes over time.
- Reproducibility requires knowing which ready subset training actually used.
- Future live-ready or streaming modes need explicit semantics rather than
  hidden dataloader behavior.

### 5. Cache And Derived Payload State

Async dataflow depends on knowing which derived payloads exist, are valid, and
are reusable.

Metadata should cover:

- Latent cache readiness and compatibility.
- Text-encoder output readiness and compatibility.
- Token-file readiness and manifest linkage.
- Cache namespace identity.
- Cache write state and completion state.
- Shard membership and shard lookup facts for future large-scale caching.
- Cache invalidation reasons.
- Partial, stale, missing, and failed payload states.

Why it matters:

- File existence is not enough to decide readiness.
- Incremental cache production needs registry updates as work completes.
- Future sharded caches require metadata for lookup, grouping, and validation.

### 6. Work Queue And Backpressure Observability

The note separates persistent state from work execution, but the metadata system
still needs to expose enough facts for observability and debugging.

Metadata or metadata-adjacent events should cover:

- Queue depth by stage.
- Stage throughput.
- Blocked-stage diagnostics.
- Readiness counts by state.
- Failure and retry counts.
- Slow or quarantined sample summaries.
- Worker/stage timestamps where useful for analysis.

Why it matters:

- Async systems are harder to debug without explicit state transitions.
- Users need to know whether they are waiting on disk, CPU, GPU, writes, or
  policy.
- Future dashboards and reports should not have to infer this from logs.

### 7. Failure Isolation And Recovery

The async note calls out failed or quarantined samples as first-class states.

Metadata should cover:

- Failure reason.
- Failure stage.
- Retry count and retry policy state.
- Quarantine marker.
- Whether the sample is excluded from ready pools.
- Whether the failure is fatal for the run, warning-level, or ignorable.

Why it matters:

- Bad samples should not poison the entire pipeline.
- Recovery and reprocessing need durable reasons, not just transient log lines.
- Dataset health should become inspectable.

### 8. Device And Execution Policy Context

The note warns that async should not mean unrestricted GPU-heavy concurrency.

Metadata should cover policy context such as:

- Which preparation stages are allowed to overlap with training.
- Whether a payload was produced on CPU, training GPU, prep GPU, or another
  worker class.
- Resource or device ownership facts that affect compatibility or diagnostics.
- Policy decisions that explain why work is waiting.

Why it matters:

- Training performance and prep throughput interact.
- Future reports should explain contention or disabled overlap.
- Reproducibility may depend on knowing the active async policy.

### 9. Streaming And Incremental Discovery Hooks

The note treats async dataflow as a prerequisite for online or streaming data.

Metadata should leave room for:

- Incrementally discovered samples.
- Remote or lazy materialization state.
- Live ingest source identity.
- Synthetic sample generation provenance.
- Windowed training views.
- Active-learning or feedback-loop provenance.

Why it matters:

- The first async implementation may still use bounded known datasets, but the
  metadata model should not make continuous discovery impossible.
- Persistent sample state is the bridge from normal datasets to streaming.

### 10. Boundary: What Metadata Should Not Own

The metadata backbone should not own the execution engine itself.

The async-data note implies metadata should not be responsible for:

- Running worker queues.
- Scheduling threads or processes.
- Performing image decode, VAE encode, or text-encoder encode.
- Performing cache writes directly.
- Choosing exact queue sizes or GPU scheduling algorithms.

Instead, metadata should own or support:

- Durable state.
- State transitions.
- Readiness and compatibility facts.
- Provenance.
- Validation and explainability.
- Observability facts emitted by the execution system.

### Extracted Coverage Additions

From this note, the ideal metadata system especially needs to cover:

- Persistent sample registry/state.
- Sample lifecycle and readiness.
- Ready-pool and epoch-snapshot identity.
- Derived payload readiness and cache namespace/shard state.
- Failure/quarantine/retry metadata.
- Async pipeline observability.
- Execution-policy context for GPU/CPU/I/O overlap.
- Streaming-compatible incremental discovery.

## Source: `docs_design/future_ideas/data_accounting.md`

The data-accounting note narrows the async-data problem to epoch semantics,
bucket coverage, and exposure fairness when samples become ready at different
times. It adds stronger requirements around training population state and
sample-level accounting.

### 1. Existence, Assignment, And Readiness Are Separate Facts

The note explicitly separates three sample states that should not collapse into
one boolean.

Metadata should cover:

- Existing set: samples known by discovery/metadata gather.
- Assignable set: samples admitted to the training population or scheduling
  structure.
- Ready set: samples whose required payloads are train-ready for the current
  mode.
- Transitions between those sets.
- The timestamp, epoch, or run window when each transition happened.

Why it matters:

- "All data" is ambiguous in an async system unless the system records all data
  that existed, was assigned, or was ready at a specific boundary.
- Assignment is the recommended fairness baseline for target exposure, so it
  needs durable identity and timing.
- Readiness alone is too late to explain or correct prep-delay bias.

### 2. Metadata-First Structural Blocking

The note recommends an early metadata/inspection pass that defines structure
before payload preparation completes.

Metadata should cover early structural facts:

- Sample identity.
- Dimensions.
- Bucket assignment.
- Caption metadata.
- Split membership.
- Repeat or weight metadata.
- Dedup identity.
- Source, class, or prior grouping.

Why it matters:

- The system can safely block on light structural facts while allowing heavy
  payload work to remain asynchronous.
- Bucket planning, epoch/view construction, and fairness accounting all depend
  on early structural facts.

### 3. Epoch Definition And Snapshot Identity

The note recommends the first async-compatible epoch model as:

> epoch = deterministic snapshot of the ready pool at epoch start

Metadata should cover:

- Which epoch model was used: strict snapshot, ready snapshot, rolling/window, or
  future deficit-aware mode.
- Snapshot identity.
- Snapshot creation boundary.
- Ready criteria used to build the snapshot.
- Included and excluded sample sets.
- Deterministic shuffle seed/context.
- Whether late-ready samples are deferred to later epochs.

Why it matters:

- Epochs no longer automatically mean "one pass over all known data."
- Reproducibility requires recording the exact ready snapshot and policy.
- Future modes may use epochs as reporting windows rather than full passes.

### 4. Bucket Coverage And Threshold Policies

The note adds bucket-readiness policy as a metadata concern.

Metadata should cover:

- Full bucket structure for the known/assigned set.
- Ready counts per bucket.
- Included/excluded buckets for a snapshot.
- Minimum total ready threshold.
- Minimum per-bucket readiness threshold.
- Underfilled bucket policy.
- Bucket-level under-service or deficit if balancing is introduced.

Why it matters:

- A bucket can be structurally known but not sufficiently ready.
- Early training can skew toward buckets that prepare quickly.
- Future sharded caches need bucket membership and readiness for planning.

### 5. Exposure Accounting

This note introduces actual seen counts and exposure fairness as first-class
metadata/system facts.

Metadata should cover:

- When a sample was discovered.
- When a sample was assigned.
- When a sample became ready.
- How many times a sample has been seen.
- How many times a sample was intended to be seen according to repeats/weights.
- Seen counts by bucket, source, class/prior group, and split where relevant.
- Per-epoch or per-window exposure summaries.

Why it matters:

- Without actual exposure tracking, early-ready samples can dominate silently.
- Late-ready samples may need later balancing.
- Dataset health and training distribution need to be explainable.

### 6. Target Exposure, Deficit, And Catch-Up Policy

The note describes long-term deficit/debt scheduling:

```text
deficit = target_seen - actual_seen
```

Metadata should cover, if/when this scheduling model appears:

- Target exposure.
- Actual exposure.
- Deficit/debt.
- When target exposure begins: existence, assignment, or readiness.
- Catch-up policy: aggressive, soft, bounded, or disabled.
- Deficit caps or smoothing settings.
- Bucket-aware deficit summaries.

Why it matters:

- Assignment-time target exposure is recommended as the long-term middle ground.
- Deficit explains why a late-ready sample may be prioritized later.
- Catch-up policy affects training distribution and should be auditable.

### 7. Source/Class Distribution Drift

The note calls out that readiness can differ by source, class, bucket, or other
dataset groupings.

Metadata should cover:

- Source or class membership.
- Group-level discovered, assigned, ready, and seen counts.
- Group-level readiness lag.
- Group-level exposure imbalance.
- Group-level inclusion/exclusion in snapshots.

Why it matters:

- Async readiness can create unintended curriculum or distribution shift.
- Fairness is not only per-sample; it can be per bucket/source/class.
- Reports need enough metadata to show skew before it becomes a silent training
  behavior.

### 8. Scheduler Model Provenance

The note outlines a staged scheduler evolution:

- Ready-snapshot epochs.
- Cross-epoch balancing.
- Deficit-aware scheduling.
- Continuous/windowed scheduling.

Metadata should cover:

- Active scheduler/accounting model.
- Scheduler policy parameters.
- Population boundary used by the scheduler.
- Whether seen counts influence selection.
- Whether deficit influences selection.
- Whether epoch is a pass, snapshot, or reporting window.

Why it matters:

- Runs using different scheduling/accounting models are not directly equivalent.
- The scheduler policy is part of run provenance, not just data-loader behavior.
- Comparisons need to know which fairness/accounting policy was active.

### Extracted Coverage Additions

From this note, the metadata system especially needs to cover:

- Existing, assignable, and ready sample sets.
- Assignment-time and readiness-time transition metadata.
- Epoch/snapshot model identity and snapshot membership.
- Bucket readiness thresholds and bucket inclusion/exclusion.
- Actual seen counts and exposure summaries.
- Target exposure, deficit/debt, and catch-up policy if advanced scheduling is
  added.
- Source/class/bucket distribution drift observability.
- Scheduler/accounting model provenance.

## Source: `docs_design/future_ideas/data_shards.md`

The data-shards note treats large-scale caching as a storage redesign split
between a persistent sample registry and sharded tensor payload stores. It adds
metadata requirements around addressability, namespace invalidation, index
integrity, shard lifecycle, and analytics/export paths.

### 1. Registry And Tensor Payloads Are Separate Concerns

The note explicitly separates:

- Sample registry / manifest store.
- Latent cache store.
- Text-encoder output cache store.

Metadata should cover the registry side as the canonical place for sample state
and cache location facts, while tensor stores own payload bytes.

Why it matters:

- The best storage shape for metadata is not necessarily the best storage shape
  for tensors.
- The registry needs point lookups, incremental updates, state transitions, and
  shard membership.
- Tensor payloads need efficient batched reads and fewer filesystem objects.

### 2. Shard Addressability

The note makes explicit shard lookup a core requirement.

Metadata should cover:

- Latent namespace.
- Latent shard ID.
- Latent row, offset, or slice mapping.
- TE namespace.
- TE shard ID.
- TE row, offset, or slice mapping.
- Bucket key for latent shard grouping.
- Caption hash for TE lookup.
- Sample ID to shard mapping.
- Caption hash to shard mapping.

Why it matters:

- Training should not need to open or parse whole shard payloads to discover
  where one sample lives.
- Batch reads should be able to group selected entries by shard.
- Shard-aware reads depend on metadata addressability, not filename guessing.

### 3. Namespace-Based Invalidation

The note recommends config-hash namespaces rather than in-place cache mutation.

Metadata should cover latent namespace inputs such as:

- Base resolution.
- Bucket steps.
- No-upscale or resize policy.
- Latent dtype.
- Model family or latent contract version.

Metadata should cover TE namespace inputs such as:

- Tokenizer identity or version.
- Text encoder identity or version.
- Max token length.
- Clip skip.
- Prompt-weighting or caption-mutation policy version.
- TE output contract version.

Why it matters:

- Latent and TE caches need independent invalidation.
- Old and new cache generations can coexist safely.
- Crash recovery and debugging are easier when writes do not mutate existing
  namespaces in place.

### 4. Shard Index Metadata

The note separates payload files from explicit indexes.

Metadata should cover shard index facts such as:

- Which sample IDs or caption hashes are present.
- Entry row, offset, or slice.
- Tensor dtype.
- Tensor shape.
- Payload key layout.
- Shard-local stats.
- Checksums or integrity facts.
- Shard format or contract version.

Why it matters:

- Simple membership and shape checks should not require full payload reads.
- Index metadata becomes the bridge between registry lookups and payload reads.
- Integrity/version facts help validation, migration, and debugging.

### 5. Shard Sizing And Grouping Policy

The note recommends tuning shards by target payload size rather than image count.

Metadata should cover:

- Target shard size policy.
- Actual shard byte size.
- Image/sample count per shard.
- Bucket or shape grouping.
- Dtype and latent shape mix.
- Batch-locality hints if sampling becomes shard-aware.

Why it matters:

- A fixed image count is not portable across resolutions, dtypes, or model
  families.
- Shard size affects read amplification, file count, and compaction pressure.
- Batch-locality metadata can help reduce scatter-gather overhead later.

### 6. Append-Only Generations, Cleanup, And Compaction

The note favors append-only shard generations over in-place mutation.

Metadata should cover:

- Shard generation identity.
- Active versus stale generation state.
- Finalized versus partial write state.
- Cleanup eligibility.
- Compaction lineage.
- Replacement relationships between old and new shard generations.
- Tombstone or deletion markers if needed.

Why it matters:

- Append-only writes reduce corruption risk and simplify crash recovery.
- Old namespaces and stale shard generations will accumulate without metadata
  describing what is active and what can be cleaned.
- Compaction and migration need lineage, not just files on disk.

### 7. Crash Recovery And Write Atomicity

The note calls out interrupted writes as a major problem to solve.

Metadata should cover:

- Write intent or in-progress state.
- Payload write completion.
- Index write completion.
- Registry update completion.
- Finalization marker.
- Recovery action or orphan detection state.

Why it matters:

- Sharded caches have larger failure blast radius than per-image files.
- A run may crash between payload, index, and registry updates.
- Recovery should be explicit and inspectable.

### 8. Caption-Hash TE Deduplication

The note treats TE cache identity as caption-hash plus TE config, not image ID.

Metadata should cover:

- Normalized caption identity.
- Caption hash.
- TE config namespace.
- Many-sample-to-one-TE-entry relationships.
- Caption mutation or prompt-weighting policy version.

Why it matters:

- Repeated captions can share TE outputs.
- Caption-related invalidation should not rebuild latent caches.
- TE cache identity must remain stable despite sample-level image identity.

### 9. Registry Export And Analytics

The note treats Parquet as a likely analytics/export companion even if SQLite is
the canonical mutable registry.

Metadata should cover:

- Export snapshot identity.
- Export creation time.
- Source registry version.
- Included namespaces and state filters.
- Analytics-friendly summaries for large scans.

Why it matters:

- The metadata backbone should serve operational lookup and offline analysis.
- Exported views must be traceable back to a registry state.
- Future run-history and data analytics need stable export provenance.

### 10. Debuggability And Manual Inspection

The note highlights that sharded systems are less manually transparent than
per-image files.

Metadata should support:

- Human-readable shard summaries.
- Sample-to-payload lookup explanations.
- Cache hit/miss explanations.
- Namespace and generation summaries.
- Tools that can inspect one sample, one shard, or one namespace.

Why it matters:

- Large-scale storage improves performance but worsens ad hoc debugging unless
  metadata makes the layout explainable.
- Support burden moves from "open one file" to "ask the registry and index."

### Extracted Coverage Additions

From this note, the metadata system especially needs to cover:

- Separation of registry metadata from tensor payload storage.
- Shard addressability: namespace, shard ID, row/offset/slice, bucket, and
  caption-hash mappings.
- Independent latent and TE namespace signatures.
- Shard index facts: membership, dtype, shape, checksums, stats, and format
  version.
- Shard sizing/grouping policy and actual shard stats.
- Append-only generation state, cleanup eligibility, and compaction lineage.
- Crash recovery/write finalization state.
- Caption-hash TE dedup relationships.
- Registry export/analytics snapshot provenance.
- Debuggability summaries for samples, shards, namespaces, and cache decisions.

## Source: `docs_design/future_ideas/streamed_training.md`

The streamed-training note extends the async/registry direction toward remote,
lazy, and continuously arriving samples. Most readiness requirements overlap
with the async-data notes, but streaming adds stronger source identity,
deduplication, recency, replay, and live-ingest concerns.

### Unique Metadata Responsibilities

Metadata should cover:

- Sample source identity by source type: local folder, static manifest, shard
  stream, remote object store, live ingest queue, or synthetic generator.
- Source-provided identifiers and origin metadata.
- Remote materialization state: known remotely, fetched locally, cached locally,
  unavailable, expired, or failed.
- Live-ingest timestamps and ordering.
- Synthetic-generation provenance if samples come from generators or feedback
  loops.
- Duplicate-control facts: content hash, persistent source ID, dedup window, and
  optional future similarity-check provenance.
- Recency and replay-buffer membership.
- Age-aware, source-aware, class-aware, or quota-based balancing policy.
- Whether online mode is using step-based training, snapshot epochs, windowed
  epochs, or replay-buffer sampling.
- Hot-path versus cold-path ownership of transformations such as caption
  mutation, tokenization, latent preparation, and TE preparation.
- Backpressure and pause/slow-discovery state when buffers are full.

Why it matters:

- Streaming data still needs persistent structure; removing manifests entirely
  would weaken reproducibility, deduplication, cache identity, and analytics.
- Online sources can resend or regenerate duplicates, so identity cannot rely
  only on arrival order.
- New samples can dominate if sampling is arrival-order based, so recency and
  replay policy become run metadata.
- Remote and live-ingest workflows need provenance that explains where a sample
  came from and when it became available locally.

### Boundary Notes

The metadata system should not own the live source, worker queues, fetch logic,
or replay scheduler implementation. It should own or support the facts those
systems need to record:

- Source identity.
- Materialization state.
- Dedup state.
- Readiness state.
- Sampling/replay policy provenance.
- Balancing and recency observability.

### Extracted Coverage Additions

From this note, the metadata system especially needs to cover:

- Source-type-specific sample identity and origin metadata.
- Remote/local materialization state.
- Live-ingest and synthetic-generation provenance.
- Duplicate-control metadata beyond simple sample IDs.
- Recency, replay-buffer, and balancing policy metadata.
- Explicit online epoch/window/step semantics.
- Hot-path versus cold-path transformation provenance.
- Backpressure state for streaming discovery and preparation.
