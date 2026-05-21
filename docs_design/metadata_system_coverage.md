# Metadata System Coverage Map

Date: 2026-05-14

This note is phase 2 of the metadata-system work. It asks what the ideal
repo-wide metadata backbone should cover at a high level. It deliberately avoids
choosing concrete implementation shapes, storage formats, class layouts, or API
boundaries for each concern.

Companion note:

- `docs_design/metadata_system_inventory.md` records what exists today.

## Framing

The ideal metadata backbone should make the repo better at answering:

- What is this thing?
- Where did it come from?
- What configuration and runtime context produced it?
- What is it compatible with?
- What can safely consume it?
- What changed between two runs, artifacts, samples, or cache entries?
- Which facts are internal runtime facts, and which facts belong in exported
  user-facing artifacts?

The key split is not "metadata vs non-metadata." The useful split is:

```
                 REPO METADATA COVERAGE

   source facts       runtime facts       exported facts
        |                  |                    |
        v                  v                    v
 +-------------+    +-------------+      +-------------+
 | discovery   |    | training    |      | checkpoint  |
 | dataset     |--->| cache       |----->| report      |
 | model       |    | optimizer   |      | model card  |
 +-------------+    +-------------+      +-------------+
        |                  |                    |
        +-------- provenance, identity, validation --------+
```

## Coverage Principles

- The system should distinguish internal metadata from exported artifact
  metadata.
- The system should preserve provenance without forcing every consumer to know
  every producer's details.
- Metadata should support validation and compatibility checks, not just
  descriptive output.
- Human-facing metadata and machine-routing metadata should not be blurred.
- Domain code should own source truth and lifecycle call sites, while normal
  recorded metadata schemas and assembly live in the central metadata system.
- Exported metadata should be intentionally lossy when needed; internal metadata
  can be richer than safetensors string key/value storage.
- Metadata should help future tools compare runs, explain artifacts, and decide
  whether cached or persisted data is reusable.

## 1. Identity Coverage

The backbone should cover stable identity for the main entities the repo works
with.

Ideal coverage:

- Training runs and sessions.
- Source datasets and dataset views.
- Individual samples and derived sample variants.
- Model families and loaded components.
- Training modes such as adapter, fine-tune, and textual inversion.
- Adapter methods and adapter artifacts.
- Objectives, timestep policies, and loss-shaping policies.
- Optimizer and scheduler plans.
- Cache entries and cache namespaces.
- Checkpoints, final artifacts, reports, samples, and sidecar files.

Questions it should answer:

- Which entity is this metadata about?
- Is this identity stable across process restarts?
- Is this identity local to one run, or comparable across runs?
- Is this identity user-facing, internal, or both?

## 2. Provenance Coverage

The backbone should describe where entities came from and what transformed them.

Ideal coverage:

- Source model paths, remote IDs, hashes, versions, and model-family
  interpretation.
- VAE, tokenizer, text encoder, denoiser, and other component provenance.
- Dataset source roots, metadata files, captions, generated tags, and
  preprocessing sources.
- Cache derivation chains from source sample to latent, text-encoder output,
  token file, or future shard.
- Adapter target provenance from model component to selected module/parameter to
  adapter module/trainable.
- Merge, continuation, resume, and conversion provenance for artifacts.
- Git revision, config source, command/runtime environment, and tool versions.

Questions it should answer:

- What upstream inputs produced this thing?
- Which transformation path did it go through?
- Can we reconstruct enough context to explain or reproduce it?
- Which provenance is required for correctness, and which is only diagnostic?

## 3. Compatibility And Reuse Coverage

The backbone should make compatibility explicit enough that code does not guess
from filenames, ad hoc keys, or broad config equality.

Ideal coverage:

- Whether a cache entry is compatible with the current data, model, VAE,
  tokenizer, caption, bucket, precision, and preprocessing settings.
- Whether an adapter artifact is compatible with the current base model family,
  component namespace, target selector vocabulary, and adapter method.
- Whether a checkpoint can be resumed with the current mode, optimizer,
  scheduler, distributed layout, and training-state expectations.
- Whether a model artifact's metadata accurately reflects objective semantics
  such as DDPM epsilon/v-prediction or RF-style flows.
- Whether logged/report artifacts correspond to the run context being analyzed.

Questions it should answer:

- Can this thing be reused safely?
- If not, which fact made it incompatible?
- Is the mismatch fatal, warning-level, or only a report annotation?
- Does compatibility depend on exact config equality or on a narrower signature?

## 4. Dataset And Sample Structure Coverage

The backbone should cover dataset structure as a first-class metadata domain,
not just as transient manifest content.

Ideal coverage:

- Sample identity, source path or URI, content hash where available, dimensions,
  media type, and alpha/mask facts.
- Caption, tag, class/prior, source, split, repeat, weight, and grouping facts.
- Bucket assignment and resize/crop-relevant facts.
- Deduplication identity and relationships between original samples and derived
  variants.
- Readiness state for required payloads such as latents, text-encoder outputs,
  token files, masks, and future mode-specific assets.
- Exposure/accounting facts needed for async or incremental training.

Questions it should answer:

- Is the sample known, structurally assigned, and ready for training?
- Which training view admitted it, and why?
- How often has it been exposed relative to intended repeats/weights?
- Which facts must be known before training can start, and which can arrive
  asynchronously?

## 5. Training Run Coverage

The backbone should describe the actual run in a way that can feed checkpoints,
reports, dashboards, and later analysis.

Ideal coverage:

- Run identity, timing, session state, seed, distributed/rank context, and Git
  revision.
- Full composed config identity plus intentionally selected key-config facts.
- Training mode, model family, loaded components, objective, timestep policy,
  loss policy, validation policy, precision, and performance policy.
- Optimizer/scheduler plan, logical groups, runtime ownership, and learning-rate
  labels.
- Dataset counts, bucket summaries, tag summaries, regularization/validation
  counts, and effective batch/step/epoch facts.
- Interrupt, resume, finalization, and artifact registration state.

Questions it should answer:

- What exactly ran?
- What should be attached to exported checkpoints?
- What should be attached to reports and dashboards?
- What is needed to resume, compare, audit, or reproduce the run?

## 6. Model And Component Coverage

The backbone should cover model-family and component identity without locking the
repo into SD-shaped assumptions.

Ideal coverage:

- Family-declared component identity, public labels, internal keys, roles, and
  capabilities.
- Loaded component provenance and runtime state.
- Component-specific precision, trainability, residency, and checkpoint behavior.
- Selector namespace metadata for parameter dumps, optimizer targeting, adapter
  targeting, and diagnostics.
- Family-specific exported model metadata, including model-spec fields.

Questions it should answer:

- Which components exist for this family?
- Which roles/capabilities does each component provide?
- Which selectors are public and stable?
- Which component facts belong in exported artifacts versus internal reports?

## 7. Adapter And Trainable Coverage

The backbone should cover adapter methods and trainable provenance cleanly enough
for saving, loading, merging, diagnostics, and optimizer grouping.

Ideal coverage:

- Adapter method identity and method-local capability facts.
- Resolved target identity, including component, selector, local path, module
  type, and target role tags.
- Adapter module and trainable parameter provenance.
- Artifact load/save/merge/continue provenance.
- Method-local reconstruction facts such as deterministic projection settings.
- User-facing adapter metadata and compatibility-oriented adapter metadata.

Questions it should answer:

- Which original model target does this adapter trainable belong to?
- What method-specific facts are required to reload or merge correctly?
- Which metadata is generic adapter metadata, and which is method-owned?
- Can diagnostics/reporting explain adapter memory and parameter ownership
  without reverse-engineering modules?

## 8. Cache And Derived Payload Coverage

The backbone should cover derived payloads as artifacts with signatures and
readiness state, not just as files that happen to exist.

Ideal coverage:

- Latent cache identity, source sample identity, VAE signature, preprocessing
  signature, bucket shape, dtype, flip/alpha/mask state, and payload keys.
- Text-encoder cache identity, caption/text identity, tokenizer/encoder
  signature, clip-skip/max-length context, dtype, and payload keys.
- Epoch-token metadata, manifest/view identity, sample count, encoder names, and
  validation signatures.
- Future sharded-cache namespace, shard membership, slice addressing, and shard
  validity metadata.
- Cache invalidation explanations.

Questions it should answer:

- What produced this derived payload?
- Which current settings can reuse it?
- Which payloads are missing, stale, partial, or ready?
- How should cache state be summarized for large datasets?

## 9. Artifact And Export Coverage

The backbone should cover every durable output the repo produces.

Ideal coverage:

- Checkpoints, final model artifacts, adapter artifacts, full-model exports,
  diffusers exports, sample images, benchmark reports, JSON reports, resource
  monitor logs, and future dashboard/run-warehouse records.
- Artifact kind, format, path/location, creation time, producer, run association,
  and relevant compatibility/provenance signatures.
- Exported user-facing metadata such as model title, author, tags, thumbnail,
  license, usage hint, and trigger phrase.
- Exported machine-facing metadata such as `modelspec.*`, legacy `ss_*`, and
  method-local adapter keys.

Questions it should answer:

- What did this run produce?
- Which metadata should travel with the artifact?
- Which richer metadata should stay in repo-owned sidecars or run records?
- How should external format limits affect exported metadata?

## 10. Observability And Analytics Coverage

The backbone should support current logging and future run-warehouse/dashboard
work without turning console output into the metadata source of truth.

Ideal coverage:

- Run lifecycle events, startup summaries, step metrics, validation metrics,
  resource-monitor events, report payloads, and artifact registrations.
- Metric label provenance, especially optimizer/logical group labels.
- Resource component breakdowns and loaded/trainable/frozen state summaries.
- Backend/sink-neutral event facts for future W&B, local warehouse, dashboard, or
  report consumers.
- Search/comparison facts for runs, artifacts, configs, cache reuse, and resource
  outcomes.

Questions it should answer:

- What should a human see now?
- What should a machine store for later?
- Which facts are events, which are state snapshots, and which are summaries?
- Can later tools compare runs without scraping logs?

## 11. Validation, Policy, And Trust Coverage

The backbone should make metadata trustworthy enough to drive behavior.

Ideal coverage:

- Required versus optional metadata by domain.
- Validation severity levels: fatal mismatch, warning, informational annotation.
- Source ownership: which domain object is authoritative for a fact, and which
  central emitter/provider is allowed to record or mutate it.
- Consumer expectations: which consumers require the fact, and what fallback is
  acceptable.
- Versioning and migration expectations for persisted metadata.
- Privacy/safety boundaries for paths, user strings, external tracker configs,
  and future shared reports.

Questions it should answer:

- Can this metadata be trusted to make a runtime decision?
- If a fact is missing, who is responsible?
- Is this field stable enough to persist?
- Should this fact be exported, redacted, normalized, or kept internal?

## 12. Comparison And Explanation Coverage

The backbone should make differences explainable.

Ideal coverage:

- Run-to-run config and metadata comparison.
- Artifact-to-run linkage.
- Cache hit/miss and invalidation reasons.
- Dataset view comparison across scans, manifests, epochs, and readiness states.
- Optimizer/scheduler plan comparison.
- Model/component/trainable comparison across families and modes.

Questions it should answer:

- Why did this run behave differently?
- Why was this cache reused or rebuilt?
- Why does this checkpoint or adapter not load cleanly?
- What changed between two artifacts that look similar by filename?

## Non-Goals For This Coverage Pass

- Do not choose a package layout.
- Do not choose dataclass names or schemas.
- Do not choose a database, file format, sidecar format, or artifact layout.
- Do not decide whether metadata lives in strategies, models, data, training, or
  a central package.
- Do not migrate current `ss_*`, `modelspec.*`, cache, or logging metadata yet.
- Do not define exact compatibility signatures yet.

## Coverage Summary

The ideal metadata backbone should cover these high-level domains:

- Identity.
- Provenance.
- Compatibility and reuse.
- Dataset and sample structure.
- Training run state.
- Model and component facts.
- Adapter and trainable facts.
- Cache and derived payload facts.
- Artifact and export facts.
- Observability and analytics facts.
- Validation, policy, and trust.
- Comparison and explanation.

The inventory shows the repo already has fragments of each domain. The next
useful step is to gather previous notes/resources and then combine the inventory
plus this coverage map into an architectural direction.
