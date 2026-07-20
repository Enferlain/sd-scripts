# Model Metadata Continuation Research

Date: 2026-07-20  
Status: pre-OpenSpec research; findings and recommended scope, not implemented

## Purpose

Research the work that follows the completed model-family metadata slice. This
note combines the model topics that were intentionally left outside that slice
with the source identity, run provenance, and query requirements identified
after implementation.

This document is intentionally under `docs_design/`. The implemented authority
remains `library/metadata/README.md`, `library/models/README.md`, and archived
OpenSpec requirements. The concrete names below are candidates for the next
OpenSpec, not facts about current production behavior.

Durable work is tracked by:

- `sd-scripts-8ck`: stable source-model identity and run provenance;
- `sd-scripts-edl`: system-wide typed metadata query and optional resolution;
- `sd-scripts-d79`: qualified model-component identities in resource facts;
- `sd-scripts-b25`: granular model structure and tensor catalog.

## Executive Conclusion

The continuation contains two model capabilities with one metadata-system
dependency between them:

1. **Model source identity and realization provenance** can and should be
   implemented next. It establishes a stable repository model-catalog identity,
   how newly supplied models are recognized or registered, what source was
   supplied and actually resolved, which components came from which sources,
   what identity evidence exists, and how the resulting run-scoped realization
   relates to produced artifacts.
2. **Granular model structure and tensor facts** should be specified now but
   cannot be completed honestly before the shared typed-query capability. Its
   purpose is not to force full tensor inventories into every run; it is to
   make one tensor fact, a filtered subset, a component aggregate, or an
   explicitly requested full catalog available through the same semantics.

The recommended work order is therefore:

```text
model identity/provenance foundation (8ck)
    -> qualified resource/component linkage (d79)
    -> shared typed metadata query capability (edl)
    -> granular model structure/tensor catalog (b25)
```

A comprehensive model-metadata OpenSpec may describe both model capabilities,
but its tasks should contain an explicit milestone pause before structural
query work. It must not invent a model-only query API to avoid the dependency.
An equally valid packaging is two model changes around the system query change;
the architectural sequencing is more important than the artifact packaging.

## What The Completed Slice Actually Provides

The archived `centralize-model-family-metadata` change established:

- versioned family declaration references;
- run-qualified model realization identities;
- ordered, family-declared top-level component identities;
- family-local typed contributions;
- typed produced-artifact facts and artifact-to-realization relationships;
- central `kuro.*`, ModelSpec, `ss_*`, and safetensors projection ownership.

It deliberately excluded source metadata ingestion, hash generation, arbitrary
module state, parameter names, tensor topology, and storage identities. Those
were correct exclusions for that migration, not a conclusion that these facts
fall outside long-term model metadata.

The current realization answers roughly:

```text
run 42 loaded an SDXL realization called "training-target"
with clip_l, clip_g, vae, and unet components
```

It does not answer:

```text
which exact base source and revision were resolved;
whether two runs used the same source content;
whether the VAE or a text encoder came from an override;
how a Diffusers source was converted into the live component;
what tensor exists at a component-local path;
what its shape, dtype, persistence, aliases, or logical size are.
```

## Current Provenance Gap

### Configured names are not source identities

`ModelConfig` currently exposes `pretrained_model_name_or_path` and `vae` as
strings. Run metadata retains configured model/VAE names, while the realization
builder receives only run, family, model-version, and loaded-component data.
There is no accepted source entity or source-to-realization relationship.

A supplied string may mean a local checkpoint, a local Diffusers directory, or
a Hugging Face repository. A branch-like Hub reference can resolve differently
over time. A local path can be modified in place. A basename loses still more
evidence. None of these values alone proves exact equality across runs.

### The successful loader knows more than the config

The actual source and fallback decisions occur inside family loaders:

| Family | Current source behavior | Provenance lost at the strategy boundary |
| --- | --- | --- |
| SD1/SD2 | A file is treated as an original checkpoint; anything else is passed to a Diffusers pipeline. An optional VAE replaces the integrated VAE. | File versus local directory versus Hub repository, resolved Hub revision, integrated versus overridden VAE, and the Diffusers-to-repo UNet conversion. |
| SDXL | The same file/Diffusers split is used. An `fp16` variant is attempted for FP16 and may fall back to the default variant. Checkpoint information is retained only as mutable strategy state. | The variant actually selected, resolved source/revision, checkpoint evidence, VAE override, and conversion of the Diffusers UNet. |
| SD3 | One unified safetensors checkpoint supplies MMDiT and may supply all encoders/VAE; CLIP-L, CLIP-G, T5-XXL, and VAE may instead come from sidecars/overrides. | Per-component source binding, integrated tensor namespace versus sidecar source, source artifact evidence, and materialization transforms. |

The common `ModelLoadingStrategy` contract returns only
`(model_version, loaded_components)`. Central code cannot reconstruct the
successful branch, fallback, or component origin from that result. Re-reading
configuration after loading would be both incomplete and sometimes wrong.

The next provenance slice should therefore extend the successful load result
with typed source observations and component bindings. Family/model loading
code remains the source of these facts; central metadata owns their schema,
validation, identity construction, emitters, and relationships. This is not a
reason to create family-local metadata mini-frameworks.

## Identity Layers The Design Must Keep Separate

The repository can and should assign one stable internal catalog identity to a
model record. That identity is the key by which the repository recognizes the
model and gathers its aliases, sources, realizations, fingerprints, and output
history. It does not need to encode or prove every stronger meaning of “the same
model” inside the identifier itself.

The record must still keep at least six meanings separate:

1. **Repository model-catalog identity** — the stable internal identity used to
   accumulate records about one recognized model. It may be opaque and remains
   stable as better evidence and new aliases are attached.
2. **Family/declaration identity** — architecture and declared component
   semantics such as SDXL with its top-level component roles.
3. **Supplied source reference** — exactly what the user/config requested,
   such as a Hub repository and revision or a local path.
4. **Resolved source snapshot/artifact** — the immutable or observed source
   actually used, such as a full Hub commit SHA or a local file with optional
   content digest.
5. **Run-scoped realization/composition** — the components materialized for
   this run, including overrides, omissions, conversions, and relevant loading
   transforms.
6. **Produced artifact** — a checkpoint, adapter, embedding, or other output
   derived from the realization.

An optional stronger fingerprint may eventually describe normalized model state
independent of packaging. It would be needed to prove that an original
checkpoint and a Diffusers directory encode the same effective weights. A file
digest or Hub commit does not prove that stronger claim, so the initial design
must not imply it. This is a fuller goal to work toward and test, not something
the first provenance implementation must solve completely.

Cross-run comparison should consequently report the level of evidence, not one
overloaded `same_model: bool`. Useful outcomes include:

- same/different supplied reference;
- same/different resolved source snapshot;
- same/different exact source artifact content;
- same/different realization composition when every relevant binding is known;
- unknown when the available evidence cannot support the requested claim.

### Repository recognition and assignment

When a model is presented to the repository, recognition should be progressive:

```text
incoming source or live model
    -> collect the evidence available at acceptable cost
    -> locate an existing model-catalog record when the evidence is sufficient
    -> otherwise create a new stable model-catalog identity
    -> attach stronger evidence or aliases later without changing that identity
```

Potential recognition evidence, from inexpensive/contextual to stronger,
includes:

- provider coordinates such as Hub repository, resolved commit, variant, and
  subfolder/file selection;
- supplied and resolved local references;
- exact artifact digests when already available or explicitly calculated;
- family/component topology and a versioned canonical tensor signature over
  normalized paths, shapes, dtypes, and counts;
- selected or complete tensor-content fingerprints when higher confidence is
  required;
- explicit known-conversion or user/provider alias relationships.

A mutable name is useful alias evidence, not the catalog identity. A file hash
is strong evidence that two serialized artifacts are byte-identical, but it is
also not the catalog identity. Fifty renamed copies may therefore attach to one
model record when their evidence matches, while one model may have multiple
artifact/source records for different serializations.

Recognition should expose at least confirmed, candidate, new/unmatched, and
unknown/insufficient-evidence outcomes. Weak similarity may suggest a candidate
without silently merging records. If stronger evidence later establishes that
two records represent the same catalog model, the design needs an alias/merge
policy that preserves historical references.

The longer-term target is a versioned family-aware state/structure fingerprint
that can survive tensor ordering and known key-format conversions. It should be
developed empirically. Required fixtures include byte-identical clones, renamed
files, reordered serialization, metadata-only changes, known checkpoint-to-
Diffusers key mappings, one changed tensor, component overrides, tied/shared
state, and unknown extra keys. Until those tests support a claim, the catalog
can still provide useful identification from weaker evidence with its limits
recorded.

## Source Evidence Requirements

### Hugging Face sources

For a Hub-backed source, retain the supplied repository and revision separately
from the resolved full commit SHA. The Hub exposes revision-to-commit resolution
and returns snapshot paths scoped by commit; Diffusers separately supports
`revision` and weight `variant`. The selected variant and relevant subfolder or
file selection are therefore part of source materialization, not merely display
metadata.

The current config does not expose a revision or variant contract and SDXL may
change its chosen variant through fallback. The OpenSpec must decide whether to
add explicit config fields now, but the loader result must record the effective
values even when defaults or fallback selected them.

### Local files

For a local file, preserve:

- the supplied reference;
- a normalized/canonical resolved path as observation context;
- file kind/format when established;
- observed size and modification evidence where useful;
- a defined digest only when it was actually calculated.

Path, size, or modification time can support change detection but cannot be
advertised as exact content identity. Hash calculation already exists as a
utility and remains separately tracked because multi-gigabyte hashing has a
real cost. Provenance must represent “digest unavailable/not requested” rather
than substituting a weaker value into a digest field.

### Local directories

A local Diffusers directory is a multi-file source. Its canonical path is a
locator, not immutable identity. Exact comparison would require a defined
manifest or content-digest policy. That policy can be deferred, provided the
initial fact explicitly records the resulting identity limitation.

### Integrated and overridden components

Component composition needs first-class edges. A base checkpoint may supply all
top-level components, while an external VAE or SD3 text encoder replaces only
one. An integrated component should be linked to its containing source plus a
stable extraction/namespace claim; an override should link to its own source.

The relationship means “materialized/derived from,” not “is byte-identical to.”
That distinction matters because current loaders convert Diffusers UNets,
change padding behavior, optionally replace modules with RamTorch, cast
precision, and perform other runtime preparation.

### External artifact metadata

Safetensors or ModelSpec/legacy header values may provide useful source claims,
but they are untrusted evidence from an external artifact. They must retain
their producer/source, be validated, and never become canonical merely because
they use `modelspec.*`, `ss_*`, or `kuro.*` spelling. Compatibility projections
must not be parsed to reconstruct facts the active run already knows.

## Provenance Lifecycle

The trainer currently files realization metadata once immediately after the
initial strategy load. Deferred denoiser loading and later model preparation can
change the loaded-component surface or live state after that point.

The OpenSpec needs an explicit lifecycle decision. A likely minimum is:

```text
load result
    -> source observations and initial component bindings
    -> realization created
deferred component load, if any
    -> realization composition updated or finalized
runtime preparation
    -> separately scoped transformation/observation facts where meaningful
```

The run-scoped realization identity may remain stable while its composition is
finalized, but accepted records must not permanently claim an absent component
that later became present. Repeated snapshots also need unambiguous newest-state
or version semantics.

Not every call to `.to(...)` or module mutation belongs in provenance. The
initial slice should record only transformations necessary to interpret the
relationship between source and realization—for example direct extraction,
component override, format conversion, or a material change explicitly owned
by the load contract. Mutable device placement and allocator residency remain
runtime/resource observations.

## Existing Structural Discovery Is Reusable But Not Canonical

The repository can already inspect models directly:

- `library/models/parameter_dump.py` enumerates named modules, parameters, and
  buffers and renders shape, dtype, trainability, and buffer persistence;
- optimization builds component-qualified live module/parameter references for
  matching and optimizer construction;
- adapter reporting counts trainable parameters and logical bytes by component;
- startup diagnostics and resource intelligence independently sum parameter
  counts and `numel * element_size` per component.

These mechanisms prove feasibility, but they currently have different purposes
and semantics:

- the parameter dump renders strings instead of typed accepted descriptors;
- optimization references contain live objects and family-local keys rather
  than durable metadata identities;
- resource estimates count parameters but not buffers and currently use bare
  component keys as owner identifiers;
- repeated implementations can disagree on alias deduplication and aggregation.

Execution code should keep live `Module` and `Parameter` references when it
needs to act on them. The metadata continuation should add qualified identities
to those references where useful and centralize reusable descriptors; it should
not turn metadata into an object registry or force optimizers to recover live
parameters from stored records.

## Required Structural Descriptor Semantics

A future catalog should cover:

- qualified realization, component, module, parameter, and buffer identities;
- component-local paths and, where applicable, state-dict/artifact keys;
- module type and ownership relationships;
- parameter versus buffer kind;
- shape, dtype, element count, element size, and logical byte count;
- `requires_grad` and buffer persistence;
- tensor-object aliases and shared/tied storage groups;
- descriptor source, observation scope, and validity/lifecycle scope;
- optional artifact shard/file/offset evidence;
- aggregation with an explicit basis.

Logical tensor identity and storage identity must remain separate. PyTorch
removes duplicate named parameters by default, while multiple tensor views can
share one storage. Raw process pointers are observation-local and must not be
used as durable storage identifiers. A resolver can assign stable-within-one-
observation alias/storage-group identifiers and retain the evidence used to
group them.

At least two aggregate sizes may be useful:

- **logical tensor bytes**: sum of descriptor `numel * element_size` values;
- **unique observed storage bytes**: sum of distinct observed storages within a
  declared observation scope.

Neither is automatically physical GPU memory. Quantization wrappers, sharding,
offload, gradients, optimizer state, allocator overhead, and temporal placement
belong to separate structural/runtime resource facts. Any resource projection
must state its basis rather than labeling all of these quantities “model size.”

## Live And Artifact Evidence Are Not Equivalent

For a live `nn.Module`, PyTorch exposes parameters, persistent and
non-persistent buffers, state-dict membership, tensor views, and storage
sharing. This supports rich realization-scoped descriptors.

A safetensors header can provide tensor keys, dtypes, shapes, and byte offsets
without loading tensor data. That makes it a strong bounded artifact resolver.
However, safetensors does not generally preserve PyTorch shared-tensor
structure, and a header alone does not identify whether a key is a parameter or
buffer in the live architecture. Artifact-backed descriptors must state those
limits and should only add stronger semantic classification when another
validated mapping supplies it.

PyTorch checkpoints can expose metadata through state dictionaries and newer
fake-tensor loading mechanisms, but pickle-backed inputs carry different trust
and loading constraints. Support for them should be an explicit resolver and
security decision, not an automatic side effect of provenance ingestion.

## Why The Shared Query Capability Is A Real Dependency

The current metadata runtime accepts known typed items and exposes complete
snapshots plus concern-specific views. It cannot accept a bounded typed question
or ask an explicit source resolver for one missing fact.

The structural requirement is intentionally more capable than `silent` versus
`full dump`:

```text
one tensor shape
one component's parameter summary
all trainable buffers matching a filter
an explicitly requested complete inventory
```

Recording every tensor on every run would avoid query design only by imposing
unbounded startup work and storage volume. A model-only query facade would
duplicate a capability needed by other metadata concerns. The shared query
OpenSpec should therefore establish:

- typed request and bounded result contracts;
- lookup against accepted/stored facts first;
- optional, explicitly supplied live/artifact resolver contexts;
- unsupported, unavailable, invalid, and stale outcomes;
- query cost/cardinality policy;
- independence of query, retention, projection, and output;
- a future seam for validity/caching without requiring an invalidation system
  in the first slice.

The model catalog then supplies model-specific query types, descriptors, and
resolvers through that system-wide contract.

## Integration Targets

Once provenance, query, and structural facts exist:

- parameter/state dumps can become projections over the canonical descriptor
  semantics rather than a separate discovery definition;
- optimization and adapter references can retain their live objects while also
  carrying qualified component/tensor identities;
- startup resource summaries can consume accepted component aggregates when
  valid, while preserving a direct-live fallback during migration;
- resource structural facts can point at accepted qualified component/tensor
  identities without cross-run collisions;
- produced artifacts can retain derivation from a realization whose complete
  component sources are known;
- inspection, reporting, recommendation, and future configuration-bound systems
  can request only the facts they need.

## Recommended OpenSpec Boundary

The next model-metadata OpenSpec should settle the full continuation but keep
the implementation milestones explicit.

### Milestone group A: source identity and provenance

- define stable repository model-catalog identity plus supplied, resolved,
  artifact/content-evidence, realization, component, and produced-artifact
  distinctions;
- define progressive recognition, candidate matching, new-ID assignment, and
  later alias/merge behavior without deriving identity from mutable names;
- extend the family load result with typed source observations and component
  bindings;
- capture SD, SDXL, and SD3 effective source/fallback behavior;
- file source records and explicit realization/component relationships;
- define evidence-strength and cross-run comparison outcomes;
- preserve a path toward tested family-aware state/structure fingerprints
  without requiring that fuller recognition goal in the first implementation;
- handle initial, deferred, and finalized realization composition;
- validate and persist external header claims as evidence, not truth;
- retain compatibility output only as projection policy.

This group is implementable before the shared query API.

### Milestone boundary: metadata query dependency

Pause the model change after provenance handoff. Complete the shared typed-query
OpenSpec and implementation before structural resolver/catalog milestones. The
OpenSpec tasks should name this dependency rather than leaving it as an implied
future concern.

### Milestone group B: qualified structural catalog

- define module/tensor/storage identities and typed descriptors;
- implement bounded live-module and safetensors-header resolvers;
- support single, filtered, aggregate, and optional full-inventory requests;
- define alias/storage deduplication and logical-byte aggregation;
- link descriptors to accepted source, realization, and component identities;
- migrate parameter dump, optimization identity references, and resource
  summaries in bounded follow-up sections;
- preserve physical/runtime resource observations as a separate concern.

## Explicit Non-Goals For The Initial Continuation

- storing every tensor descriptor for every run by default;
- tensor values, gradients, or arbitrary module attributes;
- complete normalized cross-format weight identity in the first implementation;
  the schema and tests must leave a deliberate path toward it;
- a complete cache/invalidation framework;
- physical CUDA allocation accounting inside model descriptors;
- redesigning checkpoint serialization;
- making compatibility keys canonical;
- adapter-method-local persistence metadata;
- changes under `tools/`.

## Open Questions For The OpenSpec

- Should one comprehensive model change pause across the query dependency, or
  should provenance and structural catalog be two model changes?
- Should repository model-catalog identifiers be opaque generated values,
  deterministic scoped values, or support both through an identity registry?
- What evidence threshold permits automatic catalog-record reuse, and what only
  creates a candidate/alias proposal?
- How are historical references preserved if stronger evidence later merges
  two catalog records?
- Which identity names best distinguish logical source, resolved snapshot, and
  exact content artifact without implying stronger equality than the evidence?
- Should a Hub repository plus full commit and selected variant be considered a
  resolved source identity even when the exact downloaded file set is not
  recorded?
- Which local-file digest policy is required for authoritative equality, and
  when is the cost paid?
- How should a local multi-file Diffusers directory obtain exact identity, if at
  all in the first slice?
- Which load-time transformations are semantically necessary provenance and
  which remain runtime observations?
- Does realization finalization update one identity, create a versioned record,
  or emit an event plus newest-state record?
- How are state-dict keys mapped to family-qualified component/tensor paths when
  original checkpoints use different namespaces?
- Which complete-inventory limits are safe for very large models?
- Should parameter dumps retain non-persistent buffers that can never be
  reconstructed from an artifact, and how is that source limitation rendered?

## External Technical Evidence

- [PyTorch `nn.Module` documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
  defines duplicate removal for named parameters, persistent versus
  non-persistent buffers, and state-dict membership.
- [PyTorch storage documentation](https://docs.pytorch.org/docs/stable/storage.html)
  distinguishes tensor metadata from underlying storage and documents shared
  storage and observation-local storage pointers.
- [PyTorch serialization semantics](https://docs.pytorch.org/docs/main/notes/serialization.html)
  documents parameter/persistent-buffer state dictionaries and metadata-only
  checkpoint inspection using fake tensors.
- [Safetensors metadata parsing](https://huggingface.co/docs/safetensors/en/metadata_parsing)
  shows that keys, dtypes, shapes, and byte offsets can be read from the header
  without downloading/loading complete tensor contents.
- [Safetensors shared-tensor behavior](https://huggingface.co/docs/safetensors/main/torch_shared_tensors)
  documents why ordinary safetensors files do not preserve arbitrary PyTorch
  shared-tensor structure.
- [Hugging Face Hub download documentation](https://huggingface.co/docs/huggingface_hub/main/guides/download)
  documents supplied revisions, full commit-scoped snapshots, and dry-run file
  information including resolved commit hashes.
- [Hugging Face Hub API documentation](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/hf_api)
  exposes model revision information and full repository SHA evidence.
- [Diffusers pipeline documentation](https://huggingface.co/docs/diffusers/main/api/diffusion_pipeline)
  distinguishes repository/path, revision, variant, and local-only loading
  inputs relevant to effective source materialization.
