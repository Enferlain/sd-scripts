## Context

The current resource-observability stack has three strong pieces:

- live observational counters from `library/logging/resource_monitor/`
- benchmark-report/debug reshaping in `library/logging/reports.py`
- startup component estimates and runtime-trace timing windows that already
  narrow where memory changes happened

What it still does not have is an honest model for answering questions like:

- which component or runtime owner caused this memory jump
- whether a phase increase came from weights, optimizer state, activations,
  caches, or temporary workspaces
- how much confidence to place in an attribution claim

The capability investigation and follow-up implementation already established an
important boundary:

- aggregate counters and phase-local deltas are observational
- report/debug summaries can interpret those observations
- provenance/causality is a separate design problem

This change defines that missing design boundary so future implementation can
add attribution without polluting `ResourceMonitorFacts` or overstating what the
raw counters prove.

## Goals / Non-Goals

**Goals:**

- Define a future attribution model that separates observational facts,
  best-effort attribution, and stronger causal claims.
- Keep the live resource event schema observational and additive.
- Identify the repo-owned seams where explicit attribution windows could be
  instrumented later.
- Define how unknown or unattributed resource growth should be represented
  instead of guessed away.
- Reuse existing runtime-trace and startup-diagnostic seams where that helps
  attribution without inventing a second unrelated timing vocabulary.

**Non-Goals:**

- Implementing a full provenance system in this change.
- Promising per-allocation or per-kernel ownership from CUDA allocator counters.
- Expanding host-memory fields or distributed semantics beyond the already-open
  follow-up issue for that work.
- Reclassifying startup component estimates as live resource ownership facts.
- Adding ad hoc provenance fields to `ResourceMonitorFacts`.

## Decisions

### Decision: Keep observational facts and attribution records as separate surfaces

The current live resource event/data model remains observational. Future
attribution records should be separate debug/report-oriented structures that
reference observational snapshots and timing windows rather than extending
`ResourceMonitorFacts` with guessed ownership fields.

Why:

- the current event schema already has a clear contract and low-overhead role
- attribution requires caveats, evidence, and unknown remainder that do not fit
  cleanly into flat per-event counters
- keeping the surfaces separate prevents users from reading a guessed owner
  field as if it had the same status as `gpu_allocated_mb`

Alternative considered:

- add optional owner/provenance fields directly to `ResourceMonitorFacts`

Why not:

- it would blur observational and inferred data
- it would pressure the live schema to carry partially-known or misleading
  attribution on every event

### Decision: Model attribution in explicit evidence levels

Future attribution claims should be classified into explicit evidence levels.
The initial design vocabulary is:

- **observed**: raw counters or direct snapshots with no ownership claim
- **best_effort_attribution**: repo-owned inference from explicit
  instrumentation plus known runtime context
- **causal_explanation**: stronger claim only when the repo has a direct,
  justified reason to connect a change to a specific owner or operation

Why:

- the same report may legitimately mix hard observations with weaker ownership
  hints
- evidence levels make it possible to be useful without pretending every claim
  is equally certain

Alternative considered:

- a single generic “confidence” float or string label

Why not:

- evidence categories are easier to reason about and easier to document in
  tests/specs than a freeform scoring system

### Decision: Use explicit attribution windows around known operations

Future provenance work should instrument named attribution windows around known
runtime operations such as model loading, optimizer creation,
`accelerator.prepare(...)`, cache preparation, or checkpoint save, rather than
trying to infer ownership from phase boundaries or session deltas alone.

An attribution window should be able to preserve:

- the triggering tag or operation name
- start/end observational snapshots
- any supporting runtime-trace tag/phase relationship
- candidate owners and their evidence level
- caveats and unattributed remainder

Why:

- explicit local instrumentation gives the repo a real basis for saying “this
  operation coincided with this change”
- existing phase rows are too broad to support honest ownership claims by
  themselves

Alternative considered:

- infer ownership only from existing `phase_start` / `phase_end` rows

Why not:

- the current phase windows are useful for narrowing time, not for proving
  owner

### Decision: Keep the initial owner vocabulary coarse and repo-owned

The first attribution model should use a coarse runtime-owner vocabulary rather
than pretending it can name every submodule or temporary buffer precisely. The
initial buckets should be things like:

- loaded model weights
- trainable parameters
- optimizer state
- activations / temporary workspace
- caches / preprocessing state
- checkpoint / serialization work
- unknown / unattributed remainder

Component labels may still be attached where the repo has a real declared
component surface, but component identity and runtime-owner category should stay
distinct.

Why:

- coarse buckets match what the repo can plausibly reason about from local
  instrumentation and existing component declarations
- this keeps the model useful for startup/trainer/report work without requiring
  impossible allocator-level certainty

Alternative considered:

- make the first model directly component- and submodule-granular

Why not:

- the repo does not currently have the evidence needed to make that precise in a
  trustworthy way

### Decision: Treat startup estimates and deep-mode counters as supporting evidence

Startup component estimates, runtime-trace spans, and deep-mode allocator
counters should be treated as supporting evidence for future attribution rather
than as attribution facts themselves.

Why:

- startup estimates are static or modeled values, not live ownership
- runtime trace narrows timing but does not assign owner
- deep-mode allocator counters reflect allocator behavior, not component
  identity

Alternative considered:

- project these existing signals directly into owner claims

Why not:

- each signal is useful, but none of them independently justifies a causal
  statement

### Decision: Unknown remainder is a first-class outcome

Future attribution outputs must be able to say that some or all of a resource
change is unattributed or unknown.

Why:

- temporary workspaces, allocator reuse, backend internals, and distributed
  behavior will often leave a remainder the repo cannot honestly assign
- forcing every delta into a named bucket would overstate correctness

Alternative considered:

- require every instrumented window to sum to named owners

Why not:

- that would reward misleading guesses over honest limits

## Risks / Trade-offs

- [Users may overread attribution as stronger than it is] -> Mitigation:
  preserve evidence levels and caveats in every attribution surface.
- [Instrumentation could sprawl across trainer/runtime code] -> Mitigation:
  align attribution windows with existing runtime-trace and lifecycle seams
  rather than inventing freeform probes everywhere.
- [Distributed/offload paths may not fit the initial owner model] -> Mitigation:
  keep the first owner vocabulary coarse and allow explicit unknown remainder.
- [The design may tempt early schema pollution] -> Mitigation: keep
  `ResourceMonitorFacts` observational by spec and route attribution through a
  separate surface.
- [Deep-mode counters may be mistaken for ownership proof] -> Mitigation:
  document them as supporting evidence only.

## Migration Plan

1. Land this design/spec change to settle terminology and boundaries.
2. Introduce a small repo-owned attribution record model separate from live
   resource facts.
3. Add explicit attribution windows around a narrow set of known operations,
   starting with startup/trainer/report seams.
4. Surface those records in debug/report payloads with evidence levels,
   caveats, and unknown remainder.
5. Only after the local/single-process path is credible, revisit wider
   distributed/offload semantics.

Rollback is straightforward because this change is design-only. Future
implementation should remain separable from the existing live resource schema.

## Open Questions

- Which repo-owned operations should be the first attribution windows:
  startup/model-load, optimizer creation, accelerator preparation, checkpoint
  save, or something else?
- Should future attribution records live only in report/debug payloads at first,
  or also file through observability metadata as a distinct fact type?
- How much of the current runtime-trace tag vocabulary can be reused directly
  before attribution needs narrower subphase tags?
- What is the minimum useful evidence/caveat shape for tests and docs without
  overengineering the first implementation slice?
