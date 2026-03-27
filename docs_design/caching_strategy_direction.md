# Caching Strategy Direction (Deferred Architecture Note)

## Status

This is a **design direction note**, not an implementation plan.

The current caching system is working and should not be churned immediately
just to satisfy architecture purity. This note exists so that when new model
families are introduced, we have a recorded whole-system direction to compare
against the active shape.

## Problem Statement

The current caching design splits model-specific caching behavior across
multiple peer classes:

- the training-facing caching facet in
  [`library/strategies/base/contracts.py`](/mnt/d/Projects/sd-scripts/library/strategies/base/contracts.py)
- model-family caching facet implementations such as
  [`library/strategies/sd/caching.py`](/mnt/d/Projects/sd-scripts/library/strategies/sd/caching.py)
  and
  [`library/strategies/sdxl/caching.py`](/mnt/d/Projects/sd-scripts/library/strategies/sdxl/caching.py)
- model-family `CacheBackend` implementations created by those facets
- optional model-specific cache payload concepts such as `SdxlConditioning`

That means the explicit strategy surface is not the whole model-specific caching
surface. A new model author can implement the visible caching facet and still
need to discover additional model-specific classes by tracing internals.

## What Feels Wrong In The Current Shape

Today the active design is roughly:

```python
class SdxlCachingStrategy(CachingStrategy):
    def create_latent_caching_strategy(self, cfg):
        return SdxlLatentsPipelineStrategy(...)

    def create_te_caching_strategy(self, cfg):
        return SdxlTextEncoderPipelineStrategy(...)
```

with separate peer classes:

- `SdxlLatentsPipelineStrategy`
- `SdxlTextEncoderPipelineStrategy`
- `SdxlConditioning`

This is tidy from the point of view of the generic caching engine, but it makes
the model-specific training surface feel fragmented:

- one class exists to manufacture more model-specific classes
- discoverability depends on tracing call paths rather than reading the
  explicit strategy contracts
- the data layer ends up carrying model-specific concerns farther than is ideal

## Current System Strengths

The current design is not arbitrary. It does have real strengths:

- `CachingEngine` talks to a small backend interface
- latent and text-encoder caching can vary independently
- the generic data pipeline stays simple from the engine's point of view
- `CacheBackend` is reusable in tests and benchmarks without instantiating a
  full training strategy

These are real benefits and should not be ignored.

## Current System Weaknesses

The same design also creates the long-term architecture pressure:

- the real extension points are broader than the explicit strategy contracts
- the model-specific caching concern is split across 3-4 peer classes per model
  family
- new model work is less discoverable than it should be
- support concepts like model-specific conditioning payloads start to feel
  stranded between the data and strategy layers

## Long-Term Direction

If we optimize for the cleanest long-term architecture, the best direction is:

**make `CachingStrategy` itself the full model-specific caching surface.**

In that design:

- `CachingStrategy` owns latent caching behavior directly
- `CachingStrategy` owns text-encoder caching behavior directly
- `CachingEngine` remains a generic orchestration helper
- `CachingEngine` calls the caching facet directly instead of a separate
  model-specific `CacheBackend`
- model-specific caching behavior becomes discoverable by reading the explicit
  strategy contract

Conceptually, the target shape is closer to:

```python
class CachingStrategy(ABC):
    def encode_latent_batch(...)
    def save_latent_cache(...)
    def load_latent_cache(...)
    def is_latent_cache_valid(...)

    def encode_te_batch(...)
    def save_te_cache(...)
    def load_te_cache(...)
    def is_te_cache_valid(...)
```

with model families implementing those methods directly in their caching facet.

## Why This Direction Wins Long-Term

This direction is preferred because it:

- keeps the explicit strategy contracts as the real extension surface
- reduces the number of model-specific peer classes
- improves discoverability for new model implementations
- keeps the strategy system legible: implement the contracts, not the contracts
  plus hidden follow-on classes

The important point is not whether the latent/TE split remains. It should.
The important point is whether that split is exposed as:

- direct caching-facet behavior

or:

- factory methods that manufacture more model-specific classes

The first is more honest and easier to extend.

## Tradeoffs

### Upsides

- clearer extension story for new model families
- fewer peer abstractions to keep in sync
- better alignment between architecture docs and actual implementation work
- easier to open one strategy file and understand the model-specific caching
  surface in one place

### Downsides

- the caching facet becomes heavier
- refactor cost is non-trivial because caching phase code, dataloader code,
  engine wiring, tests, and SD/SDXL implementations all need coordinated
  updates
- the current "tiny backend interface" neatness from the engine's point of view
  becomes less central
- if the repo later needs many interchangeable cache backend implementations per
  model family, that polymorphism becomes a concern to solve behind the caching
  facet instead of in front of it

## Comparison To The Conservative Option

The conservative alternative is:

- keep `CacheBackend`
- improve docs and naming
- accept the extra backend classes as part of the architecture

That is a valid short-term choice, especially while the active SD / SDXL path is
still being stabilized.

But it is weaker long-term because it accepts hidden extension points and tries
to document them better instead of removing them.

## Conditioning / Payload Implication

The same scrutiny applies to model-specific payloads such as `SdxlConditioning`.

If a model-specific payload only exists because the generic data path needs to
transport it, that should be re-examined when this caching redesign is revisited.

The preferred long-term direction is:

- keep generic data structures generic where possible
- reconstruct model-specific helper objects in model-family code when practical
- avoid introducing shared marker hierarchies unless they buy something beyond
  nominal typing

This does **not** mean `SdxlConditioning` is wrong today.
It means that once caching is reconsidered for future model families, the same
whole-system ownership question should be asked for cache payloads as well as
cache backends.

## Deferred Decision

This note records a preferred long-term direction, but it is intentionally
deferred.

We should revisit it when one of these becomes true:

- a new model family is being implemented
- the caching surface starts to feel materially harder to extend
- `CacheBackend` or model-specific cache payloads start causing architecture
  drift in additional subsystems

Until then, the active design can remain in place.
