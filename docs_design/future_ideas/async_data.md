## Research Notes: Truly Asynchronous Data Pipeline

### Summary

The current data system already has a strong **conceptual phase split**:

* scan
* cache
* epoch prep
* training

but it still executes those phases as **global barriers**:

* phase 1 finishes
* then phase 2 finishes
* then phase 3 finishes
* then phase 4 starts

This means the system is **staged**, but not yet **truly asynchronous**.

That distinction matters because one of the original motivations for the data-layer rework was to avoid workflows where preprocessing becomes a hard blocker — for example, waiting hours or even days for caching to finish before training can start at all.

The long-term goal should therefore not be “make caching faster” in isolation.
It should be:

> allow data-related concerns to run on their most appropriate device or execution path, overlap where safe, and avoid holding training hostage to full upfront preprocessing completion

This includes more than latent caching. It also touches:

* dataset discovery and registry updates
* image inspection and bucket assignment
* caption loading / mutation
* latent generation
* TE output generation
* token preparation
* cache writes
* batch readiness
* training consumption

So the future async pipeline should be understood as a **dataflow architecture**, not just a multithreaded cache builder.

---

## Why This Matters

The current barrier-based flow is operationally safe, but it creates several workflow limitations:

* training cannot begin until full preprocessing stages are complete
* expensive caching can delay iteration by many hours or days
* training remains blocked even if a useful subset of data is already ready
* the current architecture separates concerns logically, but not in runtime overlap
* different concerns cannot naturally execute on their best device without global waits

The desired future state is:

* expensive preparation no longer blocks all progress
* work can be executed on the device best suited for it
* training begins as soon as a sufficiently ready subset exists
* background preparation can continue while training runs
* throughput remains bounded by the actual bottleneck, not by phase barriers
* correctness and reproducibility remain explicit through persistent state tracking

---

## Key Architectural Observation

The current 4 phases should likely remain as **conceptual responsibilities**, but they should stop acting as **hard global runtime boundaries**.

Future design should preserve:

* scan / inspection responsibility
* cache / preprocessing responsibility
* epoch/view preparation responsibility
* training consumption responsibility

but evolve runtime behavior from:

> dataset-level phase completion

to:

> sample-level state progression

That is the main architectural shift.

Instead of saying:

* “the dataset is now in phase 2”

the system should eventually be able to say:

* this sample is discovered
* this one is bucketed
* this one has latent cache
* this one has TE cache
* this one is ready for training
* this one is failed or quarantined
* this one belongs to the current epoch snapshot

That stateful view is what makes asynchronous flow possible without losing control.

---

## Relationship to Previous Notes

This async pipeline note depends heavily on the previous two directions.

### Relationship to online / streaming data

A truly asynchronous data system is the foundation that would later make online or streaming data practical.

Without:

* bounded queues
* persistent sample state
* ready-pool semantics
* incremental preparation

there is no clean way to support:

* datasets that are not fully prebuilt
* continuously arriving samples
* remote or lazy materialization
* background preparation while training continues

So the streaming/online-data direction likely depends on the async-pipeline direction first.

### Relationship to large-scale caching architecture

Large-scale caching also depends on async dataflow.

Without:

* registry-based sample state
* explicit readiness
* shard-aware cache lookup
* background work orchestration

it is much harder to:

* populate sharded caches incrementally
* migrate away from global giant-manifest assumptions
* feed training from partially ready but valid subsets
* avoid cache-building as a monolithic stop-the-world phase

So the large-scale registry/cache redesign and the async-pipeline redesign are deeply linked.

---

## Current Architecture

### Current conceptual flow

```text id="ebwtwu"
PHASE 1: Scan
  - dataset scanning
  - manifest creation
  - bucket assignment
  - caption loading

PHASE 2: Cache
  - latent caching
  - TE output caching
  - distributed cache production

PHASE 3: Epoch Prep
  - shuffle
  - augmentation
  - warmup ordering
  - optional token-file generation

PHASE 4: Training
  - batch loading
  - tokenization / token-file usage
  - training consumption
```

### Current strengths

* responsibilities are already cleanly separated
* phase-specific logic is easier to test and reason about
* correctness and debugging are simpler
* distributed and caching semantics are easier to control

### Current limitation

The phases are still executed as **global barriers**, so the system does not yet gain the throughput or usability benefits of runtime overlap.

---

## Original Async Sketch and Its Role

The original planned pipeline was:

```text id="0zjrgv"
LOAD IMAGE -> RESIZE/CROP -> VAE ENCODE -> SAVE DISK
```

with queues between stages.

That design is still valuable, but it should now be interpreted as:

> a **sub-pipeline for one preparation path**, not the complete future data architecture

It correctly recognized that:

* different stages have different bottlenecks
* disk I/O, CPU work, GPU work, and disk writes should not block each other
* bounded queues help smooth throughput
* batching GPU work improves efficiency

But it is no longer sufficient by itself because the full future data layer also needs to model:

* persistent sample state
* cache readiness
* TE outputs as a separate concern
* partial data readiness
* failure handling
* training overlap
* snapshot/live-ready consumption
* possible future streaming inputs
* large-scale sharded registry/cache storage

So the original sketch is still useful, but it should now be treated as one piece of a broader async architecture.

---

## Recommended Future Architecture

The future async data layer likely needs **three planes**.

### 1. Persistent state plane

This is the canonical registry / manifest evolution.

Responsibilities:

* stable sample identity
* source paths / URIs / hashes
* dimensions and bucket assignment
* caption metadata
* cache namespace membership
* readiness state
* failure/retry state
* shard membership / row mapping
* timestamps / versioning

This is the persistent truth that allows async work to remain correct.

### 2. Work execution plane

This is the queue/worker system.

Responsibilities:

* discovery and inspection work
* image loading / decode / normalization
* latent preparation
* TE output preparation
* cache writes
* registry updates
* backpressure
* retries and failure isolation

This is where concurrency lives.

### 3. Consumption plane

This is the training-facing side.

Responsibilities:

* selecting ready items
* snapshotting or windowing ready pools
* batch collation
* shard-aware reads
* tokenization / token-file usage
* trainer-facing batch delivery

This is the hot path and should stay as narrow and predictable as possible.

---

## Proposed Runtime Model

The runtime should likely evolve from:

> phase-oriented full-dataset execution

to:

> queue-driven sample/job progression with explicit readiness

A likely conceptual flow would be:

```text id="2v3rj1"
discovery / scan
  -> inspect + bucket
  -> register sample state
  -> latent prep queue
  -> TE prep queue
  -> cache write queue
  -> mark sample ready
  -> ready-pool / epoch snapshot
  -> training consumption
```

This is intentionally not modeled as one single linear conveyor belt, because in practice:

* some samples may already be cached
* some may need only latent preparation
* some may need latent + TE
* some may fail inspection
* some may be skipped
* some may be deferred

So the future architecture should likely be **job- and state-driven**, not just “thread 1 → thread 2 → thread 3 → thread 4.”

---

## Core Design Shift: Sample State Machine

A truly async data pipeline probably requires an explicit sample lifecycle.

Possible states:

* `discovered`
* `inspected`
* `bucketed`
* `latent_pending`
* `latent_ready`
* `te_pending`
* `te_ready`
* `ready`
* `failed`
* `quarantined`

Depending on mode, readiness may also need more precise distinctions such as:

* ready for latent-only training
* ready for latent+TE-cached training
* ready for token-file epoch use

This state model is likely one of the key prerequisites for making async flow correct and observable.

---

## Concurrency and Execution Notes

### Important principle

“Async” should not simply mean “run everything at once.”

The actual goal is:

> overlap concerns where it improves throughput, while respecting device ownership, memory limits, and training performance

That means the pipeline should care about **where** work is best performed.

### Likely execution tendencies

**Disk / file work**

* background threads or workers
* prefetch / write queues
* should not block GPU

**CPU-heavy prep**

* decode
* resize / crop
* metadata inspection
* tokenization
* caption transforms
* good candidates for worker processes

**GPU-heavy prep**

* VAE encode
* TE encode
* should be batched carefully
* should not automatically compete with active training on the same device without policy control

### Important caution

Not all overlap is good overlap.

If the same GPU is asked to:

* run training
* encode latents
* encode TE outputs

at the same time, then:

* VRAM contention increases
* compute contention increases
* fragmentation risks increase
* training throughput may get worse instead of better

So the async design should explicitly separate:

* **asynchronous preparation**
  from
* **unrestricted concurrent GPU-heavy execution**

CPU/I/O overlap is usually safer than blindly overlapping all GPU work with training.

---

## Ready Pool and Training Start

One of the main workflow goals is to avoid waiting for full preprocessing completion.

So training should eventually be able to begin once there is a **sufficient ready subset**.

This likely implies a future model where:

* background preparation can continue
* training consumes from a ready pool
* epoch/view preparation operates on a current snapshot of ready items

This is the main mechanism by which “wait two days for caching” turns into “start training when enough data is ready, while the rest continues preparing.”

---

## Snapshot vs Live-Ready Consumption

A major design choice is whether training should consume:

### 1. Snapshot-ready data

At epoch start, freeze the current ready set and train over that view.

Benefits:

* easier reproducibility
* easier debugging
* clearer epoch semantics
* simpler trainer expectations

### 2. Live-ready data

Training draws from whatever is currently ready as the run progresses.

Benefits:

* more dynamic
* more online-data-friendly
* less idle delay

Costs:

* fuzzier epoch semantics
* harder reproducibility
* more complicated fairness and sampling behavior

### Recommendation

Start with **snapshot-per-epoch over the ready pool**.

That still allows overlap between background preparation and training, while keeping the training side much easier to reason about.

---

## Interaction with Large-Scale Caching

The async pipeline and large-scale cache design should probably evolve together.

The async layer needs:

* registry-backed sample state
* shard lookup
* explicit readiness

The large-scale cache layer needs:

* incremental population
* non-blocking production
* shard membership updates
* background write paths
* no dependence on monolithic full-dataset completion

So a future sharded cache design should likely be built assuming:

* cache entries become available incrementally
* registry is updated as shards are filled
* training may consume only currently ready shards/samples
* cleanup/compaction happen outside the hot training path

This is another reason to move away from “one big JSON manifest + stop-the-world caching pass.”

---

## Interaction with Online / Streaming Data

The async pipeline is also the likely prerequisite for online-data support.

If the system eventually wants to support:

* remote lazy data
* incrementally discovered samples
* live ingest
* synthetic sample generation
* active learning loops

then it will already need:

* sample state transitions
* bounded queues
* readiness tracking
* ready-pool consumption
* snapshot or window semantics

So the future async data pipeline should likely be designed in a way that naturally extends into streaming/online support later, even if the first implementation still assumes a bounded known dataset.

---

## Recommended Evolution Path

A full async rewrite is probably not the right first move.

A better staged path would be:

### Phase 1 — Internal async cache production

Make current cache production internally asynchronous.

Goals:

* overlap disk load, CPU prep, GPU encode, and cache write
* preserve current training barrier behavior initially
* improve cache-building throughput first
* validate queueing/backpressure design

This is the natural place for the original 4-stage sketch to land.

### Phase 2 — Registry-backed readiness

Add persistent sample state and readiness tracking.

Goals:

* know which samples are ready
* know which cache concerns are complete
* enable partial progress without ambiguity
* prepare foundation for non-blocking training start

### Phase 3 — Training from ready snapshots

Allow training to begin from a ready subset instead of waiting for total cache completion.

Goals:

* epoch prep operates on current ready snapshot
* background cache prep can continue while training runs
* training no longer waits for full dataset caching

### Phase 4 — Shard-aware async large-scale path

Tie async preparation into sharded cache / registry design.

Goals:

* registry points to shard membership
* cache production can fill shards incrementally
* batch reads can be grouped by shard
* large-scale cache no longer assumes stop-the-world prebuild

### Phase 5 — Online / streaming-capable dataflow

Only after the above is stable, extend the model toward:

* incremental sample discovery during training
* live ingest
* windowed/snapshot semantics
* future online-data workflows

---

## Important Problems to Solve

### 1. Sample readiness definition

Need a precise answer for when a sample is considered trainable in each mode.

### 2. Backpressure

Need bounded queues and clear policies so async work does not run away with RAM/disk usage.

### 3. GPU contention

Need explicit policy for when GPU-heavy prep is allowed to overlap with training.

### 4. Snapshot semantics

Need a clear definition of what an “epoch” means when readiness changes over time.

### 5. Failure isolation

Bad or slow samples should not poison the whole pipeline.

### 6. Observability

Need queue depth, stage throughput, readiness counts, and blocked-stage diagnostics.

### 7. Debuggability

Async flow is more efficient, but harder to reason about unless state transitions are explicit.

---

## Open Research Questions

* How much async overlap is beneficial before complexity outweighs gains?
* Which prep stages should be allowed to overlap with active training?
* Should latent prep and TE prep use the same job system or separate ones?
* What should the minimum ready threshold be before training begins?
* Should async mode use epoch snapshots, rolling windows, or step-based semantics?
* How should queue sizes be tuned for different hardware/storage setups?
* How should multi-GPU systems divide training and prep responsibilities?
* How tightly should async preparation be coupled to the future sharded-cache design?
* Should online-data support be a later extension of this system, or a first-class design target now?

---

## Recommendation

Treat this not as “add threads to caching,” but as a broader evolution toward:

> **persistent sample state + bounded async work queues + ready-pool consumption**

while preserving the current conceptual phase ownership.

The next practical goal should probably be:

> make cache production internally asynchronous first,
> then let training consume from ready snapshots,
> then tie that model into registry-backed large-scale cache and future online-data support

This keeps the system grounded in the realities that motivated the work in the first place:

* people do not want to wait days for preprocessing before training can even begin
* async is not just about caching speed
* each concern should run where it makes the most sense
* overlap should help training, not interfere with it
* the data layer should become a real runtime system, not just a sequence of preprocessing phases

---

## Short roadmap version

* [ ] Research **truly asynchronous data pipeline** beyond global phase barriers
* [ ] Preserve current phase ownership, but evolve runtime from **dataset-level barriers** to **sample-level state progression**
* [ ] Add **persistent sample readiness/state tracking**
* [ ] Make cache production internally async with bounded queues and backpressure
* [ ] Allow training to begin from a **ready subset** instead of waiting for total cache completion
* [ ] Prefer **snapshot-per-epoch over ready pool** as the first async training model
* [ ] Treat GPU-heavy prep overlap carefully; async should not automatically mean unrestricted same-device concurrency
* [ ] Tie async pipeline design to future **large-scale registry + sharded cache** work
* [ ] Treat async pipeline as a prerequisite/foundation for future **online / streaming data** support
* [ ] Add observability for queue depth, readiness counts, stage throughput, and blocked-stage diagnostics
