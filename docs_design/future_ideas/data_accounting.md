## Research Notes: Epochs, Buckets, and Exposure Accounting in an Asynchronous Data Pipeline

### Summary

A truly asynchronous data pipeline creates a fundamental scheduling problem:

* the system may know that data exists
* it may know enough metadata to structure that data
* but it may not yet have the fully prepared training payload for all of it

This creates tension between three goals:

* preserve useful epoch semantics
* avoid blocking training on total preprocessing completion
* avoid harming learning by exposing data in a staggered or biased way

The key architectural insight is that these concerns should be separated.

A sample can be:

* **known to exist**
* **known structurally**
* **ready for training**

Those are not the same thing, and they do not need to happen at the same time.

That distinction is what makes it possible to preserve buckets and some notion of epochs in an asynchronous system without requiring all latent/TE preparation to complete first.

---

## Why This Matters

One of the main motivations for asynchronous dataflow is to avoid workflows where training is held hostage by preprocessing — for example, having to wait days for caching before anything can begin.

But once training can start before all data is ready, an immediate question appears:

> what does an epoch mean if the data that exists, the data that is assigned to training structure, and the data that is actually ready are all different sets?

This is not only a technical batching problem. It is also a **learning-distribution problem**.

If early-ready data is repeatedly overexposed while late-ready data joins later, training can become unintentionally biased.

So this topic is really about:

* preserving structural training logic
* preserving fairness of exposure
* deciding how to account for partially ready data over time

---

## Key Architectural Observation

To reconcile asynchronous preparation with buckets and epochs, the system should separate:

### 1. Existence

The sample is known by metadata.

Possible known fields:

* path / URI
* hash
* width / height
* captions or text metadata
* class/source info
* train/val membership
* repeat or weighting metadata

### 2. Structural assignment

The sample has enough information to participate in training structure.

Examples:

* bucket assignment
* split assignment
* repeat logic
* dedup identity
* source/class grouping
* possible epoch/scheduler admission

### 3. Training readiness

The sample has all required payloads for the current mode.

Examples:

* latent cache ready
* TE cache ready
* token file entry ready if needed
* any additional mode-specific assets ready

This means future design should likely follow a rule like:

> **Everything needed to define training structure should be gathered early.**
> **Everything needed only for payload execution can become asynchronous.**

That is probably the cleanest line between “must block” and “does not need to block.”

---

## Relationship to the Broader Async Pipeline

This topic is a subproblem of the asynchronous data pipeline note.

The broader async pipeline needs:

* persistent sample state
* bounded queues
* background work execution
* ready-pool consumption

This note focuses on the next question:

> once samples become ready gradually, how should epochs, buckets, and actual training exposure be defined?

So this note should be read as the **scheduler/accounting layer** that sits on top of the async pipeline.

It also connects strongly to the previous notes:

### Relationship to online / streaming data

If samples can appear gradually or continuously, then strict full-dataset epoch semantics get even weaker. Exposure accounting becomes more important, not less.

### Relationship to large-scale caching

If cache readiness is incremental and shard population happens gradually, then the scheduler must decide how and when partially ready structured data becomes trainable.

---

## Buckets in an Async System

Buckets are much easier to preserve than strict traditional epochs.

Bucket assignment usually only needs:

* dimensions
* resize policy
* bucket settings

It does **not** require latent contents themselves.

That means bucket assignment belongs naturally to the **metadata / inspection / registry stage**, not the latent-ready stage.

This is important because it means an asynchronous system can still have stable bucket structure if it performs an early metadata pass.

### Design rule

Bucket membership should be derived from:

* dimensions
* model/data config
* resize/bucket policy

and stored in persistent sample state before full cache readiness is required.

This allows:

* bucket-aware preparation
* bucket-aware readiness tracking
* bucket-aware training selection
* shard planning for future large-scale cache layouts

---

## Metadata-First Dataset Structuring

A useful compromise is to allow an early metadata/inspection pass that establishes dataset structure before all payload preparation is complete.

This pass can gather:

* sample identity
* dimensions
* bucket assignment
* caption metadata
* split membership
* repeat metadata
* dedup identity
* source/class metadata

This still introduces some structural blocking, but it is much smaller and safer than blocking on:

* latent generation
* TE cache generation
* token preparation
* full cache completion

So the async goal is not necessarily to remove **all** blocking.

It is more specifically to reduce **payload blocking**, while allowing a lighter **structural blocking** pass if needed.

A practical split is:

### Pass A — Discovery / metadata gather

Purpose:

* build registry
* assign buckets
* define dataset structure

### Pass B — Background preparation

Purpose:

* create latent / TE / token readiness
* continue asynchronously

### Pass C — Epoch/view construction

Purpose:

* build trainable view from ready subset of known structured data

That is likely the most practical way to preserve buckets and some notion of epochs while removing full-cache startup delay.

---

## The Problem with Traditional Epochs

Traditional epoch semantics assume:

* the full dataset is known
* the full dataset is ready
* shuffling is over the full dataset
* every sample is seen according to repeats/weights within that epoch

In an asynchronous system, those assumptions break unless the pipeline reintroduces blocking.

If:

* the known set grows,
* the ready set grows,
* and samples join training at different times,

then “all data seen once” becomes ambiguous unless the system answers:

* all data that existed when?
* all data assigned when?
* all data ready when?

So strict old-style epoch semantics do not survive unchanged unless the system waits for full readiness.

That does **not** mean epochs are impossible.
It means they need a new definition.

---

## Three Candidate Epoch Models

### 1. Strict snapshot epoch

At epoch start:

* freeze the full known/assigned set
* require sufficient readiness before training proceeds
* anything not ready delays the epoch or is excluded by strict rule

Benefits:

* closest to traditional epoch meaning
* easier comparability/reproducibility

Costs:

* gives back a lot of blocking
* weakens async benefits

### 2. Ready-snapshot epoch

At epoch start:

* freeze the currently ready set
* build epoch only from that snapshot
* newly ready samples join later epochs

Benefits:

* simple
* compatible with async prep
* stable buckets and deterministic shuffle are still possible

Costs:

* epoch no longer means “all known data”
* late-ready samples enter later

### 3. Continuous / rolling epoch or window

Training consumes from the currently ready set continuously as it evolves.

Benefits:

* online-data-friendly
* minimal blocking

Costs:

* fuzzier semantics
* harder reproducibility
* harder fairness accounting

### Recommendation

The best first async-compatible epoch model is likely:

> **epoch = deterministic snapshot of the ready pool at epoch start**

This preserves:

* bucket stability
* deterministic shuffle
* easier debugging
* async overlap between prep and training

while avoiding the complexity of fully live scheduling too early.

---

## The Real Tension

There are three competing goals:

### 1. Preserve clean epoch semantics

“An epoch means all intended data is seen once according to repeats.”

### 2. Avoid waiting for total preprocessing

“Training should start before every latent/TE entry is ready.”

### 3. Avoid learning skew

“Early-ready data should not dominate while late-ready data is starved.”

A future design likely needs to treat these as tradeoffs rather than assuming all three come for free.

---

## Existing Set, Assignable Set, and Ready Set

A useful model is to explicitly distinguish:

### Existing set

All samples known by the registry/metadata pass.

### Assignable set

All samples currently allowed to participate in the training scheduling structure.

This may be:

* the full discovered set
* a filtered subset
* only samples admitted to the current training population

### Ready set

All samples whose payloads are train-ready right now.

### Seen count

How many times a sample has actually been used for training.

This distinction is critical. Without it, async scheduling tends to drift into:

* “whatever was ready first gets seen first and most often”

---

## Missing Bucket Coverage

A practical problem arises when:

* full dataset bucket structure is known
* but only part of each bucket is ready

Possible policies:

### Conservative bucket policy

Only include buckets with enough ready samples.

Benefits:

* stable batching
* easier batch quality control

Costs:

* early epochs may ignore some buckets entirely

### Aggressive bucket policy

Allow underfilled buckets immediately.

Benefits:

* training can start earlier

Costs:

* weaker batch structure
* noisier training distribution

### Threshold-based bucket policy

Require:

* minimum total ready samples
* minimum per-bucket readiness for included buckets
* defer sparse buckets until later epochs

### Recommendation

A threshold-based policy is likely the safest first version.

---

## Exposure Accounting Problem

Once samples become ready at different times, preserving fair training exposure becomes a separate problem.

If training starts from partial readiness, then:

* early-ready samples can be oversampled
* late-ready samples can be starved
* data distribution can drift over time

So async scheduling probably needs explicit **exposure accounting**.

At minimum, the system should track:

* when a sample was discovered
* when it was assigned
* when it became ready
* how many times it has been seen
* bucket/source/class membership

Without this, there is no good way to measure or correct staggered exposure bias.

---

## Scheduler Models

### Model A — Ready-snapshot epochs only

Build each epoch from the ready set at epoch start and use ordinary shuffle/repeat logic inside that snapshot.

Benefits:

* simplest
* easiest to implement
* easiest to reason about

Costs:

* only coarse correction of staggered readiness
* late-ready samples may lag across epochs

### Model B — Carryover balancing across epochs

Track seen counts across epochs and preferentially choose under-seen ready samples in later epochs.

Benefits:

* smoother correction
* still retains epoch structure

Costs:

* more scheduler complexity

### Model C — Continuous deficit/debt scheduling

Drop strict epoch-first scheduling and instead maintain target vs actual exposure over time.

This is the most interesting longer-term model.

---

## Deficit / Debt Scheduling

A useful way to think about async fairness is:

* every sample has a **target exposure**
* every sample has an **actual exposure**
* the difference is its **deficit** or **debt**

```text id="mzc8lc"
deficit = target_seen - actual_seen
```

Then the scheduler prefers ready samples with the largest deficit, subject to bucket/batch constraints.

This directly addresses:

### Early-ready oversampling

If a sample is seen many times early, its deficit shrinks, so it naturally becomes lower priority.

### Late-ready catch-up

When a sample becomes ready late, if target exposure has already been accumulating, it may begin with a larger deficit and get prioritized.

### Growing datasets

If the known set grows over time, newly assigned samples can begin accumulating target exposure from a defined point.

This is probably the cleanest long-term answer for truly async or online-capable training, but it is also significantly more complex than snapshot epochs.

---

## When Should Target Exposure Start?

A crucial policy choice is when a sample begins “deserving” training exposure.

Possible options:

### 1. From existence time

As soon as metadata says the sample exists.

Benefit:

* most globally fair

Cost:

* creates backlog for not-yet-ready samples

### 2. From assignment time

Exposure starts when the sample enters the current training population.

Benefit:

* clean middle ground
* separates discovered data from actively scheduled data

### 3. From ready time

Exposure starts only when sample is train-ready.

Benefit:

* simplest accounting

Cost:

* does not compensate for prep delay bias

### Recommendation

The most sensible long-term choice is likely:

> start target exposure from **assignment time**, not raw existence time and not only ready time

That gives the system a meaningful fairness baseline without overcommitting to everything merely discovered.

---

## Catch-Up Behavior

If a sample becomes ready late and has accumulated deficit, should it immediately catch up aggressively?

Possible policies:

### Aggressive catch-up

Late-ready samples are strongly prioritized until caught up.

Pros:

* fairer globally

Cons:

* can create sudden distribution shifts

### Soft catch-up

Deficit influences priority, but only gently.

Pros:

* smoother training dynamics

Cons:

* slower fairness correction

### Bounded catch-up

Deficit influences priority, but its effect is capped per window/epoch.

### Recommendation

If deficit-based scheduling is introduced later, bounded catch-up is probably the safest default.

---

## Bucket-Aware Deficit Scheduling

Buckets do not break deficit scheduling. They just make it hierarchical.

A useful structure is:

### Layer 1 — choose bucket

Choose bucket based on:

* desired bucket proportion
* current ready population
* bucket-level deficit or under-service

### Layer 2 — choose samples within bucket

Within that bucket, choose ready samples by:

* highest deficit
* repeats/weights
* randomness for smoothing

This is much more practical than imagining one giant half-filled global epoch list with placeholders.

A better mental model is:

* each bucket has a ready pool
* each sample has target and actual exposure
* scheduler keeps choosing the most under-served ready items while preserving batch structure

---

## Practical Evolution Path

A full continuous deficit-based scheduler is probably not the right first implementation.

A better progression is:

### Phase 1 — Ready-snapshot epochs

* epoch built only from ready set at epoch start
* deterministic shuffle inside the snapshot
* ordinary repeat logic
* actual seen counts tracked

### Phase 2 — Cross-epoch balancing

* track sample seen counts across epochs
* bias later epoch selection toward under-seen ready samples
* begin correcting async skew without abandoning epochs

### Phase 3 — Deficit-aware scheduling

* maintain target vs actual exposure
* possibly use epoch only as reporting window
* support gradually expanding assigned/ready populations more gracefully

This staged path gives you a practical first model while leaving room for a richer scheduler later.

---

## What an Epoch Could Mean in Practice

In an async-compatible system, an epoch probably needs to be redefined.

Instead of:

> one pass over all data that exists

it could mean one of:

### 1. One pass over all currently ready data at snapshot time

Best first practical meaning.

### 2. One reporting window over a continuously scheduled system

Best for future deficit/online mode.

### 3. One pass over all assigned data, but only after readiness thresholds are met

Best for stricter comparability, but more blocking.

### Recommendation

For the first async version:

> **epoch = one deterministic pass over the ready snapshot at epoch start**

That is likely the cleanest compromise.

---

## Important Risks

### 1. Early curriculum bias

Fast-to-prepare samples may dominate early training.

### 2. Bucket imbalance

Some buckets may be underrepresented until readiness catches up.

### 3. Source/class imbalance

If readiness differs by source or category, distribution drift can appear.

### 4. Reproducibility loss

Fully live scheduling makes experiments harder to compare.

### 5. Scheduler complexity

Deficit-based correction is powerful but easy to overcomplicate.

---

## Observability Needs

If the async scheduler becomes more sophisticated, the system should probably monitor:

* sample seen counts
* per-bucket seen counts
* source/class coverage
* readiness lag
* average deficit
* max deficit
* per-bucket under-service
* number of samples discovered vs assigned vs ready vs seen

Without this, it will be very hard to tell whether staggered training is behaving sensibly.

---

## Open Research Questions

* Should async mode keep strict epoch semantics at all, or only use epochs as snapshots/reporting windows?
* What is the right minimum readiness threshold before training should start?
* How should samples move from existing set into assignable set?
* Should assignment be static or dynamic?
* Should target exposure start at existence, assignment, or readiness?
* How aggressively should late-ready samples catch up?
* How much skew from staggered exposure is acceptable in practice?
* Should buckets be allowed to appear gradually, or should some minimum coverage be required first?
* How should distributed training handle per-rank fairness under async readiness?
* When does the scheduler become complex enough that step-based or window-based training is preferable to epoch language?

---

## Recommendation

Treat this as a **scheduler and accounting problem**, not just a queueing problem.

The likely first practical solution is:

> preserve bucket assignment through an early metadata/inspection pass
> build epochs from deterministic snapshots of the ready pool
> track actual sample exposure from the beginning
> and only later consider more advanced deficit-based scheduling

That gives a sane progression:

* bucket structure remains stable
* training no longer waits for full preprocessing
* early async behavior remains debuggable
* future online/streaming support stays possible
* fairness corrections can be introduced gradually instead of all at once

Longer term, a more advanced async/online-capable system may move toward:

> **existing set + assignable set + ready set + exposure accounting**

with epoch becoming either:

* a snapshot boundary, or
* a reporting window over a deficit-aware scheduler

---

## Short roadmap version

* [ ] Research **epoch semantics for async-ready data**
* [ ] Preserve **bucket assignment** through metadata-first inspection, independent of latent readiness
* [ ] Distinguish **existing**, **assignable**, and **ready** sample sets
* [ ] Track **actual seen count** per sample from the start
* [ ] First async scheduler model: **ready-snapshot epochs**
* [ ] Add **bucket readiness thresholds** for safer early training
* [ ] Investigate cross-epoch balancing for under-seen ready samples
* [ ] Research long-term **deficit/debt-based exposure scheduling**
* [ ] Add observability for sample/bucket/source exposure and readiness lag
* [ ] Treat epoch as a potentially evolving concept: full-dataset pass, ready snapshot, or reporting window depending on mode
