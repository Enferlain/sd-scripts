What you’re describing is basically an **empirical resource profile system**:

- launch a config in a controlled way
- record a stable identity for it: config hash, seed, code revision, hardware/runtime facts
- capture resource behavior at key runtime events
- store those observations
- reuse the stored profile later as a **measured prior** instead of a pure estimate

That is meaningfully different from the provenance topic.

**Why this is attractive**

It fits your performance concern well:
- normal runs do not need heavier diagnostics
- the cost is paid in dedicated profiling runs, not in every training run
- later runs can consult the profile database with effectively no runtime penalty

So instead of saying:
- “optimizer state is estimated from parameter counts”

we could eventually say things more like:
- “on this hardware/runtime/config family, startup usually lands in this range”
- “first optimizer step tends to add this much”
- “checkpoint save causes a transient spike around this range”
- “this preset historically peaks here during `accelerator.prepare(...)`”

That’s much better than hand-modeled estimates when the behavior is stable enough.

**What it allows us to do**

If done well, it could support:

- better startup resource forecasts before full training
- replacing some static estimates with measured ranges
- config-risk warnings like “this config family historically exceeds 24 GB during first train step”
- regression detection across code changes
- environment-aware modeling: same config, different GPU/driver/backend behavior
- better cache/planning decisions
- eventually smarter heuristics around offload, tiling, batch sizing, or validation cadence

It also gives us a path toward a **resource model** without paying continuous provenance overhead in the hot path.

**Important nuance**

I would not call these “concrete numbers” in the same sense as live observed counters.

They would be:
- **empirical predictions**
- **historical ranges**
- **profile-backed expectations**

That’s still very valuable, but different from:
- exact live ownership
- exact current-run attribution

So this can replace a lot of weak estimates, but not all of them.

**Why it won’t solve everything alone**

A config profile can drift based on things beyond the YAML itself:

- hardware and VRAM size
- CUDA/PyTorch versions
- xformers / attention backend
- deepspeed/offload/distributed mode
- model weights and actual loaded components
- dataset shape and bucket distribution
- cache state
- first-run allocator warmup / fragmentation
- whether validation or checkpointing is enabled
- resume state versus fresh run

So the profile key probably needs to include more than just “config hash.”
It likely needs something like:

- config hash
- git SHA
- model family / loaded component signature
- hardware/runtime signature
- relevant backend toggles
- maybe dataset/resource-shape fingerprints

**Best version of the idea**

I think the cleanest version is not “dry launch” in the literal sense, but something like:

- **probe run**
  A bounded profiling execution that intentionally stops after key milestones.
- Example milestones:
  - startup complete
  - accelerator prepared
  - first batch loaded
  - first forward/backward
  - first optimizer step
  - first checkpoint save
  - optional validation pass

That gives us measured event-level data with low conceptual ambiguity.

A truly “dry” launch that never exercises important runtime paths would miss the most useful resource behavior.

**How this relates to the current topics**

This sits adjacent to both:

- **resource provenance**
  tries to explain where a change came from in a specific run
- **empirical profiling/modeling**
  learns what usually happens for similar runs from prior observations

Those are related, but not the same.

I’d think of your idea as a third lane:
- live observation
- attribution/provenance
- offline empirical modeling

And honestly, your lane may be the better ROI path for many practical decisions.

**Why this is nice architecturally**

The repo already has pieces that fit this direction:

- resource monitor events
- benchmark report payloads
- metadata runtime / storage
- runtime trace milestones
- run/config/environment facts

So we already have most of the raw capture surface.
What’s missing is the explicit concept of:
- profile runs
- stable profile identity
- aggregation/modeling over many runs
- “use prior profile instead of estimate” consumers

**My take**

I think this is one of the best next-step ideas in the area.

It avoids the trap of:
- making normal training slower
- pretending we can always know exact live ownership
- overengineering provenance before we have enough observed history

And it gives us something directly useful:
- profile-backed forecasts and ranges
