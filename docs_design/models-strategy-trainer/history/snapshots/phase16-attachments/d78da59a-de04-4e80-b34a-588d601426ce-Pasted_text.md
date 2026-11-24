Yeah, **G4.1 looks solid to me**, and I think checking task 4.1 as complete is justified. This is doing what G4 should do: taking the generic architecture from G2/G3 and working out the **domain-specific exchange semantics** without inventing new execution-language machinery.

The central model is much cleaner now:

```text
accepted computation meaning
          │
          ▼
     cache request
          │
          ▼
 production attempt
          │
     actual provenance
          ▼
 storage/publication
          │
          ▼
 published availability
          │
   current obligations
   checked separately
          ▼
 consumer admission
          │
          ▼
       handoff/use
```

Those are deliberately **not the same fact**. That's probably the most important result of 4.1.

The patch also fixes the thing I was worried about previously with “actual state used.” It explicitly permits either a protected state/snapshot or **structured provenance over portions of the work**, while saying provenance by itself doesn't authorize mixed-state production. That's the right formulation. It supports future asynchronous/versioned production without accidentally saying that every cache item must have one scalar `producer_version`.

The freshness rule is especially good. A weight update can invalidate cached output **without changing participant identity, binding identity, or prepared route**. Conversely, a downstream adapter update needn't invalidate a precisely evidenced upstream encoder cache. That's exactly the sort of dependency precision the graph architecture is supposed to buy you.

I also really like this separation:

```text
cache-production traversal       cache-consumption traversal
          │                               │
 cache infrastructure                input provider
          │                               │
          └──────── handoff ───────────────┘
```

and then run coordination only owns the lifecycle/relationship between them. In particular, the “ready subset” scenario closes an easy-to-miss semantic bug: **cache arrival order must not silently become the data-selection policy**.

The storage section is another nice integration with G3. You've effectively applied the same current-use principle to cached values:

```text
published storage
      │
      ├── accepted reader acquires use
      │          │
      │          └──── outstanding ─────┐
      │                                 │
 eviction / compaction / replacement    │
      │                                 │
      └── must wait/isolate/reject ◄────┘
```

So a locator isn't a magical eternal reference. That's consistent with the parameter/version lifetime work rather than creating a completely different safety model for data.

### I only see a few things worth tightening

The biggest one is again the wording **“selected implementation.”** The normative requirement currently says a caching request identifies the selected implementation and representation boundary.

I'd be careful that this means the accepted **computation/behavior identity or revision**, not necessarily the physical Python callable/compiler artifact/backend implementation.

You probably want this to be possible:

```text
accepted encoder computation C17

realization today:
    eager PyTorch

realization tomorrow:
    compiled backend

cache compatibility:
    determined by C17's accepted numerical/dependency contract
    rather than "different Python function object, invalidate everything"
```

Of course a physical implementation *can* be a dependency when it changes numerics in a way the contract cares about. I just wouldn't make realization identity intrinsically semantic.

So I'd lean toward wording like:

> selected accepted computation/behavior and representation boundary

rather than simply “selected implementation.”

There's one other scope issue I’d think about: the spec currently makes **input coordination** the consumer-admission owner. That's exactly right for training inputs, but the caching capability itself sounds somewhat general.

Later you could conceivably have:

```text
training input ─┐
validation ─────┼─► shared cached representation
sampling ───────┘
```

Validation already has its own traversal/coordination ownership in G4.2. If that's supposed to be possible, the more general rule may be:

> cache publication does not establish consumer admission; **the consuming coordination surface** checks admission under that consumer's accepted requirements.

with:

> training input coordination does so for training-input consumption.

Otherwise you may accidentally make training input coordination a universal gateway for cache use by unrelated capabilities.

If caching in this change is deliberately scoped only to training-input caches, then the current wording is fine. I'd just decide that explicitly.

The last wording tweak I'd consider is **“complete usable value or bundle.”** That's correct for the current cache model, but because the design explicitly leaves room for streaming/sharded future systems, I might say:

> complete usable **publication unit** or bundle

because a future accepted cache boundary could itself be chunked:

```text
large cached representation
   ├─ chunk A → independently complete publication unit
   ├─ chunk B → independently complete publication unit
   └─ chunk C
```

You don't want “complete” accidentally interpreted later as “the entire logical representation must exist before anything can be published.” Your following language about explicit consistency boundaries already points toward this interpretation anyway.

### The failure semantics are particularly strong

This part is nicely disciplined:

```text
payload written
index/publication failed
        ↓
known external effect:
    payload exists

published readiness:
    NO

recovery:
    verify/adopt
    reissue
    remove orphan
```

without inventing a fake transaction. Same with partial job coverage: independently completed records can remain usable if the accepted policy permits it, while the whole production job remains incomplete.

That matches the optimizer/failure design you've been building: **record what actually became true instead of forcing everything into success/failure for some enclosing operation.**

And the restoration boundary feels appropriately placed. G4.1 says what cache coordination must *contribute*:

```text
production position
pending/publication state
dependency evidence
stored-resource guarantees
```

while explicitly leaving the **cross-owner coherent snapshot protocol** for G4.4. That's the right split; 4.1 isn't trying to prematurely solve restoration.

### So I'd move on to 4.2

I don't see a missing architectural question here that warrants keeping 4.1 open.

The three wording/scope things I'd consider before leaving it are:

1. `selected implementation` → make clear it's accepted computation identity/meaning, not automatically physical realization identity.
2. Decide whether cache consumer admission is specifically training-input-owned or generically owned by whichever capability consumes the value.
3. `complete usable value` → perhaps `complete accepted publication unit/value` so future streaming doesn't get excluded accidentally.

None of those undermine the exchange itself.

And this is a nice sign for the overall design: G4.1 didn't require inventing anything new in the run language. The existing notions—accepted computation, dependencies, provenance, independently progressing activity, handoff/admission, current-use protection, partial outcome, restoration contributor—were enough to describe caching. That's a decent test that the G2/G3 architecture is actually carrying its weight rather than needing a special construct every time you touch a new feature.