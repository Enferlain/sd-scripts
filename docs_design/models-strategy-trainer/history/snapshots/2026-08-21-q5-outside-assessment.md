Yeah, these two reviews are useful, and I think they mostly confirm where we already landed rather than introducing a competing direction.

The **most important new observation is GLM’s**:

> the Q5 scenarios cannot select the physical layout because all three candidate layouts can be made to satisfy them.

That’s exactly right. Milestone 10 has done its job: it established the required semantics—single authority, typed transitions, revisions/freshness, narrow projections, atomic acceptance, no family cardinality—but it can’t tell us whether those semantics live in object A or object B. The inventory itself now says essentially that. 

So I would **stop adding normal-path Q5 scenarios as a way of choosing the owner**. Failure/rollback, multi-adapter overlap, compile, EMA, etc. are useful future conformance scenarios, but they're unlikely to answer the physical ownership question either.

### Where I differ slightly from the reviewers

Both lean toward a **separate contract-owned authority**. I agree physically, but I still don't think it should necessarily become the new thing Trainer receives.

The direction already says fulfillment is a status and “the Trainer still receives a strategy,” and that after fulfillment Trainer interacts with the complete strategy through the Trainer-facing contract.  

So my preferred result is:

```text
authored TrainingStrategy
       │
       │ filing / validation / run initialization
       ▼
active per-run strategy
       │
       ├── strategy behavior / selected capabilities
       │
       └── dedicated BindingAuthority
                canonical mutable contract state
```

Trainer still conceptually has:

```python
trainer.strategy
```

not:

```python
trainer.strategy
trainer.binding_authority
```

and definitely not:

```python
trainer.bound_strategy_contract
```

unless later implementation pressure proves that a new façade has actual semantics.

The **authority itself should be a separate concern/object**, because its responsibilities have become substantial enough that storing them as random strategy fields would recreate the same state/behavior mixing we're removing.

So:

```text
public architecture:
    Trainer ↔ Strategy

internal ownership:
    Strategy/run contract governs a dedicated BindingAuthority
```

That combines the best part of “paired object” and “separate authority.”

## The criteria I’d use to decide Q5 now

GLM’s proposed criteria are better than more topology scenarios. I'd settle the choice against these:

1. **Single-writer enforceability**
   Can loader, adapter code, and Trainer infrastructure propose transitions without gaining unrestricted mutation authority?

2. **Independent testing**
   Can binding semantics—materialization, replacement, relationship transitions, stale route detection—be tested without constructing a real SDXL strategy?

3. **State/behavior separation**
   Does adding revision/freshness/invalidation bookkeeping turn strategy behavior classes into mutable runtime stores?

4. **Lifetime**
   Is the authority obviously one-per-run and created/destroyed with that strategy execution rather than with model classes or Trainer globally?

5. **Access discipline**
   Can capabilities receive scoped views rather than `authority.get_anything()`?

6. **Migration path**
   Can `LoadedModelComponent` evolve into this incrementally without rewriting every strategy at once?

A dedicated internal authority wins basically all six.

So I think **Q5 is now close enough to decide**, with one remaining mini-design pass around construction/access.

---

### I strongly agree with the conservative v1 staging

This is probably the most practically valuable recommendation in the two reviews.

Don't implement the final theoretical system on day one.

For invalidation:

```text
semantic end-state:
    precise dependency-aware invalidation
```

but v1 can safely be:

```text
accepted structural/binding transition
    → invalidate all derived preparation/
      optimization/cache/artifact projections
```

Then later:

```text
dependency set + revisions
    → invalidate only affected projections
```

The docs should preserve the stronger semantic requirement, but the OpenSpec migration can explicitly stage it.

Same for named execution routes.

Keep this settled semantic:

```text
(component, route) → one authoritative current binding
```

but v1 can simply implement:

```text
component → normal execution route
```

until compilation/sampling or another real production need introduces the second route.

That is not backing away from Q2. It's avoiding writing a generalized route registry before anything needs one.

Same with Q4. The full model is good design vocabulary:

```text
declaration
request
plan
result

product
member
resource
```

but the first migration does not need an industrial artifact transaction engine. The important first implementation could simply introduce typed:

```text
ArtifactRequest
ArtifactResult
```

plus enough resolved semantic state that SDXL/SD3/adapter persistence no longer receives the whole Trainer. The richer member/checksum/dependency model can grow underneath the already-correct semantics.

---

## Claude's missing scenarios are useful, but I'd classify them

Some are **Q5 semantic tests**:

```text
failed/rejected transition
partial multi-participant transition
mid-run arrangement amendment
multi-adapter relationship overlap
```

Some are mostly **Q2/preparation tests**:

```text
torch.compile route
multiple prepared routes
route refresh
```

Some are **distributed implementation constraints**:

```text
rank consistency
authority replication
```

And EMA is mostly another excellent **Q1/Q2 identity test**:

```text
live model
EMA model

independently evolving state
→ separate participants
```

We shouldn't let all of those become “Q5 must answer everything before we're allowed to proceed.”

---

## I also agree we need explicit v1 assumptions

Claude caught something important here.

The design currently implicitly assumes something like:

```text
one active strategy per Trainer
one coordinating Trainer lifecycle
contract state transitions are synchronous at acceptance
one logical binding authority per run
distributed ranks do not independently mutate strategy state
```

I would write these down.

Especially the distributed one.

We do **not** need to design a distributed database for binding authority. The initial contract can say conceptually:

> Strategy-state transitions are coordinated by the Trainer/runtime synchronization mechanism. Per-rank execution replicas do not constitute independent binding authorities.

Then rank consistency becomes a preparation/runtime obligation.

That keeps:

```text
logical authority
```

separate from:

```text
N physical distributed copies/wrappers
```

which matches everything we've said already.

---

## The optimization blocker definitely needs to come back

This is the clearest doc bug the reviewers found.

The notes originally had two blockers:

```text
1. binding semantics / ownership
2. standard Trainer-owned optimization profile
   + explicit research extension profile
```



Q1–Q5 resolve blocker #1.

They do **not** resolve #2.

The later wording that makes Q5 sound like the only thing remaining before everything becomes API design is slightly misleading.

After Q5 I think the flow should actually be:

```text
Binding semantic block
    Q1–Q5
        ↓ settled

Concrete binding/preparation exchanges
        ↓

Optimization ownership profile
    standard Trainer-owned path
    explicit manual/research takeover
        ↓

Optimization exchange
        ↓

Step exchange
        ↓

OpenSpec
```

Some preparation exchange work and optimization-ownership work may inform one another, so I wouldn't insist on a rigid wall between them.

---

## And yes, the docs should be normalized soon

The three-doc division I'd choose is almost exactly what the reviewers suggest:

```text
strategy_system_direction.md
    normative design direction
    settled decisions
    current open questions
    rationale where useful

strategy_system_inventory.md
    evidence
    current production behavior
    pressure tests
    milestone investigations

notes.md
    chronological checkpoint / conversational breadcrumbs
```

Right now the direction and inventory both repeat settled Q1–Q4 semantics at considerable length. That's useful during discovery, but once we move to OpenSpec it becomes dangerous because somebody will eventually edit one and not the other.

I'd make the **direction doc the single normative pre-OpenSpec source**.

Inventory can say:

```text
Milestone 8 evidence supports the
settled arrangement semantics in Direction §X
```

and retain its detailed evidence without independently defining the architecture.

I'd also explicitly enumerate:

```text
Q1 logical identity             SETTLED
Q2 execution binding cardinality SETTLED
Q3 arrangement transitions       SETTLED
Q4 persistence projection        SETTLED
Q5 binding authority ownership   CURRENT
```

Then separately:

```text
Next semantic blocker:
    optimization ownership profile
```

That would make future context recovery dramatically easier.

---

### Terminology cleanup is worth doing at the same time

I think the current emerging distinction is:

```text
participant
    strategy-contract semantic entity

component
    model/system entity that may serve as a participant
```

That means a strategy participant may correspond to:

* a model component;
* an adapter;
* learned prefix state;
* potentially some other independently managed state.

So `logical component` may actually be slightly too model-shaped as the universal term.

I wouldn't rename everything casually now, but during the write-up pass I would decide whether:

```text
ParticipantId
```

is the contract-level term, while:

```text
ModelComponent
```

remains the model-layer term.

That seems more future-proof for the system we've actually designed.

And I'd strongly prefer **autoencoder** as the generic contract vocabulary, with VAE being a particular implementation/type. The direction already naturally trends that way.

---

So my updated position after all three reviews is:

```text
Q1–Q4:
    genuinely settled

Q5 semantic requirements:
    genuinely settled

Q5 physical choice:
    choose a dedicated contract-owned
    per-run binding authority

public Trainer boundary:
    keep Strategy unless implementation
    gives us a real reason to introduce
    another façade

v1 implementation:
    deliberately conservative
    coarse invalidation
    one normal route
    minimum useful artifact plan/result

remaining major semantic work:
    optimization ownership profile
    then concrete exchanges
```

I think the two additional reviews actually make the next move clearer: **we should decide Q5 using construction/lifetime/access/migration criteria, not invent more architecture scenarios.** After that, I'd do the write-up reconciliation before going deeper, because we're now at the point where stale duplicated documentation is becoming a bigger source of confusion than the design itself.
