Yeah, this is cleaner than treating “acceptance” as one blob, and it also exposes the one thing I’d still change if your goal is **verify as completely as reasonably possible before training**.

The umbrella idea is solid:

```text
TRAINING CONTRACT SYSTEM
├── contract definition
├── strategy fulfillment
└── runtime conformance / enforcement
```

But I think **“runtime enforcement” currently covers two things with very different timing**:

```text
A. proving the concrete realization is valid before Trainer gets it
B. keeping it valid as the run mutates afterward
```

Those are related, but I wouldn’t hide both behind one phrase.

I’d make the conceptual model more like:

```text
TRAINING CONTRACT SYSTEM

1. Contract definition
   What valid training means.

2. Strategy fulfillment
   Does the authored training definition satisfy the contract?
   Produces the exact obligations for this run.

3. Run realization verification
   Do the actual models, loaders, targets, execution structure,
   optimizer realization, backend choices, etc. satisfy those obligations?

4. Run enforcement / authority
   As things are replaced, prepared, wrapped, revised, restored,
   does the live run continue to satisfy them?
```

That fits the session context better *and* your concern about “why check only step 1 and then discover everything else later?”

The flow becomes:

```text
INTENDED TRAINER / PIPELINE
          │
          ▼
 TRAINING CONTRACT DEFINITION
          │
          ▼
    AUTHORED STRATEGY
          │
          ▼
   STRATEGY FULFILLMENT
 "is the requested run valid?"
          │
          ▼
    RUN OBLIGATIONS
          │
          ▼
 REALIZATION VERIFICATION
 "can the concrete supplied world
  actually fulfill this run?"
          │
          ▼
    VERIFIED RUN STATE
          │
          ▼
 PREPARATION / PUBLICATION
          │
          ▼
        TRAINER
          │
          ▼
  ONGOING RUN AUTHORITY
 "does it remain valid as state changes?"
```

That distinction is important because otherwise **“runtime enforcement” sounds like the second half only happens after you're already running**.

And I think your original intuition was stronger than the wording Sol currently has: we should try to move **as much of realization verification as possible before expensive preparation/training**.

So the contract system's job would not merely be:

> strategy is legal → good luck loading it.

It would try to establish:

```text
✓ authored combination is meaningful
✓ selected providers can satisfy it
✓ concrete artifacts structurally match it
✓ participants and relationships can be formed
✓ optimization targets are valid
✓ execution topology is valid
✓ requested backend capabilities exist
✓ persistence/output requirements are realizable
✓ known resource constraints are acceptable
```

before the real run is committed.

Then whatever remains unknowable gets explicitly classified as such.

That gives you a much nicer definition of verification:

> **The contract system proves every contract obligation at the earliest point where authoritative evidence exists.**

That should probably be a first-class design rule.

Because then the boundaries aren't arbitrary architecture boxes. They're caused by **availability of evidence**.

For example:

```text
Strategy says:
"train SDXL UNet with LoRA on these target modules"
```

At fulfillment time, you can prove:

```text
✓ SDXL + LoRA is an admitted arrangement
✓ requested target-selection semantics exist
✓ ownership is coherent
✓ optimizer advancement semantics exist
✓ required capabilities are selected
```

Before loading full weights, perhaps checkpoint metadata lets you prove:

```text
✓ artifact is structurally SDXL
✓ expected tensors/modules exist
✓ dtype/shape expectations are satisfiable
```

After materialization:

```text
✓ actual module graph satisfies accepted structural identities
✓ actual target refs resolve
✓ tied/shared parameters obey non-overlap rules
```

After preparation:

```text
✓ prepared revisions/routes correspond to the accepted participants
✓ backend transformations preserved required identity/relationships
```

And only genuinely dynamic things remain for execution.

That's much closer to **one proof becoming progressively stronger** than to “four validators.”

---

There's also a naming issue I think you've stumbled onto.

### `Accepted run arrangement` may be too strong a name

If you've only performed strategy fulfillment, then:

```text
AcceptedRunArrangement
```

sounds like:

> this run is accepted.

But what it really means is:

> this run's **meaning** has been accepted.

Those aren't quite the same.

You could keep the term, but define it very explicitly:

> An Accepted Run Arrangement represents semantically accepted run intent plus the concrete realization obligations that remain to be discharged.

That's viable.

But something like:

```text
Resolved Run Arrangement
```

or

```text
Admitted Run Arrangement
```

would naturally imply less than “everything checked.”

Then later you can reach:

```text
Verified Run
```

or whatever final vocabulary comes out of the design.

I wouldn't choose names yet, but I think the **state progression itself** is worth making explicit before settling terminology.

---

And I strongly agree with the session-context distinction:

> **Training contract definition** should stay narrower than **Training Contract System**.

That solves your original diagram problem really nicely.

The contract definition doesn't have to stretch across the whole pipeline as some giant circle. Instead:

```text
                   TRAINING CONTRACT SYSTEM
┌────────────────────────────────────────────────────────────┐
│                                                            │
│   CONTRACT DEFINITION                                      │
│   "what must be true?"                                     │
│           │                                                │
│           ├───────────────┬──────────────────┐             │
│           ▼               ▼                  ▼             │
│     fulfillment      verification       enforcement        │
│     authored         concrete           changing live      │
│     meaning          realization        run state          │
│                                                            │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
                           TRAINER
```

This avoids turning `TrainingContract` into:

```text
validator
loader
binder
state machine
backend
checkpoint manager
Trainer supervisor
```

which would indeed muddy the whole architecture.

The definition supplies **meaning**. Different mechanisms establish and preserve that meaning.

---

I’d go one step further than the session text on this sentence:

> “The first turns the general training contract into concrete requirements for one run. The second applies those already-established requirements to concrete objects and transitions.”

I think that's basically the central architectural idea.

I'd perhaps sharpen it into:

> **Strategy fulfillment specializes the general training contract into a run-specific obligation set. Every subsequent conformance decision evaluates evidence against that same obligation set; downstream stages do not reinterpret the authored strategy or independently derive compatibility rules.**

That gives you a very strong anti-drift rule.

Then:

```text
general rules
      ↓ specialize once
run-specific obligations
      ↓
      ├─ artifact evidence
      ├─ binding evidence
      ├─ preparation evidence
      ├─ restoration evidence
      └─ runtime transition evidence
```

Everybody downstream shares **one frozen interpretation**.

That feels like the right answer to your original “how close should these things be?” question:

**Semantically, extremely close. They consume the exact same run obligations. Operationally, separate because they operate on different evidence and at different lifetimes.**

And I think the next design conversation with Sol should probably be **not “edit D2” yet**, but:

> What exactly is the run-specific obligation set produced by strategy fulfillment, what evidence can discharge each kind of obligation, and how much of that evidence can we establish before expensive runtime preparation?

Once that is clear, placement and naming will probably fall out naturally instead of us forcing boxes onto it.
