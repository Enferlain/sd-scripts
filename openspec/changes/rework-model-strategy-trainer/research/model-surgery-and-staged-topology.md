# Model Surgery and Staged Topology Changes

## Scope

Study training recipes where the model or its execution topology changes across a training run or between directly connected training stages.

The focus is on:

* parameter identity across topology changes
* copying, cloning, replacing, adding, and deleting model state
* initialization of newly introduced parameters
* optimizer and EMA migration
* trainability changes around transition boundaries
* checkpoint and resume semantics
* distributed-wrapper/backend reconstruction
* compatibility of cached or prepared state
* distinction between curriculum changes and actual topology changes

Representative cases:

1. **Qwen3.8-Flash-Next** — full attention → Qwen Sparse Attention during continued pretraining
2. **DeepSeek-V4** — dense-attention warmup → indexer warmup → sparse attention
3. **Megatron MoE upcycling** — dense FFNs → routed MoE experts
4. **Progressive Growing GANs** — add generator/discriminator layers as resolution increases

Claim labels: **Observed / Inferred / Unknown**.

---

# 1. What counts as model surgery?

I’d establish this first, because there are several different kinds.

A training stage can change:

```text
parameter topology
    add/remove/replace parameters

module topology
    replace one operation/module with another

execution topology
    same parameters, different connectivity/attention/routing

optimization topology
    new parameter groups / newly trainable state

representation topology
    new intermediate state or interface appears
```

So these are all different:

```text
freeze layer
    trainability change

change sequence length
    curriculum change

replace dense attention with sparse attention + indexer
    execution + parameter topology change

duplicate dense FFN into 8 experts + add router
    parameter + module topology change

add new generator layers
    structural growth
```

The central distinction I'd preserve is:

> **Training stage is not necessarily only a new configuration over the same model graph. A stage transition may transform the graph itself.**

---

# 2. Recipe — Qwen3.8-Flash-Next: full attention → QSA

This is probably the best modern example because the surgery happens **during the training lifecycle of the same model**, not as an unrelated conversion experiment.

The Qwen3.8 architecture initially contains:

```text
GDN
GDN
GDN
full attention
↓
repeat
```

During continued pretraining, those full-attention layers are replaced by **Qwen Sparse Attention (QSA)**. ([arXiv][1])

So the transition is approximately:

```text
before CPT

hidden
  ↓
full attention
  ↓
hidden


after surgery

hidden
  ├──────────────→ QSA indexer
  │                    ↓
  │                selected blocks
  │                    ↓
  └──────────────→ sparse core attention
                       ↓
                     hidden
```

That is more than changing an attention mask. QSA introduces a **new learned indexer** and a different attention execution path. ([arXiv][2])

## Stage 1 — indexer distillation

Qwen first keeps the dense/full attention behavior available as a teacher signal.

The full attention distributions are aggregated and converted into block-level targets. Only the QSA indexer is trained for this warmup: 1,000 steps at a reported learning rate of `1e-3`, using 256K-token sequences. ([arXiv][2])

Conceptually:

```text
existing full attention
        ↓
dense attention distribution
        ↓
block-level target
        │
        ▼
new QSA indexer
        ↓
KL distillation loss
```

The new module therefore does **not** simply appear with random state and immediately take over the backbone.

There is an explicit transition procedure.

## Stage 2 — sparse backbone training

After indexer initialization, QSA actually selects sparse context and the **entire backbone is trained under the new sparse execution pattern**. ([arXiv][2])

So:

```text
stage A
full attention authoritative
QSA indexer learns to approximate selection

              ↓ transition

stage B
QSA selection authoritative
whole backbone adapts to sparse execution
```

### Training-specific pressure

This gives us several constraints at once:

> **A newly installed component may require its own preparation/training stage before it is allowed to control the main execution path.**

And:

> **The component being replaced may temporarily remain executable after the replacement component has already been created, because it is needed to initialize or supervise the replacement.**

That's pretty important.

A simplistic mutation API like:

```text
replace(old, new)
```

isn't enough.

The real lifecycle is closer to:

```text
create(new)
↓
initialize/train new using old
↓
validate/accept
↓
switch authoritative execution
↓
possibly retire old path
```

---

# 3. Parameter identity across replacement

Qwen also gives a useful distinction between state that survives the transition and state that doesn't.

The model isn't rebuilt from zero.

Most of:

```text
embeddings
GDN layers
MoE layers
residual machinery
output head
```

continues across the stage boundary.

Only a particular execution region changes.

So a transition can conceptually have:

```text
PERSIST
    parameter A → parameter A

COPY / TRANSFORM
    parameter B → parameter B'

CREATE
    ∅ → indexer parameters

RETIRE
    dense-only execution state → unused/removed

REWIRE
    full attention path → sparse attention path
```

That suggests a useful general abstraction:

> **Topology transition needs an explicit state-mapping rule, not merely a before-model and after-model configuration.**

The mapping itself is part of the recipe.

---

# 4. Recipe — DeepSeek-V4 dense → sparse attention

DeepSeek-V4 provides an independent example with very similar pressure but somewhat different staging.

DeepSeek-V4 begins training with dense attention. Its reported training sequence progresses:

```text
4K
↓
16K
↓
64K
↓
1M
```

The first **1T tokens** use dense attention. Sparse attention is introduced when sequence length reaches 64K and remains active afterwards. ([arXiv][3])

Again, there is an intermediate stage:

```text
dense attention
       ↓
short lightning-indexer warmup
       ↓
sparse attention training
```

The authors explicitly describe this as a two-stage sparse-attention introduction. ([arXiv][3])

So this isn't simply:

```text
if step > X:
    sparse = True
```

The transition has **new trainable state that needs initialization before the new topology becomes authoritative**.

### Interesting distinction from Qwen

The broad shape is similar:

```text
dense execution
↓
learn selection/indexing mechanism
↓
sparse execution
```

but the specific attention architecture and indexer are different.

That suggests the generic lesson is *not*:

```text
Trainer understands dense_to_sparse_attention()
```

but rather:

> **Recipes may define staged transitions in which new state is prepared under one execution topology and subsequently becomes part of another execution topology.**

---

# 5. Transition can coincide with other curricula

DeepSeek is also a good warning that surgery doesn't necessarily happen in isolation.

The sparse-attention transition coincides with a sequence-length curriculum:

```text
model topology
attention mode
sequence length
```

all move together through training. ([arXiv][3])

So stage boundaries can coordinate several authorities:

```text
Stage N
├─ model topology
├─ sequence length
├─ data policy
├─ objective weights
├─ trainability
└─ optimizer/schedule behavior
```

This reinforces the conclusion from the curriculum work:

> **A stage is a coherent configuration of several training subsystems, not necessarily one scheduler changing one scalar.**

But topology is one of those subsystems, not the owner of the whole stage.

---

# 6. Dynamic execution-mode transitions can also be event-driven

DeepSeek-V4 has another interesting case that I would mention separately rather than call it topology surgery.

During instability, its training system can temporarily enable **Anticipatory Routing**.

Normally:

```text
current backbone θ_t
     ↓
current routing θ_t
```

During this mode:

```text
current backbone θ_t

routing indices produced using
historical θ_(t-Δt)
```

The system detects a loss spike, rolls back briefly, enables Anticipatory Routing for a period, then returns to ordinary training. ([arXiv][3])

No permanent model topology necessarily changes here.

But it proves something related:

> **Training-stage transitions do not have to be predetermined by step count. They may be triggered by observed training state and later reversed.**

So your eventual stage machinery shouldn't automatically mean:

```text
stage 1 until step 1000
stage 2 forever
```

Transitions can be:

```text
scheduled
metric-triggered
failure-triggered
temporary
reversible
```

I wouldn't overdesign for this yet, but I'd record it.

---

# 7. Recipe — dense model → MoE through sparse upcycling

This is the clearest example of genuine **parameter-topology surgery**.

Sparse upcycling starts with a trained dense model and converts some dense FFNs into sparse Mixture-of-Experts layers, preserving existing knowledge while adding model capacity. ([arXiv][4])

Current Megatron Core supports this directly.

Its normal upcycling path does approximately:

```text
dense Transformer

FFN
 │
 ├─ W1
 └─ W2

       ↓ upcycle

MoE layer
 ├─ router                  NEW
 │
 ├─ expert 0 ─ W1/W2 copy
 ├─ expert 1 ─ W1/W2 copy
 ├─ expert 2 ─ W1/W2 copy
 └─ ...
```

Megatron documents the default strategy as duplicating the existing dense MLP into several experts; it also supports granular upcycling where expert FFNs are narrower. ([NVIDIA Docs][5])

This is the exact opposite of ordinary checkpoint loading.

Normally:

```text
checkpoint topology
≈
target topology
```

Here:

```text
dense checkpoint topology
≠
MoE runtime topology
```

and an explicit transformation bridges the two.

---

# 8. Megatron treats surgery as a checkpoint-producing preparation step

This part is especially relevant to your architecture.

Current Megatron's implementation:

1. constructs the desired MoE model,
2. temporarily constructs the corresponding dense topology,
3. loads the dense checkpoint,
4. converts dense state into MoE state,
5. saves the **converted MoE checkpoint**,
6. destroys the temporary dense model,
7. continues from the new topology.

The code explicitly says upcycling should only be enabled for the **first conversion run**; subsequent runs use the converted checkpoint normally. Megatron also supports changing distributed parallel topology while performing the upcycle because the conversion sits on top of distributed checkpointing. ([NVIDIA Docs][6])

That makes the lifecycle look like:

```text
dense checkpoint
      │
      ▼
temporary dense model
      │
      │ state transformation
      ▼
target MoE model
      │
      ▼
converted checkpoint
      │
      └────────→ normal MoE training
```

This is very close to the preparation-coordination problem we've been talking about.

The topology mutation is **not hidden inside the ordinary train step**.

It creates a new authoritative model state before subsequent training proceeds.

### Strong design pressure

> **Some model surgery is better understood as a transactional preparation transition that produces a new model-state authority, rather than as an ordinary optimizer step.**

That one seems worth preserving almost verbatim.

---

# 9. Optimizer state cannot always cross the surgery boundary

Megatron's implementation makes this concrete.

When MoE upcycling is requested, it automatically enables `no_load_optim`; the dense optimizer state is not restored into the new MoE parameter topology. The conversion loads model state without an optimizer or scheduler and saves the converted model before normal training proceeds. ([NVIDIA Docs][6])

Which makes sense:

```text
dense FFN parameter
       │
       ├── Adam moment m
       └── Adam moment v

             ↓ duplicate into experts

expert 0
expert 1
expert 2
...
router
```

What exactly should happen to the dense optimizer moments?

There isn't one universally correct answer.

Possible policies include:

```text
discard old optimizer state

copy state to cloned parameters

transform state

initialize new state only

preserve state for untouched parameters
```

Megatron chooses a clean boundary rather than pretending dense optimizer state naturally maps to the new topology.

### Design pressure

> **Parameter-state migration and optimizer-state migration are separate decisions.**

This is probably one of the biggest findings from the topic.

Even if:

```text
old parameter → new parameter
```

has a valid mapping, it does not automatically follow that:

```text
old optimizer state → new optimizer state
```

has one.

Same issue for:

```text
EMA
gradient scaler state
per-parameter scheduler state
quantization state
```

---

# 10. A topology transition creates lineage, not necessarily identity

Sparse upcycling also gives a better vocabulary for what happens to parameters.

Consider:

```text
dense.ffn.weight
```

After conversion:

```text
moe.expert0.weight
moe.expert1.weight
moe.expert2.weight
...
```

These weights have a common **origin**, but they aren't one parameter anymore.

So it may be useful conceptually to distinguish:

```text
identity:
    "this is still the same logical state"

lineage:
    "this new state was derived from that old state"
```

For example:

```text
embedding.weight
    identity preserved

dense_ffn.weight
    identity retired
        ↓ lineage
    expert_0.weight
    expert_1.weight
    ...

router.weight
    newly created
```

That seems much more accurate than trying to preserve parameter identity through every surgery.

---

# 11. New topology can imply new distributed topology

Converting dense FFNs to experts doesn't just alter modules.

Before:

```text
dense FFN
    ↓
ordinary TP/DP
```

After:

```text
MoE
 ├─ expert 0 → rank group A
 ├─ expert 1 → rank group B
 ├─ ...
 └─ router + token dispatch
```

Expert parallelism and token dispatch now exist.

Megatron explicitly allows arbitrary expert-parallel layouts during upcycling even when the dense source checkpoint used a different parallel topology. ([NVIDIA Docs][5])

So:

> **A model-topology change may invalidate the existing physical execution topology.**

That means surgery may require rebuilding:

```text
DDP/FSDP wrappers
process groups
optimizer parameter groups
shard layouts
checkpoint mappings
compiled graphs
activation-checkpoint boundaries
```

rather than mutating a module underneath already-prepared execution machinery and hoping for the best.

For your design this argues pretty strongly for:

```text
derive target model
↓
perform surgery
↓
derive execution/distribution preparation
↓
install
```

instead of doing arbitrary surgery after the backend owns the model.

---

# 12. Recipe — Progressive Growing GANs

I'd include one non-LLM case because it's the purest form of **architecture changing during training itself**.

Progressive Growing GANs start training both generator and discriminator at low spatial resolution and progressively introduce layers that operate at higher resolutions. ([arXiv][7])

Conceptually:

```text
4×4 generator
4×4 discriminator

       ↓

add 8×8 blocks

       ↓

add 16×16 blocks

       ↓

...

       ↓

1024×1024
```

The transition doesn't instantly replace one network with another. Newly introduced resolution paths are **faded in**, blending old and new paths over a transition period.

Conceptually:

```text
old path ────────┐
                 ├─ weighted blend → output
new block/path ──┘

α: 0 → 1
```

Then the old bypass disappears from the authoritative execution once the transition completes.

This gives us something Qwen/Megatron don't:

> **Two topologies may intentionally coexist during a transition, with a continuously changing mixture determining execution.**

That's very different from atomic replacement.

---

# 13. Newly introduced parameters need an initialization policy

Across these cases, new parameters are introduced differently.

```text
Qwen QSA indexer
    newly created
    then distilled

DeepSeek lightning indexer
    newly introduced
    warmup before sparse execution

MoE experts
    derived/copied from dense FFN

MoE router
    new state

Progressive GAN block
    newly initialized
    faded into execution
```

So a generic surgery mechanism needs to distinguish at least conceptually:

```text
COPY
CLONE
TRANSFORM
INITIALIZE
DISTILL
RETAIN
RETIRE
```

These are not interchangeable.

### Useful constraint

> **“Add component” is incomplete without specifying how its initial authoritative state is obtained.**

That could come from:

```text
random initialization
another parameter
several parameters
distillation
conversion transform
external checkpoint
deterministic construction
```

---

# 14. Surgery can alter which parameters are trainable

Qwen is a nice example:

```text
QSA warmup
    indexer trainable
    backbone effectively teacher/source

then

sparse training
    indexer + backbone participate in training
```

So topology and trainability transitions can happen together.

The resulting stage transition is something like:

```text
Stage A
graph = dense + candidate indexer
trainable = indexer
objective = distillation

Stage B
graph = sparse/indexer topology
trainable = whole backbone
objective = LM pretraining
```

That's way richer than:

```text
scheduler.step()
```

and probably should be represented as a coherent recipe-stage transition.

---

# 15. Objectives can change because topology changes

This is another good distinction.

During Qwen's indexer warmup:

```text
objective = reproduce dense attention distribution
```

After the switch:

```text
objective = normal pretraining under sparse attention
```

So the new topology requires an **intermediate objective that exists only to prepare the transition**.

That objective has no reason to appear during normal training or deployment.

This gives:

> **A stage-specific loss may exist solely to establish valid state for a future execution topology.**

That's a pretty useful category.

You could call it a **transition objective**, conceptually.

---

# 16. Checkpoints around surgery need a topology identity

Imagine saving immediately before and after an upcycle:

```text
checkpoint A
model topology = dense

checkpoint B
model topology = MoE
```

Even if both came from “the same run,” they're not load-compatible with the same constructor.

Likewise:

```text
Qwen before QSA
Qwen after QSA installation
```

may require different model construction or state interpretation.

Therefore a resumable checkpoint needs more than:

```text
model state_dict
global_step
```

It needs enough information to recover:

```text
topology identity/version
current stage
installed components
active execution relationships
state transformation already completed?
```

Otherwise resume can accidentally **perform the surgery twice**.

Megatron guards exactly this kind of problem: its upcycling flag is valid only for the first conversion and refuses to overwrite an already-existing upcycling destination. ([GitHub][8])

### Design pressure

> **Topology transition must be idempotence-aware. Resume must distinguish “transition pending” from “transition already committed.”**

That fits beautifully with your coherent-publication concern.

---

# 17. Surgery should probably have a commit boundary

This is one place where the evidence points pretty strongly toward an architectural rule.

Unsafe conceptual flow:

```text
modify layer A
↓
modify optimizer
↓
modify layer B
↓ crash
```

Now what model do we have?

A safer conceptual flow is:

```text
current authoritative topology
        │
        ▼
derive target topology
        │
        ▼
transform/copy/create state
        │
        ▼
validate target
        │
        ▼
prepare optimizer/backend/checkpoint mappings
        │
        ▼
COMMIT
        │
        ▼
new authoritative topology
```

Megatron's converted-checkpoint-before-normal-training behavior is basically a practical example of this pattern. ([NVIDIA Docs][6])

So I'd put this in the note:

> **Topology-changing preparation should be treated as a coherent state transition. The old and new topology should not both be partially authoritative.**

That seems very aligned with what you've already been settling for the Trainer preparation coordinator.

---

# 18. Not every topology change should happen in-place

This is probably worth explicitly saying.

There are at least three implementation styles:

```text
IN-PLACE MUTATION
existing object
    ↓
modules replaced/added

RECONSTRUCTION
build target topology
    ↓
map old state into it

DUAL TOPOLOGY TRANSITION
old + new coexist
    ↓
gradually transfer authority
```

Examples roughly map to:

```text
QSA
    replacement / reconstruction style

Megatron upcycling
    target-model reconstruction + state conversion

Progressive GAN
    temporary dual topology / fade-in
```

A generic architecture should probably describe the **desired transition semantics** without forcing all transitions to use one physical technique.

---

# 19. Model surgery invalidates derived state

This is another place it intersects with all the previous research.

Changing model topology can invalidate:

```text
optimizer parameter groups
optimizer state
EMA
distributed shards
compiled graphs
CUDA graphs
activation-checkpoint plans
offload plans
PEFT target mappings
parameter-name filters
cached model outputs
rollout/inference replicas
checkpoint schemas
```

Some derived state may survive; some won't.

So I'd write:

> **Any state derived from model topology needs an explicit validity relationship to that topology.**

This is basically the model-side version of our cache dependency rule:

```text
data cache
    valid iff upstream representation producers unchanged

execution preparation
    valid iff relevant model topology unchanged
```

Same pattern again.

---

# 20. Topology-changing stages versus ordinary stages

I think a little distinction table would be useful here:

| Stage change                             | Requires model surgery?                                  |
| ---------------------------------------- | -------------------------------------------------------- |
| learning-rate change                     | No                                                       |
| dataset mixture change                   | No                                                       |
| sequence-length change                   | Usually no                                               |
| freeze/unfreeze existing module          | No topology change                                       |
| new loss over existing outputs           | Usually no                                               |
| install new indexer module               | Yes                                                      |
| replace dense attention with sparse path | Yes / execution topology change                          |
| dense FFN → multiple experts + router    | Yes                                                      |
| add higher-resolution generator block    | Yes                                                      |
| switch optimizer                         | No model topology change, but optimizer-state transition |
| change distributed sharding              | No logical topology change                               |

That helps keep this topic from swallowing every staged training behavior.

---

# 21. Cross-case findings

| Tempting assumption                                                | Evidence                                                                                          |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| Model topology is fixed once training starts                       | False.                                                                                            |
| A stage transition only changes hyperparameters                    | False.                                                                                            |
| Newly added components can immediately become authoritative        | False; Qwen/DeepSeek warm up indexers first.                                                      |
| Replacement is always atomic                                       | False; progressive growing uses overlapping/faded topologies.                                     |
| Old weights either survive or disappear                            | Too simple; they may be copied, cloned, transformed or become lineage for several new parameters. |
| Optimizer state follows model weights automatically                | False; Megatron explicitly drops dense optimizer state during MoE upcycling.                      |
| One checkpoint schema works throughout a run                       | Not necessarily; topology identity may change.                                                    |
| Distribution setup survives surgery                                | Not necessarily; MoE conversion introduces expert parallelism and dispatch.                       |
| New parameters always use random initialization                    | False; copied weights and distillation are common.                                                |
| Architecture transition and objective are independent              | False; transition-specific objectives may initialize the new topology.                            |
| Stage boundaries are always predetermined                          | False; DeepSeek also demonstrates event-triggered temporary execution-mode transitions.           |
| A topology mutation can safely occur anywhere in the training loop | Not generally; derived backend/optimizer/checkpoint state may need reconstruction.                |

---

# 22. Main architecture conclusions

I think these are the ones worth carrying forward.

> **Model topology should be treated as versioned authoritative state, not as an immutable assumption of the Trainer.**

> **A topology transition is a mapping from one coherent model state to another. That mapping may retain, copy, transform, initialize, distill, or retire parameter state, and optimizer/EMA state requires its own migration policy.**

> **Transition preparation and transition publication should be distinct: the new topology may need to be constructed, initialized, trained or validated before it becomes the execution authority.**

And probably the most actionable one for your current architecture:

```text
Recipe / stage authority
    declares desired transition
    and state-mapping semantics

Model authority
    constructs / validates target topology

Optimization preparation
    derives new parameter ownership/state

Backend preparation
    derives new wrapping/sharding/placement

Coordinator
    publishes them coherently

Trainer
    continues on the accepted topology
```

That is much safer than teaching the generic Trainer arbitrary operations like:

```text
add_layer()
replace_attention()
convert_to_moe()
```

Those are model/recipe-specific transformations.

The Trainer mainly needs to tolerate the fact that **the thing being trained at stage N+1 may not have the same parameter graph as the thing at stage N**.

One final distinction I'd keep in the note because it summarizes the topic nicely:

```text
checkpoint resume:
    restore the SAME topology and continue

model surgery:
    derive a NEW topology from prior state and continue
```

Those are related operations, but they are very much **not the same lifecycle event**.

---

[1]: https://arxiv.org/abs/2608.30320 "[2608.30320] On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability"
[2]: https://arxiv.org/html/2608.30320v1 "On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability"
[3]: https://arxiv.org/html/2606.19348v1 "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"
[4]: https://arxiv.org/abs/2212.05055?utm_source=chatgpt.com "Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints"
[5]: https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/user-guide/features/moe.html?utm_source=chatgpt.com "Mixture of Experts — Megatron Core"
[6]: https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.moe.upcycling_utils.html?utm_source=chatgpt.com "core.transformer.moe.upcycling_utils — Megatron Core"
[7]: https://arxiv.org/abs/1710.10196?utm_source=chatgpt.com "Progressive Growing of GANs for Improved Quality, Stability, and Variation"
[8]: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md?utm_source=chatgpt.com "Megatron-LM/megatron/core/transformer/moe/README.md at main · NVIDIA/Megatron-LM · GitHub"
