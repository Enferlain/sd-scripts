# Preference Learning and RL Post-Training

## Scope

Study preference learning and reinforcement-learning post-training as training
recipes, with emphasis on execution topology, data semantics, model roles,
generated training state, reward computation, policy versioning, and resume
requirements.

This is not a survey of alignment quality or preference-learning objectives.
The goal is to identify assumptions a generic Trainer must not make.

Representative cases:
- reward-model training from ranked preferences
- DPO
- KTO
- PPO-based RLHF
- GRPO
- multi-turn / agentic / asynchronous RL

Claim labels:
Observed / Inferred / Unknown

---

The most useful opening distinction is probably this:

```text
offline preference optimization
    dataset already contains supervision
    ↓
forward / loss / backward

online RL
    dataset may contain only prompts
    ↓
generate new experience
    ↓
score experience
    ↓
derive training targets / advantages
    ↓
forward / loss / backward
```

That difference is much bigger architecturally than the individual loss formulas.

---

# 1. Training topology axes

Before looking at specific recipes, I’d define a few independent axes.

```text
supervision source
    pairwise preference
    unary desirable/undesirable label
    learned reward model
    deterministic verifier
    environment outcome
    hybrid reward

experience source
    static/offline dataset
    current policy
    older policy
    external generator

comparison unit
    single completion
    chosen/rejected pair
    group of completions
    token trajectory
    multi-turn interaction

model roles
    policy
    reference policy
    reward model
    critic/value model
    rollout/inference realization

execution topology
    synchronous
    colocated
    separate rollout workers
    asynchronous
```

The point being:

> **“RL post-training” does not imply one fixed collection of models or one fixed execution loop.**

Current `verl`, for example, explicitly models Actor, Rollout, Critic, Reference Policy and Reward Model as distinct roles, while allowing some of them to be colocated in the same worker. ([Verl Documentation][1])

That immediately reinforces one of our recurring themes:

```text
logical role
    ≠
physical model copy
    ≠
worker
    ≠
resource pool
```

---

# 2. Recipe — Reward-model training from preferences

I’d include this because classical RLHF is actually **two different training jobs** before we even reach PPO.

InstructGPT's pipeline first collected ranked outputs and fit a reward model, then used that learned reward in RLHF. ([arXiv][2])

Conceptually:

```text
prompt
  │
  ├── candidate A
  ├── candidate B
  ├── candidate C
  └── ...
         ↓
    human ranking
         ↓
   reward-model training
         ↓
 response → scalar score
```

The important thing here isn't the ranking formula.

It's that the training artifact is **not the final generative policy**.

```text
job A:
preference data
   ↓
reward model

job B:
prompts + policy
   ↓
uses reward model
   ↓
updated policy
```

### Design pressure

> **A training pipeline may produce a model whose purpose is to participate as an objective dependency in a later training job rather than as the final deployable generator.**

And:

> **Supervision may describe relationships among multiple outputs rather than provide a direct target tensor for one output.**

This prepares us for DPO nicely.

---

# 3. Recipe — DPO: offline pairwise preference optimization

DPO is the cleanest counterexample to “preference learning means RL infrastructure.”

Its dataset is basically:

```text
prompt
chosen response
rejected response
```

and there is **no online sampling during fine-tuning and no separately trained reward model or critic**. ([arXiv][3])

The training computation is roughly:

```text
                    chosen ─┐
prompt ───────────── rejected ─┤
                              │
                     ┌────────┴────────┐
                     ↓                 ↓
                  policy           reference
                     │                 │
              chosen logp        chosen logp
              rejected logp      rejected logp
                     │                 │
                     └────────┬────────┘
                              ↓
                         DPO loss
                              ↓
                       update policy
```

TRL exposes exactly these four sequence-level values to the DPO loss:

```text
policy chosen logp
policy rejected logp
reference chosen logp
reference rejected logp
```

([Hugging Face][4])

The frozen reference isn't optimized, but its predictions participate in every objective evaluation.

### Interesting training consequences

One dataset record means **two model-facing sequences**.

And one logical objective can require several executions:

```text
one preference item
    ↓
policy(chosen)
policy(rejected)
reference(chosen)
reference(rejected)
```

Implementations can concatenate chosen and rejected sequences for execution efficiency without changing the fact that the objective operates on a **pair relationship**. ([Hugging Face][5])

This gives:

> **The unit of optimization need not correspond to one model sequence.**

And:

> **A frozen model can be objective-critical without being optimizer-owned.**

There is also an obvious caching opportunity:

```text
reference is frozen
+
dataset responses are fixed
        ↓
reference log probabilities
can in principle be precomputed
```

Whether a particular implementation does so is implementation-specific, but semantically it means reference execution is replaceable by an equivalent cached representation if its inputs and weights cannot change.

That's exactly the kind of thing our preparation/caching rules need to express.

---

# 4. Recipe — KTO: unpaired desirable / undesirable feedback

KTO is useful because it breaks the assumption that preference supervision always comes in explicit pairs.

Its examples only need a binary judgment:

```text
prompt + response
       ↓
desirable

or

prompt + response
       ↓
undesirable
```

The original KTO work explicitly targets this unary feedback rather than requiring chosen/rejected pairs. ([arXiv][6])

Yet the objective still uses the policy/reference relationship and an estimated KL reference point.

This makes **batch composition algorithmically relevant**.

TRL warns that too-small per-step batches produce a poor KTO KL estimate even if gradient accumulation gives a large nominal effective batch. ([Hugging Face][7])

That's a really useful case for us:

```text
microbatch
effective optimizer batch
objective-estimation batch
```

are not automatically the same thing.

### Design pressure

> **Batching may participate in the mathematics of the objective rather than merely controlling memory/performance.**

That means the execution backend cannot always freely transform:

```text
batch 32

into

8 × microbatch 4
```

and assume all intermediate calculations can occur independently.

Some statistics need the correct algorithmic grouping.

This issue comes back even harder with GRPO.

---

# 5. Recipe — PPO RLHF: generated experience as part of the training step

This is where the training topology changes completely.

A simplified current `verl` PPO loop is essentially:

```text
prompt
   ↓
rollout policy generates response
   ↓
old policy log probabilities
reference-policy log probabilities
critic values
reward score
   ↓
compute advantages
   ↓
update actor
update critic
```

That isn't conceptual inference on my part; `verl`'s HybridFlow guide presents almost exactly that control flow. ([Verl Documentation][8])

Expanded:

```text
                    ┌──── policy / rollout ───→ response
                    │                            │
prompt ─────────────┤                            ├─ old logprobs
                    │                            │
                    ├──── reference policy ─────┤ ref logprobs
                    │                            │
                    ├──── reward source ────────┤ reward
                    │                            │
                    └──── critic ───────────────┤ values
                                                 │
                                                 ▼
                                            advantages
                                                 │
                          ┌──────────────────────┴─────┐
                          ↓                            ↓
                     actor update                 critic update
```

This is no longer naturally describable as:

```text
dataloader
→ forward
→ loss
→ backward
→ optimizer
```

The **training data itself is partly generated inside the optimization loop**.

### Old policy state matters

PPO's policy loss compares current log probabilities against the `old_log_prob` associated with the policy that generated the rollout. Current `verl` carries both explicitly. ([Verl Documentation][9])

So a rollout isn't adequately represented by:

```text
prompt + response + reward
```

It can carry things like:

```text
response tokens
response mask
old policy logprobs
reference logprobs
reward
value estimates
advantages
returns
```

The rollout is therefore a **derived training artifact**.

And critically:

> **Its meaning depends on the policy version that generated it.**

That's a new kind of version dependency we haven't really had in ordinary SFT/diffusion training.

---

# 6. Policy and rollout can be one logical model but two execution realizations

This might be the biggest architectural lesson from modern RL training.

`verl` distinguishes:

```text
Actor
Rollout
ActorRollout
ActorRolloutRef
```

and supports training through FSDP/Megatron-style engines while rollout generation can use vLLM or SGLang. ([Verl Documentation][1])

So:

```text
                   logical policy
                        │
            ┌───────────┴───────────┐
            ↓                       ↓
     training realization     rollout realization
       FSDP / Megatron         vLLM / SGLang
            │                       │
       gradients etc.          generation cache
                                inference kernels
```

Weights must be synchronized between these worlds.

In `verl`'s agent-loop path, rollout servers are explicitly awakened and synchronized with the training engine before rollout work. ([Verl Documentation][10])

That's a major addition to our earlier ownership distinctions:

```text
model identity
    ≠
execution realization
```

Even more specifically:

> **One logical mutable model may simultaneously require a training representation and an inference-optimized representation, with an explicit synchronization boundary between them.**

That's worth preserving.

---

# 7. Reward computation is its own execution role

RL training doesn't imply “there is a reward model.”

Modern `verl` supports reward sources including:

```text
rule-based verifier
discriminative reward model
generative reward model
hybrid combinations
```

([Verl Documentation][11])

Its Reward Loop can place reward models:

```text
colocated with actor/rollout

or

on independent resources
```

and the standalone form can score trajectories as they finish rather than waiting for the entire rollout batch. ([Verl Documentation][12])

So the abstraction really needs to be:

```text
trajectory
    ↓
RewardSource
    ↓
score / reward information
```

not:

```text
reward_model.forward()
```

because a verifier might instead be:

```text
run code tests
check mathematical answer
query simulator
call external grader
combine several signals
```

### Design pressure

> **Reward computation may be non-differentiable, remote, asynchronous, stateful, or not neural at all.**

That sounds very similar to our earlier teacher/student conclusion: describe the **role/capability**, not an assumed model slot.

---

# 8. Recipe — GRPO: grouped generated experience

GRPO removes PPO's learned critic/value model and uses the rewards of several responses to the same prompt to establish a relative baseline. This is the defining structural change described in DeepSeekMath and current `verl`. ([arXiv][13])

Conceptually:

```text
                   prompt
                     │
       ┌─────────────┼──────────────┐
       ↓             ↓              ↓
   rollout 1      rollout 2      rollout N
       │             │              │
     reward        reward         reward
       │             │              │
       └─────────────┼──────────────┘
                     ↓
              group-relative
                baseline
                     ↓
                 advantages
                     ↓
               policy update
```

This is a really useful counterexample to:

```text
one input
→ one generated sample
→ one loss
```

Instead:

> **One prompt expands into an algorithmically related group of generated samples.**

And the group can't simply be treated as N unrelated batch records because their advantages depend on the other members.

So again:

```text
algorithmic group
    ≠
microbatch
    ≠
optimizer batch
```

`verl` explicitly notes that its microbatch settings are supposed to constrain memory without changing GRPO's algorithmic/convergence behavior. ([GitHub][14])

That's a useful backend constraint:

> **Physical batching may partition an algorithmic group for execution, but must preserve the logical relationship needed for later objective construction.**

---

# 9. Multi-turn / agentic RL changes a rollout into a trajectory

This is the one I'd definitely include because it's where RL ceases to look remotely like ordinary supervised training.

Instead of:

```text
prompt → completion
```

you now have:

```text
prompt
   ↓
model output
   ↓
tool call
   ↓
environment result
   ↓
model output
   ↓
tool call
   ↓
...
   ↓
final outcome
```

`verl`'s Agent Loop explicitly defines a rollout this way: a user-defined loop may repeatedly call the LLM and tools, and the resulting interaction becomes the trajectory consumed by RL training. ([Verl Documentation][10])

So tools/environments become **first-class participants in training data construction**.

They can have:

```text
their own state
latency
errors
side effects
nondeterminism
resource ownership
```

without being model components.

### Token identity becomes important

There's a surprisingly nasty implementation detail documented by `verl`: reconstructing the final conversation through chat messages can produce different token IDs from the tokens actually generated during each rollout turn.

Their guidance is essentially:

```text
do not:
tokens → text → messages → retokenize

preserve:
actual rollout token stream
```

([Verl Documentation][10])

That's very relevant to a generic training system.

The optimized trajectory must correspond to **what the generating policy actually sampled**, not merely an equivalent-looking text serialization.

So:

> **Human-readable trajectory representation and optimizer-facing trajectory representation may be different artifacts.**

That's a great constraint.

---

# 10. Async RL introduces versioning and staleness as training semantics

Synchronous PPO is comparatively simple:

```text
generate with policy N
↓
train on those rollouts
↓
produce policy N+1
↓
generate again
```

Modern systems increasingly overlap them.

As of September 2026, `verl`'s V1 trainer has explicit `colocate_async` and `separate_async` modes, an asynchronous replay buffer/TransferQueue, standalone rollout GPUs, and partial-rollout support. ([Verl Documentation][15])

That changes the topology toward:

```text
rollout workers ───────────────┐
     ↑                         │
     │ policy snapshots        ↓
     │                    trajectory queue
     │                         │
trainer ───────────────────────┘
    ↓
continuously changing policy
```

Now the system has to care about questions like:

```text
which policy version produced this trajectory?
how stale is it?
is it still valid for this update?
does it need importance correction?
should it be rejected?
```

Current `verl` includes rollout-correction machinery explicitly aimed at off-policy discrepancies. ([Verl Documentation][16])

This gives one of the strongest constraints in this whole research set:

> **Generated training examples may carry provenance tying them to a particular historical model state, and their admissibility or weighting can depend on that provenance.**

That's very different from normal static training data.

---

# 11. Partial rollout introduces resumable *experience construction*

There's an even stranger current case.

`verl`'s fully asynchronous tool-agent implementation supports interrupting an unfinished rollout around parameter synchronization, saving its current state, and later resuming the interaction rather than starting the trajectory over. ([Verl Documentation][17])

So now we have:

```text
completed training state
partial training data
partial environment interaction
```

all at once.

That's another artifact class:

```text
in-progress trajectory state
```

which is neither:

```text
model checkpoint
dataset record
final rollout
```

For long-running agent/tool environments, that could matter a lot.

I wouldn't design for it prematurely, but I'd record the constraint:

> **A generated training sample may itself have a lifecycle and resumable intermediate state.**

---

# 12. Static data and generated data should probably remain conceptually separate

Across these recipes:

```text
DPO
dataset item
    → directly optimized

KTO
dataset item
    → directly optimized

PPO
dataset prompt
    → rollout
    → derived experience
    → optimized

GRPO
dataset prompt
    → rollout group
    → group-relative experience
    → optimized

agent RL
dataset prompt
    → environment interaction
    → trajectory
    → optimized
```

So I'd explicitly distinguish:

```text
source example
generated experience
training experience
```

They may all contain tokens, but they're not the same thing.

That could become particularly important if RL eventually joins the same generic Trainer you're designing.

---

# 13. Cross-case findings

I’d end with something roughly like this:

| Assumption                                         | What the recipes show                                                                                             |
| -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Training data always exists before the step        | False; PPO/GRPO generate experience during training.                                                              |
| One dataset item corresponds to one sequence       | False; DPO needs pairs, GRPO needs rollout groups.                                                                |
| One model forward produces one loss                | False; preference objectives compare several executions/models.                                                   |
| Frozen models are incidental                       | False; reference and reward models can define the objective.                                                      |
| Reward means a neural reward model                 | False; rules, verifiers, environments and hybrid sources are valid.                                               |
| Batch partitioning is always implementation detail | False; KTO/GRPO have group/batch-dependent estimators.                                                            |
| Policy and inference model are one runtime object  | False; training and rollout engines can be separate realizations.                                                 |
| Generated samples are timeless data                | False; PPO/asynchronous RL ties them to generating policy state.                                                  |
| A trajectory is just text                          | False; exact token history, masks, tool results and provenance may matter.                                        |
| Resume state is only model + optimizer             | Not necessarily; rollout queues, algorithm controllers, critics, and potentially partial trajectories can matter. |

And I think the main architectural conclusion should be:

> **Preference/RL post-training turns “training” from a repeated model-forward optimization loop into a dataflow that may generate its own training examples, execute several model and non-model roles, derive intermediate statistics over pairs or groups, and update multiple pieces of mutable state.**

Then this more actionable one:

> **A generic Trainer should not assume that the dataloader directly supplies the object consumed by the loss. A recipe may need to transform source data into generated experience before optimization can begin.**

And probably the most important role distinction:

```text
policy identity
    ≠ rollout realization
    ≠ reference policy
    ≠ historical policy snapshot
```

Those can share ancestry—even weights at some moment—while having completely different responsibilities during training.

This topic definitely deserves its own note. It exposes a bunch of assumptions that ordinary SFT/diffusion never forces us to confront.

[1]: https://verl.readthedocs.io/en/latest/examples/ppo_code_architecture.html?utm_source=chatgpt.com "PPO Example Architecture — verl documentation"
[2]: https://arxiv.org/abs/2203.02155?utm_source=chatgpt.com "Training language models to follow instructions with human feedback"
[3]: https://arxiv.org/abs/2305.18290?utm_source=chatgpt.com "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
[4]: https://huggingface.co/docs/trl/v0.12.2/dpo_trainer?utm_source=chatgpt.com "DPO Trainer · Hugging Face"
[5]: https://huggingface.co/docs/trl/v0.7.9/en/dpo_trainer?utm_source=chatgpt.com "DPO Trainer · Hugging Face"
[6]: https://arxiv.org/abs/2402.01306?utm_source=chatgpt.com "KTO: Model Alignment as Prospect Theoretic Optimization"
[7]: https://huggingface.co/docs/trl/main/kto_trainer?utm_source=chatgpt.com "KTO Trainer · Hugging Face"
[8]: https://verl.readthedocs.io/en/v0.4.0/hybrid_flow.html?utm_source=chatgpt.com "HybridFlow Programming Guide — verl documentation"
[9]: https://verl.readthedocs.io/en/v0.5.x/_modules/verl/trainer/ppo/core_algos.html?utm_source=chatgpt.com "verl.trainer.ppo.core_algos — verl documentation"
[10]: https://verl.readthedocs.io/en/latest/advance/agent_loop.html?utm_source=chatgpt.com "Agent Loop — verl documentation"
[11]: https://verl.readthedocs.io/en/latest/extend_guide.html?utm_source=chatgpt.com "How to Extend verl — verl documentation"
[12]: https://verl.readthedocs.io/en/latest/advance/reward_loop.html?utm_source=chatgpt.com "Reward Loop — verl documentation"
[13]: https://arxiv.org/abs/2402.03300?utm_source=chatgpt.com "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
[14]: https://github.com/verl-project/verl/blob/main/docs/algo/grpo.md?utm_source=chatgpt.com "verl/docs/algo/grpo.md at main · verl-project/verl · GitHub"
[15]: https://verl.readthedocs.io/en/latest/advance/v1_async_trainer.html?utm_source=chatgpt.com "V1 Async Trainer — verl documentation"
[16]: https://verl.readthedocs.io/en/latest/algo/rollout_corr.html?utm_source=chatgpt.com "Rollout Correction — verl documentation"
[17]: https://verl.readthedocs.io/en/latest/advance/fully_async.html?utm_source=chatgpt.com "Recipe: Fully Async Policy Trainer — verl documentation"
