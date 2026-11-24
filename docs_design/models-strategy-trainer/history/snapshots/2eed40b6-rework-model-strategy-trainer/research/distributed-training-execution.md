**This is a non-main topic that will need deeper research when the topic becomes current. It is also one of the topics that specifically set up the goals for the current rework**

# Distributed Training Execution

The main framing should be:

```text
parameter ownership
    ≠
parameter residency
    ≠
parameter materialization
    ≠
compute placement
    ≠
optimizer placement
    ≠
gradient synchronization group
```

That's the real architectural lesson.

### 1. Sharding is not just "put half the model on each GPU"

Use **FSDP2 / ZeRO-3** as the clean case.

PyTorch FSDP2 turns parameters into sharded `DTensor`s. Before a sharded module executes, its parameters are all-gathered; afterwards they may be resharded. Optimizer state and gradients are likewise sharded. `reshard_after_forward` even makes the lifetime of the full parameter representation configurable. ([PyTorch Documentation][1])

So during one iteration a logical parameter can look like:

```text
persistent state
    rank-local parameter shard
             │
             │ all-gather
             ▼
temporary full parameter
             │
          forward
             │
          reshard
             ▼
rank-local shard
             │
          backward
             │
       reduce-scatter
             ▼
rank-local gradient shard
```

That immediately gives:

> **The tensor used by a forward need not be the persistently resident representation of the parameter.**

Which is very similar to what quantized training just showed us, except the alternate representation is produced by **distribution** rather than quantization.

Also worth recording: FSDP2 may use **different sharding meshes for different parameters**, explicitly including MoE experts. ([PyTorch Documentation][1])

So even:

```text
model
    ↛ one sharding policy
```

### 2. Different forms of parallelism split different semantic axes

Megatron is probably the best reference here because it exposes them all distinctly:

| Mode    | What actually gets partitioned       |
| ------- | ------------------------------------ |
| DP/FSDP | training-state replicas/shards       |
| TP      | computation inside individual layers |
| PP      | model depth / execution stages       |
| CP      | sequence/context dimension           |
| EP      | MoE experts                          |

These compose rather than substitute for one another. Megatron explicitly supports combined TP + PP + CP + EP + DP topologies. ([NVIDIA Docs][2])

This is a useful correction to a generic abstraction like:

```text
distributed = world_size
```

The actual runtime has **multiple overlapping process groups with different meanings**.

Megatron's current process-group configuration contains separate groups for TP, PP, CP, EP, expert-DP and combinations of them. ([NVIDIA Docs][3])

So:

> **Distributed membership is contextual: two ranks may be peers for one state transition but not another.**

### 3. Context parallelism demonstrates that distribution can change the forward execution graph

CP is especially useful for our research because it isn't about weight storage.

Megatron splits the sequence itself across ranks. Most operations can work locally, but attention needs K/V information from other sequence partitions, so communication becomes part of the attention execution. ([NVIDIA Docs][4])

Conceptually:

```text
same model
same parameter topology

sequence
 ├── chunk A → GPU 0 ─┐
 ├── chunk B → GPU 1 ─┤
 └── chunk C → GPU 2 ─┤
                       ↓
                distributed attention
```

So:

> **A distribution strategy may affect activation topology and forward semantics without changing parameter ownership.**

That's worth keeping separate from FSDP.

### 4. Pipeline parallelism turns one logical forward/backward into a schedule

Pipeline parallelism partitions layers and executes **microbatches through stages**.

That means the trainer can no longer naively conceptualize:

```text
batch
  ↓
forward
  ↓
backward
```

because physically it's more like:

```text
microbatch 1: stage0 → stage1 → stage2
microbatch 2:          stage0 → stage1 → stage2
microbatch 3:                   ...
```

with interleaved forward/backward scheduling and pipeline bubbles. Megatron exposes multiple PP schedules including interleaved execution. ([NVIDIA Docs][5])

Useful constraint:

> **One optimizer step may contain an internal execution schedule whose unit is a microbatch rather than the logical batch.**

That's relevant to gradient accumulation too.

### 5. Gradient synchronization isn't universally "all-reduce everything"

Megatron explicitly has to finalize gradients differently depending on the parallel topology: ordinary data-parallel gradients, tied parameters across pipeline stages, and MoE experts living on different expert groups all have different synchronization requirements. ([NVIDIA Docs][6])

So:

```text
parameter
    ↓
its ownership/topology
    ↓
determines synchronization semantics
```

rather than:

```text
backward()
↓
all_reduce(grads)
```

That's a pretty important generic-Trainer boundary.

### 6. Offload is several different mechanisms

I would strongly separate:

```text
parameter offload
optimizer-state offload
optimizer-compute offload
gradient offload
activation offload
checkpoint/recompute
```

because they have completely different lifecycles.

DeepSpeed ZeRO-Infinity can move parameters and optimizer state through:

```text
GPU
 ↕
CPU
 ↕
NVMe
```

and uses buffering/prefetching plus pipelined reads/writes to overlap storage movement with computation. ([DeepSpeed][7])

Megatron separately exposes **optimizer CPU offload**, including partial offload fractions and overlap of optimizer compute with device transfers. ([NVIDIA Docs][8])

And activation offload is different again: current Megatron can move selected intermediate activations such as QKV inputs, attention inputs or MoE activations to CPU during forward and fetch them again before backward. ([NVIDIA Docs][9])

This gives:

> **"Offloaded component" is too coarse a concept. What is offloaded may be persistent state, optimizer state, gradients, or ephemeral backward-required activations.**

### 7. Offload is scheduling, not merely placement

This might be the most useful point for your architecture.

If something lives permanently on CPU, that's placement.

But this:

```text
GPU compute layer N
       │
       ├──────── prefetch layer N+1 weights from CPU
       │
       └──────── evict/release layer N-1
```

is an **execution schedule**.

Same with activation offload:

```text
forward
   ↓
produce activation
   ↓
async D2H
   ↓
free GPU copy
   ...
backward approaches
   ↓
prefetch H2D
   ↓
consume activation
```

So:

> **Residency policy can require active orchestration synchronized with the execution graph.**

That's closer to a backend/execution responsibility than a static model property.

### 8. Checkpoints should ideally survive the distribution topology

This is another very good training-specific point.

PyTorch Distributed Checkpoint exposes state dictionaries using canonical parameter names and can reshard model/optimizer state under a different trainer count or combination of parallelisms. ([PyTorch Documentation][10])

That's evidence for treating:

```text
logical training state
       ≠
current physical shard layout
```

A bad checkpoint abstraction would serialize:

```text
rank_3_tensor_fragment
```

as the authoritative identity.

A stronger one serializes something conceptually like:

```text
parameter X
optimizer state for X
    ↓
current distributed representation is incidental
```

### Distributed-execution conclusion

The note could finish with:

```text
Model / recipe
    owns semantic components and required relationships

Execution backend
    owns physical partitioning
    communication
    materialization
    residency movement
    overlap schedules

Trainer
    coordinates logical steps / optimization / lifecycle
```

with an important qualification:

> **The backend cannot choose arbitrary partitioning blindly. The model/recipe may need to expose constraints such as tied parameters, atomic blocks, expert identity, sequence relationships, or execution ordering.**

So distribution should probably be **derived from semantic requirements**, rather than allowed to redefine them.

---

## The two strongest conclusions

For distributed execution, I'd preserve this:

> **Distribution is a physical realization of logical training state. Sharding, materialization, residency, communication group and optimizer placement may all differ without changing the semantic identity of a parameter or component.**

For data, this one:

> **The data pipeline produces a stateful sequence of model-facing representations. Caching, sampling, blending, packing, augmentation and curriculum determine that sequence and therefore participate in training semantics and resume correctness—not merely input performance.**

And there's a nice symmetry between them:

```text
MODEL SIDE
logical parameter
    ↓
shard / gather / offload / execute

DATA SIDE
logical source
    ↓
sample / transform / cache / pack / execute
```

Both are really cases of **logical identity versus physical realization**.

---

[1]: https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html?utm_source=chatgpt.com "torch.distributed.fsdp.fully_shard — PyTorch main documentation"
[2]: https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html?utm_source=chatgpt.com "Parallelism Strategies Guide — Megatron Core"
[3]: https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.process_groups_config.html?utm_source=chatgpt.com "core.process_groups_config — Megatron Core"
[4]: https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html?utm_source=chatgpt.com "Context Parallel Package — Megatron Core"
[5]: https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/core/pipeline_parallel.html?utm_source=chatgpt.com "pipeline_parallel package — Megatron Core"
[6]: https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/core/distributed.html?utm_source=chatgpt.com "distributed package — Megatron Core"
[7]: https://deepspeed.readthedocs.io/en/stable/_modules/deepspeed/runtime/zero/offload_config.html?utm_source=chatgpt.com "deepspeed.runtime.zero.offload_config — DeepSpeed 0.19.7 documentation"
[8]: https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.optimizer.optimizer_config.html?utm_source=chatgpt.com "core.optimizer.optimizer_config — Megatron Core"
[9]: https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/fine_grained_activation_offloading.html?utm_source=chatgpt.com "Fine-Grained Activation Offloading — Megatron Core"
[10]: https://docs.pytorch.org/docs/main/distributed.checkpoint.html?utm_source=chatgpt.com "Distributed Checkpoint - torch.distributed.checkpoint — PyTorch main documentation"
