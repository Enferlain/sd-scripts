# Training Architecture Research

**Some of these upstream repos can be found at `home\imi\Projects\sd-scripts-resources`. If needed, repos for further research can be cloned there.**

These notes collect concrete training recipes and framework studies used to stress-test the trainer/model-strategy architecture.

The goal is **not** to document every training method or reproduce papers. Each note looks for cases that challenge assumptions about model ownership, execution, representations, optimization, caching, staging, artifacts, and resume behavior.

Claims in recipe-oriented notes generally use:

* **Observed** — directly established by the examined source.
* **Inferred** — follows from the implementation or documented behavior.
* **Unknown** — not established by the examined sources.

Where possible, research is grounded in exact repositories, versions/commits, configs, entrypoints, and implementation locations.

## Notes

| File                                                                               | Focus                                                                                                                                                                                            |
| ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [`anima-llm-adapter.md`](anima-llm-adapter.md)                                     | Anima's native LLM conditioning adapter: representation boundaries, independent trainability, caching inside a conditioning pipeline, and artifact/runtime ownership.                            |
| [`autoencoders-vae.md`](autoencoders-vae.md)                                       | Autoencoder/VAE training across image, video, and audio; posterior/bottleneck roles, latent representations, adversarial training, normalization/packing, and deployable vs resumable artifacts. |
| [`conditioning-and-control-networks.md`](conditioning-and-control-networks.md)     | ControlNet, T2I-Adapter, IP-Adapter, and LLLite-style side networks; frozen-base gradient flow, control preprocessing, injection topology, initialization, and separately installable artifacts. |
| [`distributed-training-execution.md`](distributed-training-execution.md)           | FSDP/ZeRO, tensor/pipeline/context/expert parallelism, offload, communication groups, transient parameter materialization, and logical state vs physical distribution.                           |
| [`hackable-diffusion.md`](hackable-diffusion.md)                                   | Google's Hackable Diffusion framework as an architecture study: separation of corruption, time sampling, prediction semantics, loss, inference, sampling, and multimodal composition.            |
| [`llm-training.md`](llm-training.md)                                               | Modern LLM pretraining/post-training recipes, including sparse attention, MoE, multimodal training, autoregressive and diffusion language modeling, and architecture/optimizer co-design.        |
| [`model-surgery-and-staged-topology.md`](model-surgery-and-staged-topology.md)     | Training where model topology changes: sparse-attention transitions, dense-to-MoE upcycling, progressive growth, state migration, transition objectives, and topology-aware checkpoints.         |
| [`pixel-space.md`](pixel-space.md)                                                 | Direct/pixel-space training and what changes when learned latent compression is absent or reduced.                                                                                               |
| [`precision-and-quantization-training.md`](precision-and-quantization-training.md) | Mixed precision, quantized-base training, FP8/FP4, QAT, master weights, quantization state, and distinctions between storage, compute, optimizer, and export representations.                    |
| [`preference-and-rl-post-training.md`](preference-and-rl-post-training.md)         | Reward modeling, DPO/KTO, PPO/GRPO, rollout engines, reward sources, generated experience, policy versioning, agentic trajectories, and asynchronous RL.                                         |
| [`teacher-student-distillation.md`](teacher-student-distillation.md)               | Teacher/student relationships across distillation recipes, including same-model roles, cross-architecture teachers, offline teachers, evolving teachers, and remote inference.                   |
| [`training-data-pipeline.md`](training-data-pipeline.md)                           | Dataset transformation, caching, packing, bucketing, mixtures, streaming, resumable data state, and curricula as part of training semantics.                                                     |
| [`video-audio-training.md`](video-audio-training.md)                               | Video/audio and joint multimodal training, synchronized representations, modality-specific time/noise semantics, multi-component execution, and cross-modal relationships.                       |

## Recurring questions

Across the notes, the main questions are:

* What is a **model component**, and what is only an execution role?
* Which representations exist between raw data and the optimized model?
* Which components execute, which receive gradients, and which are optimizer-owned?
* What can be cached, and what invalidates that cache?
* Which state belongs to the model, optimizer, recipe, backend, or training run?
* Can one component participate multiple times or in multiple roles within one step?
* Can topology, trainability, data policy, or objectives change between stages?
* What must be preserved for exact resume versus deployment?
* Which relationships are model-specific, and which belong in generic training machinery?

The notes are intended as **design evidence**, not as an API specification. Repeated patterns across independent recipes are stronger signals than any single framework's class hierarchy or terminology.
