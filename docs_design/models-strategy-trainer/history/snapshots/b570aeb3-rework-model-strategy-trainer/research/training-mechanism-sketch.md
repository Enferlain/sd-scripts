# Training mechanism: working sketch

This is an **exploration record**, not an accepted design, API, implementation plan, or replacement for the [direction](../../../../docs_design/models-strategy-trainer/strategy_system_direction.md) and [OpenSpec design](../design.md). It collects concrete pressure cases as we investigate whether one training engine can be composed for different runs. Source behavior, implications, and choices we have not made are kept separate. Update this sketch as cases change the picture; move only agreed conclusions into governing artifacts.

## Working frame

The [direction](../../../../docs_design/models-strategy-trainer/strategy_system_direction.md) is the starting point, not a candidate to reopen: a pre-existing training contract guides explicit strategy authoring; fulfillment establishes an accepted run arrangement; one Trainer engine executes that arrangement without treating the authored strategy as its ordinary per-step collaborator. The current investigation asks what **composable run IR/graph** that engine needs in order to handle the repository's present behavior and the cases indexed in the [research README](README.md). A whole-run graph is a live possibility. It is different from a graph of every PyTorch tensor operation, and its granularity and execution form have to follow the capabilities.

The graph needs to expose dependencies, state ownership, changing inputs, effects, and lifecycle boundaries wherever one owner or activity must coordinate with another. Model-specific mathematics need not be converted into a universal taxonomy. Failure boundaries must reflect what actual operations can support; calling an entire training step atomic would not make it so.

## Case: alternating adversarial optimization and failure

### Observed behavior

- [Open-Sora Plan's WFVAE trainer](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/f7fa604f4e3a523d6b973e4c89a5620ed1aff65a/opensora/train/train_causalvae.py) selects generator work on even `current_step` values and discriminator work on odd ones. One selected optimizer branch runs per batch. `current_step` increments **after** that branch; validation and periodic checkpointing follow. The generator and discriminator updates are not one indivisible D+G cycle.
- The [Stable Audio autoencoder trainer](https://github.com/Stability-AI/stable-audio-tools/blob/3241adba4fc2a85cf5b29d9eb68d42f40a28e820/stable_audio_tools/training/autoencoders.py) also chooses one branch by step parity and manually advances that branch's optimizer and scheduler. On a generator branch, discriminator computation can still contribute differentiably to the generator loss while discriminator parameters do not receive that branch's update. These are distinct execution and update roles.
- In WFVAE, `scaler.step(optimizer)` may skip the actual optimizer update for non-finite gradients, while `current_step` still advances. This follows from the source and [PyTorch's AMP behavior](https://docs.pytorch.org/docs/stable/notes/amp_examples.html). Thus a scheduled/completed training iteration is not automatically proof that weights changed.
- WFVAE's checkpoint contains generator and discriminator weights, both optimizers, scaler, sampler, `current_step`, and EMA shadow. However, the inspected resume path does not explicitly load the saved EMA shadow, and it reads `start_epoch` while the shown epoch loop starts at zero. This checkpoint is evidence of intended continuation state, **not** evidence of exact replay. See also the broader [autoencoder research note](autoencoders-vae.md).

### Failure walk-through

Assume a discriminator iteration finishes, then the next generator iteration fails. These are different batches/iterations in the WFVAE example.

| Failure point | What can be said without inventing rollback |
| --- | --- |
| Before the generator optimizer acts | The preceding discriminator change remains. Generator parameters may be unchanged, but computation, RNG, gradients, or operation-owned state may already have changed. An automatic in-process retry is not generally justified. |
| During or after the generator optimizer action, before progress records completion | Generator parameters and optimizer state may have changed while the progress cursor still identifies that generator action as next. Blind retry may duplicate an update. A failed optimizer call cannot generally be assumed all-or-nothing. |
| After progress advances, before the next durable snapshot | Live state has advanced, but a process restart can recover only the last coherent saved state, not the unsaved progress. |

There is no generic transaction spanning model weights, optimizer internals, progress, data position, adaptive behavior, and checkpoint publication. The preparation-time publication guarantee in [design D8](../design.md) is a different, narrower concern; it must not be copied onto optimizer execution.

### Design pressure, not yet an API

- An accepted run must express **which optimization action is due**, what state it can update, and what later actions depend on it. Separate optimizer units may advance at different cadences. A D+G pair is not atomic by default; if some future method requires a larger logical boundary, that meaning needs an explicit mechanism and a realizable failure policy.
- Distinguish an action being attempted, the optimizer actually changing state, progress being recorded, and that state becoming durable in a snapshot. One generic `global_step` cannot be assumed to mean all four.
- After an ambiguous mutation or process/rank failure, the safe generic path is to stop using the uncertain live state and restore from a coherent runtime snapshot whose declared coverage is sufficient. Any narrower in-process recovery requires evidence about the particular operation and its owned state; the engine should not imply it for every operation.
- A snapshot needs the relevant optimizer units, progress/scheduling position, input position, and stateful accepted behavior if it claims exact same-run continuation. The trained product is a different artifact.

## Case: progressive distillation changes stage mid-run

### Observed behavior

- The pinned [OpenAI Consistency Models training loop](https://github.com/openai/consistency_models/blob/e32b69e/cm/train_util.py) has student, target, and optional teacher models plus an optimizer and EMA state. Its scale schedule depends on global progress. After a successful optimizer action, it updates EMA/target state, checks for a progressive-distillation scale change, and only then increments its progress counters.
- On a scale change, `reset_training_for_progdist()` copies student weights into the existing teacher model, constructs a fresh RAdam optimizer, resets ordinary EMA parameters, may change the LR-annealing horizon, and resets the stage-local counter. These are several live mutations in sequence, **not** an atomic transition implemented by that source.
- The source's roles are subtler than a simple “student becomes next teacher” picture: the progressive-distillation loss uses `teacher_model` at the initial scale but selects `target_model` at later scales. The copy into `teacher_model` does not by itself prove that this object supplies every later stage's supervision. The [research note](teacher-student-distillation.md) is useful for identifying the state changes, but its relationship diagram should not be treated as an exact execution trace.
- The loop saves EMA, optimizer, target, and teacher states in separate files, then writes the main model file last to reduce a restart race. It does not provide a multi-file transaction. The resume path can load the corresponding states when present, but the inspected source does not establish bit-identical input/RNG continuation. [Source](https://github.com/openai/consistency_models/blob/e32b69e/cm/train_util.py).

### Failure walk-through

If failure occurs after the student optimizer action but partway through the reset, the live objects may disagree about the stage: for example, teacher weights may have been copied while the old optimizer remains, or the new optimizer may exist while EMA and the stage-local counter still have old values. The source has no rollback for those intermediate states. If it fails after completing the reset but before a coherent checkpoint is available, a new process can resume only from an earlier complete saved state, not the unsaved transition.

Writing the main model file last helps avoid selecting one incomplete checkpoint, but does not make all files durable together or repair an interrupted live transition. This is the same distinction seen in the adversarial case: **in-process visibility**, **mutation failure**, and **crash restoration** are different promises.

### Design pressure, not yet an API

- An accepted stage policy must be able to change several dependent concerns together: selected execution behavior, participant/relationship meaning where applicable, optimizer runtime, EMA/target state, preparation freshness, and stage/global progress. The next dependent training action must not observe a mixed old/new stage.
- Where replacement candidates can be built and checked before installation, a short coherent publication boundary is plausible. Where a backend or operation must destructively mutate live state, a likely extension of [design D8](../design.md) would withdraw affected guarantees before mutation and avoid presenting failure as rollback. The exact cross-owner stage-transition mechanism remains to be designed.
- Copying student weights to an existing teacher object does not decide participant identity. Under the [run-authority direction](../design.md), identity, binding continuity, role, and lineage are separate questions; the accepted transition must resolve them rather than infer them from a Python copy operation.
- The engine needs stage-aware progress and restoration coverage. It does not need a special `ProgressiveDistillationTrainer`, nor must the core engine know the mathematics of the distillation loss.

## Related pressure: data exposure

The [future data-accounting note](../../../../docs_design/future_ideas/data_accounting.md) distinguishes discovered, assigned, ready, and seen samples, but does not define whether “seen” means fetched, used in computation, or contributed to an effective parameter update. Failure and alternating optimization make those events diverge: a batch can be processed for D but not G, or processed while AMP skips an update.

This is not a reason to settle one universal exposure policy now. It is a reason for the future input/accounting mechanism to retain enough distinct events to support the policy chosen for a particular run. Scheduling fairness, metrics, and exact restoration may use different projections of those events.

## Case: input production progresses independently of training

### Frame of reference

The current [Trainer](../../../../library/training/runners/trainer.py) finishes `run_caching()` before model/optimizer preparation and the training loop. The [caching phase](../../../../library/training/phases/caching.py) processes the manifest and waits for all ranks; the [epoch setup](../../../../library/training/phases/training_loop.py) constructs a fixed epoch manifest and dataloader from the available training manifest. [DataLoader workers](../../../../library/data/dataloader.py) can prefetch batches within that setup. Thus the current code already overlaps some batch loading, but not ongoing dataset discovery/cache production with training consumption.

The listed cases exert different pressure:

| Case | Producer/consumer relationship | Distinct correctness condition |
| --- | --- | --- |
| [Async TE offload and on-the-fly encoding](../../../../docs_design/future_ideas/async_te_offload_otf.md) | A producer encodes already-planned upcoming captions while the training consumer advances. | An output must match the exact epoch/batch, caption choice, encoder meaning, and distributed order; queue position alone does not establish that. |
| [Async caching](../../../../docs_design/future_ideas/async_data.md), [streaming](../../../../docs_design/future_ideas/streamed_training.md), and [sharded cache](../../../../docs_design/future_ideas/data_shards.md) | Discovery/preparation can publish newly ready samples while training consumes a selected ready set. | Registry/cache readiness, cache generation, failure, and selection/exposure state must be distinguishable. Snapshot-ready epochs and live-ready selection are different policies. |
| [Megatron Energon](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/megatron_energon.html) | A streaming loader supplies batches from a stateful distributed data stream. | Loader/stream position can require its own checkpoint state; training step count alone does not reconstruct it. |
| [veRL asynchronous RL](https://verl.readthedocs.io/en/latest/advance/v1_async_trainer.html) | Rollout workers produce trajectories while the trainer advances and periodically synchronizes policy weights. | A trajectory may contain multiple policy versions; admission can depend on staleness, failure, and group policy. Its async checkpoint restores finished groups but reissues pending/running prompts. This is stronger than ordinary prefetched data. |

The repository's future-data notes are proposals, not implemented guarantees. The upstream examples establish that these behaviors are possible and meaningful, not that our first migration must implement every one of them.

### Failure walk-through

- If a TE producer returns a late result for the wrong caption or encoder state, the consumer must reject or recompute it; silently pairing it with the next batch changes training meaning. A producer timeout or failure needs an accepted wait, fallback, or run-failure policy, not an assumption that an empty queue means valid empty conditioning.
- If cache payload writing succeeds but readiness publication fails, the payload may be an orphan that can be checked or retried; it is not automatically available for training. If readiness is published before the payload is usable, a selected batch may fail. The durable registry and payload consistency boundary need to be designed together; a work queue is not the source of truth.
- If training fetches a ready batch and later fails, producer progress, data exposure, and optimizer progress may disagree. As in the adversarial case, there is no generic transaction spanning dequeue, computation, optimizer update, and producer/registry state. Exact restoration needs a coherent input position or a defined way to reconstruct/reissue work alongside the model/optimizer state.
- For versioned rollout data, a produced item can be complete yet not admissible for the current policy. Freshness is a semantic check over provenance, not a queue-empty/queue-full condition. The [veRL V1 guide](https://verl.readthedocs.io/en/latest/advance/v1_async_trainer.html) demonstrates drop/wait choices and prompt reissue on checkpoint recovery; neither policy should be silently universalized.

### Design pressure, not yet an API

- The training mechanism must leave room for **longer-lived input producers and stateful providers** whose progress is not identical to a Trainer step or epoch. It need not schedule every worker task itself. A recognized data/caching or generated-experience capability could own the internal queue/workers while exposing bounded lifecycle, readiness, input delivery, error, and restoration behavior to the run.
- The handoff from producer to training needs accepted meaning: sample or work identity, relevant provenance/dependency revision, readiness, the selected batch/view, and the policy governing wait/reject/recompute. Which exact fields exist depends on the capability; there should be no universal image sample or RL trajectory type.
- The Trainer remains responsible for coordinating its execution and cleanup, but the input provider and producer retain their own state and writers. The [OpenSpec run-state split](../design.md) already names input and capability state; it does not yet define this independently progressing producer/consumer exchange.
- A snapshot-ready epoch, a live-ready stream, and versioned generated experience can all meet a narrow training input boundary while having materially different selection, admission, progress, and recovery policies. The execution representation must not make epoch completion or a fixed finite manifest a universal prerequisite.
- This is an architectural neutrality/support test for the rework, not a commitment to build streaming ingestion, async cache production, or asynchronous RL in the first production implementation.

## Capability survey: what the run IR has to express

This section works **forward from the agreed architecture**. The [research index](README.md) contains pressure cases, not a promise to implement every method in the first migration. Current code supplies working behavior but not the target IR. The question is which shared meanings the IR must retain so these behaviors can be composed without a family-specific Trainer or an unrestricted runtime strategy.

### Present repository behavior

| Current seam | Evidence | IR consequence |
| --- | --- | --- |
| Fixed setup-to-loop order | [Trainer.train()](../../../../library/training/runners/trainer.py) runs setup, caching, model preparation, optimizer preparation, startup evaluation, the training loop, and finalization in a fixed sequence. | The initial SD/SDXL/SD3 path is one lifecycle instance the graph must express, not the universal sequence of all runs. |
| Family computation and dynamic objective | The [training loop](../../../../library/training/phases/training_loop.py) invokes selected family batch behavior, advances objective state, consumes loss/observation results, then performs standard backward and optimizer mechanics. [SDXL batch behavior](../../../../library/strategies/sdxl/diffusion.py) contains representation, conditioning, prediction, and loss decisions. | Selected executable behavior, changing inputs, observations, owned state, and Trainer-owned optimization must remain connected without per-step calls to the authoring interface. A single diffusion-shaped result is not universal. |
| Caching and due capabilities | [Caching](../../../../library/training/phases/caching.py) currently finishes before training. The [loop](../../../../library/training/phases/training_loop.py) schedules validation, sampling, and saving after eligible steps. | The graph needs lifecycle work and triggers in addition to the training action; overlapping production is a future extension of this current sequential case. |
| Optimization realization | The [optimization plan](../../../../library/optimization/types.py) already separates logical and execution groups, but it contains live runtime members. Current mode/phase code creates and prepares optimizer state. | IR-level training subjects and optimizer-unit meaning must precede concrete parameter groups, wrappers, and backend handles. |
| Active aggregate strategy and mode | The [current contract](../../../../library/strategies/base/contracts.py) aggregates loading, caching, validation, sampling, persistence, preparation, and batch execution; the loop also invokes mode hooks. | These behaviors must survive as selected operations and capabilities, while their present whole-Trainer callback topology must not become the IR execution protocol. |

The graph-backed code check for these paths used project home-imi-Projects-sd-scripts, generation 2026-09-23T20:18:29Z. The cited paths had matching filesystem metadata and no recorded indexing gaps; that is a best-effort coverage signal, not proof of exhaustive discovery.

### Research capability -> required graph meaning

| Capability family in the [index](README.md) | Concrete pressure | IR/graph meaning to retain |
| --- | --- | --- |
| [Autoencoders](autoencoders-vae.md), [pixel-space](pixel-space.md), [video/audio](video-audio-training.md), [Hackable Diffusion](hackable-diffusion.md) | Several representations, modalities, losses, time axes, and model calls may participate in one action. A VAE or one image-shaped batch is not universal. | Typed producer/consumer values and composable computation regions, with compatibility supplied by selected feature/domain contracts rather than a global media taxonomy. |
| [Anima adapter](anima-llm-adapter.md), [control networks](conditioning-and-control-networks.md), [distillation](teacher-student-distillation.md) | A cache cut may sit between an encoder and trainable bridge; a frozen base can remain on the gradient path; one participant may have several roles or routes; an offline teacher need not be a live participant. | Separate participant identity, execution route, representation boundary, gradient participation, optimizer membership, relationship, and artifact membership. Edges must not infer one from another. |
| [Adversarial autoencoders](autoencoders-vae.md), [LLM training](llm-training.md) | G/D actions or Muon/AdamW units may have different membership and cadence; an auxiliary objective may coexist with the main one. | Multiple named optimization actions/units and authored scheduling/combination meaning; standard Trainer mechanics can be bound to the due unit without one universal loss or update sequence. |
| [Training data](training-data-pipeline.md), [preference/RL](preference-and-rl-post-training.md) | Source data can be transformed, cached, blended, streamed, or generated from a policy. Producers and consumers may progress independently; generated experience has provenance and admissibility rules. | Input-production activities, semantic data dependencies, readiness/admission, versioned handoffs, backpressure/failure policy, and stateful input continuation. A feedback edge across time is not an ordinary same-step DAG edge. |
| [Model surgery](model-surgery-and-staged-topology.md), [LLM curricula](llm-training.md), [distillation](teacher-student-distillation.md) | Stage changes can alter topology, trainability, objective, data policy, optimizer state, and distribution together; some growth uses overlapping old/new execution. | Authority-governed graph/state revisions, state mapping and lineage, dependency invalidation, candidate preparation, and coherent visibility before dependent work. No generic rollback or crash-atomic commit is implied. |
| [Distributed execution](distributed-training-execution.md), [precision/quantization](precision-and-quantization-training.md) | One logical participant/parameter may have shards, pipeline stages, offloaded state, master weights, quantized execution views, and scaling state. Physical scheduling can change substantially. | A semantic graph distinct from its prepared physical realization. Backend lowering may add communication, materialization, or microbatch schedules while preserving accepted identity, ownership, and constraints. |
| Current and future sampling, validation, caching, products, restoration | Capabilities have their own requests, state and outputs; trained products differ from exact-run snapshots. | Lifecycle activities and externally visible effects with named owners, declared triggers/dependencies, actual-result reporting, and contributor-based restoration. |

The table does not make all research cases first-implementation features. It tells us what a graph designed for the stated ambition must represent or leave a deliberate extension point for; cases requiring new authority or recovery semantics still need an explicit contract/engine extension.

## Working IR shape to derive from those capabilities

The most useful working model is a **hierarchical semantic run graph**. “Whole-run” means the graph can express the run's lifecycle and interactions, not that every tensor kernel, queue operation, or backend collective must become a universal node.

~~~text
contract-guided authored strategy
            |
            v
accepted semantic run graph
  participants, relationships, obligations, selected behavior
  values and state dependencies
  nested work/producer/capability/transition regions
  scheduling, authority, effects, and results
            |
            v
prepared executable graph revision
  current participant routes and backend representations
  optimizer units, selected implementations, provider handles
  prebound hot-path work and lifecycle coordination
            |
            v
one Trainer engine executes the accepted run
~~~

These are stages of **one accepted meaning**, not three independent authorities. The authored strategy can remain explicit Python composition; it need not require authors to write a textual IR. Fulfillment validates semantic choices under the pre-existing contract. Preparation resolves current objects and backend realizations. A later accepted structural transition may produce a new graph/binding/preparation revision while participant identity follows the run-authority rules, not Python object identity.

The graph needs a few distinct kinds of information:

1. **Values and representations:** operation outputs consumed by other operations, with domain/feature-specific type or compatibility evidence. It should not prescribe one tensor rank, modality, prediction field, or teacher-output type.
2. **Participants, routes, and state:** stable logical references; current prepared execution views; operation-, optimization-, input-, capability-, and Trainer-owned mutable state. Their identities and writers are not inferred from graph position.
3. **Work and nested regions:** selected model/algorithm operations can be ordinary Python or composed subgraphs. Repeated training actions, conditional/staged work, long-lived producers, and due capabilities have different lifetimes. A region exposes the boundary needed for coordination without necessarily exposing its internals.
4. **More than one edge meaning:** a value dependency is different from “must happen before,” “uses current revision,” “reads this state,” or “may mutate this state.” Feedback such as adaptive sampling or rollout-policy publication crosses an explicit state/version boundary, so a cyclic run is not mistaken for a combinational cycle.
5. **Authority and effects:** standard backward, clipping, optimizer advancement, and related mechanics remain Trainer-owned under the normal profile. The IR makes their inputs, selected unit, ordering, and results visible; it need not freeze each backend action into a separate universal node. An imperative region receives only the authority granted by an explicit profile.
6. **Transitions and failures:** a transition declares what it may replace or evolve and what must be revalidated/reprepared. Fallible construction should occur before publication where possible. Destructive in-place changes, optimizer-step ambiguity, external writes, and crash restoration retain distinct failure rules.
7. **Physical lowering:** distributed and precision backends can realize the same logical work with different wrappers, shards, compute views, and schedules. They cannot silently change authored optimization, data-admission, or objective meaning.

This model makes the **graph boundary** specific: expose a relationship when contract checking, Trainer coordination, another state owner, preparation, observability, or restoration needs to see it. Keep internal math or worker scheduling inside the selected implementation when none of those boundaries are crossed. It does not rule out a whole-run graph; it describes what that graph would be a graph *of*.

### Current and planned capability graph sketches

For today's SDXL path, the target semantic graph could show the *selected relationships* below. The brackets name work, not mandatory Python classes or one-node-per-function decomposition.

~~~text
input batch
  +--> [latent representation: cached or current VAE route] ---+
  +--> [conditioning: cached or current encoder routes] -------+--> [SDXL objective/prediction work]
objective-owned adaptive state -------------------------------+              |
                                                                           +--> optimization input
                                                                           |       |
                                                                           |       v
                                                                           |  [Trainer-owned unit update]
                                                                           |       |
                                                                           |       +--> progress / due validation, sampling, product work
                                                                           |
                                                                           +--> observations --> adaptive-state update
~~~

The graph does not require Trainer to understand latents or the SDXL loss. It does need to retain which representation producer is selected, its current dependencies, the objective's feedback state, the declared optimization input, and the capability triggers. Caching can replace one producer's execution with a validated stored value **at that boundary**; it does not turn the rest of conditioning into one undifferentiated cached “text encoder output.” This is a target projection of current behavior, not a claim that the current implementation already has such a graph.

For generated-experience training, the graph has a different lifetime and a feedback edge across policy versions:

~~~text
policy state vN --> [rollout activity] --> experience tagged vN
                        |                       |
                        |                 [admit / group / score]
                        |                       |
                        |                 [training action]
                        |                       |
                        +--- later policy publication <-- policy state vN+1
~~~

The rollout activity may keep running while training updates the policy. The edge back to it is a **new published state/version**, not same-step tensor flow. Admission decides whether a trajectory generated from vN can be used at the consumer's current version. The rollout implementation and queue stay selected capability behavior; the run graph retains their lifecycle, provenance, and handoff. If a stage transition also changes policy topology or optimizer meaning, it creates an authority-governed graph/preparation revision before the next dependent training action; producer items carry enough dependency facts to be admitted or rejected against that revision.

These examples expose why a plain per-step DAG is too small, while leaving open whether the eventual Python representation stores the whole hierarchy as explicit graph objects, structured regions with graph-like dependencies, or a lowered hybrid. That storage decision follows the semantic graph above; it does not decide the architecture by itself.

## First concrete pass: one standard SDXL training action

This pass derives the graph boundary from the [current training loop](../../../../library/training/phases/training_loop.py), [SDXL batch implementation](../../../../library/strategies/sdxl/diffusion.py), and the [worked SDXL case in the OpenSpec design](../design.md). It describes target accepted meaning, not today's object layout or proposed Python class names.

### Before repeated execution

The already active contract guides an author to select and connect SDXL representation, conditioning, objective, predictor, loss, training subjects, and supported capabilities. Fulfillment checks the completed selection and establishes a run arrangement. Its semantic graph can name the denoiser, text encoders, and VAE by authority-established participant references even if a permitted component is still unbound. It also preserves the selected implementations, which outputs feed which consumers, cache alternatives and their dependencies, objective-state initialization, the standard optimization policy, and any bounded choices exposed to run configuration.

Materialization then binds real components. Preparation resolves current execution routes for all required components—not only optimizer members—and realizes the selected optimization unit. These jobs attach current evidence and runtime objects to the accepted graph; they do not ask Trainer to reconstruct the SDXL anatomy or change what the strategy authored. A route or cache result tied to an old binding revision is unready until rechecked or replaced.

### One repetition: what the graph must connect

| Relationship in this case | Needs to be visible to the accepted run | May remain inside selected behavior or backend |
| --- | --- | --- |
| Batch to latent and conditioning values | Both are required inputs to later prediction/objective work; a compatible cache may supply one boundary instead of executing its live producer. | Tensor layout, tokenization details, VAE posterior selection, and exact encoding mathematics. |
| Current prepared components | Each operation states which participant route and freshness it needs; trainability does not determine whether a component executes. | Wrapper objects, device transfers, and distributed materialization chosen during preparation. |
| Objective state and changing inputs | Current coordinates and adaptive state can affect sampling/noise/loss; observations from this repetition can update objective-owned state for later repetitions. | DDPM or flow formulas, sampled timestep tensor format, and the adaptive algorithm's private buffers. |
| Computation result | Optimization input, accounting value, observations, and declared owned-state effects are distinguishable. | Whether a particular SDXL implementation calculates those through several Python calls or one fused call. |
| Standard optimizer action | It consumes the accepted optimization input and prepared unit; Trainer owns accumulation, synchronization, backward, clipping, stepping, zeroing, and advancement under the standard profile. | Concrete optimizer/backend handles and any fused physical schedule that preserves the accepted meaning. |
| Due validation, sampling, and product work | Their selected triggers, required current routes, requests, effects, and actual outcomes are connected to run progress. | Family generation/conversion mathematics and capability-private traversal or storage internals. |

The current source order is worth making explicit: the loop advances objective scheduling before batch computation, feeds timestep/loss observations to the objective runtime after computation, then applies the loss modifier, backward and optimizer mechanics, and runs step-triggered work after a synchronized advancement. The target graph must either preserve those authored/accepted dependencies or deliberately specify a changed training meaning; it must not accidentally reorder state feedback because two operations looked independent as tensor values.

The region containing this repetition needs a completion/failure result, but **not** a claim that all its mutations are atomic. A microbatch, synchronized optimizer action, objective-state update, and saved checkpoint are different events. If backend advancement is skipped or fails, progress and observations follow the accepted policy and actual result, not the mere presence of an edge labelled “update.”

### What this case contributes to the IR

It establishes a practical first set of graph meanings: named ephemeral values, selected executable work, participant-route requirements, operation-owned state and observations, data and ordering dependencies, a standard Trainer-owned optimization action, and capability triggers/results. A region groups work with a lifecycle and failure boundary. It does **not** establish that every run has one batch, one loss, one optimizer, diffusion timesteps, a VAE, or an SDXL-style prediction call. The research capability survey above is the check against promoting those incidental SDXL details into the common vocabulary.

## Second concrete pass: attachment, gradient path, and cache cut

The [OpenSpec SDXL PEFT case](../design.md) supplies a current-behavior migration test; the [control-network](conditioning-and-control-networks.md) and [Anima adapter](anima-llm-adapter.md) research supply different shapes that the same IR must not exclude. These are not all the same kind of “adapter node.”

| Arrangement | What actually connects | What an IR must not infer |
| --- | --- | --- |
| SDXL with a PEFT method | An adapter participant is attached to resolved host targets. That relationship changes the host's effective execution route; the SDXL computation still invokes the host route. Adapter state may be the only optimization subject, or explicitly selected base substructures may also train. | Attachment does not automatically create another forward call, make the host trainable, or put every attached parameter into one optimizer unit. |
| ControlNet-style side network | A condition producer feeds a trainable side network whose output is injected into a frozen base's execution. The loss can differentiate through the base to the side network even though base parameters are not updated. | Frozen base does not mean no-grad execution; gradient participation does not mean optimizer ownership. |
| Anima's LLM representation bridge | Multiple tokenizations and a frozen Qwen representation can feed a trainable bridge before DiT-facing conditioning. A legal cache cut can occur after Qwen and before the bridge. | “Text encoder output” is not one indivisible cache value, and equal tensor shapes do not prove compatible representation meaning. |

For PEFT, the graph must retain **attachment as a relationship that changes a prepared host route**, with its own identity, source-target evidence, revision, and permitted lifecycle. A runtime forward need not show a separate adapter operation if the selected method acts inside that host route. For a side network, the graph does need a value flowing from side computation into a selected injection surface. For the bridge, it needs two different representation meanings and a cacheable producer boundary between them. A universal “adapter” node would lose these distinctions.

The common graph question is not whether a component is simply trainable. It is whether a selected operation or route (1) must execute, (2) must allow differentiation through a particular path, (3) owns parameters selected for an optimizer unit, (4) changes another route through a governed relationship, and (5) contributes to a requested product. Those are independently declared or derived under accepted obligations. The Trainer's standard optimization machinery consumes the selected subjects; it does not discover them by walking whatever happened to execute.

This pass adds **relationship and gradient-boundary meaning** to the first case's values, routes, state, and effects. It does not require the run IR to reimplement PyTorch autograd. It does require enough accepted information to prepare the correct executable path, avoid an accidental no-grad boundary, select legal optimizer members, and invalidate a cache or route when its producer/attachment dependencies change. The exact amount of gradient information exposed on a port or region remains a concrete design choice to test with custom operations and distributed preparation.

## Third concrete pass: alternating actions and a changing run

This pass combines two pressures without pretending they are one algorithm. [WFVAE and Stable Audio](autoencoders-vae.md) demonstrate alternating generator/discriminator actions; [progressive distillation](teacher-student-distillation.md) demonstrates a same-run stage change that replaces optimizer and EMA state; [model surgery](model-surgery-and-staged-topology.md) tests actual topology change and temporary old/new overlap. The observed implementations do **not** provide a common transactional step or stage-transition API. The question here is what an accepted run must say so one Trainer can execute each meaning.

### Alternation is scheduling, not a combined optimizer step

For the WFVAE shape, each selected batch has one due action. A generator action and a discriminator action can have different computation, gradient paths, optimizer units, and progress. A generator loss may execute the discriminator differentiably without selecting its parameters for that action's optimizer update, as the Stable Audio case shows. The graph therefore needs a schedule that selects an accepted action and its unit(s), not a hard-coded “run G, then D” Trainer cycle.

~~~text
accepted schedule + current progress
              |
              v
       due action (G or D)
              |
       selected computation / gradient path
              |
       Trainer-owned advancement of selected unit
              |
       actual result + progress / observations / due work
~~~

The schedule may use one shared iteration counter, separate unit counters, or another accepted coordinate. The selected action, attempted advancement, actual optimizer effect, completed iteration, and durable checkpoint are different events. The graph must preserve those meanings even when an AMP scaler skips the parameter update. A failed G action does not undo a completed earlier D action; neither a repeated-action region nor an apparent G/D “cycle” implies rollback.

### A stage change may revise the executable graph

Progressive distillation can keep its model topology while changing teacher/target state, optimizer, EMA, objective/schedule behavior, and stage-local progress. That already exceeds an ordinary branch inside one repeated action. A stronger transition such as dense-to-MoE upcycling changes parameter and execution topology; a progressive-growing GAN may intentionally execute old and new paths together while its blend evolves. These are distinct accepted transitions, not one universal `replace_model` operation.

The working graph reading is:

~~~text
current accepted view + accepted condition/event
                       |
        governed transition request and mapping
                       |
         +-------------+------------------+
         |                                |
 replacement-only path             destructive path
 build/check/prepare target         withdraw affected guarantees
 while old view stays valid         before mutating live state
         |                          mutate/check/prepare
 publish complete target view       |             |
         |                        success       failure
         |                          |             |
         |                     publish view   remain unready
         +-------------+------------+
                       |
             next dependent action
              only if view is ready
~~~

This drawing shows **alternative safety paths**, not a prescribed order of calls. For replacement-only work, construct and check the target—including its optimization/backend consequences—before an unobservable publication of the complete prepared view, as [D8](../design.md) requires. If transition work must mutate currently authoritative objects in place, affected guarantees must be withdrawn before mutation; failure cannot be made into rollback by resetting a flag. In either path, the next dependent action must not use a mixed old/new stage. A fade-in is different again: both paths are intentionally part of one accepted current graph during a bounded interval, rather than an accidentally half-installed replacement.

The transition meaning must cover the source and target graph/obligation revisions; which participants and relationships persist, change, appear, or retire; how parameter state is retained, copied, transformed, initialized, or trained before use; and separately how optimizer, EMA, scheduler, progress, input dependencies, and backend state continue, restart, or become unready. These are **policies selected by the authored behavior and governed through run authority and the relevant state owners**, not model-surgery instructions built into Trainer. A preserved participant reference does not imply the same parameters, optimizer state, or prepared route. Conversely, a physical copy does not establish identity or lineage on its own.

This also gives a concrete restoration test: a same-run snapshot must identify the current stage and graph/topology revision, installed relationships and routes, optimization/runtime state, progress, and whether a transition has already become authoritative. Restoring an old snapshot must not silently replay a completed surgery or load old optimizer state into new parameters. A trained artifact derived from a stage is not automatically an exact-runtime snapshot.

### Minimum additional graph meanings from this pass

- **Due-action selection:** accepted cadence/condition, action-specific computation and optimizer unit(s), and progress coordinates; no mandatory paired G/D transaction or single global step semantics.
- **Action result:** distinguish attempted work, effective advancement, recorded completion, and later durability. State effects and failure are scoped to their actual owner and point of occurrence.
- **Governed stage transition:** source revision, trigger, permitted state/structure mapping, affected owners/dependencies, readiness checks, and one coherent current-state view before dependent work resumes. Pure schedule/policy changes need not pretend to be model surgery.
- **Graph coexistence when authored:** old and new execution paths can both be current during a deliberate fade-in, with an accepted changing blend and eventual retirement; this is not a failed replacement.
- **Restoration identity:** stage/topology/revision and transition status are needed to interpret saved state. Optimizer and EMA migration do not follow automatically from model-weight mapping.

This pass does not settle the Python representation, invent general in-process rollback, or decide how every source's stage policy is expressed. It narrows the IR question: a whole-run graph must support scheduled **actions** and authority-governed **revisions of the executable run**, with different failure promises for ordinary updates, replacement-only transition preparation, destructive mutation, and durable recovery.

## Fourth concrete pass: versioned experience while training continues

The [veRL V1 async guide](https://verl.readthedocs.io/en/latest/advance/v1_async_trainer.html) provides a concrete example: rollout production and policy optimization can progress independently; trained weights are periodically synchronized to rollout workers; a partial trajectory can resume under a newer policy version; and a completed prompt group may be admitted, waited on, or dropped according to staleness and group rules. Its checkpoint path restores finished groups but reissues pending/running prompts. The [GRPO and agent-loop research](preference-and-rl-post-training.md) adds that several trajectories may form one algorithmic group and a trajectory may include external interactions. Those are source facts and research pressures, not a commitment to veRL's modes, queue, thresholds, or PPO algorithm in this Trainer.

### The two activities and their handoffs

~~~text
Trainer-owned policy update --> current policy state P18
          |                           |
          |                  publish selected snapshot
          |                           v
          |                 rollout replica at P17 or P18
          |                           |
          |                  generated trajectory/group
          |                  with production provenance
          |                           |
          +<-- admit/derive training input <--+
~~~

The policy used for optimization and the policy snapshot available to a producer are not necessarily the same version at an instant. Publication is an explicit cross-activity event with a result and freshness meaning, not an implicit consequence of an optimizer step. A replica that merely executes a published view of the accepted policy is not a new participant solely because it is another process or holds an older copy. An intentionally distinct reference/teacher model, if one is selected, remains a distinct participant. This follows the existing participant-versus-route distinction; it does not decide that every versioned copy has identical operational state or can be substituted without a check.

At the producer-to-consumer boundary, the accepted run must retain **work identity and provenance**: which prompt/group or task produced the item, which policy revision(s) generated its relevant actions/log probabilities, whether the item/group is complete, and any other dependencies the selected admission rule needs. One trajectory may contain spans from different policy versions, so a single `generated_at_step` field is not always sufficient. The accepted behavior, not Trainer's generic code, defines whether mixed-version or stale experience is usable, needs correction, must wait, or should be rejected. For a groupwise objective, admitting one trajectory independently may be invalid; physical microbatching cannot erase group membership.

This gives the whole-run graph two independently advancing regions with explicit communication, rather than one repeated `get_batch()` call that secretly owns the producer's lifetime:

| Graph-visible boundary | Meaning the engine must preserve | What may remain selected capability behavior |
| --- | --- | --- |
| Producer lifecycle | Start/stop, policy-snapshot dependency, failure, capacity/backpressure, and whether unfinished work can be cancelled or resumed. | Agent/tool loop, worker allocation, load balancing, internal queue and token generation. |
| Policy publication | Which trained policy state was successfully made available to which producer view, with its version and readiness. | Transfer protocol, replica loading, and backend synchronization. |
| Experience handoff | Work/group identity, completion, provenance, ownership transfer or retention, and admission outcome. | Payload format, reward/scoring mathematics, staleness/correction policy implementation. |
| Training action | Accepted experience and selected objective feed Trainer-owned optimization, whose effective update and progress are separate from producer progress. | PPO/GRPO-specific loss construction and trajectory representation. |

This does not require a universal `Trajectory`, `ReplayBuffer`, or queue class. It does require that the accepted run expose the synchronization and handoff facts on which another owner, restoration, or Trainer coordination depends. A producer can be an operation/region with internal workers; the graph need not include one node per request or token. The future data-streaming and async-cache cases share independent production and readiness, but they do **not** automatically inherit policy-version or groupwise-admission rules.

### Recovery test: one snapshot with work in three states

Suppose the Trainer has committed policy state P18, a rollout replica has only successfully published P17, one group is finished, one is running, and one prompt is pending. A same-run snapshot cannot claim coherent continuation merely by saving P18 and the optimizer. It needs an accepted account of the producer's published version, input/work allocation and cursor, finished-but-unconsumed groups, and what happens to the running and pending work. In veRL's documented policy, finished groups return as stored experience while pending/running prompts are reissued after restoration; their old partial execution is **not** claimed to resume from the same point. Another accepted producer could persist a restorable partial trajectory instead, but only if its environment/tool state and external effects actually support that promise.

After restoration, admission must still evaluate restored or regenerated experience against the restored policy state and accepted rule. Reissuing a prompt must not silently count it as both consumed and new, and a completed group must not be trained twice because queue removal and optimizer progress were saved at different cuts. A policy-publication failure likewise does not undo an already completed Trainer optimizer update: it leaves producer freshness/availability behind and invokes the accepted wait, retry, or failure policy. No generic transaction spans the optimizer, remote replicas, external tools, and durable queue.

The strongest honest restoration claim is therefore **run- and producer-specific**. A snapshot may support same-run continuation from a coherent cut while regenerating disposable in-flight experience; that is not bit-identical replay of the interrupted world. Exact continuation of an external interaction requires more evidence and possibly a stronger capability. The snapshot/result must say which guarantee it provides, rather than labeling every saved policy plus queue as “exact.”

### Minimum additional graph meanings from this pass

- **Independent progress:** producer and optimizer activity have distinct lifetimes and coordinates; their scheduling can overlap without pretending to be one atomic step.
- **Versioned publication and provenance:** the producer's available policy view and the Trainer's current policy state are distinct; generated work carries the dependency facts needed for admission, including mixed-version spans where applicable.
- **Handoff/admission:** ready and admissible are different. Completion, group membership, selection, rejection/refill, and effective consumption have distinct outcomes.
- **Capacity and lifecycle control:** backpressure, cancellation, producer failure, and shutdown are bounded interactions visible where the engine must coordinate them, while internal worker mechanics remain selected behavior.
- **Declared recovery coverage:** finished, running, pending, and consumed work require an explicit save/reissue/drop/resume policy consistent with the saved optimizer and input state. An external side effect cannot be generically replayed or rolled back.

The case establishes that independently progressing regions and explicit communication are part of the intended run-language problem if asynchronous experience is to be supported. It does **not** settle whether they are primitive graph objects or assembled from a smaller Python-native region/handoff mechanism. That choice must retain the version, ownership, and recovery checks above rather than burying them in an opaque producer callback.

## Fifth concrete pass: one training meaning, different physical realization

The [distributed-execution](distributed-training-execution.md) and [precision/quantization](precision-and-quantization-training.md) notes test whether the graph confuses a logical participant, trainable subject, or optimizer unit with whichever tensors a backend currently exposes. [FSDP-QLoRA is a documented combined case](https://huggingface.co/docs/bitsandbytes/fsdp_qlora): a quantized base executes with trainable LoRA weights, while FSDP shards the model. This gives a concrete comparison without claiming that every FSDP, quantizer, model family, and adapter implementation can be freely combined.

### Hold the accepted training meaning fixed

Assume the authored selection is: a quantized frozen base, an attached LoRA method at accepted targets, a loss that differentiates through the effective base route, and one Trainer-owned optimization unit selecting the LoRA parameters. That numerical choice is already part of accepted meaning; comparing it to an unquantized full-model fine-tune would **not** be merely another physical lowering.

The same accepted meaning can have at least these realizations, where supported:

| Accepted meaning | Single-device realization | FSDP-QLoRA realization |
| --- | --- | --- |
| Base participant and attachment | One local quantized execution route with the LoRA attached. | Sharded quantized storage, gathered as needed for execution, with the same accepted attachment relationship. |
| Forward/gradient path | Quantized base computation allows gradients to reach LoRA weights; base weights remain frozen. | Gather/communication and dequantization are part of the prepared route; the same accepted gradient path reaches LoRA weights. |
| Optimization unit | Selects only the authored LoRA parameter substructure. | Resolves that same semantic membership to current distributed parameters, gradients, and optimizer state; shards do not become extra units. |
| Result/product | LoRA update and its declared observations; a requested LoRA artifact retains its base dependency. | Same logical update/product meaning, though numerical reduction order and the mechanics of gathering/export may differ within the accepted numerical policy. |

The bitsandbytes guide makes a non-obvious compatibility issue concrete: its 4-bit values may use a floating-point **storage container** so FSDP can shard them, while the kernel dequantizes to a separately chosen compute dtype. Storage container dtype, quantization format, compute dtype, adapter dtype, and optimizer state therefore cannot be collapsed into `model.dtype`. Backend/materialization facts must establish that a selected quantized route, FSDP wrapping, attachment, and gradient path actually work together. A known incompatible combination should fail during contract-guided fulfillment; a hardware-, shape-, or realized-backend fact that is only available later must fail at its earliest governed evidence checkpoint, not after publishing an apparently prepared route.

~~~text
accepted semantic graph
  LoRA participant --attached to--> base participant
  selected quantized forward + differentiable path
  LoRA-only optimization unit
                 |
                 v
coherent preparation job for base, LoRA, optimizer and backend
                 |
                 v
prepared realization A: local quantized route + local optimizer
prepared realization B: sharded quantized route + distributed optimizer state
                 |
                 v
same accepted action/participants/optimization meaning
~~~

FSDP2's [documented shard/gather/reshard behavior](https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html) shows why the tensor visible during forward need not be the persistently resident parameter. That is useful physical-lowering evidence, but the linked FSDP-QLoRA guide does **not** establish that its example uses FSDP2 specifically. Backend wrappers, rank-local shards, temporary full tensors, communication groups, and composite handles are preparation/execution state; none changes the base or LoRA participant reference, nor the semantic optimization-unit identity. A backend may need to prepare frozen base and trainable LoRA jointly. [D8](../design.md) already requires their routes/views and Trainer-owned optimizer/backend state to become current as one coherent result, even if several backend calls are needed.

### Physical scheduling has a boundary

The same principle extends past FSDP. Pipeline parallelism can schedule microbatches through model stages; context parallelism adds activation communication; offload adds prefetch/eviction; precision kernels may maintain scaling state. These may expand one accepted operation or Trainer-owned advancement into many physical actions. They are valid lowerings only if they satisfy the accepted data, order, gradient, update, numerical, and state-effect obligations. In particular, a physical microbatch is not an independent logical training action, and a backend collective is not a new authored model operation just because the profiler can see it.

This cannot become a blanket claim that *all* quantization is backend-only. Frozen-base quantized execution is an authored, checkable numerical selection whose storage/sharding details can be realized by a backend. Quantization-aware training that explicitly quantizes/dequantizes on the differentiable path, or uses a selected estimator, changes the algorithmic operation and gradient meaning; that belongs in accepted behavior before lowering. FP8/FP4 scaling histories and FP16 loss-scaler state may be backend-/optimization-owned continuation state, but their updates and restoration coverage still must be declared. [Transformer Engine's FP8 documentation](https://docs.nvidia.com/deeplearning/transformer-engine/examples/fp8_primer.html) demonstrates both per-tensor scaling state and operation-specific precision, rather than one stateless dtype switch.

### Failure, replacement, and restoration check

- Preparation must check backend compatibility, complete rank agreement, current binding/relationship revisions, optimizer membership, and all required participant routes before publishing a new prepared view. An optimistic replacement failure leaves the old still-valid view; destructive in-place mutation withdraws affected guarantees first, as D8 requires. A successful wrapper change need not revise participant or optimization-unit identity, but it does change the current prepared realization and its freshness evidence.
- A rank or backend failure during a physical optimizer action is **not** a failed preparation candidate and has no generic rollback. The action result must not claim effective advancement just because a logical update was scheduled. Distributed participants must stop using an unconfirmed mixed state; recovery depends on a sufficiently coherent saved runtime state.
- A resumable snapshot identifies logical participant and optimizer state plus the backend-owned continuation contributors needed for this run. It may be saved in shards and restored into a different supported placement, but rank-local fragments and wrapper object identities are not the logical identity. A LoRA trained product is separate from that snapshot and need not duplicate an unchanged quantized base if its dependency is recorded and available.

### Minimum additional graph meanings from this pass

- **Logical-to-physical mapping:** accepted participants, semantic substructures, operations, units, and relationships map to current prepared routes, parameter representations, optimizer handles, communication groups, and physical schedules without redefining their identities.
- **Realization constraints and evidence:** backend support is checked against authored numerical/gradient/ordering obligations and realized facts. An available kernel or wrapper is not, by itself, proof of compatible training meaning.
- **Scoped numerical state:** storage, compute, backward, optimizer/master, scaling, and product representations may differ; stateful precision control has an owner and restoration contribution.
- **Lowering boundary:** partitioning, fusion, and movement may stay inside a backend region; authored QAT, changed objective, changed optimization membership, or changed data/admission policy cannot be silently introduced by that region.
- **Physical failure boundary:** coherent preparation publication is narrower than transactional execution. Backend failure after an optimizer mutation retains the same ambiguity identified in the adversarial case.

This case supports a semantic graph with one or more prepared executable revisions, not a requirement to model every collective or kernel as a public IR node. It also shows why the graph must expose enough constraints to **reject an invalid lowering** while letting selected backends own their implementation schedules. The particular Python representation and evidence interface remain to be designed.

## Next synthesis check

The [research index](README.md) has now been checked against the five passes at the **architecture-boundary** level. This is not a source-by-source proof that every recipe is implementable by a not-yet-written IR. The useful result is where the current sketch has a worked example and where a meaning has only been named in the survey above:

| Research pressure | What the five passes already exercise | Concrete language pressure to test |
| --- | --- | --- |
| [Autoencoders](autoencoders-vae.md), [pixel-space](pixel-space.md), [video/audio](video-audio-training.md), [Hackable Diffusion](hackable-diffusion.md), and [LLM training](llm-training.md) | Selected representation/objective work, values with domain meaning, stateful feedback, varied optimization actions; no universal VAE, image batch, timestep, or next-token loss. | Multiple invocations of **one participant** in different roles or gradient modes; structured multi-output/multi-objective results; intermediate state such as a KV cache with an explicit gradient boundary. A name/shape alone cannot establish compatible value meaning. |
| [Anima's bridge](anima-llm-adapter.md), [control side networks](conditioning-and-control-networks.md), and [distillation](teacher-student-distillation.md) | Different attachment, side-input, cache-cut, frozen-gradient, and changing-role relationships. | Exact role-scoped route/port requirements, including one physical base used with and without an adapter; do not force each execution role to become a separate participant. |
| [Autoencoder adversarial updates](autoencoders-vae.md), [LLM optimizer partitions](llm-training.md), and [precision state](precision-and-quantization-training.md) | Independent alternating units, action results, and physical optimizer representations. | Two or more units due in one logical action, their ordering/gradient/synchronization/zeroing meanings, and a strong imperative case that requests authority beyond the standard profile. This belongs with existing G2.5/G3 work, not a new mode axis. |
| [Training-data pipeline](training-data-pipeline.md), [Anima caching](anima-llm-adapter.md), and [preference/RL input](preference-and-rl-post-training.md) | Cache dependency cuts, independent producers, generated-experience identity/admission, and restoration coverage. | Stateful selection, packing, mixture/curriculum and distributed data-group meaning in a worked input region; neither a dataloader call nor an epoch is a universal input lifecycle. Logical groups must survive physical microbatching. |
| [Model surgery](model-surgery-and-staged-topology.md), [progressive distillation](teacher-student-distillation.md), and [async RL](preference-and-rl-post-training.md) | Governed graph/preparation revisions, intentional overlap, independently progressing activities, versioned handoffs and failure/restoration distinctions. | The concrete transition and communication interface, with freshness and owned-state dependency checks. No new semantic category is apparent from this sweep; the exact mechanism remains unbuilt. |
| [Distributed execution](distributed-training-execution.md) and [precision/quantization](precision-and-quantization-training.md) | One accepted QLoRA meaning with local versus sharded realization, plus the boundary where QAT is authored computation rather than physical lowering. | Check a real lowering's constraints, result, state contributors and failure paths without exposing every collective/kernel as an authoring construct. |

The sweep therefore **does not justify freezing the IR vocabulary yet**. It also did not reveal a need for a family-specific Trainer or to reopen contract-first authoring. The next useful work is to derive a small Python-native representation from the common meanings and pressure it with three deliberately different worked examples: (1) one participant invoked more than once with different roles/gradient boundaries, (2) one logical action coordinating multiple optimizer units, and (3) one stateful input region with packing or mixture policy. All three language tests are worked below. These are focused checks of existing G2.4–G2.6/G3 tasks, not a new list of architectural decisions for the user to settle in advance. Readability and hot-path cost should be assessed on the same examples before choosing graph objects, nested regions, or a hybrid storage form.

The settled contract-first direction and Trainer ownership are inputs to this work, not alternatives under review.

## Language test 1: one participant, several uses in one action

The [MiniMax-H3 teacher-matching case](teacher-student-distillation.md) and [DiffusionGemma Sudoku case](llm-training.md) make two different demands. In MiniMax, one frozen base executes as a privileged, no-gradient teacher **without** the trainable LoRA and as an adapted student **with** it. In DiffusionGemma, one Gemma backbone executes causally to produce logits and a KV cache, then executes as a bidirectional denoiser twice; the first denoiser prediction is detached before it can self-condition the second. The concrete Sudoku recipe leaves the denoiser-to-encoder gradient path through the KV cache open. These are source behaviors, not generic `teacher` or `denoiser` slots for Trainer.

The smallest readable sketch of the *authored behavior* is ordinary selected operations connected by values. The names and call syntax below are illustrative, not a proposed API:

~~~python
# MiniMax: both views refer to the same accepted base participant.
teacher_target = privileged_prediction(base_without_lora, teacher_inputs)  # no grad
student_output = t2va_prediction(base_with_lora, student_inputs)            # grad to LoRA
loss = teacher_match(student_output, teacher_target)

# DiffusionGemma: one backbone, different uses and an intermediate value.
encoder_logits, kv = causal_prefill(gemma, clean_inputs)
draft = bidirectional_denoise(gemma, corrupted_inputs, kv)
condition = choose_for_self_condition(detach(draft), zeros)
final = bidirectional_denoise(gemma, corrupted_inputs, kv, condition)
loss = encoder_loss(encoder_logits) + diffusion_loss(final)
~~~

This author-facing form is useful only if fulfillment can preserve the important facts it implies without making Trainer inspect those Python function bodies. In MiniMax, `base_without_lora` and `base_with_lora` are **two accepted execution views of one base participant**, not separate identities or a relationship detached and reattached on every batch. The LoRA relationship remains accepted; selecting a base-only view for one call is a bounded use of it. The no-gradient teacher call cannot be inferred from the base being frozen, because the adapted student call still needs a differentiable path to LoRA parameters. Only the declared LoRA substructure belongs to the optimizer unit.

In DiffusionGemma, the causal and denoising uses have different attention/input contracts even though they refer to the same backbone. The KV cache is a typed intermediate value with a lifetime and a gradient policy, not automatically a new participant or a durable dataset cache. The first denoiser output is **computed but detached** before it is optionally selected as conditioning; the second call happens either way. The encoder logits have their own objective, while the denoiser output has another. A one-output/one-forward/one-loss operation interface would erase the method's meaning. The specific `[tokens, KV cache, logits]` payloads remain selected behavior rather than universal Trainer fields.

What the accepted graph must expose across each call boundary is therefore: the **same participant reference**, selected role-specific operation, required prepared view, input and output meanings, ordering/value dependencies, gradient cut or continuation where the next owner relies on it, and any state effects. It need not expose every attention kernel or each token. The selected implementation can hold the detailed math. A role name alone does not prove a valid view; fulfillment checks the authored combination and preparation checks that the current component/backend can provide the required views and gradient path.

There is one practical execution constraint hidden by clean pseudocode: if an implementation realizes `base_without_lora` by temporarily toggling mutable adapter state on the same module, two calls cannot overlap arbitrarily and failure must not leave the next call in the wrong mode. An implementation may instead provide independent immutable views or another safe mechanism. The accepted calls and their ordering/cleanup requirements are stable; their physical realization is backend-dependent. Per-call view selection does **not** by itself revise participant bindings or force full runtime preparation every step.

This test adds an **invocation/use** meaning distinct from participant identity and from executable implementation identity. One participant can be used repeatedly, in multiple roles, with different inputs and gradient policies in one action. That is compatible with the existing hierarchical graph frame, but the eventual Python design must make the calls readable and their cross-call constraints checkable without installing a universal model-role taxonomy.

## Language test 2: multiple optimization units due in one action

The [Qwen3.8-Flash-Next report](https://arxiv.org/html/2608.30320#S3.SS1) applies Muon to suitable linear-map matrices, AdamW to embeddings, output head, router and some other projections, and Adam without weight decay to its n-gram embedding table. Its Muon implementation also splits fused gradients into meaningful submatrices before orthogonalization and may rearrange their physical ownership for distributed execution. The [DeepSeek-V4 report](https://arxiv.org/html/2606.19348#S4.SS2) likewise uses Muon for most parameters and AdamW for selected modules. These papers establish **different optimizer policies for disjoint parts of one training recipe**; they do not define our semantic optimization units or, by themselves, specify the exact ordering or failure transaction of one of our actions.

To test the language, take an accepted recipe with one selected objective and two semantic optimization responsibilities: `matrix_unit` owns suitable linear-map substructure under a Muon policy; `other_unit` owns disjoint embeddings/router/head substructure under an AdamW policy. Both are due at the same accepted action boundary. This is our proposed *interpretation for the test*, not a claim about how either paper represents its optimizers. Qwen's separately described n-gram Adam policy could add another responsibility without changing the mechanism.

| Part of the accepted action | Meaning the language must retain |
| --- | --- |
| Authored computation | The selected implementation computes the model/objective result. One objective can feed both units; two losses do not automatically imply two units. |
| Semantic optimization plan | Each unit has its own address, disjoint participant-substructure membership, accepted policy, state, and progress. Both happen to be due now; neither is collapsed into the other. |
| Trainer-owned action | Trainer coordinates required backward/accumulation/synchronization, then clipping, advancement, scheduler transitions, and zeroing according to the accepted policies and backend constraints. A single backward may be enough for this example; the language must not require that for every case. |
| Runtime realization | Backend-specific optimizer objects, fused kernels, shards, and Qwen-style submatrix handling realize the plan. Their number and layout do not define how many semantic units exist. |
| Result and continuation | Record what actually happened for each unit, including skipped, completed, failed, or uncertain advancement. Preserve the appropriate state and coordinates for exact same-run restoration. |

Here **one action** means one accepted coordination boundary where both units are due, not an all-or-nothing optimizer transaction. A preparation attempt publishes both current unit runtimes coherently before execution, as D9 requires; this does **not** make two later in-place parameter updates atomic. If the first unit advances and the second fails, reporting only “the action failed” loses the fact that parameters may already have changed. The run needs an honest per-unit outcome, safe failure handling, and no blind replay. If a recipe truly requires all-or-nothing advancement, it needs an explicitly supported transactional mechanism or must be rejected; mere placement in one region cannot promise it.

This test also sharpens the separation between **logical groups**, **semantic units**, and **physical optimizers**. A unit is a separately managed advancement responsibility, even if its cadence is coordinated with another unit. Different optimizer methods are a good reason to test two units here, but the number of optimizer objects alone is not proof of that count. The standard profile still checks non-overlap, including tied aliases. Clipping scope, gradient reuse, ordering, scheduler advancement, and zeroing must come from accepted policy and backend support; the language should not silently infer them from table position or a fixed global `step()` sequence. Alternating adversarial updates from the earlier autoencoder pass use the same unit meanings with a different due schedule, rather than a separate kind of Trainer.

The language consequence is modest but important: an action can name a **set of due units** and their optimization input(s), while each unit retains its own identity, policy, state, and result. It must allow the accepted coordination policy to say how those units share or separate gradient work. Whether the readable Python surface expresses that as a nested region, explicit action description, or another form remains a G3 design task; this test does not select an API.

## Language test 3: stateful input selection and packing

The [Megatron Core dataset documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/core/datasets.html) gives a concrete mixture case: a blend of datasets and weights is realized as indices that choose a contributing dataset and sample for each logical position. Its [data-loading guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/data-loading.html) describes deterministic per-data-parallel-rank sample assignment and separately prepared index caches. [Megatron Energon](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/megatron_energon.html) shows a different, streaming shape with blending, packing, prefetching, and restorable data-loading position. The [Transformers padding-free guide](https://huggingface.co/docs/transformers/main/padding_free) demonstrates why packing is not just concatenating tensors: the consumer needs sequence-boundary information, and some model implementations require explicit boundary inputs rather than inferring them from positions.

For this language test, combine those pressures into an *illustrative accepted input pipeline*, not a claim that one cited system implements this exact combination:

~~~text
accepted source/mixture policy + current input state
                    |
             select logical examples
                    |
      selected transformations and packing
                    |
     model-ready values + example boundaries
                    |
          physical rank/microbatch delivery
                    |
             accepted computation
~~~

The accepted meaning must say which sources and selection/packing policies are permitted, what representation the consumer expects, and which boundaries or masks make a packed value valid for the selected operation. It need not expose every file read, worker, queue entry, or tokenizer call as a public graph node. Likewise, the authored strategy need not personally implement all data handling: the accepted arrangement can select maintained data behavior, while training input coordination owns readiness, traversal, and delivery. Selected input implementations may own private buffers but must contribute the continuation state required by the accepted restoration claim. Trainer receives the resulting input and relevant coordinates; it does not decode dataset internals or guess model-specific packing rules.

This region has state even when the model computation is stateless. For an indexed mixture, selected source/sample indices plus a position may reconstruct the next input; for a streaming pipeline, continuation may also require stream cursors, shuffle/packer buffers, RNG and worker state. Prepared index caches and cached model representations are different artifacts with different producer dependencies. Exact same-run restoration must recover the *next logical input and its meaning*, not just a global optimizer step. Prefetch complicates that claim: fetching, handing an input to computation, and accepting the action's result are distinct events. The chosen pipeline must define what is replayed or skipped after failure and preserve enough state to honor that policy; it should not pretend every `next()` call is already a completed training action.

The language also must not collapse a dataset record, logical example, packed model sequence, and physical microbatch into one universal `batch`. One packed sequence may contain several independent examples; a backend may split a logical training group into several microbatches. Source identity, example boundaries, valid target positions, and accounting units must survive whatever physical layout is chosen when the selected computation or restoration policy relies on them. The concrete payload can remain implementation-specific.

An accepted mixture or packing schedule can change with progress as ordinary stateful behavior **within its accepted bounds**; that need not re-run strategy fulfillment for each input. A new source, incompatible representation, or policy change outside those bounds requires the applicable governed transition or new fulfillment. A coupled curriculum change that also changes model topology or optimization meaning follows those concerns' existing transition rules rather than being smuggled in as a dataloader setting. Neither an epoch nor `global_step` is therefore a universal data cursor.

This test adds an **input-production region with owned continuation state** to the language: it connects accepted source/policy meaning to model-ready values, preserves logical example boundaries across physical batching, and contributes to exact restoration. It does not require a universal dataset graph or settle the concrete Python API. Together with the first two tests, it shows why a single fixed `for batch in loader: loss = model(batch); optimizer.step()` shape cannot be the framework's semantic core, even though a maintained simple path may lower to one.

## Still to specify at the design level

- The minimal graph constructs and which relationships are checked during strategy fulfillment versus materialization/preparation.
- How schedules and concurrent activities are expressed, including multiple progress coordinates and bounded runtime choices.
- How graph revisions and operation-owned state respond to changed participants, routes, relationships, and observations.
- The exact standard optimization input/result connection and the explicit authority surface of a research region.
- The concrete input-production exchange, including ownership of selection/packing state and replay policy at action failure.
- The exact snapshot/restore coverage for concurrent producers, transitions, and backend-owned state.
