# Trying to break the trainer's lowest sensible boundary

Architectural challenge — 1 October 2026

## Finding

The proposed boundary is a defensible starting point, but it is not established as the unique lowest sensible architecture. Two credible alternatives remain:

1. **Own more:** a bounded training-execution planner that can schedule differentiated regions, activation retention/recomputation, transfers, and eligible updates together, while borrowing existing autodiff, tensor compilers, allocators, and kernels.
2. **Own less:** compile accepted semantic regions directly into framework-specific executable functions or existing runtime deployments. Retain a small run supervisor and contracts, without requiring a second universal executable IR or a project-owned VM.

The report's strongest conclusion survives: there is no demonstrated reason to recreate a universal tensor framework, derivative-rule library, allocator, kernel compiler, or transport. Its weaker inference does not: delegating those mechanisms does not require delegating every numerical scheduling decision, and preserving run authority does not require interpreting a separate control plan at runtime.

The smallest defensible distinct runtime layer is a **supervisor for revision-bound regions with explicit effects and completion**. It exists to enforce obligations between independently progressing owners. Wiring ordinary function calls is insufficient justification by itself.

## Evidence and limits

This review treats `trainer-architecture-research(1).md` as a hypothesis, and grounds project requirements in the supplied `design(4).md` and `training-mechanism-sketch(4).md`. The GitHub branch still resolves to **e0a6f9bb2f7299bf576dfdd1cf5a1c6e8c9ac627**, the same commit inspected by the report. Relevant source was retrieved through GitHub, including `library/training/execution.py`, `library/performance/custom_offloading_utils.py`, `library/optimization/wrappers/cpu_offload.py`, and the asynchronous text-encoder proposal. The candidate executor explicitly says it is not used by the active Trainer.

This is architectural research, not a new performance experiment. No model training, GPU, distributed, native VM, or memory-planner benchmark was run. Published systems establish that mechanisms are credible; they do not establish their compatibility or speedup on this repository. Source links below are primary documentation, project source, or papers. Proposed combinations and ownership choices are deductions, and are identified as such.

The report already allows transparent regions, custom kernels, compiled actions, imported tensor IR, and existing actor runtimes. Merely naming those again would not be a counterexample. The challenges below identify decisions or representation costs that its conclusion still leaves unresolved. “Own less” below reduces representation and runtime implementation ownership; it does not discard any accepted semantic responsibility. “Own more” means owning a numerical planning policy and its supported execution protocol, rather than merely attaching an existing compiler to an opaque function.

## 1. The most important distinction the report underuses

Three kinds of ownership should be evaluated separately:

| Ownership | Example | Does the trainer need to implement the underlying numerical machinery? |
| --- | --- | --- |
| Semantic authority | Which loss contributes to which unit; whether a producer may use older weights | No |
| Planning authority | Which valid backward schedule, memory budget, checkpoint cut, or transfer order to select | No, if existing mechanisms expose sufficient contracts |
| Mechanism ownership | Derivative formulas, tensor allocation, compiler code generation, collectives | Usually no |

The report is right about semantic authority and mechanism delegation. It is less decisive about planning authority. A framework can know every tensor operator while lacking knowledge about an independently running text encoder, a teacher/student view pair, a pending accumulation window, or a run transition. Conversely, the trainer may know those constraints without knowing the model's arithmetic.

That gap admits a useful middle architecture. The trainer can combine framework-provided numerical summaries with its run semantics, choose a physical plan, and invoke framework mechanisms to realize it. This requires a bounded, optional numerical planning surface; it does not require a public node for every tensor operation.

## 2. Counterexample for owning more: jointly plan memory, recomputation, and numerical execution

### A project-shaped pressure case

Consider a memory-constrained run containing a no-gradient teacher view, a differentiable student view, a live conditioning producer, and CPU optimizer offload. Each implementation could individually fit within its locally chosen budget while their overlapping peaks exceed device capacity. Fixed partitioning of one global budget prevents that problem, but assigning permanent worst-case allowances may waste memory or serialize useful work.

The trainer knows when teacher work is due, whether its output is detached and releasable, how many conditioning results may be admitted, which student continuations remain live, and when an optimizer window closes. A compiler confined to one student invocation generally does not know this whole-run context. The outer controller, if it sees only an opaque callable and a scalar peak-memory estimate, cannot choose internal save/recompute cuts or finely coordinated prefetch.

For a single closed numerical action, the existing backend may solve the problem well. The counterexample is a shared budget across independently planned regions, where their decisions interact.

### Concrete prior art

[Rockmate](https://proceedings.mlr.press/v202/zhao23b.html) constructs training rematerialization plans for PyTorch models under memory constraints. [HiRemate](https://proceedings.mlr.press/v267/gusak25a.html) uses hierarchical graph partitioning and merges regional solutions into a training schedule while integrating with PyTorch Autograd. These are examples of substantial numerical planning above an existing differentiation engine.

The [Rockmate repository](https://github.com/topal-team/rockmate) also describes Offmate, which includes offloading and handles optimizer advancement. The distinction matters: coordinating movement, backward, and updates can require a broader execution interval than a forward-only model wrapper. Its documented API does not establish support for this project's independent units or windows.

The [NeurIPS rematerialization/offloading work](https://proceedings.neurips.cc/paper/2021/hash/c8461bf13fca8a2b9912ab2eb1668e4b-Abstract.html) treats those two memory techniques together rather than as independent toggles. [torch_remat](https://github.com/meta-pytorch/remat) supplies region-level save/recompute annotations and explicit replay-state hooks; it is a possible mechanism beneath a planner, rather than a whole-run planner itself.

### A bounded architecture that owns more

An optional prepared numerical region exposes:

- Selected block or region dependencies, including required backward continuations.
- Candidate implementations: retain, recompute, offload, or a supported compiled variant.
- Profiled or conservative memory and time summaries with shape/precision guards.
- Replay requirements for parameters, buffers, RNG, and owned state.
- Safe prefetch, release, and update points supplied by the numerical backend.

The project plans across these summaries using a shared resource budget. Existing framework code executes forward/backward, checkpointing, transfers, and numerical updates. The planner may select among backend-generated candidates instead of deriving every operator-level schedule.

The first useful implementation could allocate budgets to a few coarse regions and request alternate rematerialization plans. It need not start with a general solver or inspect every ATen operation. Opaque implementations remain supported with conservative reservations and less overlap.

### The correctness obligation

Replay must reproduce the accepted computation against the relevant parameter version. Recomputing with newer weights is not an equivalent lowering. Replaying an adaptive observation update or metric publication twice is also wrong. Global planning therefore needs replayable-state and effect information, not only graph connectivity.

The repository's offloading utilities contain explicit stream synchronization and `.data` rebinding. This is concrete evidence that movement and lifetime behavior already require care here, not evidence that those utilities are active in every training path or that a new planner would outperform them.

### Verdict

This is a credible reason to own substantially more than generic executable control. It extends into numerical execution scheduling while preserving delegated numerical mechanisms. The report can accommodate it under “backend-private work,” but should then stop presenting generic physical scheduling as presumptively outside project ownership.

Do not adopt it unconditionally. It wins only if whole-run budget coordination or a demonstrated model schedule improves feasible batch/resolution, throughput, or memory use over composed framework solutions. An existing maintained planner may be cheaper to integrate than to replace.

## 3. Counterexample for owning more: make differentiated-region boundaries executable

The proposed gradient contracts are necessary, but a declaration such as “gradient-bearing output” is not always enough to schedule that output's lifetime or multiple routed losses efficiently.

### A concrete routing failure

Let units U1 and U2 own parameters `a` and `b`, respectively, with:

- `L1 = a*b`, assigned only to U1.
- `L2 = a*a*b`, assigned only to U2.

At `a=2, b=3`, the authorized gradients are `(dL1/da, dL2/db) = (3, 4)`. A backward of `L1+L2` produces `(15, 6)`, which changes the accepted optimization meaning. The report recognizes this problem, but leaves its executable representation largely inside backend capability contracts.

One implementation is an explicit derivative request over the loss outputs: seed L1 and select U1's parameters; seed L2 and select U2's parameters. [PyTorch's `torch.func.vjp`](https://docs.pytorch.org/docs/2.14/generated/torch.func.vjp.html) already returns outputs and a pullback callable parameterized by output cotangents. Thus derivative requests can be represented and scheduled without writing derivative formulas.

### What a richer optional boundary could expose

Instead of only `call(inputs) -> outputs`, a supported differentiated region could expose:

1. Forward outputs and an opaque continuation or pullback handle.
2. The parameter/state versions and storage leases that continuation requires.
3. Supported derivative requests and their output-to-subject routing.
4. Whether requests can share retained work, must recompute, or support batching.
5. When the continuation may be released and which updates must wait.

This does not promise that batching VJPs will be faster. It gives the planner information needed to compare legal alternatives. A normal framework tape remains the simplest realization when all work lives in one compatible eager/compiled domain.

If the project itself composes pullbacks across region boundaries, that is genuinely a small differentiation layer. It should be described honestly and restricted to a declared profile, such as first-order reverse mode with explicit structured ports. Higher-order derivatives, mutation, aliases, distributed placement, and arbitrary custom functions are not automatically supported. Prefer letting the framework compose derivatives where possible.

### A performance counterexample to opaque backward

[Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241) splits backward work into input-gradient and parameter-gradient computations so scheduling can fill otherwise idle pipeline intervals. Current [PyTorch pipelining](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html) documents schedules using this split and extension points for custom schedules.

The relevant counterexample is the split itself: a coordinator restricted to one indivisible backward call cannot exploit it. The project need not reimplement the published schedule. It could request and compose supported split regions from a backend, and own only custom scheduling where its units, windows, or shared views need behavior the backend does not already offer.

Framework-provided forward/backward graphs are another route. [PyTorch's custom-backend interface](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_custom_backends.html) allows compilation after AOTAutograd generates backward graphs. A project can own selected partitioning or planning passes over imported numerical IR while borrowing differentiation and downstream compilation. Those integration interfaces include internal APIs and require a version support policy.

### Verdict

Delegate autodiff mechanics, but do not equate that with leaving derivative routing, residual lifetimes, and every backward scheduling choice opaque. There is a credible deeper architecture here. Whether it is worth implementing depends on a real scheduling or memory benefit; the existing gradient correctness requirement alone can often be satisfied with framework calls and a simpler controller.

## 4. Counterexample for owning more: optimizer timing is not just a final call

[PyTorch's optimizer-in-backward tutorial](https://docs.pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html) shows how post-accumulation hooks can advance parameters and release gradients during backward. This is a concrete memory opportunity from crossing the conventional backward/step boundary. Its tutorial explicitly excludes ordinary gradient accumulation from the simple recipe.

For this trainer, the eligibility conditions are stricter than “a gradient hook fired”:

- The unit's complete accepted contribution window has finished.
- Required synchronization, unscaling, and clipping are satisfied.
- Every computation, retained derivative, and recomputation needing the old parameter version has finished or uses a protected snapshot.
- The optimizer rule and accepted failure/observation policy permit this timing.

Global-norm clipping, shared parameter uses, multiple losses, and outstanding continuations can rule out early updates. Parameter-local optimizers and a supported no-accumulation profile can admit them. The accepted contract must offer that profile; it cannot silently change G3.1's clipping or phase ownership.

This is more than a custom fast kernel. It is a plan that orders numerical computation and mutation using gradient readiness and last-use information. The upper graph can retain unit identity and outcomes while a lower schedule realizes the accepted protocol.

**Verdict:** do not make optimizer-in-backward universal, but do not freeze a boundary that makes it impossible. It justifies optional update-readiness and last-use information in the differentiated execution profile. Implement the hooks or update kernels through existing framework/optimizer machinery.

## 5. Counterexample for owning less: compile the graph away

### A concrete architecture

For a supported synchronous region:

1. Validate its semantic graph, routes, unit definitions, gradients, effects, and numerical policy.
2. Generate a framework-specific function containing the selected computation and retained Trainer mechanics.
3. Insert guards and observation/failure boundaries required by the accepted contract.
4. Attach provenance and specialization metadata.
5. Invoke it from a small supervisor responsible for input admission, revisions, transitions, and unresolved outcomes.

No persistent, separately serialized, backend-neutral executable-control representation is required. No register interpreter, instruction decoder, or independent value-slot runtime is required. The semantic graph stays authoritative, and the generated function is its prepared realization. A transient lowering representation is an implementation detail.

[PyTorch's compiler guidance](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.where_to_apply_compile.html) explicitly includes applying compilation to a training step with its optimizer. That does not guarantee one fused graph, portable arbitrary optimizers, or full compatibility with distributed wrappers. It does establish that a numerical action can include more than a model forward. Generated ordinary Python remains a valid fallback where capture is unsuitable.

[JAX `lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html) supplies compiled, state-carrying repeated execution with fixed carry structure, shape, and dtype. For a supported functional training region, this can delegate its inner loop as well. A scan chunk must end before an externally required observation, input-policy decision, or transition; chunking changes the available supervision points unless those obligations are encoded inside it.

### Why this is a real simplification

The report's architecture B already permits generated code. The unresolved stronger claim is whether a distinct universal executable-plan schema and runtime must still exist alongside it. Direct lowering can avoid duplicating branch semantics, result layout, dispatch, and callable wiring across two representations.

The branch already provides a narrow hint: its 16-operation cheap fixture measures 14.41 microseconds for prepared interpretation versus 3.57 for equivalent specialized direct Python. These are the existing G3.7 measurements, not new measurements, and do not cover whole-run concurrency or numerical training. They justify taking direct lowering seriously, not choosing a native VM.

The less-owned implementation must preserve the same obligations. One generic compiler may generate different accepted programs; this does not require creating family-specific Trainers or reinstating per-step strategy authoring callbacks.

### What cannot be compiled away indiscriminately

When two input sources progress independently, the supervisor still must join their actual work identities and dependency versions. When a destructive revision is pending, it still must prevent unsafe access. When a fused numerical action fails after mutation, it still must report the supported known/uncertain outcome.

If a contract requires independently observable unit outcomes, compilation must preserve them or retain those boundaries. A broader conservative uncertainty classification is valid only when the accepted profile permits that loss of detail. It is not a free optimization of the existing contract.

**Verdict:** an executable mapping is necessary; a separate universal executable-control IR and VM are not. The report establishes the former, but not the latter. This alternative should be a first-class baseline for G5.

## 6. A stronger correctness option: functional candidate state and publication

The report is correct that arbitrary in-place optimizer actions do not have generic rollback. However, that does not exhaust the achievable execution guarantees.

For a restricted functional region, computation can produce a candidate next state from an unchanged old state. Keep old buffers valid, prohibit undeclared external effects, and make RNG/buffer/optimizer-state evolution explicit. Once the candidate's required completion is established, the authority can publish the new state bundle. A failed candidate need not have corrupted the previous committed numerical state.

This is a proposed protocol, not a built-in JAX transaction guarantee. [JAX buffer-donation documentation](https://docs.jax.dev/en/latest/buffer_donation.html) makes the crucial limitation explicit: donated inputs may be overwritten and become invalid. Donation therefore conflicts with keeping an old version as a recoverable candidate fallback. [Asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html) likewise means returning a result handle is not sufficient evidence that the candidate computation completed.

The guarantee is narrow: coherent in-process visibility of a selected state bundle. It does not include disk durability, process/device failure, distributed commit, or data cursor restoration unless separate protocols supply them. An input may already have been reserved or consumed; candidate numerical failure alone does not authorize automatic retry of the whole action.

This may be practical for a small adapter, optimizer subset, or temporary parameter view. Holding old and candidate full-model state can be prohibitively expensive. The design's ownership profiles and single-writer authority can accommodate it without promising transactions universally.

For a SAM-like method, a functional perturbed view can sometimes replace physical perturb-and-restore. But [PyTorch `functional_call`](https://docs.pytorch.org/docs/2.14/generated/torch.func.functional_call.html) does not guarantee purity: in-place parameter or buffer changes can affect supplied state. Buffer/RNG ownership, tied weights, adapters, and backend compatibility still need explicit validation.

**Verdict:** the report's conservative default survives. Its “no generic transaction” observation should not discourage stronger opt-in guarantees made possible by different numerical state representations. That is an architectural capability, not merely a stylistic rewrite.

## 7. Native coordination: the condition is broader than microseconds

The report correctly warns against a native interpreter whose every instruction calls Python. But raw dispatch fraction is an incomplete admission test for native coordination.

### A concrete workload

While the main device trains, conditioning work completes out of order, cached results occupy bounded memory, CPU optimizer work consumes host resources, and revision requests require quiescence. The important outcomes include device idle gaps, cancellation response, queue residence time, and memory-credit return latency.

A Python coordinator may handle this well. If it cannot, a native coordinator can help even when bookkeeping CPU time is a small fraction of training time: delayed readiness or credit release can create a much larger device stall. This is a scheduling effect, not a violation of Amdahl's law; the measured stalled portion must be included in the relevant cost model.

The countercondition remains important: a Rust queue does not make a long-running synchronous Python capsule interruptible. Worker/process isolation or cooperative capsule boundaries may be required, regardless of coordinator language.

### Native does not mean project-owned VM

An existing actor runtime can own messaging, process supervision, and generic futures, leaving the project to own admission, revision protocols, and outcome correlation. [Monarch](https://github.com/meta-pytorch/monarch/blob/main/README.md) provides actor meshes, supervision, and tensor-related distributed facilities. These do not automatically supply optimizer rollback or this trainer's contract semantics.

For a transformable subset, [IREE's runtime](https://iree.dev/reference/bindings/c-api/) already provides a host VM with generic control flow, calls, and device command submission. Its existence refutes the need to invent a VM merely to obtain native execution. It does not establish that this project's models, optimizers, Python extensions, or run-level protocols can lower to it economically.

[CUDA conditional graph nodes](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html) can execute device-side conditional bodies and loops. This is a narrower alternative when a numerical decision otherwise requires a host round trip. Its conditional bodies have device and node-type restrictions; it cannot replace a whole-run supervisor. Treat it as a physical lowering capability, not a CUDA-shaped public run language.

### Revised decision rule

Choose native coordination only when a supported workload demonstrates at least one of:

- Material throughput or tail-latency improvement through readiness/resource scheduling.
- A concrete deployment or controller-isolation requirement.
- Lower total implementation/operational complexity by reusing an existing runtime.
- A bounded native path with enough real work between foreign-language callbacks.

Measure actual capsule crossings, device idle time, queue memory, quiescence latency, and startup/packaging burden. Compare against specialized Python and worker isolation. A VM and an event coordinator are different deliverables: needing efficient queues does not imply needing a new instruction set.

**Verdict:** conditional native coordination survives. The report should broaden its conditions, and separate native execution from custom runtime ownership.

## 8. What remains delegated, and what can legitimately become ours

| Responsibility | Sharpened ownership decision |
| --- | --- |
| Accepted graph, participant/unit identity, phase authority, admission, revisions, restoration meaning | Project-owned; essential semantics |
| Prepared wiring and structured control | Project compiles it; may become direct code instead of a distinct runtime IR |
| Cross-region effects, completion, protected state versions, input correlation | Project-owned protocol; executor/runtime realization may be borrowed |
| Loss-to-unit derivative requests and accumulation meaning | Project-owned; framework computes derivatives |
| Numerical residual and replay contracts | Expose when scheduling needs them; opaque fallback remains valid |
| Rematerialization/offload/update schedules | Optional project-owned planner or imported planner; justify with a measured use case |
| General autodiff engine and derivative-rule coverage | Delegate; no demonstrated case for recreating them |
| Selected numerical/compiler transforms | Legitimate project extensions over existing IR; support and semantic validation required |
| Generic tensor compiler and kernel ecosystem | Delegate; selected custom kernels remain possible |
| Framework tensor allocation/coherence | Delegate; project owns outer admission budgets and leases rather than a competing allocator |
| Standard sharding, collectives, pipeline schedules, transport | Reuse; custom schedules can be owned without custom transport |
| Generic actor/event/VM machinery | Reuse where practical; a small custom implementation requires its own justification |

There is also useful prior art for stronger distributed contracts without permanent runtime overhead. [TorchTitan's SPMD type work](https://docs.pytorch.org/devlogs/distributed/2026-08-26-spmd-types-in-torchtitan/) pairs forward/backward distribution behavior and validates explicit collectives, with typechecking disabled in production. It illustrates checking more during preparation while owning less runtime machinery. It does not certify arbitrary dynamic collective order or failure recovery.

## 9. The smallest executable-control abstraction worth keeping

### Its purpose

Coordinate **externally relevant boundaries**: a boundary is relevant when another owner needs to wait, exclude mutation, check a version, consume a result, observe an outcome, or form a restoration cut.

Everything inside a boundary that has no such independent obligation may lower to one callable, compiled region, backend schedule, or existing-runtime task. The semantic graph can remain detailed for authoring and inspection without making every semantic node a runtime dispatch.

### Prepared region contract

The minimal common information is:

| Information | Why it is irreducible |
| --- | --- |
| Region identity and prepared generation | Correlate work and reject invalid prepared artifacts |
| Input/output boundary and provenance requirements | Join actual inputs and preserve semantic value meaning |
| Access to owned state, including required revisions/versions | Prevent stale use and conflicting mutation |
| Dependency and effect interval | Determine when execution is legal and when exclusion ends |
| Supported completion milestones | Distinguish host return, submitted work, value readiness, and settled mutation |
| Outcome knowledge and uncertainty scope | Stop unsafe dependent work without inventing rollback |
| Lifecycle/restoration contributor reference, when applicable | Quiesce and restore with the actual state owner |

These are semantic fields, not a mandatory seven-method ABI on every function. A pure synchronous leaf may use a much smaller interface. Asynchronous and mutating regions need the corresponding protocol traits. The same worker can carry framework tensor references directly; no serialization or FFI is required for ordinary local execution.

### Runtime protocol

1. **Admit:** select actual inputs; validate relevant prepared and producer dependencies; acquire the access/resource rights required by this attempt.
2. **Invoke or resume:** run a prepared region with scoped resources. Retain generation and storage/version protection through its declared lifetime.
3. **Await only relevant dependencies:** an incomplete input or numerical continuation must not block unrelated eligible work.
4. **Settle:** accept outputs and owner-supplied outcome facts, advance the appropriate progress, and release rights at the supported completion milestone. On failure, preserve known outcomes and gate affected uncertain state.

The essential guarantee is: **no dependent work receives authority to use a resource before the required version, input, ownership, and completion facts are established.** This holds only within the supported contracts and trusted implementation profile. Arbitrary Python/native code with unrestricted mutable handles can still violate its declaration.

### A race this layer must actually prevent

Checking revision `r7`, publishing `r8`, and then invoking a capsule against a destructively modified `r7` realization is a time-of-check/time-of-use failure. A freshness check alone is insufficient. Admission must participate in the same synchronization protocol as publication: protected snapshots, leases, or a quiescent exclusion interval.

Pinning immutable plan generation `g7` does not by itself pin the parameter values or protect an in-place-modified physical object. Plan generation, live weight version, and producer-product provenance are separate facts. Likewise, a forward read right may need to last until all dependent backward/recomputation work completes.

This is concrete work a distinct supervisor can justify. It is stronger than prebinding integer slots, and smaller than a general distributed VM.

The current candidate supplies a concrete scheduling limitation: `RunCoordinator.tick()` breaks out of its due-work loop when one attempt reports `waiting`. Later due work is therefore not examined in that tick, even if it could otherwise be ready. This is explicitly a test-only coordinator, not a production regression. It shows why pre-resolved graph wiring alone does not establish independent progress. A production supervisor should either demonstrate the required non-blocking scheduling or state which accepted profiles permit this ordering.

### What it should not require

- A universal global-step clock or one input/action/optimizer completion identity.
- A central scan of all dormant graph nodes on every event.
- A universal durable log or transaction around every operation.
- Numerical derivative formulas, tensor allocation, or a universal physical instruction set.
- A separate mutable model registry mirroring the framework's objects.
- Interpreting control whose obligations have already been discharged by a generated or backend-owned region.

For a closed synchronous run with no independent handoffs, the protocol can reduce to generated guards, function calls, and scoped outcome handling. It does not have to remain a separately instantiated runtime subsystem.

## 10. Four experiments that would decide the disputed boundaries

These are proposed discriminators, not completed tests. All variants must implement the same accepted observations, numerical/gradient policy, outcome detail, and failure boundaries. Compare one bounded arrangement at a time rather than first building a universal runtime.

| Experiment | Alternatives | What would change the ownership decision |
| --- | --- | --- |
| **Direct lowering versus distinct plan** | Existing interpreter; generated Python directly from the semantic action; compatible compiled action, all with matched guards/outcomes | Prefer direct lowering if it preserves semantics and removes enough schema/runtime work. Keep a separate executable IR only for demonstrated reused analysis or multiple real lowerings. |
| **Shared memory planning** | Independently configured teacher/student/conditioning/offload; coordinated budget allocation using existing mechanisms; optional block-level schedule | Deeper planning wins only on feasible workload size, throughput, memory, or stalled time. Include compiler workspaces, allocator behavior, replay state, and compile/solver cost. |
| **Derivative and update scheduling** | Routed framework gradients; explicit pullback/residual profile; supported backward/update split | Check the two-loss routing example, frozen gradient conduits, shared continuations, independent windows, and old-weight last use. Unsupported global clipping/early update must reject before mutation. Own more only if a real schedule benefits. |
| **Independent owners and native execution** | Specialized Python with async/process workers; an existing actor runtime; narrow native coordinator if still necessary | Out-of-order two-source joins, one stalled unrelated action, stale actual encoder results, bounded credits, cancellation/revision races, and partial failure. Measure device idle gaps, p99 readiness/quiescence, retained memory, FFI calls, and integration burden. |

Separately, exercise a small non-donating functional update with failure before candidate publication, and a donating/in-place profile that correctly reports uncertainty. This distinguishes a genuine stronger guarantee from a generic “success” flag. Its input and RNG recovery rules must remain explicit.

## 11. Proposed replacement for the report's conclusion

> Own the accepted semantic graph and the protocols required to authorize, correlate, and settle externally relevant execution regions. Compile those regions into the smallest supported executable realization; do not require a distinct universal control IR or custom VM when direct framework lowering suffices.
>
> Delegate tensor semantics, derivative primitives, generic compiler code generation, allocation, kernels, and transport. Permit project-owned numerical execution planning over supported differentiated regions when whole-run gradient, memory, replay, or update constraints provide information an isolated backend lacks. Borrow existing planning mechanisms before implementing new ones.
>
> Treat native coordination as an executor choice justified by measured scheduling behavior, deployment requirements, or reduced total complexity. Native execution does not imply project ownership of generic runtime machinery.

The boundary therefore survives as a **minimal required core**, with a justified optional extension into numerical planning. It does not survive as a uniquely proven stopping point. The next architecture decision should resolve a real coordination or memory case, not choose a lower language or construct a VM in anticipation of one.

## Primary source index

Project evidence:

- [Pinned branch](https://github.com/Enferlain/sd-scripts/tree/e0a6f9bb2f7299bf576dfdd1cf5a1c6e8c9ac627)
- [Design](https://github.com/Enferlain/sd-scripts/blob/e0a6f9bb2f7299bf576dfdd1cf5a1c6e8c9ac627/openspec/changes/rework-model-strategy-trainer/design.md)
- [Mechanism sketch](https://github.com/Enferlain/sd-scripts/blob/e0a6f9bb2f7299bf576dfdd1cf5a1c6e8c9ac627/openspec/changes/rework-model-strategy-trainer/research/training-mechanism-sketch.md)
- [Candidate executor](https://github.com/Enferlain/sd-scripts/blob/e0a6f9bb2f7299bf576dfdd1cf5a1c6e8c9ac627/library/training/execution.py)
- [Existing CPU probe results](https://github.com/Enferlain/sd-scripts/blob/e0a6f9bb2f7299bf576dfdd1cf5a1c6e8c9ac627/openspec/changes/rework-model-strategy-trainer/research/execution-cost-results.json)
- [Offloading utilities](https://github.com/Enferlain/sd-scripts/blob/e0a6f9bb2f7299bf576dfdd1cf5a1c6e8c9ac627/library/performance/custom_offloading_utils.py)
- [TorchAO CPU optimizer wrapper](https://github.com/Enferlain/sd-scripts/blob/e0a6f9bb2f7299bf576dfdd1cf5a1c6e8c9ac627/library/optimization/wrappers/cpu_offload.py)
- [Async text-encoder proposal](https://github.com/Enferlain/sd-scripts/blob/e0a6f9bb2f7299bf576dfdd1cf5a1c6e8c9ac627/docs_design/future_ideas/async_te_offload_otf.md)

External sources are linked at the claims they support. The proposed cross-region planner, pullback protocol, direct semantic lowering, and functional publication protocol are architectural inferences; none is claimed to have been implemented or benchmarked in this repository.
