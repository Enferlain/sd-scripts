# Resource-aware execution, scheduling and reusable performance knowledge

Date: 2026-11-26

Status: research / non-normative. Consolidates the November 24 runtime discussion (including the investigation initially framed as alternatives to asyncio) and the execution side of the November 26 authoring discussion. Companion: [Authoring execution requirements](execution-requirements-and-policies.md). Neither note establishes a new API, package boundary, dependency, scheduler or design decision. The recovered earlier discussion was available as a summary; the upstream mechanisms below were rechecked against public documentation and publications. No external runtime has been built or benchmarked for this note.

## 1. Research question

How can the system execute authored strategies using the available heterogeneous resources, select between valid realizations of the same behavior, and improve those choices using evidence from prior execution? The question is broader than replacing Python's event loop or parallelizing a training step. A strategy may describe a configurable whole run with several independently progressing activities, evolving participants, stateful producers, requests, optimization windows, and changing resource requirements.

The author determines the intended behavior, including the roles of existing or custom code. The IR system understands and translates that authored strategy, derives obligations, and coordinates its execution. Ordinary library code need not be written in the IR's vocabulary. A system-maintained inventory of knowledge about code, together with analysis and observation, may help the system understand it; that knowledge does not supply independent training intent or substitute another operation for one the author chose.

The resource problem includes both correctness (e.g. a requested revision remains accessible until its last use) and optimization (e.g. whether to realize it as a copy or protect the live state). The former constrains the latter. An equivalent execution alternative may still change memory, latency, throughput, energy, or failure behavior. These can be relevant requirements even when the mathematical computation is unchanged.

## 2. Existing design and relevant boundaries

**Observed in the repository:** `openspec/changes/rework-model-strategy-trainer/design.md`, branch `model-strategy-trainer`, inspected 2026-11-26; the portions discussed here correspond to the D14–D20 design. Links are to the inspected design lineage (commit `9055aa20`); line positions may shift in later revisions.

| Area | Existing capability or decision | Question for this research |
| --- | --- | --- |
| [D14, access and requested work](https://github.com/Enferlain/sd-scripts/blob/9055aa20/openspec/changes/rework-model-strategy-trainer/design.md#L2798-L2865) | Separates selected revision, protected live access, detached snapshots, request reservations, admission and placement. | Can some physical realization choices be delegated while the accepted revision and protection guarantees remain intact? |
| [D15, runtime model](https://github.com/Enferlain/sd-scripts/blob/9055aa20/openspec/changes/rework-model-strategy-trainer/design.md#L3183-L3227) | Per-process/rank coordinator, exposed execution boundaries, worker execution, device completion. | Where can a scheduler intervene without per-operation Python dispatch or changing ownership? |
| [D15, capacity and turns](https://github.com/Enferlain/sd-scripts/blob/9055aa20/openspec/changes/rework-model-strategy-trainer/design.md#L3250-L3385) | Device admission, resource observations, stream ordering, held resources, rank coordination; currently overlap requires permission. | How can measured occupancy, alternative realizations and progress constraints inform admission? The overlap rule is an open review item, not automatically superseded. |
| [D15, leaving the device](https://github.com/Enferlain/sd-scripts/blob/9055aa20/openspec/changes/rework-model-strategy-trainer/design.md#L3386-L3445) | Separates route-internal rematerialization from between-use residency; current placement constraints reflect supported implementations. | Can the system choose among exact supported moves, retention and offload without inventing mandatory author permissions? |
| [D16, retained dependencies](https://github.com/Enferlain/sd-scripts/blob/9055aa20/openspec/changes/rework-model-strategy-trainer/design.md#L3896-L3948) | Leases, retained products, release at last use, atomic multi-resource acquisition, lifecycle conformance. | Resource accounting must include everything kept by live work, not merely the call's temporary peak. |
| D17–D20 | Activities, request lifecycle, owners, state updates, training units and per-call modes. | Scheduling does not replace the authority that defines a valid update or settlement. |

D15 already records useful evidence: on one RTX 3080 experiment, alternating separate streams with large allocations under a constrained memory budget incurred repeated synchronization; sharing a stream for work taking turns was substantially cheaper. D15 also documents a single-step SDXL bitwise comparison with checkpointing on/off and explicitly limits the observation. These are examples of what reusable measurements could inform, not universal scheduling laws.

The currently documented implementation declarations, explicit permissions and access realizations are themselves being reconsidered in the live authoring discussion. Do not treat their presence in the old design as approval of their final placement or author-facing form. This note takes the existing lifecycle/protection concepts as research inputs, not as an instruction to preserve every current interface.

## 3. External systems and applicability

### 3.1 Realm: event-driven heterogeneous execution substrate

**Observed:** Realm exposes asynchronous task launch with event preconditions, events and user events, processor and memory abstractions, reservations, physical instances and copies, completion queues, and operation profiling. It originated beneath Legion and is available as a standalone C++ runtime. Its completion queue provides a useful pattern for consuming many completion notifications without polling each event separately.

Sources: [Realm overview](https://legion.stanford.edu/realm/), [repository](https://github.com/StanfordLegion/realm), [events tutorial](https://legion.stanford.edu/tutorial/realm/events_basic.html), [completion queues](https://legion.stanford.edu/tutorial/realm/completion_queue.html), [profiling](https://legion.stanford.edu/realm/tutorial/realm/profiling.html).

**Inferred relevance:** Realm is the strongest candidate from the earlier discussion for investigating selective reuse of a low-level task/event substrate, especially completion, dependency propagation and native execution across heterogeneous resources. It is not a replacement for the authored strategy, accepted representation, run authority or training-specific semantics. A native completion queue could be useful even if no other Realm functionality were adopted.

**Unknown:** integration with the existing Python runtime and PyTorch's allocator, streams and event ordering; GIL-free callback and object-lifetime behavior on free-threaded CPython; build/package cost; per-task overhead; cancellation and exception propagation; interaction with distributed training and vendor-specific device adapters. Native workers, GPU support and asynchronous events do not establish compatibility by themselves. Inspect actual APIs and test these boundaries before suggesting adoption.

### 3.2 Legion / Regent: logical access and physical mapping

**Observed:** Legion represents task data as logical regions, with privileges and coherence information; mapping callbacks choose processors, physical memory instances, variants, priorities and profiling requests. The runtime manages data movement and ordering consistent with the logical requirements. The mapper can receive measurements and adapt its choices. Realm provides the lower-level execution machinery. Legion explicitly separates logical correctness from mapping; it does not claim to automatically derive an optimal decomposition or mapping for arbitrary applications.

Sources: [overview](https://legion.stanford.edu/overview/index.html), [mapper API](https://legion.stanford.edu/mapper/index.html), [custom mapper tutorial](https://legion.stanford.edu/tutorial/custom_mappers.html).

**Inferred relevance:** strong conceptual reference for separating the accepted work/dependency representation from resource realization. A Legion mapper is not equivalent to the Kuroiro strategy author. Kuroiro may need to derive more information from ordinary authored Python than Legion asks its application author to supply. Directly adopting Legion's region model would require demonstrating that it preserves Kuroiro's state ownership, version and effect semantics without demanding Legion-specific declarations in library code. See the companion note for Bishop, the higher-level mapping-policy DSL.

### 3.3 StarPU: history-informed scheduling and data residency

**Observed:** StarPU provides heterogeneous task scheduling around codelets and managed data handles. It can manage coherence and transfer, select execution workers, and exploit calibrated task performance models. Its official tutorial documents models saved to files for later reuse, calibration of new observations, and explicit invalidation when kernel code changes. The Task Graph Market exports task footprints, architecture, sample counts and timing statistics; it also illustrates static scheduling directives and prefetch guidance.

Sources: [StarPU quick start](https://starpu.gitlabpages.inria.fr/quickGuide.html), [performance model calibration](https://starpu.gitlabpages.inria.fr/tutorials/2024-05-Saclay/fortran.html), [Task Graph Market](https://starpu.gitlabpages.inria.fr/market.html), [original runtime paper](https://doi.org/10.1002/cpe.1631).

**Inferred relevance:** strongest reference for the reusable performance-knowledge part of this investigation. Compare code identity, input/resource footprint, target architecture and observations rather than assuming a task's cost is a constant. Cold-start calibration, uncertainty, transfer cost, and adaptation are relevant even if Kuroiro keeps its own coordinator.

**Unknown:** whether selective reuse of the performance-model machinery is feasible without adopting StarPU's task and data ownership model. PyTorch tensors and autograd state already have storage/lifetime mechanisms; a second coherence authority could be incorrect. Investigate integration at an opaque execution boundary before considering wholesale data registration. History-based duration predictions do not prove resource bounds or semantic equivalence.

### 3.4 Parla: Python-oriented heterogeneous orchestration

**Observed:** Parla is a Python task-based orchestration framework for CPU/GPU work on a shared-memory machine. It supports task dependencies, resource-aware scheduling, task variants and managed data movement, with an emphasis on integration with scientific Python. The published work and project describe Python tasks and array-oriented data management rather than a universal ML training authority.

Sources: [Parla.py repository](https://github.com/ut-parla/Parla.py), [SC22 paper and abstract](https://sc22.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap519.html).

**Inferred relevance:** useful as a reference for Python-facing task submission, resource admission and multi-GPU orchestration. Its single-process emphasis is relevant to the current per-process coordinator. The repository API and paper are worth inspecting alongside a minimal native substrate experiment.

**Unknown:** current free-threaded Python support, Torch tensor integration, event compatibility, operational maintenance and release compatibility. Do not infer these from the Python API or from the performance results in the paper.

### 3.5 IREE Stream: resource lifetimes in compiler IR

**Observed:** IREE's Stream dialect represents resource lifetime, affinity, asynchronous execution and timepoints. Passes schedule allocations and propagate or eliminate host waits where valid. It connects compiler reasoning about resource flow to physical asynchronous execution.

Sources: [Stream dialect](https://iree.dev/reference/mlir-dialects/Stream/), [Stream passes](https://iree.dev/reference/mlir-passes/Stream/).

**Inferred relevance:** particularly useful for investigating how Kuroiro could represent asynchronous completion, storage lifetimes and affinity through its own IR and lowering rather than making every scheduler decision a runtime callback. It is not a drop-in runtime for arbitrary Python/PyTorch training activities; IREE's compiler-visible resources differ from opaque Python calls, evolving model state and long-lived activities.

### 3.6 Checkmate and Dynamic Tensor Rematerialization (DTR)

**Observed:** Checkmate formulates tensor retention/recomputation under a memory budget as a constrained optimization problem. DTR instead uses an online eviction and rematerialization heuristic suitable for dynamic computation. These establish two different planning horizons for the same class of memory tradeoff.

Sources: [Checkmate, MLSys 2020](https://proceedings.mlsys.org/paper_files/paper/2020/hash/0b816ae8f06f8dd3543dc3d9ef196cab-Abstract.html), [DTR, ICLR 2021](https://arxiv.org/abs/2006.09616), [DTR project](https://sampl.cs.washington.edu/projects/dtr.html).

**Inferred relevance:** a static schedule may be effective for a repeatable forward/backward region; online choices may be more suitable when shapes or workloads change. Neither implies that the coordinator should reason about every tensor operation. Domain/backend mechanisms may offer valid rematerialization plans over regions, while the coordinator only needs their resource, dependency and completion properties. Recomputability, side effects, random state, mutable buffers, parameter revisions and numerical requirements must be established for each alternative.

### 3.7 Candidate assessment

| Candidate | Main value | Initial research stance |
| --- | --- | --- |
| Realm | Native task/event/completion substrate | Investigate narrow integration and overhead; no replacement decision. |
| Legion / Regent | Logical requirements vs mapping policy | Architecture reference; full adoption is a substantially larger integration. |
| StarPU | Persistent calibrated costs, heterogeneous task/data scheduling | Inspect model reuse and policies first; data ownership compatibility is critical. |
| Parla | Python task/resource orchestration | Reference and possible prototype comparison; compatibility unverified. |
| IREE Stream | Asynchronous resource IR and compiler lifetime reasoning | Study representation/lowering, not assume runtime adoption. |
| Checkmate / DTR | Memory-constrained planning at different horizons | Research eligible backend mechanisms and measurements. |
| asyncio, thread pools, structured concurrency | Host control and waiting | Useful baseline; do not confuse event-loop overhead with the resource-planning problem. |

None has been demonstrated to be a drop-in solution for Kuroiro. Distinguish reuse of a library, a specific algorithm, an interface concept, and a benchmark reference. Adoption of a heavyweight framework would need to outperform or materially simplify a measured baseline, not just duplicate already-owned responsibilities.

## 4. Candidate scheduling model to investigate

A hierarchical resource-aware approach is the working research hypothesis from November 24, not a prescribed component layout. Existing system stages could make decisions at different scopes:

- **Preparation:** derive dependencies and obligations from accepted IR, inspect available resources and capabilities, enumerate supported realizations, select/precompute suitable alternatives for repeatable regions. Compare bounded static solvers or heuristics only where useful.
- **Admission:** consider pending work, freshness/protection, capacity, resident state, retained results, outstanding transfers, rank/collective order, fairness and progress. Do not hold a scarce lease across an unrelated wait merely to simplify scheduling.
- **Physical execution:** submit eligible work to worker/device mechanisms; observe physical completion and release resources when safe. Python return is not device completion.
- **Adaptation:** use observations to revise later choices at valid boundaries, subject to accepted semantics and requirements. Do not replay an uncertain mutation as a harmless retry.

These are responsibilities, not a requirement for four modules, a second translator, or a solver in every admission. The system's own IR remains the execution description. Whether a decision belongs in lowering, an existing owner/backend realization or coordinator admission depends on the information required and the cost of making it there.

### Resource accounting

At time `t`, a candidate admission must account for distinct live storage on the relevant device/memory domain. An illustrative decomposition is persistent resident state + live retained intermediates + in-flight transfer/staging + newly required temporary/working storage + allocator/external overhead. This is not a directly summable set of independent peak statistics: aliases, sharing, buffers reused across phases, allocator reservations, pending frees, and concurrent peaks must be handled explicitly to avoid double-counting or undercounting.

Useful resource classes to investigate include VRAM, pageable and pinned host memory, CPU worker capacity, transfer/communication bandwidth, disk/network I/O and accelerator execution capacity. Resource capacity and the cost of consuming a resource are different facts. A free-memory estimate is not an allocation guarantee; unexplained device activity and allocator fragmentation may invalidate it. Hard caps require enforceable mechanisms, conservative reservation, or an explicit supported failure policy, not merely a forecast.

An illustrative admission invariant is `committed occupancy + reserved additional demand <= usable capacity`, with occupancy/claims reconciled against actual completion and allocation evidence. The model must also consider wait cycles across resources and progress when outstanding work is retaining memory. A greedy scheduler that admits a consumer and leaves no room for its producer can deadlock despite each individual job fitting in isolation.

### Scheduling policies and objectives

Possible experiments: ready-queue/FIFO baseline, priority with starvation prevention, locality/transfer-cost scoring, critical-path or predicted earliest-finish heuristics, memory-pressure-aware admission, and bounded constraint optimization for stable preparation regions. Optimize only among feasible realizations; do not relax a requirement implicitly to achieve a cheaper plan. Unknown cost can lead to conservative choice or controlled calibration, not a fabricated precise estimate.

Use multiple horizons: a recurring training region may amortize expensive planning; dynamic request handling may favor small online decisions. Preserve a fast path with prepared straight-line execution where the IR does not need individual call-level dispatch. Evaluate solver time, profiling overhead and decision latency against actual workload costs.

## 5. Reusable evidence across runs

The earlier discussion emphasizes learning from execution history rather than re-deriving everything for every run. StarPU provides a concrete precedent; Kuroiro's code-identity discussion and existing resource observations give local context.

Distinguish these classes of knowledge:

| Class | Examples | Use |
| --- | --- | --- |
| Code/semantic evidence | Supported dtype/layout, effects, replay behavior, required resource relationships | Establish whether a realization can preserve obligations; evidence scope matters. |
| Target capabilities | Available memory domains, event mechanisms, supported kernels, collective capabilities | Rule in/out mechanisms for a target. |
| Observed performance | Host launch time, device duration, actual peak, transfer cost, residency pressure | Rank valid alternatives; predictions carry uncertainty. |
| Run outcome | Chosen realization, actual resource allocations, failures, achieved metrics | Explain decisions, validate predictions and improve future choices. |

A possible identity/compatibility context for a measurement (not a storage schema): code and selected implementation evidence; input shape/structure and relevant value regimes; dtype, precision and layout; backend/library/driver versions where relevant; device model, capacity and topology; selected realization and co-running work; measurement scope and sample count. Code identity alone is insufficient. Different devices, changing batch shapes, altered kernels, allocator behavior or co-resident tasks may produce very different results. Reuse should be conditional; uncertainty and invalidation must be visible.

Useful outputs of calibration include duration distributions, measured peaks, bandwidth estimates, failure rates and the conditions under which a mechanism was tested. A cold-start estimate should not be treated as a proven bound. Do not require library-local declarations to store this information. A catalog/evidence system may expose it to authoring tools and to preparation, but it does not decide what the author intended. Physical storage layout, append-only behavior, identity format and whether this shares infrastructure with other metadata are independent open implementation questions.

Evidence should distinguish an intended plan, decisions actually taken, and actual completion/effects. A replay may constrain physical choices more tightly than a fresh run, depending on the requested guarantee. Identical plans do not imply bit-identical execution. Distributed ranks may need coordinated but different local choices; compatible collective order is a separate obligation.

## 6. Representative cases and experiments

Use the live configurable SDXL sketch as an initial case, not as the boundary of the system. The broader cases below can be settings/branches of realistic strategies; they are not instructions to create one minimal strategy per case.

| Case | What to vary | What to establish |
| --- | --- | --- |
| Validation reads request-time revision | Retain live state, detached realization, placement on another device or shared device | Correct observed revision; full state preservation; measured memory/time; no unintended owner updates. |
| Live/frozen/cached encoders and adapters | Residency, movement and eligible recomputation | Frozen differentiable conduit still propagates gradients; mode independent of recording; storage references stay valid. |
| Data producers with feedback from training | Prefetch and backpressure | No reordered/future-dependent samples; bounded live work; request cancellation and settlement remain correct. |
| Accumulation, several units or delayed derivatives | Competing admissions and retained graphs | Last physical use protects state; no update while a retained result needs it; progress under finite capacity. |
| Different GPUs / multiple ranks | Unequal capacity, transfer paths and collective participation | Valid local mappings with compatible communication; failure doesn't strand another rank. |
| Stochastic/effectful work | Recompute, retry, overlap | No duplicated owner effects; correct RNG context; no unsafe retry after ambiguous mutation. |
| Repeated runs with changing shapes or code | Cold calibration vs compatible prior evidence vs stale evidence | Reuse helps rather than degrades decisions; incompatible data rejected or downweighted; accounting includes decision overhead. |

Suggested prototype sequence (not a commitment to scheduling milestones):

1. Establish measurements for the current coordinator: submit-to-admit, host submission time, device completion latency, idle gaps, peak resident/retained/temporary occupancy, transfer overhead, and full run throughput. Compare no-op and actual Torch/CUDA work. Include free-threaded Python, GIL detection and exception propagation rather than benchmarking only microtasks.
2. Test a narrow event/worker substrate experiment for Realm and, if justified, a Parla or native baseline. Use real Torch streams and allocator ownership. Verify event completion, object lifetimes, cancellation and shutdown before comparing performance.
3. Prototype history-informed admission/realization choices with StarPU-style cost keys and a simple heuristic. Compare against FIFO and fixed settings with measurements both withheld and reused. Measure total overhead and prediction errors.
4. Investigate memory-constrained realizations in a repeatable region and a dynamic region. Compare fixed checkpoint/offload choices, a bounded preparation solver and online policies only where the backend can support them safely.
5. Revisit multi-rank coordination and requirements using the companion note. No numerical benchmark substitutes for lifecycle or protection conformance.

Acceptance evidence should include outcomes that falsify the scheduler's assumptions (e.g. invalidated performance data, retained-state capacity exhaustion, a stochastic replay mismatch), not just fast paths.

## 7. Open questions and non-decisions

- Which existing IR boundaries are adequate for scheduling, and where does analysis need finer-grained information? Do not assume all library functions become individually dispatched tasks.
- What evidence is needed to authorize overlap, copy, eviction, rematerialization and migration? When is it derivable, observed, or supplied through a supported contract?
- Can an external runtime be integrated at a narrow physical boundary without creating a second source of state/ownership truth? What does it cost to build, maintain and deploy on supported targets?
- Which resource estimates can be enforced, and how does the system distinguish unsupported, infeasible, not yet planned and runtime-failed work?
- Which decisions should remain adaptive during execution? How are their actual outcomes and relevant evidence recorded for recovery and reproducibility?
- How should the existing overlap/offload permission rules and explicit access realizations be reviewed in light of delegated choices? The answer depends on actual cases and guarantees, not on removing all controls by fiat.

**Research conclusion:** a resource-aware planner/admission mechanism informed by reusable measurements is a coherent direction. Realm, Legion, StarPU, Parla, IREE and tensor-rematerialization research provide complementary mechanisms, not one ready-made system. Preserve the distinction between the accepted authored computation, derived obligations, physical alternatives and measured costs. No specific runtime, policy API, catalog schema or solver has been selected.
