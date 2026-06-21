## Purpose

This reconciliation records how the current resource-monitor, metadata, startup
estimate, and report surfaces map into the governing resource-intelligence
design. It is the implementation reference for migration work and supersedes
the historical `phase-resource-accounting` direction, archived at
`openspec/changes/archive/2026-06-11-phase-resource-accounting/`.

## Settled Observation Shape

An observation is still one measured value with explicit resource kind,
measurement kind, value, unit, source, and scope. However, the current monitor
collects several measurements at one lifecycle or sampler boundary. Repeating
all shared context for every scalar measurement would inflate retained
telemetry and lose the fact that the values were collected together.

The canonical production and retention unit is therefore an **observation
frame**:

```text
ResourceObservationFrame
  frame identity
  run/process/rank/host context
  lifecycle or collection event identity
  phase/step/epoch context
  collector/policy/quality context
  timestamp
  measurements[]
    measurement identity
    resource kind
    measurement kind
    value
    unit
    source
    resource/device scope
```

The frame is an ingestion and relationship container, not a fifth semantic
fact class. Every nested measurement remains an observation and must be
individually addressable so profiles and accounting statements can cite the
specific evidence they use.

The first-slice scalar `ResourceObservationFacts` remains useful for isolated
observations and as the semantic measurement model. Telemetry migration must
add a frame representation or equivalent metadata-runtime batch contract before
expanding current multi-measurement events into durable canonical facts.

## Current Surface Disposition

### Resource-Monitor Event Context

| Current value | Target disposition |
| --- | --- |
| `ts` | Observation-frame timestamp. |
| `event` | Frame lifecycle/collection event identity, not a measured resource value. |
| `run_identifier` | Required run relationship on every frame and durable derived fact. |
| `rank`, `world_size` | Process/rank context and relationships; not measurements. |
| `mode` | Collection-policy context retained on the frame or collector policy identity. |
| `device_scope` | Collection-request context; individual device measurements retain concrete device scope. |
| `config_name`, `git_sha`, `git_dirty` | Run facts referenced through the run identity rather than repeated as resource measurements. Compatibility JSONL may continue to repeat them. |
| `global_step`, `epoch`, `phase` | Optional relational runtime context. Phase is not the primary identity for all resource facts. |
| `duration_ms` | Runtime/lifecycle observation associated with the frame; it is not resource ownership. |
| `steps_per_sec`, `samples_per_sec` | Performance observations associated with the frame, retained only where required by resource profiles/reports. |
| `dropped_samples` | Degraded-observability/ingestion fact. It must remain observable and must not be treated as a resource measurement. |
| `collection_ms` | Collector-cost observation used for operational policy and profiling. |
| `deep_window_active` | Diagnostic-window/policy context. |

### Measured Resource Values

| Current value | Canonical observation mapping |
| --- | --- |
| `gpu_allocated_mb` | Aggregate CUDA allocator observation: resource `gpu_memory`, measurement `allocated`, unit `MiB`, source `torch_cuda_allocator`. |
| `gpu_allocated_by_device_mb` | One device-scoped allocator observation per device; aggregate remains a compatibility projection or explicitly identified aggregate observation. |
| `gpu_reserved_mb` | Aggregate CUDA allocator observation: measurement `reserved`. |
| `gpu_reserved_by_device_mb` | One device-scoped allocator `reserved` observation per device. |
| `gpu_peak_allocated_mb` | Window-bound aggregate allocator peak observation; frame/window context must define the reset boundary. |
| `gpu_peak_allocated_by_device_mb` | One device-scoped allocator peak observation per device with the same window boundary. |
| `gpu_used_mb` | Aggregate visible-device-used observation preserving NVML or torch fallback source and quality. |
| `gpu_used_by_device_mb` | One device-scoped visible-device-used observation per device. |
| `cpu_rss_mb` | Process-scoped host-memory RSS observation. |
| `cpu_vms_mb` | Process-scoped host-memory VMS observation. |
| `deep_alloc_retries` | Window-bound CUDA allocator diagnostic counter observation. |
| `deep_ooms` | Window-bound CUDA allocator diagnostic counter observation. |
| `deep_active_mb` | CUDA allocator diagnostic memory observation. |
| `deep_reserved_mb` | CUDA allocator diagnostic memory observation distinct from the normal reserved collector fact. |
| `deep_inactive_split_mb` | CUDA allocator fragmentation diagnostic observation. |

### Event Types

| Current event | Target disposition |
| --- | --- |
| `session_start`, `session_end` | Lifecycle observation frames containing boundary measurements. |
| `phase_start`, `phase_end` | Runtime-context observation frames containing boundary measurements. They support phase views but do not make phases the universal resource identity. |
| scheduled `step_sample` | Step-context observation frame containing synchronous snapshot measurements. |
| background-sampler `step_sample` | Collection observation frame. The compatibility projection may retain the current event name, but canonical facts must not imply a known training step when none was recorded. |
| deep counters attached to other events | Measurements in the relevant frame with explicit diagnostic-window context; later migration may use dedicated diagnostic frames if that produces clearer retention semantics. |

### Existing Metadata Facts

| Current surface | Target disposition |
| --- | --- |
| `ResourceMonitorFacts` | Narrow no-frame compatibility fallback only. Canonical observation frames are filed alone on the normal path; do not restore bundled mirror filing. |
| `ResourceObservationFacts` | Accepted scalar observation semantics and isolated-observation path. Extend with or complement by an accepted frame/batch representation before telemetry migration. |
| `StructuralResourceFacts` | Canonical structural fact. |
| `ResourceProfileFacts` | Canonical durable derived profile. |
| `ResourceAccountingFacts` | Canonical evidence-constrained accounting statement. |
| `ResourceAccountingGapFacts` | Canonical unresolved accounting result; never an owner. |
| `AnalyticsSnapshotFacts` containing report resource payloads | Compatibility/report artifact snapshot, not a canonical resource database. |

### Startup Estimates

| Current estimate | Target disposition |
| --- | --- |
| loaded component parameter bytes | Structural fact owned by the model component, basis `measured_structure`. |
| trainable parameter bytes | Structural fact owned by the component or optimization group, basis `measured_structure`. |
| frozen parameter bytes | Derived structural view from total minus trainable bytes; retain only when useful as a query/projection. |
| gradient bytes | Estimated structural fact with explicit estimation method and validity assumptions until runtime state can provide a trustworthy measurement. |
| optimizer-state bytes | Estimated structural fact with optimizer-family method/version; replace or supplement with measured optimizer tensor state where available. |
| total startup estimate | Profile/report projection, not an independent observed fact. |
| DeepSpeed/ZeRO caveat | Validity/quality metadata on affected structural estimates and profiles. |

### Report-Derived Values

| Current report value | Target disposition |
| --- | --- |
| paired phase rows | Resource-run view/projection over original boundary frames. |
| session and phase peaks | Versioned profile values when reusable; otherwise report projections over accepted observations. |
| per-device debug rows | Resource-run view/projection over device-scoped observations. |
| session/phase deltas | Derived profile or presentation value; never ownership evidence by itself. |
| explanatory resource observations | Presentation-only interpretation unless a reusable, versioned profile question is defined. |
| component-memory estimate rows | Projection over structural facts. |
| report `resource_monitor` payload | Compatibility report projection, not an independent canonical schema. |
| report JSONL event count/path | Artifact/projection metadata linked to the run. |

## Historical Phase-Accounting Disposition

The historical `phase-resource-accounting` change is superseded by this change.
Its useful requirements are retained as follows:

- Declared owner scopes remain a valid source of bounded operation/window
  evidence.
- Structural accounting remains distinct from operation/window accounting.
- Accounting gaps remain explicit results and must never become fallback
  owners.
- Operation-local movement must not become a persistent ownership claim.
- Heavy accounting diagnostics remain explicitly configured or bounded.
- Reports should present observed, profiled, accounted, and gap values plainly
  without creating a competing accounting database.

The following historical directions are rejected:

- Phases are not the primary identity for all resource intelligence.
- Accounting is not report-local or deferred until after report design.
- Metadata is not merely downstream; it owns accepted schemas, validation,
  relationships, storage, queries, and projections.
- Owner-scope events are not independently canonical. They become accepted
  evidence facts or observation frames and flow through the same metadata
  boundary.

## Ordering Consequences

1. Settle and implement observation-frame/batch semantics before expanding
   current bundled events into scalar durable observations.
2. Define retention and bounded metadata ingestion against realistic frames,
   including active-training in-memory behavior and optional durable storage.
3. Migrate one complete monitor path through canonical facts, metadata-owned
   projection, and resource-run query before broad collector expansion.
4. Introduce owner scopes only when their evidence can be retained and queried
   through the governing fact model.
