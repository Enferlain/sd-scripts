## Telemetry Ingestion First-Slice Results

This note records the task-3 ingestion decisions and representative synthetic
measurements used to pressure-test them. The benchmark is intentionally a
directional local measurement; task 10 adds coarse migration acceptance gates
that guard against pathological regressions without making CI depend on exact
machine timing.

## Implemented Contract

- `ResourceObservationFrameFacts` carries shared runtime/collection context.
- `ResourceObservationMeasurementFacts` keeps each measurement individually
  addressable.
- Frame emission produces one frame record, lightweight measurement records,
  frame/run/scope relationships, and measurement/frame/device relationships.
- `MetadataRuntime.file_many(...)` routes explicit batches through one
  backend/store batch.
- `MetadataRuntime.buffer(...)` validates and queues high-frequency telemetry
  without backend or storage I/O.
- The bounded buffer supports `drop_oldest` and `drop_newest`.
- `flush_buffer(...)` limits each synchronous flush by item count.
- Concurrent telemetry flushes are serialized without holding the producer
  queue lock during backend/storage work.
- Drops and failed flushes are observable through `MetadataBufferReport` and a
  `metadata_ingestion_degraded` event on the next successful flush.
- Capacity drops and backend-ingestion failures have distinct counters.
- SQLite batch ingestion uses one transaction per metadata batch.

## Retention Policy

The first contract does not silently sample or summarize accepted facts.
Producers choose which facts enter the telemetry buffer. Once selected:

- capacity bounds active-run queued telemetry
- `drop_oldest` preserves the newest diagnostic context
- `drop_newest` preserves the earliest accepted context
- a failed flush drops only that bounded batch rather than requeueing storage
  pressure onto the producer path
- every drop contributes to explicit degraded-ingestion counters

`flush_buffer(...)` is still a synchronous storage operation: item-count
bounding limits the work submitted but cannot impose a time limit on an
arbitrary backend. Resource-domain producers use `buffer(...)` on the hot path;
orchestration must schedule flushes outside latency-critical training sections.

Future collector policy may select or summarize frames before buffering, but it
must not hide that behavior inside metadata storage.

## Representative Measurements

Measured locally on June 11, 2026 using `uv run python`, `tracemalloc`, the
in-memory backend, and in-memory SQLite. Each two-measurement case used 1,000
sample boundaries:

| Shape | Time per sample boundary | Retained Python memory | Produced facts |
| --- | ---: | ---: | --- |
| legacy bundled event, single filing | 3575 us | 0.70 MiB | 1,000 events |
| two independent scalar observations | 4179 us | 2.98 MiB | 2,000 records / 3,000 edges |
| two-measurement observation frames, one in-memory batch | 3997 us | 3.15 MiB | 3,000 records / 4,000 edges |
| two-measurement observation frames, one SQLite batch | 4137 us | 0.18 MiB after write | 3,000 records / 4,000 edges |

The two-measurement frame is not smaller than independent scalar observations:
it intentionally pays for frame identity and containment relationships. Its
value is preserving co-collection truth and avoiding repeated context as the
number of measurements per boundary grows. This confirms that retention and
durable-store selection remain necessary; observation frames are not a memory
optimization by themselves.

A second 500-boundary comparison with eight measurements and realistic repeated
runtime context demonstrated that crossover:

| Shape | Time per sample boundary | Retained Python memory |
| --- | ---: | ---: |
| eight independent scalar observations | 16001 us | 7.22 MiB |
| one frame with eight measurements | 9903 us | 5.52 MiB |

## Acceptance Interpretation

The buffered producer path performs validation plus bounded in-memory queue
work only. It never performs backend or storage I/O. Flushes are explicit,
bounded by item count, serialized against other telemetry flushes, and intended
for orchestration-selected boundaries outside latency-critical training work.
Backend failure is converted into a degraded report rather than escaping the
buffered telemetry path.

## Migration Acceptance Thresholds

The final migration gate treats accepted metadata facts as authoritative and
allows compatibility JSONL/report outputs only as projections or fallbacks. The
runtime thresholds are intentionally broad, deterministic sanity checks for the
synthetic unit benchmark in `tests/unit/metadata/test_telemetry_ingestion.py`.
They are not product performance targets.

| Policy | Representative frames | Max ingestion wall time | Min ingestion throughput | Max query/projection wall time | Max traced Python peak | Max JSONL bytes/event |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `off` | 0 | N/A | N/A | N/A | N/A | N/A |
| `basic` | 25 | 5 s | 5 frames/s | 5 s | 32 MiB | 4096 |
| `sampled` | 250 | 15 s | 5 frames/s | 5 s | 64 MiB | 4096 |
| `deep` | 250 | 20 s | 5 frames/s | 5 s | 96 MiB | 6144 |

Compatibility equivalence uses exact equality for event count, total resource
event count, total JSONL event count, session start/end payloads, GPU-used
session peaks, phase rows, and debug rows. Profile and accounting views are not
required to appear in JSONL fallback output; they remain accepted metadata
facts and metadata-owned exports.
