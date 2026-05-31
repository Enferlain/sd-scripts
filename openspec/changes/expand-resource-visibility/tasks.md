## 1. Host Memory Breadth

- [x] 1.1 Add `cpu_vms_mb` to the resource-monitor live event/data model and metadata-facing observability facts.
- [x] 1.2 Thread `cpu_vms_mb` through resource-monitor metadata filing and benchmark-report payload generation/rendering.
- [x] 1.3 Add focused tests covering the new host-memory field in raw events, metadata emitters, and run-report output.

## 2. Richer GPU Visibility

- [x] 2.1 Design and implement an additive schema for per-device GPU readings alongside the existing aggregate fields.
- [x] 2.2 Extend report/snapshot payloads to preserve per-device GPU visibility without dropping the current aggregate summaries.
- [x] 2.3 Add focused tests for local-device and `all_visible` collection behavior.

## 3. Report / Debug Expansions

- [ ] 3.1 Add richer report/debug resource summaries that build on the expanded live schema instead of inventing separate raw collection paths.
- [ ] 3.2 Document which resource surfaces are live metadata versus report/debug-only outputs.
