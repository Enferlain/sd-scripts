## 1. Scheduler runtime metadata modeling

- [x] 1.1 Extend `library/optimization/types.py` with explicit scheduler/runtime plan metadata for scheduler mode and target selection.
- [x] 1.2 Add focused unit coverage for scheduler/runtime metadata plan behavior and scheduler compatibility paths.

## 2. Optimizer phase and scheduler integration

- [x] 2.1 Add a shared scheduler-runtime resolver and update `get_scheduler_fix(...)` to consume plan metadata when present.
- [x] 2.2 Update the optimizer phase to populate scheduler/runtime metadata on plan-aware build results before scheduler construction.

## 3. Validation

- [x] 3.1 Run focused scheduler/optimizer-phase tests and record the change in `CHANGELOG.md` if the implementation lands in this working session.
