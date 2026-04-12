## 1. Execution-group plan modeling

- [x] 1.1 Extend `library/optimization/types.py` with explicit execution-group-oriented plan fields and compatibility accessors.
- [x] 1.2 Add focused unit coverage for execution-group naming, compatibility accessors, and legacy materialization behavior.

## 2. Base-path integration

- [x] 2.1 Update the shared fine-tune grouping helper to return execution-group-oriented data.
- [x] 2.2 Update `FineTuneMode.build_optimizer_params(...)` to build the optimization plan through explicit execution groups.

## 3. Validation

- [x] 3.1 Run focused execution-group/fine-tune tests and record the change in `CHANGELOG.md` if the implementation lands in this working session.
