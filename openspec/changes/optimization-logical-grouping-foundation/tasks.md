## 1. Optimization plan foundation

- [x] 1.1 Extend `library/optimization/types.py` with logical-group and optimization-plan types while preserving legacy parameter-group materialization.
- [x] 1.2 Add focused unit coverage for logical-group ordering, plan metadata, and legacy materialization compatibility.

## 2. Base-path mode integration

- [x] 2.1 Update the base training-mode contract so the optimizer build seam can return shared plan metadata instead of only fragmented tuple fields.
- [x] 2.2 Update `FineTuneMode.build_optimizer_params(...)` to build the first base-path optimization plan from denoiser and text-encoder groups.

## 3. Trainer and logging integration

- [x] 3.1 Update the optimizer phase to consume the optimization plan, build runtime optimizer/scheduler objects, and store plan metadata on the trainer.
- [x] 3.2 Update startup diagnostics and step logging helpers to use logical-group metadata instead of relying on raw `optimizer.param_groups` plus `lr_descriptions`.

## 4. Validation

- [x] 4.1 Update focused orchestration tests for the new plan-carrying contract and logging behavior.
- [ ] 4.2 Run targeted optimizer/training tests and record the change in `CHANGELOG.md` if the implementation lands in this working session.
