## 1. Shared grouping foundation

- [x] 1.1 Add a shared base-path grouping helper module under `library/optimization/` that builds denoiser/text-encoder execution groups and logical groups.
- [x] 1.2 Add focused unit coverage for the shared grouping helper, including deterministic order and configured learning-rate behavior.

## 2. Fine-tune integration

- [x] 2.1 Update `FineTuneMode.build_optimizer_params(...)` to use the shared grouping helper instead of assembling groups inline.
- [x] 2.2 Update focused fine-tune optimizer tests to assert the shared-grouping-backed result shape and preserved logical-group metadata.

## 3. Validation

- [x] 3.1 Run focused grouping/fine-tune tests and record the change in `CHANGELOG.md` if the implementation lands in this working session.
