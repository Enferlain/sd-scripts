## 1. Step Metric Payload Coverage

- [x] 1.1 Add typed step-event construction and flat-rendering tests for base loss metrics.
- [x] 1.2 Add focused tests for scaled loss-modifier metrics and modifier learning-rate metrics.
- [x] 1.3 Add focused tests for validation loss metrics.
- [x] 1.4 Add focused tests for max-norm, average key norm, gradient norm, and combined norm metrics.
- [x] 1.5 Add focused tests for sampler EMA summary, per-bin EMA, entropy ratio, and timestep histogram metrics.

## 2. Learning-Rate Metric Hardening

- [x] 2.1 Add tests that define explicit `ValueError` behavior when LR labels are fewer than scheduler LR values.
- [x] 2.2 Extract or update LR metric helper logic so `generate_step_logs()` validates label count before indexing.
- [x] 2.3 Preserve existing optimization-plan label priority and legacy `lr_descriptions` behavior with focused tests.
- [x] 2.4 Preserve DAdapt and Prodigy `lr/d*lr/<label>` behavior under the validated label path.

## 3. Tracker Routing Coverage

- [x] 3.1 Add tests showing `step_logging()` routes step value and global step correctly.
- [x] 3.2 Add tests showing `epoch_logging()` uses epoch as step value while preserving global step for W&B payloads.
- [x] 3.3 Add tests for TensorBoard, W&B, and generic Accelerate tracker routing in `log_metrics_to_trackers()`.
- [x] 3.4 Copy W&B payloads per backend and encode non-mutation behavior in tests.

## 4. Verification And Documentation

- [x] 4.1 Run focused logging tests for `tests/unit/logging/test_step_logging.py`.
- [x] 4.2 Run focused training-loop tests if any call-site behavior changes.
- [x] 4.3 Update `CHANGELOG.md` with the completed metric hardening behavior.
- [x] 4.4 Confirm no adapter config reporting, metadata-system work, resource-monitor redesign, or dashboard/UI changes are included in this change.
