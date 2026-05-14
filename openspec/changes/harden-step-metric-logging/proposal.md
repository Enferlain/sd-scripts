## Why

Step metrics are one of the main runtime observability surfaces for training runs, but the current implementation is only partially specified and only partially covered by tests. The code emits loss, LR, validation, max-norm, loss-modifier, sampler, and tracker-routing metrics, but several categories have weak or implicit behavior, and LR-label mismatches currently fail by accident rather than through an intentional contract.

This matters now because the logging system is being separated from durable metadata/reporting concerns. Before future dashboards, run warehouses, or metadata consumers rely on step metrics, the metric payload and tracker-routing behavior should be explicit, tested, and stable.

## What Changes

- Define a typed internal step-metric event for the metric categories that `library/logging/metrics.py` owns.
- Add focused coverage for every currently emitted metric category.
- Make LR label mismatch behavior intentional, with a clear error instead of an accidental `IndexError`.
- Cover tracker routing behavior for TensorBoard, W&B, and other Accelerate trackers.
- Render the typed event to the existing flat tracker payload so current callers and metric key names remain compatible.
- Keep the change narrow: do not redesign metadata, adapter config reporting, resource monitoring, or the dashboard UI as part of this work.
- Preserve existing metric key names unless a test exposes an unsafe or accidental behavior that needs a deliberate compatibility decision.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `training-observability`: clarify the step-metric payload and tracker-routing contract for per-step scalar metrics.

## Impact

- `library/logging/metrics.py`: typed step event, flat tracker rendering, LR label validation, tracker routing tests, and small helper extraction.
- `library/training/phases/training_loop.py`: only touched if the metric generation call contract needs a narrow adjustment.
- `tests/unit/logging/test_step_logging.py`: expanded coverage for emitted metric categories and routing behavior.
- `tests/unit/training/phases/test_training_loop.py`: expanded coverage only if call-site behavior changes.
- No external dependencies are expected.
