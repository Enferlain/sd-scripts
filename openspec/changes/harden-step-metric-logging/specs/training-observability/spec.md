## ADDED Requirements

### Requirement: Step metric generation covers all current metric categories
The training observability metrics concern SHALL build a typed internal step-metric event for the current runtime metric categories emitted by the training loop and render that event to the existing stable flat tracker payload.

#### Scenario: Base loss metrics are emitted
- **WHEN** step metrics are generated with current and average loss values
- **THEN** the step event MUST store those loss values
- **AND** the rendered payload MUST include `loss/current` and `loss/average` with those values

#### Scenario: Loss modifier metrics are emitted when available
- **WHEN** step metrics are generated with scaled current and average loss values
- **THEN** the step event MUST store those scaled loss values
- **AND** the rendered payload MUST include `loss/current_scaled` and `loss/average_scaled`

#### Scenario: Validation loss metrics are emitted when available
- **WHEN** step metrics are generated with current and average validation loss values
- **THEN** the step event MUST store those validation loss values
- **AND** the rendered payload MUST include `loss/current_val_loss` and `loss/average_val_loss`

#### Scenario: Max norm metrics are emitted when available
- **WHEN** step metrics are generated with max-norm regularization facts
- **THEN** the step event MUST store those norm facts
- **AND** the rendered payload MUST include `max_norm/keys_scaled`, `max_norm/max_key_norm`, and `norm/avg_key_norm` for the provided values

#### Scenario: Optional norm metrics are emitted when available
- **WHEN** step metrics are generated with gradient or combined norm values
- **THEN** the payload MUST include `norm/avg_grad_norm` or `norm/avg_combined_norm` for the provided values

#### Scenario: Modifier learning rates are emitted when available
- **WHEN** step metrics are generated with modifier learning-rate values
- **THEN** the step event MUST store the modifier learning-rate values
- **AND** the rendered payload MUST include one `lr/<modifier_name>` entry for each modifier learning rate

#### Scenario: Sampler metrics are emitted when sampler state is available
- **WHEN** step metrics are generated with sampler EMA state, entropy ratio, and sampled timesteps
- **THEN** the step event MUST store structured sampler facts
- **AND** the rendered payload MUST include sampler EMA summary keys, per-bin EMA keys, entropy ratio, and sampled timestep histogram keys

### Requirement: Step metric learning-rate labels are explicit and validated
The training observability metrics concern SHALL pair scheduler learning-rate values with explicit trainer-facing labels and fail clearly when the labels do not match the scheduler output.

#### Scenario: Optimization-plan labels take priority
- **WHEN** step metrics are generated with an optimization plan that provides learning-rate descriptions
- **THEN** those labels MUST be used for `lr/<label>` keys instead of legacy `lr_descriptions`

#### Scenario: Legacy labels remain supported
- **WHEN** step metrics are generated without an optimization plan and with legacy learning-rate descriptions
- **THEN** those labels MUST be used for `lr/<label>` keys

#### Scenario: Mismatched label count fails clearly
- **WHEN** the number of learning-rate labels is less than the number of scheduler learning-rate values
- **THEN** step metric generation MUST raise a `ValueError` that identifies the label count and scheduler learning-rate count

#### Scenario: DAdapt and Prodigy derived rates keep labeled keys
- **WHEN** step metrics are generated for DAdapt or Prodigy optimizers
- **THEN** the payload MUST include `lr/d*lr/<label>` entries that use the same explicit labels as the base learning-rate entries

### Requirement: Tracker routing preserves backend-specific step semantics
The training observability metrics concern SHALL route step metric payloads to supported Accelerate tracker backends with explicit backend-specific step semantics.

#### Scenario: TensorBoard receives step argument
- **WHEN** step metrics are routed to a TensorBoard tracker
- **THEN** the tracker MUST receive the payload with `step` set to the step value selected by the caller

#### Scenario: W&B receives global step and epoch payload fields
- **WHEN** step metrics are routed to a W&B tracker
- **THEN** the tracker payload MUST include `global_step` and `epoch` values from the routing call
- **AND** those fields MUST NOT be added to the caller-owned payload object

#### Scenario: Other trackers receive step argument
- **WHEN** step metrics are routed to a non-TensorBoard and non-W&B tracker
- **THEN** the tracker MUST receive the payload with `step` set to the step value selected by the caller

#### Scenario: Epoch logging uses epoch step value
- **WHEN** epoch-level metrics are routed through the metrics concern
- **THEN** TensorBoard and other step-based trackers MUST receive the epoch value as their step argument
- **AND** W&B MUST receive the global step and epoch values in its payload
