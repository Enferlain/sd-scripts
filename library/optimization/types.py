from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from torch import nn


@dataclass(slots=True)
class ParameterGroup:
    """Typed optimizer parameter-group description.

    Phase 1 keeps the runtime payload compatible with the existing optimizer
    factories while giving the optimization layer one shared group shape.
    """

    params: list[Any]
    lr: float | None = None
    label: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def to_optimizer_dict(self) -> dict[str, Any]:
        group: dict[str, Any] = {"params": self.params}
        if self.lr is not None:
            group["lr"] = self.lr
        group.update(self.options)
        return group


@dataclass(slots=True)
class LogicalParameterGroup:
    """Stable trainer-facing optimization group metadata.

    Logical groups model user-facing grouping intent independently from the
    optimizer's runtime ``param_groups`` layout. In the first pass, the base
    fine-tune path keeps a straightforward 1:1 mapping to execution groups,
    but wrappers/offload/fused follow-up work can evolve execution grouping
    later without forcing trainer logging/diagnostics to change shape.
    """

    key: str
    params: list[Any]
    lr: float | None = None
    label: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    execution_group_indices: tuple[int, ...] = ()

    @property
    def metric_name(self) -> str:
        return self.label or self.key

    @property
    def parameter_count(self) -> int:
        return sum(param.numel() for param in self.params if isinstance(param, nn.Parameter))


@dataclass(slots=True)
class SchedulerRuntimeMetadata:
    """Explicit scheduler/runtime ownership metadata for optimizer orchestration."""

    mode: Literal["external", "embedded", "none"] = "external"
    target: Literal["optimizer", "base_optimizer"] = "optimizer"


@dataclass(slots=True)
class OptimizerRuntimeMetadata:
    """Explicit optimizer runtime behavior metadata for orchestration."""

    supports_train_eval_toggle: bool = False


@dataclass(slots=True)
class OptimizationPlan:
    """Shared optimizer-planning payload for trainer-facing orchestration."""

    logical_groups: list[LogicalParameterGroup] = field(default_factory=list)
    execution_groups: list[ParameterGroup] = field(default_factory=list)
    scheduler_runtime: SchedulerRuntimeMetadata | None = None
    optimizer_runtime: OptimizerRuntimeMetadata | None = None

    @property
    def lr_descriptions(self) -> list[str]:
        return [group.metric_name for group in self.logical_groups]

    @property
    def parameter_groups(self) -> list[ParameterGroup]:
        """Compatibility accessor for older plan consumers."""
        return self.execution_groups

    def materialize_execution_groups(self) -> list[Any]:
        return materialize_parameter_groups(self.execution_groups)


@dataclass(slots=True)
class OptimizerBuildResult:
    """Optimizer construction result plus normalized plan metadata."""

    optimizer_name: str
    optimizer_args: Any
    optimizer: Any
    optimization_plan: OptimizationPlan | None = None

    @property
    def lr_descriptions(self) -> list[str]:
        if self.optimization_plan is None:
            return []
        return self.optimization_plan.lr_descriptions


def build_parameter_group(
    params: Iterable[Any],
    *,
    lr: float | None = None,
    label: str | None = None,
    **options: Any,
) -> ParameterGroup:
    """Build a typed parameter group from any parameter iterable."""
    return ParameterGroup(params=list(params), lr=lr, label=label, options=options)


def build_logical_parameter_group(
    key: str,
    params: Iterable[Any],
    *,
    lr: float | None = None,
    label: str | None = None,
    execution_group_indices: Iterable[int] = (),
    **options: Any,
) -> LogicalParameterGroup:
    """Build a stable logical optimization group from any parameter iterable."""
    return LogicalParameterGroup(
        key=key,
        params=list(params),
        lr=lr,
        label=label,
        options=options,
        execution_group_indices=tuple(execution_group_indices),
    )


def build_module_parameter_group(
    module: nn.Module,
    *,
    lr: float | None = None,
    label: str | None = None,
    **options: Any,
) -> ParameterGroup:
    """Build a typed parameter group directly from a module.

    The materialized optimizer dict keeps `named_params` alongside `params`
    so optimizers that need parameter-name context can still build through
    the shared factory path.
    """
    named_params = list(module.named_parameters())
    return build_parameter_group((param for _, param in named_params), lr=lr, label=label, named_params=named_params, **options)


def materialize_parameter_groups(trainable_params: Any) -> Any:
    """Convert typed parameter groups into the legacy optimizer dict payload."""
    if not isinstance(trainable_params, list):
        return trainable_params

    materialized: list[Any] = []
    for group in trainable_params:
        if isinstance(group, ParameterGroup):
            materialized.append(group.to_optimizer_dict())
        else:
            materialized.append(group)
    return materialized
