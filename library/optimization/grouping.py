from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from torch import nn

from library.config.dataclasses.optimizer import LearningRatesConfig
from library.optimization.types import (
    LogicalParameterGroup,
    ParameterGroup,
    build_logical_parameter_group,
    build_module_parameter_group,
)


@dataclass(slots=True)
class GroupingResult:
    """Shared logical/execution grouping payload for optimizer planning."""

    execution_groups: list[ParameterGroup]
    logical_groups: list[LogicalParameterGroup]

    @property
    def parameter_groups(self) -> list[ParameterGroup]:
        """Compatibility accessor for older grouping consumers."""
        return self.execution_groups


def _resolve_text_encoder_lr(learning_rates: LearningRatesConfig, index: int) -> float:
    te_lr_raw = learning_rates.text_encoders
    if te_lr_raw is None:
        return learning_rates.base
    if isinstance(te_lr_raw, (int, float)):
        return te_lr_raw
    return te_lr_raw[index] if index < len(te_lr_raw) else learning_rates.base


def build_finetune_grouping(
    *,
    denoiser: nn.Module | None,
    train_denoiser: bool,
    text_encoders: Sequence[nn.Module],
    te_train_flags: Sequence[bool],
    learning_rates: LearningRatesConfig,
) -> GroupingResult:
    """Build the base fine-tune logical/execution groups in trainer-facing order."""
    execution_groups: list[ParameterGroup] = []
    logical_groups: list[LogicalParameterGroup] = []

    if train_denoiser:
        assert denoiser is not None, "denoiser must be loaded before build_finetune_grouping"
        denoiser_lr = learning_rates.denoiser if learning_rates.denoiser is not None else learning_rates.base
        parameter_group = build_module_parameter_group(denoiser, lr=denoiser_lr, label="denoiser")
        execution_groups.append(parameter_group)
        logical_groups.append(
            build_logical_parameter_group(
                "denoiser",
                parameter_group.params,
                lr=denoiser_lr,
                label="denoiser",
                execution_group_indices=(len(execution_groups) - 1,),
            )
        )

    for index, (text_encoder, train_flag) in enumerate(zip(text_encoders, te_train_flags)):
        if not train_flag:
            continue

        text_encoder_lr = _resolve_text_encoder_lr(learning_rates, index)
        label = f"text_encoder{index + 1}"
        parameter_group = build_module_parameter_group(text_encoder, lr=text_encoder_lr, label=label)
        execution_groups.append(parameter_group)
        logical_groups.append(
            build_logical_parameter_group(
                label,
                parameter_group.params,
                lr=text_encoder_lr,
                label=label,
                execution_group_indices=(len(execution_groups) - 1,),
            )
        )

    return GroupingResult(execution_groups=execution_groups, logical_groups=logical_groups)
