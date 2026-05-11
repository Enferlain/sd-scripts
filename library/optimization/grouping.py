from __future__ import annotations

import fnmatch
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from torch import nn

from library.adapters.shared import AdapterTrainableParameterRef, get_trainable_parameter_refs
from library.adapters.runtime.targets import AdapterResolvedTargets, build_component_module_targets
from library.config.dataclasses.optimizer import LearningRateGroupConfig, LearningRatesConfig
from library.models import NamedParameterComponentNames
from library.optimization.targets import (
    OptimizationTargetRef,
    build_parameter_target_ref,
    resolve_parameter_owner_modules,
)
from library.optimization.types import (
    LogicalParameterGroup,
    ParameterGroup,
    build_parameter_group,
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


@dataclass(slots=True)
class NamedParameterRef:
    """Resolved live parameter with a stable full name for matching."""

    target_ref: OptimizationTargetRef

    @property
    def component_key(self) -> str:
        return self.target_ref.component_key

    @property
    def component_label(self) -> str:
        return self.target_ref.component

    @property
    def local_name(self) -> str:
        return self.target_ref.path

    @property
    def full_name(self) -> str:
        return self.target_ref.selector

    @property
    def param(self) -> nn.Parameter:
        return self.target_ref.obj


@dataclass(slots=True)
class FinetuneSelection:
    """Resolved live parameter selection for the base fine-tune path."""

    selected_by_component: dict[str, list[NamedParameterRef]]

    @property
    def train_denoiser(self) -> bool:
        return bool(self.selected_by_component.get("denoiser", []))

    @property
    def te_train_flags(self) -> list[bool]:
        text_encoder_labels = sorted(
            (label for label in self.selected_by_component if label.startswith("text_encoder")),
            key=lambda label: int(label.removeprefix("text_encoder")),
        )
        return [bool(self.selected_by_component[label]) for label in text_encoder_labels]


@dataclass(slots=True)
class AdapterTargetSelection:
    """Optimization-owned resolved target selection for adapter training."""

    resolved_targets: AdapterResolvedTargets
    train_denoiser: bool
    te_train_flags: list[bool]

    @property
    def train_any_text_encoder(self) -> bool:
        return any(self.te_train_flags)


def _coerce_learning_rate_group_config(group: Any, source: str) -> LearningRateGroupConfig:
    if isinstance(group, LearningRateGroupConfig):
        return group

    if isinstance(group, dict):
        raw_name = group.get("name", "")
        raw_lr = group.get("lr", 0.0)
        raw_match = group.get("match", [])
    else:
        raw_name = getattr(group, "name", "")
        raw_lr = getattr(group, "lr", 0.0)
        raw_match = getattr(group, "match", [])

    if isinstance(raw_match, str):
        match = [raw_match]
    elif isinstance(raw_match, Sequence):
        match = list(raw_match)
    else:
        raise ValueError(f"{source} must define 'match' as a string or list of strings")

    try:
        lr = float(raw_lr)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must define 'lr' as a number") from exc

    return LearningRateGroupConfig(name=str(raw_name), lr=lr, match=match)


def _load_learning_rate_groups_file(groups_file: str) -> list[LearningRateGroupConfig]:
    path = Path(groups_file).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path

    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("optimizer.learning_rates.groups_file must point to a .yaml or .yml file")
    if not path.is_file():
        raise ValueError(f"optimizer.learning_rates.groups_file does not exist: {path}")

    with path.open(encoding="utf-8") as f:
        loaded = yaml.safe_load(f)

    if loaded is None:
        return []

    raw_groups: Any
    if isinstance(loaded, list):
        raw_groups = loaded
    elif isinstance(loaded, dict) and "groups" in loaded:
        raw_groups = loaded["groups"]
    else:
        raise ValueError("optimizer.learning_rates.groups_file must contain either a top-level list of groups or a top-level 'groups' list")

    if not isinstance(raw_groups, list):
        raise ValueError("optimizer.learning_rates.groups_file must define groups as a list")

    return [
        _coerce_learning_rate_group_config(group, f"optimizer.learning_rates.groups_file[{index}]")
        for index, group in enumerate(raw_groups)
    ]


def resolve_learning_rate_groups(learning_rates: LearningRatesConfig) -> list[LearningRateGroupConfig]:
    """Normalize inline/file-backed learning-rate groups into one resolved list."""
    inline_groups = list(getattr(learning_rates, "groups", None) or [])
    groups_file = getattr(learning_rates, "groups_file", None)

    if inline_groups and groups_file:
        raise ValueError("optimizer.learning_rates.groups and optimizer.learning_rates.groups_file cannot be set at the same time")

    if groups_file:
        return _load_learning_rate_groups_file(groups_file)

    return [
        _coerce_learning_rate_group_config(group, f"optimizer.learning_rates.groups[{index}]") for index, group in enumerate(inline_groups)
    ]


def _resolve_text_encoder_lr(learning_rates: LearningRatesConfig, index: int) -> float | None:
    te_lr_raw = learning_rates.text_encoders
    if te_lr_raw is None:
        return learning_rates.base
    if isinstance(te_lr_raw, (int, float)):
        return te_lr_raw
    return te_lr_raw[index] if index < len(te_lr_raw) else learning_rates.base


def _is_positive_lr(lr: float | None) -> bool:
    return lr is not None and lr > 0


def _matches_pattern(name: str, pattern: str) -> bool:
    if pattern.startswith("re:"):
        return re.search(pattern[3:], name) is not None
    return fnmatch.fnmatchcase(name, pattern)


def resolve_adapter_target_selection(
    *,
    model_type: str,
    denoiser: nn.Module | None,
    text_encoders: Sequence[nn.Module],
    learning_rates: LearningRatesConfig,
) -> AdapterTargetSelection:
    """Resolve optimization-owned adapter targets for the current PEFT path.

    Optimization still decides which top-level components are in scope from
    learning-rate policy, then expands those selected components into
    concrete module targets for adapter runtimes to consume without
    re-owning target discovery.
    """

    denoiser_lr = learning_rates.denoiser if learning_rates.denoiser is not None else learning_rates.base
    te_train_flags = [_is_positive_lr(_resolve_text_encoder_lr(learning_rates, index)) for index, _ in enumerate(text_encoders)]
    train_denoiser = _is_positive_lr(denoiser_lr)

    return AdapterTargetSelection(
        resolved_targets=build_component_module_targets(
            model_type=model_type,
            text_encoders=list(text_encoders),
            vae=None,
            denoiser=denoiser,
            include_text_encoders=te_train_flags,
            include_denoiser=train_denoiser,
        ),
        train_denoiser=train_denoiser,
        te_train_flags=te_train_flags,
    )


def _resolve_adapter_component_lr(ref: AdapterTrainableParameterRef, learning_rates: LearningRatesConfig) -> float | None:
    if ref.component_key == "denoiser":
        return learning_rates.denoiser if learning_rates.denoiser is not None else learning_rates.base
    if ref.component_key.startswith("text_encoder"):
        suffix = ref.component_key.removeprefix("text_encoder")
        if not suffix.isdigit():
            raise ValueError(f"Adapter trainable ref '{ref.name}' has malformed text-encoder component key '{ref.component_key}'")
        index = int(suffix) - 1
        return _resolve_text_encoder_lr(learning_rates, index)
    return learning_rates.base


def build_adapter_grouping(
    *,
    adapter,
    learning_rates: LearningRatesConfig,
) -> GroupingResult:
    """Build optimization-owned groups from repo-owned adapter trainable refs."""

    refs = get_trainable_parameter_refs(adapter)
    grouped_refs: dict[str, list[AdapterTrainableParameterRef]] = {}
    grouped_lrs: dict[str, float] = {}

    for ref in refs:
        lr = _resolve_adapter_component_lr(ref, learning_rates)
        if not _is_positive_lr(lr):
            continue

        group_label = ref.component
        grouped_refs.setdefault(group_label, []).append(ref)
        grouped_lrs[group_label] = lr

    execution_groups: list[ParameterGroup] = []
    logical_groups: list[LogicalParameterGroup] = []
    for group_label, group_refs in grouped_refs.items():
        group_index = len(execution_groups)
        params = [ref.param for ref in group_refs]
        execution_groups.append(
            build_parameter_group(
                params,
                lr=grouped_lrs[group_label],
                label=group_label,
                metadata={"param_names": [ref.name for ref in group_refs]},
            )
        )
        logical_groups.append(
            build_logical_parameter_group(
                group_label,
                params,
                lr=grouped_lrs[group_label],
                label=group_label,
                execution_group_indices=(group_index,),
            )
        )

    return GroupingResult(execution_groups=execution_groups, logical_groups=logical_groups)


def _build_component_parameter_refs(
    *,
    component_key: str,
    component_label: str,
    root_module: nn.Module,
    tags: frozenset[str],
) -> list[NamedParameterRef]:
    owner_modules = resolve_parameter_owner_modules(root_module)
    return [
        NamedParameterRef(
            target_ref=build_parameter_target_ref(
                component=component_label,
                component_key=component_key,
                path=name,
                parameter=param,
                owner_module_path=owner_modules.get(id(param), (None, None))[0],
                owner_module_type=owner_modules.get(id(param), (None, None))[1],
                tags=tags | frozenset({"parameter_target"}),
            )
        )
        for name, param in root_module.named_parameters()
    ]


def _collect_component_named_parameters(
    *,
    denoiser: nn.Module | None,
    text_encoders: Sequence[nn.Module],
    component_names: NamedParameterComponentNames | None = None,
) -> dict[str, list[NamedParameterRef]]:
    component_params: dict[str, list[NamedParameterRef]] = {}

    if denoiser is not None:
        public_denoiser_label = component_names.denoiser_name if component_names is not None else "denoiser"
        component_params["denoiser"] = _build_component_parameter_refs(
            component_key="denoiser",
            component_label=public_denoiser_label,
            root_module=denoiser,
            tags=frozenset({"denoiser"}),
        )

    for index, text_encoder in enumerate(text_encoders):
        internal_label = f"text_encoder{index + 1}"
        public_label = (
            component_names.text_encoder_names[index]
            if component_names is not None and index < len(component_names.text_encoder_names)
            else internal_label
        )
        component_params[internal_label] = _build_component_parameter_refs(
            component_key=internal_label,
            component_label=public_label,
            root_module=text_encoder,
            tags=frozenset({"text_encoder"}),
        )

    return component_params


def _resolve_group_matches(
    group: LearningRateGroupConfig,
    component_params: dict[str, list[NamedParameterRef]],
) -> list[NamedParameterRef]:
    if not group.name.strip():
        raise ValueError("optimizer.learning_rates.groups entries must define a non-empty name")
    if group.lr <= 0:
        raise ValueError(f"optimizer.learning_rates.groups '{group.name}' must use a positive learning rate")
    if not group.match:
        raise ValueError(f"optimizer.learning_rates.groups '{group.name}' must define at least one match pattern")

    matches: list[NamedParameterRef] = []
    seen_param_ids: set[int] = set()
    for refs in component_params.values():
        for ref in refs:
            if any(_matches_pattern(ref.full_name, pattern) for pattern in group.match):
                param_id = id(ref.param)
                if param_id not in seen_param_ids:
                    matches.append(ref)
                    seen_param_ids.add(param_id)

    if not matches:
        raise ValueError(f"optimizer.learning_rates.groups '{group.name}' did not match any named parameters")
    return matches


def resolve_finetune_trainability(
    *,
    denoiser: nn.Module | None,
    text_encoders: Sequence[nn.Module],
    learning_rates: LearningRatesConfig,
    groups: Sequence[LearningRateGroupConfig] | None = None,
    component_names: NamedParameterComponentNames | None = None,
) -> tuple[bool, list[bool]]:
    """Resolve component-level trainability from baseline LRs plus explicit groups."""
    selection = resolve_finetune_selection(
        denoiser=denoiser,
        text_encoders=text_encoders,
        learning_rates=learning_rates,
        groups=groups,
        component_names=component_names,
    )
    return selection.train_denoiser, selection.te_train_flags


def resolve_finetune_selection(
    *,
    denoiser: nn.Module | None,
    text_encoders: Sequence[nn.Module],
    learning_rates: LearningRatesConfig,
    groups: Sequence[LearningRateGroupConfig] | None = None,
    component_names: NamedParameterComponentNames | None = None,
) -> FinetuneSelection:
    """Resolve selected live parameters from baseline LRs plus explicit groups."""
    component_params = _collect_component_named_parameters(
        denoiser=denoiser,
        text_encoders=text_encoders,
        component_names=component_names,
    )

    denoiser_lr = learning_rates.denoiser if learning_rates.denoiser is not None else learning_rates.base
    te_lrs = [_resolve_text_encoder_lr(learning_rates, index) for index, _ in enumerate(text_encoders)]

    selected_param_ids: set[int] = set()
    if _is_positive_lr(denoiser_lr):
        selected_param_ids.update(id(ref.param) for ref in component_params.get("denoiser", []))
    for index, lr in enumerate(te_lrs):
        if _is_positive_lr(lr):
            label = f"text_encoder{index + 1}"
            selected_param_ids.update(id(ref.param) for ref in component_params.get(label, []))

    for group in groups or []:
        if group.lr <= 0:
            continue
        for ref in _resolve_group_matches(group, component_params):
            selected_param_ids.add(id(ref.param))

    selected_by_component = {
        label: [ref for ref in refs if id(ref.param) in selected_param_ids] for label, refs in component_params.items()
    }
    return FinetuneSelection(selected_by_component=selected_by_component)


def build_finetune_grouping(
    *,
    denoiser: nn.Module | None,
    train_denoiser: bool,
    text_encoders: Sequence[nn.Module],
    te_train_flags: Sequence[bool],
    learning_rates: LearningRatesConfig,
    groups: Sequence[LearningRateGroupConfig] | None = None,
    component_names: NamedParameterComponentNames | None = None,
) -> GroupingResult:
    """Build the base fine-tune logical/execution groups in trainer-facing order."""
    execution_groups: list[ParameterGroup] = []
    logical_groups: list[LogicalParameterGroup] = []
    component_params = _collect_component_named_parameters(
        denoiser=denoiser,
        text_encoders=text_encoders,
        component_names=component_names,
    )
    assigned_param_ids: set[int] = set()
    seen_group_names: set[str] = set()
    overridden_components: set[str] = set()

    for group in groups or []:
        if group.name in seen_group_names:
            raise ValueError(f"optimizer.learning_rates.groups contains duplicate group name '{group.name}'")
        seen_group_names.add(group.name)

        matched_refs = _resolve_group_matches(group, component_params)
        overlapping_names = [ref.full_name for ref in matched_refs if id(ref.param) in assigned_param_ids]
        if overlapping_names:
            raise ValueError(
                f"optimizer.learning_rates.groups '{group.name}' overlaps with already assigned parameters, first overlap: {overlapping_names[0]}"
            )

        execution_groups.append(
            build_parameter_group(
                [ref.param for ref in matched_refs],
                lr=group.lr,
                label=group.name,
                metadata={"param_names": [ref.full_name for ref in matched_refs]},
            )
        )
        logical_groups.append(
            build_logical_parameter_group(
                group.name,
                [ref.param for ref in matched_refs],
                lr=group.lr,
                label=group.name,
                execution_group_indices=(len(execution_groups) - 1,),
            )
        )

        assigned_param_ids.update(id(ref.param) for ref in matched_refs)
        overridden_components.update(ref.component_key for ref in matched_refs)

    if train_denoiser:
        assert denoiser is not None, "denoiser must be loaded before build_finetune_grouping"
        denoiser_lr = learning_rates.denoiser if learning_rates.denoiser is not None else learning_rates.base
        remaining_refs = [ref for ref in component_params.get("denoiser", []) if id(ref.param) not in assigned_param_ids]
        if _is_positive_lr(denoiser_lr) and remaining_refs:
            if "denoiser" not in overridden_components:
                parameter_group = build_module_parameter_group(denoiser, lr=denoiser_lr, label="denoiser")
            else:
                parameter_group = build_parameter_group(
                    [ref.param for ref in remaining_refs],
                    lr=denoiser_lr,
                    label="denoiser",
                    metadata={"param_names": [ref.full_name for ref in remaining_refs]},
                )
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
        remaining_refs = [ref for ref in component_params.get(label, []) if id(ref.param) not in assigned_param_ids]
        if _is_positive_lr(text_encoder_lr) and remaining_refs:
            if label not in overridden_components:
                parameter_group = build_module_parameter_group(text_encoder, lr=text_encoder_lr, label=label)
            else:
                parameter_group = build_parameter_group(
                    [ref.param for ref in remaining_refs],
                    lr=text_encoder_lr,
                    label=label,
                    metadata={"param_names": [ref.full_name for ref in remaining_refs]},
                )
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
