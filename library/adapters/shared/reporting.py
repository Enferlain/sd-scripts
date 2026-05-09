from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .trainables import get_trainable_parameter_refs


@dataclass(frozen=True, slots=True)
class AdapterComponentReportRow:
    label: str
    component_key: str
    modules_trainable: int
    modules_total: int
    adapter_modules_trainable: int
    adapter_modules_total: int
    params_trainable: int
    params_total: int
    param_bytes_trainable: int
    param_bytes_total: int


def build_adapter_component_report_rows(adapter: Any) -> list[AdapterComponentReportRow]:
    """Build adapter-owned per-component reporting rows from trainable provenance."""

    refs = get_trainable_parameter_refs(adapter)
    grouped: dict[str, dict[str, Any]] = {}

    for ref in refs:
        if ref.adapter_module_path is None:
            raise TypeError(f"AdapterTrainableParameterRef '{ref.name}' is missing adapter_module_path")

        group = grouped.setdefault(
            ref.component_key,
            {
                "label": ref.component,
                "module_paths_total": set(),
                "module_paths_trainable": set(),
                "adapter_module_paths_total": set(),
                "adapter_module_paths_trainable": set(),
                "params_total": 0,
                "params_trainable": 0,
                "param_bytes_total": 0,
                "param_bytes_trainable": 0,
            },
        )
        group["module_paths_total"].add(ref.target_path)
        group["adapter_module_paths_total"].add(ref.adapter_module_path)

        param_count = ref.param.numel()
        bytes_count = param_count * ref.param.element_size()
        group["params_total"] += param_count
        group["param_bytes_total"] += bytes_count

        if ref.param.requires_grad:
            group["module_paths_trainable"].add(ref.target_path)
            group["adapter_module_paths_trainable"].add(ref.adapter_module_path)
            group["params_trainable"] += param_count
            group["param_bytes_trainable"] += bytes_count

    rows: list[AdapterComponentReportRow] = []
    for component_key, group in grouped.items():
        rows.append(
            AdapterComponentReportRow(
                label=group["label"],
                component_key=component_key,
                modules_trainable=len(group["module_paths_trainable"]),
                modules_total=len(group["module_paths_total"]),
                adapter_modules_trainable=len(group["adapter_module_paths_trainable"]),
                adapter_modules_total=len(group["adapter_module_paths_total"]),
                params_trainable=group["params_trainable"],
                params_total=group["params_total"],
                param_bytes_trainable=group["param_bytes_trainable"],
                param_bytes_total=group["param_bytes_total"],
            )
        )

    return rows
