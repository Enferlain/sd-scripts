from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from torch import nn

from library.adapters.shared.reporting import build_adapter_component_report_rows
from library.models import resolve_component_names
from library.optimization.types import OptimizationPlan


@dataclass(frozen=True, slots=True)
class DiagnosticAlias:
    alias: str
    target: str


@dataclass(frozen=True, slots=True)
class DiagnosticModuleCount:
    label: str
    trainable: int
    total: int


@dataclass(frozen=True, slots=True)
class DiagnosticRow:
    label: str
    component_key: str
    modules_trainable: int
    modules_total: int
    params_trainable: int
    params_total: int
    extra_module_counts: tuple[DiagnosticModuleCount, ...] = ()
    param_bytes_trainable: int = 0
    param_bytes_total: int = 0


@dataclass(frozen=True, slots=True)
class OptimizerRow:
    label: str
    parameter_count: int
    learning_rate: float | None


@dataclass(frozen=True, slots=True)
class TrainingStartupSummary:
    runtime_rows: list[tuple[str, str]]
    dataset_rows: list[tuple[str, str]]
    schedule_rows: list[tuple[str, str]]
    component_rows: list[DiagnosticRow]
    optimizer_name: str
    optimizer_rows: list[OptimizerRow]
    aliases: list[DiagnosticAlias]


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _iter_parameterized_leaf_modules(module: nn.Module):
    for child in module.modules():
        if any(child.children()):
            continue
        child_params = list(child.parameters(recurse=False))
        if child_params:
            yield child, child_params


def summarize_component_trainability(module: nn.Module) -> dict[str, int]:
    total_modules = 0
    trainable_modules = 0
    for _mod, mod_params in _iter_parameterized_leaf_modules(module):
        total_modules += 1
        if any(p.requires_grad for p in mod_params):
            trainable_modules += 1

    total_params = 0
    trainable_params = 0
    for param in module.parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count

    return {
        "modules_total": total_modules,
        "modules_trainable": trainable_modules,
        "params_total": total_params,
        "params_trainable": trainable_params,
    }


def build_public_component_name_map(model_type: str | None, *, text_encoder_count: int) -> dict[str, str]:
    if not model_type:
        return {}

    component_names = resolve_component_names(model_type)
    if component_names is None:
        return {}

    mapping = {"denoiser": component_names.denoiser_name, "vae": component_names.vae_name}
    for index in range(text_encoder_count):
        key = f"text_encoder{index + 1}"
        if index < len(component_names.text_encoder_names):
            mapping[key] = component_names.text_encoder_names[index]
    return mapping


def build_diagnostic_rows_from_components(
    components: list[tuple[str, nn.Module]],
    *,
    public_name_map: dict[str, str] | None = None,
) -> list[DiagnosticRow]:
    rows: list[DiagnosticRow] = []
    name_map = public_name_map or {}

    for component_key, module in components:
        stats = summarize_component_trainability(module)
        param_bytes_total = 0
        param_bytes_trainable = 0
        for param in module.parameters():
            bytes_count = param.numel() * param.element_size()
            param_bytes_total += bytes_count
            if param.requires_grad:
                param_bytes_trainable += bytes_count

        rows.append(
            DiagnosticRow(
                label=name_map.get(component_key, component_key),
                component_key=component_key,
                modules_trainable=stats["modules_trainable"],
                modules_total=stats["modules_total"],
                params_trainable=stats["params_trainable"],
                params_total=stats["params_total"],
                param_bytes_trainable=param_bytes_trainable,
                param_bytes_total=param_bytes_total,
            )
        )

    return rows


def build_adapter_diagnostic_rows(adapter: Any) -> list[DiagnosticRow]:
    rows: list[DiagnosticRow] = []
    for report_row in build_adapter_component_report_rows(adapter):
        rows.append(
            DiagnosticRow(
                label=report_row.label,
                component_key=report_row.component_key,
                modules_trainable=report_row.modules_trainable,
                modules_total=report_row.modules_total,
                params_trainable=report_row.params_trainable,
                params_total=report_row.params_total,
                extra_module_counts=(
                    DiagnosticModuleCount(
                        label="adapter modules",
                        trainable=report_row.adapter_modules_trainable,
                        total=report_row.adapter_modules_total,
                    ),
                ),
                param_bytes_trainable=report_row.param_bytes_trainable,
                param_bytes_total=report_row.param_bytes_total,
            )
        )
    return rows


def build_total_diagnostic_row(rows: Iterable[DiagnosticRow]) -> DiagnosticRow:
    row_list = list(rows)
    extra_counts_by_label: dict[str, DiagnosticModuleCount] = {}
    for row in row_list:
        for count in row.extra_module_counts:
            existing = extra_counts_by_label.get(count.label)
            if existing is None:
                extra_counts_by_label[count.label] = DiagnosticModuleCount(
                    label=count.label,
                    trainable=count.trainable,
                    total=count.total,
                )
            else:
                extra_counts_by_label[count.label] = DiagnosticModuleCount(
                    label=count.label,
                    trainable=existing.trainable + count.trainable,
                    total=existing.total + count.total,
                )
    return DiagnosticRow(
        label="total",
        component_key="total",
        modules_trainable=sum(row.modules_trainable for row in row_list),
        modules_total=sum(row.modules_total for row in row_list),
        params_trainable=sum(row.params_trainable for row in row_list),
        params_total=sum(row.params_total for row in row_list),
        extra_module_counts=tuple(extra_counts_by_label.values()),
        param_bytes_trainable=sum(row.param_bytes_trainable for row in row_list),
        param_bytes_total=sum(row.param_bytes_total for row in row_list),
    )


def build_optimizer_rows(
    optimizer: Any,
    *,
    optimization_plan: OptimizationPlan | None = None,
    lr_descriptions: list[str] | None = None,
    public_name_map: dict[str, str] | None = None,
) -> list[OptimizerRow]:
    rows: list[OptimizerRow] = []
    label_map = public_name_map or {}

    if optimization_plan is not None and optimization_plan.logical_groups:
        runtime_groups = getattr(optimizer, "param_groups", [])
        for index, logical_group in enumerate(optimization_plan.logical_groups):
            learning_rate = logical_group.lr
            if logical_group.execution_group_indices and logical_group.execution_group_indices[0] < len(runtime_groups):
                learning_rate = runtime_groups[logical_group.execution_group_indices[0]].get("lr", learning_rate)

            label = logical_group.metric_name if logical_group.metric_name else f"group {index}"
            rows.append(
                OptimizerRow(
                    label=label_map.get(label, label),
                    parameter_count=logical_group.parameter_count,
                    learning_rate=learning_rate,
                )
            )
        return rows

    runtime_descriptions = list(lr_descriptions or [])
    for index, group in enumerate(getattr(optimizer, "param_groups", [])):
        label = runtime_descriptions[index] if index < len(runtime_descriptions) else f"group {index}"
        parameter_count = sum(param.numel() for param in group["params"] if isinstance(param, nn.Parameter))
        rows.append(
            OptimizerRow(
                label=label_map.get(label, label),
                parameter_count=parameter_count,
                learning_rate=group.get("lr"),
            )
        )
    return rows


def build_trainer_diagnostic_rows(trainer: Any) -> tuple[list[DiagnosticRow], list[DiagnosticAlias]]:
    adapter = getattr(trainer, "adapter", None)
    if adapter is not None and hasattr(adapter, "describe_trainable_parameter_refs"):
        return build_adapter_diagnostic_rows(adapter), []

    diagnostics = trainer.mode.get_diagnostics_components(trainer)
    if isinstance(diagnostics, tuple) and len(diagnostics) == 2:
        components, aliases = diagnostics
    else:
        components = diagnostics or []
        aliases = None
    model_cfg = getattr(getattr(trainer, "cfg", None), "model", None)
    public_name_map = build_public_component_name_map(
        getattr(model_cfg, "model_type", None),
        text_encoder_count=len(getattr(trainer, "text_encoders", [])),
    )
    alias_rows = [DiagnosticAlias(alias=alias, target=target) for alias, target in (aliases or [])]
    return build_diagnostic_rows_from_components(components, public_name_map=public_name_map), alias_rows


def build_training_startup_summary(
    *,
    trainer: Any,
    component_rows: list[DiagnosticRow],
    aliases: list[DiagnosticAlias],
) -> TrainingStartupSummary:
    cfg = trainer.cfg
    num_train_images = sum(entry.num_repeats for entry in trainer.train_manifest.entries.values() if not entry.is_reg)
    num_reg_images = sum(entry.num_repeats for entry in trainer.train_manifest.entries.values() if entry.is_reg)
    num_val_images = sum(entry.num_repeats for entry in trainer.val_manifest.entries.values()) if trainer.val_manifest else 0

    effective_batch = cfg.training.train_batch_size * trainer.accelerator.num_processes * cfg.training.gradient_accumulation_steps
    public_name_map = {row.component_key: row.label for row in component_rows}

    return TrainingStartupSummary(
        runtime_rows=[
            ("mode", type(trainer.mode).__name__),
            *((("method", trainer.adapter_method_name),) if getattr(trainer, "adapter_method_name", None) else ()),
            ("strategy", type(trainer.strategies).__name__),
            ("precision", _format_scalar(cfg.performance.precision.mixed_precision)),
            ("grad_ckpt", _format_scalar(cfg.performance.memory.gradient_checkpointing)),
            ("xformers", _format_scalar(cfg.performance.attention.xformers)),
            ("deepspeed", _format_scalar(cfg.performance.deepspeed.deepspeed)),
        ],
        dataset_rows=[
            ("train images", f"{num_train_images:,}"),
            ("validation images", f"{num_val_images:,}"),
            ("reg images", f"{num_reg_images:,}"),
        ],
        schedule_rows=[
            ("batches/epoch", f"{trainer.num_batches_per_epoch:,}"),
            ("epochs", f"{trainer.num_train_epochs:,}"),
            ("batch/device", f"{cfg.training.train_batch_size:,}"),
            ("grad accum", f"{cfg.training.gradient_accumulation_steps:,}"),
            ("effective batch", f"{effective_batch:,}"),
            ("total steps", f"{trainer.max_train_steps:,}"),
        ],
        component_rows=component_rows,
        optimizer_name=trainer.optimizer_name,
        optimizer_rows=build_optimizer_rows(
            trainer.optimizer,
            optimization_plan=trainer.optimization_plan,
            lr_descriptions=None if trainer.optimization_plan is not None else trainer.lr_descriptions,
            public_name_map=public_name_map,
        ),
        aliases=aliases,
    )


def render_training_startup_summary(summary: TrainingStartupSummary) -> str:
    lines: list[str] = []

    def add_section(title: str, rows: list[tuple[str, str]]) -> None:
        if not rows:
            return
        lines.append(title)
        label_width = max(len(label) for label, _ in rows)
        for label, value in rows:
            lines.append(f"  {label:<{label_width}}: {value}")
        lines.append("")

    add_section("training run", summary.runtime_rows)
    add_section("dataset", summary.dataset_rows)
    add_section("schedule", summary.schedule_rows)

    if summary.component_rows:
        lines.append("components")
        all_rows = [*summary.component_rows, build_total_diagnostic_row(summary.component_rows)]
        label_width = max(len("component"), *(len(row.label) for row in all_rows))
        module_trainable_width = max(len(f"{row.modules_trainable:,}") for row in all_rows)
        module_total_width = max(len(f"{row.modules_total:,}") for row in all_rows)
        extra_module_labels: list[str] = []
        extra_module_trainable_widths: dict[str, int] = {}
        extra_module_total_widths: dict[str, int] = {}
        for row in all_rows:
            for count in row.extra_module_counts:
                if count.label not in extra_module_labels:
                    extra_module_labels.append(count.label)
                extra_module_trainable_widths[count.label] = max(
                    extra_module_trainable_widths.get(count.label, 0),
                    len(f"{count.trainable:,}"),
                )
                extra_module_total_widths[count.label] = max(
                    extra_module_total_widths.get(count.label, 0),
                    len(f"{count.total:,}"),
                )
        module_width = max(len("modules"), module_trainable_width + 1 + module_total_width)
        param_trainable_width = max(len(f"{row.params_trainable:,}") for row in all_rows)
        param_total_width = max(len(f"{row.params_total:,}") for row in all_rows)
        param_width = max(len("params"), param_trainable_width + 1 + param_total_width)
        status_values: list[str] = []
        for row in all_rows:
            status_values.append("frozen" if row.params_trainable == 0 else f"{(row.params_trainable / row.params_total) * 100:.1f}%")
        status_width = max(len("trainable"), *(len(value) for value in status_values))

        header_columns = [
            ("component", label_width, "<"),
            ("modules", module_width, ">"),
            *[
                (
                    extra_label,
                    max(
                        len(extra_label),
                        extra_module_trainable_widths[extra_label] + 1 + extra_module_total_widths[extra_label],
                    ),
                    ">",
                )
                for extra_label in extra_module_labels
            ],
            ("params", param_width, "<"),
            ("trainable", status_width, ">"),
        ]

        header = "  " + " | ".join(
            f"{label:<{width}}" if align == "<" else f"{label:>{width}}" for label, width, align in header_columns
        )
        separator = "  " + "-+-".join("-" * width for _, width, _ in header_columns)
        lines.append(header)
        lines.append(separator)

        for row in summary.component_rows:
            status = "frozen" if row.params_trainable == 0 else f"{(row.params_trainable / row.params_total) * 100:.1f}%"
            rendered_columns = [
                f"{row.label:<{label_width}}",
                f"{row.modules_trainable:>{module_trainable_width},}/{row.modules_total:>{module_total_width},}".rjust(module_width),
            ]
            extra_counts = {count.label: count for count in row.extra_module_counts}
            for extra_label in extra_module_labels:
                count = extra_counts.get(extra_label)
                extra_width = max(
                    len(extra_label),
                    extra_module_trainable_widths[extra_label] + 1 + extra_module_total_widths[extra_label],
                )
                value = (
                    ""
                    if count is None
                    else f"{count.trainable:>{extra_module_trainable_widths[extra_label]},}/{count.total:>{extra_module_total_widths[extra_label]},}"
                )
                rendered_columns.append(value.rjust(extra_width))
            rendered_columns.append(
                f"{row.params_trainable:>{param_trainable_width},}/{row.params_total:>{param_total_width},}".ljust(param_width)
            )
            rendered_columns.append(status.rjust(status_width))
            lines.append("  " + " | ".join(rendered_columns))

        total_row = build_total_diagnostic_row(summary.component_rows)
        total_status = "0.0%" if total_row.params_total == 0 else f"{(total_row.params_trainable / total_row.params_total) * 100:.1f}%"
        total_rendered_columns = [
            f"{total_row.label:<{label_width}}",
            f"{total_row.modules_trainable:>{module_trainable_width},}/{total_row.modules_total:>{module_total_width},}".rjust(module_width),
        ]
        total_extra_counts = {count.label: count for count in total_row.extra_module_counts}
        for extra_label in extra_module_labels:
            count = total_extra_counts.get(extra_label)
            extra_width = max(
                len(extra_label),
                extra_module_trainable_widths[extra_label] + 1 + extra_module_total_widths[extra_label],
            )
            value = (
                ""
                if count is None
                else f"{count.trainable:>{extra_module_trainable_widths[extra_label]},}/{count.total:>{extra_module_total_widths[extra_label]},}"
            )
            total_rendered_columns.append(value.rjust(extra_width))
        total_rendered_columns.append(
            f"{total_row.params_trainable:>{param_trainable_width},}/{total_row.params_total:>{param_total_width},}".ljust(param_width)
        )
        total_rendered_columns.append(total_status.rjust(status_width))
        lines.append("  " + " | ".join(total_rendered_columns))
        if summary.aliases:
            alias_text = ", ".join(f"{alias.alias} -> {alias.target}" for alias in summary.aliases)
            lines.append(f"  aliases: {alias_text}")
        lines.append("")

    lines.append("optimizer")
    lines.append(f"  {summary.optimizer_name}")
    for row in summary.optimizer_rows:
        lr_text = "?" if row.learning_rate is None else f"{row.learning_rate:g}"
        lines.append(f"  - {row.label}: lr={lr_text}, params={row.parameter_count:,}")

    return "\n".join(lines)


def diagnostic_rows_to_memory_rows(rows: list[DiagnosticRow], optimizer_name: str) -> list[dict[str, Any]]:
    optimizer_state_multiplier = 2 if "adam" in optimizer_name.lower() or "lion" in optimizer_name.lower() else 1
    rendered_rows: list[dict[str, Any]] = []
    for row in rows:
        rendered_rows.append(
            {
                "name": row.label,
                "modules_total": row.modules_total,
                "modules_trainable": row.modules_trainable,
                "params_total": row.params_total,
                "params_trainable": row.params_trainable,
                "param_mb": row.param_bytes_total / (1024 * 1024),
                "trainable_mb": row.param_bytes_trainable / (1024 * 1024),
                "gradient_mb_est": row.param_bytes_trainable / (1024 * 1024),
                "optimizer_state_mb_est": (row.param_bytes_trainable * optimizer_state_multiplier) / (1024 * 1024),
            }
        )
        for count in row.extra_module_counts:
            key = count.label.replace(" ", "_")
            rendered_rows[-1][f"{key}_total"] = count.total
            rendered_rows[-1][f"{key}_trainable"] = count.trainable
    return rendered_rows
