"""Shared metadata and YAML formatting helpers for model inspection output."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import nn
from library.models import build_selector_name


def derive_parameter_dump_identifier(model_type: str, model_version: str | None) -> str:
    """Build a readable identifier for inspection output."""
    if model_version is None or model_version == model_type:
        return model_type
    return f"{model_type}-{model_version}"




def _format_scalar_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        return value
    return str(value)


def _format_inline_record(record: dict[str, object]) -> str:
    rendered_items = [f"{key}: {_format_scalar_value(value)}" for key, value in record.items()]
    return f"{{{', '.join(rendered_items)}}}"


def _parameter_record(param: torch.nn.Parameter, *, include_kind: bool) -> dict[str, object]:
    record: dict[str, object] = {
        "shape": list(param.shape),
        "dtype": str(param.dtype).removeprefix("torch."),
        "requires_grad": bool(param.requires_grad),
    }
    if include_kind:
        record = {"kind": "parameter", **record}
    return record


def _buffer_record(module: nn.Module, name: str, buffer: torch.Tensor) -> dict[str, object]:
    return {
        "kind": "buffer",
        "shape": list(buffer.shape),
        "dtype": str(buffer.dtype).removeprefix("torch."),
        "requires_grad": bool(getattr(buffer, "requires_grad", False)),
        "persistent": _buffer_is_persistent(module, name),
    }


def _iter_component_parameter_lines(
    module: nn.Module,
    *,
    component_name: str,
    trainable_only: bool = False,
    include_kind: bool = False,
    indent: str = "    ",
) -> Iterable[str]:
    for name, param in sorted(module.named_parameters(), key=lambda item: item[0]):
        if trainable_only and not param.requires_grad:
            continue
        selector_name = build_selector_name(component_name, name)
        yield f"{indent}{selector_name}: {_format_inline_record(_parameter_record(param, include_kind=include_kind))}"


def _iter_component_buffer_lines(module: nn.Module, *, component_name: str, indent: str = "      ") -> Iterable[str]:
    for name, buffer in sorted(module.named_buffers(), key=lambda item: item[0]):
        selector_name = build_selector_name(component_name, name)
        yield f"{indent}{selector_name}: {_format_inline_record(_buffer_record(module, name, buffer))}"


def _iter_component_module_lines(module: nn.Module, *, indent: str = "      ") -> Iterable[str]:
    for name, child in module.named_modules():
        module_name = name or "<root>"
        yield f"{indent}{module_name}: {_format_inline_record({'type': child.__class__.__name__})}"


def format_named_parameter_dump(
    *,
    identifier: str,
    components: Sequence[tuple[str, nn.Module]],
    trainable_only: bool = False,
) -> str:
    """Render named parameters grouped by top-level component."""
    lines = [f"identifier: {identifier}", "components:"]

    for component_name, module in components:
        parameter_lines = list(
            _iter_component_parameter_lines(module, component_name=component_name, trainable_only=trainable_only)
        )
        if trainable_only and not parameter_lines:
            continue

        lines.append(f"  {component_name}:")
        lines.extend(parameter_lines)

    return "\n".join(lines) + "\n"


def _buffer_is_persistent(module: nn.Module, name: str) -> bool:
    module_path, _, buffer_name = name.rpartition(".")
    owner = module.get_submodule(module_path) if module_path else module
    return buffer_name not in owner._non_persistent_buffers_set


def format_component_state_dump(
    *,
    identifier: str,
    components: Sequence[tuple[str, nn.Module]],
) -> str:
    """Render parameters and buffers grouped by top-level component."""
    lines = [f"identifier: {identifier}", "components:"]

    for component_name, module in components:
        lines.append(f"  {component_name}:")
        parameter_lines = list(
            _iter_component_parameter_lines(module, component_name=component_name, include_kind=True, indent="      ")
        )
        if parameter_lines:
            lines.append("    parameters:")
            lines.extend(parameter_lines)
        else:
            lines.append("    parameters: {}")

        buffer_lines = list(_iter_component_buffer_lines(module, component_name=component_name))
        if buffer_lines:
            lines.append("    buffers:")
            lines.extend(buffer_lines)
        else:
            lines.append("    buffers: {}")

    return "\n".join(lines) + "\n"


def format_component_module_dump(
    *,
    identifier: str,
    components: Sequence[tuple[str, nn.Module]],
) -> str:
    """Render named modules and types grouped by top-level component."""
    lines = [f"identifier: {identifier}", "components:"]

    for component_name, module in components:
        lines.append(f"  {component_name}:")
        lines.append("    modules:")
        lines.extend(_iter_component_module_lines(module))

    return "\n".join(lines) + "\n"
