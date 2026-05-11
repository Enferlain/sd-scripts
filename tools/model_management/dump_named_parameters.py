"""Inspect loaded model runtimes and dump parameters, modules, or state to YAML."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path
import re
from types import SimpleNamespace
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from library.config.dataclasses.data import DataConfig
from library.config.dataclasses.model import ModelConfig
from library.config.dataclasses.performance import PerformanceConfig
from library.config.dataclasses.training import TrainingConfig
from library.models import build_component_module_pairs
from library.models.parameter_dump import (
    derive_parameter_dump_identifier,
    format_component_module_dump,
    format_component_state_dump,
    format_named_parameter_dump,
)

_WINDOWS_ABS_PATH_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")
_DTYPE_CHOICES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}
_CONTAINER_MODULE_TYPES = (
    torch.nn.ModuleList,
    torch.nn.Sequential,
    torch.nn.ModuleDict,
    torch.nn.ParameterList,
    torch.nn.ParameterDict,
)
_SUMMARY_IGNORED_TYPES = {"Identity"}


class _ToolAccelerator:
    """Small accelerator stub sufficient for strategy loading helpers."""

    def __init__(self, device: str) -> None:
        self.device = torch.device(device)
        self.state = SimpleNamespace(num_processes=1, local_process_index=0)

    def wait_for_everyone(self) -> None:
        return None


def supported_model_types() -> tuple[str, ...]:
    """Return model families supported by the active strategy path."""
    from library.strategies.factory import _STRATEGY_REGISTRY

    return tuple(_STRATEGY_REGISTRY)


def build_strategy(cfg):
    """Build the active training strategy for a minimal inspection runtime config."""
    from library.strategies.factory import build_training_strategy

    return build_training_strategy(cfg)


def validate_model_type(parser: argparse.ArgumentParser, model_type: str) -> None:
    supported_types = supported_model_types()
    if model_type in supported_types:
        return

    supported = ", ".join(supported_types)
    parser.error(f"Unsupported --model-type {model_type!r}. Supported model types: {supported}")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a loaded model runtime and emit YAML views.")
    parser.add_argument("--model-type", required=True, help="Model family to inspect")
    parser.add_argument("--model-path", required=True, help="Checkpoint path or diffusers path")
    parser.add_argument("--vae", default=None, help="Optional VAE path")
    parser.add_argument("--clip-l", default=None, help="Optional SD3 CLIP-L sidecar path")
    parser.add_argument("--clip-g", default=None, help="Optional SD3 CLIP-G sidecar path")
    parser.add_argument("--t5xxl", default=None, help="Optional SD3 T5-XXL sidecar path")
    parser.add_argument("--device", default="cpu", help="Device to load the model on for inspection")
    parser.add_argument("--dtype", default="float32", choices=tuple(_DTYPE_CHOICES), help="Weight dtype for loading")
    parser.add_argument("--disable-mmap", action="store_true", help="Disable safetensors mmap loading when supported")
    parser.add_argument(
        "--view",
        default="parameters",
        choices=("parameters", "modules", "summary", "state"),
        help="Inspection view to render",
    )
    parser.add_argument("--identifier", default=None, help="Optional identifier override for the YAML header")
    parser.add_argument("--component", action="append", default=[], help="Optional component filter, may be passed multiple times")
    parser.add_argument("--trainable-only", action="store_true", help="Only include parameters with requires_grad=True")
    parser.add_argument("--output", default=None, help="Optional output path; prints to stdout when omitted")
    return parser


def parse_args() -> argparse.Namespace:
    parser = setup_parser()
    args = parser.parse_args()

    validate_model_type(parser, args.model_type)

    if args.trainable_only and args.view != "parameters":
        parser.error("--trainable-only only applies to the parameters view")

    return args


def normalize_path(path: str | None) -> str | None:
    if not path:
        return path

    original = Path(path)
    if original.exists():
        return str(original)

    match = _WINDOWS_ABS_PATH_RE.match(path)
    if match is None:
        return path

    drive, remainder = match.groups()
    converted = Path("/mnt") / drive.lower() / remainder.replace("\\", "/")
    return str(converted) if converted.exists() else path


def mixed_precision_from_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    return "no"


def build_runtime_cfg(args: argparse.Namespace) -> SimpleNamespace:
    """Build the minimal runtime config needed to reuse the strategy loading path."""
    model_config = ModelConfig(
        model_type=args.model_type,
        pretrained_model_name_or_path=normalize_path(args.model_path),
        vae=normalize_path(args.vae),
    )
    for attr_name, value in (
        ("clip_l", args.clip_l),
        ("clip_g", args.clip_g),
        ("t5xxl", args.t5xxl),
    ):
        if value is not None:
            setattr(model_config, attr_name, normalize_path(value))

    training_config = TrainingConfig(max_token_length=None, clip_skip=None)
    data_config = DataConfig()
    data_config.caching.disable_mmap_load_safetensors = bool(args.disable_mmap)
    performance_config = PerformanceConfig()
    performance_config.precision.mixed_precision = mixed_precision_from_dtype(_DTYPE_CHOICES[args.dtype])

    return SimpleNamespace(
        mode="finetune",
        model=model_config,
        training=training_config,
        data=data_config,
        performance=performance_config,
        objective=SimpleNamespace(prediction="epsilon"),
    )


def load_components(args: argparse.Namespace) -> tuple[str, list[tuple[str, torch.nn.Module]]]:
    """Load the model through the active strategy path and group top-level components."""
    cfg = build_runtime_cfg(args)
    strategy = build_strategy(cfg)
    accelerator = _ToolAccelerator(args.device)
    weight_dtype = _DTYPE_CHOICES[args.dtype]
    model_version, loaded_components = strategy.load_target_model(cfg, weight_dtype, accelerator)
    components = build_component_module_pairs(loaded_components)
    identifier = args.identifier or derive_parameter_dump_identifier(args.model_type, model_version)
    return identifier, components


def filter_components(
    components: list[tuple[str, torch.nn.Module]],
    requested_components: list[str],
) -> list[tuple[str, torch.nn.Module]]:
    if not requested_components:
        return components

    requested = set(requested_components)
    return [(name, module) for name, module in components if name in requested]


def render_view(
    *,
    view: str,
    identifier: str,
    components: list[tuple[str, torch.nn.Module]],
    trainable_only: bool,
) -> str:
    if view == "modules":
        return format_component_module_dump(identifier=identifier, components=components)
    if view == "summary":
        return format_component_summary_dump(identifier=identifier, components=components)
    if view == "state":
        return format_component_state_dump(identifier=identifier, components=components)
    return format_named_parameter_dump(identifier=identifier, components=components, trainable_only=trainable_only)


def _is_container_module(module: torch.nn.Module) -> bool:
    return isinstance(module, _CONTAINER_MODULE_TYPES)


def _iter_summary_child_modules(module: torch.nn.Module):
    for child in module.children():
        if _is_container_module(child):
            yield from _iter_summary_child_modules(child)
            continue
        yield child


def build_component_summary(
    components: list[tuple[str, torch.nn.Module]],
) -> dict[str, dict[str, list[str]]]:
    summary: dict[str, dict[str, list[str]]] = {}

    for component_name, module in components:
        component_summary: dict[str, list[str]] = {}

        for _, child_module in module.named_modules():
            if _is_container_module(child_module):
                continue

            child_types: list[str] = []
            for summary_child in _iter_summary_child_modules(child_module):
                child_type = summary_child.__class__.__name__
                if child_type in _SUMMARY_IGNORED_TYPES or child_type in child_types:
                    continue
                child_types.append(child_type)

            if not child_types:
                continue

            composite_type = child_module.__class__.__name__
            existing_types = component_summary.setdefault(composite_type, [])
            for child_type in child_types:
                if child_type not in existing_types:
                    existing_types.append(child_type)

        summary[component_name] = component_summary

    return summary


def format_component_summary_dump(
    *,
    identifier: str,
    components: list[tuple[str, torch.nn.Module]],
) -> str:
    lines = [f"identifier: {identifier}", "components:"]
    summary = build_component_summary(components)

    for component_name, composite_summary in summary.items():
        if not composite_summary:
            lines.append(f"  {component_name}: {{}}")
            continue

        lines.append(f"  {component_name}:")
        for composite_type, child_types in composite_summary.items():
            lines.append(f"    {composite_type}:")
            for child_type in child_types:
                lines.append(f"      - {child_type}")

    return "\n".join(lines) + "\n"


def write_output(rendered: str, output_path: str | None) -> None:
    if output_path is None:
        print(rendered, end="")
        return

    resolved_output = Path(output_path).expanduser()
    if not resolved_output.is_absolute():
        resolved_output = Path.cwd() / resolved_output
    resolved_output.write_text(rendered, encoding="utf-8")


def main() -> None:
    args = parse_args()
    identifier, components = load_components(args)
    filtered_components = filter_components(components, args.component)
    rendered = render_view(
        view=args.view,
        identifier=identifier,
        components=filtered_components,
        trainable_only=args.trainable_only,
    )
    write_output(rendered, args.output)


if __name__ == "__main__":
    main()
