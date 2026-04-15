"""Dump live model named parameters grouped by component in a compact YAML shape."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from types import SimpleNamespace
import sys

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _ToolAccelerator:
    """Small accelerator stub sufficient for strategy loading helpers."""

    def __init__(self, device: str) -> None:
        self.device = torch.device(device)
        self.state = SimpleNamespace(num_processes=1, local_process_index=0)

    def wait_for_everyone(self) -> None:
        return None


_WINDOWS_ABS_PATH_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump live model named parameters grouped by component.")
    parser.add_argument("--config-name", required=True, help="Hydra config name, for example presets/sdxl_finetune")
    parser.add_argument("--override", action="append", default=[], help="Hydra override, may be passed multiple times")
    parser.add_argument("--device", default="cpu", help="Device to load the model on for inspection")
    parser.add_argument("--identifier", default=None, help="Optional identifier override for the YAML header")
    parser.add_argument("--component", action="append", default=[], help="Optional component filter, may be passed multiple times")
    parser.add_argument("--trainable-only", action="store_true", help="Only include parameters with requires_grad=True")
    parser.add_argument("--output", default=None, help="Optional output path; prints to stdout when omitted")
    return parser.parse_args()


def _load_repo_dependencies():
    from library.config.config_validation import prepare_config
    from library.config.schemas import register_all
    from library.models.parameter_dump import (
        derive_parameter_dump_identifier,
        format_named_parameter_dump,
        resolve_named_parameter_components,
    )
    from library.strategies.factory import build_training_strategy
    from library.utils.torch_utils import prepare_dtype

    return {
        "prepare_config": prepare_config,
        "register_all": register_all,
        "derive_parameter_dump_identifier": derive_parameter_dump_identifier,
        "format_named_parameter_dump": format_named_parameter_dump,
        "resolve_named_parameter_components": resolve_named_parameter_components,
        "build_training_strategy": build_training_strategy,
        "prepare_dtype": prepare_dtype,
    }


def _normalize_windows_path(path: str | None) -> str | None:
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


def _normalize_model_paths(cfg) -> None:
    model_cfg = cfg.model
    model_cfg.pretrained_model_name_or_path = _normalize_windows_path(model_cfg.pretrained_model_name_or_path)
    model_cfg.vae = _normalize_windows_path(model_cfg.vae)

    for attr in ("clip_l", "clip_g", "t5xxl"):
        if hasattr(model_cfg, attr):
            setattr(model_cfg, attr, _normalize_windows_path(getattr(model_cfg, attr)))


def main() -> None:
    args = _parse_args()
    deps = _load_repo_dependencies()

    config_dir = REPO_ROOT / "configs"

    deps["register_all"]()
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=args.config_name, overrides=args.override)
    GlobalHydra.instance().clear()

    deps["prepare_config"](cfg)
    _normalize_model_paths(cfg)

    strategy = deps["build_training_strategy"](cfg)
    accelerator = _ToolAccelerator(args.device)
    weight_dtype, _ = deps["prepare_dtype"](cfg.performance.precision)
    model_version, text_encoders, vae, denoiser = strategy.load_target_model(cfg, weight_dtype, accelerator)

    components = deps["resolve_named_parameter_components"](
        model_type=cfg.model.model_type,
        text_encoders=text_encoders,
        vae=vae,
        denoiser=denoiser,
    )
    if args.component:
        requested = set(args.component)
        components = [(name, module) for name, module in components if name in requested]

    identifier = args.identifier or deps["derive_parameter_dump_identifier"](cfg.model.model_type, model_version)
    rendered = deps["format_named_parameter_dump"](
        identifier=identifier,
        components=components,
        trainable_only=args.trainable_only,
    )

    if args.output:
        output_path = Path(args.output).expanduser()
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        output_path.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
