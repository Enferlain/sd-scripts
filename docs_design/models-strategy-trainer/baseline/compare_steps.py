"""Compare two step recordings written by ``record_steps.py``.

    uv run python compare_steps.py <recording-a> <recording-b> [report.json]

Reports, per step and per recorded value, whether the two recordings match
bit for bit and otherwise the largest absolute and relative difference. Exits
with status 1 when anything differs.

``aggregates`` in the report compares gradients and parameter updates as a
whole, per step. Element-wise differences of an Adam update say little: the
first update of every element is the learning rate with the sign of its
gradient, so rounding-level noise in a near-zero gradient flips a whole
update.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a * b).sum() / (a.norm() * b.norm()))


def aggregates(first: Path, second: Path) -> dict[str, object]:
    """Whole-step agreement of gradients and parameter updates."""
    rows: dict[str, object] = {}
    before_a = load_file(str(first / "initial.safetensors"))["param_sample"].double()
    before_b = load_file(str(second / "initial.safetensors"))["param_sample"].double()
    for path in sorted(first.glob("step_*.safetensors")):
        if not (second / path.name).exists():
            continue
        left, right = load_file(str(path)), load_file(str(second / path.name))
        norm_a = float((left["grad_l2"].double() ** 2).sum().sqrt())
        norm_b = float((right["grad_l2"].double() ** 2).sum().sqrt())
        grad_a, grad_b = left["grad_sample"].double(), right["grad_sample"].double()
        update_a = left["param_sample"].double() - before_a
        update_b = right["param_sample"].double() - before_b
        changed = (update_a != 0) | (update_b != 0)
        rows[path.stem] = {
            "loss": [float(left["final_loss"]), float(right["final_loss"])],
            "global_gradient_norm": [norm_a, norm_b],
            "global_gradient_norm_relative_difference": abs(norm_a - norm_b) / norm_a,
            "sampled_gradient_cosine": _cosine(grad_a, grad_b),
            "sampled_update_cosine": _cosine(update_a, update_b),
            "sampled_elements_updated_fraction": float(changed.double().mean()),
            "updated_elements_with_opposite_sign_fraction": float(((update_a * update_b) < 0).sum() / changed.sum()),
            "largest_sampled_update": float(update_a.abs().max()),
        }
        before_a, before_b = left["param_sample"].double(), right["param_sample"].double()
    return rows


def compare(first: Path, second: Path) -> dict[str, object]:
    report: dict[str, object] = {"a": str(first), "b": str(second), "files": {}}
    identical = True
    names = sorted(path.name for path in first.glob("*.safetensors"))
    missing = sorted(set(names) ^ {path.name for path in second.glob("*.safetensors")})
    if missing:
        report["files_only_in_one"] = missing
        identical = False
    for name in names:
        if not (second / name).exists():
            continue
        left, right = load_file(str(first / name)), load_file(str(second / name))
        rows: dict[str, object] = {}
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                rows[key] = "only in one recording"
                identical = False
                continue
            a, b = left[key], right[key]
            if a.shape != b.shape or a.dtype != b.dtype:
                rows[key] = f"shape or dtype differs: {tuple(a.shape)} {a.dtype} vs {tuple(b.shape)} {b.dtype}"
                identical = False
                continue
            if torch.equal(a, b) or (
                a.is_floating_point() and torch.equal(a.nan_to_num(nan=0.0), b.nan_to_num(nan=0.0)) and torch.equal(a.isnan(), b.isnan())
            ):
                continue
            identical = False
            difference = (a.double() - b.double()).abs()
            scale = torch.maximum(a.double().abs(), b.double().abs()).clamp_min(1e-30)
            rows[key] = {
                "max_abs": difference.max().item(),
                "max_rel": (difference / scale).max().item(),
                "differing_elements": int((difference > 0).sum()),
                "elements": a.numel(),
            }
        report["files"][name] = rows or "identical"  # type: ignore[index]
    report["identical"] = identical
    report["aggregates"] = aggregates(first, second)
    return report


if __name__ == "__main__":
    result = compare(Path(sys.argv[1]), Path(sys.argv[2]))
    text = json.dumps(result, indent=1)
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(text)
    print(text if len(text) < 6000 else text[:6000] + "\n...")
    sys.exit(0 if result["identical"] else 1)
