"""Repo-owned LoHa (LoRA-Hadamard Product) method implementation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


SUPPORTED_MODULE_TYPES = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
_LOHA_EXPORT_WEIGHT_KEYS = (
    "hada_w1_a",
    "hada_w1_b",
    "hada_w2_a",
    "hada_w2_b",
    "hada_t1",
    "hada_t2",
    "alpha",
    "dora_scale",
)
_LOHA_DETECTION_KEYS = ("hada_w1_a",)


@dataclass(frozen=True, slots=True)
class LohaConfig:
    """Repo-owned config for constructing a LoHa module on one target module."""

    multiplier: float = 1.0
    lora_dim: int | None = None
    alpha: float | Tensor | None = 1
    dropout: float = 0.0
    rank_dropout: float = 0.0
    module_dropout: float = 0.0
    use_tucker: bool = False
    use_scalar: bool = False
    rank_dropout_scale: bool = False
    weight_decompose: bool = False
    wd_on_output: bool = True
    bypass_mode: bool | None = None
    rs_lora: bool = False

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "multiplier": self.multiplier,
            "lora_dim": self.lora_dim,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "rank_dropout": self.rank_dropout,
            "module_dropout": self.module_dropout,
            "use_tucker": self.use_tucker,
            "use_scalar": self.use_scalar,
            "rank_dropout_scale": self.rank_dropout_scale,
            "weight_decompose": self.weight_decompose,
            "wd_on_output": self.wd_on_output,
            "bypass_mode": self.bypass_mode,
            "rs_lora": self.rs_lora,
        }


def _reshape_diff_weight(diff_weight: Tensor, shape: tuple[int, ...] | None) -> Tensor:
    if shape is None:
        return diff_weight
    return diff_weight.reshape(shape)


def _build_loha_diff_weight(
    hada_w1_a: Tensor,
    hada_w1_b: Tensor,
    hada_w2_a: Tensor,
    hada_w2_b: Tensor,
    hada_t1: Tensor | None,
    hada_t2: Tensor | None,
    gamma: Tensor,
) -> Tensor:
    if hada_t1 is not None and hada_t2 is not None:
        rebuild1 = torch.einsum("i j ..., j r, i p -> p r ...", hada_t1, hada_w1_b, hada_w1_a)
        rebuild2 = torch.einsum("i j ..., j r, i p -> p r ...", hada_t2, hada_w2_b, hada_w2_a)
        return rebuild1 * rebuild2 * gamma

    return ((hada_w1_a @ hada_w1_b) * (hada_w2_a @ hada_w2_b)) * gamma


class LohaModule(nn.Module):
    """LoHa method module bound to one resolved target module.

    This class owns the LoHa-specific parameter layout, diff-weight
    reconstruction, and export/load behavior for a single target module.
    The surrounding runtime is responsible for orchestration across many
    resolved targets.
    """

    export_weight_keys = _LOHA_EXPORT_WEIGHT_KEYS
    detection_keys = _LOHA_DETECTION_KEYS

    @classmethod
    def from_target_module(cls, lora_name: str, org_module: nn.Module, *, config: LohaConfig) -> LohaModule:
        """Build a LoHa module from a target module plus repo-owned config."""

        return cls(lora_name, org_module, **config.as_kwargs())

    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,
        *,
        multiplier: float = 1.0,
        lora_dim: int | None = None,
        alpha: float | Tensor | None = 1,
        dropout: float = 0.0,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        use_tucker: bool = False,
        use_scalar: bool = False,
        rank_dropout_scale: bool = False,
        weight_decompose: bool = False,
        wd_on_output: bool = True,
        bypass_mode: bool | None = None,
        rs_lora: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()
        if not isinstance(org_module, SUPPORTED_MODULE_TYPES):
            raise ValueError(f"{type(org_module).__name__} is not supported in LoHa algo.")
        if lora_dim is None or lora_dim <= 0:
            raise ValueError(f"LoHa rank must be positive, got {lora_dim}.")
        if weight_decompose and bypass_mode:
            raise ValueError("LoHa weight_decompose is incompatible with bypass_mode.")

        self.lora_name = lora_name
        self.multiplier = multiplier
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.rank_dropout_scale = rank_dropout_scale
        self.bypass_mode = bool(bypass_mode)
        self.rs_lora = rs_lora
        self.org_module = [org_module]
        self.org_forward = org_module.forward
        self.adapter_target = None

        if isinstance(org_module, nn.Linear):
            self.module_type = "linear"
            self.shape = (org_module.out_features, org_module.in_features)
            self.op = F.linear
            self.kw_dict: dict[str, Any] = {}
        elif isinstance(org_module, nn.Conv1d):
            self.module_type = "conv1d"
            self.shape = (org_module.out_channels, org_module.in_channels, *org_module.kernel_size)
            self.op = F.conv1d
            self.kw_dict = {
                "stride": org_module.stride,
                "padding": org_module.padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups,
            }
        elif isinstance(org_module, nn.Conv2d):
            self.module_type = "conv2d"
            self.shape = (org_module.out_channels, org_module.in_channels, *org_module.kernel_size)
            self.op = F.conv2d
            self.kw_dict = {
                "stride": org_module.stride,
                "padding": org_module.padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups,
            }
        else:
            self.module_type = "conv3d"
            self.shape = (org_module.out_channels, org_module.in_channels, *org_module.kernel_size)
            self.op = F.conv3d
            self.kw_dict = {
                "stride": org_module.stride,
                "padding": org_module.padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups,
            }

        self.lora_dim = lora_dim
        self.tucker = False

        weight_shape = self.shape
        if self.module_type.startswith("conv"):
            in_dim = org_module.in_channels
            kernel_size = org_module.kernel_size
            out_dim = org_module.out_channels
            self.shape = (out_dim, in_dim, *kernel_size)
            self.tucker = use_tucker and any(size != 1 for size in kernel_size)
            if self.tucker:
                weight_shape = self.shape
            else:
                weight_shape = (out_dim, in_dim * math.prod(kernel_size))

        if self.tucker:
            self.hada_t1 = nn.Parameter(torch.empty(lora_dim, lora_dim, *weight_shape[2:]))
            self.hada_w1_a = nn.Parameter(torch.empty(lora_dim, weight_shape[0]))
            self.hada_w1_b = nn.Parameter(torch.empty(lora_dim, weight_shape[1]))
            self.hada_t2 = nn.Parameter(torch.empty(lora_dim, lora_dim, *weight_shape[2:]))
            self.hada_w2_a = nn.Parameter(torch.empty(lora_dim, weight_shape[0]))
            self.hada_w2_b = nn.Parameter(torch.empty(lora_dim, weight_shape[1]))
        else:
            self.hada_w1_a = nn.Parameter(torch.empty(weight_shape[0], lora_dim))
            self.hada_w1_b = nn.Parameter(torch.empty(lora_dim, weight_shape[1]))
            self.hada_w2_a = nn.Parameter(torch.empty(weight_shape[0], lora_dim))
            self.hada_w2_b = nn.Parameter(torch.empty(lora_dim, weight_shape[1]))
            self.register_parameter("hada_t1", None)
            self.register_parameter("hada_t2", None)

        self.wd = weight_decompose
        self.wd_on_output = wd_on_output
        if self.wd:
            org_weight = org_module.weight.detach().cpu().float()
            self.dora_norm_dims = org_weight.dim() - 1
            if self.wd_on_output:
                scale = torch.norm(org_weight.reshape(org_weight.shape[0], -1), dim=1, keepdim=True)
                scale = scale.reshape(org_weight.shape[0], *[1] * self.dora_norm_dims)
            else:
                scale = torch.norm(org_weight.transpose(1, 0).reshape(org_weight.shape[1], -1), dim=1, keepdim=True)
                scale = scale.reshape(org_weight.shape[1], *[1] * self.dora_norm_dims).transpose(1, 0)
            self.dora_scale = nn.Parameter(scale.float())
        else:
            self.register_parameter("dora_scale", None)

        if isinstance(alpha, Tensor):
            alpha = float(alpha.detach().float().item())
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        r_factor = math.sqrt(lora_dim) if self.rs_lora else lora_dim
        self.scale = alpha / r_factor
        self.register_buffer("alpha", torch.tensor(alpha * (lora_dim / r_factor), dtype=torch.float32))

        if use_scalar:
            self.scalar = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("scalar", torch.tensor(1.0), persistent=False)

        # These initializers intentionally mirror the established LyCORIS LoHa
        # defaults so the absorbed repo-owned method keeps the same starting
        # behavior while we migrate away from vendored code.
        if self.tucker:
            nn.init.normal_(self.hada_t1, std=0.1)
            nn.init.normal_(self.hada_t2, std=0.1)
        nn.init.normal_(self.hada_w1_b, std=1.0)
        nn.init.normal_(self.hada_w1_a, std=0.1)
        nn.init.normal_(self.hada_w2_b, std=1.0)
        if use_scalar:
            nn.init.normal_(self.hada_w2_a, std=0.1)
        else:
            nn.init.constant_(self.hada_w2_a, 0.0)

    @property
    def dtype(self) -> torch.dtype:
        return self.hada_w1_a.dtype

    @property
    def device(self) -> torch.device:
        return self.hada_w1_a.device

    @property
    def required_export_weight_keys(self) -> tuple[str, ...]:
        required_keys = [
            "hada_w1_a",
            "hada_w1_b",
            "hada_w2_a",
            "hada_w2_b",
            "alpha",
        ]
        if self.tucker:
            required_keys.extend(("hada_t1", "hada_t2"))
        if self.wd:
            required_keys.append("dora_scale")
        return tuple(required_keys)

    def _reset_scalar_to_identity(self) -> None:
        if getattr(self, "scalar", None) is None:
            return
        identity = torch.ones_like(self.scalar)
        with torch.no_grad():
            self.scalar.copy_(identity)

    @classmethod
    def algo_check(cls, state_dict: Mapping[str, Tensor], lora_name: str) -> bool:
        return any(f"{lora_name}.{key}" in state_dict for key in cls.detection_keys)

    @classmethod
    def extract_state_dict(cls, state_dict: Mapping[str, Tensor], lora_name: str) -> list[Tensor | None]:
        return [state_dict.get(f"{lora_name}.{key}", None) for key in cls.export_weight_keys]

    @classmethod
    def make_module_from_state_dict(
        cls,
        lora_name: str,
        orig_module: nn.Module,
        hada_w1_a: Tensor,
        hada_w1_b: Tensor,
        hada_w2_a: Tensor,
        hada_w2_b: Tensor,
        hada_t1: Tensor | None,
        hada_t2: Tensor | None,
        alpha: Tensor,
        dora_scale: Tensor | None,
    ) -> LohaModule:
        config = LohaConfig(
            multiplier=1.0,
            lora_dim=hada_w1_b.size(0),
            alpha=float(alpha.detach().float().item()),
            use_tucker=hada_t1 is not None,
            weight_decompose=dora_scale is not None,
        )
        module = cls.from_target_module(lora_name, orig_module, config=config)
        with torch.no_grad():
            module.hada_w1_a.copy_(hada_w1_a)
            module.hada_w1_b.copy_(hada_w1_b)
            module.hada_w2_a.copy_(hada_w2_a)
            module.hada_w2_b.copy_(hada_w2_b)
            if hada_t1 is not None and module.hada_t1 is not None and module.hada_t2 is not None:
                module.hada_t1.copy_(hada_t1)
                module.hada_t2.copy_(hada_t2)
            if dora_scale is not None and module.dora_scale is not None:
                module.dora_scale.copy_(dora_scale)
        module._reset_scalar_to_identity()
        return module

    def export_state_dict(self) -> dict[str, Tensor]:
        """Export the repo-owned LoHa checkpoint payload for this module."""

        state = {
            "alpha": self.alpha.detach(),
            "hada_w1_a": self.hada_w1_a.detach() * self.scalar.detach().to(self.hada_w1_a.device),
            "hada_w1_b": self.hada_w1_b.detach(),
            "hada_w2_a": self.hada_w2_a.detach(),
            "hada_w2_b": self.hada_w2_b.detach(),
        }
        if self.tucker and self.hada_t1 is not None and self.hada_t2 is not None:
            state["hada_t1"] = self.hada_t1.detach()
            state["hada_t2"] = self.hada_t2.detach()
        if self.wd and self.dora_scale is not None:
            state["dora_scale"] = self.dora_scale.detach()
        return state

    @torch.no_grad()
    def load_export_state_dict(self, weights: Mapping[str, Tensor]) -> None:
        """Load the repo-owned LoHa checkpoint payload for this module."""

        self.hada_w1_a.copy_(weights["hada_w1_a"])
        self.hada_w1_b.copy_(weights["hada_w1_b"])
        self.hada_w2_a.copy_(weights["hada_w2_a"])
        self.hada_w2_b.copy_(weights["hada_w2_b"])
        if self.tucker and self.hada_t1 is not None and self.hada_t2 is not None:
            if weights.get("hada_t1") is not None:
                self.hada_t1.copy_(weights["hada_t1"])
            if weights.get("hada_t2") is not None:
                self.hada_t2.copy_(weights["hada_t2"])
        if self.wd and self.dora_scale is not None and weights.get("dora_scale") is not None:
            self.dora_scale.copy_(weights["dora_scale"])
        self._reset_scalar_to_identity()

    def get_org_weight_for_compute(self, device: torch.device) -> Tensor:
        return self.org_module[0].weight.to(device, non_blocking=True)

    def get_org_bias_for_compute(self, device: torch.device) -> Tensor | None:
        bias = self.org_module[0].bias
        if bias is None:
            return None
        return bias.to(device, non_blocking=True)

    def apply_to(self) -> None:
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward

    @torch.no_grad()
    def merge_to(self, multiplier: float = 1.0) -> None:
        current_weight = self.org_module[0].weight
        merged_weight, merged_bias = self.get_merged_weight(multiplier, current_weight.shape, current_weight.device)
        self.org_module[0].weight.copy_(merged_weight.to(current_weight))
        if merged_bias is not None:
            if self.org_module[0].bias is None:
                self.org_module[0].bias = nn.Parameter(merged_bias.to(current_weight))
            else:
                self.org_module[0].bias.copy_(merged_bias.to(self.org_module[0].bias))

    def get_weight(self, shape: tuple[int, ...] | None) -> Tensor:
        """Build the LoHa diff weight tensor for the requested target shape."""

        gamma = torch.tensor(self.scale, dtype=self.hada_w1_b.dtype, device=self.hada_w1_b.device)
        weight = _build_loha_diff_weight(
            self.hada_w1_a,
            self.hada_w1_b,
            self.hada_w2_a,
            self.hada_w2_b,
            self.hada_t1,
            self.hada_t2,
            gamma,
        )
        weight = _reshape_diff_weight(weight, shape)
        if self.training and self.rank_dropout:
            drop = (torch.rand(weight.size(0), device=weight.device) > self.rank_dropout).to(weight.dtype)
            drop = drop.view(-1, *[1] * len(weight.shape[1:]))
            if self.rank_dropout_scale:
                mean_drop = drop.mean()
                if mean_drop > 0:
                    drop /= mean_drop
            weight = weight * drop
        return weight

    def get_diff_weight(self, multiplier: float = 1.0, shape: tuple[int, ...] | None = None, device=None) -> tuple[Tensor, None]:
        diff = self.get_weight(shape) * multiplier
        if device is not None:
            diff = diff.to(device)
        return diff, None

    def apply_weight_decompose(self, weight: Tensor, multiplier: float = 1.0) -> Tensor:
        if self.dora_scale is None:
            return weight
        weight = weight.to(self.dora_scale.dtype)
        if self.wd_on_output:
            weight_norm = weight.reshape(weight.shape[0], -1).norm(dim=1).reshape(weight.shape[0], *[1] * self.dora_norm_dims)
        else:
            weight_norm = (
                weight.transpose(0, 1)
                .reshape(weight.shape[1], -1)
                .norm(dim=1, keepdim=True)
                .reshape(weight.shape[1], *[1] * self.dora_norm_dims)
                .transpose(0, 1)
            )
        weight_norm = weight_norm + torch.finfo(weight.dtype).eps
        scale = self.dora_scale.to(weight.device) / weight_norm
        if multiplier != 1:
            scale = multiplier * (scale - 1) + 1
        return weight * scale

    def get_merged_weight(self, multiplier: float = 1.0, shape: tuple[int, ...] | None = None, device=None) -> tuple[Tensor, None]:
        """Return the original target weight with this LoHa module merged in."""

        if device is None:
            device = self.device
        diff = self.get_weight(shape).to(device)
        weight = self.get_org_weight_for_compute(device)
        if weight.dtype != diff.dtype:
            weight = weight.to(diff.dtype)
        if self.wd:
            merged = self.apply_weight_decompose(weight + diff * self.scalar.to(diff.device), multiplier)
        else:
            merged = weight + diff * self.scalar.to(diff.device) * multiplier
        return merged, None

    @torch.no_grad()
    def apply_max_norm(self, max_norm: float, device=None) -> tuple[int | Tensor, Tensor]:
        orig_norm = (self.get_weight(self.shape) * self.scalar.to(self.hada_w1_a.device)).norm()
        norm = torch.clamp(orig_norm, max_norm / 2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = (desired / norm).to(self.hada_w1_a.device)
        scaled = norm != desired
        if scaled:
            self.scalar.mul_(ratio)
            return scaled, orig_norm * ratio
        return 0, orig_norm

    @torch.no_grad()
    def get_norm(self, device=None) -> Tensor:
        return self.get_weight(self.shape).norm()

    def bypass_forward_diff(self, x: Tensor, scale: float = 1.0) -> Tensor:
        diff_weight = self.get_weight(self.shape).to(self.dtype) * self.scalar.to(self.device) * scale
        return self.op(x, diff_weight, None, **self.kw_dict)

    def bypass_forward(self, x: Tensor, scale: float = 1.0) -> Tensor:
        return self.org_forward(x) + self.bypass_forward_diff(x, scale=scale)

    def forward(self, x: Tensor, *args, **kwargs):
        if self.module_dropout and self.training and torch.rand(1).item() < self.module_dropout:
            bias = self.get_org_bias_for_compute(x.device)
            if bias is not None:
                bias = bias.to(x.dtype, non_blocking=True)
            return self.op(x, self.get_org_weight_for_compute(x.device).to(self.dtype), bias, **self.kw_dict)

        if self.bypass_mode:
            return self.bypass_forward(x, scale=self.multiplier)

        diff_weight = self.get_weight(self.shape).to(self.dtype) * self.scalar.to(self.device)
        weight = self.get_org_weight_for_compute(x.device).to(self.dtype)
        if self.wd:
            weight = self.apply_weight_decompose(weight + diff_weight, self.multiplier)
        else:
            weight = weight + diff_weight * self.multiplier
        bias = self.get_org_bias_for_compute(x.device)
        if bias is not None:
            bias = bias.to(x.dtype, non_blocking=True)
        return self.op(x, weight, bias, **self.kw_dict)
