"""Repo-owned ABBA (Hadamard Product Adaptation) method implementation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


SUPPORTED_MODULE_TYPES = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
_ABBA_EXPORT_WEIGHT_KEYS = (
    "lora_up1.weight",
    "lora_down1.weight",
    "lora_up2.weight",
    "lora_down2.weight",
    "alpha",
    "dora_scale",
)
_ABBA_DETECTION_KEYS = ("lora_up1.weight",)


@dataclass(frozen=True, slots=True)
class AbbaConfig:
    """Repo-owned config for constructing an ABBA module on one target module."""

    multiplier: float = 1.0
    lora_dim: int | None = None
    alpha: float | Tensor | None = 1
    dropout: float = 0.0
    rank_dropout: float = 0.0
    module_dropout: float = 0.0
    use_scalar: bool = False
    rank_dropout_scale: bool = False
    weight_decompose: bool = False
    wd_on_output: bool = True
    bypass_mode: bool | None = None

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "multiplier": self.multiplier,
            "lora_dim": self.lora_dim,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "rank_dropout": self.rank_dropout,
            "module_dropout": self.module_dropout,
            "use_scalar": self.use_scalar,
            "rank_dropout_scale": self.rank_dropout_scale,
            "weight_decompose": self.weight_decompose,
            "wd_on_output": self.wd_on_output,
            "bypass_mode": self.bypass_mode,
        }


def _reshape_diff_weight(diff_weight: Tensor, shape: tuple[int, ...] | None) -> Tensor:
    if shape is None:
        return diff_weight
    return diff_weight.reshape(shape)


class AbbaModule(nn.Module):
    """ABBA method module bound to one resolved target module."""

    export_weight_keys = _ABBA_EXPORT_WEIGHT_KEYS
    detection_keys = _ABBA_DETECTION_KEYS

    @classmethod
    def from_target_module(cls, lora_name: str, org_module: nn.Module, *, config: AbbaConfig) -> AbbaModule:
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
        use_scalar: bool = False,
        rank_dropout_scale: bool = False,
        weight_decompose: bool = False,
        wd_on_output: bool = True,
        bypass_mode: bool | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        if not isinstance(org_module, SUPPORTED_MODULE_TYPES):
            raise ValueError(f"{type(org_module).__name__} is not supported in ABBA algo.")
        if lora_dim is None or lora_dim < 2:
            raise ValueError(f"ABBA rank must be at least 2, got {lora_dim}.")
        if weight_decompose and bypass_mode:
            raise ValueError("ABBA weight_decompose is incompatible with bypass_mode.")

        self.lora_name = lora_name
        self.multiplier = multiplier
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.rank_dropout_scale = rank_dropout_scale
        self.bypass_mode = bool(bypass_mode)
        self.drop = nn.Identity() if dropout == 0 else nn.Dropout(dropout)
        self.org_module = [org_module]
        self.org_forward = org_module.forward
        self.adapter_target = None

        self.r1 = lora_dim // 2
        self.r2 = lora_dim - self.r1
        self.lora_dim = lora_dim
        self.isconv = not isinstance(org_module, nn.Linear)

        if isinstance(org_module, nn.Linear):
            self.module_type = "linear"
            self.shape = (org_module.out_features, org_module.in_features)
            self.op = F.linear
            self.kw_dict: dict[str, Any] = {}
            self.lora_down1 = nn.Linear(org_module.in_features, self.r1, bias=False)
            self.lora_up1 = nn.Linear(self.r1, org_module.out_features, bias=False)
            self.lora_down2 = nn.Linear(org_module.in_features, self.r2, bias=False)
            self.lora_up2 = nn.Linear(self.r2, org_module.out_features, bias=False)
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
            self.lora_down1 = nn.Conv1d(
                org_module.in_channels,
                self.r1,
                org_module.kernel_size,
                org_module.stride,
                org_module.padding,
                bias=False,
            )
            self.lora_up1 = nn.Conv1d(self.r1, org_module.out_channels, 1, bias=False)
            self.lora_down2 = nn.Conv1d(
                org_module.in_channels,
                self.r2,
                org_module.kernel_size,
                org_module.stride,
                org_module.padding,
                bias=False,
            )
            self.lora_up2 = nn.Conv1d(self.r2, org_module.out_channels, 1, bias=False)
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
            self.lora_down1 = nn.Conv2d(
                org_module.in_channels,
                self.r1,
                org_module.kernel_size,
                org_module.stride,
                org_module.padding,
                bias=False,
            )
            self.lora_up1 = nn.Conv2d(self.r1, org_module.out_channels, 1, bias=False)
            self.lora_down2 = nn.Conv2d(
                org_module.in_channels,
                self.r2,
                org_module.kernel_size,
                org_module.stride,
                org_module.padding,
                bias=False,
            )
            self.lora_up2 = nn.Conv2d(self.r2, org_module.out_channels, 1, bias=False)
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
            self.lora_down1 = nn.Conv3d(
                org_module.in_channels,
                self.r1,
                org_module.kernel_size,
                org_module.stride,
                org_module.padding,
                bias=False,
            )
            self.lora_up1 = nn.Conv3d(self.r1, org_module.out_channels, 1, bias=False)
            self.lora_down2 = nn.Conv3d(
                org_module.in_channels,
                self.r2,
                org_module.kernel_size,
                org_module.stride,
                org_module.padding,
                bias=False,
            )
            self.lora_up2 = nn.Conv3d(self.r2, org_module.out_channels, 1, bias=False)

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
        self.scale = (float(alpha) ** 2) / math.sqrt(self.r1 * self.r2)
        self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32))

        if use_scalar:
            self.scalar = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer("scalar", torch.tensor(1.0), persistent=False)

        self._initialize_parameters()

    @property
    def dtype(self) -> torch.dtype:
        return self.lora_up1.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.lora_up1.weight.device

    @property
    def required_export_weight_keys(self) -> tuple[str, ...]:
        required_keys = [
            "lora_up1.weight",
            "lora_down1.weight",
            "lora_up2.weight",
            "lora_down2.weight",
            "alpha",
        ]
        if self.wd:
            required_keys.append("dora_scale")
        return tuple(required_keys)

    def _initialize_parameters(self) -> None:
        org_weight = self.org_module[0].weight.detach().float()
        flat_weight = org_weight.reshape(org_weight.shape[0], -1)
        svd_rank = min(self.r1, flat_weight.shape[0], flat_weight.shape[1])

        up1_data = torch.zeros(flat_weight.shape[0], self.r1, dtype=torch.float32)
        down1_data = torch.zeros(self.r1, flat_weight.shape[1], dtype=torch.float32)
        if svd_rank > 0:
            u, s, v = torch.svd_lowrank(flat_weight, q=svd_rank, niter=10)
            s_sqrt = torch.sqrt(s)
            up1_data[:, :svd_rank] = u[:, :svd_rank] * s_sqrt[:svd_rank].unsqueeze(0)
            down1_data[:svd_rank, :] = s_sqrt[:svd_rank].unsqueeze(1) * v[:, :svd_rank].transpose(0, 1)

        with torch.no_grad():
            self.lora_up1.weight.copy_(up1_data.view_as(self.lora_up1.weight).to(self.lora_up1.weight))
            self.lora_down1.weight.copy_(down1_data.view_as(self.lora_down1.weight).to(self.lora_down1.weight))
            nn.init.kaiming_uniform_(self.lora_down2.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_up2.weight)

    def _reset_scalar_to_identity(self) -> None:
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
        up1: Tensor,
        down1: Tensor,
        up2: Tensor,
        down2: Tensor,
        alpha: Tensor,
        dora_scale: Tensor | None,
    ) -> AbbaModule:
        config = AbbaConfig(
            multiplier=1.0,
            lora_dim=down1.size(0) + down2.size(0),
            alpha=float(alpha.detach().float().item()),
            weight_decompose=dora_scale is not None,
        )
        module = cls.from_target_module(lora_name, orig_module, config=config)
        with torch.no_grad():
            module.lora_up1.weight.copy_(up1)
            module.lora_down1.weight.copy_(down1)
            module.lora_up2.weight.copy_(up2)
            module.lora_down2.weight.copy_(down2)
            if dora_scale is not None and module.dora_scale is not None:
                module.dora_scale.copy_(dora_scale)
        module._reset_scalar_to_identity()
        return module

    def export_state_dict(self) -> dict[str, Tensor]:
        """Export the repo-owned ABBA checkpoint payload for this module."""

        scalar = self.scalar.detach().to(self.lora_up1.weight.device)
        state = {
            "alpha": self.alpha.detach(),
            "lora_up1.weight": self.lora_up1.weight.detach() * scalar,
            "lora_down1.weight": self.lora_down1.weight.detach(),
            "lora_up2.weight": self.lora_up2.weight.detach(),
            "lora_down2.weight": self.lora_down2.weight.detach(),
        }
        if self.wd and self.dora_scale is not None:
            state["dora_scale"] = self.dora_scale.detach()
        return state

    @torch.no_grad()
    def load_export_state_dict(self, weights: Mapping[str, Tensor]) -> None:
        """Load the repo-owned ABBA checkpoint payload for this module."""

        self.lora_up1.weight.copy_(weights["lora_up1.weight"])
        self.lora_down1.weight.copy_(weights["lora_down1.weight"])
        self.lora_up2.weight.copy_(weights["lora_up2.weight"])
        self.lora_down2.weight.copy_(weights["lora_down2.weight"])
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

    def _build_diff_weight(self) -> Tensor:
        up1 = self.lora_up1.weight
        down1 = self.lora_down1.weight
        up2 = self.lora_up2.weight
        down2 = self.lora_down2.weight
        if self.isconv:
            up1 = up1.reshape(up1.shape[0], -1)
            down1 = down1.reshape(down1.shape[0], -1)
            up2 = up2.reshape(up2.shape[0], -1)
            down2 = down2.reshape(down2.shape[0], -1)
        delta_w1 = up1 @ down1
        delta_w2 = up2 @ down2
        weight = (delta_w1 * delta_w2).reshape(self.shape)
        scale = torch.as_tensor(self.scale, dtype=weight.dtype, device=weight.device)
        return weight * scale * self.scalar.to(device=weight.device, dtype=weight.dtype)

    def _apply_rank_dropout(self, weight: Tensor) -> Tensor:
        if not self.training or self.rank_dropout == 0:
            return weight
        drop = (torch.rand(weight.size(0), device=weight.device) > self.rank_dropout).to(weight.dtype)
        drop = drop.view(-1, *[1] * len(weight.shape[1:]))
        if self.rank_dropout_scale:
            mean_drop = drop.mean()
            if mean_drop > 0:
                drop = drop / mean_drop
        return weight * drop

    def get_weight(self, shape: tuple[int, ...] | None = None) -> Tensor:
        weight = self._apply_rank_dropout(self._build_diff_weight())
        return _reshape_diff_weight(weight, shape)

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
        """Return the original target weight with this ABBA module merged in."""

        if device is None:
            device = self.device
        diff = self.get_weight(shape).to(device)
        weight = self.get_org_weight_for_compute(device)
        if weight.dtype != diff.dtype:
            weight = weight.to(diff.dtype)
        if self.wd:
            merged = self.apply_weight_decompose(weight + diff, multiplier)
        else:
            merged = weight + diff * multiplier
        return merged, None

    @torch.no_grad()
    def apply_max_norm(self, max_norm: float, device=None) -> tuple[int | Tensor, Tensor]:
        orig_norm = self.get_weight(self.shape).norm()
        norm = torch.clamp(orig_norm, max_norm / 2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = (desired / norm).to(self.scalar.device)
        scaled = norm != desired
        if scaled:
            self.scalar.mul_(ratio)
            return scaled, orig_norm * ratio
        return 0, orig_norm

    @torch.no_grad()
    def get_norm(self, device=None) -> Tensor:
        return self.get_weight(self.shape).norm()

    def _apply_target_op(self, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        compute_dtype = weight.dtype
        x_compute = x.to(device=weight.device, dtype=compute_dtype)
        if bias is not None and bias.dtype != compute_dtype:
            bias = bias.to(compute_dtype)
        return self.op(x_compute, weight, bias, **self.kw_dict).to(dtype=x.dtype)

    def _apply_base_op(self, x: Tensor) -> Tensor:
        base_weight = self.get_org_weight_for_compute(x.device)
        if base_weight.dtype != self.dtype:
            base_weight = base_weight.to(self.dtype)
        bias = self.get_org_bias_for_compute(x.device)
        if bias is not None and bias.dtype != self.dtype:
            bias = bias.to(self.dtype)
        return self._apply_target_op(x, base_weight, bias)

    def bypass_forward_diff(self, x: Tensor, scale: float = 1.0) -> Tensor:
        if not self.isconv:
            x_compute = x.to(device=self.device, dtype=self.dtype)
            a1 = self.lora_down1.weight.to(self.dtype)
            b1 = self.lora_up1.weight.to(self.dtype)
            a2 = self.lora_down2.weight.to(self.dtype)
            b2 = self.lora_up2.weight.to(self.dtype)
            a_star = (a1.unsqueeze(1) * a2.unsqueeze(0)).reshape(self.r1 * self.r2, -1)
            b_star = (b1.unsqueeze(2) * b2.unsqueeze(1)).reshape(b1.shape[0], self.r1 * self.r2)
            diff = F.linear(F.linear(x_compute, a_star), b_star)
            diff = diff * (self.scale * scale) * self.scalar.to(device=x_compute.device, dtype=x_compute.dtype)
            if self.training and self.rank_dropout:
                drop = (torch.rand(diff.shape[-1], device=diff.device) > self.rank_dropout).to(diff.dtype)
                if self.rank_dropout_scale:
                    mean_drop = drop.mean()
                    if mean_drop > 0:
                        drop = drop / mean_drop
                diff = diff * drop.view(*([1] * (diff.dim() - 1)), -1)
            return self.drop(diff).to(dtype=x.dtype)

        diff_weight, _ = self.get_diff_weight(multiplier=scale, shape=self.shape, device=x.device)
        return self.drop(self._apply_target_op(x, diff_weight.to(self.dtype), None))

    def bypass_forward(self, x: Tensor, scale: float = 1.0) -> Tensor:
        return self._apply_base_op(x) + self.bypass_forward_diff(x, scale=scale)

    def forward(self, x: Tensor, *args, **kwargs):
        if self.module_dropout and self.training and torch.rand(1).item() < self.module_dropout:
            return self._apply_base_op(x)

        if self.bypass_mode:
            return self.bypass_forward(x, scale=self.multiplier)

        if self.wd:
            merged_weight, _ = self.get_merged_weight(self.multiplier, self.shape, x.device)
            bias = self.get_org_bias_for_compute(x.device)
            x_compute = self.drop(x.to(device=merged_weight.device, dtype=merged_weight.dtype))
            if bias is not None and bias.dtype != merged_weight.dtype:
                bias = bias.to(merged_weight.dtype)
            return self.op(x_compute, merged_weight, bias, **self.kw_dict).to(dtype=x.dtype)

        if self.dropout:
            base_output = self._apply_base_op(x)
            diff_weight = self.get_weight(self.shape).to(device=x.device, dtype=self.dtype)
            delta_output = self._apply_target_op(x, diff_weight, None)
            return base_output + self.drop(delta_output * self.multiplier)

        merged_weight, _ = self.get_merged_weight(self.multiplier, self.shape, x.device)
        bias = self.get_org_bias_for_compute(x.device)
        if bias is not None and bias.dtype != merged_weight.dtype:
            bias = bias.to(merged_weight.dtype)
        return self._apply_target_op(x, merged_weight, bias)
