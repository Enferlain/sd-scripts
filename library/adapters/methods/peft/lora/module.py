from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn


SUPPORTED_MODULE_TYPES = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)

_LORA_EXPORT_WEIGHT_KEYS = ("alpha", "lora_up.weight", "lora_down.weight")
_LORA_DETECTION_KEYS = ("lora_down.weight", "lora_up.weight")


@dataclass(slots=True)
class LoraConfig:
    multiplier: float = 1.0
    lora_dim: int = 4
    alpha: float | Tensor | None = 1.0
    dropout: float = 0.0
    rank_dropout: float = 0.0
    module_dropout: float = 0.0

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "multiplier": self.multiplier,
            "lora_dim": self.lora_dim,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "rank_dropout": self.rank_dropout,
            "module_dropout": self.module_dropout,
        }


def _is_pointwise_kernel(kernel_size: tuple[int, ...]) -> bool:
    return all(size == 1 for size in kernel_size)


class LoraModule(nn.Module):
    """Repo-owned LoRA module bound to one resolved target module.

    The normal training path is optimized for autocast-driven mixed precision:
    keep the wrapped forward shape close to legacy LoRA and avoid explicit
    per-call dtype shuffles in the hot path. A narrow fallback path handles
    unusual no-autocast dtype mismatches explicitly.
    """

    export_weight_keys = _LORA_EXPORT_WEIGHT_KEYS
    detection_keys = _LORA_DETECTION_KEYS
    required_export_weight_keys = ("lora_up.weight", "lora_down.weight")

    @classmethod
    def from_target_module(cls, lora_name: str, org_module: nn.Module, *, config: LoraConfig) -> LoraModule:
        return cls(lora_name, org_module, **config.as_kwargs())

    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,
        *,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float | Tensor | None = 1.0,
        dropout: float = 0.0,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        **_: Any,
    ) -> None:
        super().__init__()
        if not isinstance(org_module, SUPPORTED_MODULE_TYPES):
            raise ValueError(f"{type(org_module).__name__} is not supported in LoRA.")
        if lora_dim <= 0:
            raise ValueError(f"LoRA rank must be positive, got {lora_dim}.")

        self.lora_name = lora_name
        self.multiplier = multiplier
        self.lora_dim = int(lora_dim)
        self.dropout = float(dropout)
        self.rank_dropout = float(rank_dropout)
        self.module_dropout = float(module_dropout)
        self.adapter_target = None
        # Keep the target module in a mutable container so the wrapped reference
        # stays valid if later runtime code swaps the bound module object.
        self.org_module = [org_module]
        self.org_forward = org_module.forward

        if isinstance(alpha, Tensor):
            alpha_value = float(alpha.detach().float().item())
        elif alpha is None or alpha == 0:
            alpha_value = float(self.lora_dim)
        else:
            alpha_value = float(alpha)
        self.scale = alpha_value / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha_value, dtype=torch.float32))

        if isinstance(org_module, nn.Linear):
            self.module_type = "linear"
            self.shape = (org_module.out_features, org_module.in_features)
            self.in_dim = org_module.in_features
            self.out_dim = org_module.out_features
            self.kernel_size: tuple[int, ...] = ()
            self.op = F.linear
            self.kw_dict: dict[str, Any] = {}
            self.lora_down = nn.Linear(self.in_dim, self.lora_dim, bias=False)
            self.lora_up = nn.Linear(self.lora_dim, self.out_dim, bias=False)
        else:
            self.in_dim = org_module.in_channels
            self.out_dim = org_module.out_channels
            self.kernel_size = tuple(org_module.kernel_size)
            self.shape = (self.out_dim, self.in_dim, *self.kernel_size)
            self.kw_dict = {
                "stride": org_module.stride,
                "padding": org_module.padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups,
            }
            module_cls: type[nn.Conv1d] | type[nn.Conv2d] | type[nn.Conv3d]
            if isinstance(org_module, nn.Conv1d):
                self.module_type = "conv1d"
                self.op = F.conv1d
                module_cls = nn.Conv1d
            elif isinstance(org_module, nn.Conv2d):
                self.module_type = "conv2d"
                self.op = F.conv2d
                module_cls = nn.Conv2d
            else:
                self.module_type = "conv3d"
                self.op = F.conv3d
                module_cls = nn.Conv3d
            self.lora_down = module_cls(
                self.in_dim,
                self.lora_dim,
                org_module.kernel_size,
                org_module.stride,
                org_module.padding,
                dilation=org_module.dilation,
                groups=org_module.groups,
                bias=False,
            )
            self.lora_up = module_cls(
                self.lora_dim,
                self.out_dim,
                (1,) * len(self.kernel_size),
                (1,) * len(self.kernel_size),
                bias=False,
            )

        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

    @property
    def dtype(self) -> torch.dtype:
        return self.lora_down.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.lora_down.weight.device

    @classmethod
    def algo_check(cls, state_dict: Mapping[str, Tensor], lora_name: str) -> bool:
        return any(f"{lora_name}.{key}" in state_dict for key in cls.detection_keys)

    @classmethod
    def extract_state_dict(cls, state_dict: Mapping[str, Tensor], lora_name: str) -> list[Tensor | None]:
        return [state_dict.get(f"{lora_name}.{key}") for key in cls.export_weight_keys]

    @classmethod
    def make_module_from_state_dict(
        cls,
        lora_name: str,
        orig_module: nn.Module,
        up: Tensor,
        down: Tensor,
        alpha: Tensor | None,
    ) -> LoraModule:
        config = LoraConfig(
            multiplier=1.0,
            lora_dim=down.shape[0],
            alpha=float(alpha.detach().float().item()) if alpha is not None else down.shape[0],
        )
        module = cls.from_target_module(lora_name, orig_module, config=config)
        module.load_export_state_dict(
            {
                "lora_up.weight": up,
                "lora_down.weight": down,
                **({"alpha": alpha} if alpha is not None else {}),
            }
        )
        return module

    def export_state_dict(self) -> dict[str, Tensor]:
        return {
            "alpha": cast(Tensor, self.alpha).detach(),
            "lora_up.weight": self.lora_up.weight.detach(),
            "lora_down.weight": self.lora_down.weight.detach(),
        }

    @torch.no_grad()
    def load_export_state_dict(self, weights: Mapping[str, Tensor]) -> None:
        exported_alpha = weights.get("alpha")
        if exported_alpha is not None:
            alpha_value = float(exported_alpha.detach().float().item())
            expected_scale = alpha_value / self.lora_dim
            self.alpha.copy_(torch.tensor(alpha_value, device=self.alpha.device, dtype=self.alpha.dtype))
            self.scale = expected_scale

        down = weights["lora_down.weight"]
        up = weights["lora_up.weight"]
        if down.shape != self.lora_down.weight.shape:
            raise ValueError(
                f"LoRA down weight shape mismatch while loading {self.lora_name!r}: "
                f"runtime expects {tuple(self.lora_down.weight.shape)}, weights store {tuple(down.shape)}."
            )
        if up.shape != self.lora_up.weight.shape:
            raise ValueError(
                f"LoRA up weight shape mismatch while loading {self.lora_name!r}: "
                f"runtime expects {tuple(self.lora_up.weight.shape)}, weights store {tuple(up.shape)}."
            )

        self.lora_down.weight.copy_(down.to(device=self.device, dtype=self.dtype))
        self.lora_up.weight.copy_(up.to(device=self.device, dtype=self.dtype))

    def apply_to(self) -> None:
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward  # type: ignore[assignment]

    def _is_autocast_active(self) -> bool:
        if torch.is_autocast_enabled():
            return True
        try:
            return torch.is_autocast_enabled("cpu")
        except TypeError:
            cpu_autocast_enabled = getattr(torch, "is_autocast_cpu_enabled", None)
            return bool(cpu_autocast_enabled()) if callable(cpu_autocast_enabled) else False

    def _org_dtype(self) -> torch.dtype:
        return self.org_module[0].weight.dtype

    def _needs_explicit_dtype_fallback(self, x: Tensor) -> bool:
        if self._is_autocast_active():
            return False

        org_dtype = self._org_dtype()
        return x.dtype != org_dtype or x.dtype != self.dtype or self.dtype != org_dtype

    def _apply_rank_dropout(self, x: Tensor) -> tuple[Tensor, float]:
        if self.rank_dropout <= 0.0 or not self.training:
            return x, self.scale

        mask_shape = [1] * x.ndim
        mask_shape[0] = x.shape[0]
        mask_shape[1] = self.lora_dim
        mask = (torch.rand(mask_shape, device=x.device) > self.rank_dropout).to(x.dtype)
        x = x * mask
        scale = self.scale * (1.0 / (1.0 - self.rank_dropout))
        return x, scale

    def _forward_with_explicit_dtype_fallback(self, x: Tensor) -> Tensor:
        """Handle unusual no-autocast dtype mismatch cases explicitly."""
        org_dtype = self._org_dtype()
        org_input = x if x.dtype == org_dtype else x.to(org_dtype)
        org_forwarded = self.org_forward(org_input)

        if self.module_dropout > 0.0 and self.training and torch.rand(1, device=x.device) < self.module_dropout:
            return org_forwarded.to(x.dtype) if org_forwarded.dtype != x.dtype else org_forwarded

        lora_input = x if x.dtype == self.dtype else x.to(self.dtype)
        lora_hidden = self.lora_down(lora_input)

        if self.dropout > 0.0 and self.training:
            lora_hidden = F.dropout(lora_hidden, p=self.dropout)

        lora_hidden, scale = self._apply_rank_dropout(lora_hidden)
        lora_out = self.lora_up(lora_hidden)
        if lora_out.dtype != org_forwarded.dtype:
            lora_out = lora_out.to(org_forwarded.dtype)

        output = org_forwarded + lora_out * (self.multiplier * scale)
        if output.dtype != x.dtype:
            output = output.to(x.dtype)
        return output

    def forward(self, x: Tensor) -> Tensor:
        # Keep the common mixed-precision path as close to legacy LoRA as
        # possible and only pay explicit dtype-alignment costs when autocast is
        # not available to manage the computation types for us.
        if self._needs_explicit_dtype_fallback(x):
            return self._forward_with_explicit_dtype_fallback(x)

        org_forwarded = self.org_forward(x)

        if self.module_dropout > 0.0 and self.training and torch.rand(1, device=x.device) < self.module_dropout:
            return org_forwarded

        lora_hidden = self.lora_down(x)

        if self.dropout > 0.0 and self.training:
            lora_hidden = F.dropout(lora_hidden, p=self.dropout)

        lora_hidden, scale = self._apply_rank_dropout(lora_hidden)
        lora_out = self.lora_up(lora_hidden)
        return org_forwarded + lora_out * (self.multiplier * scale)

    def _get_compute_device(self, device: torch.device | None) -> torch.device:
        if device is not None:
            return device
        return self.device

    def _conv_weight_from_factors(self, up_weight: Tensor, down_weight: Tensor) -> Tensor:
        if _is_pointwise_kernel(self.kernel_size):
            flat_up = up_weight.reshape(self.out_dim, self.lora_dim)
            flat_down = down_weight.reshape(self.lora_dim, self.in_dim)
            return (flat_up @ flat_down).reshape(self.shape)

        spatial_dims = len(self.kernel_size)
        permute_order = (1, 0, *range(2, 2 + spatial_dims))
        down_input = down_weight.permute(permute_order)
        if self.module_type == "conv1d":
            delta = F.conv1d(down_input, up_weight)
        elif self.module_type == "conv2d":
            delta = F.conv2d(down_input, up_weight)
        else:
            delta = F.conv3d(down_input, up_weight)
        return delta.permute(permute_order).contiguous()

    def get_diff_weight(
        self,
        multiplier: float = 1.0,
        shape: tuple[int, ...] | torch.Size | None = None,
        device: torch.device | None = None,
    ) -> tuple[Tensor, None]:
        compute_device = self._get_compute_device(device)
        up_weight = self.lora_up.weight.to(device=compute_device, dtype=self.dtype)
        down_weight = self.lora_down.weight.to(device=compute_device, dtype=self.dtype)

        if self.module_type == "linear":
            diff_weight = up_weight @ down_weight
        else:
            diff_weight = self._conv_weight_from_factors(up_weight, down_weight)

        diff_weight = diff_weight * (multiplier * self.scale)
        if shape is not None:
            diff_weight = diff_weight.reshape(shape)
        return diff_weight, None

    def get_merged_weight(
        self,
        multiplier: float = 1.0,
        shape: tuple[int, ...] | torch.Size | None = None,
        device: torch.device | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        compute_device = self._get_compute_device(device)
        org_weight = self.org_module[0].weight.to(device=compute_device, dtype=self.dtype)
        org_bias = self.org_module[0].bias
        diff_weight, _ = self.get_diff_weight(multiplier=multiplier, shape=shape or org_weight.shape, device=compute_device)
        merged_weight = org_weight + diff_weight
        merged_bias = org_bias.to(device=compute_device, dtype=self.dtype) if org_bias is not None else None
        return merged_weight, merged_bias

    @torch.no_grad()
    def merge_to(self, multiplier: float = 1.0) -> None:
        current_weight = self.org_module[0].weight
        merged_weight, merged_bias = self.get_merged_weight(multiplier=multiplier, shape=current_weight.shape, device=current_weight.device)
        self.org_module[0].weight.copy_(merged_weight.to(current_weight))
        current_bias = self.org_module[0].bias
        if current_bias is not None and merged_bias is not None:
            current_bias.copy_(merged_bias.to(current_bias))
