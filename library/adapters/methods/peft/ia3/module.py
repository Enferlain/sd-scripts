"""Repo-owned IA3 method implementation."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from library.adapters.runtime.precision import cast_input_for_compute, restore_output_dtype


SUPPORTED_MODULE_TYPES = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
_IA3_EXPORT_WEIGHT_KEYS = ("weight", "on_input")
_IA3_DETECTION_KEYS = ("weight",)


def _reshape_weight(weight: Tensor, shape: tuple[int, ...] | None) -> Tensor:
    if shape is None:
        return weight
    return weight.reshape(shape)


def _as_python_bool(value: Tensor | None, *, default: bool) -> bool:
    if value is None:
        return default
    return bool(int(value.detach().cpu().item()))


@dataclass(frozen=True, slots=True)
class Ia3Config:
    """Repo-owned config for constructing an IA3 module on one target module."""

    multiplier: float = 1.0
    train_on_input: bool = False
    module_dropout: float = 0.0
    bypass_mode: bool | None = None

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "multiplier": self.multiplier,
            "train_on_input": self.train_on_input,
            "module_dropout": self.module_dropout,
            "bypass_mode": self.bypass_mode,
        }


class Ia3Module(nn.Module):
    """IA3 method module bound to one resolved target module."""

    export_weight_keys = _IA3_EXPORT_WEIGHT_KEYS
    detection_keys = _IA3_DETECTION_KEYS

    @classmethod
    def from_target_module(cls, lora_name: str, org_module: nn.Module, *, config: Ia3Config) -> Ia3Module:
        return cls(lora_name, org_module, **config.as_kwargs())

    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,
        *,
        multiplier: float = 1.0,
        train_on_input: bool = False,
        module_dropout: float = 0.0,
        bypass_mode: bool | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        if not isinstance(org_module, SUPPORTED_MODULE_TYPES):
            raise ValueError(f"{type(org_module).__name__} is not supported in IA3 algo.")

        self.lora_name = lora_name
        self.multiplier = multiplier
        self.module_dropout = module_dropout
        self.bypass_mode = bool(bypass_mode)
        self.org_module = [org_module]
        self.org_forward = org_module.forward
        self.adapter_target = None

        if isinstance(org_module, nn.Linear):
            self.module_type = "linear"
            self.shape = (org_module.out_features, org_module.in_features)
            self.out_dim = org_module.out_features
            self.in_dim = org_module.in_features
            self.op = F.linear
            self.kw_dict: dict[str, Any] = {}
        else:
            self.out_dim = org_module.out_channels
            self.in_dim = org_module.in_channels
            self.shape = (self.out_dim, self.in_dim, *org_module.kernel_size)
            self.kw_dict = {
                "stride": org_module.stride,
                "padding": org_module.padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups,
            }
            if isinstance(org_module, nn.Conv1d):
                self.module_type = "conv1d"
                self.op = F.conv1d
            elif isinstance(org_module, nn.Conv2d):
                self.module_type = "conv2d"
                self.op = F.conv2d
            else:
                self.module_type = "conv3d"
                self.op = F.conv3d

        self.train_on_input = bool(train_on_input)
        train_dim = self.in_dim if self.train_on_input else self.out_dim
        self.weight = nn.Parameter(torch.empty(train_dim, device=org_module.weight.device, dtype=org_module.weight.dtype))
        nn.init.zeros_(self.weight)
        self.register_buffer("on_input", torch.tensor(int(self.train_on_input), dtype=torch.int64))

    @property
    def dtype(self) -> torch.dtype:
        return self.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.weight.device

    @property
    def required_export_weight_keys(self) -> tuple[str, ...]:
        return self.export_weight_keys

    @classmethod
    def algo_check(cls, state_dict: Mapping[str, Tensor], lora_name: str) -> bool:
        return any(f"{lora_name}.{key}" in state_dict for key in cls.detection_keys)

    @classmethod
    def extract_state_dict(cls, state_dict: Mapping[str, Tensor], lora_name: str) -> list[Tensor | None]:
        return [state_dict.get(f"{lora_name}.{key}", None) for key in cls.export_weight_keys]

    @classmethod
    def _infer_train_on_input(cls, orig_module: nn.Module, weight: Tensor) -> bool:
        flat_size = int(weight.numel())
        if isinstance(orig_module, nn.Linear):
            in_dim = orig_module.in_features
            out_dim = orig_module.out_features
        else:
            in_dim = orig_module.in_channels
            out_dim = orig_module.out_channels

        if flat_size == in_dim and flat_size != out_dim:
            return True
        if flat_size == out_dim and flat_size != in_dim:
            return False
        raise ValueError("IA3 checkpoints must store 'on_input' when input and output dimensions are ambiguous.")

    @classmethod
    def make_module_from_state_dict(
        cls,
        lora_name: str,
        orig_module: nn.Module,
        weight: Tensor,
        on_input: Tensor | None,
    ) -> Ia3Module:
        if on_input is None:
            train_on_input = cls._infer_train_on_input(orig_module, weight)
        else:
            train_on_input = _as_python_bool(on_input, default=False)
        module = cls.from_target_module(
            lora_name,
            orig_module,
            config=Ia3Config(multiplier=1.0, train_on_input=train_on_input),
        )
        module.load_export_state_dict(
            {
                "weight": weight,
                **({"on_input": on_input} if on_input is not None else {}),
            }
        )
        return module

    def export_state_dict(self) -> dict[str, Tensor]:
        return {
            "weight": self.weight.detach(),
            "on_input": self.on_input.detach(),
        }

    @torch.no_grad()
    def load_export_state_dict(self, weights: Mapping[str, Tensor]) -> None:
        stored_on_input = weights.get("on_input")
        if stored_on_input is not None:
            train_on_input = _as_python_bool(stored_on_input, default=self.train_on_input)
            if train_on_input != self.train_on_input:
                raise ValueError(
                    f"IA3 train_on_input mismatch while loading {self.lora_name!r}: "
                    f"runtime expects {self.train_on_input}, weights store {train_on_input}."
                )

        exported_weight = weights["weight"].reshape(-1)
        if exported_weight.numel() != self.weight.numel():
            raise ValueError(
                f"IA3 weight shape mismatch while loading {self.lora_name!r}: "
                f"runtime expects {self.weight.numel()} values, weights store {exported_weight.numel()}."
            )
        self.weight.copy_(exported_weight.to(device=self.device, dtype=self.dtype))

    def get_org_weight_for_compute(self, device: torch.device) -> Tensor:
        return self.org_module[0].weight.to(device, non_blocking=True)

    def get_org_bias_for_compute(self, device: torch.device) -> Tensor | None:
        bias = self.org_module[0].bias
        if bias is None:
            return None
        return bias.to(device, non_blocking=True)

    def apply_to(self) -> None:
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward  # type: ignore[assignment]

    @torch.no_grad()
    def merge_to(self, multiplier: float = 1.0) -> None:
        current_weight = self.org_module[0].weight
        merged_weight, merged_bias = self.get_merged_weight(multiplier, current_weight.shape, current_weight.device)
        self.org_module[0].weight.copy_(merged_weight.to(current_weight))
        current_bias = self.org_module[0].bias
        if current_bias is not None and merged_bias is not None:
            self.org_module[0].bias.copy_(merged_bias.to(current_bias))

    def _cast_for_compute(self, x: Tensor, dtype: torch.dtype) -> Tensor:
        return cast_input_for_compute(x, dtype)

    def _scale_weight_view(self, scale_vector: Tensor) -> Tensor:
        if self.module_type == "linear":
            if self.train_on_input:
                return scale_vector.view(1, -1)
            return scale_vector.view(-1, 1)
        spatial_dims = len(self.shape) - 2
        if self.train_on_input:
            return scale_vector.view(1, -1, *([1] * spatial_dims))
        return scale_vector.view(-1, 1, *([1] * spatial_dims))

    def _scale_activation_view(self, scale_vector: Tensor) -> Tensor:
        if self.module_type == "linear":
            return scale_vector.view(1, -1)
        return scale_vector.view(1, -1, *([1] * (len(self.shape) - 2)))

    def _apply_target_op(self, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        if self.module_type == "linear":
            return F.linear(x, weight, bias)
        if self.module_type == "conv1d":
            return F.conv1d(x, weight, bias, **self.kw_dict)
        if self.module_type == "conv2d":
            return F.conv2d(x, weight, bias, **self.kw_dict)
        return F.conv3d(x, weight, bias, **self.kw_dict)

    def _weight_scale_delta(self, multiplier: float, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        return self.weight.to(device=device, dtype=dtype) * multiplier

    def _build_scaled_weight_bias(self, multiplier: float, *, device: torch.device, diff: bool) -> tuple[Tensor, Tensor | None]:
        base_weight = self.get_org_weight_for_compute(device)
        compute_dtype = base_weight.dtype
        scale_delta = self._weight_scale_delta(multiplier, device=device, dtype=compute_dtype)
        scale_vector = scale_delta if diff else scale_delta + 1.0
        scaled_weight = base_weight * self._scale_weight_view(scale_vector)

        base_bias = self.get_org_bias_for_compute(device)
        if base_bias is None or self.train_on_input:
            return scaled_weight, (None if diff else base_bias)

        if base_bias.dtype != compute_dtype:
            base_bias = base_bias.to(compute_dtype)
        scaled_bias = base_bias * scale_vector
        return scaled_weight, scaled_bias

    def get_diff_weight(self, multiplier: float = 1.0, shape=None, device=None):
        diff_weight, diff_bias = self._build_scaled_weight_bias(multiplier, device=device or self.device, diff=True)
        return _reshape_weight(diff_weight, shape), diff_bias

    def get_merged_weight(self, multiplier: float = 1.0, shape=None, device=None):
        merged_weight, merged_bias = self._build_scaled_weight_bias(multiplier, device=device or self.device, diff=False)
        return _reshape_weight(merged_weight, shape), merged_bias

    @torch.no_grad()
    def apply_max_norm(self, max_norm: float, device=None) -> tuple[bool, Tensor] | tuple[int, Tensor]:
        diff_weight, diff_bias = self.get_diff_weight(multiplier=self.multiplier, device=device or self.device)
        norm_sq = diff_weight.float().pow(2).sum()
        if diff_bias is not None:
            norm_sq = norm_sq + diff_bias.float().pow(2).sum()
        orig_norm = norm_sq.sqrt()
        if orig_norm <= max_norm:
            return 0, orig_norm

        ratio = max_norm / orig_norm.clamp(min=torch.finfo(orig_norm.dtype).eps)
        self.weight.mul_(ratio.to(device=self.weight.device, dtype=self.weight.dtype))
        return True, orig_norm * ratio

    @torch.no_grad()
    def get_norm(self, device=None):
        diff_weight, diff_bias = self.get_diff_weight(multiplier=self.multiplier, device=device or self.device)
        norm_sq = diff_weight.float().pow(2).sum()
        if diff_bias is not None:
            norm_sq = norm_sq + diff_bias.float().pow(2).sum()
        return norm_sq.sqrt()

    def _bypass_forward(self, x: Tensor, scale: float = 1.0, diff: bool = False) -> Tensor:
        compute_dtype = self.dtype
        x_compute = self._cast_for_compute(x, compute_dtype)
        base_weight = self.get_org_weight_for_compute(x.device)
        if base_weight.dtype != compute_dtype:
            base_weight = base_weight.to(compute_dtype)
        bias = self.get_org_bias_for_compute(x.device)
        if bias is not None and bias.dtype != compute_dtype:
            bias = bias.to(compute_dtype)

        scale_delta = self._weight_scale_delta(scale, device=x.device, dtype=compute_dtype)
        scale_vector = scale_delta if diff else scale_delta + 1.0

        if self.train_on_input:
            x_scaled = x_compute * self._scale_activation_view(scale_vector)
            result = self._apply_target_op(x_scaled, base_weight, None if diff else bias)
        else:
            result = self._apply_target_op(x_compute, base_weight, bias)
            result = result * self._scale_activation_view(scale_vector)

        return restore_output_dtype(result, x.dtype)

    def bypass_forward_diff(self, x, scale=1):
        return self._bypass_forward(x, scale=scale, diff=True)

    def bypass_forward(self, x, scale=1):
        return self._bypass_forward(x, scale=scale, diff=False)

    def forward(self, x, *args, **kwargs):
        if self.module_dropout and self.training and torch.rand(1).item() < self.module_dropout:
            return self.org_forward(x)
        if self.bypass_mode:
            return self.bypass_forward(x, self.multiplier)

        compute_dtype = self.dtype
        x_compute = self._cast_for_compute(x, compute_dtype)
        merged_weight, merged_bias = self.get_merged_weight(multiplier=self.multiplier, device=x.device)
        if merged_weight.dtype != compute_dtype:
            merged_weight = merged_weight.to(compute_dtype)
        if merged_bias is not None and merged_bias.dtype != compute_dtype:
            merged_bias = merged_bias.to(compute_dtype)
        result = self._apply_target_op(x_compute, merged_weight, merged_bias)
        return restore_output_dtype(result, x.dtype)
