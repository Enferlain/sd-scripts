"""Repo-owned LoCon (LyCORIS convolutional LoRA) method implementation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from library.adapters.runtime.precision import cast_input_for_compute, restore_output_dtype


SUPPORTED_MODULE_TYPES = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
LOCON_INIT_MODES = ("lycoris_legacy", "zero_delta_he", "random_nonzero")
_LOCON_EXPORT_WEIGHT_KEYS = (
    "lora_up.weight",
    "lora_down.weight",
    "lora_mid.weight",
    "alpha",
    "dora_scale",
)
_LOCON_DETECTION_KEYS = ("lora_up.weight",)


@dataclass(frozen=True, slots=True)
class LoconConfig:
    """Repo-owned config for constructing a LoCon module on one target module."""

    multiplier: float = 1.0
    lora_dim: int | None = None
    alpha: float | Tensor | None = 1
    dropout: float = 0.0
    rank_dropout: float = 0.0
    module_dropout: float = 0.0
    use_tucker: bool = False
    use_scalar: bool = False
    init_mode: str = "lycoris_legacy"
    rank_dropout_scale: bool = False
    weight_decompose: bool = False
    wd_on_output: bool = True
    bypass_mode: bool | None = None
    rs_lora: bool = False
    orthogonalize: bool = False

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
            "init_mode": self.init_mode,
            "rank_dropout_scale": self.rank_dropout_scale,
            "weight_decompose": self.weight_decompose,
            "wd_on_output": self.wd_on_output,
            "bypass_mode": self.bypass_mode,
            "rs_lora": self.rs_lora,
            "orthogonalize": self.orthogonalize,
        }


def _rebuild_tucker(core: Tensor, up: Tensor, down: Tensor) -> Tensor:
    return torch.einsum("i j ..., i p, j r -> p r ...", core, up, down)


def _reshape_diff_weight(diff_weight: Tensor, shape: tuple[int, ...] | None) -> Tensor:
    if shape is None:
        return diff_weight
    return diff_weight.reshape(shape)


class LoconModule(nn.Module):
    """LoCon method module bound to one resolved target module."""

    export_weight_keys = _LOCON_EXPORT_WEIGHT_KEYS
    detection_keys = _LOCON_DETECTION_KEYS

    @classmethod
    def from_target_module(cls, lora_name: str, org_module: nn.Module, *, config: LoconConfig) -> LoconModule:
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
        init_mode: str = "lycoris_legacy",
        rank_dropout_scale: bool = False,
        weight_decompose: bool = False,
        wd_on_output: bool = True,
        bypass_mode: bool | None = None,
        rs_lora: bool = False,
        orthogonalize: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()
        if not isinstance(org_module, SUPPORTED_MODULE_TYPES):
            raise ValueError(f"{type(org_module).__name__} is not supported in LoCon algo.")
        if init_mode not in LOCON_INIT_MODES:
            raise ValueError(f"Unknown LoCon init_mode {init_mode!r}; expected one of {LOCON_INIT_MODES}.")
        if lora_dim is None or lora_dim <= 0:
            raise ValueError(f"LoCon rank must be positive, got {lora_dim}.")
        if weight_decompose and bypass_mode:
            raise ValueError("LoCon weight_decompose is incompatible with bypass_mode.")

        self.lora_name = lora_name
        self.multiplier = multiplier
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.init_mode = init_mode
        self.rank_dropout_scale = rank_dropout_scale
        self.bypass_mode = bool(bypass_mode)
        self.rs_lora = rs_lora
        self.use_orthogonal_weights = orthogonalize
        if self.use_orthogonal_weights and not use_scalar:
            use_scalar = True
        self.use_scalar = use_scalar
        self.org_module = [org_module]
        self.org_forward = org_module.forward
        self.adapter_target = None

        if isinstance(org_module, nn.Linear):
            self.module_type = "linear"
            self.isconv = False
            self.shape = (org_module.out_features, org_module.in_features)
            self.op = F.linear
            self.down_op = F.linear
            self.up_op = F.linear
            self.kw_dict: dict[str, Any] = {}

            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)
            self.lora_mid = None
            self.tucker = False
        else:
            self.isconv = True
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            if isinstance(org_module, nn.Conv1d):
                kernel_size = org_module.kernel_size
                stride = org_module.stride
                padding = org_module.padding
                dilation = org_module.dilation
                groups = org_module.groups
                self.tucker = use_tucker and any(size != 1 for size in kernel_size)
                self.module_type = "conv1d"
                self.op = F.conv1d
                if self.tucker:
                    self.lora_down = nn.Conv1d(in_dim, lora_dim, 1, bias=False)
                    self.lora_mid = nn.Conv1d(lora_dim, lora_dim, kernel_size, stride, padding, bias=False)
                else:
                    self.lora_down = nn.Conv1d(in_dim, lora_dim, kernel_size, stride, padding, bias=False)
                    self.lora_mid = None
                self.lora_up = nn.Conv1d(lora_dim, out_dim, 1, bias=False)
            elif isinstance(org_module, nn.Conv2d):
                kernel_size = org_module.kernel_size
                stride = org_module.stride
                padding = org_module.padding
                dilation = org_module.dilation
                groups = org_module.groups
                self.tucker = use_tucker and any(size != 1 for size in kernel_size)
                self.module_type = "conv2d"
                self.op = F.conv2d
                if self.tucker:
                    self.lora_down = nn.Conv2d(in_dim, lora_dim, 1, bias=False)
                    self.lora_mid = nn.Conv2d(lora_dim, lora_dim, kernel_size, stride, padding, bias=False)
                else:
                    self.lora_down = nn.Conv2d(in_dim, lora_dim, kernel_size, stride, padding, bias=False)
                    self.lora_mid = None
                self.lora_up = nn.Conv2d(lora_dim, out_dim, 1, bias=False)
            else:
                kernel_size = org_module.kernel_size
                stride = org_module.stride
                padding = org_module.padding
                dilation = org_module.dilation
                groups = org_module.groups
                self.tucker = use_tucker and any(size != 1 for size in kernel_size)
                self.module_type = "conv3d"
                self.op = F.conv3d
                if self.tucker:
                    self.lora_down = nn.Conv3d(in_dim, lora_dim, 1, bias=False)
                    self.lora_mid = nn.Conv3d(lora_dim, lora_dim, kernel_size, stride, padding, bias=False)
                else:
                    self.lora_down = nn.Conv3d(in_dim, lora_dim, kernel_size, stride, padding, bias=False)
                    self.lora_mid = None
                self.lora_up = nn.Conv3d(lora_dim, out_dim, 1, bias=False)

            self.down_op = self.op
            self.up_op = self.op
            self.shape = (out_dim, in_dim, *kernel_size)
            self.kw_dict = {
                "stride": stride,
                "padding": padding,
                "dilation": dilation,
                "groups": groups,
            }

        self.lora_dim = lora_dim

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

        self._initialize_parameters(use_scalar=use_scalar)

    @property
    def dtype(self) -> torch.dtype:
        return self.lora_down.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.lora_down.weight.device

    @property
    def required_export_weight_keys(self) -> tuple[str, ...]:
        required_keys = ["lora_up.weight", "lora_down.weight", "alpha"]
        if self.tucker:
            required_keys.append("lora_mid.weight")
        if self.wd:
            required_keys.append("dora_scale")
        return tuple(required_keys)

    def _reset_scalar_to_identity(self) -> None:
        if getattr(self, "scalar", None) is None:
            return
        identity = torch.ones_like(self.scalar)
        with torch.no_grad():
            self.scalar.copy_(identity)

    def _initialize_weight_tensor(self, tensor: Tensor) -> None:
        if self.use_orthogonal_weights:
            nn.init.orthogonal_(tensor)
        else:
            nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))

    def _initialize_parameters(self, *, use_scalar: bool) -> None:
        if self.init_mode == "lycoris_legacy":
            self._initialize_weight_tensor(self.lora_down.weight)
            if self.tucker and self.lora_mid is not None:
                self._initialize_weight_tensor(self.lora_mid.weight)
            if self.use_orthogonal_weights or use_scalar:
                self._initialize_weight_tensor(self.lora_up.weight)
            else:
                nn.init.constant_(self.lora_up.weight, 0.0)
            return

        self._initialize_weight_tensor(self.lora_down.weight)
        self._initialize_weight_tensor(self.lora_up.weight)
        if self.tucker and self.lora_mid is not None:
            self._initialize_weight_tensor(self.lora_mid.weight)

        if self.init_mode == "zero_delta_he":
            if use_scalar:
                with torch.no_grad():
                    self.scalar.zero_()
            else:
                nn.init.zeros_(self.lora_up.weight)
            return

        if self.init_mode == "random_nonzero" and use_scalar:
            with torch.no_grad():
                self.scalar.fill_(1.0)

    def _orthogonalize(self, weight_matrix: Tensor) -> Tensor:
        if not self.use_orthogonal_weights or not self.training:
            return weight_matrix

        shape = weight_matrix.shape
        dimcount = len(shape)
        if dimcount == 0:
            return weight_matrix
        if dimcount > 2:
            weight_matrix = weight_matrix.reshape(len(weight_matrix), -1)
        elif dimcount < 2:
            weight_matrix = weight_matrix.reshape(1, -1)

        original_dtype = weight_matrix.dtype
        weight_matrix_fp32 = weight_matrix.to(torch.float32)
        rows, cols = weight_matrix_fp32.shape
        if rows >= cols:
            q, r = torch.linalg.qr(weight_matrix_fp32)
            weight_matrix_fp32 = q * torch.diag(r)
        else:
            q, r = torch.linalg.qr(weight_matrix_fp32.T)
            weight_matrix_fp32 = (q * torch.diag(r)).T
        return weight_matrix_fp32.to(original_dtype).reshape(shape).contiguous()

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
        up: Tensor,
        down: Tensor,
        mid: Tensor | None,
        alpha: Tensor,
        dora_scale: Tensor | None,
    ) -> LoconModule:
        config = LoconConfig(
            multiplier=1.0,
            lora_dim=down.size(0),
            alpha=float(alpha.detach().float().item()),
            use_tucker=mid is not None,
            weight_decompose=dora_scale is not None,
        )
        module = cls.from_target_module(lora_name, orig_module, config=config)
        with torch.no_grad():
            module.lora_up.weight.copy_(up)
            module.lora_down.weight.copy_(down)
            if mid is not None and module.lora_mid is not None:
                module.lora_mid.weight.copy_(mid)
            if dora_scale is not None and module.dora_scale is not None:
                module.dora_scale.copy_(dora_scale)
        module._reset_scalar_to_identity()
        return module

    def export_state_dict(self) -> dict[str, Tensor]:
        state = {
            "alpha": cast(Tensor, self.alpha).detach(),
            "lora_up.weight": self.lora_up.weight.detach() * self.scalar.detach().to(self.lora_up.weight.device),
            "lora_down.weight": self.lora_down.weight.detach(),
        }
        if self.tucker and self.lora_mid is not None:
            state["lora_mid.weight"] = self.lora_mid.weight.detach()
        if self.wd and self.dora_scale is not None:
            state["dora_scale"] = self.dora_scale.detach()
        return state

    @torch.no_grad()
    def load_export_state_dict(self, weights: Mapping[str, Tensor]) -> None:
        self.lora_up.weight.copy_(weights["lora_up.weight"])
        self.lora_down.weight.copy_(weights["lora_down.weight"])
        if self.tucker and self.lora_mid is not None and weights.get("lora_mid.weight") is not None:
            self.lora_mid.weight.copy_(weights["lora_mid.weight"])
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
        self.org_module[0].forward = self.forward  # type: ignore[assignment]

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

    def make_weight(self, device: torch.device | None = None) -> Tensor:
        up = self._orthogonalize(self.lora_up.weight.to(device))
        down = self._orthogonalize(self.lora_down.weight.to(device))
        if self.tucker and self.lora_mid is not None:
            mid = self._orthogonalize(self.lora_mid.weight.to(device))
            up_reduced = up.view(up.size(0), -1).transpose(0, 1)
            down_reduced = down.view(down.size(0), -1)
            weight = _rebuild_tucker(mid, up_reduced, down_reduced)
        else:
            weight = up.view(up.size(0), -1) @ down.view(down.size(0), -1)

        weight = weight.view(self.shape)
        if self.training and self.rank_dropout:
            drop = (torch.rand(weight.size(0), device=weight.device) > self.rank_dropout).to(weight.dtype)
            drop = drop.view(-1, *[1] * len(weight.shape[1:]))
            if self.rank_dropout_scale:
                mean_drop = drop.mean()
                if mean_drop > 0:
                    drop /= mean_drop
            weight = weight * drop
        return weight * self.scalar.to(weight.device)

    def get_weight(self, shape: tuple[int, ...] | None) -> Tensor:
        return _reshape_diff_weight(self.make_weight(self.device) * self.scale, shape)

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
        if device is None:
            device = self.device
        diff = self.get_diff_weight(multiplier=1.0, shape=shape, device=device)[0]
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
        orig_norm = self.make_weight(device or self.device).norm() * self.scale
        norm = torch.clamp(orig_norm, max_norm / 2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = (desired / norm).to(self.device)
        scaled = norm != desired
        if scaled:
            self.scalar.mul_(ratio)
            return scaled, orig_norm * ratio
        return 0, orig_norm

    @torch.no_grad()
    def get_norm(self, device=None) -> Tensor:
        return self.make_weight(device or self.device).norm()

    def _apply_module_op(self, x: Tensor, module: nn.Module, weight: Tensor) -> Tensor:
        if isinstance(module, nn.Linear):
            return F.linear(x, weight, None)
        if isinstance(module, nn.Conv1d):
            return F.conv1d(
                x,
                weight,
                None,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
            )
        if isinstance(module, nn.Conv2d):
            return F.conv2d(
                x,
                weight,
                None,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
            )
        module = cast(nn.Conv3d, module)
        return F.conv3d(
            x,
            weight,
            None,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )

    def _apply_target_op(self, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        if self.module_type == "linear":
            return F.linear(x, weight, bias)
        if self.module_type == "conv1d":
            return F.conv1d(x, weight, bias, **self.kw_dict)
        if self.module_type == "conv2d":
            return F.conv2d(x, weight, bias, **self.kw_dict)
        return F.conv3d(x, weight, bias, **self.kw_dict)

    def _cast_for_compute(self, x: Tensor, dtype: torch.dtype) -> Tensor:
        return cast_input_for_compute(x, dtype)

    def bypass_forward_diff(self, x: Tensor, scale: float = 1.0) -> Tensor:
        compute_dtype = self.dtype
        x_compute = self._cast_for_compute(x, compute_dtype)

        down_weight = self._orthogonalize(self.lora_down.weight).to(device=x.device, dtype=compute_dtype)
        mid = self._apply_module_op(x_compute, self.lora_down, down_weight)

        if self.tucker and self.lora_mid is not None:
            mid_weight = self._orthogonalize(self.lora_mid.weight).to(device=x.device, dtype=compute_dtype)
            mid = self._apply_module_op(mid, self.lora_mid, mid_weight)

        if self.rank_dropout and self.training:
            drop = (torch.rand(self.lora_dim, device=mid.device) > self.rank_dropout).to(mid.dtype)
            if self.rank_dropout_scale:
                mean_drop = drop.mean()
                if mean_drop > 0:
                    drop /= mean_drop
            if len(mid.shape) == 4:
                drop = drop.view(1, -1, 1, 1)
            elif len(mid.shape) == 5:
                drop = drop.view(1, -1, 1, 1, 1)
            elif len(mid.shape) == 3:
                drop = drop.view(1, -1, 1)
            else:
                drop = drop.view(*([1] * (len(mid.shape) - 1)), -1)
            mid = mid * drop

        up_weight = self._orthogonalize(self.lora_up.weight).to(device=x.device, dtype=compute_dtype)
        up = self._apply_module_op(mid, self.lora_up, up_weight)
        diff = up * self.scalar.to(device=x.device, dtype=compute_dtype) * self.scale * scale
        if self.training and self.dropout:
            diff = F.dropout(diff, p=self.dropout, training=True)
        return diff

    def bypass_forward(self, x: Tensor, scale: float = 1.0) -> Tensor:
        compute_dtype = self.dtype
        x_compute = self._cast_for_compute(x, compute_dtype)
        org_weight = self.get_org_weight_for_compute(x.device).to(dtype=compute_dtype)
        bias = self.get_org_bias_for_compute(x.device)
        if bias is not None:
            bias = bias.to(dtype=compute_dtype)
        base = self._apply_target_op(x_compute, org_weight, bias)
        diff = self.bypass_forward_diff(x, scale=scale)
        result = base + diff
        return restore_output_dtype(result, x.dtype)

    def forward(self, x: Tensor, *args, **kwargs):
        if self.module_dropout and self.training and torch.rand(1).item() < self.module_dropout:
            org_weight = self.get_org_weight_for_compute(x.device).to(dtype=self.dtype)
            bias = self.get_org_bias_for_compute(x.device)
            if bias is not None:
                bias = bias.to(dtype=self.dtype)
            result = self._apply_target_op(self._cast_for_compute(x, self.dtype), org_weight, bias)
            return restore_output_dtype(result, x.dtype)

        if self.bypass_mode:
            return self.bypass_forward(x, scale=self.multiplier)

        compute_dtype = self.dtype
        x_compute = self._cast_for_compute(x, compute_dtype)
        diff_weight = self.make_weight(x.device).to(dtype=compute_dtype) * self.scale
        org_weight = self.get_org_weight_for_compute(x.device).to(dtype=compute_dtype)

        if self.wd:
            weight = self.apply_weight_decompose(org_weight + diff_weight, self.multiplier)
            if self.training and self.dropout:
                x_compute = F.dropout(x_compute, p=self.dropout, training=True)
        else:
            weight = org_weight + diff_weight * self.multiplier

        bias = self.get_org_bias_for_compute(x.device)
        if bias is not None:
            bias = bias.to(dtype=compute_dtype)
        result = self._apply_target_op(x_compute, weight, bias)
        return restore_output_dtype(result, x.dtype)
