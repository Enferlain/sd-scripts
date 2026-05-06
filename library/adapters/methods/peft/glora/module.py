"""Repo-owned GLoRA (Generalized LoRA) method implementation."""

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
_GLORA_EXPORT_WEIGHT_KEYS = (
    "a1.weight",
    "a2.weight",
    "b1.weight",
    "b2.weight",
    "bm.weight",
    "alpha",
)
_GLORA_DETECTION_KEYS = ("a1.weight",)


@dataclass(frozen=True, slots=True)
class GloraConfig:
    """Repo-owned config for constructing a GLoRA module on one target module."""

    multiplier: float = 1.0
    lora_dim: int | None = None
    alpha: float | Tensor | None = 1
    dropout: float = 0.0
    rank_dropout: float = 0.0
    module_dropout: float = 0.0
    use_tucker: bool = False
    use_scalar: bool = False
    rank_dropout_scale: bool = False
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
            "rank_dropout_scale": self.rank_dropout_scale,
            "bypass_mode": self.bypass_mode,
            "rs_lora": self.rs_lora,
            "orthogonalize": self.orthogonalize,
        }


def _reshape_diff_weight(diff_weight: Tensor, shape: tuple[int, ...] | None) -> Tensor:
    if shape is None:
        return diff_weight
    return diff_weight.reshape(shape)


def _rebuild_tucker(core: Tensor, up: Tensor, down: Tensor) -> Tensor:
    return torch.einsum("m n ..., i m, n j -> i j ...", core, up, down)


class GloraModule(nn.Module):
    """GLoRA method module bound to one resolved target module."""

    export_weight_keys = _GLORA_EXPORT_WEIGHT_KEYS
    detection_keys = _GLORA_DETECTION_KEYS

    @classmethod
    def from_target_module(cls, lora_name: str, org_module: nn.Module, *, config: GloraConfig) -> GloraModule:
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
        bypass_mode: bool | None = None,
        rs_lora: bool = False,
        orthogonalize: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()
        if not isinstance(org_module, SUPPORTED_MODULE_TYPES):
            raise ValueError(f"{type(org_module).__name__} is not supported in GLoRA algo.")
        if lora_dim is None or lora_dim <= 0:
            raise ValueError(f"GLoRA rank must be positive, got {lora_dim}.")

        self.lora_name = lora_name
        self.multiplier = multiplier
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.rank_dropout_scale = rank_dropout_scale
        self.bypass_mode = bool(bypass_mode)
        self.rs_lora = rs_lora
        self.use_orthogonal_weights = orthogonalize
        if self.use_orthogonal_weights and not use_scalar:
            use_scalar = True
        self.use_scalar = use_scalar
        self.drop = nn.Identity() if dropout == 0 else nn.Dropout(dropout)
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
            self.a2 = nn.Linear(in_dim, lora_dim, bias=False)
            self.a1 = nn.Linear(lora_dim, in_dim, bias=False)
            self.b2 = nn.Linear(in_dim, lora_dim, bias=False)
            self.b1 = nn.Linear(lora_dim, out_dim, bias=False)
            self.tucker = False
            self.register_module("bm", None)
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
                self.module_type = "conv1d"
                self.op = F.conv1d
                module_cls: type[nn.Module] = nn.Conv1d
            elif isinstance(org_module, nn.Conv2d):
                kernel_size = org_module.kernel_size
                stride = org_module.stride
                padding = org_module.padding
                dilation = org_module.dilation
                groups = org_module.groups
                self.module_type = "conv2d"
                self.op = F.conv2d
                module_cls = nn.Conv2d
            else:
                kernel_size = org_module.kernel_size
                stride = org_module.stride
                padding = org_module.padding
                dilation = org_module.dilation
                groups = org_module.groups
                self.module_type = "conv3d"
                self.op = F.conv3d
                module_cls = nn.Conv3d

            self.down_op = self.op
            self.up_op = self.op
            self.kw_dict = {
                "stride": stride,
                "padding": padding,
                "dilation": dilation,
                "groups": groups,
            }
            self.shape = (out_dim, in_dim, *kernel_size)
            self.tucker = use_tucker and any(size != 1 for size in kernel_size)

            self.a2 = module_cls(in_dim, lora_dim, 1, bias=False)
            self.a1 = module_cls(lora_dim, in_dim, 1, bias=False)
            if self.tucker:
                self.b2 = module_cls(in_dim, lora_dim, 1, bias=False)
                self.bm = module_cls(lora_dim, lora_dim, kernel_size, stride, padding, bias=False)
            else:
                self.b2 = module_cls(in_dim, lora_dim, kernel_size, stride, padding, bias=False)
                self.register_module("bm", None)
            self.b1 = module_cls(lora_dim, out_dim, 1, bias=False)

        self.lora_dim = lora_dim

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
        return self.a1.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.a1.weight.device

    @property
    def required_export_weight_keys(self) -> tuple[str, ...]:
        required_keys = ["a1.weight", "a2.weight", "b1.weight", "b2.weight", "alpha"]
        if self.tucker:
            required_keys.append("bm.weight")
        return tuple(required_keys)

    def _initialize_weight_tensor(self, tensor: Tensor) -> None:
        if self.use_orthogonal_weights:
            nn.init.orthogonal_(tensor)
        else:
            nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))

    def _initialize_parameters(self, *, use_scalar: bool) -> None:
        self._initialize_weight_tensor(self.a1.weight)
        self._initialize_weight_tensor(self.b1.weight)
        if self.use_orthogonal_weights or use_scalar:
            self._initialize_weight_tensor(self.a2.weight)
            self._initialize_weight_tensor(self.b2.weight)
        else:
            nn.init.zeros_(self.a2.weight)
            nn.init.zeros_(self.b2.weight)
        if self.tucker and self.bm is not None:
            self._initialize_weight_tensor(self.bm.weight)

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
        a1: Tensor,
        a2: Tensor,
        b1: Tensor,
        b2: Tensor,
        bm: Tensor | None,
        alpha: Tensor,
    ) -> GloraModule:
        config = GloraConfig(
            multiplier=1.0,
            lora_dim=a2.size(0),
            alpha=float(alpha.detach().float().item()),
            use_tucker=bm is not None,
        )
        module = cls.from_target_module(lora_name, orig_module, config=config)
        with torch.no_grad():
            module.a1.weight.copy_(a1)
            module.a2.weight.copy_(a2)
            module.b1.weight.copy_(b1)
            module.b2.weight.copy_(b2)
            if bm is not None and module.bm is not None:
                module.bm.weight.copy_(bm)
        module._reset_scalar_to_identity()
        return module

    def _reset_scalar_to_identity(self) -> None:
        identity = torch.ones_like(self.scalar)
        with torch.no_grad():
            if isinstance(self.scalar, nn.Parameter):
                self.scalar.copy_(identity)
            else:
                self.scalar.copy_(identity)

    def export_state_dict(self) -> dict[str, Tensor]:
        scale_tensor = self.scalar.detach().to(self.a2.weight.device)
        state = {
            "alpha": cast(Tensor, self.alpha).detach(),
            "a1.weight": self.a1.weight.detach(),
            "a2.weight": self.a2.weight.detach() * scale_tensor,
            "b1.weight": self.b1.weight.detach(),
            "b2.weight": self.b2.weight.detach() * scale_tensor,
        }
        if self.tucker and self.bm is not None:
            state["bm.weight"] = self.bm.weight.detach()
        return state

    @torch.no_grad()
    def load_export_state_dict(self, weights: Mapping[str, Tensor]) -> None:
        self.a1.weight.copy_(weights["a1.weight"])
        self.a2.weight.copy_(weights["a2.weight"])
        self.b1.weight.copy_(weights["b1.weight"])
        self.b2.weight.copy_(weights["b2.weight"])
        if self.tucker and self.bm is not None and weights.get("bm.weight") is not None:
            self.bm.weight.copy_(weights["bm.weight"])
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
        merged_weight, _ = self.get_merged_weight(multiplier, current_weight.shape, current_weight.device)
        self.org_module[0].weight.copy_(merged_weight.to(current_weight))

    def _make_rank_dropout_masks(self, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        drop_a = (torch.rand(self.lora_dim, device=device) > self.rank_dropout).to(dtype)
        drop_b = (torch.rand(self.lora_dim, device=device) > self.rank_dropout).to(dtype)
        if self.rank_dropout_scale:
            mean_a = drop_a.mean()
            mean_b = drop_b.mean()
            if mean_a > 0:
                drop_a /= mean_a
            if mean_b > 0:
                drop_b /= mean_b
        return drop_a, drop_b

    def make_weight(self, device=None) -> Tensor:
        wa1 = self._orthogonalize(self.a1.weight.to(device)).view(self.a1.weight.size(0), -1)
        wa2 = self._orthogonalize(self.a2.weight.to(device)).view(self.a2.weight.size(0), -1)

        orig = self.get_org_weight_for_compute(device or self.device)
        if orig.dtype != wa1.dtype:
            orig = orig.to(wa1.dtype)

        if self.tucker and self.bm is not None:
            wb1 = self._orthogonalize(self.b1.weight.to(device))
            wb2 = self._orthogonalize(self.b2.weight.to(device))
            wbm = self._orthogonalize(self.bm.weight.to(device))
            wb = _rebuild_tucker(wbm, wb1.reshape(wb1.size(0), wb1.size(1)), wb2.reshape(wb2.size(0), wb2.size(1)))
        else:
            wb1 = self._orthogonalize(self.b1.weight.to(device)).view(self.b1.weight.size(0), -1)
            wb2 = self._orthogonalize(self.b2.weight.to(device)).view(self.b2.weight.size(0), -1)
            wb = (wb1 @ wb2).view(*orig.shape)
        if orig.dim() > 2:
            w_wa1 = torch.einsum("o i ..., i j -> o j ...", orig, wa1)
            w_wa2 = torch.einsum("o i ..., i j -> o j ...", w_wa1, wa2)
        else:
            w_wa2 = (orig @ wa1) @ wa2
        return (wb + w_wa2) * self.scale * self.scalar.to(device=device)

    def get_diff_weight(self, multiplier=1.0, shape=None, device=None):
        weight = self.make_weight(device or self.device) * multiplier
        return _reshape_diff_weight(weight, shape), None

    def get_merged_weight(self, multiplier=1.0, shape=None, device=None):
        diff_weight, _ = self.get_diff_weight(multiplier, shape, device)
        weight = self.get_org_weight_for_compute(diff_weight.device)
        if weight.dtype != diff_weight.dtype:
            weight = weight.to(diff_weight.dtype)
        return weight + diff_weight, None

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        orig_norm = self.make_weight(device or self.device).norm() * self.scale
        norm = torch.clamp(orig_norm, min=max_norm / 2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = desired.cpu() / norm.cpu()

        scaled = norm != desired
        if scaled:
            self.scalar *= ratio
            return scaled, orig_norm * ratio
        return 0, orig_norm

    @torch.no_grad()
    def get_norm(self, device=None):
        return self.make_weight(device or self.device).norm()

    def _apply_target_op(self, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        if self.module_type == "linear":
            return F.linear(x, weight, bias)
        if self.module_type == "conv1d":
            return F.conv1d(x, weight, bias, **self.kw_dict)
        if self.module_type == "conv2d":
            return F.conv2d(x, weight, bias, **self.kw_dict)
        return F.conv3d(x, weight, bias, **self.kw_dict)

    def _apply_module_op(self, x: Tensor, module: nn.Module, weight: Tensor) -> Tensor:
        if isinstance(module, nn.Linear):
            return F.linear(x, weight, None)
        if isinstance(module, nn.Conv1d):
            return F.conv1d(x, weight, None, stride=module.stride, padding=module.padding, dilation=module.dilation, groups=module.groups)
        if isinstance(module, nn.Conv2d):
            return F.conv2d(x, weight, None, stride=module.stride, padding=module.padding, dilation=module.dilation, groups=module.groups)
        module = cast(nn.Conv3d, module)
        return F.conv3d(x, weight, None, stride=module.stride, padding=module.padding, dilation=module.dilation, groups=module.groups)

    def _cast_for_compute(self, x: Tensor, dtype: torch.dtype) -> Tensor:
        return cast_input_for_compute(x, dtype)

    def _dropout_view(self, drop: Tensor, dims: int) -> Tensor:
        if dims >= 4:
            return drop.view(1, -1, *([1] * (dims - 2)))
        return drop.view(*([1] * (dims - 1)), -1)

    def _bypass_forward(self, x: Tensor, scale: float = 1.0, diff: bool = False) -> Tensor:
        compute_dtype = self.dtype
        x_compute = self._cast_for_compute(x, compute_dtype)
        effective_scale = self.scale * scale

        wa1 = self._orthogonalize(self.a1.weight).to(x.device, dtype=compute_dtype)
        wa2 = self._orthogonalize(self.a2.weight).to(x.device, dtype=compute_dtype)
        wb1 = self._orthogonalize(self.b1.weight).to(x.device, dtype=compute_dtype)
        wb2 = self._orthogonalize(self.b2.weight).to(x.device, dtype=compute_dtype)

        ax_mid = self._apply_module_op(x_compute, self.a2, wa2)
        bx_mid = self._apply_module_op(x_compute, self.b2, wb2)

        if self.rank_dropout and self.training:
            drop_a, drop_b = self._make_rank_dropout_masks(ax_mid.device, ax_mid.dtype)
            ax_mid = ax_mid * self._dropout_view(drop_a, len(x_compute.shape))
            bx_mid = bx_mid * self._dropout_view(drop_b, len(x_compute.shape))

        ax = self._apply_module_op(ax_mid, self.a1, wa1)
        if self.tucker and self.bm is not None:
            wbm = self._orthogonalize(self.bm.weight).to(x.device, dtype=compute_dtype)
            bx_mid = self._apply_module_op(bx_mid, self.bm, wbm)
        bx = self._apply_module_op(bx_mid, self.b1, wb1)

        scale_tensor = self.scalar.to(device=x.device, dtype=compute_dtype)
        ax = self.drop(ax) * scale_tensor * effective_scale
        bx = self.drop(bx) * scale_tensor * effective_scale

        base_input = torch.zeros_like(x_compute) if diff else x_compute
        result = self.org_forward(base_input + ax) + bx
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
        weight = self.get_org_weight_for_compute(x.device)
        if weight.dtype != compute_dtype:
            weight = weight.to(compute_dtype)
        diff_weight, _ = self.get_diff_weight(multiplier=self.multiplier, device=x.device)
        weight = weight + diff_weight.to(weight.dtype)

        bias = self.get_org_bias_for_compute(x.device)
        if self.dropout:
            x_compute = self.drop(x_compute)

        if bias is not None and bias.dtype != compute_dtype:
            bias = bias.to(compute_dtype)
        result = self._apply_target_op(x_compute, weight, bias)
        return restore_output_dtype(result, x.dtype)
