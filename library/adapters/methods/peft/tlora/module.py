"""Repo-owned TLora (timestep-aware LoRA) method implementation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


TloraSigType = Literal["principal", "last", "middle"]
SUPPORTED_MODULE_TYPES = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
_TLORA_EXPORT_WEIGHT_KEYS = (
    "q_layer.weight",
    "p_layer.weight",
    "lambda_layer",
    "alpha",
    "base_q",
    "base_p",
    "base_lambda",
)
_TLORA_DETECTION_KEYS = ("lambda_layer",)


def compute_timestep_mask(
    timestep: int,
    *,
    max_timestep: int,
    max_rank: int,
    min_rank: int = 1,
    alpha: float = 1.0,
) -> Tensor:
    """Compute a TLora rank mask for one timestep."""

    if max_rank <= 0:
        raise ValueError("TLora max_rank must be positive.")
    if max_timestep <= 0:
        raise ValueError("TLora max_timestep must be positive.")
    if min_rank <= 0:
        raise ValueError("TLora min_rank must be positive.")
    if min_rank > max_rank:
        raise ValueError("TLora min_rank cannot exceed max_rank.")
    if alpha <= 0.0:
        raise ValueError("TLora mask alpha must be greater than 0.0.")

    active_rank = int(((max_timestep - timestep) / max_timestep) ** alpha * (max_rank - min_rank)) + min_rank
    active_rank = max(min_rank, min(active_rank, max_rank))
    mask = torch.zeros((1, max_rank), dtype=torch.float32)
    mask[:, :active_rank] = 1.0
    return mask


def compute_timestep_mask_batch(
    timesteps: torch.Tensor,
    *,
    max_timestep: int,
    max_rank: int,
    min_rank: int = 1,
    alpha: float = 1.0,
) -> Tensor:
    """Compute TLora rank masks for a batch of timesteps."""

    if max_rank <= 0:
        raise ValueError("TLora max_rank must be positive.")
    if max_timestep <= 0:
        raise ValueError("TLora max_timestep must be positive.")
    if min_rank <= 0:
        raise ValueError("TLora min_rank must be positive.")
    if min_rank > max_rank:
        raise ValueError("TLora min_rank cannot exceed max_rank.")
    if alpha <= 0.0:
        raise ValueError("TLora mask alpha must be greater than 0.0.")

    timesteps_float = timesteps.reshape(-1).to(dtype=torch.float32)
    active_rank = ((max_timestep - timesteps_float) / max_timestep) ** alpha * (max_rank - min_rank) + min_rank
    active_rank = active_rank.to(dtype=torch.long).clamp(min=min_rank, max=max_rank)
    rank_indices = torch.arange(max_rank, device=timesteps.device).unsqueeze(0)
    return (rank_indices < active_rank.unsqueeze(1)).to(dtype=torch.float32)


@dataclass(frozen=True, slots=True)
class TloraConfig:
    """Repo-owned config for constructing a TLora module on one target module."""

    multiplier: float = 1.0
    lora_dim: int | None = None
    alpha: float | Tensor | None = 1
    dropout: float = 0.0
    module_dropout: float = 0.0
    use_scalar: bool = False
    bypass_mode: bool | None = None
    sig_type: TloraSigType = "principal"
    use_data_init: bool = True

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "multiplier": self.multiplier,
            "lora_dim": self.lora_dim,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "module_dropout": self.module_dropout,
            "use_scalar": self.use_scalar,
            "bypass_mode": self.bypass_mode,
            "sig_type": self.sig_type,
            "use_data_init": self.use_data_init,
        }


class TloraModule(nn.Module):
    """TLora method module bound to one resolved target module."""

    export_weight_keys = _TLORA_EXPORT_WEIGHT_KEYS
    detection_keys = _TLORA_DETECTION_KEYS

    @classmethod
    def from_target_module(cls, lora_name: str, org_module: nn.Module, *, config: TloraConfig) -> TloraModule:
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
        module_dropout: float = 0.0,
        use_scalar: bool = False,
        bypass_mode: bool | None = None,
        sig_type: TloraSigType = "principal",
        use_data_init: bool = True,
        initialize_from_svd: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        if not isinstance(org_module, SUPPORTED_MODULE_TYPES):
            raise ValueError(f"{type(org_module).__name__} is not supported in TLora algo.")
        if lora_dim is None or lora_dim <= 0:
            raise ValueError(f"TLora rank must be positive, got {lora_dim}.")

        self.lora_name = lora_name
        self.multiplier = multiplier
        self.dropout = dropout
        self.module_dropout = module_dropout
        self.bypass_mode = bool(bypass_mode)
        self.use_data_init = use_data_init
        self.sig_type = sig_type
        self.drop = nn.Identity() if dropout == 0 else nn.Dropout(dropout)
        self.org_module = [org_module]
        self.org_forward = org_module.forward
        self.adapter_target = None
        self.lora_dim = int(lora_dim)
        self._active_timestep_mask: Tensor | None = None

        if isinstance(org_module, nn.Linear):
            self.module_type = "linear"
            self.isconv = False
            self.shape = (org_module.out_features, org_module.in_features)
            self.op = F.linear
            self.down_op = F.linear
            self.up_op = F.linear
            self.kw_dict: dict[str, Any] = {}
            self.kw_dict_down: dict[str, Any] = {}
            self.kw_dict_up: dict[str, Any] = {}
            self.q_layer = nn.Linear(org_module.in_features, self.lora_dim, bias=False)
            self.p_layer = nn.Linear(self.lora_dim, org_module.out_features, bias=False)
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            flat_in_dim = in_dim
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

            self.shape = (out_dim, in_dim, *kernel_size)
            self.kw_dict = {
                "stride": stride,
                "padding": padding,
                "dilation": dilation,
                "groups": groups,
            }
            self.down_op = self.op
            self.up_op = self.op
            self.kw_dict_down = {
                "stride": stride,
                "padding": (0,) * len(kernel_size),
                "dilation": (1,) * len(kernel_size),
                "groups": 1,
            }
            self.kw_dict_up = {
                "stride": (1,) * len(kernel_size),
                "padding": (0,) * len(kernel_size),
                "dilation": (1,) * len(kernel_size),
                "groups": 1,
            }
            self.q_layer = module_cls(in_dim, self.lora_dim, 1, bias=False)
            self.p_layer = module_cls(self.lora_dim, out_dim, 1, bias=False)
            flat_in_dim = in_dim * math.prod(kernel_size)

        self.lambda_layer = nn.Parameter(torch.ones(1, self.lora_dim, dtype=torch.float32))

        if isinstance(alpha, Tensor):
            alpha = float(alpha.detach().float().item())
        alpha = self.lora_dim if alpha is None or alpha == 0 else float(alpha)
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

        if use_scalar:
            self.scalar = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            self.register_buffer("scalar", torch.tensor(1.0, dtype=torch.float32), persistent=False)

        if initialize_from_svd:
            self._initialize_from_svd(org_module, out_dim=out_dim, flat_in_dim=flat_in_dim)
        else:
            self._initialize_placeholder_parameters()

        self.register_buffer("base_q", self.q_layer.weight.detach().clone())
        self.register_buffer("base_p", self.p_layer.weight.detach().clone())
        self.register_buffer("base_lambda", self.lambda_layer.detach().clone())

    @property
    def dtype(self) -> torch.dtype:
        return self.q_layer.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.q_layer.weight.device

    @property
    def required_export_weight_keys(self) -> tuple[str, ...]:
        return self.export_weight_keys

    def _initialize_placeholder_parameters(self) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.q_layer.weight)
            nn.init.zeros_(self.p_layer.weight)
            self.lambda_layer.fill_(1.0)

    def _initialize_from_svd(self, org_module: nn.Module, *, out_dim: int, flat_in_dim: int) -> None:
        weight = org_module.weight.detach().float()
        if self.isconv:
            weight = weight.reshape(out_dim, -1)

        if self.use_data_init:
            svd_input = weight
        else:
            svd_input = torch.normal(mean=0.0, std=1 / self.lora_dim, size=(out_dim, flat_in_dim), dtype=torch.float32)

        u, s, vh = torch.linalg.svd(svd_input, full_matrices=False)

        if self.sig_type == "principal":
            q_init = vh[: self.lora_dim]
            p_init = u[:, : self.lora_dim]
            lambda_init = s[: self.lora_dim]
        elif self.sig_type == "last":
            q_init = vh[-self.lora_dim :]
            p_init = u[:, -self.lora_dim :]
            lambda_init = s[-self.lora_dim :]
        else:
            start = max((vh.shape[0] - self.lora_dim) // 2, 0)
            q_init = vh[start : start + self.lora_dim]
            p_init = u[:, start : start + self.lora_dim]
            lambda_init = s[start : start + self.lora_dim]

        if q_init.shape[0] < self.lora_dim:
            pad_size = self.lora_dim - q_init.shape[0]
            q_init = F.pad(q_init, (0, 0, 0, pad_size))
            p_init = F.pad(p_init, (0, pad_size))
            lambda_init = F.pad(lambda_init, (0, pad_size), value=1e-6)

        with torch.no_grad():
            if self.isconv:
                kernel_ones = [1] * (len(self.shape) - 2)
                self.q_layer.weight.copy_(
                    q_init[:, : self.shape[1]].reshape(self.lora_dim, self.shape[1], *kernel_ones).contiguous()
                )
                self.p_layer.weight.copy_(
                    p_init[: self.shape[0], :].reshape(self.shape[0], self.lora_dim, *kernel_ones).contiguous()
                )
            else:
                self.q_layer.weight.copy_(q_init.contiguous())
                self.p_layer.weight.copy_(p_init.contiguous())
            self.lambda_layer.copy_(lambda_init.unsqueeze(0).contiguous())

    def _reset_scalar_to_identity(self) -> None:
        identity = torch.ones_like(self.scalar)
        with torch.no_grad():
            self.scalar.copy_(identity)

    def _set_alpha_value(self, alpha_value: float) -> None:
        alpha_tensor = torch.tensor(alpha_value, device=self.alpha.device, dtype=self.alpha.dtype)
        with torch.no_grad():
            self.alpha.copy_(alpha_tensor)
        self.scale = alpha_value / self.lora_dim

    def set_timestep_mask(self, mask: Tensor | None) -> None:
        self._active_timestep_mask = mask

    def clear_timestep_mask(self) -> None:
        self._active_timestep_mask = None

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
        q_weight: Tensor,
        p_weight: Tensor,
        lambda_weight: Tensor,
        alpha: Tensor,
        base_q: Tensor | None = None,
        base_p: Tensor | None = None,
        base_lambda: Tensor | None = None,
    ) -> TloraModule:
        config = TloraConfig(
            multiplier=1.0,
            lora_dim=int(q_weight.shape[0]),
            alpha=float(alpha.detach().float().item()),
        )
        module = cls(
            lora_name,
            orig_module,
            initialize_from_svd=False,
            **config.as_kwargs(),
        )
        with torch.no_grad():
            module.q_layer.weight.copy_(q_weight)
            module.p_layer.weight.copy_(p_weight)
            module.lambda_layer.copy_(lambda_weight)
            if base_q is not None:
                module.base_q.copy_(base_q)
            else:
                module.base_q.copy_(q_weight)
            if base_p is not None:
                module.base_p.copy_(base_p)
            else:
                module.base_p.copy_(p_weight)
            if base_lambda is not None:
                module.base_lambda.copy_(base_lambda)
            else:
                module.base_lambda.copy_(lambda_weight)
        module._set_alpha_value(float(alpha.detach().float().item()))
        module._reset_scalar_to_identity()
        return module

    def export_state_dict(self) -> dict[str, Tensor]:
        """Export the repo-owned TLora checkpoint payload for this module."""

        scalar = self.scalar.detach().to(device=self.lambda_layer.device, dtype=self.lambda_layer.dtype)
        return {
            "q_layer.weight": self.q_layer.weight.detach(),
            "p_layer.weight": self.p_layer.weight.detach(),
            "lambda_layer": self.lambda_layer.detach() * scalar,
            "alpha": self.alpha.detach(),
            "base_q": self.base_q.detach(),
            "base_p": self.base_p.detach(),
            "base_lambda": self.base_lambda.detach() * scalar,
        }

    @torch.no_grad()
    def load_export_state_dict(self, weights: Mapping[str, Tensor]) -> None:
        """Load the repo-owned TLora checkpoint payload for this module."""

        self.q_layer.weight.copy_(weights["q_layer.weight"])
        self.p_layer.weight.copy_(weights["p_layer.weight"])
        self.lambda_layer.copy_(weights["lambda_layer"])
        self.base_q.copy_(weights["base_q"])
        self.base_p.copy_(weights["base_p"])
        self.base_lambda.copy_(weights["base_lambda"])
        self._set_alpha_value(float(weights["alpha"].detach().float().item()))
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

    def _get_mask(self, device: torch.device) -> Tensor:
        mask = self._active_timestep_mask
        if mask is None:
            mask = torch.ones(1, self.lora_dim, device=device, dtype=torch.float32)
        if mask.shape[1] > self.lora_dim:
            mask = mask[:, : self.lora_dim]
        elif mask.shape[1] < self.lora_dim:
            mask = F.pad(mask, (0, self.lora_dim - mask.shape[1]), value=1.0)
        return mask.to(device=device, dtype=torch.float32)

    def _has_batched_mask(self) -> bool:
        mask = self._active_timestep_mask
        return mask is not None and mask.shape[0] > 1

    @staticmethod
    def _reshape_mask_for_output(mask: Tensor, output: Tensor) -> Tensor:
        if output.dim() == mask.dim():
            return mask
        if output.dim() > mask.dim():
            if output.shape[1] == mask.shape[-1]:
                return mask.view(*mask.shape, *([1] * (output.dim() - mask.dim())))
            middle_dims = output.dim() - mask.dim()
            return mask.view(mask.shape[0], *([1] * middle_dims), mask.shape[1])
        return mask

    def _apply_target_op(self, x: Tensor, weight: Tensor, bias: Tensor | None, *, kwargs: Mapping[str, Any] | None = None) -> Tensor:
        compute_dtype = weight.dtype
        x_compute = x.to(device=weight.device, dtype=compute_dtype)
        bias_compute = bias
        if bias_compute is not None and bias_compute.dtype != compute_dtype:
            bias_compute = bias_compute.to(compute_dtype)
        return self.op(x_compute, weight, bias_compute, **(dict(kwargs or self.kw_dict))).to(dtype=x.dtype)

    def _apply_base_op(self, x: Tensor) -> Tensor:
        base_weight = self.get_org_weight_for_compute(x.device)
        if base_weight.dtype != self.dtype:
            base_weight = base_weight.to(self.dtype)
        bias = self.get_org_bias_for_compute(x.device)
        if bias is not None and bias.dtype != self.dtype:
            bias = bias.to(self.dtype)
        return self._apply_target_op(x, base_weight, bias)

    def _get_diff_weight_components(self, device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        mask = self._get_mask(device)
        if mask.shape[0] > 1:
            raise ValueError("TLora batched timestep masks require bypass-mode execution and cannot be merged into one weight delta.")
        q = self.q_layer.weight.to(device)
        p = self.p_layer.weight.to(device)
        lam = self.lambda_layer.to(device) * mask
        q_base = self.base_q.to(device)
        p_base = self.base_p.to(device)
        lam_base = self.base_lambda.to(device) * mask
        return q, p, lam, q_base, p_base, lam_base

    def get_diff_weight(
        self,
        multiplier: float = 1.0,
        shape: tuple[int, ...] | None = None,
        device: torch.device | None = None,
    ) -> tuple[Tensor, None]:
        if device is None:
            device = self.device

        q, p, lam, q_base, p_base, lam_base = self._get_diff_weight_components(device)
        if self.isconv:
            q_2d = q.reshape(self.lora_dim, -1)
            p_2d = p.reshape(self.shape[0], self.lora_dim)
            q_base_2d = q_base.reshape(self.lora_dim, -1)
            p_base_2d = p_base.reshape(self.shape[0], self.lora_dim)
            current = p_2d @ (lam.T * q_2d)
            base = p_base_2d @ (lam_base.T * q_base_2d)
            kernel_ones = [1] * (len(self.shape) - 2)
            diff = (current - base).reshape(self.shape[0], self.shape[1], *kernel_ones)
        else:
            current = p @ (lam.T * q)
            base = p_base @ (lam_base.T * q_base)
            diff = current - base

        diff = diff * self.scalar.to(device=device, dtype=diff.dtype) * self.scale * multiplier
        if shape is not None:
            diff = diff.reshape(shape)
        return diff, None

    def get_merged_weight(
        self,
        multiplier: float = 1.0,
        shape: tuple[int, ...] | None = None,
        device: torch.device | None = None,
    ) -> tuple[Tensor, None]:
        if device is None:
            device = self.device

        diff, _ = self.get_diff_weight(multiplier=multiplier, shape=shape, device=device)
        weight = self.get_org_weight_for_compute(device)
        if weight.dtype != diff.dtype:
            weight = weight.to(diff.dtype)

        if self.isconv and diff.shape != weight.shape:
            kernel_size = weight.shape[2:]
            if all(size == 1 for size in kernel_size):
                merged = weight + diff
            else:
                diff_expanded = torch.zeros_like(weight)
                padding = self.kw_dict.get("padding", tuple(size // 2 for size in kernel_size))
                center_slices = (slice(None), slice(None)) + tuple(slice(offset, offset + 1) for offset in padding)
                diff_expanded[center_slices] = diff
                merged = weight + diff_expanded
        else:
            merged = weight + diff

        return merged, None

    def orthogonality_regularization(self) -> tuple[Tensor, Tensor]:
        """Compute TLora orthogonality regularization for the current factors."""

        if self.isconv:
            q = self.q_layer.weight.reshape(self.lora_dim, -1)
            p = self.p_layer.weight.reshape(self.shape[0], self.lora_dim)
        else:
            q = self.q_layer.weight
            p = self.p_layer.weight

        identity = torch.eye(self.lora_dim, device=q.device, dtype=q.dtype)
        p_reg = torch.sum((p.T @ p - identity) ** 2)
        q_reg = torch.sum((q @ q.T - identity) ** 2)
        return p_reg, q_reg

    def bypass_forward_diff(self, x: Tensor, scale: float = 1.0) -> Tensor:
        device = x.device
        dtype = self.dtype
        mask = self._get_mask(device)

        lam = self.lambda_layer.to(device=device, dtype=dtype) * mask.to(dtype=dtype)
        lam_base = self.base_lambda.to(device=device, dtype=dtype) * mask.to(dtype=dtype)

        if self.isconv:
            q_out = self.down_op(x.to(device=device, dtype=dtype), self.q_layer.weight.to(dtype), None, **self.kw_dict_down)
            q_out_scaled = q_out * self._reshape_mask_for_output(lam, q_out)
            current_output = self.up_op(q_out_scaled, self.p_layer.weight.to(dtype), None, **self.kw_dict_up)

            q_base_out = self.down_op(x.to(device=device, dtype=dtype), self.base_q.to(dtype), None, **self.kw_dict_down)
            q_base_scaled = q_base_out * self._reshape_mask_for_output(lam_base, q_base_out)
            base_output = self.up_op(q_base_scaled, self.base_p.to(dtype), None, **self.kw_dict_up)
        else:
            x_compute = x.to(device=device, dtype=dtype)
            q_out = self.down_op(x_compute, self.q_layer.weight.to(dtype), None)
            q_out_scaled = q_out * self._reshape_mask_for_output(lam, q_out)
            current_output = self.up_op(q_out_scaled, self.p_layer.weight.to(dtype), None)

            q_base_out = self.down_op(x_compute, self.base_q.to(dtype), None)
            q_base_scaled = q_base_out * self._reshape_mask_for_output(lam_base, q_base_out)
            base_output = self.up_op(q_base_scaled, self.base_p.to(dtype), None)

        diff = current_output - base_output
        scaled = diff * self.scalar.to(device=device, dtype=dtype) * self.scale * scale
        return self.drop(scaled).to(dtype=x.dtype)

    def bypass_forward(self, x: Tensor, scale: float = 1.0) -> Tensor:
        return self._apply_base_op(x) + self.bypass_forward_diff(x, scale=scale)

    @torch.no_grad()
    def apply_max_norm(self, max_norm: float, device=None) -> tuple[int | Tensor, Tensor]:
        diff, _ = self.get_diff_weight(multiplier=1.0, device=device)
        original_norm = diff.norm() * self.scale
        norm = torch.clamp(original_norm, max_norm / 2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = (desired / norm).to(self.scalar.device)
        scaled = norm != desired
        if scaled:
            self.scalar.mul_(ratio)
            return scaled, original_norm * ratio
        return 0, original_norm

    def forward(self, x: Tensor, *args, **kwargs):
        if self.module_dropout and self.training and torch.rand((), device=x.device).item() < self.module_dropout:
            return self._apply_base_op(x)

        if self.bypass_mode or self._has_batched_mask() or (
            self.isconv and (any(size != 1 for size in self.shape[2:]) or self.kw_dict.get("groups", 1) > 1)
        ):
            return self.bypass_forward(x, scale=self.multiplier)

        base = self.org_forward(x, *args, **kwargs)
        diff_weight, _ = self.get_diff_weight(multiplier=self.multiplier, device=x.device)
        delta = self._apply_target_op(x, diff_weight.to(dtype=self.dtype), None)
        return base + delta
