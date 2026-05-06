"""Repo-owned OFT (Orthogonal Finetuning) method implementation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from library.adapters.runtime.precision import cast_input_for_compute, restore_output_dtype


SUPPORTED_MODULE_TYPES = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
_OFT_EXPORT_WEIGHT_KEYS = (
    "oft_blocks",
    "rescale",
    "alpha",
)
_OFT_DETECTION_KEYS = ("oft_blocks",)


def factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    """Factorize one output dimension using the absorbed LyCORIS search rule."""

    if factor > 0 and (dimension % factor) == 0:
        m = factor
        n = dimension // factor
        if m > n:
            n, m = m, n
        return m, n
    if factor < 0:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        m, n = new_m, new_n
    if m > n:
        n, m = m, n
    return m, n


@dataclass(frozen=True, slots=True)
class OftConfig:
    """Repo-owned config for constructing an OFT module on one target module."""

    multiplier: float = 1.0
    factor: int | None = None
    constraint: float = 0.0
    rescaled: bool = False
    dropout: float = 0.0
    rank_dropout: float = 0.0
    module_dropout: float = 0.0
    bypass_mode: bool | None = None
    block_size_override: int | None = None

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "multiplier": self.multiplier,
            "factor": self.factor,
            "constraint": self.constraint,
            "rescaled": self.rescaled,
            "dropout": self.dropout,
            "rank_dropout": self.rank_dropout,
            "module_dropout": self.module_dropout,
            "bypass_mode": self.bypass_mode,
            "block_size_override": self.block_size_override,
        }


def _reshape_weight(weight: Tensor, shape: tuple[int, ...] | None) -> Tensor:
    if shape is None:
        return weight
    return weight.reshape(shape)


def _validate_probability(value: float, *, field_name: str) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"OFT {field_name} must be between 0.0 and 1.0 inclusive, got {value}.")


def _upper_triangle_entry_count(block_size: int) -> int:
    return block_size * (block_size - 1) // 2


def _infer_block_size_from_entry_count(entry_count: int) -> int:
    discriminant = 1 + 8 * entry_count
    root = math.isqrt(discriminant)
    if root * root != discriminant:
        raise ValueError(f"OFT compact block entry count {entry_count} does not describe a triangular block size.")
    block_size = (1 + root) // 2
    if _upper_triangle_entry_count(block_size) != entry_count:
        raise ValueError(f"OFT compact block entry count {entry_count} does not describe a triangular block size.")
    return block_size


def _infer_block_size_from_export_blocks(oft_blocks: Tensor) -> int:
    if oft_blocks.ndim == 3:
        if oft_blocks.shape[1] != oft_blocks.shape[2]:
            raise ValueError(f"OFT full block tensor must be square, got shape {tuple(oft_blocks.shape)}.")
        return int(oft_blocks.shape[1])
    if oft_blocks.ndim == 2:
        return _infer_block_size_from_entry_count(int(oft_blocks.shape[1]))
    raise ValueError(f"OFT exported blocks must be rank 2 or 3, got ndim={oft_blocks.ndim}.")


class OftModule(nn.Module):
    """OFT method module bound to one resolved target module."""

    export_weight_keys = _OFT_EXPORT_WEIGHT_KEYS
    detection_keys = _OFT_DETECTION_KEYS

    @classmethod
    def from_target_module(cls, lora_name: str, org_module: nn.Module, *, config: OftConfig) -> OftModule:
        return cls(lora_name, org_module, **config.as_kwargs())

    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,
        *,
        multiplier: float = 1.0,
        factor: int | None = None,
        constraint: float = 0.0,
        rescaled: bool = False,
        dropout: float = 0.0,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        bypass_mode: bool | None = None,
        block_size_override: int | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        if not isinstance(org_module, SUPPORTED_MODULE_TYPES):
            raise ValueError(f"{type(org_module).__name__} is not supported in OFT algo.")
        if block_size_override is None and (factor is None or factor <= 0):
            raise ValueError(f"OFT factor must be positive, got {factor}.")
        if block_size_override is not None and block_size_override <= 0:
            raise ValueError(f"OFT block_size_override must be positive, got {block_size_override}.")
        if constraint < 0.0:
            raise ValueError(f"OFT constraint must be non-negative, got {constraint}.")
        _validate_probability(dropout, field_name="dropout")
        _validate_probability(rank_dropout, field_name="rank_dropout")
        _validate_probability(module_dropout, field_name="module_dropout")

        self.lora_name = lora_name
        self.multiplier = multiplier
        self.factor = factor
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.bypass_mode = bool(bypass_mode)
        self.org_module = [org_module]
        self.org_forward = org_module.forward
        self.adapter_target = None

        if isinstance(org_module, nn.Linear):
            self.module_type = "linear"
            self.shape = (org_module.out_features, org_module.in_features)
            self.kw_dict: dict[str, Any] = {}
        elif isinstance(org_module, nn.Conv1d):
            self.module_type = "conv1d"
            self.shape = (org_module.out_channels, org_module.in_channels, *org_module.kernel_size)
            self.kw_dict = {
                "stride": org_module.stride,
                "padding": org_module.padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups,
            }
        elif isinstance(org_module, nn.Conv2d):
            self.module_type = "conv2d"
            self.shape = (org_module.out_channels, org_module.in_channels, *org_module.kernel_size)
            self.kw_dict = {
                "stride": org_module.stride,
                "padding": org_module.padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups,
            }
        else:
            self.module_type = "conv3d"
            self.shape = (org_module.out_channels, org_module.in_channels, *org_module.kernel_size)
            self.kw_dict = {
                "stride": org_module.stride,
                "padding": org_module.padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups,
            }

        out_dim = self.shape[0]
        if block_size_override is not None:
            if out_dim % block_size_override != 0:
                raise ValueError(
                    f"OFT block_size_override={block_size_override} must evenly divide output dimension {out_dim}."
                )
            self.block_size = int(block_size_override)
            self.block_num = out_dim // self.block_size
        else:
            self.block_size, self.block_num = factorization(out_dim, factor)
        self.constraint = constraint * out_dim
        self.register_buffer("alpha", torch.tensor(constraint, dtype=torch.float32))
        upper_rows, upper_cols = torch.triu_indices(self.block_size, self.block_size, offset=1)
        self.register_buffer("upper_rows", upper_rows, persistent=False)
        self.register_buffer("upper_cols", upper_cols, persistent=False)
        self.oft_blocks = nn.Parameter(torch.zeros(self.block_num, _upper_triangle_entry_count(self.block_size)))
        self.register_buffer("I", torch.eye(self.block_size, dtype=torch.float32), persistent=False)

        if rescaled:
            scale_shape = (out_dim, *[1] * (len(self.shape) - 1))
            self.rescale = nn.Parameter(torch.ones(scale_shape, dtype=torch.float32))
        else:
            self.register_parameter("rescale", None)

        self.rank_drop = nn.Identity() if rank_dropout == 0.0 else nn.Dropout(rank_dropout)

    @property
    def dtype(self) -> torch.dtype:
        return self.oft_blocks.dtype

    @property
    def device(self) -> torch.device:
        return self.oft_blocks.device

    @property
    def required_export_weight_keys(self) -> tuple[str, ...]:
        required = ["oft_blocks", "alpha"]
        if self.rescale is not None:
            required.append("rescale")
        return tuple(required)

    @classmethod
    def algo_check(cls, state_dict: Mapping[str, Tensor], lora_name: str) -> bool:
        oft_blocks = state_dict.get(f"{lora_name}.oft_blocks")
        return oft_blocks is not None and oft_blocks.ndim in {2, 3}

    @classmethod
    def extract_state_dict(cls, state_dict: Mapping[str, Tensor], lora_name: str) -> list[Tensor | None]:
        return [state_dict.get(f"{lora_name}.{key}", None) for key in cls.export_weight_keys]

    @classmethod
    def make_module_from_state_dict(
        cls,
        lora_name: str,
        orig_module: nn.Module,
        oft_blocks: Tensor | None,
        rescale: Tensor | None,
        alpha: Tensor | None,
    ) -> OftModule:
        if oft_blocks is None:
            raise ValueError(f"OFT state dict for {lora_name!r} is missing oft_blocks.")
        if alpha is None:
            raise ValueError(f"OFT state dict for {lora_name!r} is missing alpha.")

        block_size = _infer_block_size_from_export_blocks(oft_blocks)
        config = OftConfig(
            multiplier=1.0,
            factor=block_size,
            constraint=float(alpha.detach().float().item()),
            rescaled=rescale is not None,
            block_size_override=block_size,
        )
        module = cls.from_target_module(lora_name, orig_module, config=config)
        with torch.no_grad():
            module.load_export_state_dict({"oft_blocks": oft_blocks, "rescale": rescale})
        return module

    def export_state_dict(self) -> dict[str, Tensor]:
        state = {
            "alpha": self.alpha.detach(),
            "oft_blocks": self.oft_blocks.detach(),
        }
        if self.rescale is not None:
            state["rescale"] = self.rescale.detach()
        return state

    @torch.no_grad()
    def load_export_state_dict(self, weights: Mapping[str, Tensor]) -> None:
        compact_blocks = self._compact_oft_blocks(weights["oft_blocks"])
        self.oft_blocks.copy_(compact_blocks.to(device=self.oft_blocks.device, dtype=self.oft_blocks.dtype))
        if self.rescale is not None and weights.get("rescale") is not None:
            self.rescale.copy_(weights["rescale"])

    def get_org_weight_for_compute(self, device: torch.device) -> Tensor:
        return self.org_module[0].weight.to(device, non_blocking=True)

    def get_org_bias_for_compute(self, device: torch.device) -> Tensor | None:
        bias = self.org_module[0].bias
        if bias is None:
            return None
        return bias.to(device, non_blocking=True)

    def _identity_for_compute(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        return self.I.to(device=device, dtype=dtype)

    def _compact_oft_blocks(self, oft_blocks: Tensor) -> Tensor:
        if oft_blocks.ndim == 2:
            expected_shape = (self.block_num, _upper_triangle_entry_count(self.block_size))
            if tuple(oft_blocks.shape) != expected_shape:
                raise ValueError(
                    f"OFT compact block tensor must have shape {expected_shape}, got {tuple(oft_blocks.shape)}."
                )
            return oft_blocks
        if oft_blocks.ndim == 3:
            expected_shape = (self.block_num, self.block_size, self.block_size)
            if tuple(oft_blocks.shape) != expected_shape:
                raise ValueError(
                    f"OFT full block tensor must have shape {expected_shape}, got {tuple(oft_blocks.shape)}."
                )
            q = oft_blocks - oft_blocks.transpose(1, 2)
            rows = self.upper_rows.to(device=oft_blocks.device)
            cols = self.upper_cols.to(device=oft_blocks.device)
            return q[:, rows, cols]
        raise ValueError(f"OFT exported blocks must be rank 2 or 3, got ndim={oft_blocks.ndim}.")

    def _expand_oft_blocks(self, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        upper_values = self.oft_blocks.to(device=device, dtype=dtype)
        q = torch.zeros(self.block_num, self.block_size, self.block_size, device=device, dtype=dtype)
        rows = self.upper_rows.to(device=device)
        cols = self.upper_cols.to(device=device)
        q[:, rows, cols] = upper_values
        q[:, cols, rows] = -upper_values
        return q

    def _cast_for_compute(self, x: Tensor, dtype: torch.dtype) -> Tensor:
        return cast_input_for_compute(x, dtype)

    def _restore_result_dtype(self, result: Tensor, dtype: torch.dtype) -> Tensor:
        return restore_output_dtype(result, dtype)

    def _apply_target_op(self, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        if self.module_type == "linear":
            return F.linear(x, weight, bias)
        if self.module_type == "conv1d":
            return F.conv1d(x, weight, bias, **self.kw_dict)
        if self.module_type == "conv2d":
            return F.conv2d(x, weight, bias, **self.kw_dict)
        return F.conv3d(x, weight, bias, **self.kw_dict)

    def _compute_base_result(self, x: Tensor) -> Tensor:
        weight = self.get_org_weight_for_compute(x.device).to(self.dtype)
        bias = self.get_org_bias_for_compute(x.device)
        if bias is not None:
            bias = bias.to(device=x.device, dtype=self.dtype)
        result = self._apply_target_op(self._cast_for_compute(x, self.dtype), weight, bias)
        return self._restore_result_dtype(result, x.dtype)

    def _apply_plain_dropout(self, r: Tensor) -> Tensor:
        if not self.training or self.dropout == 0.0:
            return r
        keep_mask = torch.rand(self.block_num, device=r.device) >= self.dropout
        identity = self._identity_for_compute(r.device, r.dtype)
        return torch.where(keep_mask.view(self.block_num, 1, 1), r, identity)

    def _broadcast_rescale(self, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        if self.rescale is None:
            raise ValueError("OFT rescale weights were requested but this module is not configured for rescaling.")
        rescale = self.rescale.to(device=device, dtype=dtype)
        if self.module_type == "linear":
            return rescale
        return rescale.transpose(0, -1)

    def get_r(self, *, device: torch.device | None = None) -> Tensor:
        if device is None:
            device = self.device
        q = self._expand_oft_blocks(device=device, dtype=self.oft_blocks.dtype)
        normed_q = q
        if self.constraint > 0:
            q_norm = torch.norm(q) + 1e-8
            if q_norm > self.constraint:
                normed_q = q * self.constraint / q_norm

        identity = self._identity_for_compute(device, torch.float32)
        q_float = normed_q.to(torch.float32)
        return (identity + q_float) @ torch.inverse(identity - q_float)

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
                self.org_module[0].bias = nn.Parameter(merged_bias.to(device=current_weight.device, dtype=current_weight.dtype))
            else:
                existing_bias = self.org_module[0].bias
                self.org_module[0].bias.copy_(merged_bias.to(device=existing_bias.device, dtype=existing_bias.dtype))

    def _rescale_for_output_axis_tensor(self, *, device: torch.device, dtype: torch.dtype, ndim: int) -> Tensor:
        if self.rescale is None:
            raise ValueError("OFT rescale weights were requested but this module is not configured for rescaling.")
        return self.rescale.to(device=device, dtype=dtype).reshape(self.shape[0], *([1] * (ndim - 1)))

    def _transform_output_axis_tensor(
        self,
        tensor: Tensor,
        *,
        scale: float = 1.0,
        device: torch.device | None = None,
        diff: bool = False,
    ) -> Tensor:
        if device is None:
            device = tensor.device
        r = self.get_r(device=device)
        r = self._apply_plain_dropout(r)

        tensor_dtype = tensor.dtype
        target_dtype = torch.promote_types(tensor_dtype, r.dtype)
        identity = self._identity_for_compute(device, target_dtype)
        tensor = tensor.to(device=device, dtype=target_dtype)
        flat_tensor = tensor
        _, *shape = flat_tensor.shape
        reshaped_tensor = flat_tensor.view(self.block_num, self.block_size, *shape)

        base = torch.zeros_like(identity) if diff else identity
        transform = self.rank_drop(r.to(target_dtype) * scale) - scale * identity + base
        transformed = torch.einsum("k n m, k n ... -> k m ...", transform, reshaped_tensor).view(-1, *shape)

        if self.rescale is not None:
            rescale = self._rescale_for_output_axis_tensor(device=device, dtype=target_dtype, ndim=flat_tensor.ndim)
            transformed = rescale * transformed
            if diff:
                transformed = transformed + (rescale - 1) * flat_tensor
        return transformed.to(tensor_dtype)

    def make_weight(self, *, scale: float = 1.0, device: torch.device | None = None, diff: bool = False) -> Tensor:
        org_weight = self.get_org_weight_for_compute(device or self.device)
        return self._transform_output_axis_tensor(org_weight, scale=scale, device=device, diff=diff)

    def make_bias(self, *, scale: float = 1.0, device: torch.device | None = None, diff: bool = False) -> Tensor | None:
        if device is None:
            device = self.device
        org_bias = self.get_org_bias_for_compute(device)
        if org_bias is None:
            return None
        transformed_bias = self._transform_output_axis_tensor(org_bias.view(-1, 1), scale=scale, device=device, diff=diff)
        return transformed_bias.view(-1)

    def get_diff_weight(
        self, multiplier: float = 1.0, shape: tuple[int, ...] | None = None, device=None
    ) -> tuple[Tensor, Tensor | None]:
        diff = self.make_weight(scale=multiplier, device=device, diff=True)
        return _reshape_weight(diff, shape), self.make_bias(scale=multiplier, device=device, diff=True)

    def get_merged_weight(
        self, multiplier: float = 1.0, shape: tuple[int, ...] | None = None, device=None
    ) -> tuple[Tensor, Tensor | None]:
        weight = self.make_weight(scale=multiplier, device=device, diff=False)
        return _reshape_weight(weight, shape), self.make_bias(scale=multiplier, device=device, diff=False)

    @torch.no_grad()
    def apply_max_norm(self, max_norm: float, device=None) -> tuple[bool, Tensor] | tuple[int, Tensor]:
        device = device or self.device
        orig_norm = self._expand_oft_blocks(device=device, dtype=self.oft_blocks.dtype).norm()
        # Keep the same vendor-side regularization shape: clamp very small norms
        # up to half-threshold, then cap anything larger than the requested max.
        norm = torch.clamp(orig_norm, min=max_norm / 2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = desired / norm
        scaled = bool(norm != desired)
        if scaled:
            self.oft_blocks.mul_(ratio.to(self.oft_blocks.device))
            return scaled, orig_norm * ratio
        return False, orig_norm

    @torch.no_grad()
    def get_norm(self, device=None) -> Tensor:
        device = device or self.device
        return self._expand_oft_blocks(device=device, dtype=self.oft_blocks.dtype).norm()

    def _bypass_forward(self, x: Tensor, *, scale: float = 1.0, diff: bool = False) -> Tensor:
        r = self.get_r(device=x.device)
        r = self._apply_plain_dropout(r)
        base_output = self._compute_base_result(x)
        compute_dtype = torch.promote_types(base_output.dtype, r.dtype)
        working_output = base_output
        if self.module_type != "linear":
            working_output = working_output.transpose(1, -1)
        working_output = working_output.to(compute_dtype)
        *shape, _ = working_output.shape
        flat_output = working_output.reshape(*shape, -1)
        reshaped_output = flat_output.reshape(*shape, self.block_num, self.block_size)

        identity = self._identity_for_compute(x.device, compute_dtype)
        # Plain dropout already acts on the OFT block transforms in `_apply_plain_dropout`.
        # Reapplying an output-shaped mask here only recreates the broken vendor path and
        # introduces invalid broadcasting between batch/output dims and the block matrices.
        transform = r.to(compute_dtype) * scale + (1 - scale) * identity
        transformed = torch.einsum("k n m, ... k n -> ... k m", transform, reshaped_output).reshape(*shape, -1)
        if diff:
            out = transformed - flat_output
        else:
            out = transformed

        if self.rescale is not None:
            rescale = self._broadcast_rescale(device=x.device, dtype=compute_dtype)
            out = rescale * out + (rescale - 1) * flat_output

        if self.module_type != "linear":
            out = out.transpose(1, -1)
        return self._restore_result_dtype(out, base_output.dtype)

    def bypass_forward_diff(self, x: Tensor, scale: float = 1.0) -> Tensor:
        return self._bypass_forward(x, scale=scale, diff=True)

    def bypass_forward(self, x: Tensor, scale: float = 1.0) -> Tensor:
        return self._bypass_forward(x, scale=scale, diff=False)

    def forward(self, x: Tensor, *args, **kwargs):
        if self.module_dropout and self.training and torch.rand(1).item() < self.module_dropout:
            return self._compute_base_result(x)

        if self.bypass_mode:
            return self.bypass_forward(x, scale=self.multiplier)

        weight, bias = self.get_merged_weight(multiplier=self.multiplier, shape=self.shape, device=x.device)
        compute_dtype = weight.dtype
        if bias is not None:
            bias = bias.to(device=x.device, dtype=compute_dtype)
        result = self._apply_target_op(self._cast_for_compute(x, compute_dtype), weight.to(compute_dtype), bias)
        return self._restore_result_dtype(result, x.dtype)
