"""Repo-owned DyLoRA method implementation."""

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
_DYLORA_EXPORT_WEIGHT_KEYS = ("lora_up.weight", "lora_down.weight", "alpha", "block_size")
_DYLORA_DETECTION_KEYS = ("lora_up.weight", "lora_down.weight")


def _reshape_diff_weight(diff_weight: Tensor, shape: tuple[int, ...] | None) -> Tensor:
    if shape is None:
        return diff_weight
    return diff_weight.reshape(shape)


def _as_python_int(value: Tensor | None, *, default: int) -> int:
    if value is None:
        return default
    return int(value.detach().cpu().item())


@dataclass(frozen=True, slots=True)
class DyloraConfig:
    """Repo-owned config for constructing a DyLoRA module on one target module."""

    multiplier: float = 1.0
    lora_dim: int | None = None
    alpha: float | Tensor | None = 1
    block_size: int = 1
    module_dropout: float = 0.0
    bypass_mode: bool | None = None

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "multiplier": self.multiplier,
            "lora_dim": self.lora_dim,
            "alpha": self.alpha,
            "block_size": self.block_size,
            "module_dropout": self.module_dropout,
            "bypass_mode": self.bypass_mode,
        }


class DyloraModule(nn.Module):
    """DyLoRA method module bound to one resolved target module."""

    export_weight_keys = _DYLORA_EXPORT_WEIGHT_KEYS
    detection_keys = _DYLORA_DETECTION_KEYS

    @classmethod
    def from_target_module(cls, lora_name: str, org_module: nn.Module, *, config: DyloraConfig) -> DyloraModule:
        return cls(lora_name, org_module, **config.as_kwargs())

    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,
        *,
        multiplier: float = 1.0,
        lora_dim: int | None = None,
        alpha: float | Tensor | None = 1,
        block_size: int = 1,
        module_dropout: float = 0.0,
        bypass_mode: bool | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        if not isinstance(org_module, SUPPORTED_MODULE_TYPES):
            raise ValueError(f"{type(org_module).__name__} is not supported in DyLoRA algo.")
        if lora_dim is None or lora_dim <= 0:
            raise ValueError(f"DyLoRA rank must be positive, got {lora_dim}.")
        if block_size <= 0:
            raise ValueError(f"DyLoRA block_size must be positive, got {block_size}.")
        if lora_dim % block_size != 0:
            raise ValueError(f"DyLoRA rank {lora_dim} must be divisible by block_size {block_size}.")

        self.lora_name = lora_name
        self.multiplier = multiplier
        self.module_dropout = module_dropout
        self.bypass_mode = bool(bypass_mode)
        self.org_module = [org_module]
        self.org_forward = org_module.forward
        self.adapter_target = None

        self.block_size = block_size
        self.lora_dim = lora_dim
        self.block_count = lora_dim // block_size

        if isinstance(org_module, nn.Linear):
            self.module_type = "linear"
            self.shape = (org_module.out_features, org_module.in_features)
            self.out_dim = org_module.out_features
            self.in_dim = org_module.in_features
            self.op = F.linear
            self.kw_dict: dict[str, Any] = {}
            up_shape = (self.block_count, self.out_dim, self.block_size)
            down_shape = (self.block_count, self.block_size, self.in_dim)
        else:
            self.out_dim = org_module.out_channels
            self.in_dim = org_module.in_channels
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            dilation = org_module.dilation
            groups = org_module.groups
            self.shape = (self.out_dim, self.in_dim, *kernel_size)
            self.kw_dict = {
                "stride": stride,
                "padding": padding,
                "dilation": dilation,
                "groups": groups,
            }
            up_kernel = (1,) * len(kernel_size)
            down_shape = (self.block_count, self.block_size, self.in_dim, *kernel_size)
            up_shape = (self.block_count, self.out_dim, self.block_size, *up_kernel)
            if isinstance(org_module, nn.Conv1d):
                self.module_type = "conv1d"
                self.op = F.conv1d
            elif isinstance(org_module, nn.Conv2d):
                self.module_type = "conv2d"
                self.op = F.conv2d
            else:
                self.module_type = "conv3d"
                self.op = F.conv3d

        self.lora_down_blocks = nn.Parameter(torch.empty(down_shape, device=org_module.weight.device, dtype=org_module.weight.dtype))
        self.lora_up_blocks = nn.Parameter(torch.empty(up_shape, device=org_module.weight.device, dtype=org_module.weight.dtype))

        if isinstance(alpha, Tensor):
            alpha = float(alpha.detach().float().item())
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = float(alpha) / self.lora_dim
        self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32))

        nn.init.kaiming_uniform_(self.lora_down_blocks, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up_blocks)

    @property
    def dtype(self) -> torch.dtype:
        return self.lora_down_blocks.dtype

    @property
    def device(self) -> torch.device:
        return self.lora_down_blocks.device

    @property
    def required_export_weight_keys(self) -> tuple[str, ...]:
        return ("lora_up.weight", "lora_down.weight", "alpha")

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
        alpha: Tensor,
        block_size: Tensor | None,
    ) -> DyloraModule:
        inferred_rank = down.size(0)
        inferred_block_size = _as_python_int(block_size, default=1)
        if inferred_rank % inferred_block_size != 0:
            raise ValueError(
                f"DyLoRA state dict rank {inferred_rank} is not divisible by stored block_size {inferred_block_size}."
            )

        config = DyloraConfig(
            multiplier=1.0,
            lora_dim=inferred_rank,
            alpha=float(alpha.detach().float().item()),
            block_size=inferred_block_size,
        )
        module = cls.from_target_module(lora_name, orig_module, config=config)
        module.load_export_state_dict(
            {
                "lora_up.weight": up,
                "lora_down.weight": down,
                "alpha": alpha,
                **({"block_size": block_size} if block_size is not None else {}),
            }
        )
        return module

    def export_state_dict(self) -> dict[str, Tensor]:
        return {
            "alpha": cast(Tensor, self.alpha).detach(),
            "block_size": torch.tensor(self.block_size, dtype=torch.int64, device=self.device),
            "lora_up.weight": self._assemble_up_blocks().detach(),
            "lora_down.weight": self._assemble_down_blocks().detach(),
        }

    @torch.no_grad()
    def load_export_state_dict(self, weights: Mapping[str, Tensor]) -> None:
        exported_block_size = weights.get("block_size")
        if exported_block_size is not None:
            block_size_value = _as_python_int(exported_block_size, default=self.block_size)
            if block_size_value != self.block_size:
                raise ValueError(
                    f"DyLoRA block_size mismatch while loading {self.lora_name!r}: "
                    f"runtime expects {self.block_size}, weights store {block_size_value}."
                )

        self.lora_up_blocks.copy_(self._split_up_weight(weights["lora_up.weight"]).to(device=self.device, dtype=self.dtype))
        self.lora_down_blocks.copy_(self._split_down_weight(weights["lora_down.weight"]).to(device=self.device, dtype=self.dtype))

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

    def _assemble_up_blocks(self, active_blocks: int | None = None, *, train_current_only: bool = False) -> Tensor:
        block_limit = active_blocks if active_blocks is not None else self.block_count
        blocks = []
        for index in range(block_limit):
            block = self.lora_up_blocks[index]
            if train_current_only and index < block_limit - 1:
                # DyLoRA trains only the sampled block while still rebuilding
                # the prefix up to that block for the effective update.
                block = block.detach()
            blocks.append(block)
        return torch.cat(blocks, dim=1)

    def _assemble_down_blocks(self, active_blocks: int | None = None, *, train_current_only: bool = False) -> Tensor:
        block_limit = active_blocks if active_blocks is not None else self.block_count
        blocks = []
        for index in range(block_limit):
            block = self.lora_down_blocks[index]
            if train_current_only and index < block_limit - 1:
                block = block.detach()
            blocks.append(block)
        return torch.cat(blocks, dim=0)

    def _split_up_weight(self, up_weight: Tensor) -> Tensor:
        if up_weight.size(1) != self.lora_dim:
            raise ValueError(
                f"DyLoRA up weight rank mismatch for {self.lora_name!r}: expected {self.lora_dim}, got {up_weight.size(1)}."
            )
        chunks = list(up_weight.split(self.block_size, dim=1))
        if len(chunks) != self.block_count:
            raise ValueError(
                f"DyLoRA up weight block count mismatch for {self.lora_name!r}: expected {self.block_count}, got {len(chunks)}."
            )
        return torch.stack(chunks, dim=0)

    def _split_down_weight(self, down_weight: Tensor) -> Tensor:
        if down_weight.size(0) != self.lora_dim:
            raise ValueError(
                f"DyLoRA down weight rank mismatch for {self.lora_name!r}: expected {self.lora_dim}, got {down_weight.size(0)}."
            )
        chunks = list(down_weight.split(self.block_size, dim=0))
        if len(chunks) != self.block_count:
            raise ValueError(
                f"DyLoRA down weight block count mismatch for {self.lora_name!r}: expected {self.block_count}, got {len(chunks)}."
            )
        return torch.stack(chunks, dim=0)

    def _sample_active_blocks(self) -> int:
        return int(torch.randint(1, self.block_count + 1, (1,), device=self.device).item())

    def _dynamic_scale(self, active_blocks: int | None) -> float:
        if active_blocks is None:
            return self.scale
        active_rank = active_blocks * self.block_size
        return self.scale * math.sqrt(self.lora_dim / active_rank)

    def make_weight(
        self,
        device: torch.device | None = None,
        *,
        active_blocks: int | None = None,
        train_current_only: bool = False,
    ) -> Tensor:
        up = self._assemble_up_blocks(active_blocks, train_current_only=train_current_only)
        down = self._assemble_down_blocks(active_blocks, train_current_only=train_current_only)
        if device is not None:
            up = up.to(device)
            down = down.to(device)
        weight = up.reshape(self.out_dim, -1) @ down.reshape(down.size(0), -1)
        return weight.view(self.shape)

    def get_weight(self, shape: tuple[int, ...] | None) -> Tensor:
        return _reshape_diff_weight(self.make_weight(self.device) * self.scale, shape)

    def get_diff_weight(
        self,
        multiplier: float = 1.0,
        shape: tuple[int, ...] | None = None,
        device=None,
        *,
        active_blocks: int | None = None,
        train_current_only: bool = False,
    ) -> tuple[Tensor, None]:
        diff = self.make_weight(
            device or self.device,
            active_blocks=active_blocks,
            train_current_only=train_current_only,
        ) * (self._dynamic_scale(active_blocks) * multiplier)
        diff = _reshape_diff_weight(diff, shape)
        if device is not None:
            diff = diff.to(device)
        return diff, None

    def get_merged_weight(self, multiplier: float = 1.0, shape: tuple[int, ...] | None = None, device=None) -> tuple[Tensor, None]:
        if device is None:
            device = self.device
        diff = self.get_diff_weight(multiplier=multiplier, shape=shape, device=device)[0]
        weight = self.get_org_weight_for_compute(device)
        if weight.dtype != diff.dtype:
            weight = weight.to(diff.dtype)
        return weight + diff, None

    @torch.no_grad()
    def apply_max_norm(self, max_norm: float, device=None) -> tuple[int | Tensor, Tensor]:
        orig_norm = self.get_diff_weight(device=device or self.device)[0].norm()
        norm = torch.clamp(orig_norm, min=max_norm / 2)
        desired = torch.clamp(norm, max=max_norm)
        if torch.equal(norm, desired):
            return 0, orig_norm
        ratio = (desired / norm).to(self.device)
        gain = torch.sqrt(ratio)
        self.lora_up_blocks.mul_(gain)
        self.lora_down_blocks.mul_(gain)
        return 1, orig_norm * ratio

    @torch.no_grad()
    def get_norm(self, device=None) -> Tensor:
        return self.get_diff_weight(device=device or self.device)[0].norm()

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

    def bypass_forward_diff(
        self,
        x: Tensor,
        scale: float = 1.0,
        *,
        active_blocks: int | None = None,
        train_current_only: bool = False,
    ) -> Tensor:
        compute_dtype = self.dtype
        x_compute = self._cast_for_compute(x, compute_dtype)
        diff_weight = self.get_diff_weight(
            multiplier=scale,
            shape=self.shape,
            device=x.device,
            active_blocks=active_blocks,
            train_current_only=train_current_only,
        )[0].to(dtype=compute_dtype)
        return self._apply_target_op(x_compute, diff_weight, None)

    def bypass_forward(
        self,
        x: Tensor,
        scale: float = 1.0,
        *,
        active_blocks: int | None = None,
        train_current_only: bool = False,
    ) -> Tensor:
        compute_dtype = self.dtype
        x_compute = self._cast_for_compute(x, compute_dtype)
        org_weight = self.get_org_weight_for_compute(x.device).to(dtype=compute_dtype)
        bias = self.get_org_bias_for_compute(x.device)
        if bias is not None:
            bias = bias.to(dtype=compute_dtype)
        base = self._apply_target_op(x_compute, org_weight, bias)
        diff = self.bypass_forward_diff(
            x,
            scale=scale,
            active_blocks=active_blocks,
            train_current_only=train_current_only,
        )
        result = base + diff.to(base.dtype)
        return restore_output_dtype(result, x.dtype)

    def forward(self, x: Tensor, *args, **kwargs):
        if self.module_dropout > 0 and self.training and torch.rand(1).item() < self.module_dropout:
            org_weight = self.get_org_weight_for_compute(x.device).to(dtype=self.dtype)
            bias = self.get_org_bias_for_compute(x.device)
            if bias is not None:
                bias = bias.to(dtype=self.dtype)
            result = self._apply_target_op(self._cast_for_compute(x, self.dtype), org_weight, bias)
            return restore_output_dtype(result, x.dtype)

        active_blocks = self._sample_active_blocks() if self.training else None
        # DyLoRA's training trick updates only the sampled block while using
        # the prefix through that block to build the effective low-rank update.
        train_current_only = self.training

        if self.bypass_mode:
            return self.bypass_forward(
                x,
                scale=self.multiplier,
                active_blocks=active_blocks,
                train_current_only=train_current_only,
            )

        compute_dtype = self.dtype
        x_compute = self._cast_for_compute(x, compute_dtype)
        org_weight = self.get_org_weight_for_compute(x.device).to(dtype=compute_dtype)
        diff_weight = self.get_diff_weight(
            multiplier=self.multiplier,
            shape=self.shape,
            device=x.device,
            active_blocks=active_blocks,
            train_current_only=train_current_only,
        )[0].to(dtype=compute_dtype)
        bias = self.get_org_bias_for_compute(x.device)
        if bias is not None:
            bias = bias.to(dtype=compute_dtype)
        result = self._apply_target_op(x_compute, org_weight + diff_weight, bias)
        return restore_output_dtype(result, x.dtype)
