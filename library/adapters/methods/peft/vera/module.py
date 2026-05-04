from __future__ import annotations

import math
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.init import _calculate_correct_fan

try:
    from transformers.pytorch_utils import Conv1D as TransformersConv1D
except (ImportError, AttributeError):  # pragma: no cover - optional dependency
    TransformersConv1D = None

_SUPPORTED_MODULE_TYPES_LIST: list[type[nn.Module]] = [nn.Linear]
if TransformersConv1D is not None:
    _SUPPORTED_MODULE_TYPES_LIST.append(TransformersConv1D)
SUPPORTED_MODULE_TYPES = tuple(_SUPPORTED_MODULE_TYPES_LIST)

_VERA_SHARED_EXPORT_WEIGHT_KEYS = ("vera_A", "vera_B")
_VERA_MODULE_EXPORT_WEIGHT_KEYS = ("vera_lambda_b", "vera_lambda_d")


def _kaiming_uniform_with_generator(shape: tuple[int, ...], *, generator: torch.Generator) -> Tensor:
    tensor = torch.empty(shape, dtype=torch.float32)
    fan = _calculate_correct_fan(tensor, "fan_in")
    gain = math.sqrt(2.0)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        return tensor.uniform_(-bound, bound, generator=generator)


def _is_transformers_conv1d(module: nn.Module) -> bool:
    return TransformersConv1D is not None and isinstance(module, TransformersConv1D)


def get_target_feature_shape(module: nn.Module) -> tuple[int, int]:
    if isinstance(module, nn.Linear):
        return int(module.out_features), int(module.in_features)
    if _is_transformers_conv1d(module):
        weight_shape = getattr(module.weight, "ds_shape", None) or tuple(module.weight.shape)
        if len(weight_shape) != 2:
            raise ValueError(
                f"VeRA Conv1D target expects a 2D weight tensor, got shape {tuple(weight_shape)} for {type(module).__name__}."
            )
        in_features, out_features = (int(weight_shape[0]), int(weight_shape[1]))
        return out_features, in_features
    raise ValueError(f"{type(module).__name__} is not supported in VeRA.")


@dataclass(slots=True)
class VeraConfig:
    multiplier: float = 1.0
    rank: int = 256
    dropout: float = 0.0
    d_initial: float = 0.1
    init_weights: bool = True

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "multiplier": self.multiplier,
            "rank": self.rank,
            "dropout": self.dropout,
            "d_initial": self.d_initial,
            "init_weights": self.init_weights,
        }


class VeraSharedProjectionBank(nn.Module):
    """Runtime-owned VeRA projection tensors shared across adapted targets."""

    export_weight_keys = _VERA_SHARED_EXPORT_WEIGHT_KEYS

    def __init__(
        self,
        *,
        rank: int,
        max_in_features: int,
        max_out_features: int,
        projection_prng_key: int = 0,
        save_projection: bool = True,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"VeRA rank must be positive, got {rank}.")
        if max_in_features <= 0 or max_out_features <= 0:
            raise ValueError(
                "VeRA shared projection bank requires positive feature sizes, "
                f"got in_features={max_in_features}, out_features={max_out_features}."
            )

        self.rank = int(rank)
        self.max_in_features = int(max_in_features)
        self.max_out_features = int(max_out_features)
        self.projection_prng_key = int(projection_prng_key)
        self.save_projection = bool(save_projection)

        vera_A, vera_B = self._initialize_projection_tensors(projection_prng_key=self.projection_prng_key)
        self.register_buffer("vera_A", vera_A, persistent=self.save_projection)
        self.register_buffer("vera_B", vera_B, persistent=self.save_projection)

    def _initialize_projection_tensors(self, *, projection_prng_key: int) -> tuple[Tensor, Tensor]:
        generator = torch.Generator(device="cpu").manual_seed(int(projection_prng_key))
        vera_A = _kaiming_uniform_with_generator((self.rank, self.max_in_features), generator=generator)
        vera_B = _kaiming_uniform_with_generator((self.max_out_features, self.rank), generator=generator)
        return vera_A, vera_B

    @classmethod
    def from_state_dict(
        cls,
        vera_A: Tensor,
        vera_B: Tensor,
        *,
        save_projection: bool = True,
    ) -> VeraSharedProjectionBank:
        bank = cls(
            rank=int(vera_A.shape[0]),
            max_in_features=int(vera_A.shape[1]),
            max_out_features=int(vera_B.shape[0]),
            projection_prng_key=0,
            save_projection=save_projection,
        )
        bank.load_export_state_dict({"vera_A": vera_A, "vera_B": vera_B})
        return bank

    def export_state_dict(self) -> dict[str, Tensor]:
        if not self.save_projection:
            return {}
        return {
            "vera_A": self.vera_A.detach(),
            "vera_B": self.vera_B.detach(),
        }

    @torch.no_grad()
    def reset_from_prng_key(self, projection_prng_key: int | None = None) -> None:
        if projection_prng_key is not None:
            self.projection_prng_key = int(projection_prng_key)
        vera_A, vera_B = self._initialize_projection_tensors(projection_prng_key=self.projection_prng_key)
        self.vera_A.copy_(vera_A.to(device=self.vera_A.device, dtype=self.vera_A.dtype))
        self.vera_B.copy_(vera_B.to(device=self.vera_B.device, dtype=self.vera_B.dtype))

    @torch.no_grad()
    def load_export_state_dict(self, weights: Mapping[str, Tensor]) -> None:
        vera_A = weights["vera_A"]
        vera_B = weights["vera_B"]
        expected_A_shape = tuple(self.vera_A.shape)
        expected_B_shape = tuple(self.vera_B.shape)
        if tuple(vera_A.shape) != expected_A_shape:
            raise ValueError(
                "VeRA shared projection shape mismatch for vera_A: "
                f"runtime expects {expected_A_shape}, weights store {tuple(vera_A.shape)}."
            )
        if tuple(vera_B.shape) != expected_B_shape:
            raise ValueError(
                "VeRA shared projection shape mismatch for vera_B: "
                f"runtime expects {expected_B_shape}, weights store {tuple(vera_B.shape)}."
            )
        self.vera_A.copy_(vera_A.to(device=self.vera_A.device, dtype=self.vera_A.dtype))
        self.vera_B.copy_(vera_B.to(device=self.vera_B.device, dtype=self.vera_B.dtype))

    def get_projection_slices(
        self,
        *,
        in_features: int,
        out_features: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        if in_features > self.max_in_features:
            raise ValueError(
                f"VeRA target requires in_features={in_features}, but the shared bank only stores {self.max_in_features}."
            )
        if out_features > self.max_out_features:
            raise ValueError(
                f"VeRA target requires out_features={out_features}, but the shared bank only stores {self.max_out_features}."
            )
        vera_A = self.vera_A[:, :in_features].to(device=device, dtype=dtype)
        vera_B = self.vera_B[:out_features, :].to(device=device, dtype=dtype)
        return vera_A, vera_B


class VeraModule(nn.Module):
    """Repo-owned VeRA module bound to one resolved Linear target."""

    export_weight_keys = _VERA_MODULE_EXPORT_WEIGHT_KEYS
    detection_keys = _VERA_MODULE_EXPORT_WEIGHT_KEYS
    required_export_weight_keys = _VERA_MODULE_EXPORT_WEIGHT_KEYS

    @classmethod
    def from_target_module(
        cls,
        lora_name: str,
        org_module: nn.Module,
        *,
        shared_bank: VeraSharedProjectionBank,
        config: VeraConfig,
    ) -> VeraModule:
        return cls(lora_name, org_module, shared_bank=shared_bank, **config.as_kwargs())

    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,
        *,
        shared_bank: VeraSharedProjectionBank,
        multiplier: float = 1.0,
        rank: int = 256,
        dropout: float = 0.0,
        d_initial: float = 0.1,
        init_weights: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        if not isinstance(org_module, SUPPORTED_MODULE_TYPES):
            raise ValueError(f"{type(org_module).__name__} is not supported in VeRA.")
        if rank <= 0:
            raise ValueError(f"VeRA rank must be positive, got {rank}.")

        self.lora_name = lora_name
        self.multiplier = float(multiplier)
        self.rank = int(rank)
        self.dropout = float(dropout)
        self.d_initial = float(d_initial)
        self.adapter_target = None
        self.org_module = [org_module]
        self.org_forward = org_module.forward
        self.out_features, self.in_features = get_target_feature_shape(org_module)
        self.fan_in_fan_out = _is_transformers_conv1d(org_module)
        self.shape = (self.out_features, self.in_features)
        self.dropout_layer = nn.Dropout(p=self.dropout) if self.dropout > 0.0 else nn.Identity()
        self._shared_bank_ref = weakref.ref(shared_bank)

        self.vera_lambda_b = nn.Parameter(torch.ones(self.out_features))
        self.vera_lambda_d = nn.Parameter(torch.randn(self.rank))
        if init_weights:
            self.reset_vera_parameters(d_initial=self.d_initial)

    @property
    def shared_bank(self) -> VeraSharedProjectionBank:
        shared_bank = self._shared_bank_ref()
        if shared_bank is None:
            raise RuntimeError(f"VeRA module {self.lora_name!r} lost access to its shared projection bank.")
        return shared_bank

    @property
    def dtype(self) -> torch.dtype:
        return self.org_module[0].weight.dtype

    @property
    def device(self) -> torch.device:
        return self.org_module[0].weight.device

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
        shared_bank: VeraSharedProjectionBank,
        lambda_b: Tensor,
        lambda_d: Tensor,
    ) -> VeraModule:
        config = VeraConfig(
            multiplier=1.0,
            rank=int(lambda_d.shape[0]),
            init_weights=False,
        )
        module = cls.from_target_module(lora_name, orig_module, shared_bank=shared_bank, config=config)
        module.load_export_state_dict(
            {
                "vera_lambda_b": lambda_b,
                "vera_lambda_d": lambda_d,
            }
        )
        return module

    def reset_vera_parameters(self, *, d_initial: float = 0.1) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.vera_lambda_d).fill_(float(d_initial))
            nn.init.zeros_(self.vera_lambda_b)

    def export_state_dict(self) -> dict[str, Tensor]:
        return {
            "vera_lambda_b": self.vera_lambda_b.detach(),
            "vera_lambda_d": self.vera_lambda_d.detach(),
        }

    @torch.no_grad()
    def load_export_state_dict(self, weights: Mapping[str, Tensor]) -> None:
        lambda_b = weights["vera_lambda_b"]
        lambda_d = weights["vera_lambda_d"]
        if tuple(lambda_b.shape) != tuple(self.vera_lambda_b.shape):
            raise ValueError(
                f"VeRA lambda_b shape mismatch while loading {self.lora_name!r}: "
                f"runtime expects {tuple(self.vera_lambda_b.shape)}, weights store {tuple(lambda_b.shape)}."
            )
        if tuple(lambda_d.shape) != tuple(self.vera_lambda_d.shape):
            raise ValueError(
                f"VeRA lambda_d shape mismatch while loading {self.lora_name!r}: "
                f"runtime expects {tuple(self.vera_lambda_d.shape)}, weights store {tuple(lambda_d.shape)}."
            )

        self.vera_lambda_b.copy_(lambda_b.to(device=self.vera_lambda_b.device, dtype=self.vera_lambda_b.dtype))
        self.vera_lambda_d.copy_(lambda_d.to(device=self.vera_lambda_d.device, dtype=self.vera_lambda_d.dtype))

    def apply_to(self) -> None:
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward  # type: ignore[assignment]

    def _cast_for_compute(self, x: Tensor) -> Tensor:
        if x.dtype == self.dtype:
            return x
        return x.to(self.dtype)

    def _get_compute_device(self, device: torch.device | None) -> torch.device:
        if device is not None:
            return device
        return self.device

    def _get_delta_components(
        self,
        *,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        compute_dtype = self.vera_lambda_d.dtype
        vera_A, vera_B = self.shared_bank.get_projection_slices(
            in_features=self.in_features,
            out_features=self.out_features,
            device=device,
            dtype=compute_dtype,
        )
        lambda_d = self.vera_lambda_d.to(device=device, dtype=compute_dtype)
        lambda_b = self.vera_lambda_b.to(device=device, dtype=compute_dtype)

        if device.type == "cpu" and compute_dtype in {torch.float16, torch.bfloat16}:
            vera_A = vera_A.float()
            vera_B = vera_B.float()
            lambda_d = lambda_d.float()
            lambda_b = lambda_b.float()
        return vera_A, vera_B, lambda_b, lambda_d

    def forward(self, x: Tensor) -> Tensor:
        compute_input = self._cast_for_compute(x)
        org_forwarded = self.org_forward(compute_input)

        vera_A, vera_B, lambda_b, lambda_d = self._get_delta_components(device=compute_input.device)
        adapter_input = compute_input.to(dtype=lambda_d.dtype)
        adapter_input = self.dropout_layer(adapter_input)
        adapter_hidden = F.linear(adapter_input, vera_A)
        adapter_hidden = adapter_hidden * lambda_d
        adapter_out = F.linear(adapter_hidden, vera_B)
        adapter_out = adapter_out * lambda_b
        if adapter_out.dtype != org_forwarded.dtype:
            adapter_out = adapter_out.to(org_forwarded.dtype)
        output = org_forwarded + adapter_out * self.multiplier
        if output.dtype != x.dtype:
            output = output.to(x.dtype)
        return output

    def get_diff_weight(
        self,
        multiplier: float = 1.0,
        shape: tuple[int, ...] | torch.Size | None = None,
        device: torch.device | None = None,
    ) -> tuple[Tensor, None]:
        compute_device = self._get_compute_device(device)
        vera_A, vera_B, lambda_b, lambda_d = self._get_delta_components(device=compute_device)
        diff_weight = (lambda_b.unsqueeze(-1) * vera_B) @ (lambda_d.unsqueeze(-1) * vera_A)
        diff_weight = diff_weight * multiplier
        if diff_weight.dtype != self.dtype and self.dtype not in {torch.float16, torch.bfloat16}:
            diff_weight = diff_weight.to(self.dtype)
        if self.fan_in_fan_out:
            # Transformers Conv1D stores weights as (in_features, out_features)
            # while the VeRA delta is computed in logical Linear order.
            diff_weight = diff_weight.transpose(0, 1)
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
        merged_weight = org_weight + diff_weight.to(dtype=org_weight.dtype)
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
