"""Repo-owned LoKr (LoRA-Kronecker Product) method implementation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn


SUPPORTED_MODULE_TYPES = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
LOKR_INIT_MODES = ("lycoris_legacy", "zero_delta_he", "random_nonzero")
_LOKR_EXPORT_WEIGHT_KEYS = (
    "lokr_w1",
    "lokr_w1_a",
    "lokr_w1_b",
    "lokr_w2",
    "lokr_w2_a",
    "lokr_w2_b",
    "lokr_t2",
    "alpha",
    "dora_scale",
)
_LOKR_DETECTION_KEYS = ("lokr_w1", "lokr_w1_a")


@dataclass(frozen=True, slots=True)
class LokrConfig:
    """Repo-owned config for constructing a LoKr module on one target module."""

    multiplier: float = 1.0
    lora_dim: int | None = None
    alpha: float | Tensor | None = 1
    dropout: float = 0.0
    rank_dropout: float = 0.0
    module_dropout: float = 0.0
    use_tucker: bool = False
    use_scalar: bool = False
    init_mode: str = "lycoris_legacy"
    decompose_both: bool = False
    factor: int = -1
    full_matrix: bool = False
    rank_dropout_scale: bool = False
    weight_decompose: bool = False
    wd_on_output: bool = True
    bypass_mode: bool | None = None
    rs_lora: bool = False
    unbalanced_factorization: bool = False
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
            "decompose_both": self.decompose_both,
            "factor": self.factor,
            "full_matrix": self.full_matrix,
            "rank_dropout_scale": self.rank_dropout_scale,
            "weight_decompose": self.weight_decompose,
            "wd_on_output": self.wd_on_output,
            "bypass_mode": self.bypass_mode,
            "rs_lora": self.rs_lora,
            "unbalanced_factorization": self.unbalanced_factorization,
            "orthogonalize": self.orthogonalize,
        }


def factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    """Split a dimension into the LoKr Kronecker scale and weight factors."""

    if dimension <= 0:
        raise ValueError(f"LoKr factorization dimension must be positive, got {dimension}.")
    if factor == 0:
        raise ValueError("LoKr factor must be -1 or a positive divisor/search bound.")
    if factor > 0 and dimension % factor == 0:
        return factor, dimension // factor

    search_bound = dimension if factor < 0 else factor
    first, second = 1, dimension
    best_sum = first + second
    while first < second:
        candidate = first + 1
        while dimension % candidate != 0:
            candidate += 1
        other = dimension // candidate
        if candidate + other > best_sum or candidate > search_bound:
            break
        first, second = candidate, other
        best_sum = first + second

    if first > second:
        first, second = second, first
    return first, second


def _make_kron_weight(w1: Tensor, w2: Tensor, scale: float | Tensor) -> Tensor:
    for _ in range(w2.dim() - w1.dim()):
        w1 = w1.unsqueeze(-1)
    weight = torch.kron(w1, w2.contiguous())
    if scale != 1:
        weight = weight * scale
    return weight


def _rebuild_tucker(tensor: Tensor, wa: Tensor, wb: Tensor) -> Tensor:
    return torch.einsum("i j ..., i p, j r -> p r ...", tensor, wa, wb)


def _reshape_diff_weight(diff_weight: Tensor, shape: tuple[int, ...] | None) -> Tensor:
    if shape is None:
        return diff_weight
    return diff_weight.reshape(shape)


def _infer_lokr_factor(
    *,
    w1: Tensor | None,
    w1_a: Tensor | None,
    w1_b: Tensor | None,
    w2: Tensor | None,
    w2_a: Tensor | None,
    w2_b: Tensor | None,
) -> int:
    if w1 is None:
        if w1_a is None or w1_b is None:
            raise ValueError("LoKr state dict must include either lokr_w1 or both lokr_w1_a/lokr_w1_b.")
        w1_shape = (w1_a.size(0), w1_b.size(1))
    else:
        w1_shape = w1.shape

    if w2 is None:
        if w2_a is None or w2_b is None:
            raise ValueError("LoKr state dict must include either lokr_w2 or both lokr_w2_a/lokr_w2_b.")
        w2_shape = (w2_a.size(0), w2_b.size(1))
    else:
        w2_shape = w2.shape
    out_dim = w1_shape[0] * w2_shape[0]
    in_dim = w1_shape[1] * w2_shape[1]
    shape_s = [w1_shape[0], w1_shape[1]]

    if shape_s[0] == factorization(out_dim, -1)[0] and shape_s[1] == factorization(in_dim, -1)[0]:
        return -1

    shape_group_1 = (w1_shape[0], w2_shape[0])
    shape_group_2 = (w1_shape[1], w2_shape[1])
    weight_shape = (out_dim, in_dim)
    factor1 = max(w1_shape)
    factor2 = max(w2_shape)
    if weight_shape[0] % factor1 == 0 and weight_shape[1] % factor1 == 0 and factor1 in shape_group_1 and factor1 in shape_group_2:
        return factor1
    if weight_shape[0] % factor2 == 0 and weight_shape[1] % factor2 == 0 and factor2 in shape_group_1 and factor2 in shape_group_2:
        return factor2
    return min(factor1, factor2)


class LokrModule(nn.Module):
    """LoKr method module bound to one resolved target module."""

    export_weight_keys = _LOKR_EXPORT_WEIGHT_KEYS
    detection_keys = _LOKR_DETECTION_KEYS

    @classmethod
    def from_target_module(cls, lora_name: str, org_module: nn.Module, *, config: LokrConfig) -> LokrModule:
        """Build a LoKr module from a target module plus repo-owned config."""

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
        decompose_both: bool = False,
        factor: int = -1,
        full_matrix: bool = False,
        rank_dropout_scale: bool = False,
        weight_decompose: bool = False,
        wd_on_output: bool = True,
        bypass_mode: bool | None = None,
        rs_lora: bool = False,
        unbalanced_factorization: bool = False,
        orthogonalize: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()
        if not isinstance(org_module, SUPPORTED_MODULE_TYPES):
            raise ValueError(f"{type(org_module).__name__} is not supported in LoKr algo.")
        if init_mode not in LOKR_INIT_MODES:
            raise ValueError(f"Unknown LoKr init_mode {init_mode!r}; expected one of {LOKR_INIT_MODES}.")
        if lora_dim is None or lora_dim <= 0:
            raise ValueError(f"LoKr rank must be positive, got {lora_dim}.")
        if factor == 0:
            raise ValueError("LoKr factor must be -1 or a positive integer.")
        if weight_decompose and bypass_mode:
            raise ValueError("LoKr weight_decompose is incompatible with bypass_mode.")
        if dropout:
            raise ValueError("LoKr plain dropout is disabled; use rank_dropout or module_dropout instead.")
        if orthogonalize:
            use_scalar = True

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
        self.use_w1 = False
        self.use_w2 = False
        self.full_matrix = full_matrix

        out_dim, in_dim, *kernel_size = self.shape
        in_m, in_n = factorization(in_dim, factor)
        out_l, out_k = factorization(out_dim, factor)
        if unbalanced_factorization:
            out_l, out_k = out_k, out_l
        shape = ((out_l, out_k), (in_m, in_n), *kernel_size)
        self.tucker = bool(use_tucker and kernel_size and any(size != 1 for size in kernel_size))

        if decompose_both and lora_dim < max(shape[0][0], shape[1][0]) / 2 and not self.full_matrix:
            self.lokr_w1_a = nn.Parameter(torch.empty(shape[0][0], lora_dim))
            self.lokr_w1_b = nn.Parameter(torch.empty(lora_dim, shape[1][0]))
            self.register_parameter("lokr_w1", None)
        else:
            self.use_w1 = True
            self.lokr_w1 = nn.Parameter(torch.empty(shape[0][0], shape[1][0]))
            self.register_parameter("lokr_w1_a", None)
            self.register_parameter("lokr_w1_b", None)

        if lora_dim >= max(shape[0][1], shape[1][1]) / 2 or self.full_matrix:
            self.use_w2 = True
            self.lokr_w2 = nn.Parameter(torch.empty(shape[0][1], shape[1][1], *kernel_size))
            self.register_parameter("lokr_w2_a", None)
            self.register_parameter("lokr_w2_b", None)
            self.register_parameter("lokr_t2", None)
        elif self.tucker:
            self.lokr_t2 = nn.Parameter(torch.empty(lora_dim, lora_dim, *shape[2:]))
            self.lokr_w2_a = nn.Parameter(torch.empty(lora_dim, shape[0][1]))
            self.lokr_w2_b = nn.Parameter(torch.empty(lora_dim, shape[1][1]))
            self.register_parameter("lokr_w2", None)
        else:
            self.lokr_w2_a = nn.Parameter(torch.empty(shape[0][1], lora_dim))
            self.lokr_w2_b = nn.Parameter(torch.empty(lora_dim, shape[1][1] * math.prod(kernel_size or (1,))))
            self.register_parameter("lokr_w2", None)
            self.register_parameter("lokr_t2", None)

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
        full_unfactored = self.use_w1 and self.use_w2
        r_factor = lora_dim
        if self.rs_lora and not full_unfactored:
            r_factor = math.sqrt(lora_dim)
        if full_unfactored:
            alpha = lora_dim
        self.scale = alpha / r_factor
        self.register_buffer("alpha", torch.tensor(alpha * (lora_dim / r_factor), dtype=torch.float32))

        if use_scalar:
            self.scalar = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("scalar", torch.tensor(1.0), persistent=False)

        self._initialize_parameters(use_scalar=use_scalar)

    @property
    def dtype(self) -> torch.dtype:
        for parameter in self.parameters(recurse=False):
            if parameter is not None:
                return parameter.dtype
        return cast(Tensor, self.alpha).dtype

    @property
    def device(self) -> torch.device:
        for parameter in self.parameters(recurse=False):
            if parameter is not None:
                return parameter.device
        return cast(Tensor, self.alpha).device

    @property
    def required_export_weight_keys(self) -> tuple[str, ...]:
        required_keys = ["alpha"]
        if self.use_w1:
            required_keys.append("lokr_w1")
        else:
            required_keys.extend(("lokr_w1_a", "lokr_w1_b"))
        if self.use_w2:
            required_keys.append("lokr_w2")
        else:
            required_keys.extend(("lokr_w2_a", "lokr_w2_b"))
            if self.tucker:
                required_keys.append("lokr_t2")
        if self.wd:
            required_keys.append("dora_scale")
        return tuple(required_keys)

    def _reset_scalar_to_identity(self) -> None:
        if getattr(self, "scalar", None) is None:
            return
        identity = torch.ones_like(self.scalar)
        with torch.no_grad():
            self.scalar.copy_(identity)

    def _initialize_parameters(self, *, use_scalar: bool) -> None:
        # `zero_delta_he` currently shares LoKr's LyCORIS-compatible zero-effect
        # policy: initialize the active factors with Kaiming/orthogonal weights,
        # then keep the initial delta at zero through lokr_w2/lokr_w2_b or scalar.
        if self.use_w2:
            if self.use_orthogonal_weights:
                nn.init.orthogonal_(self.lokr_w2)
            elif self.init_mode == "random_nonzero" or use_scalar:
                nn.init.kaiming_uniform_(self.lokr_w2, a=math.sqrt(5))
            else:
                nn.init.zeros_(self.lokr_w2)
        else:
            if self.tucker:
                if self.use_orthogonal_weights:
                    nn.init.orthogonal_(self.lokr_t2)
                else:
                    nn.init.kaiming_uniform_(self.lokr_t2, a=math.sqrt(5))
            if self.use_orthogonal_weights:
                nn.init.orthogonal_(self.lokr_w2_a)
            else:
                nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
            if self.use_orthogonal_weights or self.init_mode == "random_nonzero" or use_scalar:
                nn.init.kaiming_uniform_(self.lokr_w2_b, a=math.sqrt(5))
            else:
                nn.init.zeros_(self.lokr_w2_b)

        if self.use_w1:
            if self.use_orthogonal_weights:
                nn.init.orthogonal_(self.lokr_w1)
            else:
                nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
        else:
            if self.use_orthogonal_weights:
                nn.init.orthogonal_(self.lokr_w1_a)
                nn.init.orthogonal_(self.lokr_w1_b)
            else:
                nn.init.kaiming_uniform_(self.lokr_w1_a, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.lokr_w1_b, a=math.sqrt(5))

        if self.init_mode == "random_nonzero" and use_scalar:
            with torch.no_grad():
                self.scalar.fill_(1.0)

    def _orthogonalize(self, weight_matrix: Tensor) -> Tensor:
        if not self.use_orthogonal_weights or not self.training:
            return weight_matrix
        shape = weight_matrix.shape
        if len(shape) == 0:
            return weight_matrix
        if len(shape) > 2:
            weight_matrix = weight_matrix.reshape(len(weight_matrix), -1)
        elif len(shape) < 2:
            weight_matrix = weight_matrix.reshape(1, -1)

        orig_dtype = weight_matrix.dtype
        weight_matrix_fp32 = weight_matrix.to(torch.float32)
        rows, cols = weight_matrix.shape
        if rows >= cols:
            q, r = torch.linalg.qr(weight_matrix_fp32)
            weight_matrix_fp32 = q * torch.diag(r)
        else:
            q, r = torch.linalg.qr(weight_matrix_fp32.T)
            weight_matrix_fp32 = (q * torch.diag(r)).T
        return weight_matrix_fp32.to(orig_dtype).reshape(shape).contiguous()

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
        lokr_w1: Tensor | None,
        lokr_w1_a: Tensor | None,
        lokr_w1_b: Tensor | None,
        lokr_w2: Tensor | None,
        lokr_w2_a: Tensor | None,
        lokr_w2_b: Tensor | None,
        lokr_t2: Tensor | None,
        alpha: Tensor,
        dora_scale: Tensor | None,
    ) -> LokrModule:
        if alpha is None:
            raise ValueError(f"LoKr state dict for {lora_name!r} is missing alpha.")
        full_matrix = lokr_w1 is not None and lokr_w2 is not None
        if lokr_w1_a is not None:
            lora_dim = lokr_w1_a.size(1)
        elif lokr_w2_a is not None:
            if lokr_t2 is not None:
                lora_dim = lokr_w2_a.size(0)
            else:
                lora_dim = lokr_w2_a.size(1)
        else:
            lora_dim = 1

        factor = _infer_lokr_factor(
            w1=lokr_w1,
            w1_a=lokr_w1_a,
            w1_b=lokr_w1_b,
            w2=lokr_w2,
            w2_a=lokr_w2_a,
            w2_b=lokr_w2_b,
        )
        config = LokrConfig(
            multiplier=1.0,
            lora_dim=lora_dim,
            alpha=float(alpha.detach().float().item()),
            use_tucker=lokr_t2 is not None,
            decompose_both=lokr_w1 is None and lokr_w2 is None,
            factor=factor,
            weight_decompose=dora_scale is not None,
            full_matrix=full_matrix,
        )
        module = cls.from_target_module(lora_name, orig_module, config=config)
        with torch.no_grad():
            if lokr_w1 is not None:
                module.lokr_w1.copy_(lokr_w1)
            else:
                if lokr_w1_a is None or lokr_w1_b is None:
                    raise ValueError(f"LoKr state dict for {lora_name!r} is missing lokr_w1 decomposition tensors.")
                module.lokr_w1_a.copy_(lokr_w1_a)
                module.lokr_w1_b.copy_(lokr_w1_b)
            if lokr_w2 is not None:
                module.lokr_w2.copy_(lokr_w2)
            else:
                if lokr_w2_a is None or lokr_w2_b is None:
                    raise ValueError(f"LoKr state dict for {lora_name!r} is missing lokr_w2 decomposition tensors.")
                module.lokr_w2_a.copy_(lokr_w2_a)
                module.lokr_w2_b.copy_(lokr_w2_b)
            if lokr_t2 is not None and module.lokr_t2 is not None:
                module.lokr_t2.copy_(lokr_t2)
            if dora_scale is not None and module.dora_scale is not None:
                module.dora_scale.copy_(dora_scale)
        module._reset_scalar_to_identity()
        return module

    def export_state_dict(self) -> dict[str, Tensor]:
        """Export the repo-owned LoKr checkpoint payload for this module."""

        state = {"alpha": cast(Tensor, self.alpha).detach()}
        scalar = self.scalar.detach().to(self.device)
        if self.use_w1:
            state["lokr_w1"] = self.lokr_w1.detach() * scalar.to(self.lokr_w1.device)
        else:
            state["lokr_w1_a"] = self.lokr_w1_a.detach() * scalar.to(self.lokr_w1_a.device)
            state["lokr_w1_b"] = self.lokr_w1_b.detach()

        if self.use_w2:
            state["lokr_w2"] = self.lokr_w2.detach()
        else:
            state["lokr_w2_a"] = self.lokr_w2_a.detach()
            state["lokr_w2_b"] = self.lokr_w2_b.detach()
            if self.tucker and self.lokr_t2 is not None:
                state["lokr_t2"] = self.lokr_t2.detach()
        if self.wd and self.dora_scale is not None:
            state["dora_scale"] = self.dora_scale.detach()
        return state

    @torch.no_grad()
    def load_export_state_dict(self, weights: Mapping[str, Tensor]) -> None:
        """Load the repo-owned LoKr checkpoint payload for this module."""

        if self.use_w1:
            self.lokr_w1.copy_(weights["lokr_w1"])
        else:
            self.lokr_w1_a.copy_(weights["lokr_w1_a"])
            self.lokr_w1_b.copy_(weights["lokr_w1_b"])
        if self.use_w2:
            self.lokr_w2.copy_(weights["lokr_w2"])
        else:
            self.lokr_w2_a.copy_(weights["lokr_w2_a"])
            self.lokr_w2_b.copy_(weights["lokr_w2_b"])
            if self.tucker and self.lokr_t2 is not None and weights.get("lokr_t2") is not None:
                self.lokr_t2.copy_(weights["lokr_t2"])
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

    def _cast_input_for_op(self, x: Tensor, weight: Tensor) -> Tensor:
        if x.dtype == weight.dtype:
            return x
        return x.to(weight.dtype)

    def _cast_bias_for_op(self, bias: Tensor | None, weight: Tensor) -> Tensor | None:
        if bias is None:
            return None
        return bias.to(weight.dtype, non_blocking=True)

    def _restore_result_dtype(self, result: Tensor, original_dtype: torch.dtype) -> Tensor:
        if result.dtype == original_dtype:
            return result
        return result.to(original_dtype)

    def _compute_base_result(self, x: Tensor) -> Tensor:
        weight = self.get_org_weight_for_compute(x.device).to(self.dtype)
        bias = self._cast_bias_for_op(self.get_org_bias_for_compute(x.device), weight)
        result = self.op(self._cast_input_for_op(x, weight), weight, bias, **self.kw_dict)
        return self._restore_result_dtype(result, x.dtype)

    def apply_to(self) -> None:
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward  # type: ignore[invalid-assignment]

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
        """Build the LoKr diff weight tensor for the requested target shape."""

        if self.use_w1:
            w1 = self._orthogonalize(self.lokr_w1)
        else:
            w1 = self._orthogonalize(self.lokr_w1_a) @ self._orthogonalize(self.lokr_w1_b)

        if self.use_w2:
            w2 = self._orthogonalize(self.lokr_w2)
        elif self.tucker:
            w2 = _rebuild_tucker(self.lokr_t2, self._orthogonalize(self.lokr_w2_a), self._orthogonalize(self.lokr_w2_b))
        else:
            w2 = self._orthogonalize(self.lokr_w2_a) @ self._orthogonalize(self.lokr_w2_b)

        weight = _make_kron_weight(w1, w2, self.scale)
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
        """Return the original target weight with this LoKr module merged in."""

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
        orig_norm = (self.get_weight(self.shape) * self.scalar.to(self.device)).norm()
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
        return self.get_weight(self.shape).norm()

    def bypass_forward_diff(self, x: Tensor, scale: float = 1.0) -> Tensor:
        diff_weight = self.get_weight(self.shape).to(self.dtype) * self.scalar.to(self.device) * scale
        result = self.op(self._cast_input_for_op(x, diff_weight), diff_weight, None, **self.kw_dict)
        return self._restore_result_dtype(result, x.dtype)

    def bypass_forward(self, x: Tensor, scale: float = 1.0) -> Tensor:
        return self._compute_base_result(x) + self.bypass_forward_diff(x, scale=scale)

    def forward(self, x: Tensor, *args, **kwargs):
        if self.module_dropout and self.training and torch.rand(1).item() < self.module_dropout:
            return self._compute_base_result(x)

        if self.bypass_mode:
            return self.bypass_forward(x, scale=self.multiplier)

        diff_weight = self.get_weight(self.shape).to(self.dtype) * self.scalar.to(self.device)
        weight = self.get_org_weight_for_compute(x.device).to(self.dtype)
        if self.wd:
            weight = self.apply_weight_decompose(weight + diff_weight, self.multiplier)
        else:
            weight = weight + diff_weight * self.multiplier
        bias = self._cast_bias_for_op(self.get_org_bias_for_compute(x.device), weight)
        result = self.op(self._cast_input_for_op(x, weight), weight, bias, **self.kw_dict)
        return self._restore_result_dtype(result, x.dtype)
