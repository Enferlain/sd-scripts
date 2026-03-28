"""SD3 MMDiT model definitions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import logging
import math
from typing import Optional

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from library.performance import custom_offloading_utils


logger = logging.getLogger(__name__)


memory_efficient_attention = None
try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    memory_efficient_attention = None


@dataclass
class SD3Params:
    patch_size: int
    depth: int
    num_patches: int
    pos_embed_max_size: int
    adm_in_channels: int
    qk_norm: Optional[str]
    x_block_self_attn_layers: list[int]
    context_embedder_in_features: int
    context_embedder_out_features: int
    model_type: str


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    scaling_factor: float | None = None,
    offset: float | None = None,
):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    if scaling_factor is not None:
        grid = grid / scaling_factor
    if offset is not None:
        grid = grid - offset

    grid = grid.reshape([2, 1, grid_size, grid_size])
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid):
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_scaled_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int | tuple[int, int],
    cls_token: bool = False,
    extra_tokens: int = 0,
    sample_size: int = 64,
    base_size: int = 16,
):
    """Create scaled positional embeddings for resolutions beyond the base grid."""

    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / grid_size[0]
    grid_w = np.arange(grid_size[1], dtype=np.float32) / grid_size[1]

    scale_h = base_size * grid_size[0] / sample_size
    scale_w = base_size * grid_size[1] / sample_size

    shift_h = scale_h * (grid_size[0] - sample_size) / (2 * grid_size[0])
    shift_w = scale_w * (grid_size[1] - sample_size) / (2 * grid_size[1])

    grid_h = grid_h * scale_h - shift_h
    grid_w = grid_w * scale_w - shift_w

    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


def get_1d_sincos_pos_embed_from_grid_torch(
    embed_dim: int,
    pos,
    device=None,
    dtype: torch.dtype = torch.float32,
):
    omega = torch.arange(embed_dim // 2, device=device, dtype=dtype)
    omega *= 2.0 / embed_dim
    omega = 1.0 / 10000**omega
    out = torch.outer(pos.reshape(-1), omega)
    return torch.cat([out.sin(), out.cos()], dim=1)


def get_2d_sincos_pos_embed_torch(
    embed_dim: int,
    w: int,
    h: int,
    val_center: float = 7.5,
    val_magnitude: float = 7.5,
    device=None,
    dtype: torch.dtype = torch.float32,
):
    small = min(h, w)
    val_h = (h / small) * val_magnitude
    val_w = (w / small) * val_magnitude
    grid_h, grid_w = torch.meshgrid(
        torch.linspace(-val_h + val_center, val_h + val_center, h, device=device, dtype=dtype),
        torch.linspace(-val_w + val_center, val_w + val_center, w, device=device, dtype=dtype),
        indexing="ij",
    )
    emb_h = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid_h, device=device, dtype=dtype)
    emb_w = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid_w, device=device, dtype=dtype)
    return torch.cat([emb_w, emb_h], dim=1)


def modulate(x: torch.Tensor, shift: torch.Tensor | None, scale: torch.Tensor):
    if shift is None:
        shift = torch.zeros_like(scale)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def default(x, default_value):
    if x is None:
        return default_value
    return x


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(dtype=t.dtype)
    return embedding


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int | None = 256,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 512,
        norm_layer=None,
        flatten: bool = True,
        bias: bool = True,
        strict_img_size: bool = True,
        dynamic_img_pad: bool = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad
        if img_size is not None:
            self.img_size = img_size
            self.grid_size = img_size // patch_size
            self.num_patches = self.grid_size**2
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size, bias=bias)
        self.norm = nn.Identity() if norm_layer is None else norm_layer(embed_dim)

    def forward(self, x: torch.Tensor):
        _, _, h, w = x.shape

        if self.dynamic_img_pad:
            pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
            pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


class UnPatch(nn.Module):
    def __init__(self, hidden_size: int = 512, patch_size: int = 4, out_channels: int = 3):
        super().__init__()
        self.patch_size = patch_size
        self.c = out_channels
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size**2 * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )

    def forward(self, x: torch.Tensor, cmod: torch.Tensor, h: int | None = None, w: int | None = None):
        b, n, _ = x.shape
        p = self.patch_size
        c = self.c
        if h is None and w is None:
            w = h = int(n**0.5)
            assert h * w == n
        else:
            h = h // p if h else n // (w // p)
            w = w // p if w else n // h
            assert h * w == n

        shift, scale = self.adaLN_modulation(cmod).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)

        x = x.view(b, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(b, c, h * p, w * p)


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer=lambda: nn.GELU(),
        norm_layer=None,
        bias: bool = True,
        use_conv: bool = False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        layer = partial(nn.Conv1d, kernel_size=1) if use_conv else nn.Linear
        self.fc1 = layer(in_features, hidden_features, bias=bias)
        self.fc2 = layer(hidden_features, out_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        return self.fc2(x)


class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_size: int, freq_embed_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_embed_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.freq_embed_size = freq_embed_size

    def forward(self, t: torch.Tensor, dtype: torch.dtype | None = None, **kwargs):
        del kwargs
        t_freq = timestep_embedding(t, self.freq_embed_size).to(dtype)
        return self.mlp(t_freq)


class Embedder(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor):
        return self.mlp(x)


def rmsnorm(x: torch.Tensor, eps: float = 1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        elementwise_affine: bool = False,
        eps: float = 1e-6,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.eps = eps
        self.learnable_scale = elementwise_affine
        if self.learnable_scale:
            self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor):
        x = rmsnorm(x, eps=self.eps)
        if self.learnable_scale:
            return x * self.weight.to(device=x.device, dtype=x.dtype)
        return x


class SwiGLUFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None = None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class AttentionLinears(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        pre_only: bool = False,
        qk_norm: str | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if not pre_only:
            self.proj = nn.Linear(dim, dim)
        self.pre_only = pre_only

        if qk_norm == "rms":
            self.ln_q = RMSNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6)
            self.ln_k = RMSNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6)
        elif qk_norm == "ln":
            self.ln_q = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6)
            self.ln_k = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6)
        elif qk_norm is None:
            self.ln_q = nn.Identity()
            self.ln_k = nn.Identity()
        else:
            raise ValueError(qk_norm)

    def pre_attention(self, x: torch.Tensor):
        b, l, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.reshape(b, l, -1, self.head_dim).chunk(3, dim=2)
        q = self.ln_q(q).reshape(q.shape[0], q.shape[1], -1)
        k = self.ln_k(k).reshape(q.shape[0], q.shape[1], -1)
        return q, k, v

    def post_attention(self, x: torch.Tensor):
        assert not self.pre_only
        return self.proj(x)


MEMORY_LAYOUTS = {
    "torch": (
        lambda x, head_dim: x.reshape(x.shape[0], x.shape[1], -1, head_dim).transpose(1, 2),
        lambda x: x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1),
    ),
    "xformers": (
        lambda x, head_dim: x.reshape(x.shape[0], x.shape[1], -1, head_dim),
        lambda x: x.reshape(x.shape[0], x.shape[1], -1),
    ),
    "math": (
        lambda x, head_dim: x.reshape(x.shape[0], x.shape[1], -1, head_dim).transpose(1, 2),
        lambda x: x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1),
    ),
}


def vanilla_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask, scale: float | None = None):
    if scale is None:
        scale = math.sqrt(q.size(-1))
    scores = torch.bmm(q, k.transpose(-1, -2)) / scale
    if mask is not None:
        mask = einops.rearrange(mask, "b ... -> b (...)")
        max_neg_value = -torch.finfo(scores.dtype).max
        mask = einops.repeat(mask, "b j -> (b h) j", h=q.size(-3))
        scores = scores.masked_fill(~mask, max_neg_value)
    p_attn = F.softmax(scores, dim=-1)
    return torch.bmm(p_attn, v)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    head_dim: int,
    mask=None,
    scale: float | None = None,
    mode: str = "xformers",
):
    pre_attn_layout = MEMORY_LAYOUTS[mode][0]
    post_attn_layout = MEMORY_LAYOUTS[mode][1]
    q = pre_attn_layout(q, head_dim)
    k = pre_attn_layout(k, head_dim)
    v = pre_attn_layout(v, head_dim)

    if mode == "torch":
        assert scale is None
        scores = F.scaled_dot_product_attention(q, k.to(q), v.to(q), mask)
    elif mode == "xformers":
        if memory_efficient_attention is None:
            raise RuntimeError("xformers attention requested but xformers is not available")
        scores = memory_efficient_attention(q, k.to(q), v.to(q), mask, scale=scale)
    else:
        scores = vanilla_attention(q, k.to(q), v.to(q), mask, scale=scale)

    return post_attn_layout(scores)


class SingleDiTBlock(nn.Module):
    """A DiT block with gated adaptive layer norm conditioning."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: str = "xformers",
        qkv_bias: bool = False,
        pre_only: bool = False,
        rmsnorm: bool = False,
        scale_mod_only: bool = False,
        swiglu: bool = False,
        qk_norm: str | None = None,
        x_block_self_attn: bool = False,
        **block_kwargs,
    ):
        del block_kwargs
        super().__init__()
        assert attn_mode in MEMORY_LAYOUTS
        self.attn_mode = attn_mode
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6) if not rmsnorm else RMSNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.attn = AttentionLinears(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, pre_only=pre_only, qk_norm=qk_norm)

        self.x_block_self_attn = x_block_self_attn
        if self.x_block_self_attn:
            assert not pre_only
            assert not scale_mod_only
            self.attn2 = AttentionLinears(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, pre_only=False, qk_norm=qk_norm)

        if not pre_only:
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6) if not rmsnorm else RMSNorm(
                hidden_size, elementwise_affine=False, eps=1e-6
            )
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if not pre_only:
            if not swiglu:
                self.mlp = MLP(
                    in_features=hidden_size,
                    hidden_features=mlp_hidden_dim,
                    act_layer=lambda: nn.GELU(approximate="tanh"),
                )
            else:
                self.mlp = SwiGLUFeedForward(dim=hidden_size, hidden_dim=mlp_hidden_dim, multiple_of=256)
        self.scale_mod_only = scale_mod_only
        if self.x_block_self_attn:
            n_mods = 9
        elif not scale_mod_only:
            n_mods = 6 if not pre_only else 2
        else:
            n_mods = 4 if not pre_only else 1
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, n_mods * hidden_size))
        self.pre_only = pre_only

    def pre_attention(self, x: torch.Tensor, c: torch.Tensor):
        if not self.pre_only:
            if not self.scale_mod_only:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
            else:
                shift_msa = None
                shift_mlp = None
                scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=-1)
            qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
            return qkv, (x, gate_msa, shift_mlp, scale_mlp, gate_mlp)

        if not self.scale_mod_only:
            shift_msa, scale_msa = self.adaLN_modulation(c).chunk(2, dim=-1)
        else:
            shift_msa = None
            scale_msa = self.adaLN_modulation(c)
        qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
        return qkv, None

    def pre_attention_x(self, x: torch.Tensor, c: torch.Tensor):
        assert self.x_block_self_attn
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_msa2, scale_msa2, gate_msa2 = (
            self.adaLN_modulation(c).chunk(9, dim=1)
        )
        x_norm = self.norm1(x)
        qkv = self.attn.pre_attention(modulate(x_norm, shift_msa, scale_msa))
        qkv2 = self.attn2.pre_attention(modulate(x_norm, shift_msa2, scale_msa2))
        return qkv, qkv2, (x, gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_msa2)

    def post_attention(
        self,
        attn: torch.Tensor,
        x: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor | None,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
    ):
        assert not self.pre_only
        x = x + gate_msa.unsqueeze(1) * self.attn.post_attention(attn)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

    def post_attention_x(
        self,
        attn: torch.Tensor,
        attn2: torch.Tensor,
        x: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor | None,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        gate_msa2: torch.Tensor,
        attn1_dropout: float = 0.0,
    ):
        assert not self.pre_only
        if attn1_dropout > 0.0:
            attn1_dropout_mask = torch.bernoulli(torch.full((attn.size(0), 1, 1), 1 - attn1_dropout, device=attn.device))
            attn_ = gate_msa.unsqueeze(1) * self.attn.post_attention(attn) * attn1_dropout_mask
        else:
            attn_ = gate_msa.unsqueeze(1) * self.attn.post_attention(attn)
        x = x + attn_
        x = x + gate_msa2.unsqueeze(1) * self.attn2.post_attention(attn2)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class MMDiTBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        pre_only = kwargs.pop("pre_only")
        x_block_self_attn = kwargs.pop("x_block_self_attn")

        self.context_block = SingleDiTBlock(*args, pre_only=pre_only, **kwargs)
        self.x_block = SingleDiTBlock(*args, pre_only=False, x_block_self_attn=x_block_self_attn, **kwargs)

        self.head_dim = self.x_block.attn.head_dim
        self.mode = self.x_block.attn_mode
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def _forward(self, context: torch.Tensor | None, x: torch.Tensor, c: torch.Tensor):
        ctx_qkv, ctx_intermediate = self.context_block.pre_attention(context, c)

        if self.x_block.x_block_self_attn:
            x_qkv, x_qkv2, x_intermediates = self.x_block.pre_attention_x(x, c)
        else:
            x_qkv, x_intermediates = self.x_block.pre_attention(x, c)

        ctx_len = ctx_qkv[0].size(1)

        q = torch.concat((ctx_qkv[0], x_qkv[0]), dim=1)
        k = torch.concat((ctx_qkv[1], x_qkv[1]), dim=1)
        v = torch.concat((ctx_qkv[2], x_qkv[2]), dim=1)

        attn = attention(q, k, v, head_dim=self.head_dim, mode=self.mode)
        ctx_attn_out = attn[:, :ctx_len]
        x_attn_out = attn[:, ctx_len:]

        if self.x_block.x_block_self_attn:
            x_q2, x_k2, x_v2 = x_qkv2
            attn2 = attention(x_q2, x_k2, x_v2, self.x_block.attn2.num_heads, mode=self.mode)
            x = self.x_block.post_attention_x(x_attn_out, attn2, *x_intermediates)
        else:
            x = self.x_block.post_attention(x_attn_out, *x_intermediates)

        if not self.context_block.pre_only:
            context = self.context_block.post_attention(ctx_attn_out, *ctx_intermediate)
        else:
            context = None

        return context, x

    def forward(self, *args, **kwargs):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, *args, use_reentrant=False, **kwargs)
        return self._forward(*args, **kwargs)


class MMDiT(nn.Module):
    """Diffusion model with an MMDiT backbone."""

    POS_EMBED_MAX_RATIO = 1.5

    def __init__(
        self,
        input_size: int | None = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        depth: int = 28,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = False,
        adm_in_channels: int | None = None,
        context_embedder_in_features: int | None = None,
        context_embedder_out_features: int | None = None,
        use_checkpoint: bool = False,
        register_length: int = 0,
        attn_mode: str = "torch",
        rmsnorm: bool = False,
        scale_mod_only: bool = False,
        swiglu: bool = False,
        out_channels: int | None = None,
        pos_embed_scaling_factor: float | None = None,
        pos_embed_offset: float | None = None,
        pos_embed_max_size: int | None = None,
        num_patches=None,
        qk_norm: str | None = None,
        x_block_self_attn_layers: list[int] | None = None,
        qkv_bias: bool = True,
        pos_emb_random_crop_rate: float = 0.0,
        use_scaled_pos_embed: bool = False,
        pos_embed_latent_sizes: list[int] | None = None,
        model_type: str = "sd3m",
    ):
        super().__init__()
        self._model_type = model_type
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        default_out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = default(out_channels, default_out_channels)
        self.patch_size = patch_size
        self.pos_embed_scaling_factor = pos_embed_scaling_factor
        self.pos_embed_offset = pos_embed_offset
        self.pos_embed_max_size = pos_embed_max_size
        self.x_block_self_attn_layers = x_block_self_attn_layers or []
        self.pos_emb_random_crop_rate = pos_emb_random_crop_rate
        self.gradient_checkpointing = use_checkpoint

        self.hidden_size = 64 * depth
        num_heads = depth
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(
            input_size,
            patch_size,
            in_channels,
            self.hidden_size,
            bias=True,
            strict_img_size=self.pos_embed_max_size is None,
        )
        self.t_embedder = TimestepEmbedding(self.hidden_size)

        self.y_embedder = None
        if adm_in_channels is not None:
            self.y_embedder = Embedder(adm_in_channels, self.hidden_size)

        if context_embedder_in_features is not None:
            self.context_embedder = nn.Linear(context_embedder_in_features, context_embedder_out_features)
        else:
            self.context_embedder = nn.Identity()

        self.register_length = register_length
        if self.register_length > 0:
            self.register = nn.Parameter(torch.randn(1, register_length, self.hidden_size))

        if num_patches is not None:
            self.register_buffer("pos_embed", torch.empty(1, num_patches, self.hidden_size))
        else:
            self.pos_embed = None

        self.use_checkpoint = use_checkpoint
        self.joint_blocks = nn.ModuleList(
            [
                MMDiTBlock(
                    self.hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_mode=attn_mode,
                    qkv_bias=qkv_bias,
                    pre_only=i == depth - 1,
                    rmsnorm=rmsnorm,
                    scale_mod_only=scale_mod_only,
                    swiglu=swiglu,
                    qk_norm=qk_norm,
                    x_block_self_attn=(i in self.x_block_self_attn_layers),
                )
                for i in range(depth)
            ]
        )
        for block in self.joint_blocks:
            block.gradient_checkpointing = use_checkpoint

        self.final_layer = UnPatch(self.hidden_size, patch_size, self.out_channels)

        self.blocks_to_swap = None
        self.offloader = None
        self.num_blocks = len(self.joint_blocks)

        self.enable_scaled_pos_embed(use_scaled_pos_embed, pos_embed_latent_sizes)

    def enable_scaled_pos_embed(self, use_scaled_pos_embed: bool, latent_sizes: list[int] | None):
        self.use_scaled_pos_embed = use_scaled_pos_embed

        if self.use_scaled_pos_embed:
            if self.pos_embed is not None:
                self.pos_embed = self.pos_embed.cpu()

            assert latent_sizes, "latent_sizes must be provided when scaled positional embeddings are enabled"
            latent_sizes = sorted(set(latent_sizes))
            patched_sizes = [latent_size // self.patch_size for latent_size in latent_sizes]

            max_areas: list[int] = []
            for i in range(1, len(patched_sizes)):
                prev_area = patched_sizes[i - 1] ** 2
                area = patched_sizes[i] ** 2
                max_areas.append((prev_area + area) // 2)
            max_areas.append(int((patched_sizes[-1] * MMDiT.POS_EMBED_MAX_RATIO) ** 2))

            self.resolution_area_to_latent_size = [(area, latent_size) for area, latent_size in zip(max_areas, patched_sizes)]
            self.resolution_pos_embeds: dict[int, torch.Tensor] = {}
            for patched_size in patched_sizes:
                grid_size = int(patched_size * MMDiT.POS_EMBED_MAX_RATIO)
                pos_embed = get_scaled_2d_sincos_pos_embed(self.hidden_size, grid_size, sample_size=patched_size)
                self.resolution_pos_embeds[patched_size] = torch.from_numpy(pos_embed).float().unsqueeze(0)
        else:
            self.resolution_area_to_latent_size = None
            self.resolution_pos_embeds = None

    @property
    def model_type(self):
        return self._model_type

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
        for block in self.joint_blocks:
            block.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        for block in self.joint_blocks:
            block.disable_gradient_checkpointing()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        if self.pos_embed is not None:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                int(self.pos_embed.shape[-2] ** 0.5),
                scaling_factor=self.pos_embed_scaling_factor,
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        if getattr(self, "y_embedder", None) is not None:
            nn.init.normal_(self.y_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.y_embedder.mlp[2].weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.joint_blocks:
            nn.init.constant_(block.x_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.x_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.context_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.context_block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def set_pos_emb_random_crop_rate(self, rate: float):
        self.pos_emb_random_crop_rate = rate

    def cropped_pos_embed(self, h: int, w: int, device=None, random_crop: bool = False):
        p = self.x_embedder.patch_size
        h = (h + 1) // p
        w = (w + 1) // p
        if self.pos_embed is None:
            return get_2d_sincos_pos_embed_torch(self.hidden_size, w, h, device=device)
        assert self.pos_embed_max_size is not None
        assert h <= self.pos_embed_max_size, (h, self.pos_embed_max_size)
        assert w <= self.pos_embed_max_size, (w, self.pos_embed_max_size)

        if not random_crop:
            top = (self.pos_embed_max_size - h) // 2
            left = (self.pos_embed_max_size - w) // 2
        else:
            top = torch.randint(0, self.pos_embed_max_size - h + 1, (1,)).item()
            left = torch.randint(0, self.pos_embed_max_size - w + 1, (1,)).item()

        spatial_pos_embed = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, self.pos_embed.shape[-1])
        spatial_pos_embed = spatial_pos_embed[:, top : top + h, left : left + w, :]
        return spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])

    def cropped_scaled_pos_embed(self, h: int, w: int, device=None, dtype=None, random_crop: bool = False):
        p = self.x_embedder.patch_size
        h = (h + 1) // p
        w = (w + 1) // p

        area = h * w
        patched_size = None
        for area_, patched_size_ in self.resolution_area_to_latent_size:
            if area <= area_:
                patched_size = patched_size_
                break
        if patched_size is None:
            patched_size = self.resolution_area_to_latent_size[-1][1]

        pos_embed = self.resolution_pos_embeds[patched_size]
        pos_embed_size = round(math.sqrt(pos_embed.shape[1]))
        if h > pos_embed_size or w > pos_embed_size:
            logger.warning(
                "Add new pos_embed for size %sx%s as it exceeds the scaled pos_embed size %s. Image is too tall or wide.",
                h,
                w,
                pos_embed_size,
            )
            patched_size = max(h, w)
            grid_size = int(patched_size * MMDiT.POS_EMBED_MAX_RATIO)
            pos_embed_size = grid_size
            pos_embed = get_scaled_2d_sincos_pos_embed(self.hidden_size, grid_size, sample_size=patched_size)
            pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0)
            self.resolution_pos_embeds[patched_size] = pos_embed
            logger.info("Added pos_embed for size %sx%s", patched_size, patched_size)

            area = pos_embed_size**2
            self.resolution_area_to_latent_size.append((area, patched_size))
            self.resolution_area_to_latent_size = sorted(self.resolution_area_to_latent_size)

        if not random_crop:
            top = (pos_embed_size - h) // 2
            left = (pos_embed_size - w) // 2
        else:
            top = torch.randint(0, pos_embed_size - h + 1, (1,)).item()
            left = torch.randint(0, pos_embed_size - w + 1, (1,)).item()

        if pos_embed.device != device:
            pos_embed = pos_embed.to(device)
            self.resolution_pos_embeds[patched_size] = pos_embed
        if pos_embed.dtype != dtype:
            pos_embed = pos_embed.to(dtype)
            self.resolution_pos_embeds[patched_size] = pos_embed

        spatial_pos_embed = pos_embed.reshape(1, pos_embed_size, pos_embed_size, pos_embed.shape[-1])
        spatial_pos_embed = spatial_pos_embed[:, top : top + h, left : left + w, :]
        return spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])

    def enable_block_swap(self, num_blocks: int, device: torch.device):
        self.blocks_to_swap = num_blocks

        assert self.blocks_to_swap <= self.num_blocks - 2, (
            f"Cannot swap more than {self.num_blocks - 2} blocks. Requested: {self.blocks_to_swap} blocks."
        )

        self.offloader = custom_offloading_utils.ModelOffloader(self.joint_blocks, self.blocks_to_swap, device)
        print(f"SD3: Block swap enabled. Swapping {num_blocks} blocks, total blocks: {self.num_blocks}, device: {device}.")

    def move_to_device_except_swap_blocks(self, device: torch.device):
        if self.blocks_to_swap:
            save_blocks = self.joint_blocks
            self.joint_blocks = nn.ModuleList()

        self.to(device)

        if self.blocks_to_swap:
            self.joint_blocks = save_blocks

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader.prepare_block_devices_before_forward(self.joint_blocks)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ):
        pos_emb_random_crop = False if self.pos_emb_random_crop_rate == 0.0 else torch.rand(1).item() < self.pos_emb_random_crop_rate

        _, _, h, w = x.shape

        if not self.use_scaled_pos_embed:
            pos_embed = self.cropped_pos_embed(h, w, device=x.device, random_crop=pos_emb_random_crop).to(dtype=x.dtype)
        else:
            pos_embed = self.cropped_scaled_pos_embed(h, w, device=x.device, dtype=x.dtype, random_crop=pos_emb_random_crop)
        x = self.x_embedder(x) + pos_embed
        del pos_embed

        c = self.t_embedder(t, dtype=x.dtype)
        if y is not None and self.y_embedder is not None:
            y = self.y_embedder(y)
            c = c + y

        if context is not None:
            context = self.context_embedder(context)

        if self.register_length > 0:
            context = torch.cat(
                (einops.repeat(self.register, "1 ... -> b ...", b=x.shape[0]), default(context, torch.Tensor([]).type_as(x))),
                1,
            )

        if not self.blocks_to_swap:
            for block in self.joint_blocks:
                context, x = block(context, x, c)
        else:
            for block_idx, block in enumerate(self.joint_blocks):
                self.offloader.wait_for_block(block_idx)
                context, x = block(context, x, c)
                self.offloader.submit_move_blocks(self.joint_blocks, block_idx)

        x = self.final_layer(x, c, h, w)
        return x[:, :, :h, :w]


def create_sd3_mmdit(params: SD3Params, attn_mode: str = "torch") -> MMDiT:
    return MMDiT(
        input_size=None,
        pos_embed_max_size=params.pos_embed_max_size,
        patch_size=params.patch_size,
        in_channels=16,
        adm_in_channels=params.adm_in_channels,
        context_embedder_in_features=params.context_embedder_in_features,
        context_embedder_out_features=params.context_embedder_out_features,
        depth=params.depth,
        mlp_ratio=4,
        qk_norm=params.qk_norm,
        x_block_self_attn_layers=params.x_block_self_attn_layers,
        num_patches=params.num_patches,
        attn_mode=attn_mode,
        model_type=params.model_type,
    )


__all__ = [
    "AttentionLinears",
    "Embedder",
    "MMDiT",
    "MMDiTBlock",
    "MLP",
    "PatchEmbed",
    "RMSNorm",
    "SD3Params",
    "SingleDiTBlock",
    "SwiGLUFeedForward",
    "TimestepEmbedding",
    "UnPatch",
    "attention",
    "create_sd3_mmdit",
    "default",
    "get_1d_sincos_pos_embed_from_grid",
    "get_1d_sincos_pos_embed_from_grid_torch",
    "get_2d_sincos_pos_embed",
    "get_2d_sincos_pos_embed_from_grid",
    "get_2d_sincos_pos_embed_torch",
    "get_scaled_2d_sincos_pos_embed",
    "modulate",
    "rmsnorm",
    "timestep_embedding",
]
