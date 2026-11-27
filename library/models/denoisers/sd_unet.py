# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2022 kohya-ss
# SPDX-FileCopyrightText: 2022-2023 The HuggingFace Team
# SPDX-FileCopyrightText: 2021 OpenAI
# SPDX-FileCopyrightText: 2022 Machine Vision and Learning Group, LMU Munich
# SPDX-FileCopyrightText: 2023 Stability AI
# Derived from kohya-ss/sd-scripts library/sdxl_original_unet.py, itself based on Hugging Face Diffusers;
# structure follows openai/guided-diffusion, CompVis/latent-diffusion and Stability-AI/generative-models.
# Modified. See THIRD_PARTY_NOTICES.md.

"""The latent diffusion UNet that SD 1.x, 2.x and SDXL are released as.

The structure follows the reference implementation (``openaimodel``'s
``UNetModel`` in the CompVis and Stability code, with ``legacy: False``), so
its parameters carry the original released checkpoints' names and load them
as they are. Each release ships its values as a :class:`SDUNetConfig` in its
assembly. Only SDXL's values (:mod:`library.models.sdxl.assembly`) are written
and checked so far: against the checkpoint's keys and shapes and, through the
SDXL step, bit for bit against today's model. SD 1.x and 2.x come with their
releases, and their values are checked then; shapes alone do not check them,
since how attention splits into heads leaves every weight's shape the same.

Attention runs through PyTorch's scaled dot-product attention. Gradient
checkpointing is the component's own: :meth:`SDUNet.checkpoint_points` names
the blocks that may recompute their forward inside the backward, and
:meth:`SDUNet.checkpointing` turns them on, which a route's preparation does.
A block checkpoints whenever it is on and grad mode records, in whatever
module mode the call runs, and always non-reentrant.
"""

import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

GROUPS = 32  # every normalization's group count


@dataclass(frozen=True)
class SDUNetConfig:
    """A release's values for the UNet.

    ``channel_mult`` gives each level's width as a multiple of
    ``model_channels``; ``transformer_depth`` gives each level's attention
    depth, 0 for none. ``adm_in_channels`` is the width of the vector
    conditioning, such as SDXL's pooled text and sizes, or ``None`` for a
    release without one. Attention is split into heads either by count,
    ``num_heads`` at every level as SD 1.x does, or by width, ``head_dim``
    channels per head as SDXL does; exactly one is given, as in the
    reference's ``num_heads`` and ``num_head_channels``.
    """

    in_channels: int
    out_channels: int
    model_channels: int
    channel_mult: tuple[int, ...]
    num_res_blocks: int
    transformer_depth: tuple[int, ...]
    middle_transformer_depth: int
    context_dim: int
    adm_in_channels: int | None
    linear_projection: bool
    num_heads: int | None = None
    head_dim: int | None = None

    def __post_init__(self) -> None:
        if len(self.transformer_depth) != len(self.channel_mult):
            raise ValueError(f"{len(self.channel_mult)} levels and {len(self.transformer_depth)} attention depths")
        if (self.num_heads is None) == (self.head_dim is None):
            raise ValueError("attention is split by a head count or by a head width: give exactly one of num_heads and head_dim")
        # The releases built here all attend in the middle; without it, the middle block's layout differs.
        if self.middle_transformer_depth < 1:
            raise ValueError(f"the middle block attends at depth 1 or more, not {self.middle_transformer_depth}")

    def heads(self, channels: int) -> tuple[int, int]:
        """The attention's head count and head width at a level of ``channels`` channels."""
        if self.head_dim is not None:
            count, width = channels // self.head_dim, self.head_dim
        else:
            assert self.num_heads is not None
            count, width = self.num_heads, channels // self.num_heads
        if count * width != channels:
            raise ValueError(f"{channels} channels do not split into {count} heads of {width}")
        return count, width

    @property
    def time_embed_dim(self) -> int:
        return 4 * self.model_channels


def timestep_embedding(timesteps: torch.Tensor, width: int) -> torch.Tensor:
    """Each timestep's sinusoidal embedding, cosines then sines, computed in float32."""
    half = width // 2
    exponent = -math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
    angles = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    if width % 2 == 1:
        embedding = F.pad(embedding, (0, 1, 0, 0))
    return embedding


class GroupNorm32(nn.GroupNorm):
    """Group normalization that computes in float32 when its weights are float32.

    The reference always computes ``super().forward(x.float())``, which
    assumes float32 norm weights and fails with low-precision ones; kohya's
    condition keeps low-precision weights working. Under autocast the two are
    the same, since autocast computes ``group_norm`` in float32 whatever the
    weights are. They differ only without autocast and with low-precision norm
    weights, where float32 norms are a choice of weight storage.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # GroupNorm's own parameter name
        x = input
        if self.weight.dtype != torch.float32:
            return super().forward(x)
        return super().forward(x.float()).type(x.dtype)


class Checkpointed(nn.Module):
    """A block whose forward may be recomputed inside the backward, once its route turns that on."""

    checkpointed = False

    def run(self, body, *inputs):
        if self.checkpointed and torch.is_grad_enabled():
            return checkpoint(body, *inputs, use_reentrant=False)
        return body(*inputs)


class ResnetBlock2D(Checkpointed):
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int):
        super().__init__()
        self.in_layers = nn.Sequential(GroupNorm32(GROUPS, in_channels), nn.SiLU(), nn.Conv2d(in_channels, out_channels, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, out_channels))
        self.out_layers = nn.Sequential(
            GroupNorm32(GROUPS, out_channels),
            nn.SiLU(),
            nn.Identity(),  # where the reference has dropout, keeping its parameter names
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.skip_connection = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def body(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        h = h + self.emb_layers(emb).type(h.dtype)[:, :, None, None]
        return self.skip_connection(x) + self.out_layers(h)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return self.run(self.body, x, emb)


class Downsample2D(Checkpointed):
    def __init__(self, channels: int, out_channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.run(self.op, x)


class Upsample2D(Checkpointed):
    def __init__(self, channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, out_channels, 3, padding=1)

    def body(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() * 2 > 2**31:  # nearest upsampling of a large channels-last tensor fails (pytorch#141831)
            x = x.contiguous()
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.run(self.body, x)


class CrossAttention(nn.Module):
    """Attention of a sequence over itself, or over a context when one is given."""

    def __init__(self, query_dim: int, context_dim: int | None, heads: int, head_dim: int):
        super().__init__()
        inner = heads * head_dim
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner, bias=False)
        self.to_k = nn.Linear(context_dim or query_dim, inner, bias=False)
        self.to_v = nn.Linear(context_dim or query_dim, inner, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(inner, query_dim)])

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        context = (context if context is not None else x).to(x.dtype)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (self.to_q(x), self.to_k(context), self.to_v(context)))
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        return self.to_out[0](rearrange(out, "b h n d -> b n (h d)", h=self.heads))


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        inner = 4 * dim
        self.net = nn.ModuleList([GEGLU(dim, inner), nn.Identity(), nn.Linear(inner, dim)])  # the reference's dropout slot kept for names

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            x = module(x)
        return x


class BasicTransformerBlock(Checkpointed):
    def __init__(self, dim: int, heads: int, head_dim: int, context_dim: int):
        super().__init__()
        self.attn1 = CrossAttention(dim, None, heads, head_dim)
        self.ff = FeedForward(dim)
        self.attn2 = CrossAttention(dim, context_dim, heads, head_dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def body(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        return self.ff(self.norm3(x)) + x

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.run(self.body, x, context)


class Transformer2DModel(nn.Module):
    def __init__(self, channels: int, heads: int, head_dim: int, depth: int, context_dim: int, linear_projection: bool):
        super().__init__()
        inner = heads * head_dim
        self.linear_projection = linear_projection
        self.norm = nn.GroupNorm(GROUPS, channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(channels, inner) if linear_projection else nn.Conv2d(channels, inner, 1)
        self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(inner, heads, head_dim, context_dim) for _ in range(depth)])
        self.proj_out = nn.Linear(inner, channels) if linear_projection else nn.Conv2d(inner, channels, 1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        residual = x
        h = self.norm(x)
        if not self.linear_projection:
            h = self.proj_in(h)
        inner = h.shape[1]
        h = h.permute(0, 2, 3, 1).reshape(batch, height * width, inner)
        if self.linear_projection:
            h = self.proj_in(h)
        for block in self.transformer_blocks:
            h = block(h, context)
        if self.linear_projection:
            h = self.proj_out(h)
        h = h.reshape(batch, height, width, inner).permute(0, 3, 1, 2).contiguous()
        if not self.linear_projection:
            h = self.proj_out(h)
        return h + residual


class SDUNet(nn.Module):
    """The UNet for one release's :class:`SDUNetConfig`: noisy latents, timesteps, context and vector in, the prediction out."""

    def __init__(self, config: SDUNetConfig):
        super().__init__()
        self.config = config
        mc, emb = config.model_channels, config.time_embed_dim
        self.time_embed = nn.Sequential(nn.Linear(mc, emb), nn.SiLU(), nn.Linear(emb, emb))
        if config.adm_in_channels is not None:
            self.label_emb = nn.Sequential(nn.Sequential(nn.Linear(config.adm_in_channels, emb), nn.SiLU(), nn.Linear(emb, emb)))

        def attention(channels: int, depth: int) -> Transformer2DModel:
            return Transformer2DModel(channels, *config.heads(channels), depth, config.context_dim, config.linear_projection)

        self.input_blocks = nn.ModuleList([nn.Sequential(nn.Conv2d(config.in_channels, mc, 3, padding=1))])
        skips = [mc]
        channels = mc
        last = len(config.channel_mult) - 1
        for level, (mult, depth) in enumerate(zip(config.channel_mult, config.transformer_depth, strict=True)):
            for _ in range(config.num_res_blocks):
                layers: list[nn.Module] = [ResnetBlock2D(channels, mult * mc, emb)]
                channels = mult * mc
                if depth:
                    layers.append(attention(channels, depth))
                self.input_blocks.append(nn.ModuleList(layers))
                skips.append(channels)
            if level != last:
                self.input_blocks.append(nn.Sequential(Downsample2D(channels, channels)))
                skips.append(channels)

        self.middle_block = nn.ModuleList(
            [
                ResnetBlock2D(channels, channels, emb),
                attention(channels, config.middle_transformer_depth),
                ResnetBlock2D(channels, channels, emb),
            ]
        )

        self.output_blocks = nn.ModuleList([])
        for level in reversed(range(len(config.channel_mult))):
            mult, depth = config.channel_mult[level], config.transformer_depth[level]
            for i in range(config.num_res_blocks + 1):
                layers = [ResnetBlock2D(channels + skips.pop(), mult * mc, emb)]
                channels = mult * mc
                if depth:
                    layers.append(attention(channels, depth))
                if level and i == config.num_res_blocks:
                    layers.append(Upsample2D(channels, channels))
                self.output_blocks.append(nn.ModuleList(layers))

        self.out = nn.ModuleList([GroupNorm32(GROUPS, mc), nn.SiLU(), nn.Conv2d(mc, config.out_channels, 3, padding=1)])

    def checkpoint_points(self) -> list[Checkpointed]:
        """The blocks that may recompute their forward inside the backward."""
        return [module for module in self.modules() if isinstance(module, Checkpointed)]

    def checkpointing(self, on: bool) -> None:
        """Turns gradient checkpointing at every checkpoint point on or off."""
        for module in self.checkpoint_points():
            module.checkpointed = on

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        factor = 2 ** (len(self.config.channel_mult) - 1)  # each downsampling halves the sides, and each upsampling doubles them
        if any(side % factor for side in x.shape[-2:]):
            raise ValueError(
                f"latents of {tuple(x.shape[-2:])} do not halve evenly {len(self.config.channel_mult) - 1} times; each side must be a multiple of {factor}"
            )
        timesteps = timesteps.expand(x.shape[0])
        emb = self.time_embed(timestep_embedding(timesteps, self.config.model_channels).to(x.dtype))
        if self.config.adm_in_channels is not None:
            if y is None or y.shape[0] != x.shape[0] or y.dtype != x.dtype:
                raise ValueError("the vector conditioning must have the latents' batch size and dtype")
            emb = emb + self.label_emb(y)
        elif y is not None:
            raise ValueError("this UNet takes no vector conditioning, and one was given")

        def call(block: nn.Sequential | nn.ModuleList, h: torch.Tensor) -> torch.Tensor:
            for layer in block:
                if isinstance(layer, ResnetBlock2D):
                    h = layer(h, emb)
                elif isinstance(layer, Transformer2DModel):
                    h = layer(h, context)
                else:
                    h = layer(h)
            return h

        skips = []
        h = x
        for block in self.input_blocks:
            h = call(block, h)  # ty: ignore[invalid-argument-type]  # ModuleList iterates as Module
            skips.append(h)
        h = call(self.middle_block, h)
        for block in self.output_blocks:
            h = call(block, torch.cat([h, skips.pop()], dim=1))  # ty: ignore[invalid-argument-type]  # ModuleList iterates as Module
        return call(self.out, h.type(x.dtype))


__all__ = ["SDUNet", "SDUNetConfig", "timestep_embedding"]
