"""The latent diffusion UNet: SDXL's values against the released checkpoint, and the component's own gradient checkpointing."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from library.models.denoisers.sd_unet import (
    BasicTransformerBlock,
    CrossAttention,
    Downsample2D,
    ResnetBlock2D,
    SDUNet,
    SDUNetConfig,
    Transformer2DModel,
    Upsample2D,
)
from library.models.sdxl.assembly import UNET, UNET_PREFIX

CHECKPOINT = Path("/mnt/d/stable-diffusion-webui-reForge/models/Stable-diffusion/21862-seele_pop3_v-pred.safetensors")

SMALL = SDUNetConfig(
    in_channels=4,
    out_channels=4,
    model_channels=32,
    channel_mult=(1, 2),
    num_res_blocks=1,
    transformer_depth=(0, 1),
    middle_transformer_depth=1,
    context_dim=16,
    adm_in_channels=8,
    head_dim=16,
    linear_projection=True,
)


def small() -> tuple[SDUNet, tuple[torch.Tensor, ...]]:
    torch.manual_seed(0)
    unet = SDUNet(SMALL)
    inputs = (torch.randn(2, 4, 8, 8), torch.tensor([10, 500]), torch.randn(2, 3, 16), torch.randn(2, 8))
    return unet, inputs


def body_calls(unet: SDUNet) -> list[int]:
    """Counts how often the first residual block's body runs, forward and recomputation alike."""
    calls = [0]
    block = next(m for m in unet.modules() if isinstance(m, ResnetBlock2D))
    block.in_layers.register_forward_hook(lambda *_: calls.__setitem__(0, calls[0] + 1))
    return calls


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="needs the baseline's base model")
def test_sdxl_s_values_give_exactly_the_released_checkpoint_s_unet_parameters() -> None:
    from safetensors import safe_open

    with torch.device("meta"):
        unet = SDUNet(UNET)
    ours = {name: tuple(tensor.shape) for name, tensor in unet.state_dict().items()}
    with safe_open(str(CHECKPOINT), "pt", device="cpu") as f:
        released = {key.removeprefix(UNET_PREFIX): tuple(f.get_slice(key).get_shape()) for key in f.keys() if key.startswith(UNET_PREFIX)}  # noqa: SIM118 (a safetensors file, not a dict)
    assert ours == released


def test_checkpointing_recomputes_in_the_backward_and_changes_nothing() -> None:
    unet, inputs = small()
    plain = unet(*inputs)
    plain.square().sum().backward()
    expected = [p.grad.clone() for p in unet.parameters() if p.grad is not None]
    unet.zero_grad(set_to_none=True)

    unet.checkpointing(True)
    calls = body_calls(unet)
    again = unet(*inputs)
    again.square().sum().backward()
    assert calls[0] == 2, "the forward, then its recomputation inside the backward"
    assert torch.equal(again, plain)
    grads = [p.grad for p in unet.parameters() if p.grad is not None]
    assert len(grads) == len(expected) and all(torch.equal(a, b) for a, b in zip(grads, expected, strict=True))


def test_checkpointing_follows_grad_mode_not_module_mode() -> None:
    unet, inputs = small()
    unet.checkpointing(True)
    unet.eval()  # a frozen network on the gradient path keeps evaluation mode (D20) and still checkpoints
    calls = body_calls(unet)
    unet(*inputs).sum().backward()
    assert calls[0] == 2
    with torch.no_grad():
        unet(*inputs)
    assert calls[0] == 3, "nothing to recompute without grad mode"


def test_every_residual_attention_and_resampling_block_is_a_checkpoint_point() -> None:
    unet, _ = small()
    points = list(unet.checkpoint_points())
    for kind in (ResnetBlock2D, BasicTransformerBlock, Downsample2D, Upsample2D):
        every = [m for m in unet.modules() if isinstance(m, kind)]
        assert every and [m for m in points if isinstance(m, kind)] == every, f"each {kind.__name__}, once"
    assert len(points) == len(set(map(id, points)))


def test_a_release_without_vector_conditioning_has_no_label_embedding_and_needs_no_vector() -> None:
    config = replace(SMALL, adm_in_channels=None)
    torch.manual_seed(0)
    unet = SDUNet(config)
    assert not hasattr(unet, "label_emb")
    assert unet(torch.randn(1, 4, 8, 8), torch.tensor([3]), torch.randn(1, 3, 16)).shape == (1, 4, 8, 8)


def test_a_vector_of_another_dtype_or_batch_is_refused() -> None:
    unet, (x, t, context, y) = small()
    with pytest.raises(ValueError, match="batch size and dtype"):
        unet(x, t, context, y.double())
    with pytest.raises(ValueError, match="batch size and dtype"):
        unet(x, t, context, y[:1])


def test_a_vector_given_to_a_release_without_vector_conditioning_is_refused() -> None:
    torch.manual_seed(0)
    unet = SDUNet(replace(SMALL, adm_in_channels=None))
    x, t, context, y = small()[1]
    with pytest.raises(ValueError, match="takes no vector conditioning"):
        unet(x, t, context, y)


def test_the_middle_block_attends() -> None:
    with pytest.raises(ValueError, match="the middle block attends at depth 1 or more, not 0"):
        replace(SMALL, middle_transformer_depth=0)


def test_levels_and_attention_depths_must_pair_up() -> None:
    with pytest.raises(ValueError, match="2 levels and 3 attention depths"):
        replace(SMALL, transformer_depth=(0, 1, 1))


def test_heads_by_count_keep_their_number_and_widen_with_the_level_as_sd_1_x_s_do() -> None:
    config = replace(SMALL, head_dim=None, num_heads=2)
    assert config.heads(32) == (2, 16) and config.heads(64) == (2, 32)
    torch.manual_seed(0)
    unet = SDUNet(config)
    attentions = [m for m in unet.modules() if isinstance(m, CrossAttention)]
    assert {m.heads for m in attentions} == {2}
    x, t, context, y = small()[1]
    assert unet(x, t, context, y).shape == x.shape


def test_heads_by_width_grow_in_number_with_the_level_as_sdxl_s_do() -> None:
    assert SMALL.heads(32) == (2, 16) and SMALL.heads(64) == (4, 16)


def test_exactly_one_of_head_count_and_head_width_is_given() -> None:
    with pytest.raises(ValueError, match="exactly one of num_heads and head_dim"):
        replace(SMALL, num_heads=2)
    with pytest.raises(ValueError, match="exactly one of num_heads and head_dim"):
        replace(SMALL, head_dim=None)


def test_a_level_that_does_not_split_into_its_heads_is_refused() -> None:
    with pytest.raises(ValueError, match="do not split"):
        replace(SMALL, head_dim=None, num_heads=3).heads(32)


@pytest.mark.parametrize("linear", [True, False])
def test_both_projections_map_inner_width_back_to_the_channels_when_they_differ(linear: bool) -> None:
    block = Transformer2DModel(64, 2, 24, 1, 16, linear_projection=linear)  # 48 inner channels for 64
    x = torch.randn(1, 64, 4, 4)
    assert block(x, torch.randn(1, 3, 16)).shape == x.shape


def test_latents_whose_sides_do_not_halve_evenly_at_every_level_are_refused() -> None:
    unet, (_, t, context, y) = small()  # two levels: one halving, so sides must be even
    with pytest.raises(ValueError, match="each side must be a multiple of 2"):
        unet(torch.randn(2, 4, 8, 7), t, context, y)
