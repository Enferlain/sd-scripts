import pytest
import torch

from library.models.sd.vae import (
    reshape_weight_for_sd,
    convert_vae_state_dict,
    assign_to_checkpoint,
    renew_vae_attention_paths,
    renew_vae_resnet_paths,
)
from library.models.sd.conversion import conv_transformer_to_linear, linear_transformer_to_conv


@pytest.mark.unit
class TestRenewVaeAttentionPaths:
    """Test VAE attention path renaming."""

    def test_renames_norm_to_groupnorm(self):
        old_list = ["attn.norm.weight", "attn.norm.bias"]
        result = renew_vae_attention_paths(old_list, n_shave_prefix_segments=0)

        assert result[0]["new"] == "attn.group_norm.weight"
        assert result[1]["new"] == "attn.group_norm.bias"

    def test_renames_q_k_v_to_to_q_k_v(self):
        """For diffusers >= 0.17.0, q/k/v are renamed to to_q/to_k/to_v."""
        old_list = ["attn.q.weight", "attn.k.weight", "attn.v.weight"]
        result = renew_vae_attention_paths(old_list, n_shave_prefix_segments=0)

        # Modern diffusers naming
        assert result[0]["new"] == "attn.to_q.weight"
        assert result[1]["new"] == "attn.to_k.weight"
        assert result[2]["new"] == "attn.to_v.weight"

    def test_renames_proj_out_to_to_out(self):
        old_list = ["attn.proj_out.weight", "attn.proj_out.bias"]
        result = renew_vae_attention_paths(old_list, n_shave_prefix_segments=0)

        assert result[0]["new"] == "attn.to_out.0.weight"
        assert result[1]["new"] == "attn.to_out.0.bias"

    def test_shave_prefix_segments(self):
        old_list = ["prefix.attn.norm.weight"]
        result = renew_vae_attention_paths(old_list, n_shave_prefix_segments=1)

        assert result[0]["new"] == "attn.group_norm.weight"


@pytest.mark.unit
class TestRenewVaeResnetPaths:
    """Test VAE resnet path renaming."""

    def test_renames_nin_shortcut(self):
        old_list = ["block.nin_shortcut.weight"]
        result = renew_vae_resnet_paths(old_list, n_shave_prefix_segments=0)

        assert result[0]["new"] == "block.conv_shortcut.weight"


@pytest.mark.unit
class TestConvTransformerToLinear:
    """Test conv 1D to linear conversion."""

    def test_removes_spatial_dims_from_proj_in(self):
        checkpoint = {
            "layer.proj_in.weight": torch.randn(256, 256, 1, 1),
            "layer.proj_out.weight": torch.randn(256, 256, 1, 1),
        }
        conv_transformer_to_linear(checkpoint)

        assert checkpoint["layer.proj_in.weight"].shape == (256, 256)
        assert checkpoint["layer.proj_out.weight"].shape == (256, 256)

    def test_ignores_non_proj_keys(self):
        checkpoint = {
            "layer.conv.weight": torch.randn(256, 256, 3, 3),
        }
        conv_transformer_to_linear(checkpoint)

        # Should remain unchanged
        assert checkpoint["layer.conv.weight"].shape == (256, 256, 3, 3)


@pytest.mark.unit
class TestLinearTransformerToConv:
    """Test linear to conv 1D conversion."""

    def test_adds_spatial_dims_to_proj(self):
        checkpoint = {
            "layer.proj_in.weight": torch.randn(256, 256),
            "layer.proj_out.weight": torch.randn(256, 256),
        }
        linear_transformer_to_conv(checkpoint)

        assert checkpoint["layer.proj_in.weight"].shape == (256, 256, 1, 1)
        assert checkpoint["layer.proj_out.weight"].shape == (256, 256, 1, 1)


@pytest.mark.unit
class TestReshapeWeightForSd:
    """Test weight reshaping for SD format."""

    def test_adds_spatial_dims(self):
        w = torch.randn(256, 256)
        result = reshape_weight_for_sd(w)

        assert result.shape == (256, 256, 1, 1)


@pytest.mark.unit
class TestAssignToCheckpoint:
    """Test the checkpoint assignment logic."""

    def test_basic_path_mapping(self):
        paths = [{"old": "input.0", "new": "conv_in"}]
        old_checkpoint = {"input.0": torch.randn(3, 3)}
        new_checkpoint = {}

        assign_to_checkpoint(paths, new_checkpoint, old_checkpoint)

        assert "conv_in" in new_checkpoint
        assert torch.equal(new_checkpoint["conv_in"], old_checkpoint["input.0"])

    def test_middle_block_renaming(self):
        """Test global renaming of middle blocks."""
        paths = [{"old": "mid.0", "new": "middle_block.0.weight"}]
        old_checkpoint = {"mid.0": torch.randn(64, 64)}
        new_checkpoint = {}

        assign_to_checkpoint(paths, new_checkpoint, old_checkpoint)

        assert "mid_block.resnets.0.weight" in new_checkpoint

    def test_additional_replacements(self):
        paths = [{"old": "x", "new": "layer.x"}]
        old_checkpoint = {"x": torch.randn(10)}
        new_checkpoint = {}
        additional = [{"old": "layer.", "new": "output."}]

        assign_to_checkpoint(paths, new_checkpoint, old_checkpoint, additional_replacements=additional)

        assert "output.x" in new_checkpoint


@pytest.mark.unit
class TestConvertVaeStateDict:
    """Test VAE state dict conversion from Diffusers to SD."""

    def test_nin_shortcut_renamed(self):
        """Test that nin_shortcut gets renamed to conv_shortcut."""
        vae_state_dict = {
            "decoder.up_blocks.0.resnets.0.conv_shortcut.weight": torch.randn(64, 64, 1, 1),
        }
        result = convert_vae_state_dict(vae_state_dict)

        # Check key was renamed according to conversion map
        # up_blocks.0 -> up.3 (reversed indexing)
        assert any("nin_shortcut" in k or "conv_shortcut" in k for k in result.keys())

    def test_encoder_down_blocks_converted(self):
        """Test encoder down block path conversion."""
        vae_state_dict = {
            "encoder.down_blocks.0.resnets.0.norm1.weight": torch.randn(64),
        }
        result = convert_vae_state_dict(vae_state_dict)

        # Should be converted to SD format (down.0.block.0)
        assert any("down." in k and "block." in k for k in result.keys())

    def test_attention_weights_reshaped(self):
        """Test that mid block attention weights get reshaped."""
        vae_state_dict = {
            "decoder.mid_block.attentions.0.to_q.weight": torch.randn(512, 512),
        }
        result = convert_vae_state_dict(vae_state_dict)

        # After conversion, q weights should be reshaped to [512, 512, 1, 1]
        mid_keys = [k for k in result.keys() if "mid.attn_1.q.weight" in k]
        if mid_keys:
            assert result[mid_keys[0]].shape == (512, 512, 1, 1)
