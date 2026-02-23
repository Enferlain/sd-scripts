"""
Unit tests for library/losses/loss_weighting.py

Tests for SNR-based loss weighting functions using lightweight fake schedulers.
These tests verify correct formulas, clamping behavior, and masking semantics.
"""

import torch
from types import SimpleNamespace

from library.losses.loss_weighting import (
    apply_snr_weight,
    scale_v_prediction_loss_like_noise_prediction,
    get_snr_scale,
    add_v_prediction_like_loss,
    apply_debiased_estimation,
    apply_masked_loss,
)


# --- Fixtures ---


def make_fake_scheduler(all_snr_vals: list) -> SimpleNamespace:
    """Create a fake scheduler with all_snr as a simple tensor.

    This avoids instantiating real DDPMScheduler - the functions only need
    noise_scheduler.all_snr[t] to work.
    """
    sch = SimpleNamespace()
    sch.all_snr = torch.tensor(all_snr_vals, dtype=torch.float32)
    return sch


# --- Tests for apply_snr_weight ---


class TestApplySnrWeight:
    """Tests for apply_snr_weight function."""

    def test_standard_no_clamp(self):
        """When gamma is high enough, no clamping occurs - weight = 1."""
        loss = torch.tensor([1.0, 2.0, 3.0])
        timesteps = torch.tensor([0, 1, 2], dtype=torch.int64)
        sch = make_fake_scheduler([1.0, 2.0, 4.0])

        out = apply_snr_weight(loss, timesteps, sch, gamma=10.0, v_prediction=False)

        # snr = [1,2,4], min_snr_gamma = same (gamma=10), weight = min_snr_gamma / snr = 1
        expected = loss.clone()
        assert torch.allclose(out, expected)

    def test_clamped_snr(self):
        """When SNR exceeds gamma, it gets clamped."""
        loss = torch.tensor([1.0, 2.0, 3.0])
        timesteps = torch.tensor([0, 1, 2], dtype=torch.int64)
        sch = make_fake_scheduler([0.5, 2.0, 4.0])

        gamma = 1.0
        out = apply_snr_weight(loss, timesteps, sch, gamma=gamma, v_prediction=False)

        snr = torch.tensor([0.5, 2.0, 4.0])
        min_snr = torch.minimum(snr, torch.full_like(snr, gamma))  # [0.5, 1.0, 1.0]
        expected = loss * (min_snr / snr)  # [1*1, 2*0.5, 3*0.25]
        assert torch.allclose(out, expected)

    def test_v_prediction_mode(self):
        """v_prediction uses (snr + 1) in denominator."""
        loss = torch.tensor([1.0, 2.0, 3.0])
        timesteps = torch.tensor([0, 1, 2], dtype=torch.int64)
        sch = make_fake_scheduler([1.0, 2.0, 4.0])

        gamma = 10.0
        out = apply_snr_weight(loss, timesteps, sch, gamma=gamma, v_prediction=True)

        snr = torch.tensor([1.0, 2.0, 4.0])
        min_snr = snr  # no clamp with high gamma
        weight = min_snr / (snr + 1)  # [1/2, 2/3, 4/5]
        expected = loss * weight
        assert torch.allclose(out, expected)

    def test_v_prediction_with_clamping(self):
        """v_prediction with gamma clamping."""
        loss = torch.tensor([1.0, 2.0])
        timesteps = torch.tensor([0, 1], dtype=torch.int64)
        sch = make_fake_scheduler([0.5, 4.0])

        gamma = 1.0
        out = apply_snr_weight(loss, timesteps, sch, gamma=gamma, v_prediction=True)

        snr = torch.tensor([0.5, 4.0])
        min_snr = torch.minimum(snr, torch.full_like(snr, gamma))  # [0.5, 1.0]
        weight = min_snr / (snr + 1)  # [0.5/1.5, 1.0/5.0]
        expected = loss * weight
        assert torch.allclose(out, expected)


# --- Tests for get_snr_scale ---


class TestGetSnrScale:
    """Tests for get_snr_scale function."""

    def test_basic(self):
        """Basic SNR scale computation."""
        sch = make_fake_scheduler([0.5, 1.0, 3.0])
        timesteps = torch.tensor([0, 2], dtype=torch.int64)

        scale = get_snr_scale(timesteps, sch)

        snr = torch.tensor([0.5, 3.0])
        snr_clamped = torch.minimum(snr, torch.ones_like(snr) * 1000)
        expected = snr_clamped / (snr_clamped + 1)
        assert torch.allclose(scale, expected)

    def test_clamp_very_large_snr(self):
        """Very large SNR (like at t=0) gets clamped to 1000."""
        sch = make_fake_scheduler([1e9])
        timesteps = torch.tensor([0], dtype=torch.int64)

        scale = get_snr_scale(timesteps, sch)

        # Clamped to 1000
        snr = torch.tensor([1000.0])
        expected = snr / (snr + 1)
        assert torch.allclose(scale, expected)

    def test_all_timesteps(self):
        """Test with multiple timesteps covering different SNR values."""
        sch = make_fake_scheduler([100.0, 10.0, 1.0, 0.1])
        timesteps = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

        scale = get_snr_scale(timesteps, sch)

        snr = torch.tensor([100.0, 10.0, 1.0, 0.1])
        expected = snr / (snr + 1)
        assert torch.allclose(scale, expected)


# --- Tests for scale_v_prediction_loss_like_noise_prediction ---


class TestScaleVPredictionLossLikeNoisePrediction:
    """Tests for scale_v_prediction_loss_like_noise_prediction function."""

    def test_basic(self):
        """Verify it multiplies loss by get_snr_scale."""
        sch = make_fake_scheduler([1.0, 3.0])
        timesteps = torch.tensor([0, 1], dtype=torch.int64)
        loss = torch.tensor([2.0, 4.0])

        out = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, sch)

        scale = get_snr_scale(timesteps, sch)
        expected = loss * scale
        assert torch.allclose(out, expected)

    def test_with_batch(self):
        """Test with larger batch size."""
        sch = make_fake_scheduler([0.5, 1.0, 2.0, 4.0])
        timesteps = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        loss = torch.tensor([1.0, 2.0, 3.0, 4.0])

        out = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, sch)

        snr = torch.tensor([0.5, 1.0, 2.0, 4.0])
        scale = snr / (snr + 1)
        expected = loss * scale
        assert torch.allclose(out, expected)


# --- Tests for add_v_prediction_like_loss ---


class TestAddVPredictionLikeLoss:
    """Tests for add_v_prediction_like_loss function."""

    def test_basic_formula(self):
        """Verify formula: loss' = loss + loss/scale * v_pred_like_loss."""
        sch = make_fake_scheduler([1.0, 3.0])
        timesteps = torch.tensor([0, 1], dtype=torch.int64)
        loss = torch.tensor([2.0, 4.0])
        v_like = torch.tensor([0.5, 1.0])

        out = add_v_prediction_like_loss(loss, timesteps, sch, v_like)

        scale = get_snr_scale(timesteps, sch)
        expected = loss + loss / scale * v_like
        assert torch.allclose(out, expected)

    def test_zero_v_like(self):
        """When v_pred_like_loss is zero, output equals original loss."""
        sch = make_fake_scheduler([1.0, 2.0])
        timesteps = torch.tensor([0, 1], dtype=torch.int64)
        loss = torch.tensor([2.0, 4.0])
        v_like = torch.tensor([0.0, 0.0])

        out = add_v_prediction_like_loss(loss, timesteps, sch, v_like)

        assert torch.allclose(out, loss)


# --- Tests for apply_debiased_estimation ---


class TestApplyDebiasedEstimation:
    """Tests for apply_debiased_estimation function."""

    def test_eps_prediction(self):
        """Epsilon prediction uses 1/sqrt(snr) weighting."""
        sch = make_fake_scheduler([4.0, 9.0])  # sqrt([4,9]) = [2,3]
        timesteps = torch.tensor([0, 1], dtype=torch.int64)
        loss = torch.tensor([2.0, 3.0])

        out = apply_debiased_estimation(loss, timesteps, sch, v_prediction=False)

        snr = torch.tensor([4.0, 9.0])
        weight = 1 / torch.sqrt(snr)  # [0.5, 0.333...]
        expected = weight * loss
        assert torch.allclose(out, expected)

    def test_v_prediction(self):
        """V-prediction uses 1/(snr+1) weighting."""
        sch = make_fake_scheduler([4.0, 9.0])
        timesteps = torch.tensor([0, 1], dtype=torch.int64)
        loss = torch.tensor([2.0, 3.0])

        out = apply_debiased_estimation(loss, timesteps, sch, v_prediction=True)

        snr = torch.tensor([4.0, 9.0])
        weight = 1 / (snr + 1)  # [0.2, 0.1]
        expected = weight * loss
        assert torch.allclose(out, expected)

    def test_very_large_snr_clamped(self):
        """Very large SNR is clamped to 1000 before applying weight."""
        sch = make_fake_scheduler([1e12])
        timesteps = torch.tensor([0], dtype=torch.int64)
        loss = torch.tensor([1.0])

        out = apply_debiased_estimation(loss, timesteps, sch, v_prediction=False)

        # SNR clamped to 1000
        weight = 1 / torch.sqrt(torch.tensor([1000.0]))
        expected = weight * loss
        assert torch.allclose(out, expected)


# --- Tests for apply_masked_loss ---


class TestApplyMaskedLoss:
    """Tests for apply_masked_loss function."""

    def test_with_conditioning_images(self):
        """conditioning_images in [-1,1] converted to [0,1], R channel used."""
        # loss shape: (B, C, H, W)
        loss = torch.ones((2, 1, 4, 4), dtype=torch.float32)
        # conditioning_images: (B, 3, Hc, Wc) in [-1,1]
        # All zeros -> 0.5 after /2+0.5
        cond = torch.zeros((2, 3, 2, 2), dtype=torch.float32)
        batch = {"conditioning_images": cond}

        out = apply_masked_loss(loss, batch)

        # mask_image resized from 2x2 to 4x4 via area, all values remain 0.5
        assert out.shape == loss.shape
        assert torch.allclose(out, torch.full_like(loss, 0.5))

    def test_conditioning_images_varied_values(self):
        """Test with varied conditioning image values."""
        loss = torch.ones((1, 1, 2, 2), dtype=torch.float32)
        # -1 -> 0.0, 1 -> 1.0 after /2+0.5
        cond = torch.tensor([[[[-1.0, 1.0], [0.0, 0.5]]]], dtype=torch.float32)
        cond = cond.expand(1, 3, 2, 2).clone()  # Expand to 3 channels (RGB)
        batch = {"conditioning_images": cond}

        out = apply_masked_loss(loss, batch)

        # R channel: [-1,1,0,0.5] -> [0, 1, 0.5, 0.75] after /2+0.5
        expected_mask = torch.tensor([[[[0.0, 1.0], [0.5, 0.75]]]], dtype=torch.float32)
        expected = loss * expected_mask
        assert torch.allclose(out, expected)

    def test_with_alpha_masks(self):
        """alpha_masks in [0,1] used directly."""
        loss = torch.ones((1, 1, 4, 4), dtype=torch.float32)
        alpha = torch.full((1, 4, 4), 0.25, dtype=torch.float32)  # already 0..1
        batch = {"alpha_masks": alpha}

        out = apply_masked_loss(loss, batch)

        # alpha_masks unsqueezed -> (1, 1, 4, 4)
        assert torch.allclose(out, torch.full_like(loss, 0.25))

    def test_alpha_masks_varied_values(self):
        """Test alpha mask with varied values."""
        loss = torch.ones((1, 1, 2, 2), dtype=torch.float32) * 2.0
        alpha = torch.tensor([[[0.0, 0.5], [0.75, 1.0]]], dtype=torch.float32)
        batch = {"alpha_masks": alpha}

        out = apply_masked_loss(loss, batch)

        expected = loss * alpha.unsqueeze(1)
        assert torch.allclose(out, expected)

    def test_no_mask_returns_unchanged(self):
        """When no mask keys present, return original loss unchanged."""
        loss = torch.ones((1, 1, 4, 4), dtype=torch.float32)
        batch = {}

        out = apply_masked_loss(loss, batch)

        # Should return the same tensor object
        assert out is loss

    def test_alpha_masks_none_returns_unchanged(self):
        """When alpha_masks is None, return original loss unchanged."""
        loss = torch.ones((1, 1, 4, 4), dtype=torch.float32)
        batch = {"alpha_masks": None}

        out = apply_masked_loss(loss, batch)

        assert out is loss

    def test_conditioning_images_takes_priority(self):
        """conditioning_images takes priority over alpha_masks."""
        loss = torch.ones((1, 1, 2, 2), dtype=torch.float32)
        cond = torch.zeros((1, 3, 2, 2), dtype=torch.float32)  # -> 0.5 mask
        alpha = torch.ones((1, 2, 2), dtype=torch.float32)  # -> 1.0 mask
        batch = {"conditioning_images": cond, "alpha_masks": alpha}

        out = apply_masked_loss(loss, batch)

        # conditioning_images used, not alpha_masks
        expected = torch.full_like(loss, 0.5)
        assert torch.allclose(out, expected)

    def test_mask_resize_interpolation(self):
        """Test that mask is resized to match loss spatial dimensions."""
        loss = torch.ones((1, 1, 8, 8), dtype=torch.float32)
        # Small alpha mask that will be upscaled
        alpha = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)  # 2x2
        batch = {"alpha_masks": alpha}

        out = apply_masked_loss(loss, batch)

        # Check shape matches
        assert out.shape == loss.shape
        # Area interpolation from 2x2 to 8x8: each 2x2 source pixel covers 4x4 region
        # Top-left quadrant should be 1.0, top-right 0.0, etc.
        assert torch.allclose(out[:, :, :4, :4], torch.ones((1, 1, 4, 4)))
        assert torch.allclose(out[:, :, :4, 4:], torch.zeros((1, 1, 4, 4)))
