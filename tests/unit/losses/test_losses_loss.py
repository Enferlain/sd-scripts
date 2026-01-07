import torch
import pytest
import math
from unittest.mock import MagicMock
from library.losses.loss import (
    LossRecorder,
    EMARecorder,
    stable_mse_loss,
    stable_l1_loss,
    stable_huber_loss,
    stable_smooth_l1_loss,
    stable_log_cosh_loss,
    stable_msle_loss,
    x_sigmoid_loss,
    standard_deviation_loss,
    conditional_loss,
    soft_welsch_loss,
    get_huber_threshold_if_needed,
)


@pytest.fixture
def predictions():
    return torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)


@pytest.fixture
def targets():
    return torch.tensor([1.5, 2.0, 2.5], dtype=torch.float32)


class TestLossRecorder:
    def test_initialization(self):
        recorder = LossRecorder()
        assert recorder.loss_list == []
        assert recorder.loss_total == 0.0

    def test_moving_average_empty(self):
        recorder = LossRecorder()
        assert recorder.moving_average == 0

    def test_add_epoch_zero(self):
        recorder = LossRecorder()
        recorder.add(epoch=0, step=0, loss=1.0)
        assert recorder.loss_list == [1.0]
        assert recorder.loss_total == 1.0
        assert recorder.moving_average == 1.0

        recorder.add(epoch=0, step=1, loss=2.0)
        assert recorder.loss_list == [1.0, 2.0]
        assert recorder.loss_total == 3.0
        assert recorder.moving_average == 1.5

    def test_add_subsequent_epoch(self):
        recorder = LossRecorder()
        # Initial epoch
        recorder.add(epoch=0, step=0, loss=1.0)
        recorder.add(epoch=0, step=1, loss=2.0)

        # Second epoch - should update existing step
        recorder.add(epoch=1, step=0, loss=0.5)
        # Old total was 3.0. Removed 1.0 (step 0), added 0.5. New total 2.5
        assert recorder.loss_list[0] == 0.5
        assert recorder.loss_total == 2.5
        assert recorder.moving_average == 1.25  # 2.5 / 2

        # Second epoch step 1
        recorder.add(epoch=1, step=1, loss=1.0)
        # Old total 2.5. Removed 2.0 (step 1), added 1.0. New total 1.5
        assert recorder.loss_list[1] == 1.0
        assert recorder.loss_total == 1.5
        assert recorder.moving_average == 0.75

    def test_add_subsequent_epoch_expansion(self):
        recorder = LossRecorder()
        recorder.add(epoch=0, step=0, loss=1.0)

        # Jump to step 2 in epoch 1
        recorder.add(epoch=1, step=2, loss=3.0)
        assert len(recorder.loss_list) == 3
        assert recorder.loss_list[2] == 3.0
        # step 1 should be filled with 0.0
        assert recorder.loss_list[1] == 0.0


class TestEMARecorder:
    def test_initialization(self):
        ema = EMARecorder(smoothing=0.1)
        assert ema.smoothing == 0.1
        assert ema.beta == 0.9
        assert ema.ema == 0.0
        assert ema.num_updates == 0

    def test_smoothing_bounds(self):
        with pytest.raises(ValueError):
            EMARecorder(smoothing=1.1)
        with pytest.raises(ValueError):
            EMARecorder(smoothing=-0.1)

    def test_update_logic(self):
        ema = EMARecorder(smoothing=0.5)
        ema.add(10.0)
        assert ema.ema == 5.0
        assert ema.num_updates == 1
        assert ema.average == 10.0

        ema.add(20.0)
        assert ema.ema == 12.5
        assert ema.num_updates == 2
        assert math.isclose(ema.average, 16.666666, rel_tol=1e-5)


class TestStableLosses:
    def test_stable_mse_loss(self, predictions, targets):
        loss = stable_mse_loss(predictions, targets)
        assert math.isclose(loss.item(), 0.1666666, rel_tol=1e-5)
        # Check dtype explicitly mentioned in feedback
        assert loss.dtype == torch.float64

        loss_sum = stable_mse_loss(predictions, targets, reduction="sum")
        assert math.isclose(loss_sum.item(), 0.5, rel_tol=1e-5)

    def test_stable_mse_loss_zero_diff(self):
        pred = torch.tensor([1.0, 2.0])
        targ = torch.tensor([1.0, 2.0])
        loss = stable_mse_loss(pred, targ)
        assert loss.item() < 1e-30

    def test_stable_l1_loss(self, predictions, targets):
        loss = stable_l1_loss(predictions, targets)
        assert math.isclose(loss.item(), 0.3333333, rel_tol=1e-5)
        assert loss.dtype == torch.float64

    def test_stable_huber_loss(self, predictions, targets):
        loss = stable_huber_loss(predictions, targets, delta=1.0)
        assert math.isclose(loss.item(), 0.0833333, rel_tol=1e-5)

        pred_large = torch.tensor([10.0])
        targ_large = torch.tensor([0.0])
        loss_large = stable_huber_loss(pred_large, targ_large, delta=1.0)
        assert math.isclose(loss_large.item(), 9.5, rel_tol=1e-5)

    def test_stable_smooth_l1_loss_eps_behavior(self):
        # If predictions == targets, loss should be eps (due to quadratic.add(eps))
        loss = stable_smooth_l1_loss(torch.tensor([0.0]), torch.tensor([0.0]), beta=1.0)
        assert loss.item() > 0

    def test_stable_smooth_l1_loss(self, predictions, targets):
        # This test passes because the value is large enough that eps doesn't matter
        loss = stable_smooth_l1_loss(predictions, targets, beta=1.0)
        assert math.isclose(loss.item(), 0.0833333, rel_tol=1e-5)

    @pytest.mark.parametrize(
        "loss_fn",
        [
            soft_welsch_loss,
            x_sigmoid_loss,
            stable_log_cosh_loss,
        ],
    )
    def test_loss_returns_valid_scalar(self, loss_fn, predictions, targets):
        loss = loss_fn(predictions, targets)
        assert isinstance(loss.item(), float)
        assert loss.item() >= 0

    def test_stable_msle_loss_negatives(self):
        # Test valid inputs
        pred = torch.tensor([1.0, 2.0])
        targ = torch.tensor([1.0, 2.0])
        loss = stable_msle_loss(pred, targ)
        assert loss.item() < 1e-5

        # Test negative inputs (should fail or produce NaN if not handled,
        # but function assumes inputs are valid for log, typical use case is non-negative pixel values)
        # We just verify it calculates without immediate crash on valid data

    def test_standard_deviation_loss_behavior(self, predictions, targets):
        # n = 3
        # squared_diff = [0.25, 0.0, 0.25]
        # sum = 0.5
        # mean_sq = 0.5/3 = 0.1666...
        # sqrt = 0.4082...
        # Function returns this scalar as "mean"
        loss = standard_deviation_loss(predictions, targets, reduction="mean")
        assert math.isclose(loss.item(), 0.408248, rel_tol=1e-4)

        # reduction='none' returns the logic-calculated scalar (std is a batch statistic)
        loss_none = standard_deviation_loss(predictions, targets, reduction="none")
        assert loss_none.dim() == 0


class TestConditionalLoss:
    def test_dispatcher(self, predictions, targets):
        # L2
        loss_l2 = conditional_loss(predictions, targets, loss_type="l2", reduction="mean")
        assert math.isclose(loss_l2.item(), 0.1666666, rel_tol=1e-5)

        # L1
        loss_l1 = conditional_loss(predictions, targets, loss_type="l1", reduction="mean")
        assert math.isclose(loss_l1.item(), 0.3333333, rel_tol=1e-5)

        # Huber with arg
        huber_c = torch.tensor(1.0)
        loss_dh = conditional_loss(predictions, targets, loss_type="standard_huber", reduction="mean", huber_c=huber_c)
        assert math.isclose(loss_dh.item(), 0.0833333, rel_tol=1e-5)

        # Huber custom variant
        loss_h = conditional_loss(predictions, targets, loss_type="huber", reduction="mean", huber_c=huber_c)
        # 2 * c * (sqrt(diff^2 + c^2) - c)
        # diffs: 0.5, 0.0, 0.5
        # diff^2: 0.25, 0.0, 0.25
        # inner: sqrt(0.25+1) = 1.118, sqrt(1)=1, sqrt(1.25)=1.118
        # terms: 1.118-1 = 0.118, 0, 0.118
        # sum: 0.236
        # mult by 2*c (2): 0.472
        # mean: 0.472 / 3 = 0.157...
        assert loss_h.item() > 0

        # Invalid type
        with pytest.raises(NotImplementedError):
            conditional_loss(predictions, targets, loss_type="invalid_loss_type", reduction="mean")


class TestGetHuberThreshold:
    def test_constant_schedule(self):
        from library.config.dataclasses.loss import LossConfig, HuberConfig

        loss_config = LossConfig(loss_type="huber")
        huber_config = HuberConfig(huber_schedule="constant", huber_c=0.1, huber_scale=1.0)

        timesteps = torch.tensor([1, 2, 3])
        noise_scheduler = MagicMock()

        result = get_huber_threshold_if_needed(loss_config, huber_config, timesteps, noise_scheduler)
        assert math.isclose(result.item(), 0.1, rel_tol=1e-5)

    def test_exponential_schedule(self):
        from library.config.dataclasses.loss import LossConfig, HuberConfig

        loss_config = LossConfig(loss_type="huber")
        huber_config = HuberConfig(huber_schedule="exponential", huber_c=0.1, huber_scale=1.0)

        timesteps = torch.tensor([0, 50, 100], dtype=torch.float32)
        noise_scheduler = MagicMock()
        noise_scheduler.config.num_train_timesteps = 100

        # Check callable and reasonable output
        result = get_huber_threshold_if_needed(loss_config, huber_config, timesteps, noise_scheduler)
        assert result.shape == timesteps.shape
        assert torch.all(result > 0)

    def test_snr_schedule(self):
        from library.config.dataclasses.loss import LossConfig, HuberConfig

        loss_config = LossConfig(loss_type="huber")
        huber_config = HuberConfig(huber_schedule="snr", huber_c=0.1)

        timesteps = torch.tensor([0, 1])
        noise_scheduler = MagicMock()
        noise_scheduler.alphas_cumprod = torch.tensor([0.9, 0.8])

        result = get_huber_threshold_if_needed(loss_config, huber_config, timesteps, noise_scheduler)
        assert result.shape == timesteps.shape

    def test_not_needed(self):
        from library.config.dataclasses.loss import LossConfig, HuberConfig

        loss_config = LossConfig(loss_type="l2")  # Not huber
        huber_config = HuberConfig()
        timesteps = torch.tensor([1])
        result = get_huber_threshold_if_needed(loss_config, huber_config, timesteps, None)
        assert result is None
