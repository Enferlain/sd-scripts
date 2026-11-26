"""The loss pieces a strategy composes: derivatives, mass, and the reference step's numbers bit for bit on GPU."""

from pathlib import Path

import pytest
import torch

from library.losses.elementwise import l2_f64
from library.losses.reductions import Loss, per_sample_mean, sample_mean, weigh
from library.losses.weightings import min_snr_v_weight

ROOT = Path(__file__).resolve().parents[3]
BASELINE = ROOT / "docs_design/models-strategy-trainer/baseline"
STEP = BASELINE / "steps/finetune/step_000.safetensors"
REFERENCE = BASELINE / "reference/finetune_step_000.safetensors"


def test_l2_is_float64_and_differentiable_in_both_arguments() -> None:
    prediction = torch.tensor([[1.0, 2.0]], requires_grad=True)
    target = torch.tensor([[0.5, 4.0]], requires_grad=True)
    errors = l2_f64(prediction, target)
    assert errors.dtype == torch.float64 and torch.equal(errors, torch.tensor([[0.25, 4.0]], dtype=torch.float64))
    errors.sum().backward()
    assert prediction.grad is not None and torch.equal(prediction.grad, torch.tensor([[1.0, -4.0]])), "2 (prediction - target)"
    assert target.grad is not None and torch.equal(target.grad, torch.tensor([[-1.0, 4.0]])), (
        "plain math: whether a target is fixed is the strategy's"
    )


def test_a_loss_reports_the_number_of_samples_it_divided_by_as_its_mass() -> None:
    errors = torch.arange(24, dtype=torch.float64).reshape(3, 2, 4)
    losses = per_sample_mean(errors)
    assert torch.equal(losses, errors.mean(dim=(1, 2)))
    assert torch.equal(per_sample_mean(losses), losses), "one value per sample is already each sample's mean"
    loss = sample_mean(weigh(losses, torch.tensor([1.0, 0.0, 2.0])))
    assert isinstance(loss, Loss) and loss.mass == 3.0
    assert torch.equal(loss.value, (losses * torch.tensor([1.0, 0.0, 2.0])).mean())


def test_min_snr_for_v_prediction_caps_the_signal_to_noise_ratio() -> None:
    snr = torch.tensor([0.0, 1.0, 5.0, 100.0])
    weights = min_snr_v_weight(snr, 5.0)
    assert weights.dtype == torch.float32
    assert torch.equal(weights, torch.tensor([0.0, 0.5, 5.0 / 6.0, 5.0 / 101.0])), "min(snr, 5) / (snr + 1), finite at zero SNR"


@pytest.mark.requires_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.skipif(not (STEP.exists() and REFERENCE.exists()), reason="needs the baseline recording and its reference")
def test_the_loss_reproduces_the_reference_step_bit_for_bit() -> None:
    """On the card the reference was made on: the per-sample squared error, the weighted per-sample loss and the loss, from its output and target."""
    from safetensors.torch import load_file

    device = torch.device("cuda", 1 if torch.cuda.device_count() > 1 else 0)
    recorded, reference = load_file(str(STEP)), load_file(str(REFERENCE))
    prediction, target = reference["output"].to(device), reference["target"].to(device)
    snr = reference["snr"].to(device)

    squared_error = per_sample_mean(l2_f64(prediction, target))
    assert torch.equal(squared_error.cpu(), reference["squared_error"])
    weighted = weigh(weigh(squared_error, recorded["batch/loss_weights"].to(device)), min_snr_v_weight(snr, 5.0))
    assert torch.equal(weighted.cpu(), reference["per_sample_loss"])
    loss = sample_mean(weighted)
    assert torch.equal(loss.value.cpu(), reference["loss"]) and loss.mass == 2.0
