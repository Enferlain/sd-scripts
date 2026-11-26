"""The DDPM formulation and its schedules: Diffusers as the reference, and the reference step's numbers bit for bit on GPU."""

from pathlib import Path
from typing import Any

import pytest
import torch

from library.objectives.formulations.ddpm import ddpm_noise, ddpm_snr, ddpm_v_target
from library.objectives.schedules import NoiseSchedule, scaled_linear, zero_terminal_snr

ROOT = Path(__file__).resolve().parents[3]
BASELINE = ROOT / "docs_design/models-strategy-trainer/baseline"
STEP = BASELINE / "steps/finetune/step_000.safetensors"
REFERENCE = BASELINE / "reference/finetune_step_000.safetensors"


def sdxl_schedule() -> NoiseSchedule:
    """The schedule the SDXL step trains with: SDXL's released one, rescaled to zero terminal SNR."""
    schedule = zero_terminal_snr(scaled_linear(0.00085, 0.012, 1000))
    assert isinstance(schedule, NoiseSchedule)
    return schedule


def test_the_scaled_linear_schedule_equals_diffusers() -> None:
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler  # the tests' reference implementation (D21)

    reference = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    schedule = scaled_linear(0.00085, 0.012, 1000)
    assert torch.equal(schedule.betas, reference.betas)
    assert torch.equal(schedule.alphas_cumprod, reference.alphas_cumprod)


def test_zero_terminal_snr_ends_at_zero_and_keeps_the_first_step() -> None:
    released = scaled_linear(0.00085, 0.012, 1000)
    rescaled = sdxl_schedule()
    assert rescaled.alphas_cumprod[-1] == 0 and rescaled.snr[-1] == 0, "the last step keeps no signal"
    assert torch.allclose(rescaled.alphas_cumprod[0], released.alphas_cumprod[0], rtol=1e-6), "the first step's signal stays"
    assert bool((rescaled.snr[:-1] > 0).all()) and bool((rescaled.snr[1:] <= rescaled.snr[:-1]).all()), "SNR falls step by step"
    assert torch.equal(released.betas, scaled_linear(0.00085, 0.012, 1000).betas), "the transform returns a new schedule"


def test_noising_and_the_target_follow_diffusers_on_the_host() -> None:
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    reference = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    schedule = scaled_linear(0.00085, 0.012, 1000)
    generator = torch.Generator().manual_seed(0)
    latents, noise = torch.randn(3, 4, 8, 8, generator=generator), torch.randn(3, 4, 8, 8, generator=generator)
    timesteps: Any = torch.tensor([0, 500, 999])  # Diffusers annotates IntTensor
    for dtype in (torch.float32, torch.bfloat16):  # the schedule is cast to the latents' dtype, as Diffusers casts it
        x, e = latents.to(dtype), noise.to(dtype)
        assert torch.equal(ddpm_noise(schedule, x, e, timesteps), reference.add_noise(x, e, timesteps))
        assert torch.equal(ddpm_v_target(schedule, x, e, timesteps), reference.get_velocity(x, e, timesteps))


def test_a_schedule_holds_float32_betas_on_the_host() -> None:
    with pytest.raises(ValueError, match="one float32 per step"):
        NoiseSchedule(torch.ones(4, dtype=torch.float64))


def test_noising_and_the_target_pass_on_the_derivative_of_latents_and_noise() -> None:
    schedule = sdxl_schedule()
    latents = torch.randn(2, 4, 2, 2, requires_grad=True)
    noise = torch.randn(2, 4, 2, 2, requires_grad=True)
    timesteps = torch.tensor([10, 900])
    (ddpm_noise(schedule, latents, noise, timesteps).sum() + ddpm_v_target(schedule, latents, noise, timesteps).sum()).backward()
    assert latents.grad is not None and noise.grad is not None


@pytest.mark.requires_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.skipif(not (STEP.exists() and REFERENCE.exists()), reason="needs the baseline recording and its reference")
def test_the_formulation_reproduces_the_reference_step_bit_for_bit() -> None:
    """On the card the reference was made on: the noisy latents, the v target and the SNR of the recorded fine-tune step 0."""
    from safetensors.torch import load_file

    device = torch.device("cuda", 1 if torch.cuda.device_count() > 1 else 0)
    recorded, reference = load_file(str(STEP)), load_file(str(REFERENCE))
    latents, noise = recorded["batch/latents"].to(device), recorded["noise"].to(device)
    timesteps = recorded["timesteps"].to(device)
    assert torch.equal(timesteps.cpu(), reference["timesteps"])
    schedule = sdxl_schedule()

    noisy = ddpm_noise(schedule, latents, noise, timesteps).to(torch.bfloat16)  # the denoiser's input dtype, as the step casts it
    assert torch.equal(noisy.cpu(), reference["noisy_latents"])
    assert torch.equal(ddpm_v_target(schedule, latents, noise, timesteps).cpu(), reference["target"])
    assert torch.equal(ddpm_snr(schedule, timesteps).cpu(), reference["snr"])
