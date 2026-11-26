"""One real SDXL fine-tune step on the GPU through the Trainer, from a test-only accepted run (task 16.2).

The step is the baseline's recorded fine-tune step 0: its batch and random
draws come from the recording, and the result must equal the reference made
in this environment by today's code (``baseline/reference_step.py``) bit for
bit: the UNet's inputs and output, the target, the losses, the gradient norm
and every parameter after the AdamW8bit step.

The library code is plain (``library/models/sdxl/assembly.py``,
``library/objectives``, ``library/losses``). What the IR needs to know about
each function is stated here, standing in for the strategy: the declarations
wrap the plain functions on the system side. The UNet's calls run under bf16
autocast in their declared wrapper until route precision setups exist, and
the UNet route's preparation turns on its gradient checkpointing.

The step runs in its own process: deterministic cuBLAS needs its setting
before CUDA starts, as the reference did.
"""

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from library.iro.frontend import implementation
from library.iro.seams import Effect
from library.losses.elementwise import l2_f64
from library.losses.reductions import per_sample_mean, sample_mean, weigh
from library.losses.weightings import min_snr_v_weight
from library.models.denoisers.sd_unet import SDUNet
from library.models.sdxl import assembly
from library.objectives.formulations.ddpm import ddpm_noise, ddpm_snr, ddpm_v_target
from library.objectives.schedules import scaled_linear, zero_terminal_snr
from library.training.units import DERIVATIVE
from library.training.units.standard import installed as standard

ROOT = Path(__file__).resolve().parents[3]
BASELINE = ROOT / "docs_design/models-strategy-trainer/baseline"
RECORDING = BASELINE / "steps/finetune"
REFERENCE = BASELINE / "reference/finetune_step_000"
CHECKPOINT = Path("/mnt/d/stable-diffusion-webui-reForge/models/Stable-diffusion/21862-seele_pop3_v-pred.safetensors")
DTYPE = torch.bfloat16
GAMMA = 5.0

STRATEGY = """iro.strategy @sdxl_step uses(@load_unet, @schedule, @batch, @timesteps, @noisy, @context, @vector, @v_target, @snr, @loss_weights, @min_snr, @denoise, @l2_f64, @per_sample_mean, @weigh, @sample_mean, @standard.contribute, @standard.advance) {
  %unet = iro.resource @unet : participant materialize(@load_unet, "CHECKPOINT")
  %unit = iro.train.unit @denoiser of %unet optimizer @adamw8bit(lr 1e-05, weight_decay 0.01) max_grad_norm 1.0
  iro.sequence @once capture(%unet, %unit) {
    iro.train.window %unit members 1 {
      %schedule = iro.call @schedule() -> schedule
      %batch = iro.call @batch() -> batch
      %t = iro.call @timesteps(%batch) -> timesteps
      %noisy = iro.call @noisy(%schedule, %batch) -> latents
      %context = iro.call @context(%batch) -> context
      %vector = iro.call @vector(%batch) -> vector
      %target = iro.call @v_target(%schedule, %batch) -> target
      %snr = iro.call @snr(%schedule, %t) -> snr
      %lw = iro.call @loss_weights(%batch) -> weights
      %sw = iro.call @min_snr(%snr) -> weights
      %loss = iro.access %unet protected_current {
      ^(%u):
        %prediction = iro.call @denoise(%u, %noisy, %t, %context, %vector) -> retained prediction
        %errors = iro.call @l2_f64(%prediction, %target) -> retained errors
        %losses = iro.call @per_sample_mean(%errors) -> retained losses
        %weighted = iro.call @weigh(%losses, %lw) -> retained losses
        %snr_weighted = iro.call @weigh(%weighted, %sw) -> retained losses
        iro.call @sample_mean(%snr_weighted) -> retained loss
      }
      iro.train.contribute %loss to %unit
    }
  }
}
"""


@dataclass(frozen=True)
class Batch:
    """The recorded step's batch and random draws, on the device."""

    latents: torch.Tensor
    hidden_states1: torch.Tensor
    hidden_states2: torch.Tensor
    pooled2: torch.Tensor
    loss_weights: torch.Tensor
    original_hw: list
    crop_top_left: list
    target_hw: list
    noise: torch.Tensor
    timesteps: torch.Tensor


SEEN: dict[str, torch.Tensor] = {}  # what the step computed, for the comparison; module state, which no code record covers
BATCH: list[Batch] = []  # the recorded batch, set before the run; module state, as the unit tests' batches are


def sha256(tensor: torch.Tensor) -> str:
    """The SHA-256 of a tensor's exact bytes, as the reference computes it."""
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def recorded_batch(device: torch.device) -> Batch:
    """The recording's batch and draws, on the device."""
    recorded = load_file(str(RECORDING / "step_000.safetensors"))
    notes = json.loads((RECORDING / "step_000.json").read_text())
    count = len(notes["batch/image_ids"])
    return Batch(
        recorded["batch/latents"].to(device),
        recorded["batch/text_encoder_outputs/hidden_state1"],
        recorded["batch/text_encoder_outputs/hidden_state2"],
        recorded["batch/text_encoder_outputs/pool2"],
        recorded["batch/loss_weights"].to(device),
        [notes[f"batch/conditionings/{i}/original_size_hw"] for i in range(count)],
        [notes[f"batch/conditionings/{i}/crop_top_left"] for i in range(count)],
        [notes[f"batch/conditionings/{i}/target_size_hw"] for i in range(count)],
        recorded["noise"].to(device),
        recorded["timesteps"].to(device),
    )


# The step's calls: the plain library functions, wrapped where the step decides a dtype or records a value.


def seen(name: str, value: torch.Tensor) -> torch.Tensor:
    SEEN[name] = value.detach()
    return value


def load_unet(path: str):
    return assembly.load_unet(path, torch.device("cuda", torch.accelerator.current_device_index()), DTYPE)


def schedule():
    return zero_terminal_snr(scaled_linear(assembly.BETA_START, assembly.BETA_END, assembly.TRAIN_STEPS))


def batch():
    return BATCH[0]


def timesteps(b):
    return b.timesteps


def loss_weights(b):
    return b.loss_weights


def noisy(s, b):
    return seen("noisy_latents", ddpm_noise(s, b.latents, b.noise, b.timesteps).to(DTYPE))  # the UNet's input dtype


def sizes(b):
    return assembly.size_embedding(b.original_hw, b.crop_top_left, b.target_hw, b.latents.device, DTYPE)


def context(b):
    return seen("context", assembly.conditioning(b.hidden_states1, b.hidden_states2, b.pooled2, sizes(b), DTYPE)[0])


def vector(b):
    return seen("vector", assembly.conditioning(b.hidden_states1, b.hidden_states2, b.pooled2, sizes(b), DTYPE)[1])


def v_target(s, b):
    return seen("target", ddpm_v_target(s, b.latents, b.noise, b.timesteps))


def snr(s, t):
    return seen("snr", ddpm_snr(s, t))


def min_snr(values):
    return min_snr_v_weight(values, GAMMA)


def denoise(unet, latents, t, ctx, vec):
    with torch.autocast(device_type="cuda", dtype=DTYPE):  # the route's precision, until route precision setups exist
        return seen("output", assembly.denoise(unet, latents, t, ctx, vec))


def l2(prediction, target):
    return l2_f64(prediction.float(), target.float())  # today's loss reads both in float32 first


def mean(errors):
    return seen("squared_error", per_sample_mean(errors))


def final(losses):
    loss = sample_mean(seen("per_sample_loss", losses))
    seen("loss", loss.value)
    return loss


def installed() -> dict:
    """The step's implementations, declared here for the plain functions above."""
    device = [Effect.DEVICE]
    derivative = {DERIVATIVE: [(0, 0)]}
    declared = {
        "load_unet": implementation(inputs=("str",), outputs=("unet",), name="load_unet")(load_unet),
        "schedule": implementation(outputs=("schedule",), name="schedule")(schedule),
        "batch": implementation(outputs=("batch",), name="batch")(batch),
        "timesteps": implementation(inputs=("batch",), outputs=("timesteps",), name="timesteps")(timesteps),
        "loss_weights": implementation(inputs=("batch",), outputs=("weights",), name="loss_weights")(loss_weights),
        "noisy": implementation(inputs=("schedule", "batch"), outputs=("latents",), effects=device, name="noisy")(noisy),
        "context": implementation(inputs=("batch",), outputs=("context",), effects=device, name="context")(context),
        "vector": implementation(inputs=("batch",), outputs=("vector",), effects=device, name="vector")(vector),
        "v_target": implementation(inputs=("schedule", "batch"), outputs=("target",), effects=device, name="v_target")(v_target),
        "snr": implementation(inputs=("schedule", "timesteps"), outputs=("snr",), effects=device, name="snr")(snr),
        "min_snr": implementation(inputs=("snr",), outputs=("weights",), effects=device, name="min_snr")(min_snr),
        "denoise": implementation(
            inputs=("protected_current lease", "latents", "timesteps", "context", "vector"),
            outputs=("retained prediction",),
            effects=device,
            name="denoise",
            relations=derivative,
        )(denoise),
        "l2_f64": implementation(
            inputs=("retained prediction", "target"), outputs=("retained errors",), effects=device, name="l2_f64", relations=derivative
        )(l2),
        "per_sample_mean": implementation(
            inputs=("retained errors",), outputs=("retained losses",), effects=device, name="per_sample_mean", relations=derivative
        )(mean),
        "weigh": implementation(
            inputs=("retained losses", "weights"), outputs=("retained losses",), effects=device, name="weigh", relations=derivative
        )(weigh),
        "sample_mean": implementation(
            inputs=("retained losses",), outputs=("retained loss",), effects=device, name="sample_mean", relations=derivative
        )(final),
    }
    return {**declared, **standard()}


def prepare_unet(unet: object) -> object:
    """The UNet route's preparation: its gradient checkpointing turned on, as the reference ran it."""
    assert isinstance(unet, SDUNet)
    unet.checkpointing(True)
    return unet


def run_step() -> dict:
    """Runs the step through the Trainer and returns what to compare with the reference."""
    from library.iro.devices import Host
    from library.iro.devices.cuda import Cuda
    from library.iro.prepare import HOST, DeviceFacts, Target
    from library.runstate.authority import NORMAL
    from library.training.trainer import Trainer
    from library.training.words import contract
    from tests.unit.iro.accepted_runs import accepted_run

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    index = 1 if torch.cuda.device_count() > 1 else 0
    torch.cuda.set_device(index)
    device = Cuda(index)
    BATCH[:] = [recorded_batch(device.device)]
    implementations = installed()
    run = accepted_run(STRATEGY.replace("CHECKPOINT", str(CHECKPOINT)), contract(), installed=implementations)
    trainer = Trainer(
        run,
        contract(),
        implementations,
        Target((HOST, DeviceFacts.of(device)), device=device.name),
        {"host": Host(), device.name: device},
        preparations={("unet", NORMAL): prepare_unet},
    )
    outcome = trainer.train()
    if outcome.status != "succeeded":
        return {"status": outcome.status, "error": str(outcome.error)}
    authority, optimization = trainer.authority, trainer.optimization
    assert authority is not None and optimization is not None
    binding = authority.section.at("unet").binding
    assert binding is not None and isinstance(binding.value, torch.nn.Module)
    unet = binding.value
    last = optimization.live("denoiser").last
    assert last is not None and isinstance(last.norm, torch.Tensor)
    return {
        "status": outcome.status,
        "tensors": {name: value.cpu() for name, value in SEEN.items()} | {"grad_norm": last.norm.detach().cpu()},
        "param_sha256": {f"denoiser.{name}": sha256(p) for name, p in unet.named_parameters()},
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
    }


def main() -> None:
    """Runs the step in this process and writes what it found to the path given."""
    result = run_step()
    out = Path(sys.argv[1])
    if result["status"] == "succeeded":
        torch.save(result["tensors"], out.with_suffix(".pt"))
        result["tensors"] = sorted(result["tensors"])
    out.write_text(json.dumps(result))


@pytest.mark.requires_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.skipif(not CHECKPOINT.exists(), reason="needs the baseline's base model")
@pytest.mark.skipif(not REFERENCE.with_suffix(".json").exists(), reason="needs the reference step")
def test_one_sdxl_step_through_the_trainer_equals_the_reference_bit_for_bit(tmp_path: Path) -> None:
    out = tmp_path / "step.json"
    env = os.environ | {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    done = subprocess.run(
        [sys.executable, "-m", "tests.unit.training_core.test_sdxl_step", str(out)], cwd=ROOT, env=env, capture_output=True, text=True
    )
    assert done.returncode == 0 and out.exists(), done.stderr[-4000:]
    result = json.loads(out.read_text())
    assert result["status"] == "succeeded", result.get("error")
    ours = torch.load(out.with_suffix(".pt"))
    reference = load_file(str(REFERENCE.with_suffix(".safetensors")))
    differing = [name for name, value in ours.items() if not torch.equal(value, reference[name])]
    assert not differing, f"differ from the reference: {differing}"
    record = json.loads(REFERENCE.with_suffix(".json").read_text())
    wanted = {entry["name"]: entry["param_sha256"] for entry in record["parameters"]}
    assert result["param_sha256"].keys() == wanted.keys()
    changed = [name for name, digest in wanted.items() if result["param_sha256"][name] != digest]
    assert not changed, f"{len(changed)} of {len(wanted)} parameters differ after the step, the first {changed[:3]}"


if __name__ == "__main__":
    main()
