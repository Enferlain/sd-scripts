"""SDXL training fed by a producer through a stream, on the RTX 3080 (task 16.2).

Two runs, each in a fresh process started the way a launcher starts a run's:

- **Comparison with the baseline.** A producer hands the recorded, cached
  batches through a bounded stream and a repeat of windows trains on them:
  the baseline's workload, 50 steps of batch size 2. The run reports its
  seconds per step and peak memory in the training loop, its GIL state, and
  the GPU's tensor-unlock state, read from a BF16 GEMM, for reading against
  the baseline's recorded runs (``baseline/README.md``). Attention is SDPA
  where the baseline used xformers, and the recorded batches repeat, so the
  numbers compare as a reading, not an equality.
- **Coordination.** The producer also does real GPU work on the same card
  before it hands each batch over, and a handler reads the UNet's live
  weights with protected access after each step, computing a validation loss.
  Its training losses must equal the comparison run's step for step: neither
  the producer's device work nor the handler's reads change what training
  computes.
- **Oracle.** The coordination run's steps again, the validation read in
  plain sequence after each step instead of by the handler. A protected
  current read sees one whole version, current when it starts, so each of
  the handler's reads must have seen the weights of the step it was
  requested after or of a later one, and computed from them the value the
  plain read computed. A read is told apart by a fingerprint of the weights
  it saw, every parameter's float64 sum.

Work runs in turn on a device today, so the coordination run is no evidence
that concurrent GPU execution is safe. Encoding live with the SDXL VAE waits
for the component (task 18.1); the producer's GPU work stands in for it.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from library.iro.frontend import implementation
from library.iro.seams import Effect
from library.losses.reductions import per_sample_mean, sample_mean
from library.models.sdxl import assembly
from library.objectives.formulations.ddpm import ddpm_noise, ddpm_v_target
from library.objectives.schedules import scaled_linear, zero_terminal_snr
from library.performance.precision import mixed_precision
from library.training.units import DERIVATIVE
from tests.unit.training_core import test_sdxl_step as step

ROOT = Path(__file__).resolve().parents[3]
RECORDING = step.RECORDING
STEPS = 50
CHECKED_STEPS = 6  # the coordination run's steps, compared with the comparison run's first ones

STRATEGY = """iro.strategy @sdxl_window uses(@load_unet, @schedule, @produce, @timesteps, @noisy, @context, @vector, @v_target, @snr, @loss_weights, @min_snr, @denoise, @l2_f64, @per_sample_mean, @weigh, @sample_mean, @standard.contribute, @standard.advance HANDLER_USES) {
  %unet = iro.resource @unet : participant materialize(@load_unet, "CHECKPOINT") realize(@mixed_precision, "bfloat16")
  %unit = iro.train.unit @denoiser of %unet optimizer @adamw8bit(lr 1e-05, weight_decay 0.01) max_grad_norm 1.0
  %batches = iro.stream @batches : batch
             bounds(outstanding 2, ready 2)
  iro.parallel @all capture(%batches, %unet, %unit) {
    iro.producer @data into %batches {
    ^(%index: int):
      %b = iro.call @produce(%index) -> batch
      iro.yield %b
    }
    iro.repeat @training STEPS times capture(%batches, %unet, %unit) {
    ^(%k: int):
      iro.train.window %unit members 1 {
        %c = iro.take %batches
        %batch = iro.borrow %c
        %schedule = iro.call @schedule() -> schedule
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
        iro.dispose %c consumed
      }
      REQUEST
    }
    HANDLER
    iro.stop @data after @training
  }
}
"""

REQUEST = "iro.request @validate(%k) from @training at settled_success"
ORACLE = """%v = iro.access %unet protected_current {
      ^(%u):
        iro.call @validate_loss(%u, %k) -> float
      }"""
HANDLER = """iro.handler @validate sources [@training] max 1 capture(%unet) {
    ^(%x: int):
      %v = iro.access %unet protected_current {
      ^(%u):
        iro.call @validate_loss(%u, %x) -> float
      }
    }"""


# Module state, which no code record covers: the recorded batches, the run's mode and what it saw.
RECORDED = 3
"""The recorded steps the producer cycles through."""
BATCHES: list[step.Batch] = []
MODE: list[str] = ["comparison"]
LOSSES: list[torch.Tensor] = []
VALIDATIONS: list[tuple[int, float, str]] = []  # the step a read was requested after, its value, and the weights it saw
SCHEDULE: list[object] = []


def recorded(index: int) -> step.Batch:
    """Recorded step ``index``'s batch and draws, on the host."""
    tensors = load_file(str(RECORDING / f"step_{index:03d}.safetensors"))
    notes = json.loads((RECORDING / f"step_{index:03d}.json").read_text())
    count = len(notes["batch/image_ids"])
    return step.Batch(
        tensors["batch/latents"],
        tensors["batch/text_encoder_outputs/hidden_state1"],
        tensors["batch/text_encoder_outputs/hidden_state2"],
        tensors["batch/text_encoder_outputs/pool2"],
        tensors["batch/loss_weights"],
        [notes[f"batch/conditionings/{i}/original_size_hw"] for i in range(count)],
        [notes[f"batch/conditionings/{i}/crop_top_left"] for i in range(count)],
        [notes[f"batch/conditionings/{i}/target_size_hw"] for i in range(count)],
        tensors["noise"],
        tensors["timesteps"],
    )


def produce(index):
    """The next batch on the device. In the coordination run it does real GPU work first, standing in for encoding."""
    device = torch.device("cuda", torch.accelerator.current_device_index())
    b = BATCHES[index % len(BATCHES)]
    latents = b.latents.to(device)
    if MODE[0] == "coordination":
        pixels = torch.ones(2, 3, 1024, 1024, device=device) * 0.5
        work = torch.nn.functional.avg_pool2d(pixels, 8).flatten(1)
        (work @ work.T).sum()  # real device work, whose result nothing reads
        latents = (latents * 2.0) / 2.0  # an exact round trip through the device
    return step.Batch(
        latents,
        b.hidden_states1,
        b.hidden_states2,
        b.pooled2,
        b.loss_weights.to(device),
        b.original_hw,
        b.crop_top_left,
        b.target_hw,
        b.noise.to(device),
        b.timesteps.to(device),
    )


def schedule():
    if not SCHEDULE:
        SCHEDULE.append(zero_terminal_snr(scaled_linear(assembly.BETA_START, assembly.BETA_END, assembly.TRAIN_STEPS)))
    return SCHEDULE[0]


def final(losses):
    loss = sample_mean(losses)
    LOSSES.append(loss.value.detach())
    return loss


def fingerprint(unet) -> str:
    """Which weights a read saw: a digest of every parameter's float64 sum, each computed on the device."""
    sums = torch.stack([p.detach().sum(dtype=torch.float64) for p in unet.parameters()])
    return hashlib.sha256(sums.cpu().numpy().tobytes()).hexdigest()


def validate_loss(unet, x):
    seen = fingerprint(unet)
    b = produce(0)
    s = schedule()
    noisy = ddpm_noise(s, b.latents, b.noise, b.timesteps).to(step.DTYPE)
    context, vector = assembly.conditioning(b.hidden_states1, b.hidden_states2, b.pooled2, step.sizes(b), step.DTYPE)
    prediction = assembly.denoise(unet, noisy, b.timesteps, context, vector)
    loss = per_sample_mean((prediction.float() - ddpm_v_target(s, b.latents, b.noise, b.timesteps).float()) ** 2).mean()
    VALIDATIONS.append((int(x), float(loss), seen))
    return float(loss)


def installed() -> dict:
    found = step.installed()
    device = [Effect.DEVICE]
    found["produce"] = implementation(inputs=("int",), outputs=("batch",), effects=device, name="produce")(produce)
    found["schedule"] = implementation(outputs=("schedule",), name="schedule")(schedule)
    found["sample_mean"] = implementation(
        inputs=("retained losses",), outputs=("retained loss",), effects=device, name="sample_mean", relations={DERIVATIVE: [(0, 0)]}
    )(final)
    found["validate_loss"] = implementation(
        inputs=("protected_current lease", "int"), outputs=("float",), effects=device, name="validate_loss"
    )(validate_loss)
    return found


def bf16_tflops(device: torch.device) -> float:
    """A BF16 8192x8192 GEMM's rate, which tells the GPU's tensor-unlock state (baseline README, GPU state)."""
    a = torch.randn(8192, 8192, device=device, dtype=torch.bfloat16)
    for _ in range(3):
        a @ a
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(20):
        a @ a
    torch.cuda.synchronize(device)
    return 20 * 2 * 8192**3 / (time.perf_counter() - started) / 1e12


def run(mode: str, steps: int, index: int, deterministic: bool = True) -> dict:
    from library.iro.devices import Host
    from library.iro.devices.cuda import Cuda
    from library.iro.prepare import HOST, DeviceFacts, Target
    from library.runstate.authority import NORMAL
    from library.training.trainer import Trainer
    from library.training.words import contract
    from tests.unit.iro.accepted_runs import accepted_run

    MODE[0] = mode
    device = Cuda(index, deterministic=deterministic)
    torch.cuda.set_device(index)
    BATCHES[:] = [recorded(i) for i in range(RECORDED)]
    text = STRATEGY.replace("CHECKPOINT", str(step.CHECKPOINT)).replace("STEPS", str(steps))
    if mode == "coordination":
        text = text.replace("HANDLER_USES", ", @validate_loss").replace("REQUEST", REQUEST).replace("HANDLER", HANDLER)
    elif mode == "oracle":  # the validation in plain sequence after each step, as the program orders it
        text = text.replace("HANDLER_USES", ", @validate_loss").replace("REQUEST", ORACLE).replace("    HANDLER\n", "")
    else:
        text = text.replace(" HANDLER_USES", "").replace("      REQUEST\n", "").replace("    HANDLER\n", "")
    implementations = installed()
    trainer = Trainer(
        accepted_run(text, contract(), installed=implementations),
        contract(),
        implementations,
        Target((HOST, DeviceFacts.of(device)), device=device.name),
        {"host": Host(), device.name: device},
        preparations={("unet", NORMAL): step.prepare_unet},
        realizations={"mixed_precision": mixed_precision},
    )
    unlocked = bf16_tflops(device.device)
    executable = trainer.realize()
    torch.cuda.reset_peak_memory_stats(device.device)
    started = time.perf_counter()
    outcome = executable.execute()
    torch.cuda.synchronize(device.device)
    seconds = time.perf_counter() - started
    if outcome.status != "succeeded":
        return {"status": outcome.status, "error": str(outcome.error)}
    return {
        "status": outcome.status,
        "steps": steps,
        "seconds_per_step": seconds / steps,
        "training_loop_seconds": seconds,
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device.device) / 2**20,
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device.device) / 2**20,
        "gil": [fact.values[0] for fact in executable.facts if fact.kind == "gil"],
        "bf16_gemm_tflops": unlocked,
        "gpu": torch.cuda.get_device_name(device.device),
        "deterministic": deterministic,
        "losses": [float(loss) for loss in LOSSES],
        "validations": VALIDATIONS,
    }


def main() -> None:
    mode, steps, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    index = int(sys.argv[4]) if len(sys.argv) > 4 else (1 if torch.cuda.device_count() > 1 else 0)  # the RTX 3080, the reference's card
    deterministic = sys.argv[5] != "nondeterministic" if len(sys.argv) > 5 else True  # timing like the baseline's runs: off
    out.write_text(json.dumps(run(mode, steps, index, deterministic)))


LAUNCH = """
from library.iro import gil
gil.keep_off()  # as a launcher does, before anything else loads (D15)
import runpy, sys
sys.argv = ["test_sdxl_window", *sys.argv[1:]]
runpy.run_module("tests.unit.training_core.test_sdxl_window", run_name="__main__")
"""


def launched(mode: str, steps: int, out: Path) -> dict:
    env = {name: value for name, value in os.environ.items() if name != "CUBLAS_WORKSPACE_CONFIG"}
    done = subprocess.run(
        [sys.executable, "-c", LAUNCH, mode, str(steps), str(out)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800
    )
    assert done.returncode == 0 and out.exists(), done.stderr[-4000:]
    result = json.loads(out.read_text())
    assert result["status"] == "succeeded", result.get("error")
    return result


@pytest.mark.requires_gpu
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.skipif(not step.CHECKPOINT.exists(), reason="needs the baseline's base model")
@pytest.mark.skipif(not (RECORDING / "step_002.safetensors").exists(), reason="needs the baseline's recorded steps")
def test_sdxl_training_fed_through_a_stream_and_read_by_a_protected_handler(tmp_path: Path) -> None:
    comparison = launched("comparison", STEPS, tmp_path / "comparison.json")
    coordination = launched("coordination", CHECKED_STEPS, tmp_path / "coordination.json")
    oracle = launched("oracle", CHECKED_STEPS, tmp_path / "oracle.json")
    report = {key: value for key, value in comparison.items() if key not in ("losses", "validations")}
    print("\ncomparison run:", json.dumps(report, indent=1))
    print("coordination run: seconds per step", coordination["seconds_per_step"], "validations", coordination["validations"])

    assert comparison["gil"] == ["off"] and coordination["gil"] == ["off"]
    assert len(comparison["losses"]) == STEPS and all(loss == loss for loss in comparison["losses"])
    assert coordination["losses"] == comparison["losses"][:CHECKED_STEPS], (
        "the producer's device work and the handler's protected reads change nothing training computes"
    )
    assert [x for x, _, _ in coordination["validations"]] == list(range(CHECKED_STEPS)), "one validation after each step"
    reference = load_file(str(step.REFERENCE.with_suffix(".safetensors")))
    assert comparison["losses"][0] == float(reference["loss"]), "the first step's loss is the reference step's, bit for bit"
    cycle = RECORDED  # filled in the run's process, not this one
    repeated = [i for i in range(STEPS - cycle) if comparison["losses"][i] == comparison["losses"][i + cycle]]
    assert not repeated, f"a batch seen again gives another loss, since the steps between changed the weights; the same at {repeated}"
    assert oracle["losses"] == comparison["losses"][:CHECKED_STEPS], "the oracle trains as the comparison run does"
    assert [x for x, _, _ in oracle["validations"]] == list(range(CHECKED_STEPS))
    versions = {seen: (x, value) for x, value, seen in oracle["validations"]}
    assert len(versions) == CHECKED_STEPS, "each step leaves weights of its own, so a fingerprint tells which step a read saw"
    for k, value, seen in coordination["validations"]:
        assert seen in versions, f"the handler requested after step {k} saw weights no step left: a torn read"
        j, expected = versions[seen]
        assert j >= k, f"the handler requested after step {k} saw step {j}'s weights, older than its request"
        assert value == expected, f"from step {j}'s weights the handler computed {value}, and the plain read {expected}"


if __name__ == "__main__":
    main()
