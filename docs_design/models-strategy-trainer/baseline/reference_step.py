"""A reference for the SDXL fine-tune step, made in this environment by today's code (task 16.2).

The baseline README's pass criteria (refined 2026-11-16) check exact correctness here, not against the
Windows recording: the new system's step must equal a reference step made in this environment by
running today's model code on the recorded step's inputs with PyTorch's deterministic algorithms on,
bit for bit: model output, loss, gradients and the optimizer update. This script makes that reference.

The recording (steps/finetune/step_000) supplies the step's batch, its random draws (noise and
timesteps) and its configuration. Everything else is computed here by today's functions, in the order
today's trainer calls them (library/strategies/sdxl/diffusion.py, library/training/phases/training_loop.py):
the DDPM schedule with zero-terminal SNR and its noising, the cached text conditioning, SDXL's size
conditioning and the UNet call (bf16 weights from the base model, SDPA attention, autocast bf16), the
v-prediction target, the float64 l2 loss with the batch's loss weights and min-SNR weighting, Accelerate's
backward, clipping, and bitsandbytes' AdamW8bit step. Gradient checkpointing is non-reentrant, the mode
the new system uses, where today's trainer defaults to reentrant; xformers is not installed here, so
attention is SDPA. Both are noted in the output.

The step runs twice from the same initial state, and nothing is written unless both runs agree bit for
bit. The output holds the step's tensors and, for every trainable tensor in the recording's order, the
same summaries the recording keeps (sum, L2 norm, 64 strided samples) and the SHA-256 of its exact bytes,
for its gradient before the optimizer step and for its value after it.

Before the step, the loaded weights are checked against the recording's initial summaries, so the
reference starts from the recording's base model.

Usage, from the repository root (needs the base model at its Windows path and about 18 GB on the GPU):

    .venv-wsl/bin/python -I docs_design/models-strategy-trainer/baseline/reference_step.py [--device cuda:1] [--check]

With --check it writes nothing and compares a fresh run, in this new process, with the stored reference
bit for bit, exiting with status 1 on any difference; that is how reproducibility across processes,
which a later test relies on, is shown. It sets CUBLAS_WORKSPACE_CONFIG=:4096:8 itself, before CUDA
starts, as deterministic cuBLAS needs.
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must precede CUDA's start for deterministic cuBLAS

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import library.models.sdxl.unet as sdxl_unet  # noqa: E402
from accelerate import init_empty_weights  # noqa: E402
from library.losses.huber import get_huber_threshold_if_needed  # noqa: E402
from library.losses.loss import conditional_loss  # noqa: E402
from library.models.sdxl.conversion import _load_state_dict_on_device  # noqa: E402
from library.objectives.ddpm import build_ddpm_noise_scheduler, build_ddpm_training_target, post_process_ddpm_loss  # noqa: E402
from library.strategies.base.context import StrategyPhase  # noqa: E402
from library.strategies.sdxl.conditioning import SdxlConditioning, SdxlConditioningStrategy  # noqa: E402
from library.strategies.sdxl.denoiser import SdxlDenoiserCallingStrategy  # noqa: E402
from library.utils.torch_utils import set_torch_cuda_reduced_precision  # noqa: E402

RECORDING = HERE / "steps" / "finetune"
OUT = HERE / "reference"
NAME = "finetune_step_000"
SAMPLES = 64  # as the recording keeps them: record_steps.py
WEIGHT_DTYPE = torch.bfloat16  # performance.precision.mixed_precision: bf16; the checkpoint's UNet tensors are bf16


class Accelerator:
    """What today's strategy functions read from Accelerate's accelerator in a one-process bf16 run."""

    def __init__(self, device: torch.device):
        self.device = device

    def autocast(self) -> torch.autocast:
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)


def windows_path(path: str) -> Path:
    """The WSL path of a Windows path the recording's config names, such as D:/models/x."""
    if len(path) > 2 and path[1] == ":":
        return Path("/mnt") / path[0].lower() / path[3:]
    return Path(path)


def sha256(tensor: torch.Tensor) -> str:
    """The SHA-256 of a tensor's exact bytes: contiguous, in its own dtype, little-endian, on the host."""
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def summarize(tensors: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    """Per-tensor sum, L2 norm and 64 strided samples, as record_steps.py keeps them."""
    sums = torch.zeros(len(tensors), dtype=torch.float64)
    norms = torch.zeros(len(tensors), dtype=torch.float64)
    samples = torch.zeros(len(tensors), SAMPLES, dtype=torch.float32)
    for index, tensor in enumerate(tensors):
        flat = tensor.detach().reshape(-1)
        wide = flat.double()
        sums[index] = wide.sum().item()
        norms[index] = wide.norm().item()
        positions = (torch.arange(SAMPLES, dtype=torch.long) * flat.numel()) // SAMPLES
        samples[index] = flat[positions.to(flat.device)].float().cpu()
    return {"sum": sums, "l2": norms, "sample": samples}


def load_unet(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    """Today's SDXL UNet loaded as today's loader does (library/models/sdxl/conversion.py), and its initial state on the host."""
    state: dict[str, torch.Tensor] = {}
    with safe_open(str(path), "pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith("model.diffusion_model."):
                state[key.removeprefix("model.diffusion_model.")] = f.get_tensor(key)
    initial = dict(state)  # the loader consumes its dictionary; these host tensors stay for the second run
    with init_empty_weights():
        unet = sdxl_unet.SdxlUNet2DConditionModel()
    _load_state_dict_on_device(unet, state, device=device, dtype=WEIGHT_DTYPE)
    unet.set_use_sdpa(True)
    unet.enable_gradient_checkpointing()
    unet.train()
    unet.requires_grad_(True)
    return unet, {name: tensor.to(WEIGHT_DTYPE) for name, tensor in initial.items()}


def check_initial(trainable: list[torch.Tensor]) -> dict:
    """The loaded weights against the recording's initial summaries: samples must be identical; sums and norms are reported."""
    initial = load_file(str(RECORDING / "initial.safetensors"))
    ours = summarize(trainable)
    found = {
        "samples_identical": bool(torch.equal(ours["sample"], initial["param_sample"])),
        "sum_max_relative": float(((ours["sum"] - initial["param_sum"]).abs() / initial["param_sum"].abs().clamp_min(1e-300)).max()),
        "l2_max_relative": float(((ours["l2"] - initial["param_l2"]).abs() / initial["param_l2"].clamp_min(1e-300)).max()),
    }
    if not found["samples_identical"]:
        raise SystemExit(f"the loaded weights are not the recording's base model: {found}")
    return found


def run_step(cfg: Any, unet: torch.nn.Module, trainable: list[torch.Tensor], recorded: dict, notes: dict, device: torch.device) -> dict:
    """One training step as today's trainer computes it, from the recording's batch and random draws."""
    import bitsandbytes as bnb  # loads Triton, which turns the GIL on

    accelerator = Accelerator(device)
    optimizer = bnb.optim.AdamW8bit(trainable, lr=float(cfg.optimizer.learning_rates.denoiser), weight_decay=0.01, betas=(0.9, 0.999))
    scheduler = build_ddpm_noise_scheduler(cfg, device)

    conditionings = [
        SdxlConditioning(
            original_size_hw=tuple(notes[f"batch/conditionings/{i}/original_size_hw"]),
            crop_top_left=tuple(notes[f"batch/conditionings/{i}/crop_top_left"]),
            target_size_hw=tuple(notes[f"batch/conditionings/{i}/target_size_hw"]),
        )
        for i in range(len(notes["batch/image_ids"]))
    ]
    batch = {
        "text_encoder_outputs": {
            "hidden_state1": recorded["batch/text_encoder_outputs/hidden_state1"],
            "hidden_state2": recorded["batch/text_encoder_outputs/hidden_state2"],
            "pool2": recorded["batch/text_encoder_outputs/pool2"],
        },
        "conditionings": conditionings,
        "loss_weights": recorded["batch/loss_weights"],
    }

    # process_batch: cached latents as they are, cached text conditioning in the weight dtype.
    latents = recorded["batch/latents"].to(device)
    text_conds = SdxlConditioningStrategy._get_cached_text_conds(None, batch, accelerator, WEIGHT_DTYPE)  # ty: ignore[invalid-argument-type]

    # get_noise_pred_and_target: prepare_ddpm_training_inputs with the recorded draws, then the denoiser call.
    noise = recorded["noise"].to(device)
    timesteps = recorded["timesteps"].to(device)
    noisy_latents = scheduler.add_noise(latents, noise, timesteps).to(WEIGHT_DTYPE)
    if cfg.performance.memory.gradient_checkpointing:
        for x in noisy_latents:
            x.requires_grad_(True)
        for t in text_conds:
            t.requires_grad_(True)
    captured: dict[str, torch.Tensor] = {}
    hook = unet.register_forward_pre_hook(lambda _m, args: captured.update(context=args[2].detach(), vector=args[3].detach()))
    try:
        prediction = SdxlDenoiserCallingStrategy().call_denoiser(
            cfg,
            accelerator,
            unet,
            noisy_latents,
            timesteps,
            text_conds,
            batch,
            WEIGHT_DTYPE,
            phase=StrategyPhase.TRAIN,
            global_step=0,
            is_train=True,
            train_denoiser=True,
        )
    finally:
        hook.remove()
    target = build_ddpm_training_target(scheduler, latents, noise, timesteps, cfg.objective.prediction)

    # process_batch's loss, then the no-op loss modifier's mean.
    huber_c = get_huber_threshold_if_needed(
        cfg.loss, cfg.loss.huber, timesteps, num_train_timesteps=int(scheduler.config.num_train_timesteps), alphas_cumprod=scheduler.alphas_cumprod
    )
    elementwise = conditional_loss(prediction.float(), target.float(), cfg.loss.loss_type, "none", huber_c, scale=float(cfg.loss.loss_scale))
    squared_error = elementwise.mean([1, 2, 3])
    weighted = post_process_ddpm_loss(squared_error * batch["loss_weights"].to(device), cfg, timesteps, scheduler)
    if cfg.loss.loss_multiplier:
        weighted.mul_(float(cfg.loss.loss_multiplier))
    loss = weighted.mean()

    # The training loop: Accelerate's backward, clipping, the optimizer step.
    (loss / int(cfg.training.gradient_accumulation_steps)).backward()
    norm = torch.nn.utils.clip_grad_norm_(trainable, float(cfg.optimizer.max_grad_norm))
    missing = [i for i, p in enumerate(trainable) if p.grad is None]
    if missing:
        raise RuntimeError(f"{len(missing)} trainable tensors received no gradient, the first at position {missing[0]}")
    gradients = [p.grad for p in trainable]
    result: dict[str, Any] = {
        "tensors": {
            "noisy_latents": noisy_latents.detach().cpu(),
            "timesteps": timesteps.cpu(),
            "context": captured["context"].cpu(),
            "vector": captured["vector"].cpu(),
            "output": prediction.detach().cpu(),
            "target": target.detach().cpu(),
            "snr": torch.stack([scheduler.all_snr[t] for t in timesteps]).cpu(),
            "squared_error": squared_error.detach().cpu(),
            "per_sample_loss": weighted.detach().cpu(),
            "loss": loss.detach().cpu(),
            "grad_norm": norm.detach().cpu(),
        },
        "grad": summarize(gradients),
        "grad_sha256": [sha256(g) for g in gradients],
    }
    optimizer.step()
    result["param"] = summarize(trainable)
    result["param_sha256"] = [sha256(p) for p in trainable]
    optimizer.zero_grad(set_to_none=True)
    return result


def same(a: dict, b: dict) -> list[str]:
    """Where two runs differ: tensor names and digest lists that are not bit-identical."""
    differing = [name for name in a["tensors"] if not torch.equal(a["tensors"][name], b["tensors"][name])]
    for part in ("grad", "param"):
        differing += [f"{part}_{key}" for key in a[part] if not torch.equal(a[part][key], b[part][key])]
        differing += [f"{part}_sha256"] if a[f"{part}_sha256"] != b[f"{part}_sha256"] else []
    return differing


def against_recording(result: dict, recorded: dict) -> dict:
    """How the reference reads against the Windows recording: exact where the computation is, close where kernels differ."""

    def compare(ours: torch.Tensor, theirs: torch.Tensor) -> dict:
        ours, theirs = ours.double(), theirs.double()
        return {
            "identical": bool(torch.equal(ours, theirs)),
            "exact_fraction": float((ours == theirs).double().mean()),
            "relative_l2": float((ours - theirs).norm() / theirs.norm().clamp_min(1e-300)),
        }

    pairs = {
        "noisy_latents": recorded["noisy_latents"],
        "context": recorded["model_call_0/arg2"],
        "vector": recorded["model_call_0/arg3"],
        "output": recorded["model_call_0/output"],
        "target": recorded["target"],
        "per_sample_loss": recorded["per_sample_loss"],
        "loss": recorded["loss"],
    }
    found = {name: compare(result["tensors"][name], theirs) for name, theirs in pairs.items()}
    found["grad_global_norm"] = compare(result["grad"]["l2"].norm().reshape(1), recorded["grad_l2"].norm().reshape(1))
    found["grad_sample"] = compare(result["grad"]["sample"], recorded["grad_sample"])
    found["param_sample_after_step"] = compare(result["param"]["sample"], recorded["param_sample"])
    return found


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def environment(device: torch.device) -> dict:
    import accelerate
    import bitsandbytes
    import diffusers

    return {
        "git_head": git("rev-parse", "HEAD"),
        "git_dirty_library_files": [line for line in git("status", "--porcelain", "--", "library").splitlines()],
        "python": f"{platform.python_version()}{sys.abiflags}",
        "gil_enabled": getattr(sys, "_is_gil_enabled", lambda: True)(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "bitsandbytes": bitsandbytes.__version__,
        "accelerate": accelerate.__version__,
        "diffusers": diffusers.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "attention": "sdpa (xformers is not installed in this environment)",
        "gradient_checkpointing": "non-reentrant (today's trainer defaults to reentrant)",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda:1", help="the GPU to run on; PyTorch numbers the RTX 3080 cuda:1 on this machine")
    parser.add_argument("--check", action="store_true", help="write nothing; compare a fresh run with the stored reference bit for bit")
    arguments = parser.parse_args()
    device = torch.device(arguments.device)
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    sdxl_unet.USE_REENTRANT = False

    cfg = OmegaConf.load(RECORDING / "config.yaml")
    set_torch_cuda_reduced_precision(cfg.performance.precision)  # leaves PyTorch's defaults: both switches are off in the recording
    recorded = load_file(str(RECORDING / "step_000.safetensors"))
    notes = json.loads((RECORDING / "step_000.json").read_text())
    names = [entry["name"] for entry in json.loads((RECORDING / "parameters.json").read_text())]
    checkpoint = windows_path(str(cfg.model.pretrained_model_name_or_path))

    started = time.perf_counter()
    unet, initial = load_unet(checkpoint, device)
    by_name = dict(unet.named_parameters())
    trainable = [by_name[name.removeprefix("denoiser.")] for name in names]
    print(f"loaded {len(trainable)} trainable tensors from {checkpoint.name} in {time.perf_counter() - started:.0f} s", flush=True)
    initial_check = check_initial(trainable)
    print(f"the loaded weights match the recording's initial summaries: {initial_check}", flush=True)

    runs = []
    for attempt in range(2):
        if attempt:
            with torch.no_grad():
                for name, parameter in unet.named_parameters():
                    parameter.copy_(initial[name])
        started = time.perf_counter()
        runs.append(run_step(cfg, unet, trainable, recorded, notes, device))
        print(f"run {attempt + 1}: loss {float(runs[-1]['tensors']['loss']):.12g} in {time.perf_counter() - started:.0f} s", flush=True)
    differing = same(*runs)
    if differing:
        raise SystemExit(f"the two runs differ, so this environment is not deterministic for this step: {differing}")
    result = runs[0]
    compared = against_recording(result, recorded)
    tensors = dict(result["tensors"])
    for part in ("grad", "param"):
        tensors |= {f"{part}_{key}": value for key, value in result[part].items()}

    if arguments.check:
        stored = load_file(str(OUT / f"{NAME}.safetensors"))
        stored_record = json.loads((OUT / f"{NAME}.json").read_text())
        differing = sorted(set(stored) ^ set(tensors)) + [name for name in tensors if name in stored and not torch.equal(stored[name], tensors[name])]
        for part in ("grad", "param"):
            if [entry[f"{part}_sha256"] for entry in stored_record["parameters"]] != result[f"{part}_sha256"]:
                differing.append(f"{part}_sha256")
        if differing:
            raise SystemExit(f"this run differs from the stored reference: {differing}")
        print("this run, in a new process, matches the stored reference bit for bit")
        return

    OUT.mkdir(exist_ok=True)
    save_file(tensors, str(OUT / f"{NAME}.safetensors"))
    record = {
        "made": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "recording": {
            "step": str((RECORDING / "step_000.safetensors").relative_to(ROOT)),
            "step_sha256": hashlib.sha256((RECORDING / "step_000.safetensors").read_bytes()).hexdigest(),
            "checkpoint": str(cfg.model.pretrained_model_name_or_path),
        },
        "settings": {
            "prediction": str(cfg.objective.prediction),
            "zero_terminal_snr": bool(cfg.loss.regularization.zero_terminal_snr),
            "loss_type": str(cfg.loss.loss_type),
            "min_snr_gamma": float(cfg.loss.snr.min_snr_gamma),
            "max_grad_norm": float(cfg.optimizer.max_grad_norm),
            "optimizer": "bitsandbytes.optim.AdamW8bit(lr=1e-05, weight_decay=0.01, betas=(0.9, 0.999))",
            "autocast": "bfloat16",
        },
        "environment": environment(device),
        "initial_against_recording": initial_check,
        "repeat": "two runs from the same initial state, in one process, agreed bit for bit; run with --check to compare a new process",
        "digests": (
            "SHA-256 of each tensor's exact bytes: contiguous, in its own dtype (shape and dtype are listed beside it), in memory "
            "order on this little-endian machine. grad_sha256 is the gradient after clipping and before the optimizer step; "
            "param_sha256 is the value after the optimizer step. The order is the recording's parameters.json, which equals the "
            "UNet's named_parameters order."
        ),
        "notes": {
            "output": "the UNet's bfloat16 output under autocast; today's trainer's Accelerate wrapper casts it to float32, an exact cast",
            "grad_norm": "clip_grad_norm_'s total norm, in bfloat16 as torch computes it from bfloat16 gradients; it is below max_grad_norm, so clipping multiplied every gradient by exactly 1 and is not exercised by this step",
        },
        "parameters": [
            {"name": name, "shape": list(p.shape), "dtype": str(p.dtype), "grad_sha256": g, "param_sha256": v}
            for name, p, g, v in zip(names, trainable, result["grad_sha256"], result["param_sha256"], strict=True)
        ],
        "against_recording": compared,
    }
    (OUT / f"{NAME}.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({"against_recording": compared}, indent=1))
    print(f"wrote {OUT / NAME}.safetensors and .json; peak allocated {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
