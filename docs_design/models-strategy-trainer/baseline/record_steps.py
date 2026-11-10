"""Record a few training steps of the pinned baseline trainer.

This launcher runs the pinned revision's own ``train.py`` and only adds
observers. It does not change what the trainer computes. Run it from the root
of a checkout of the pinned revision, with that checkout's environment:

    python record_steps.py --config-name=<name> training.max_train_steps=3

Environment variables:

    RMST_BASELINE_OUT           Directory that receives the recordings (required).
    RMST_BASELINE_STEPS         Number of steps to record (default 3).
    RMST_BASELINE_FULL_INITIAL  Set to 1 to also write every trainable tensor
                                in full, before the first step, to
                                ``initial_full.safetensors``.

Each recorded step is written as ``step_NNN.safetensors`` with a
``step_NNN.json`` sidecar. ``parameters.json`` names the trainable tensors in
the order used by every per-parameter array.

Limits:

- Gradients are read where the trainer calls ``optimizer.step()``, which is
  after gradient synchronization and clipping.
- A recorded step is one optimizer step with one batch. Gradient accumulation
  is refused, because the observers would count micro-batches.
- Parameters and gradients are stored as per-tensor sums, norms and samples.
  Optimizer state is not stored.
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.getcwd())

import hydra
import torch
from safetensors.torch import save_file

import train as pinned_train
from library.strategies.sdxl import diffusion as sdxl_diffusion

OUT = Path(os.environ["RMST_BASELINE_OUT"])
STEPS = int(os.environ.get("RMST_BASELINE_STEPS", "3"))
SAMPLES = 64
MAX_FULL_ELEMENTS = 16_000_000
FULL_INITIAL = os.environ.get("RMST_BASELINE_FULL_INITIAL") == "1"


def _cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to("cpu").contiguous()


def _collect(prefix: str, value: Any, tensors: dict[str, torch.Tensor], notes: dict[str, Any]) -> None:
    """Store tensors in full and describe everything else."""
    if isinstance(value, torch.Tensor):
        notes[prefix] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        if value.numel() <= MAX_FULL_ELEMENTS:
            tensors[prefix] = _cpu(value)
        else:
            notes[prefix]["omitted"] = "too large; summary only"
            notes[prefix]["sum"] = float(value.detach().double().sum())
    elif isinstance(value, dict):
        for key, item in value.items():
            _collect(f"{prefix}/{key}", item, tensors, notes)
    elif isinstance(value, (list, tuple)):
        if all(isinstance(item, (str, int, float, bool, type(None))) for item in value):
            notes[prefix] = list(value)
        else:
            for index, item in enumerate(value):
                _collect(f"{prefix}/{index}", item, tensors, notes)
    elif isinstance(value, (str, int, float, bool, type(None))):
        notes[prefix] = value
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            _collect(f"{prefix}/{field.name}", getattr(value, field.name), tensors, notes)
    else:
        notes[prefix] = f"<{type(value).__name__}>"


def _drop_duplicates(tensors: dict[str, torch.Tensor], notes: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Keep one copy of bit-identical tensors and note the others as aliases."""
    kept: dict[str, torch.Tensor] = {}
    for key, tensor in tensors.items():
        match = next(
            (
                name
                for name, other in kept.items()
                if other.dtype == tensor.dtype and other.shape == tensor.shape and torch.equal(other, tensor)
            ),
            None,
        )
        if match is not None and tensor.numel() > 64:
            notes.setdefault(key, {})["same_as"] = match
        else:
            kept[key] = tensor
    return kept


def _sample_indices(count: int) -> torch.Tensor:
    return (torch.arange(SAMPLES, dtype=torch.long) * count) // SAMPLES


def _summarize(tensors: list[torch.Tensor | None]) -> dict[str, torch.Tensor]:
    """Per-tensor sum, L2 norm and fixed strided samples, in tensor order."""
    sums = torch.zeros(len(tensors), dtype=torch.float64)
    norms = torch.zeros(len(tensors), dtype=torch.float64)
    samples = torch.zeros(len(tensors), SAMPLES, dtype=torch.float32)
    present = torch.zeros(len(tensors), dtype=torch.bool)
    for index, tensor in enumerate(tensors):
        if tensor is None:
            continue
        flat = tensor.detach().reshape(-1)
        wide = flat.double()
        sums[index] = wide.sum().item()
        norms[index] = wide.norm().item()
        samples[index] = flat[_sample_indices(flat.numel()).to(flat.device)].float().cpu()
        present[index] = True
    return {"sum": sums, "l2": norms, "sample": samples, "present": present}


class Recorder:
    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer
        self.step = 0
        self.installed = False
        self.tensors: dict[str, torch.Tensor] = {}
        self.notes: dict[str, Any] = {}
        self.parameters: list[torch.nn.Parameter] = []
        self.model_calls = 0
        OUT.mkdir(parents=True, exist_ok=True)

    @property
    def active(self) -> bool:
        return self.step < STEPS

    def put(self, prefix: str, value: Any) -> None:
        if self.active:
            _collect(prefix, value, self.tensors, self.notes)

    # ----- one-time installation once the trainer has built its runtime -----

    def install_runtime(self) -> None:
        trainer = self.trainer
        names: dict[int, str] = {}
        modules: list[tuple[str, Any]] = [("denoiser", trainer.denoiser), ("trainable_model", trainer.trainable_model)]
        for index, encoder in enumerate(trainer.text_encoders or []):
            modules.append((f"text_encoder_{index}", encoder))
        for label, module in modules:
            if isinstance(module, torch.nn.Module):
                for name, parameter in module.named_parameters():
                    names.setdefault(id(parameter), f"{label}.{name}")

        described = []
        for group_index, group in enumerate(trainer.optimizer.param_groups):
            for position, parameter in enumerate(group["params"]):
                self.parameters.append(parameter)
                described.append(
                    {
                        "name": names.get(id(parameter), f"group{group_index}.param{position}"),
                        "group": group_index,
                        "shape": list(parameter.shape),
                        "dtype": str(parameter.dtype),
                    }
                )
        (OUT / "parameters.json").write_text(json.dumps(described, indent=1))
        initial = {f"param_{key}": value for key, value in _summarize(list(self.parameters)).items()}
        save_file(initial, str(OUT / "initial.safetensors"))
        if FULL_INITIAL:
            full = {item["name"]: _cpu(parameter) for item, parameter in zip(described, self.parameters, strict=True)}
            if len(full) != len(described):
                raise ValueError("trainable tensor names are not unique; cannot write initial_full.safetensors")
            save_file(full, str(OUT / "initial_full.safetensors"))

        denoiser = trainer.denoiser
        if isinstance(denoiser, torch.nn.Module):
            denoiser.register_forward_pre_hook(self.on_model_input, with_kwargs=True)
            denoiser.register_forward_hook(self.on_model_output)

        original_apply = trainer.loss_modifier.apply

        def apply(*args: Any, **kwargs: Any) -> Any:
            output = original_apply(*args, **kwargs)
            self.put("final_loss", output.loss)
            return output

        trainer.loss_modifier.apply = apply

        original_step = trainer.optimizer.step

        def step(*args: Any, **kwargs: Any) -> Any:
            if self.active:
                gradients = _summarize([parameter.grad for parameter in self.parameters])
                self.tensors.update({f"grad_{key}": value for key, value in gradients.items()})
                self.notes["learning_rates"] = [group["lr"] for group in trainer.optimizer.param_groups]
            result = original_step(*args, **kwargs)
            if self.active:
                after = _summarize(list(self.parameters))
                self.tensors.update({f"param_{key}": value for key, value in after.items()})
                self.finish_step()
            return result

        trainer.optimizer.step = step
        self.installed = True

    # ----- observers -----

    def on_model_input(self, _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        call = self.model_calls
        for index, value in enumerate(args):
            self.put(f"model_call_{call}/arg{index}", value)
        for key, value in kwargs.items():
            self.put(f"model_call_{call}/{key}", value)

    def on_model_output(self, _module: Any, _args: Any, output: Any) -> None:
        self.put(f"model_call_{self.model_calls}/output", output)
        self.model_calls += 1

    def finish_step(self) -> None:
        self.notes["global_step"] = int(self.trainer.global_step)
        self.notes["model_calls"] = self.model_calls
        stem = OUT / f"step_{self.step:03d}"
        save_file(_drop_duplicates(self.tensors, self.notes), str(stem) + ".safetensors")
        Path(str(stem) + ".json").write_text(json.dumps(self.notes, indent=1, default=str))
        self.tensors, self.notes, self.model_calls = {}, {}, 0
        self.step += 1


def _install(trainer: Any, strategies: Any) -> None:
    recorder = Recorder(trainer)

    original_inputs = sdxl_diffusion.prepare_ddpm_training_inputs

    def prepare_inputs(*args: Any, **kwargs: Any) -> Any:
        noise, noisy_latents, timesteps = original_inputs(*args, **kwargs)
        if kwargs.get("is_train", True):
            recorder.put("noise", noise)
            recorder.put("noisy_latents", noisy_latents)
            recorder.put("timesteps", timesteps)
        return noise, noisy_latents, timesteps

    sdxl_diffusion.prepare_ddpm_training_inputs = prepare_inputs

    original_target = strategies.get_noise_pred_and_target

    def noise_pred_and_target(cfg: Any, accelerator: Any, objective_runtime: Any, latents: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_target(cfg, accelerator, objective_runtime, latents, *args, **kwargs)
        if kwargs.get("is_train", True):
            recorder.put("latents", latents)
            recorder.put("text_encoder_conds", args[1] if len(args) > 1 else None)
            recorder.put("noise_pred", result[0])
            recorder.put("target", result[1])
            recorder.put("weighting", result[3])
        return result

    strategies.get_noise_pred_and_target = noise_pred_and_target

    original_batch = strategies.process_batch

    def process_batch(batch: Any, *args: Any, **kwargs: Any) -> Any:
        training = kwargs.get("is_train", True)
        if training and not recorder.installed:
            recorder.install_runtime()
        if training:
            recorder.put("batch", batch)
        output = original_batch(batch, *args, **kwargs)
        if training:
            recorder.put("loss", output.loss)
            recorder.put("per_sample_loss", output.per_sample_loss)
            recorder.put("loss_timesteps", output.timesteps)
        return output

    strategies.process_batch = process_batch


class RecordingTrainer(pinned_train.Trainer):
    def __init__(self, cfg: Any, strategies: Any, mode: Any, **kwargs: Any) -> None:
        if int(cfg.training.gradient_accumulation_steps) != 1:
            raise ValueError("record_steps.py records one batch per optimizer step; set gradient_accumulation_steps=1")
        super().__init__(cfg, strategies, mode, **kwargs)
        _install(self, strategies)


def _write_environment() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gil = getattr(sys, "_is_gil_enabled", None)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "gil_enabled": gil() if gil is not None else True,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "argv": sys.argv[1:],
        "steps": STEPS,
    }
    (OUT / "environment.json").write_text(json.dumps(environment, indent=1))


@hydra.main(version_base=None, config_path=str(Path(os.getcwd()) / "configs"), config_name=None)
def main(cfg: Any) -> None:
    """Same entry as the pinned ``train.py``, with the recording Trainer."""
    pinned_train.train(cfg)


if __name__ == "__main__":
    from library.config.schemas import register_run

    _write_environment()
    pinned_train.Trainer = RecordingTrainer
    register_run()
    main()
