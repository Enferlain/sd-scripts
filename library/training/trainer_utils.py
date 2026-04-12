import logging
import math
import random
import time
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import TorchDynamoPlugin
from torch import nn

import library.performance.deepspeed_utils as deepspeed_utils

from library.config.dataclasses.performance import (
    PrecisionConfig,
    CompilationConfig,
    DistributedConfig,
    DeepSpeedConfig,
)
from library.config.dataclasses.output import LoggingConfig, SavingConfig
from library.config.dataclasses.training import TrainingConfig
from library.logging.step_logging import append_lr_to_logs_with_names
from library.optimization.types import OptimizationPlan
from library.utils.compile_env import prepare_windows_compiler_env_for_torch_compile


logger = logging.getLogger(__name__)


@dataclass
class AcceleratorConfig:
    """Computed accelerator setup plus any deferred side-effect instructions."""

    gradient_accumulation_steps: int
    mixed_precision: str | None
    log_with: str | None
    project_dir: str | None
    kwargs_handlers: list[Any]
    dynamo_plugin: TorchDynamoPlugin | None
    deepspeed_plugin: Any
    configure_wandb: bool = False
    wandb_api_key: str | None = None


def all_reduce_trainable(accelerator: Accelerator, trainable_model: nn.Module) -> None:
    """Synchronize gradients manually for the trainable model."""
    for param in trainable_model.parameters():
        if param.grad is not None:
            param.grad = accelerator.reduce(param.grad, reduction="mean")


def switch_rng_state(val_seed: int, accelerator: Accelerator) -> tuple[Any, Any, Any, Any]:
    """Store current RNG states and set the validation seed."""
    cpu_rng_state = torch.get_rng_state()
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()

    gpu_rng_state = None
    if accelerator.device.type == "cuda":
        gpu_rng_state = torch.cuda.get_rng_state()
    elif accelerator.device.type == "xpu":
        gpu_rng_state = torch.xpu.get_rng_state()

    random.seed(val_seed)
    np.random.seed(val_seed)
    torch.manual_seed(val_seed)
    if accelerator.device.type == "cuda":
        torch.cuda.manual_seed_all(val_seed)

    return (cpu_rng_state, gpu_rng_state, python_rng_state, numpy_rng_state)


def restore_rng_state(rng_states: tuple[Any, Any, Any, Any], accelerator: Accelerator) -> None:
    """Restore RNG states after validation."""
    cpu_rng_state, gpu_rng_state, python_rng_state, numpy_rng_state = rng_states

    torch.set_rng_state(cpu_rng_state)
    random.setstate(python_rng_state)
    np.random.set_state(numpy_rng_state)

    if gpu_rng_state is not None:
        if accelerator.device.type == "cuda":
            torch.cuda.set_rng_state(gpu_rng_state)
        elif accelerator.device.type == "xpu":
            torch.xpu.set_rng_state(gpu_rng_state)


def _iter_parameterized_leaf_modules(module: nn.Module):
    """Yield leaf modules that own parameters directly (no recursion)."""
    for child in module.modules():
        if any(child.children()):
            continue
        child_params = list(child.parameters(recurse=False))
        if child_params:
            yield child, child_params


def summarize_component_trainability(module: nn.Module) -> dict[str, int]:
    """Summarize module/parameter counts for a model component."""
    total_modules = 0
    trainable_modules = 0
    for _mod, mod_params in _iter_parameterized_leaf_modules(module):
        total_modules += 1
        if any(p.requires_grad for p in mod_params):
            trainable_modules += 1

    total_params = 0
    trainable_params = 0
    for param in module.parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count

    return {
        "modules_total": total_modules,
        "modules_trainable": trainable_modules,
        "params_total": total_params,
        "params_trainable": trainable_params,
    }


def log_training_diagnostics(
    accelerator: Accelerator,
    cfg: Any,
    mode: Any,
    strategies: Any,
    components: list[tuple[str, nn.Module]],
    optimizer: Any,
    optimizer_name: str,
    lr_descriptions: list[str] | None = None,
    optimization_plan: OptimizationPlan | None = None,
    aliases: list[tuple[str, str]] | None = None,
) -> None:
    """Emit a compact training diagnostics block.

    Prints context line, per-component layer/param stats, optimizer
    group summary, and a legend. Designed to be mode-agnostic — works
    identically for PEFT and fine-tune.
    """
    if not components:
        return

    # --- Context line ---
    mode_name = type(mode).__name__
    strategy_name = type(strategies).__name__
    accelerator.print("")
    accelerator.print(
        f"  mode={mode_name}  strategy={strategy_name}  "
        f"precision={cfg.performance.precision.mixed_precision}  "
        f"grad_ckpt={cfg.performance.memory.gradient_checkpointing}  "
        f"xformers={cfg.performance.attention.xformers}  "
        f"deepspeed={cfg.performance.deepspeed.deepspeed}"
    )

    effective_batch = cfg.training.train_batch_size * accelerator.num_processes * cfg.training.gradient_accumulation_steps
    accelerator.print(
        f"  batch: per_device={cfg.training.train_batch_size}  "
        f"grad_accum={cfg.training.gradient_accumulation_steps}  "
        f"effective={effective_batch}  max_steps={cfg.training.max_train_steps}"
    )

    # --- Per-component stats ---
    accelerator.print("")
    accelerator.print("  components:")

    agg_mods = 0
    agg_mods_train = 0
    agg_params = 0
    agg_params_train = 0

    for name, module in components:
        stats = summarize_component_trainability(module)
        agg_mods += stats["modules_total"]
        agg_mods_train += stats["modules_trainable"]
        agg_params += stats["params_total"]
        agg_params_train += stats["params_trainable"]

        # Format: right-align numbers for readability
        mt = stats["modules_trainable"]
        ml = stats["modules_total"]
        pt = stats["params_trainable"]
        pl = stats["params_total"]
        pct = f"{pt / pl * 100:.1f}%" if pl > 0 else "0.0%"
        status = "frozen" if pt == 0 else pct

        accelerator.print(f"    {name + ':':<20s} modules {mt:>5,}/{ml:>5,} trainable   params {pt:>13,}/{pl:>13,}  ({status})")

    # Aggregate
    agg_pct = f"{agg_params_train / agg_params * 100:.1f}%" if agg_params > 0 else "0.0%"
    accelerator.print(
        f"    {'total:':<20s} "
        f"modules {agg_mods_train:>5,}/{agg_mods:>5,} trainable   "
        f"params {agg_params_train:>13,}/{agg_params:>13,}  ({agg_pct})"
    )

    if aliases:
        alias_text = ", ".join(f"{alias} -> {target}" for alias, target in aliases)
        accelerator.print(f"    aliases: {alias_text}")

    # --- Optimizer groups ---
    accelerator.print("")
    accelerator.print(f"  optimizer: {optimizer_name}")
    if optimization_plan is not None and optimization_plan.logical_groups:
        runtime_groups = getattr(optimizer, "param_groups", [])
        for i, logical_group in enumerate(optimization_plan.logical_groups):
            group_lr = logical_group.lr
            if logical_group.execution_group_indices and logical_group.execution_group_indices[0] < len(runtime_groups):
                group_lr = runtime_groups[logical_group.execution_group_indices[0]].get("lr", group_lr)

            label = logical_group.metric_name if logical_group.metric_name else f"group {i}"
            accelerator.print(f"    {label}  params={logical_group.parameter_count:,}  lr={group_lr}")
    elif hasattr(optimizer, "param_groups"):
        lr_names = list(lr_descriptions or [])
        for i, group in enumerate(optimizer.param_groups):
            group_lr = group.get("lr", "?")
            param_count = sum(p.numel() for p in group["params"] if isinstance(p, torch.Tensor))
            # Try to label the group from lr_descriptions
            label = lr_names[i] if i < len(lr_names) else f"group {i}"
            accelerator.print(f"    {label}  params={param_count:,}  lr={group_lr}")

    accelerator.print("")


def prepare_accelerator(
    precision_config: PrecisionConfig,
    compilation_config: CompilationConfig,
    distributed_config: DistributedConfig,
    deepspeed_config: DeepSpeedConfig,
    logging_config: LoggingConfig | None = None,
    training_config: TrainingConfig | None = None,
):
    """
    Prepare accelerator with optional deepspeed plugin.

    Args:
        precision_config: Precision settings (mixed_precision).
        compilation_config: Torch compile settings.
        distributed_config: DDP settings (gradient_as_bucket_view, static_graph).
        deepspeed_config: DeepSpeed settings.
        logging_config: Optional logging settings (logging_dir, log_with, wandb settings).
        training_config: Optional training settings (gradient_accumulation_steps).

    Returns:
        Accelerator: The prepared accelerator object.
    """
    accelerator_config = compute_accelerator_config(
        precision_config,
        compilation_config,
        distributed_config,
        deepspeed_config,
        logging_config=logging_config,
        training_config=training_config,
    )

    if accelerator_config.configure_wandb:
        try:
            import wandb
        except ImportError:
            raise ImportError("No wandb") from None
        if accelerator_config.project_dir is not None:
            os.makedirs(accelerator_config.project_dir, exist_ok=True)
            os.environ["WANDB_DIR"] = accelerator_config.project_dir
        if accelerator_config.wandb_api_key is not None:
            wandb.login(key=accelerator_config.wandb_api_key)

    accelerator = Accelerator(
        gradient_accumulation_steps=accelerator_config.gradient_accumulation_steps,
        mixed_precision=accelerator_config.mixed_precision,
        log_with=accelerator_config.log_with,
        project_dir=accelerator_config.project_dir,
        kwargs_handlers=accelerator_config.kwargs_handlers,
        dynamo_plugin=accelerator_config.dynamo_plugin,
        deepspeed_plugin=accelerator_config.deepspeed_plugin,
    )
    return accelerator


def compute_accelerator_config(
    precision_config: PrecisionConfig,
    compilation_config: CompilationConfig,
    distributed_config: DistributedConfig,
    deepspeed_config: DeepSpeedConfig,
    logging_config: LoggingConfig | None = None,
    training_config: TrainingConfig | None = None,
) -> AcceleratorConfig:
    """Compute accelerator constructor arguments without filesystem/network side effects."""

    # Handle logging directory
    if logging_config is None or logging_config.logging_dir is None:
        logging_dir = None
    else:
        log_prefix = "" if logging_config.log_prefix is None else logging_config.log_prefix
        run_name = log_prefix + time.strftime("%Y%m%d%H%M%S", time.localtime())
        logging_dir = os.path.join(logging_config.logging_dir, run_name)

    # Handle log_with setting
    if logging_config is None or logging_config.log_with is None:
        if logging_dir is not None:
            log_with = "tensorboard"
        else:
            log_with = None
    else:
        log_with = logging_config.log_with
        if log_with in ["tensorboard", "all"] and logging_dir is None:
            raise ValueError("logging_dir is required when log_with is tensorboard")
    configure_wandb = log_with in ["wandb", "all"]

    # torch.compile options
    if compilation_config.torch_compile:
        prepare_windows_compiler_env_for_torch_compile()
        dynamo_plugin = TorchDynamoPlugin(
            backend="inductor",  # type: ignore[arg-type] - accelerate accepts str at runtime
            mode="default",
            fullgraph=False,
            dynamic=True,
            use_regional_compilation=True,
        )
    else:
        dynamo_plugin = None

    # DDP kwargs
    kwargs_handlers = [
        (
            DistributedDataParallelKwargs(
                gradient_as_bucket_view=distributed_config.ddp_gradient_as_bucket_view, static_graph=distributed_config.ddp_static_graph
            )
            if distributed_config.ddp_gradient_as_bucket_view or distributed_config.ddp_static_graph
            else None
        ),
    ]
    kwargs_handlers = [i for i in kwargs_handlers if i is not None]

    # Deepspeed plugin
    deepspeed_plugin = deepspeed_utils.prepare_deepspeed_plugin(deepspeed_config, precision_config, training_config)

    # Gradient accumulation steps
    gradient_accumulation_steps = training_config.gradient_accumulation_steps if training_config else 1

    return AcceleratorConfig(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=precision_config.mixed_precision,
        log_with=log_with,
        project_dir=logging_dir,
        kwargs_handlers=kwargs_handlers,
        dynamo_plugin=dynamo_plugin,
        deepspeed_plugin=deepspeed_plugin,
        configure_wandb=configure_wandb,
        wandb_api_key=logging_config.wandb_api_key if logging_config is not None else None,
    )


def append_lr_to_logs(logs, lr_scheduler, optimizer_type, including_denoiser=True):
    """
    Append learning rate to logs.

    Args:
        logs: The logs to append to.
        lr_scheduler: The learning rate scheduler.
        optimizer_type: The optimizer type.
        including_denoiser: Whether to include denoiser learning rate.
    """
    names = []
    if including_denoiser:
        names.append("denoiser")
    names.append("text_encoder1")
    names.append("text_encoder2")

    append_lr_to_logs_with_names(logs, lr_scheduler, optimizer_type, names)


def determine_grad_sync_context(precision_config: PrecisionConfig | None, accelerator, sync_gradients, training_model, auxiliary_model=None):
    """
    Determine the gradient synchronization context.

    Args:
        precision_config: Precision configuration (not currently used).
        accelerator: The accelerator object.
        sync_gradients: Whether to sync gradients (not currently used).
        training_model: The training model.
        auxiliary_model: Optional sidecar model that also participates in gradient accumulation.

    Returns:
        ContextManager: The gradient synchronization context.
    """
    # Note: Previously considered no_sync for full_bf16, but accumulate() handles this correctly
    if auxiliary_model is not None:
        return accelerator.accumulate(training_model, auxiliary_model)
    else:
        return accelerator.accumulate(training_model)


def calculate_initial_step(
    training_config: TrainingConfig,
    saving_config: SavingConfig,
    train_dataloader,
    accelerator,
    steps_from_state,
):
    """
    Calculate initial step and epoch for training start/resume.

    Handles:
    - initial_epoch/initial_step from config
    - steps_from_state from resume
    - skip_until_initial_step logic

    Args:
        training_config: Training settings.
        saving_config: Saving settings.
        train_dataloader: Training data loader
        accelerator: HuggingFace Accelerator
        steps_from_state: Steps loaded from saved state (or None)

    Returns:
        Tuple of (initial_step, epoch_to_start)
    """
    initial_step = 0
    if training_config.initial_epoch is not None or training_config.initial_step is not None:
        # if initial_epoch or initial_step is specified, steps_from_state is ignored even when resuming
        if steps_from_state is not None:
            logger.warning("steps from the state is ignored because initial_step is specified")
        if training_config.initial_step is not None:
            initial_step = training_config.initial_step
        else:
            # num steps per epoch is calculated by num_processes and gradient_accumulation_steps
            initial_step = (training_config.initial_epoch - 1) * math.ceil(
                len(train_dataloader) / accelerator.num_processes / training_config.gradient_accumulation_steps
            )
    else:
        # if initial_epoch and initial_step are not specified, steps_from_state is used when resuming
        if steps_from_state is not None:
            initial_step = steps_from_state

    if initial_step > 0:
        assert training_config.max_train_steps > initial_step, (
            f"max_train_steps should be greater than initial step: {training_config.max_train_steps} vs {initial_step}"
        )

    epoch_to_start = 0
    if initial_step > 0:
        if training_config.skip_until_initial_step:
            # if skip_until_initial_step is specified, load data and discard it to ensure the same data is used
            if not saving_config.resume:
                logger.info("initial_step is specified but not resuming. lr scheduler will be started from the beginning")
            logger.info(f"skipping {initial_step} steps")
            initial_step *= training_config.gradient_accumulation_steps

            # set epoch to start to make initial_step less than len(train_dataloader)
            epoch_to_start = initial_step // math.ceil(len(train_dataloader) / training_config.gradient_accumulation_steps)
        else:
            # if not, only epoch no is skipped for informative purpose
            epoch_to_start = initial_step // math.ceil(len(train_dataloader) / training_config.gradient_accumulation_steps)
            initial_step = 0  # do not skip

    return initial_step, epoch_to_start
