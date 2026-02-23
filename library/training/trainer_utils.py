import logging
import math
import time
import os
from typing import Any

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
from library.config.dataclasses.output import LoggingConfig
from library.config.dataclasses.training import TrainingConfig
from library.config.dataclasses.validation import ValidationConfig
from library.logging.step_logging import append_lr_to_logs_with_names
from library.utils.common_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


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
    lr_descriptions: list[str],
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
    precision = getattr(cfg.performance.precision, "mixed_precision", "fp32")
    grad_ckpt = getattr(cfg.performance.memory, "gradient_checkpointing", False)
    xformers = getattr(cfg.performance.attention, "xformers", False)
    deepspeed = getattr(getattr(cfg.performance, "deepspeed", None), "deepspeed", False)

    accelerator.print("")
    accelerator.print(
        f"  mode={mode_name}  strategy={strategy_name}  "
        f"precision={precision}  grad_ckpt={grad_ckpt}  xformers={xformers}  deepspeed={deepspeed}"
    )

    batch_size = cfg.training.train_batch_size
    grad_accum = cfg.training.gradient_accumulation_steps
    effective_batch = batch_size * accelerator.num_processes * grad_accum
    accelerator.print(
        f"  batch: per_device={batch_size}  grad_accum={grad_accum}  effective={effective_batch}  max_steps={cfg.training.max_train_steps}"
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
    if hasattr(optimizer, "param_groups"):
        for i, group in enumerate(optimizer.param_groups):
            group_lr = group.get("lr", "?")
            param_count = sum(p.numel() for p in group["params"] if isinstance(p, torch.Tensor))
            # Try to label the group from lr_descriptions
            label = lr_descriptions[i] if i < len(lr_descriptions) else f"group {i}"
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

    # Handle logging directory
    if logging_config is None or logging_config.logging_dir is None:
        logging_dir = None
    else:
        log_prefix = "" if logging_config.log_prefix is None else logging_config.log_prefix
        logging_dir = logging_config.logging_dir + "/" + log_prefix + time.strftime("%Y%m%d%H%M%S", time.localtime())

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
        if log_with in ["wandb", "all"]:
            try:
                import wandb
            except ImportError:
                raise ImportError("No wandb") from None
            if logging_dir is not None:
                os.makedirs(logging_dir, exist_ok=True)
                os.environ["WANDB_DIR"] = logging_dir
            if logging_config.wandb_api_key is not None:
                wandb.login(key=logging_config.wandb_api_key)

    # torch.compile options
    if compilation_config.torch_compile:
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

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=precision_config.mixed_precision,
        log_with=log_with,
        project_dir=logging_dir,
        kwargs_handlers=kwargs_handlers,
        dynamo_plugin=dynamo_plugin,
        deepspeed_plugin=deepspeed_plugin,
    )
    return accelerator


def calculate_val_loss_check(
    validation_config: ValidationConfig,
    training_config: TrainingConfig,
    global_step: int,
    epoch_step: int,
    val_dataloader,
    train_dataloader_or_num_batches: int | object,
) -> bool:
    """Check if validation should be run at this step.

    Args:
        validation_config: ValidationConfig object with validation settings.
        training_config: TrainingConfig object for max_train_steps.
        global_step: Current global training step.
        epoch_step: Current step within the epoch.
        val_dataloader: Validation dataloader (if None, returns False).
        train_dataloader_or_num_batches: Training dataloader or int (num_batches_per_epoch).

    Returns:
        bool: True if validation should be run, False otherwise.
    """
    if val_dataloader is None:
        return False

    # Support both dataloader (len()) and int (direct value)
    if isinstance(train_dataloader_or_num_batches, int):
        num_batches = train_dataloader_or_num_batches
    else:
        num_batches = len(train_dataloader_or_num_batches)  # TODO: Expected type 'Sized', got 'object' instead

    if global_step != 0 and global_step < training_config.max_train_steps:
        if validation_config.validate_every_n_steps is not None:
            if global_step % int(validation_config.validate_every_n_steps) != 0:
                return False
        else:
            if epoch_step != num_batches - 1:
                return False
    return True


def append_lr_to_logs(logs, lr_scheduler, optimizer_type, including_unet=True):
    """
    Append learning rate to logs.

    Args:
        logs: The logs to append to.
        lr_scheduler: The learning rate scheduler.
        optimizer_type: The optimizer type.
        including_unet: Whether to include UNet learning rate.
    """
    names = []
    if including_unet:
        names.append("unet")
    names.append("text_encoder1")
    names.append("text_encoder2")

    append_lr_to_logs_with_names(logs, lr_scheduler, optimizer_type, names)


def determine_grad_sync_context(precision_config: PrecisionConfig | None, accelerator, sync_gradients, training_model, edm2_model=None):
    """
    Determine the gradient synchronization context.

    Args:
        precision_config: Precision configuration (not currently used).
        accelerator: The accelerator object.
        sync_gradients: Whether to sync gradients (not currently used).
        training_model: The training model.
        edm2_model: Optional EDM2 model.

    Returns:
        ContextManager: The gradient synchronization context.
    """
    # Note: Previously considered no_sync for full_bf16, but accumulate() handles this correctly
    if edm2_model is not None:
        return accelerator.accumulate(training_model, edm2_model)
    else:
        return accelerator.accumulate(training_model)


def calculate_initial_step(cfg, train_dataloader, accelerator, steps_from_state):
    """
    Calculate initial step and epoch for training start/resume.

    Handles:
    - initial_epoch/initial_step from config
    - steps_from_state from resume
    - skip_until_initial_step logic

    Args:
        cfg: Training configuration
        train_dataloader: Training data loader
        accelerator: HuggingFace Accelerator
        steps_from_state: Steps loaded from saved state (or None)

    Returns:
        Tuple of (initial_step, epoch_to_start)
    """
    initial_step = 0
    if cfg.training.initial_epoch is not None or cfg.training.initial_step is not None:
        # if initial_epoch or initial_step is specified, steps_from_state is ignored even when resuming
        if steps_from_state is not None:
            logger.warning("steps from the state is ignored because initial_step is specified")
        if cfg.training.initial_step is not None:
            initial_step = cfg.training.initial_step
        else:
            # num steps per epoch is calculated by num_processes and gradient_accumulation_steps
            initial_step = (cfg.training.initial_epoch - 1) * math.ceil(
                len(train_dataloader) / accelerator.num_processes / cfg.training.gradient_accumulation_steps
            )
    else:
        # if initial_epoch and initial_step are not specified, steps_from_state is used when resuming
        if steps_from_state is not None:
            initial_step = steps_from_state

    if initial_step > 0:
        assert cfg.training.max_train_steps > initial_step, (
            f"max_train_steps should be greater than initial step: {cfg.training.max_train_steps} vs {initial_step}"
        )

    epoch_to_start = 0
    if initial_step > 0:
        if cfg.training.skip_until_initial_step:
            # if skip_until_initial_step is specified, load data and discard it to ensure the same data is used
            if not cfg.output.saving.resume:
                logger.info("initial_step is specified but not resuming. lr scheduler will be started from the beginning")
            logger.info(f"skipping {initial_step} steps")
            initial_step *= cfg.training.gradient_accumulation_steps

            # set epoch to start to make initial_step less than len(train_dataloader)
            epoch_to_start = initial_step // math.ceil(len(train_dataloader) / cfg.training.gradient_accumulation_steps)
        else:
            # if not, only epoch no is skipped for informative purpose
            epoch_to_start = initial_step // math.ceil(len(train_dataloader) / cfg.training.gradient_accumulation_steps)
            initial_step = 0  # do not skip

    return initial_step, epoch_to_start
