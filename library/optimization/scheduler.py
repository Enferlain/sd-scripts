import importlib
import logging
from typing import Any

import torch
import transformers
from diffusers.optimization import (
    SchedulerType as DiffusersSchedulerType,
    TYPE_TO_SCHEDULER_FUNCTION as DIFFUSERS_TYPE_TO_SCHEDULER_FUNCTION,
)
from torch.optim import Optimizer
from transformers import SchedulerType
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION

from library.config.dataclasses.optimizer import SchedulerConfig, OptimizerConfig
from library.config.dataclasses.training import TrainingConfig
from library.optimization.arguments import parse_key_value_args
from library.optimization.loading import load_target
from library.optimization.registry import (
    OPT_CAP_NO_EXTERNAL_SCHEDULER,
    OPT_CAP_SCHEDULER_ON_BASE_OPTIMIZER,
    SchedulerRegistration,
    get_configured_optimizer_name,
    get_optimizer_registration,
    get_scheduler_registration,
    is_schedulefree_optimizer_name,
    is_wrapper_optimizer_name,
)
from library.optimization.optimizer_utils import parse_string_to_type
from library.optimization.types import OptimizationPlan, SchedulerRuntimeMetadata


logger = logging.getLogger(__name__)


def resolve_scheduler_runtime_metadata(
    scheduler_config: SchedulerConfig,
    optimizer_config: OptimizerConfig,
    optimizer: Optimizer,
) -> SchedulerRuntimeMetadata:
    """Resolve scheduler ownership/target metadata for the current optimizer runtime."""
    optimizer_name = get_configured_optimizer_name(optimizer_config)
    optimizer_registration = get_optimizer_registration(optimizer_name)
    scheduler_registration = get_scheduler_registration(scheduler_config.lr_scheduler)

    if optimizer_registration is not None and optimizer_registration.supports(OPT_CAP_NO_EXTERNAL_SCHEDULER):
        return SchedulerRuntimeMetadata(mode="none", target="optimizer")

    if (
        optimizer_registration is None
        and optimizer_name.lower().split(".")[0] != "prodigyplus"
        and is_schedulefree_optimizer_name(optimizer_name)
    ):
        return SchedulerRuntimeMetadata(mode="none", target="optimizer")

    if scheduler_registration is not None and scheduler_registration.kind == "optimizer_embedded":
        return SchedulerRuntimeMetadata(mode="embedded", target="optimizer")

    if (optimizer_registration is not None and optimizer_registration.supports(OPT_CAP_SCHEDULER_ON_BASE_OPTIMIZER)) or (
        (optimizer_registration is None and is_wrapper_optimizer_name(optimizer_name)) or optimizer_config.optimizer_schedulefree_wrapper
    ):
        return SchedulerRuntimeMetadata(mode="external", target="base_optimizer")

    return SchedulerRuntimeMetadata(mode="external", target="optimizer")


def _wrap_requires_no_warmup(name: str, num_warmup_steps: int | None, scheduler):
    if num_warmup_steps is not None and num_warmup_steps != 0:
        raise ValueError(f"{name} does not require `num_warmup_steps`. Set None or 0.")
    return scheduler


def _build_custom_scheduler(lr_scheduler_type: str, optimizer: Optimizer, lr_scheduler_kwargs: dict[str, Any]):
    logger.info(f"use {lr_scheduler_type} | {lr_scheduler_kwargs} as lr_scheduler")
    if "." not in lr_scheduler_type:
        lr_scheduler_module = torch.optim.lr_scheduler
    else:
        values = lr_scheduler_type.split(".")
        lr_scheduler_module = importlib.import_module(".".join(values[:-1]))
        lr_scheduler_type = values[-1]
    lr_scheduler_class = getattr(lr_scheduler_module, lr_scheduler_type)
    return lr_scheduler_class(optimizer, **lr_scheduler_kwargs)


def _build_embedded_scheduler(name: str, optimizer: Optimizer):
    if not name.startswith("adafactor"):
        raise ValueError(f"Unsupported embedded scheduler registration: {name}")

    assert isinstance(optimizer, transformers.optimization.Adafactor), "adafactor scheduler must be used with Adafactor optimizer"
    initial_lr = float(name.split(":")[1])
    return transformers.optimization.AdafactorSchedule(optimizer, initial_lr)


def _build_diffusers_scheduler(
    registration: SchedulerRegistration,
    optimizer: Optimizer,
    lr_scheduler_kwargs: dict[str, Any],
):
    if registration.name != DiffusersSchedulerType.PIECEWISE_CONSTANT.value:
        raise ValueError(f"Unsupported diffusers scheduler registration: {registration.name}")

    schedule_name = DiffusersSchedulerType(registration.name)
    schedule_func = DIFFUSERS_TYPE_TO_SCHEDULER_FUNCTION[schedule_name]
    return schedule_func(optimizer, **lr_scheduler_kwargs)


def _build_torch_scheduler(
    registration: SchedulerRegistration,
    optimizer: Optimizer,
    lr_scheduler_kwargs: dict[str, Any],
    num_training_steps: int,
):
    if registration.target is None:
        raise ValueError(f"Missing target for torch scheduler registration: {registration.name}")

    scheduler_class = load_target(registration.target)
    if registration.name == "cosineannealinglr":
        return scheduler_class(
            optimizer,
            T_max=num_training_steps,
            eta_min=lr_scheduler_kwargs.get("min_lr", 1e-8),
            last_epoch=lr_scheduler_kwargs.get("last_epoch", -1),
        )

    logger.debug("Torch scheduler registration %s has no custom builder; calling target directly", registration.name)
    return scheduler_class(optimizer, **lr_scheduler_kwargs)


def _build_transformers_scheduler(
    registration: SchedulerRegistration,
    optimizer: Optimizer,
    lr_scheduler_kwargs: dict[str, Any],
    num_warmup_steps: int | None,
    num_training_steps: int,
    num_decay_steps: int | None,
    num_stable_steps: int,
    num_cycles: int,
    power: float,
    timescale: int | None,
    min_lr_ratio: float | None,
):
    schedule_name = SchedulerType(registration.name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[schedule_name]

    if schedule_name == SchedulerType.CONSTANT:
        return _wrap_requires_no_warmup(registration.name, num_warmup_steps, schedule_func(optimizer, **lr_scheduler_kwargs))

    if num_warmup_steps is None:
        raise ValueError(f"{schedule_name} requires `num_warmup_steps`, please provide that argument.")

    if schedule_name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **lr_scheduler_kwargs)

    if schedule_name == SchedulerType.INVERSE_SQRT:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, timescale=timescale, **lr_scheduler_kwargs)

    if schedule_name == SchedulerType.COSINE_WITH_RESTARTS:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            **lr_scheduler_kwargs,
        )

    if schedule_name == SchedulerType.POLYNOMIAL:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            power=power,
            **lr_scheduler_kwargs,
        )

    if schedule_name == SchedulerType.COSINE_WITH_MIN_LR:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles / 2,
            min_lr_rate=min_lr_ratio,
            **lr_scheduler_kwargs,
        )

    if schedule_name in {SchedulerType.LINEAR, SchedulerType.COSINE}:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            **lr_scheduler_kwargs,
        )

    if num_decay_steps is None:
        raise ValueError(f"{schedule_name} requires `num_decay_steps`, please provide that argument.")

    if schedule_name == SchedulerType.WARMUP_STABLE_DECAY:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_stable_steps=num_stable_steps,
            num_decay_steps=num_decay_steps,
            num_cycles=num_cycles / 2.0,
            min_lr_ratio=min_lr_ratio if min_lr_ratio is not None else 0.0,
            **lr_scheduler_kwargs,
        )

    return schedule_func(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_decay_steps=num_decay_steps,
        **lr_scheduler_kwargs,
    )


def _build_registered_scheduler(
    registration: SchedulerRegistration | None,
    optimizer: Optimizer,
    lr_scheduler_kwargs: dict[str, Any],
    num_warmup_steps: int | None,
    num_training_steps: int,
    num_decay_steps: int | None,
    num_stable_steps: int,
    num_cycles: int,
    power: float,
    timescale: int | None,
    min_lr_ratio: float | None,
):
    if registration is None:
        return None

    if registration.kind == "optimizer_embedded":
        return _wrap_requires_no_warmup(registration.name, num_warmup_steps, _build_embedded_scheduler(registration.name, optimizer))

    if registration.kind == "diffusers":
        return _wrap_requires_no_warmup(
            registration.name,
            num_warmup_steps,
            _build_diffusers_scheduler(registration, optimizer, lr_scheduler_kwargs),
        )

    if registration.kind == "torch":
        return _wrap_requires_no_warmup(
            registration.name,
            num_warmup_steps,
            _build_torch_scheduler(registration, optimizer, lr_scheduler_kwargs, num_training_steps),
        )

    if registration.kind == "transformers":
        return _build_transformers_scheduler(
            registration,
            optimizer,
            lr_scheduler_kwargs,
            num_warmup_steps,
            num_training_steps,
            num_decay_steps,
            num_stable_steps,
            num_cycles,
            power,
            timescale,
            min_lr_ratio,
        )

    raise ValueError(f"Unknown scheduler registration kind: {registration.kind}")


def _assert_ranger21_scheduler_compatibility(
    optimizer_name: str,
    optimizer_config: OptimizerConfig,
    scheduler_config: SchedulerConfig,
    num_warmup_steps: int | None,
):
    if optimizer_name.lower() != "ranger21":
        return

    optimizer_kwargs = parse_key_value_args(optimizer_config.optimizer_args)
    if optimizer_kwargs.get("disable_lr_scheduler", False):
        return

    scheduler_name = (scheduler_config.lr_scheduler or "constant").strip().lower()
    has_external_scheduler = scheduler_name != "constant" or scheduler_config.lr_scheduler_type or num_warmup_steps not in {None, 0}
    if not has_external_scheduler:
        return

    raise ValueError(
        "Ranger21 manages learning-rate scheduling internally unless `disable_lr_scheduler=True`. "
        "Use `lr_scheduler='constant'` with no warmup, or disable the internal scheduler before configuring an external scheduler."
    )


# Modified version of get_scheduler() function from diffusers.optimizer.get_scheduler
# Add some checking and features to the original function.
def get_scheduler_fix(
    scheduler_config: SchedulerConfig,
    optimizer_config: OptimizerConfig,
    training_config: TrainingConfig,
    optimizer: Optimizer,
    num_processes: int,
    optimization_plan: OptimizationPlan | None = None,
):
    """
    Unified API to get any scheduler from its name.

    Args:
        scheduler_config (SchedulerConfig): Configuration for the scheduler.
        optimizer_config (OptimizerConfig): Configuration for the optimizer.
        training_config (TrainingConfig): Configuration for training settings.
        optimizer (Optimizer): The optimizer instance.
        num_processes (int): The number of processes (GPUs) used for training.

    Returns:
        The configured scheduler.
    """
    scheduler_runtime = optimization_plan.scheduler_runtime if optimization_plan is not None else None
    if scheduler_runtime is None:
        scheduler_runtime = resolve_scheduler_runtime_metadata(scheduler_config, optimizer_config, optimizer)

    optimizer_name = get_configured_optimizer_name(optimizer_config)

    if scheduler_runtime.mode == "none":
        return get_dummy_scheduler(optimizer)

    if scheduler_runtime.target == "base_optimizer":
        optimizer = getattr(optimizer, "base_optimizer", optimizer)  # Get wrapped base optimizer

    name = scheduler_config.lr_scheduler
    scheduler_registration = get_scheduler_registration(name)
    num_training_steps = training_config.max_train_steps * num_processes  # * args.gradient_accumulation_steps
    num_warmup_steps: int | None = (
        int(scheduler_config.lr_warmup_steps * num_training_steps)
        if isinstance(scheduler_config.lr_warmup_steps, float)
        else scheduler_config.lr_warmup_steps
    )
    _assert_ranger21_scheduler_compatibility(
        optimizer_name,
        optimizer_config,
        scheduler_config,
        num_warmup_steps,
    )

    temp_lr_decay_steps = (
        parse_string_to_type(scheduler_config.lr_decay_steps)
        if scheduler_config.lr_decay_steps is not None
        else scheduler_config.lr_decay_steps or 0
    )

    num_decay_steps: int | None = (
        int(temp_lr_decay_steps * num_training_steps) if isinstance(temp_lr_decay_steps, float) else temp_lr_decay_steps
    )

    if name == SchedulerType.WARMUP_STABLE_DECAY and (num_decay_steps is None or num_decay_steps == 0):
        num_decay_steps = num_warmup_steps

    num_stable_steps = num_training_steps - num_warmup_steps - num_decay_steps
    num_cycles = scheduler_config.lr_scheduler_num_cycles
    power = scheduler_config.lr_scheduler_power
    timescale = scheduler_config.lr_scheduler_timescale
    min_lr_ratio = scheduler_config.lr_scheduler_min_lr_ratio

    lr_scheduler_kwargs = parse_key_value_args(scheduler_config.lr_scheduler_args)

    if scheduler_config.lr_scheduler_type:
        return _wrap_requires_no_warmup(
            scheduler_config.lr_scheduler_type,
            num_warmup_steps,
            _build_custom_scheduler(scheduler_config.lr_scheduler_type, optimizer, lr_scheduler_kwargs),
        )

    display_name = scheduler_registration.name if scheduler_registration is not None else name
    logger.info(f"use {display_name} | {lr_scheduler_kwargs} as lr_scheduler")

    registered_scheduler = _build_registered_scheduler(
        scheduler_registration,
        optimizer,
        lr_scheduler_kwargs,
        num_warmup_steps,
        num_training_steps,
        num_decay_steps,
        num_stable_steps,
        num_cycles,
        power,
        timescale,
        min_lr_ratio,
    )
    if registered_scheduler is not None:
        return registered_scheduler

    logger.warning("Scheduler %s is not registered; falling back to legacy transformers dispatch", name)
    fallback_registration = SchedulerRegistration(name=name)
    return _build_transformers_scheduler(
        fallback_registration,
        optimizer,
        lr_scheduler_kwargs,
        num_warmup_steps,
        num_training_steps,
        num_decay_steps,
        num_stable_steps,
        num_cycles,
        power,
        timescale,
        min_lr_ratio,
    )


def get_dummy_scheduler(optimizer: Optimizer) -> Any:
    """
    Creates a dummy scheduler for schedule-free optimizers.

    This scheduler supports only empty step() and get_last_lr() methods, and is used mainly for logging purposes.
    It is not intended to be wrapped by accelerator as it is not a subclass of torch.optim.lr_scheduler._LRScheduler.

    Args:
        optimizer (Optimizer): The optimizer instance.

    Returns:
        Any: A dummy scheduler instance.
    """

    # dummy scheduler for schedulefree optimizer. supports only empty step(), get_last_lr() and optimizers.
    # this scheduler is used for logging only.
    # this isn't to be wrapped by accelerator because this class is not a subclass of torch.optim.lr_scheduler._LRScheduler
    class DummyScheduler:
        def __init__(self, optimizer: Optimizer):
            self.optimizer = optimizer

        def step(self):
            pass

        def get_last_lr(self):
            return [group["lr"] for group in self.optimizer.param_groups]

    return DummyScheduler(optimizer)
