import ast
import importlib
import logging
from typing import Optional, Any

import torch
import transformers
from diffusers.optimization import SchedulerType as DiffusersSchedulerType, \
    TYPE_TO_SCHEDULER_FUNCTION as DIFFUSERS_TYPE_TO_SCHEDULER_FUNCTION
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import SchedulerType
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION

from library.config.dataclasses.optimizer import SchedulerConfig, OptimizerConfig
from library.config.dataclasses.training import TrainingConfig
from library.optimizers.optimizer_utils import parse_string_to_type
from library.utils.common_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# Modified version of get_scheduler() function from diffusers.optimizer.get_scheduler
# Add some checking and features to the original function.
def get_scheduler_fix(scheduler_config: SchedulerConfig, optimizer_config: OptimizerConfig, training_config: TrainingConfig, optimizer: Optimizer, num_processes: int):
    """
    Unified API to get any scheduler from its name.
    """
    optimizer_type = optimizer_config.optimizer_type
    # if schedulefree optimizer, return dummy scheduler
    if optimizer_type.lower().split(".")[0] not in {"LoraEasyCustomOptimizer".lower(),
                                                         "prodigyplus".lower()} and optimizer_type.lower().endswith("schedulefree".lower()):
        return get_dummy_scheduler(optimizer)

    # Need to apply scheduler to base_optimizer
    if optimizer_type.lower().endswith("schedulefreewrapper".lower()) or optimizer_type.lower().endswith("snoo_asgd".lower()):
        optimizer = optimizer.base_optimizer  # FIXME: UNRESOLVED ATTRIBUTE

    name = scheduler_config.lr_scheduler
    num_training_steps = training_config.max_train_steps * num_processes  # * args.gradient_accumulation_steps
    num_warmup_steps: Optional[int] = (
        int(scheduler_config.lr_warmup_steps * num_training_steps) if isinstance(scheduler_config.lr_warmup_steps,
                                                                     float) else scheduler_config.lr_warmup_steps
    )

    temp_lr_decay_steps = parse_string_to_type(
        scheduler_config.lr_decay_steps) if scheduler_config.lr_decay_steps is not None else scheduler_config.lr_decay_steps or 0

    num_decay_steps: Optional[int] = (
        int(temp_lr_decay_steps * num_training_steps) if isinstance(temp_lr_decay_steps, float) else temp_lr_decay_steps
    )

    # TODO add inputs to UI to support setting decay steps
    if name == SchedulerType.WARMUP_STABLE_DECAY and (num_decay_steps is None or num_decay_steps == 0):
        num_decay_steps = num_warmup_steps

    num_stable_steps = num_training_steps - num_warmup_steps - num_decay_steps
    num_cycles = scheduler_config.lr_scheduler_num_cycles
    power = scheduler_config.lr_scheduler_power
    timescale = scheduler_config.lr_scheduler_timescale
    min_lr_ratio = scheduler_config.lr_scheduler_min_lr_ratio

    lr_scheduler_kwargs = {}  # get custom lr_scheduler kwargs
    if scheduler_config.lr_scheduler_args is not None and len(scheduler_config.lr_scheduler_args) > 0:
        for arg in scheduler_config.lr_scheduler_args:
            key, value = arg.split("=")
            value = ast.literal_eval(value)
            lr_scheduler_kwargs[key] = value

    def wrap_check_needless_num_warmup_steps(return_vals):
        if num_warmup_steps is not None and num_warmup_steps != 0:
            raise ValueError(f"{name} does not require `num_warmup_steps`. Set None or 0.")
        return return_vals

    # using any lr_scheduler from other library
    if scheduler_config.lr_scheduler_type:
        lr_scheduler_type = scheduler_config.lr_scheduler_type
        logger.info(f"use {lr_scheduler_type} | {lr_scheduler_kwargs} as lr_scheduler")
        if "." not in lr_scheduler_type:  # default to use torch.optim
            lr_scheduler_module = torch.optim.lr_scheduler
        else:
            values = lr_scheduler_type.split(".")
            lr_scheduler_module = importlib.import_module(".".join(values[:-1]))
            lr_scheduler_type = values[-1]
        lr_scheduler_class = getattr(lr_scheduler_module, lr_scheduler_type)
        lr_scheduler = lr_scheduler_class(optimizer, **lr_scheduler_kwargs)
        return wrap_check_needless_num_warmup_steps(lr_scheduler)
    else:
        logger.info(f"use {name} | {lr_scheduler_kwargs} as lr_scheduler")

    if name.startswith("adafactor"):
        assert (
                type(optimizer) == transformers.optimization.Adafactor
        ), f"adafactor scheduler must be used with Adafactor optimizer / adafactor schedulerはAdafactorオプティマイザと同時に使ってください"
        initial_lr = float(name.split(":")[1])
        # logger.info(f"adafactor scheduler init lr {initial_lr}")
        return wrap_check_needless_num_warmup_steps(transformers.optimization.AdafactorSchedule(optimizer, initial_lr))

    if name == DiffusersSchedulerType.PIECEWISE_CONSTANT.value:
        name = DiffusersSchedulerType(name)
        schedule_func = DIFFUSERS_TYPE_TO_SCHEDULER_FUNCTION[name]
        return schedule_func(optimizer, **lr_scheduler_kwargs)  # step_rules and last_epoch are given as kwargs

    if name.lower() == 'CosineAnnealingLR'.lower():
        return wrap_check_needless_num_warmup_steps(CosineAnnealingLR(optimizer,
                                                                      T_max=num_training_steps,
                                                                      eta_min=lr_scheduler_kwargs.get("min_lr", 1e-8),
                                                                      last_epoch=lr_scheduler_kwargs.get("last_epoch",
                                                                                                         -1)))

    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    if name == SchedulerType.CONSTANT:
        return wrap_check_needless_num_warmup_steps(schedule_func(optimizer, **lr_scheduler_kwargs))

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **lr_scheduler_kwargs)

    if name == SchedulerType.INVERSE_SQRT:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, timescale=timescale, **lr_scheduler_kwargs)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    if name == SchedulerType.COSINE_WITH_RESTARTS:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            **lr_scheduler_kwargs,
        )

    if name == SchedulerType.POLYNOMIAL:
        return schedule_func(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, power=power,
            **lr_scheduler_kwargs
        )

    if name == SchedulerType.COSINE_WITH_MIN_LR:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles / 2,
            min_lr_rate=min_lr_ratio,
            **lr_scheduler_kwargs,
        )

    # these schedulers do not require `num_decay_steps`
    if name == SchedulerType.LINEAR or name == SchedulerType.COSINE:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            **lr_scheduler_kwargs,
        )

    # All other schedulers require `num_decay_steps`

    if num_decay_steps is None:
        raise ValueError(f"{name} requires `num_decay_steps`, please provide that argument.")
    if name == SchedulerType.WARMUP_STABLE_DECAY:
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


def get_dummy_scheduler(optimizer: Optimizer) -> Any:
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
