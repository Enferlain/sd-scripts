from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from library.config.dataclasses.optimizer import OptimizerConfig


OPT_CAP_TRAIN_EVAL_TOGGLE = "train_eval_toggle"
OPT_CAP_NO_EXTERNAL_SCHEDULER = "no_external_scheduler"
OPT_CAP_SCHEDULER_ON_BASE_OPTIMIZER = "scheduler_on_base_optimizer"


def _normalize_name(name: str | None) -> str:
    return (name or "").strip().lower()


@dataclass(frozen=True, slots=True)
class OptimizerRegistration:
    """Repo-owned metadata for known optimizer integration shapes."""

    name: str
    target: str | None = None
    kind: Literal["optimizer", "wrapper", "offload_wrapper"] = "optimizer"
    aliases: tuple[str, ...] = ()
    capabilities: frozenset[str] = field(default_factory=frozenset)

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities


@dataclass(frozen=True, slots=True)
class SchedulerRegistration:
    """Repo-owned metadata for known scheduler integration shapes."""

    name: str
    kind: Literal["scheduler", "no_op_scheduler", "optimizer_embedded"] = "scheduler"
    aliases: tuple[str, ...] = ()


def get_configured_optimizer_name(optimizer_config: OptimizerConfig) -> str:
    """Resolve the effective optimizer name from config compatibility flags."""
    optimizer_type = optimizer_config.optimizer_type
    if optimizer_config.use_8bit_adam:
        return "AdamW8bit"
    if optimizer_config.use_lion_optimizer:
        return "Lion"
    return optimizer_type or "AdamW"


_OPTIMIZER_REGISTRATIONS = [
    OptimizerRegistration(name="adamw", target="torch.optim.AdamW"),
    OptimizerRegistration(name="lion", target="lion_pytorch.Lion"),
    OptimizerRegistration(name="sgdnesterov", target="torch.optim.SGD"),
    OptimizerRegistration(name="adafactor"),
    OptimizerRegistration(name="adamw8bit"),
    OptimizerRegistration(name="lion8bit"),
    OptimizerRegistration(name="sgdnesterov8bit"),
    OptimizerRegistration(name="pagedadamw"),
    OptimizerRegistration(name="pagedadamw8bit"),
    OptimizerRegistration(name="pagedadamw32bit"),
    OptimizerRegistration(name="pagedlion8bit"),
    OptimizerRegistration(name="dadaptation", aliases=("dadaptadampreprint",)),
    OptimizerRegistration(name="dadaptadagrad"),
    OptimizerRegistration(name="dadaptadam"),
    OptimizerRegistration(name="dadaptadan"),
    OptimizerRegistration(name="dadaptadanip"),
    OptimizerRegistration(name="dadaptlion"),
    OptimizerRegistration(name="dadaptsgd"),
    OptimizerRegistration(name="prodigy"),
    OptimizerRegistration(
        name="radamschedulefree",
        target="schedulefree.RAdamScheduleFree",
        capabilities=frozenset({OPT_CAP_TRAIN_EVAL_TOGGLE, OPT_CAP_NO_EXTERNAL_SCHEDULER}),
    ),
    OptimizerRegistration(
        name="adamwschedulefree",
        target="schedulefree.AdamWScheduleFree",
        capabilities=frozenset({OPT_CAP_TRAIN_EVAL_TOGGLE, OPT_CAP_NO_EXTERNAL_SCHEDULER}),
    ),
    OptimizerRegistration(
        name="sgdschedulefree",
        target="schedulefree.SGDScheduleFree",
        capabilities=frozenset({OPT_CAP_TRAIN_EVAL_TOGGLE, OPT_CAP_NO_EXTERNAL_SCHEDULER}),
    ),
    OptimizerRegistration(
        name="schedulefreewrapper",
        kind="wrapper",
        capabilities=frozenset({OPT_CAP_TRAIN_EVAL_TOGGLE, OPT_CAP_SCHEDULER_ON_BASE_OPTIMIZER}),
    ),
    OptimizerRegistration(
        name="snoo_asgd",
        kind="wrapper",
        capabilities=frozenset({OPT_CAP_SCHEDULER_ON_BASE_OPTIMIZER}),
    ),
]

_SCHEDULER_REGISTRATIONS = [
    SchedulerRegistration(name="constant"),
    SchedulerRegistration(name="constant_with_warmup"),
    SchedulerRegistration(name="inverse_sqrt"),
    SchedulerRegistration(name="cosine_with_restarts"),
    SchedulerRegistration(name="polynomial"),
    SchedulerRegistration(name="cosine_with_min_lr"),
    SchedulerRegistration(name="linear"),
    SchedulerRegistration(name="cosine"),
    SchedulerRegistration(name="warmup_stable_decay"),
    SchedulerRegistration(name="piecewise_constant"),
    SchedulerRegistration(name="cosineannealinglr", aliases=("cosineannealinglr", "CosineAnnealingLR")),
    SchedulerRegistration(name="adafactor", kind="optimizer_embedded"),
]

_OPTIMIZER_BY_NAME: dict[str, OptimizerRegistration] = {}
for registration in _OPTIMIZER_REGISTRATIONS:
    for key in (registration.name, *registration.aliases):
        _OPTIMIZER_BY_NAME[_normalize_name(key)] = registration

_SCHEDULER_BY_NAME: dict[str, SchedulerRegistration] = {}
for registration in _SCHEDULER_REGISTRATIONS:
    for key in (registration.name, *registration.aliases):
        _SCHEDULER_BY_NAME[_normalize_name(key)] = registration


def get_optimizer_registration(name: str | None) -> OptimizerRegistration | None:
    """Return repo metadata for a known optimizer name, if any."""
    return _OPTIMIZER_BY_NAME.get(_normalize_name(name))


def get_scheduler_registration(name: str | None) -> SchedulerRegistration | None:
    """Return repo metadata for a known scheduler name, if any."""
    return _SCHEDULER_BY_NAME.get(_normalize_name(name))
