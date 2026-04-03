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
    wrapper_style: Literal["wrap_optimizer"] | None = None
    backend: Literal["torch", "bitsandbytes", "dadaptation", "prodigy", "transformers", "schedulefree", "repo"] | None = None
    aliases: tuple[str, ...] = ()
    capabilities: frozenset[str] = field(default_factory=frozenset)

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities


@dataclass(frozen=True, slots=True)
class SchedulerRegistration:
    """Repo-owned metadata for known scheduler integration shapes."""

    name: str
    target: str | None = None
    kind: Literal["transformers", "diffusers", "torch", "optimizer_embedded"] = "transformers"
    aliases: tuple[str, ...] = ()


def get_configured_optimizer_name(optimizer_config: OptimizerConfig) -> str:
    """Resolve the effective optimizer name from config compatibility flags."""
    optimizer_type = optimizer_config.optimizer_type
    if optimizer_config.use_8bit_adam:
        return "AdamW8bit"
    if optimizer_config.use_lion_optimizer:
        return "Lion"
    return optimizer_type or "AdamW"


def is_schedulefree_optimizer_name(name: str | None) -> bool:
    normalized_name = _normalize_name(name)
    return normalized_name.endswith("schedulefree")


def is_wrapper_optimizer_name(name: str | None) -> bool:
    normalized_name = _normalize_name(name)
    return normalized_name.endswith("schedulefreewrapper") or normalized_name.endswith("snoo_asgd")


_OPTIMIZER_REGISTRATIONS = [
    OptimizerRegistration(name="adamw", target="torch.optim.AdamW", backend="torch"),
    OptimizerRegistration(name="adabelief", target="library.optimization.optimizers.adabelief.AdaBelief", backend="repo"),
    OptimizerRegistration(name="adan", target="library.optimization.optimizers.adan.Adan", backend="repo"),
    OptimizerRegistration(
        name="adamw8bitkahan",
        target="library.optimization.optimizers.adamw_8bit_kahan.AdamW8bitKahan",
        backend="bitsandbytes",
    ),
    OptimizerRegistration(name="lion", target="lion_pytorch.Lion", backend="torch"),
    OptimizerRegistration(name="sgdnesterov", target="torch.optim.SGD", backend="torch"),
    OptimizerRegistration(name="adafactor", target="transformers.optimization.Adafactor", backend="transformers"),
    OptimizerRegistration(name="adamw8bit", target="bitsandbytes.optim.AdamW8bit", backend="bitsandbytes"),
    OptimizerRegistration(name="lion8bit", target="bitsandbytes.optim.Lion8bit", backend="bitsandbytes"),
    OptimizerRegistration(name="sgdnesterov8bit", target="bitsandbytes.optim.SGD8bit", backend="bitsandbytes"),
    OptimizerRegistration(name="pagedadamw", target="bitsandbytes.optim.PagedAdamW", backend="bitsandbytes"),
    OptimizerRegistration(name="pagedadamw8bit", target="bitsandbytes.optim.PagedAdamW8bit", backend="bitsandbytes"),
    OptimizerRegistration(name="pagedadamw32bit", target="bitsandbytes.optim.PagedAdamW32bit", backend="bitsandbytes"),
    OptimizerRegistration(name="pagedlion8bit", target="bitsandbytes.optim.PagedLion8bit", backend="bitsandbytes"),
    OptimizerRegistration(
        name="dadaptation",
        target="dadaptation.experimental.DAdaptAdamPreprint",
        backend="dadaptation",
        aliases=("dadaptadampreprint",),
    ),
    OptimizerRegistration(name="dadaptadagrad", target="dadaptation.DAdaptAdaGrad", backend="dadaptation"),
    OptimizerRegistration(name="dadaptadam", target="dadaptation.DAdaptAdam", backend="dadaptation"),
    OptimizerRegistration(name="dadaptadan", target="dadaptation.DAdaptAdan", backend="dadaptation"),
    OptimizerRegistration(name="dadaptadanip", target="dadaptation.experimental.DAdaptAdanIP", backend="dadaptation"),
    OptimizerRegistration(name="dadaptlion", target="dadaptation.DAdaptLion", backend="dadaptation"),
    OptimizerRegistration(name="dadaptsgd", target="dadaptation.DAdaptSGD", backend="dadaptation"),
    OptimizerRegistration(name="prodigy", target="prodigyopt.Prodigy", backend="prodigy"),
    OptimizerRegistration(
        name="radamschedulefree",
        target="schedulefree.RAdamScheduleFree",
        backend="schedulefree",
        capabilities=frozenset({OPT_CAP_TRAIN_EVAL_TOGGLE, OPT_CAP_NO_EXTERNAL_SCHEDULER}),
    ),
    OptimizerRegistration(
        name="adamwschedulefree",
        target="schedulefree.AdamWScheduleFree",
        backend="schedulefree",
        capabilities=frozenset({OPT_CAP_TRAIN_EVAL_TOGGLE, OPT_CAP_NO_EXTERNAL_SCHEDULER}),
    ),
    OptimizerRegistration(
        name="sgdschedulefree",
        target="schedulefree.SGDScheduleFree",
        backend="schedulefree",
        capabilities=frozenset({OPT_CAP_TRAIN_EVAL_TOGGLE, OPT_CAP_NO_EXTERNAL_SCHEDULER}),
    ),
    OptimizerRegistration(
        name="schedulefreewrapper",
        target="schedulefree.ScheduleFreeWrapper",
        kind="wrapper",
        wrapper_style="wrap_optimizer",
        backend="schedulefree",
        capabilities=frozenset({OPT_CAP_TRAIN_EVAL_TOGGLE, OPT_CAP_SCHEDULER_ON_BASE_OPTIMIZER}),
    ),
    OptimizerRegistration(
        name="snoo_asgd",
        target="library.optimization.wrappers.SNOOASGD",
        kind="wrapper",
        wrapper_style="wrap_optimizer",
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
    SchedulerRegistration(name="piecewise_constant", kind="diffusers"),
    SchedulerRegistration(
        name="cosineannealinglr",
        target="torch.optim.lr_scheduler.CosineAnnealingLR",
        kind="torch",
        aliases=("cosineannealinglr", "CosineAnnealingLR"),
    ),
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
