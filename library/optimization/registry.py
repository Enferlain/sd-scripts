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
    wrapper_style: Literal["wrap_optimizer", "wrap_optimizer_with_base_kwargs"] | None = None
    backend: Literal["torch", "bitsandbytes", "torchao", "dadaptation", "prodigy", "transformers", "schedulefree", "repo"] | None = None
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
    OptimizerRegistration(name="adopt", target="library.optimization.optimizers.adopt.ADOPT", backend="repo"),
    OptimizerRegistration(
        name="adoptemamixschedulefree",
        target="library.optimization.optimizers.adopt.ADOPTEMAMixScheduleFree",
        backend="repo",
        capabilities=frozenset({OPT_CAP_TRAIN_EVAL_TOGGLE, OPT_CAP_NO_EXTERNAL_SCHEDULER}),
    ),
    OptimizerRegistration(name="adoptmars", target="library.optimization.optimizers.adopt.ADOPTMARS", backend="repo"),
    OptimizerRegistration(
        name="adoptmarsschedulefree",
        target="library.optimization.optimizers.adopt.ADOPTMARSScheduleFree",
        backend="repo",
        capabilities=frozenset({OPT_CAP_TRAIN_EVAL_TOGGLE, OPT_CAP_NO_EXTERNAL_SCHEDULER}),
    ),
    OptimizerRegistration(
        name="adoptnesterovschedulefree",
        target="library.optimization.optimizers.adopt.ADOPTNesterovScheduleFree",
        backend="repo",
        capabilities=frozenset({OPT_CAP_TRAIN_EVAL_TOGGLE, OPT_CAP_NO_EXTERNAL_SCHEDULER}),
    ),
    OptimizerRegistration(
        name="adoptschedulefree",
        target="library.optimization.optimizers.adopt.ADOPTScheduleFree",
        backend="repo",
        capabilities=frozenset({OPT_CAP_TRAIN_EVAL_TOGGLE, OPT_CAP_NO_EXTERNAL_SCHEDULER}),
        aliases=("ADOPTScheduleFree",),
    ),
    OptimizerRegistration(
        name="adoptaoschedulefree",
        target="library.optimization.optimizers.adopt.ADOPTAOScheduleFree",
        backend="torchao",
        capabilities=frozenset({OPT_CAP_TRAIN_EVAL_TOGGLE, OPT_CAP_NO_EXTERNAL_SCHEDULER}),
        aliases=("ADOPTAOScheduleFree",),
    ),
    OptimizerRegistration(name="abmog", target="library.optimization.optimizers.abmog.ABMOG", backend="repo"),
    OptimizerRegistration(
        name="fadoptemamixschedulefree",
        target="library.optimization.optimizers.adopt.FADOPTEMAMixScheduleFree",
        backend="repo",
        capabilities=frozenset({OPT_CAP_TRAIN_EVAL_TOGGLE, OPT_CAP_NO_EXTERNAL_SCHEDULER}),
    ),
    OptimizerRegistration(
        name="fadoptmarsschedulefree",
        target="library.optimization.optimizers.adopt.FADOPTMARSScheduleFree",
        backend="repo",
        capabilities=frozenset({OPT_CAP_TRAIN_EVAL_TOGGLE, OPT_CAP_NO_EXTERNAL_SCHEDULER}),
    ),
    OptimizerRegistration(
        name="fadoptnesterovschedulefree",
        target="library.optimization.optimizers.adopt.FADOPTNesterovScheduleFree",
        backend="repo",
        capabilities=frozenset({OPT_CAP_TRAIN_EVAL_TOGGLE, OPT_CAP_NO_EXTERNAL_SCHEDULER}),
    ),
    OptimizerRegistration(
        name="fadoptschedulefree",
        target="library.optimization.optimizers.adopt.FADOPTScheduleFree",
        backend="repo",
        capabilities=frozenset({OPT_CAP_TRAIN_EVAL_TOGGLE, OPT_CAP_NO_EXTERNAL_SCHEDULER}),
    ),
    OptimizerRegistration(name="fadoptmars", target="library.optimization.optimizers.adopt.FADOPTMARS", backend="repo"),
    OptimizerRegistration(name="ademamix", target="library.optimization.optimizers.ademamix.AdEMAMix", backend="repo"),
    OptimizerRegistration(name="adabelief", target="library.optimization.optimizers.adabelief.AdaBelief", backend="repo"),
    OptimizerRegistration(name="adai", target="library.optimization.optimizers.adai.Adai", backend="repo"),
    OptimizerRegistration(name="adan", target="library.optimization.optimizers.adan.Adan", backend="repo"),
    OptimizerRegistration(name="adammini", target="library.optimization.optimizers.adammini.AdamMini", backend="repo"),
    OptimizerRegistration(name="adamw4bitao", target="library.optimization.optimizers.adamw.AdamW4bitAO", backend="torchao"),
    OptimizerRegistration(name="adamw8bitao", target="library.optimization.optimizers.adamw.AdamW8bitAO", backend="torchao"),
    OptimizerRegistration(name="alice", target="library.optimization.optimizers.alice.Alice", backend="repo"),
    OptimizerRegistration(name="adamwfp8ao", target="library.optimization.optimizers.adamw.AdamWfp8AO", backend="torchao"),
    OptimizerRegistration(name="bcos", target="library.optimization.optimizers.bcos.BCOS", backend="repo"),
    OptimizerRegistration(name="came", target="library.optimization.optimizers.came.CAME", backend="repo"),
    OptimizerRegistration(name="cstableadamw", target="library.optimization.optimizers.cstableadamw.CStableAdamW", backend="repo"),
    OptimizerRegistration(name="compass", target="library.optimization.optimizers.compass.Compass", backend="repo"),
    OptimizerRegistration(
        name="compass8bitbnb",
        target="library.optimization.optimizers.compass.Compass8BitBNB",
        backend="bitsandbytes",
    ),
    OptimizerRegistration(name="compassadopt", target="library.optimization.optimizers.compass.CompassADOPT", backend="repo"),
    OptimizerRegistration(name="compassadoptmars", target="library.optimization.optimizers.compass.CompassADOPTMARS", backend="repo"),
    OptimizerRegistration(name="compassao", target="library.optimization.optimizers.compass.CompassAO", backend="torchao"),
    OptimizerRegistration(name="compassplus", target="library.optimization.optimizers.compass.CompassPlus", backend="repo"),
    OptimizerRegistration(name="dehaze", target="library.optimization.optimizers.dehaze.Dehaze", backend="repo"),
    OptimizerRegistration(name="fftdescent", target="library.optimization.optimizers.fftdescent.FFTDescent", backend="repo"),
    OptimizerRegistration(name="fcompass", target="library.optimization.optimizers.compass.FCompass", backend="repo"),
    OptimizerRegistration(name="fcompassadopt", target="library.optimization.optimizers.compass.FCompassADOPT", backend="repo"),
    OptimizerRegistration(name="fcompassadoptmars", target="library.optimization.optimizers.compass.FCompassADOPTMARS", backend="repo"),
    OptimizerRegistration(name="fcompassplus", target="library.optimization.optimizers.compass.FCompassPlus", backend="repo"),
    OptimizerRegistration(name="farmscrop", target="library.optimization.optimizers.farmscrop.FARMSCrop", backend="repo"),
    OptimizerRegistration(name="farmscropv2", target="library.optimization.optimizers.farmscrop.FARMSCropV2", backend="repo"),
    OptimizerRegistration(name="fishmonger", target="library.optimization.optimizers.fishmonger.FishMonger", backend="repo"),
    OptimizerRegistration(
        name="fishmonger8bitbnb",
        target="library.optimization.optimizers.fishmonger.FishMonger8BitBNB",
        backend="bitsandbytes",
    ),
    OptimizerRegistration(name="fira", target="library.optimization.optimizers.fira.Fira", backend="repo"),
    OptimizerRegistration(name="fmarscrop", target="library.optimization.optimizers.fmarscrop.FMARSCrop", backend="repo"),
    OptimizerRegistration(name="fmarscropv2", target="library.optimization.optimizers.fmarscrop.FMARSCropV2", backend="repo"),
    OptimizerRegistration(
        name="fmarscropv2exmachina",
        target="library.optimization.optimizers.fmarscrop.FMARSCropV2ExMachina",
        backend="repo",
    ),
    OptimizerRegistration(name="fmarscropv3", target="library.optimization.optimizers.fmarscrop.FMARSCropV3", backend="repo"),
    OptimizerRegistration(
        name="fmarscropv3exmachina",
        target="library.optimization.optimizers.fmarscrop.FMARSCropV3ExMachina",
        backend="repo",
    ),
    OptimizerRegistration(name="galore", target="library.optimization.optimizers.galore.GaLore", backend="repo"),
    OptimizerRegistration(name="glyph", target="library.optimization.optimizers.glyph.Glyph", backend="repo"),
    OptimizerRegistration(name="gooddog", target="library.optimization.optimizers.gooddog.GOODDOG", backend="repo"),
    OptimizerRegistration(name="grokfastadamw", target="library.optimization.optimizers.grokfast.GrokFastAdamW", backend="repo"),
    OptimizerRegistration(name="laprop", target="library.optimization.optimizers.laprop.LaProp", backend="repo"),
    OptimizerRegistration(name="lamb", target="library.optimization.optimizers.lamb.Lamb", backend="repo"),
    OptimizerRegistration(name="lpfadamw", target="library.optimization.optimizers.lpf_adamw.LPFAdamW", backend="repo"),
    OptimizerRegistration(name="mythical", target="library.optimization.optimizers.mythical.Mythical", backend="repo"),
    OptimizerRegistration(name="momentuscaution", target="library.optimization.optimizers.momentus_caution.MomentusCaution", backend="repo"),
    OptimizerRegistration(name="oagopt", target="library.optimization.optimizers.oagopt.OAGOpt", backend="repo"),
    OptimizerRegistration(name="ocgopt", target="library.optimization.optimizers.ocgopt.OCGOpt", backend="repo"),
    OptimizerRegistration(name="projectiveadam", target="library.optimization.optimizers.projective_adam.ProjectiveAdam", backend="repo"),
    OptimizerRegistration(name="racs", target="library.optimization.optimizers.racs.RACS", backend="repo"),
    OptimizerRegistration(name="ranger21", target="library.optimization.optimizers.ranger21.Ranger21", backend="repo"),
    OptimizerRegistration(name="remaster", target="library.optimization.optimizers.remaster.REMASTER", backend="repo"),
    OptimizerRegistration(name="rmsprop", target="library.optimization.optimizers.rmsprop.RMSProp", backend="repo"),
    OptimizerRegistration(name="rmspropadopt", target="library.optimization.optimizers.rmsprop.RMSPropADOPT", backend="repo"),
    OptimizerRegistration(name="rmspropadoptmars", target="library.optimization.optimizers.rmsprop.RMSPropADOPTMARS", backend="repo"),
    OptimizerRegistration(name="scgopt", target="library.optimization.optimizers.scgopt.SCGOpt", backend="repo"),
    OptimizerRegistration(name="scion", target="library.optimization.optimizers.scion.SCION", backend="repo"),
    OptimizerRegistration(name="singstate", target="library.optimization.optimizers.singstate.SingState", backend="repo"),
    OptimizerRegistration(name="scorn", target="library.optimization.optimizers.scorn.SCORN", backend="repo"),
    OptimizerRegistration(name="scornmachina", target="library.optimization.optimizers.scorn.SCORNMachina", backend="repo"),
    OptimizerRegistration(name="scalableshampoo", target="library.optimization.optimizers.shampoo.ScalableShampoo", backend="repo"),
    OptimizerRegistration(name="sgdsai", target="library.optimization.optimizers.sgd_sai.SGDSaI", backend="repo"),
    OptimizerRegistration(name="soap", target="library.optimization.optimizers.soap.SOAP", backend="repo"),
    OptimizerRegistration(name="stablespam", target="library.optimization.optimizers.spam.StableSPAM", backend="repo"),
    OptimizerRegistration(name="simplifiedademamix", target="library.optimization.optimizers.ademamix.SimplifiedAdEMAMix", backend="repo"),
    OptimizerRegistration(
        name="simplifiedademamixexm", target="library.optimization.optimizers.ademamix.SimplifiedAdEMAMixExM", backend="repo"
    ),
    OptimizerRegistration(name="talon", target="library.optimization.optimizers.talon.TALON", backend="repo"),
    OptimizerRegistration(name="vsgd", target="library.optimization.optimizers.vsgd.VSGD", backend="repo"),
    OptimizerRegistration(name="wiwiopt", target="library.optimization.optimizers.experimental.wiwiopt.WiwiOpt", backend="repo"),
    OptimizerRegistration(
        name="adamw8bitkahan",
        target="library.optimization.optimizers.adamw.AdamW8bitKahan",
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
        target="library.optimization.wrappers.ScheduleFreeWrapper",
        kind="wrapper",
        wrapper_style="wrap_optimizer",
        backend="repo",
        capabilities=frozenset({OPT_CAP_TRAIN_EVAL_TOGGLE, OPT_CAP_SCHEDULER_ON_BASE_OPTIMIZER}),
    ),
    OptimizerRegistration(
        name="snoo_asgd",
        target="library.optimization.wrappers.SNOOASGD",
        kind="wrapper",
        wrapper_style="wrap_optimizer",
        capabilities=frozenset({OPT_CAP_SCHEDULER_ON_BASE_OPTIMIZER}),
    ),
    OptimizerRegistration(
        name="cpuoffloadoptimizer",
        target="library.optimization.wrappers.CPUOffloadOptimizerWrapper",
        kind="wrapper",
        wrapper_style="wrap_optimizer_with_base_kwargs",
        backend="torchao",
        aliases=("CPUOffloadOptimizer",),
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
    SchedulerRegistration(
        name="cosineannealingwarmrestarts",
        target="library.optimization.schedulers.warm_restarts.CosineAnnealingWarmRestarts",
        kind="torch",
        aliases=("CosineAnnealingWarmRestarts", "cosine_warm_restarts"),
    ),
    SchedulerRegistration(
        name="rexannealingwarmrestarts",
        target="library.optimization.schedulers.warm_restarts.RexAnnealingWarmRestarts",
        kind="torch",
        aliases=("RexAnnealingWarmRestarts", "rex_warm_restarts"),
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
