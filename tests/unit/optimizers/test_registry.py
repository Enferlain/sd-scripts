import pytest

from library.config.dataclasses.optimizer import OptimizerConfig
from library.optimization.registry import (
    OPT_CAP_NO_EXTERNAL_SCHEDULER,
    OPT_CAP_SCHEDULER_ON_BASE_OPTIMIZER,
    OPT_CAP_TRAIN_EVAL_TOGGLE,
    get_configured_optimizer_name,
    get_optimizer_registration,
    get_scheduler_registration,
)


@pytest.mark.training
@pytest.mark.unit
class TestOptimizerRegistry:
    @pytest.mark.parametrize(
        ("optimizer_name", "expected"),
        [
            (
                "ScheduleFreeWrapper",
                {
                    "target": "library.optimization.wrappers.ScheduleFreeWrapper",
                    "backend": "repo",
                    "kind": "wrapper",
                    "wrapper_style": "wrap_optimizer",
                    "capabilities": [OPT_CAP_SCHEDULER_ON_BASE_OPTIMIZER, OPT_CAP_TRAIN_EVAL_TOGGLE],
                },
            ),
            (
                "AdamWScheduleFree",
                {
                    "target": "schedulefree.AdamWScheduleFree",
                    "backend": "schedulefree",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [OPT_CAP_NO_EXTERNAL_SCHEDULER],
                },
            ),
            (
                "AdaBelief",
                {
                    "target": "library.optimization.optimizers.adabelief.AdaBelief",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "AdEMAMix",
                {
                    "target": "library.optimization.optimizers.ademamix.AdEMAMix",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "ADOPT",
                {
                    "target": "library.optimization.optimizers.adopt.ADOPT",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "ADOPTEMAMixScheduleFree",
                {
                    "target": "library.optimization.optimizers.adopt_schedulefree.ADOPTEMAMixScheduleFree",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [OPT_CAP_NO_EXTERNAL_SCHEDULER, OPT_CAP_TRAIN_EVAL_TOGGLE],
                },
            ),
            (
                "ADOPTMARS",
                {
                    "target": "library.optimization.optimizers.adopt.ADOPTMARS",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "ADOPTMARSScheduleFree",
                {
                    "target": "library.optimization.optimizers.adopt_schedulefree.ADOPTMARSScheduleFree",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [OPT_CAP_NO_EXTERNAL_SCHEDULER, OPT_CAP_TRAIN_EVAL_TOGGLE],
                },
            ),
            (
                "ADOPTAOScheduleFree",
                {
                    "target": "library.optimization.optimizers.adopt_schedulefree_ao.ADOPTAOScheduleFree",
                    "backend": "torchao",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [OPT_CAP_NO_EXTERNAL_SCHEDULER, OPT_CAP_TRAIN_EVAL_TOGGLE],
                },
            ),
            (
                "ADOPTNesterovScheduleFree",
                {
                    "target": "library.optimization.optimizers.adopt_schedulefree.ADOPTNesterovScheduleFree",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [OPT_CAP_NO_EXTERNAL_SCHEDULER, OPT_CAP_TRAIN_EVAL_TOGGLE],
                },
            ),
            (
                "ADOPTScheduleFree",
                {
                    "target": "library.optimization.optimizers.adopt_schedulefree.ADOPTScheduleFree",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [OPT_CAP_NO_EXTERNAL_SCHEDULER, OPT_CAP_TRAIN_EVAL_TOGGLE],
                },
            ),
            (
                "FADOPTEMAMixScheduleFree",
                {
                    "target": "library.optimization.optimizers.adopt_schedulefree.FADOPTEMAMixScheduleFree",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [OPT_CAP_NO_EXTERNAL_SCHEDULER, OPT_CAP_TRAIN_EVAL_TOGGLE],
                },
            ),
            (
                "FADOPTMARS",
                {
                    "target": "library.optimization.optimizers.adopt.FADOPTMARS",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "FADOPTMARSScheduleFree",
                {
                    "target": "library.optimization.optimizers.adopt_schedulefree.FADOPTMARSScheduleFree",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [OPT_CAP_NO_EXTERNAL_SCHEDULER, OPT_CAP_TRAIN_EVAL_TOGGLE],
                },
            ),
            (
                "FADOPTNesterovScheduleFree",
                {
                    "target": "library.optimization.optimizers.adopt_schedulefree.FADOPTNesterovScheduleFree",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [OPT_CAP_NO_EXTERNAL_SCHEDULER, OPT_CAP_TRAIN_EVAL_TOGGLE],
                },
            ),
            (
                "FADOPTScheduleFree",
                {
                    "target": "library.optimization.optimizers.adopt_schedulefree.FADOPTScheduleFree",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [OPT_CAP_NO_EXTERNAL_SCHEDULER, OPT_CAP_TRAIN_EVAL_TOGGLE],
                },
            ),
            (
                "FCompass",
                {
                    "target": "library.optimization.optimizers.fcompass.FCompass",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "FCompassADOPT",
                {
                    "target": "library.optimization.optimizers.fcompass.FCompassADOPT",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "FCompassADOPTMARS",
                {
                    "target": "library.optimization.optimizers.fcompass.FCompassADOPTMARS",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "FCompassPlus",
                {
                    "target": "library.optimization.optimizers.fcompass.FCompassPlus",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "Fira",
                {
                    "target": "library.optimization.optimizers.fira.Fira",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "GaLore",
                {
                    "target": "library.optimization.optimizers.galore.GaLore",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "Adan",
                {
                    "target": "library.optimization.optimizers.adan.Adan",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "AdamW4bitAO",
                {
                    "target": "library.optimization.optimizers.adamw_low_bit.AdamW4bitAO",
                    "backend": "torchao",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "AdamW8bitAO",
                {
                    "target": "library.optimization.optimizers.adamw_low_bit.AdamW8bitAO",
                    "backend": "torchao",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "Alice",
                {
                    "target": "library.optimization.optimizers.alice.Alice",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "AdamWfp8AO",
                {
                    "target": "library.optimization.optimizers.adamw_low_bit.AdamWfp8AO",
                    "backend": "torchao",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "ABMOG",
                {
                    "target": "library.optimization.optimizers.abmog.ABMOG",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "BCOS",
                {
                    "target": "library.optimization.optimizers.bcos.BCOS",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "CAME",
                {
                    "target": "library.optimization.optimizers.came.CAME",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "CStableAdamW",
                {
                    "target": "library.optimization.optimizers.cstableadamw.CStableAdamW",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "Compass",
                {
                    "target": "library.optimization.optimizers.compass.Compass",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "Compass8BitBNB",
                {
                    "target": "library.optimization.optimizers.compass.Compass8BitBNB",
                    "backend": "bitsandbytes",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "CompassADOPT",
                {
                    "target": "library.optimization.optimizers.compass.CompassADOPT",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "CompassADOPTMARS",
                {
                    "target": "library.optimization.optimizers.compass.CompassADOPTMARS",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "CompassAO",
                {
                    "target": "library.optimization.optimizers.compass.CompassAO",
                    "backend": "torchao",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "CompassPlus",
                {
                    "target": "library.optimization.optimizers.compass.CompassPlus",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "Ranger21",
                {
                    "target": "library.optimization.optimizers.ranger21.Ranger21",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "ScalableShampoo",
                {
                    "target": "library.optimization.optimizers.shampoo.ScalableShampoo",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "LaProp",
                {
                    "target": "library.optimization.optimizers.laprop.LaProp",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "LPFAdamW",
                {
                    "target": "library.optimization.optimizers.lpf_adamw.LPFAdamW",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "Lamb",
                {
                    "target": "library.optimization.optimizers.lamb.Lamb",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "SGDSaI",
                {
                    "target": "library.optimization.optimizers.sgd_sai.SGDSaI",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "SOAP",
                {
                    "target": "library.optimization.optimizers.soap.SOAP",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "SimplifiedAdEMAMix",
                {
                    "target": "library.optimization.optimizers.ademamix.SimplifiedAdEMAMix",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "SimplifiedAdEMAMixExM",
                {
                    "target": "library.optimization.optimizers.ademamix.SimplifiedAdEMAMixExM",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "Adai",
                {
                    "target": "library.optimization.optimizers.adai.Adai",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "VSGD",
                {
                    "target": "library.optimization.optimizers.vsgd.VSGD",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "RACS",
                {
                    "target": "library.optimization.optimizers.racs.RACS",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "RMSProp",
                {
                    "target": "library.optimization.optimizers.rmsprop.RMSProp",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "RMSPropADOPT",
                {
                    "target": "library.optimization.optimizers.rmsprop.RMSPropADOPT",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "RMSPropADOPTMARS",
                {
                    "target": "library.optimization.optimizers.rmsprop.RMSPropADOPTMARS",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "SCION",
                {
                    "target": "library.optimization.optimizers.scion.SCION",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "SingState",
                {
                    "target": "library.optimization.optimizers.singstate.SingState",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "SCORN",
                {
                    "target": "library.optimization.optimizers.scorn.SCORN",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "SCORNMachina",
                {
                    "target": "library.optimization.optimizers.scornmachina.SCORNMachina",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "Dehaze",
                {
                    "target": "library.optimization.optimizers.dehaze.Dehaze",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "FFTDescent",
                {
                    "target": "library.optimization.optimizers.fftdescent.FFTDescent",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "FARMSCrop",
                {
                    "target": "library.optimization.optimizers.farmscrop.FARMSCrop",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "FARMSCropV2",
                {
                    "target": "library.optimization.optimizers.farmscrop_v2.FARMSCropV2",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "FMARSCrop",
                {
                    "target": "library.optimization.optimizers.fmarscrop.FMARSCrop",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "FMARSCropV2",
                {
                    "target": "library.optimization.optimizers.fmarscrop_v2.FMARSCropV2",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "FishMonger",
                {
                    "target": "library.optimization.optimizers.fishmonger.FishMonger",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "FishMonger8BitBNB",
                {
                    "target": "library.optimization.optimizers.fishmonger.FishMonger8BitBNB",
                    "backend": "bitsandbytes",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "GOODDOG",
                {
                    "target": "library.optimization.optimizers.gooddog.GOODDOG",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "GrokFastAdamW",
                {
                    "target": "library.optimization.optimizers.grokfast.GrokFastAdamW",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "Glyph",
                {
                    "target": "library.optimization.optimizers.glyph.Glyph",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "Mythical",
                {
                    "target": "library.optimization.optimizers.mythical.Mythical",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "OAGOpt",
                {
                    "target": "library.optimization.optimizers.oagopt.OAGOpt",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "OCGOpt",
                {
                    "target": "library.optimization.optimizers.ocgopt.OCGOpt",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "ProjectiveAdam",
                {
                    "target": "library.optimization.optimizers.projective_adam.ProjectiveAdam",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "SCGOpt",
                {
                    "target": "library.optimization.optimizers.scgopt.SCGOpt",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "WiwiOpt",
                {
                    "target": "library.optimization.optimizers.wiwiopt.WiwiOpt",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "StableSPAM",
                {
                    "target": "library.optimization.optimizers.spam.StableSPAM",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "TALON",
                {
                    "target": "library.optimization.optimizers.talon.TALON",
                    "backend": "repo",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "AdamW8bitKahan",
                {
                    "target": "library.optimization.optimizers.adamw_8bit_kahan.AdamW8bitKahan",
                    "backend": "bitsandbytes",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
            (
                "Adafactor",
                {
                    "target": "transformers.optimization.Adafactor",
                    "backend": "transformers",
                    "kind": "optimizer",
                    "wrapper_style": None,
                    "capabilities": [],
                },
            ),
        ],
    )
    def test_optimizer_registration_metadata(self, optimizer_name, expected):
        """Optimizer registrations should expose stable target/backend/capability metadata."""
        registration = get_optimizer_registration(optimizer_name)

        assert registration is not None
        assert registration.target == expected["target"]
        assert registration.backend == expected["backend"]
        assert registration.kind == expected["kind"]
        assert registration.wrapper_style == expected["wrapper_style"]
        for capability in expected["capabilities"]:
            assert registration.supports(capability)

    def test_configured_optimizer_name_respects_compat_flags(self):
        """Compatibility flags should resolve through the shared optimizer-name helper."""
        config = OptimizerConfig(use_8bit_adam=True)

        assert get_configured_optimizer_name(config) == "AdamW8bit"


@pytest.mark.training
@pytest.mark.unit
class TestSchedulerRegistry:
    def test_scheduler_registry_resolves_builtin_alias(self):
        """Known scheduler aliases should resolve through the shared registry."""
        registration = get_scheduler_registration("CosineAnnealingLR")

        assert registration is not None
        assert registration.name == "cosineannealinglr"

    @pytest.mark.parametrize(
        ("scheduler_name", "target", "kind"),
        [
            (
                "CosineAnnealingLR",
                "torch.optim.lr_scheduler.CosineAnnealingLR",
                "torch",
            ),
            (
                "CosineAnnealingWarmRestarts",
                "library.optimization.schedulers.warm_restarts.CosineAnnealingWarmRestarts",
                "torch",
            ),
            (
                "RexAnnealingWarmRestarts",
                "library.optimization.schedulers.warm_restarts.RexAnnealingWarmRestarts",
                "torch",
            ),
        ],
    )
    def test_scheduler_registration_metadata(self, scheduler_name, target, kind):
        """Scheduler registrations should expose stable target and dispatch-kind metadata."""
        registration = get_scheduler_registration(scheduler_name)

        assert registration is not None
        assert registration.target == target
        assert registration.kind == kind
