import pytest
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config.dataclasses.optimizer import LearningRatesConfig, OptimizerConfig, SchedulerConfig
from library.config.dataclasses.training import TrainingConfig
from library.optimization.optimizer_factory import get_optimizer
from library.optimization.optimizer_utils import get_optimizer_train_eval_fn, is_schedulefree_optimizer
from library.optimization.scheduler import get_scheduler_fix
from library.optimization.schedulers import CosineAnnealingWarmRestarts, RexAnnealingWarmRestarts


def _build_optimizer_and_training_config(
    mock_model_parameters,
    *,
    optimizer_type: str = "AdamW",
    learning_rate: float = 1e-4,
    scheduler_config: SchedulerConfig | None = None,
    optimizer_args: list[str] | None = None,
    max_train_steps: int = 25,
):
    optimizer_config = OptimizerConfig(
        optimizer_type=optimizer_type,
        learning_rates=LearningRatesConfig(base=learning_rate),
        optimizer_args=optimizer_args or [],
        scheduler=scheduler_config or SchedulerConfig(),
    )
    _, _, optimizer = get_optimizer(
        optimizer_config,
        optimizer_config.learning_rates,
        optimizer_config.scheduler,
        mock_model_parameters,
    )
    training_config = TrainingConfig(max_train_steps=max_train_steps)
    return optimizer_config, training_config, optimizer


def _assert_group_values(group: dict, expected_values: dict):
    for key, value in expected_values.items():
        assert group[key] == value


@pytest.mark.training
@pytest.mark.unit
class TestAbsorbedWrappers:
    def test_registered_schedulefree_wrapper_builds_from_base_optimizer(self, mock_model_parameters):
        """Explicit wrapper optimizers should build their base optimizer through the shared wrapper path."""
        config = OptimizerConfig(
            optimizer_type="ScheduleFreeWrapper",
            learning_rates=LearningRatesConfig(base=3e-4),
            optimizer_args=[
                "base_optimizer_type=AdamW",
                "base_optimizer.weight_decay=0.01",
                "momentum=0.95",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            mock_model_parameters,
        )

        assert "ScheduleFreeWrapper" in optimizer_name
        assert hasattr(optimizer, "base_optimizer")
        assert optimizer.base_optimizer.__class__.__name__ == "AdamW"
        assert optimizer.base_optimizer.param_groups[0]["weight_decay"] == 0.01

    def test_registered_schedulefree_wrapper_schedules_base_optimizer(self, mock_model_parameters):
        """Registered wrappers should route schedulers onto their base optimizer when declared."""
        optimizer_config, training_config, optimizer = _build_optimizer_and_training_config(
            mock_model_parameters,
            optimizer_type="ScheduleFreeWrapper",
            learning_rate=3e-4,
            optimizer_args=[
                "base_optimizer_type=AdamW",
                "base_optimizer.weight_decay=0.01",
                "momentum=0.95",
            ],
            scheduler_config=SchedulerConfig(lr_scheduler="constant_with_warmup", lr_warmup_steps=5),
        )

        scheduler = get_scheduler_fix(
            optimizer_config.scheduler,
            optimizer_config,
            training_config,
            optimizer,
            num_processes=1,
        )

        assert scheduler.optimizer is optimizer.base_optimizer

    def test_registered_schedulefree_wrapper_is_schedulefree_for_train_eval_handling(self, mock_model_parameters):
        """Repo-owned wrapper registrations should advertise schedule-free train/eval behavior."""
        config = OptimizerConfig(
            optimizer_type="ScheduleFreeWrapper",
            learning_rates=LearningRatesConfig(base=3e-4),
            optimizer_args=[
                "base_optimizer_type=AdamW",
                "base_optimizer.weight_decay=0.01",
                "momentum=0.95",
            ],
        )
        _, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            mock_model_parameters,
        )

        assert is_schedulefree_optimizer(optimizer, config)
        train_fn, eval_fn = get_optimizer_train_eval_fn(optimizer, config)
        assert callable(train_fn)
        assert callable(eval_fn)

    def test_registered_snoo_asgd_builds_from_base_optimizer(self, mock_model_parameters):
        """Repo-owned wrappers should be able to wrap a base optimizer with namespaced base args."""
        config = OptimizerConfig(
            optimizer_type="snoo_asgd",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=[
                "base_optimizer_type=AdamW",
                "base_optimizer.weight_decay=0.02",
                "alpha=0.5",
                "t0=0",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            mock_model_parameters,
        )

        assert "SNOOASGD" in optimizer_name
        assert optimizer.base_optimizer.__class__.__name__ == "AdamW"
        assert optimizer.base_optimizer.param_groups[0]["weight_decay"] == 0.02
        assert optimizer.alpha == 0.5

    def test_registered_snoo_asgd_schedules_base_optimizer(self, mock_model_parameters):
        """Wrapper registrations should drive scheduler-on-base-optimizer routing generically."""
        optimizer_config, training_config, optimizer = _build_optimizer_and_training_config(
            mock_model_parameters,
            optimizer_type="snoo_asgd",
            optimizer_args=[
                "base_optimizer_type=AdamW",
                "base_optimizer.weight_decay=0.02",
                "alpha=0.5",
                "t0=0",
            ],
            scheduler_config=SchedulerConfig(lr_scheduler="constant_with_warmup", lr_warmup_steps=5),
        )

        scheduler = get_scheduler_fix(
            optimizer_config.scheduler,
            optimizer_config,
            training_config,
            optimizer,
            num_processes=1,
        )

        assert scheduler.optimizer is optimizer.base_optimizer


@pytest.mark.training
@pytest.mark.unit
class TestAbsorbedOptimizers:
    @pytest.mark.parametrize(
        ("optimizer_type", "learning_rate", "optimizer_args", "expected_name", "expected_group_values"),
        [
            (
                "AdaBelief",
                2e-4,
                ["weight_decay=0.02", "rectify=True", "cautious=True"],
                "AdaBelief",
                {"weight_decay": 0.02, "rectify": True, "cautious": True},
            ),
            (
                "ADOPT",
                1e-4,
                ["weight_decay=0.01", "weight_decouple=True", "clip=0.5", "update_strategy='grams'"],
                "ADOPT",
                {
                    "weight_decay": 0.01,
                    "weight_decouple": True,
                    "clip": 0.5,
                    "update_strategy": "grams",
                },
            ),
            (
                "ADOPTEMAMixScheduleFree",
                2.5e-3,
                [
                    "weight_decay=0.01",
                    "weight_decouple=True",
                    "adaptive_clip=0.0",
                    "alpha=4.0",
                    "t_alpha_beta3=20",
                    "r=0.5",
                ],
                "ADOPTEMAMixScheduleFree",
                {
                    "weight_decay": 0.01,
                    "weight_decouple": True,
                    "adaptive_clip": 0.0,
                    "alpha": 4.0,
                    "t_alpha_beta3": 20,
                    "r": 0.5,
                },
            ),
            (
                "AdEMAMix",
                1e-4,
                ["weight_decay=0.01", "weight_decouple=True", "alpha=4.0", "update_strategy='grams'"],
                "AdEMAMix",
                {"weight_decay": 0.01, "weight_decouple": True, "alpha": 4.0, "update_strategy": "grams"},
            ),
            (
                "ADOPTMARS",
                2.5e-3,
                [
                    "weight_decay=0.01",
                    "weight_decouple=True",
                    "stable_weight_decay=True",
                    "adaptive_clip=0.5",
                    "adaptive_clip_type='unit'",
                    "gamma=0.05",
                    "cautious=False",
                ],
                "ADOPTMARS",
                {
                    "weight_decay": 0.01,
                    "weight_decouple": True,
                    "stable_weight_decay": True,
                    "adaptive_clip": 0.5,
                    "adaptive_clip_type": "unit",
                    "gamma": 0.05,
                    "cautious": False,
                },
            ),
            (
                "ADOPTMARSScheduleFree",
                2.5e-3,
                [
                    "weight_decay=0.01",
                    "weight_decouple=True",
                    "gamma=0.05",
                    "adaptive_clip_type='unit'",
                    "r=0.25",
                ],
                "ADOPTMARSScheduleFree",
                {
                    "weight_decay": 0.01,
                    "weight_decouple": True,
                    "gamma": 0.05,
                    "adaptive_clip_type": "unit",
                    "r": 0.25,
                },
            ),
            (
                "ADOPTAOScheduleFree",
                5e-4,
                [
                    "weight_decay=0.01",
                    "state_precision='parameter'",
                    "block_size=0",
                    "adaptive_clip=0.0",
                    "torch_compile=False",
                ],
                "ADOPTAOScheduleFree",
                {
                    "weight_decay": 0.01,
                    "adaptive_clip": 0.0,
                },
            ),
            (
                "ADOPTNesterovScheduleFree",
                2.5e-3,
                [
                    "weight_decay=0.01",
                    "weight_decouple=True",
                    "debias_beta3=True",
                    "adaptive_clip=0.0",
                    "r=0.5",
                ],
                "ADOPTNesterovScheduleFree",
                {
                    "weight_decay": 0.01,
                    "weight_decouple": True,
                    "debias_beta3": True,
                    "adaptive_clip": 0.0,
                    "r": 0.5,
                },
            ),
            (
                "Adan",
                1e-4,
                ["weight_decay=0.01", "max_grad_norm=0.5", "use_gc=True", "update_strategy='grams'"],
                "Adan",
                {"weight_decay": 0.01, "max_grad_norm": 0.5, "update_strategy": "grams"},
            ),
            (
                "Alice",
                1e-2,
                [
                    "alpha=0.3",
                    "alpha_c=0.15",
                    "update_interval=8",
                    "rank=8",
                    "leading_basis=4",
                    "gamma=1.03",
                    "adam_lr=1e-4",
                ],
                "Alice",
                {
                    "alpha": 0.3,
                    "alpha_c": 0.15,
                    "update_interval": 8,
                    "rank": 8,
                    "leading_basis": 4,
                    "gamma": 1.03,
                    "adam_lr": 1e-4,
                },
            ),
            (
                "FADOPTMARS",
                2.5e-3,
                [
                    "weight_decay=0.02",
                    "weight_decouple=True",
                    "stable_weight_decay=True",
                    "adaptive_clip=0.5",
                    "adaptive_clip_type='layer'",
                    "fisher_clip=0.75",
                    "gamma=0.05",
                    "cautious=False",
                ],
                "FADOPTMARS",
                {
                    "weight_decay": 0.02,
                    "weight_decouple": True,
                    "stable_weight_decay": True,
                    "adaptive_clip": 0.5,
                    "adaptive_clip_type": "layer",
                    "fisher_clip": 0.75,
                    "gamma": 0.05,
                    "cautious": False,
                },
            ),
            (
                "FADOPTEMAMixScheduleFree",
                2.5e-3,
                [
                    "weight_decay=0.02",
                    "weight_decouple=True",
                    "fisher_clip=0.75",
                    "alpha=4.0",
                    "t_alpha_beta3=20",
                ],
                "FADOPTEMAMixScheduleFree",
                {
                    "weight_decay": 0.02,
                    "weight_decouple": True,
                    "fisher_clip": 0.75,
                    "alpha": 4.0,
                    "t_alpha_beta3": 20,
                },
            ),
            (
                "FADOPTMARSScheduleFree",
                2.5e-3,
                [
                    "weight_decay=0.02",
                    "weight_decouple=True",
                    "fisher_clip=0.75",
                    "gamma=0.05",
                    "weight_decay_lr_decouple=True",
                    "weight_decay_lr_max=1e-3",
                ],
                "FADOPTMARSScheduleFree",
                {
                    "weight_decay": 0.02,
                    "weight_decouple": True,
                    "fisher_clip": 0.75,
                    "gamma": 0.05,
                    "weight_decay_lr_decouple": True,
                    "weight_decay_lr_max": 1e-3,
                },
            ),
            (
                "FADOPTNesterovScheduleFree",
                2.5e-3,
                [
                    "weight_decay=0.02",
                    "weight_decouple=True",
                    "fisher_clip=0.5",
                    "debias_beta3=True",
                    "adaptive_clip=0.0",
                ],
                "FADOPTNesterovScheduleFree",
                {
                    "weight_decay": 0.02,
                    "weight_decouple": True,
                    "fisher_clip": 0.5,
                    "debias_beta3": True,
                    "adaptive_clip": 0.0,
                },
            ),
            (
                "FADOPTScheduleFree",
                2.5e-3,
                [
                    "weight_decay=0.02",
                    "weight_decouple=True",
                    "fisher_clip=0.5",
                    "adaptive_clip_type='unit'",
                    "debias_beta2=True",
                ],
                "FADOPTScheduleFree",
                {
                    "weight_decay": 0.02,
                    "weight_decouple": True,
                    "fisher_clip": 0.5,
                    "adaptive_clip_type": "unit",
                    "debias_beta2": True,
                },
            ),
            (
                "Compass",
                1e-4,
                ["weight_decay=0.01", "weight_decouple=True", "clip=0.5", "adaptive_clipping=True", "update_strategy='grams'"],
                "Compass",
                {
                    "weight_decay": 0.01,
                    "weight_decouple": True,
                    "clip": 0.5,
                    "adaptive_clipping": True,
                    "update_strategy": "grams",
                },
            ),
            (
                "CompassADOPT",
                1e-4,
                [
                    "weight_decay=0.01",
                    "weight_decouple=True",
                    "factor_second_moment=True",
                    "compass_second_moment_smoothing=False",
                    "use_stable_spam_clipping=False",
                ],
                "CompassADOPT",
                {
                    "weight_decay": 0.01,
                    "weight_decouple": True,
                    "factor_second_moment": True,
                    "compass_second_moment_smoothing": False,
                    "use_stable_spam_clipping": False,
                },
            ),
            (
                "CompassADOPTMARS",
                1e-4,
                ["weight_decay=0.02", "weight_decouple=True", "factor_second_moment=True", "gamma=0.05", "cautious=False"],
                "CompassADOPTMARS",
                {
                    "weight_decay": 0.02,
                    "weight_decouple": True,
                    "factor_second_moment": True,
                    "gamma": 0.05,
                    "cautious": False,
                },
            ),
            (
                "CompassPlus",
                1e-4,
                ["weight_decay=0.01", "use_lookahead=True", "lookahead_merge_time=3", "use_softplus=True"],
                "CompassPlus",
                {
                    "weight_decay": 0.01,
                    "use_lookahead": True,
                    "lookahead_merge_time": 3,
                    "use_softplus": True,
                },
            ),
            (
                "FCompass",
                1e-4,
                ["weight_decay=0.01", "amp_fac=1.5", "clip=0.5", "centralization=0.5"],
                "FCompass",
                {"weight_decay": 0.01, "amp_fac": 1.5, "clip": 0.5, "centralization": 0.5},
            ),
            (
                "FCompassADOPT",
                1e-4,
                ["weight_decay=0.01", "weight_decouple=True", "fisher_clip=0.5", "compass_second_moment_smoothing=False"],
                "FCompassADOPT",
                {
                    "weight_decay": 0.01,
                    "weight_decouple": True,
                    "fisher_clip": 0.5,
                    "compass_second_moment_smoothing": False,
                },
            ),
            (
                "FCompassADOPTMARS",
                1e-4,
                ["weight_decay=0.02", "weight_decouple=True", "fisher_clip=0.75", "gamma=0.05"],
                "FCompassADOPTMARS",
                {
                    "weight_decay": 0.02,
                    "weight_decouple": True,
                    "fisher_clip": 0.75,
                    "gamma": 0.05,
                },
            ),
            (
                "FCompassPlus",
                1e-4,
                ["weight_decay=0.01", "use_lookahead=True", "lookahead_merge_time=3", "use_softplus=False"],
                "FCompassPlus",
                {
                    "weight_decay": 0.01,
                    "use_lookahead": True,
                    "lookahead_merge_time": 3,
                    "use_softplus": False,
                },
            ),
            (
                "Fira",
                1e-4,
                ["weight_decay=0.01", "rank=4", "update_proj_gap=2", "scale=0.5", "projection_type='std'"],
                "Fira",
                {
                    "weight_decay": 0.01,
                    "rank": 4,
                    "update_proj_gap": 2,
                    "scale": 0.5,
                    "projection_type": "std",
                },
            ),
            (
                "GaLore",
                1e-4,
                ["weight_decay=0.01", "rank=4", "update_proj_gap=2", "scale=0.5", "projection_type='std'"],
                "GaLore",
                {
                    "weight_decay": 0.01,
                    "rank": 4,
                    "update_proj_gap": 2,
                    "scale": 0.5,
                    "projection_type": "std",
                },
            ),
            (
                "LaProp",
                4e-4,
                ["weight_decay=0.02", "centered=True", "ams_bound=True", "cautious=True"],
                "LaProp",
                {"weight_decay": 0.02, "centered": True, "ams_bound": True},
            ),
            (
                "Lamb",
                1e-4,
                [
                    "weight_decay=0.01",
                    "rectify=True",
                    "pre_norm=True",
                    "adanorm=True",
                    "max_grad_norm=0.5",
                ],
                "Lamb",
                {
                    "weight_decay": 0.01,
                    "rectify": True,
                    "adanorm": True,
                    "max_grad_norm": 0.5,
                },
            ),
            (
                "LPFAdamW",
                2e-4,
                ["weight_decay=0.03", "amp_fac=1.5", "centralization=0.25", "betas=(0.9, 0.92, 0.999)"],
                "LPFAdamW",
                {
                    "weight_decay": 0.03,
                    "amp_fac": 1.5,
                    "centralization": 0.25,
                    "betas": (0.9, 0.92, 0.999),
                },
            ),
            (
                "Ranger21",
                5e-4,
                [
                    "num_iterations=20",
                    "weight_decay=0.01",
                    "beta0=0.8",
                    "lookahead_merge_time=4",
                    "disable_lr_scheduler=True",
                    "use_softplus=False",
                ],
                "Ranger21",
                {
                    "weight_decay": 0.01,
                    "weight_decouple": True,
                    "fixed_decay": False,
                    "adam_debias": False,
                },
            ),
            (
                "Dehaze",
                1e-3,
                [
                    "weight_decay=0.01",
                    "weight_decay_rate=0.99",
                    "stage1_atan2=False",
                    "stage2_atan2=True",
                    "adaptive_muon=True",
                    "stochastic_fp=False",
                    "torch_compile=False",
                ],
                "Dehaze",
                {
                    "weight_decay": 0.01,
                    "weight_decay_rate": 0.99,
                    "stage1_atan2": False,
                    "stage2_atan2": True,
                    "adaptive_muon": True,
                    "stochastic_fp": False,
                    "torch_compile": False,
                },
            ),
            (
                "GOODDOG",
                1e-3,
                [
                    "weight_decay=0.01",
                    "weight_decay_rate=0.99",
                    "invariant=True",
                    "adaptive_muon=True",
                    "orthograd=True",
                    "stochastic_fp=False",
                ],
                "GOODDOG",
                {
                    "weight_decay": 0.01,
                    "weight_decay_rate": 0.99,
                    "invariant": True,
                    "adaptive_muon": True,
                    "orthograd": True,
                    "stochastic_fp": False,
                },
            ),
            (
                "SCION",
                1e-3,
                [
                    "momentum=0.2",
                    "constraint=True",
                    "norm_type=7",
                    "norm_kwargs={'transpose': True}",
                    "scale=2.5",
                    "weight_decay=0.01",
                    "weight_decouple=False",
                    "use_focus=True",
                    "focus_beta=0.9",
                    "adaptive_clip=0.1",
                    "adaptive_clip_type='unit'",
                    "update_strategy='both'",
                    "use_stable_spam_clipping=True",
                    "ssc_t_max=8",
                    "torch_compile=False",
                ],
                "SCION",
                {
                    "momentum": 0.2,
                    "constraint": True,
                    "norm_type": 7,
                    "norm_kwargs": {"transpose": True},
                    "scale": 2.5,
                    "weight_decay": 0.01,
                    "weight_decouple": False,
                    "use_focus": True,
                    "focus_beta": 0.9,
                    "adaptive_clip": 0.1,
                    "adaptive_clip_type": "unit",
                    "update_strategy": "both",
                    "use_stable_spam_clipping": True,
                    "torch_compile": False,
                },
            ),
            (
                "ScalableShampoo",
                1e-3,
                [
                    "weight_decay=0.01",
                    "block_size=32",
                    "start_preconditioning_step=1",
                    "preconditioning_compute_steps=1",
                    "statistics_compute_steps=1",
                    "graft_type=2",
                    "use_svd=False",
                ],
                "ScalableShampoo",
                {
                    "weight_decay": 0.01,
                    "decoupled_weight_decay": False,
                    "decoupled_learning_rate": True,
                    "moving_average_for_momentum": False,
                    "nesterov": True,
                },
            ),
            (
                "StableSPAM",
                1e-3,
                [
                    "weight_decay=0.01",
                    "gamma1=0.8",
                    "gamma2=0.9999",
                    "gamma3=0.99",
                    "t_max=10",
                    "eta_min=0.25",
                    "update_proj_gap=5",
                    "use_adopt=True",
                    "update_strategy='grams'",
                ],
                "StableSPAM",
                {
                    "weight_decay": 0.01,
                    "use_adopt": True,
                    "update_strategy": "grams",
                },
            ),
            (
                "Mythical",
                1e-3,
                [
                    "weight_decay=0.01",
                    "weight_decay_rate=0.99",
                    "amp=1.5",
                    "orthograd=True",
                    "adaptive_ema=True",
                    "atan2=True",
                    "warmup=True",
                    "cautious_min=0.25",
                    "stochastic_fp=False",
                ],
                "Mythical",
                {
                    "weight_decay": 0.01,
                    "weight_decay_rate": 0.99,
                    "amp": 1.5,
                    "orthograd": True,
                    "adaptive_ema": True,
                    "atan2": True,
                    "warmup": True,
                    "cautious_min": 0.25,
                    "stochastic_fp": False,
                },
            ),
            (
                "SGDSaI",
                1e-3,
                ["momentum=0.8", "weight_decay=0.01", "cautious=True"],
                "SGDSaI",
                {"momentum": 0.8, "weight_decay": 0.01, "cautious": True},
            ),
            (
                "SOAP",
                3e-3,
                [
                    "weight_decay=0.02",
                    "precondition_frequency=2",
                    "max_precondition_dim=32",
                    "merge_dims=True",
                    "precondition_1d=True",
                    "correct_bias=False",
                ],
                "SOAP",
                {
                    "weight_decay": 0.02,
                    "precondition_frequency": 2,
                    "max_precondition_dim": 32,
                    "merge_dims": True,
                    "precondition_1d": True,
                    "correct_bias": False,
                },
            ),
            (
                "SimplifiedAdEMAMix",
                1e-4,
                ["weight_decay=0.01", "alpha=0.5", "state_storage_dtype='float32'", "update_strategy='both'"],
                "SimplifiedAdEMAMix",
                {"weight_decay": 0.01, "alpha": 0.5, "update_strategy": "both"},
            ),
            (
                "SimplifiedAdEMAMixExM",
                2e-4,
                ["weight_decay=0.02", "alpha=0.8", "update_strategy='grams'", "state_storage_dtype='float32'"],
                "SimplifiedAdEMAMixExM",
                {"weight_decay": 0.02, "alpha": 0.8, "update_strategy": "grams"},
            ),
            (
                "Adai",
                3e-4,
                ["weight_decay=0.01", "weight_decouple=True", "dampening=0.8", "use_gc=True"],
                "Adai",
                {"weight_decay": 0.01, "weight_decouple": True, "dampening": 0.8},
            ),
            (
                "VSGD",
                5e-3,
                ["ghattg=20.0", "ps=1e-6", "tau1=0.75", "tau2=0.85", "stochastic_fp=False"],
                "VSGD",
                {"stochastic_fp": False, "weight_decouple": True},
            ),
            (
                "RACS",
                2e-3,
                ["beta=0.85", "alpha=0.05", "gamma=1.02", "adam_lr=1e-4", "adam_weight_decay=0.01"],
                "RACS",
                {
                    "beta": 0.85,
                    "alpha": 0.05,
                    "gamma": 1.02,
                    "adam_lr": 1e-4,
                    "adam_weight_decay": 0.01,
                },
            ),
            (
                "RMSProp",
                1e-3,
                ["weight_decay=0.01", "clip=0.5", "rectify_variance=True", "clip_loc='both'"],
                "RMSProp",
                {"weight_decay": 0.01, "clip": 0.5, "rectify_variance": True, "clip_loc": "both"},
            ),
            (
                "RMSPropADOPT",
                5e-4,
                ["weight_decay=0.02", "weight_decouple=True", "factor_second_moment=True", "debias_beta=False"],
                "RMSPropADOPT",
                {
                    "weight_decay": 0.02,
                    "weight_decouple": True,
                    "factor_second_moment": True,
                    "debias_beta": False,
                },
            ),
            (
                "RMSPropADOPTMARS",
                5e-4,
                ["weight_decay=0.03", "weight_decouple=True", "factor_second_moment=True", "gamma=0.05"],
                "RMSPropADOPTMARS",
                {
                    "weight_decay": 0.03,
                    "weight_decouple": True,
                    "factor_second_moment": True,
                    "gamma": 0.05,
                },
            ),
        ],
    )
    def test_repo_owned_optimizer_construction(
        self,
        mock_model_parameters,
        optimizer_type,
        learning_rate,
        optimizer_args,
        expected_name,
        expected_group_values,
    ):
        """Repo-owned absorbed optimizers should build through the shared registry path."""
        config = OptimizerConfig(
            optimizer_type=optimizer_type,
            learning_rates=LearningRatesConfig(base=learning_rate),
            optimizer_args=optimizer_args,
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            mock_model_parameters,
        )

        assert expected_name in optimizer_name
        _assert_group_values(optimizer.param_groups[0], expected_group_values)

        if optimizer_type == "Adan":
            assert optimizer.use_gc is True
        if optimizer_type == "Alice":
            assert optimizer.param_groups[0]["_lr_ratio"] == 0.01
        if optimizer_type == "AdEMAMix":
            assert optimizer.param_groups[0]["adopt"] is False
        if optimizer_type == "CompassPlus":
            assert optimizer.use_lookahead is True
        if optimizer_type == "FCompassPlus":
            assert optimizer.use_lookahead is True
        if optimizer_type == "Fira":
            assert optimizer.param_groups[0]["projection_type"] == "std"
        if optimizer_type == "GaLore":
            assert optimizer.param_groups[0]["projection_type"] == "std"
        if optimizer_type == "LaProp":
            assert optimizer.cautious is True
        if optimizer_type == "Lamb":
            assert optimizer.pre_norm is True
        if optimizer_type == "Dehaze":
            assert optimizer._init_lr == learning_rate
        if optimizer_type == "GOODDOG":
            assert optimizer._init_lr == learning_rate
        if optimizer_type == "SGDSaI":
            assert optimizer.has_warmup is False
        if optimizer_type == "Mythical":
            assert optimizer._init_lr == learning_rate
        if optimizer_type == "SOAP":
            assert optimizer.data_format == "channels_first"
        if optimizer_type == "Ranger21":
            assert optimizer.beta0 == 0.8
            assert optimizer.disable_lr_scheduler is True
            assert optimizer.lookahead_merge_time == 4
            assert optimizer.use_softplus is False
        if optimizer_type == "SCION":
            assert optimizer.ssc_t_max == 8
            assert optimizer.warmup is not None
        if optimizer_type == "ScalableShampoo":
            assert optimizer.block_size == 32
            assert optimizer.start_preconditioning_step == 1
            assert optimizer.preconditioning_compute_steps == 1
        if optimizer_type == "StableSPAM":
            assert optimizer.gamma1 == 0.8
            assert optimizer.gamma2 == 0.9999
            assert optimizer.gamma3 == 0.99
            assert optimizer.t_max == 10
            assert optimizer.update_proj_gap == 5
            assert optimizer.warmup is not None
        if optimizer_type == "Adai":
            assert optimizer.use_gc is True
        if optimizer_type == "VSGD":
            assert optimizer.param_groups[0]["pa2"] > 1.0
        if optimizer_type == "RMSPropADOPT":
            assert optimizer.param_groups[0]["adaptive_clip_type"] == "layer"
        if optimizer_type == "RMSPropADOPTMARS":
            assert optimizer.param_groups[0]["adaptive_clip_type"] == "layer"
        if optimizer_type == "SimplifiedAdEMAMix":
            assert optimizer.state_storage_dtype == torch.float32
        if optimizer_type == "SimplifiedAdEMAMixExM":
            assert optimizer.state_storage_dtype == torch.float32
        if optimizer_type == "ADOPTAOScheduleFree":
            assert optimizer.block_size == 0
            assert optimizer.min_quant_size == 4096
            assert optimizer.state_precision == "parameter"
            assert optimizer.torch_compile is False

    @pytest.mark.parametrize(
        ("optimizer_type", "optimizer_args", "expected_class_name", "expected_block_size"),
        [
            ("AdamW8bitAO", ["weight_decay=0.01", "block_size=512", "bf16_stochastic_round=True"], "AdamW8bitAO", 512),
            ("AdamW4bitAO", ["weight_decay=0.02", "block_size=384", "bf16_stochastic_round=True"], "AdamW4bitAO", 384),
            ("AdamWfp8AO", ["weight_decay=0.03", "block_size=640", "bf16_stochastic_round=True"], "AdamWfp8AO", 640),
        ],
    )
    def test_repo_owned_torchao_adamw_family_construction(
        self,
        mock_model_parameters,
        optimizer_type,
        optimizer_args,
        expected_class_name,
        expected_block_size,
    ):
        """TorchAO AdamW family variants should construct through the shared registry path."""
        pytest.importorskip("torchao.optim.adam")
        config = OptimizerConfig(
            optimizer_type=optimizer_type,
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=optimizer_args,
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            mock_model_parameters,
        )

        assert expected_class_name in optimizer_name
        assert optimizer.__class__.__name__ == expected_class_name
        assert optimizer.param_groups[0]["weight_decay"] > 0.0
        assert optimizer.block_size == expected_block_size
        assert optimizer.bf16_stochastic_round is True

    def test_registered_adamw8bitkahan_builds_repo_owned_augmentation(self, mock_model_parameters):
        """Optimizer augmentations should construct through the same shared registry path."""
        pytest.importorskip("bitsandbytes")
        config = OptimizerConfig(
            optimizer_type="AdamW8bitKahan",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=[
                "weight_decay=0.01",
                "stabilize=False",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            mock_model_parameters,
        )

        assert "AdamW8bitKahan" in optimizer_name
        assert optimizer.stabilize is False

    def test_registered_galore_initializes_projector_for_ranked_2d_parameters(self, mock_model_parameters):
        """GaLore should create a projector only for ranked 2D parameter groups during step execution."""
        config = OptimizerConfig(
            optimizer_type="GaLore",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=[
                "weight_decay=0.01",
                "rank=4",
                "update_proj_gap=1",
                "scale=0.5",
                "projection_type='std'",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            mock_model_parameters,
        )

        for parameter in mock_model_parameters:
            parameter.grad = torch.randn_like(parameter)

        optimizer.step()

        matrix_param = next(parameter for parameter in mock_model_parameters if parameter.ndim == 2)
        vector_param = next(parameter for parameter in mock_model_parameters if parameter.ndim == 1)

        assert "GaLore" in optimizer_name
        assert "projector" in optimizer.state[matrix_param]
        assert "projector" not in optimizer.state[vector_param]
        assert optimizer.state[matrix_param]["projector"].rank == 4
        assert optimizer.state[matrix_param]["projector"].projection_type == "std"

    def test_registered_fira_initializes_projector_for_ranked_2d_parameters(self, mock_model_parameters):
        """Fira should create a projector only for ranked 2D parameter groups during step execution."""
        config = OptimizerConfig(
            optimizer_type="Fira",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=[
                "weight_decay=0.01",
                "rank=4",
                "update_proj_gap=1",
                "scale=0.5",
                "projection_type='std'",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            mock_model_parameters,
        )

        for parameter in mock_model_parameters:
            parameter.grad = torch.randn_like(parameter)

        optimizer.step()

        matrix_param = next(parameter for parameter in mock_model_parameters if parameter.ndim == 2)
        vector_param = next(parameter for parameter in mock_model_parameters if parameter.ndim == 1)

        assert "Fira" in optimizer_name
        assert "projector" in optimizer.state[matrix_param]
        assert "projector" not in optimizer.state[vector_param]
        assert optimizer.state[matrix_param]["projector"].rank == 4
        assert optimizer.state[matrix_param]["projector"].projection_type == "std"

    def test_registered_compassplus_initializes_lookahead_state_on_first_step(self, mock_model_parameters):
        """CompassPlus should initialize lookahead state and advance its lookahead counter on step."""
        config = OptimizerConfig(
            optimizer_type="CompassPlus",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=[
                "weight_decay=0.01",
                "use_lookahead=True",
                "lookahead_merge_time=2",
                "use_softplus=False",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            mock_model_parameters,
        )

        for parameter in mock_model_parameters:
            parameter.grad = torch.randn_like(parameter)

        optimizer.step()

        first_parameter = optimizer.param_groups[0]["params"][0]
        state = optimizer.state[first_parameter]

        assert "CompassPlus" in optimizer_name
        assert optimizer.lookahead_step == 1
        assert "lookahead_params" in state
        assert state["lookahead_params"].shape == first_parameter.shape

    def test_registered_compassadopt_initializes_factored_second_moment_state(self):
        """CompassADOPT should factor second-moment state for matrix parameters on the first optimization step."""
        matrix_param = torch.nn.Parameter(torch.randn(64, 64))
        vector_param = torch.nn.Parameter(torch.randn(16))
        parameters = [matrix_param, vector_param]

        config = OptimizerConfig(
            optimizer_type="CompassADOPT",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=[
                "weight_decay=0.01",
                "weight_decouple=True",
                "factor_second_moment=True",
                "use_stable_spam_clipping=False",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            parameters,
        )

        for parameter in parameters:
            parameter.grad = torch.randn_like(parameter)

        optimizer.step()

        assert "CompassADOPT" in optimizer_name
        assert isinstance(optimizer.state[matrix_param]["exp_avg_sq"], list)
        assert len(optimizer.state[matrix_param]["exp_avg_sq"]) == 5
        assert isinstance(optimizer.state[vector_param]["exp_avg_sq"], torch.Tensor)
        assert optimizer.state[matrix_param]["exp_avg"].shape == matrix_param.shape

    def test_registered_compassadoptmars_initializes_previous_grad_state(self):
        """CompassADOPTMARS should initialize factored second-moment and previous-grad state on the first step."""
        matrix_param = torch.nn.Parameter(torch.randn(64, 64))
        vector_param = torch.nn.Parameter(torch.randn(16))
        parameters = [matrix_param, vector_param]

        config = OptimizerConfig(
            optimizer_type="CompassADOPTMARS",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=[
                "weight_decay=0.02",
                "weight_decouple=True",
                "factor_second_moment=True",
                "gamma=0.05",
                "cautious=False",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            parameters,
        )

        for parameter in parameters:
            parameter.grad = torch.randn_like(parameter)

        optimizer.step()

        matrix_state = optimizer.state[matrix_param]
        vector_state = optimizer.state[vector_param]

        assert "CompassADOPTMARS" in optimizer_name
        assert isinstance(matrix_state["exp_avg_sq"], list)
        assert len(matrix_state["exp_avg_sq"]) == 5
        assert isinstance(vector_state["exp_avg_sq"], torch.Tensor)
        assert matrix_state["previous_grad"].shape == matrix_param.shape
        assert torch.count_nonzero(matrix_state["previous_grad"]).item() > 0

    def test_registered_soap_initializes_preconditioner_state_on_first_step(self, mock_model_parameters):
        """SOAP should initialize preconditioner state on the first optimization step."""
        config = OptimizerConfig(
            optimizer_type="SOAP",
            learning_rates=LearningRatesConfig(base=3e-3),
            optimizer_args=[
                "weight_decay=0.02",
                "precondition_frequency=2",
                "max_precondition_dim=32",
                "merge_dims=False",
                "precondition_1d=True",
                "correct_bias=False",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            mock_model_parameters,
        )

        for parameter in mock_model_parameters:
            parameter.grad = torch.randn_like(parameter)

        optimizer.step()

        matrix_param = next(parameter for parameter in mock_model_parameters if parameter.ndim == 2)
        vector_param = next(parameter for parameter in mock_model_parameters if parameter.ndim == 1)

        assert "SOAP" in optimizer_name
        assert optimizer.param_groups[0]["precondition_frequency"] == 2
        assert "GG" in optimizer.state[matrix_param]
        assert "Q" in optimizer.state[matrix_param]
        assert optimizer.state[matrix_param]["Q"] is not None
        assert len(optimizer.state[matrix_param]["GG"]) == matrix_param.ndim
        assert len(optimizer.state[vector_param]["GG"]) == 1

    def test_registered_ranger21_initializes_state_and_updates_internal_lr_on_first_step(self, mock_model_parameters):
        """Ranger21 should initialize optimizer state and advance its built-in LR schedule on step."""
        config = OptimizerConfig(
            optimizer_type="Ranger21",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=[
                "num_iterations=10",
                "weight_decay=0.01",
                "beta0=0.8",
                "lookahead_merge_time=3",
                "num_warm_up_iterations=5",
                "num_warm_down_iterations=1",
                "disable_lr_scheduler=False",
                "use_softplus=False",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            mock_model_parameters,
        )

        for parameter in optimizer.param_groups[0]["params"]:
            parameter.grad = torch.ones_like(parameter)

        optimizer.step()

        first_parameter = optimizer.param_groups[0]["params"][0]
        state = optimizer.state[first_parameter]

        assert "Ranger21" in optimizer_name
        assert state["lookahead_params"].shape == first_parameter.shape
        assert "grad_ma" in state
        assert "variance_ma" in state
        assert "neg_grad_ma" in state
        assert "max_variance_ma" in state
        assert optimizer.current_lr == pytest.approx(2e-5)
        assert optimizer.lookahead_step == 1

    def test_registered_scion_init_and_first_step_work_without_focus(self, mock_model_parameters):
        """SCION should initialize LMO state and take a first step through the default non-focus path."""
        config = OptimizerConfig(
            optimizer_type="SCION",
            learning_rates=LearningRatesConfig(base=1e-3),
            optimizer_args=[
                "momentum=0.15",
                "scale=1.5",
                "constraint=False",
                "weight_decay=0.01",
                "use_focus=False",
                "norm_type=1",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            mock_model_parameters,
        )

        first_parameter = optimizer.param_groups[0]["params"][0]
        optimizer.init()
        initial_parameter = first_parameter.detach().clone()

        for parameter in optimizer.param_groups[0]["params"]:
            parameter.grad = torch.randn_like(parameter)

        optimizer.step()

        state = optimizer.state[first_parameter]

        assert "SCION" in optimizer_name
        assert optimizer.param_groups[0]["step"] == 1
        assert "d" in state
        assert "pbar" not in state
        assert state["d"].shape == first_parameter.shape
        assert not torch.equal(first_parameter, initial_parameter)

    def test_registered_dehaze_initializes_stage_state_on_first_step(self, mock_model_parameters):
        """Dehaze should initialize its dual denominator and sign state on the first optimization step."""
        config = OptimizerConfig(
            optimizer_type="Dehaze",
            learning_rates=LearningRatesConfig(base=1e-3),
            optimizer_args=[
                "adaptive_muon=True",
                "stage1_atan2=False",
                "stage2_atan2=True",
                "stochastic_fp=False",
                "torch_compile=False",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            mock_model_parameters,
        )

        for parameter in optimizer.param_groups[0]["params"]:
            parameter.grad = torch.randn_like(parameter)

        optimizer.step()

        first_parameter = optimizer.param_groups[0]["params"][0]
        state = optimizer.state[first_parameter]

        assert "Dehaze" in optimizer_name
        assert optimizer.param_groups[0]["step"] == 1
        assert "stage1_emasq" in state
        assert "stage2_emasq" in state
        assert "sign_momentum" in state
        assert state["stage1_emasq"].shape == first_parameter.shape
        assert state["stage2_emasq"].shape == first_parameter.shape
        assert state["sign_momentum"].shape == first_parameter.shape

    def test_registered_gooddog_initializes_dual_denom_state_on_first_step(self, mock_model_parameters):
        """GOODDOG should initialize both denominator buffers and sign momentum on the first optimization step."""
        config = OptimizerConfig(
            optimizer_type="GOODDOG",
            learning_rates=LearningRatesConfig(base=1e-3),
            optimizer_args=[
                "invariant=True",
                "adaptive_muon=True",
                "orthograd=True",
                "stochastic_fp=False",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            mock_model_parameters,
        )

        for parameter in optimizer.param_groups[0]["params"]:
            parameter.grad = torch.randn_like(parameter)

        optimizer.step()

        first_parameter = optimizer.param_groups[0]["params"][0]
        state = optimizer.state[first_parameter]

        assert "GOODDOG" in optimizer_name
        assert optimizer.param_groups[0]["step"] == 1
        assert "stage1_emasq" in state
        assert "stage2_emasq" in state
        assert "sign_momentum" in state
        assert state["stage1_emasq"].shape == first_parameter.shape
        assert state["stage2_emasq"].shape == first_parameter.shape
        assert state["sign_momentum"].shape == first_parameter.shape

    def test_registered_stablespam_resets_projection_buffers_on_gap_for_float32_params(self):
        """StableSPAM should write reset projection buffers back into float32 state when the gap triggers."""
        parameter = torch.nn.Parameter(torch.zeros(4, 4))
        parameters = [parameter]

        config = OptimizerConfig(
            optimizer_type="StableSPAM",
            learning_rates=LearningRatesConfig(base=1e-3),
            optimizer_args=[
                "weight_decay=0.0",
                "update_proj_gap=2",
                "use_adopt=False",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            parameters,
        )

        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        first_step_exp_avg = optimizer.state[parameter]["exp_avg"].clone()

        parameter.grad = torch.arange(1, 17, dtype=parameter.dtype).view_as(parameter)
        optimizer.step()

        state = optimizer.state[parameter]

        assert "StableSPAM" in optimizer_name
        assert optimizer.total_step == 2
        assert state["step"] == 1
        assert not torch.allclose(state["exp_avg"], first_step_exp_avg)
        assert state["exp_avg"].std().item() > 0.0
        assert state["exp_avg_sq"].std().item() > 0.0

    def test_registered_mythical_initializes_running_state_on_first_step(self, mock_model_parameters):
        """Mythical should initialize EMA, squared EMA, and previous-gradient state on the first optimization step."""
        config = OptimizerConfig(
            optimizer_type="Mythical",
            learning_rates=LearningRatesConfig(base=1e-3),
            optimizer_args=[
                "amp=1.5",
                "orthograd=True",
                "adaptive_ema=True",
                "atan2=True",
                "warmup=True",
                "cautious_min=0.25",
                "stochastic_fp=False",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            mock_model_parameters,
        )

        for parameter in optimizer.param_groups[0]["params"]:
            parameter.grad = torch.randn_like(parameter)

        optimizer.step()

        first_parameter = optimizer.param_groups[0]["params"][0]
        state = optimizer.state[first_parameter]

        assert "Mythical" in optimizer_name
        assert optimizer.param_groups[0]["step"] == 1
        assert "ema" in state
        assert "ema_squared" in state
        assert "prev_grad" in state
        assert state["ema"].shape == first_parameter.shape
        assert state["ema_squared"].shape == first_parameter.shape
        assert state["prev_grad"].shape == first_parameter.shape

    def test_registered_scalable_shampoo_initializes_preconditioner_and_graft_state(self, mock_model_parameters):
        """ScalableShampoo should initialize its preconditioner and graft state on the first optimization step."""
        config = OptimizerConfig(
            optimizer_type="ScalableShampoo",
            learning_rates=LearningRatesConfig(base=1e-3),
            optimizer_args=[
                "weight_decay=0.01",
                "block_size=32",
                "start_preconditioning_step=1",
                "preconditioning_compute_steps=1",
                "statistics_compute_steps=1",
                "graft_type=2",
                "use_svd=False",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            mock_model_parameters,
        )

        for parameter in optimizer.param_groups[0]["params"]:
            parameter.grad = torch.randn_like(parameter)

        optimizer.step()

        matrix_param = next(parameter for parameter in optimizer.param_groups[0]["params"] if parameter.ndim == 2)
        state = optimizer.state[matrix_param]
        pre_conditioner = state["pre_conditioner"]

        assert "ScalableShampoo" in optimizer_name
        assert "momentum" in state
        assert "graft" in state
        assert pre_conditioner.rank >= 1
        assert len(pre_conditioner.statistics) == len(pre_conditioner.pre_conditioners)
        assert len(pre_conditioner.pre_conditioners) > 0

    @pytest.mark.parametrize(
        ("optimizer_type", "optimizer_args"),
        [
            ("ADOPTScheduleFree", ["weight_decay=0.01", "r=0.5", "weight_lr_power=1.5", "adaptive_clip=0.0", "debias_beta2=True"]),
            ("ADOPTEMAMixScheduleFree", ["weight_decay=0.01", "adaptive_clip=0.0", "alpha=4.0", "t_alpha_beta3=20"]),
            ("ADOPTNesterovScheduleFree", ["weight_decay=0.01", "adaptive_clip=0.0", "debias_beta3=True"]),
            ("ADOPTMARSScheduleFree", ["weight_decay=0.01", "gamma=0.05", "adaptive_clip=0.0"]),
            (
                "ADOPTAOScheduleFree",
                ["weight_decay=0.01", "state_precision='parameter'", "block_size=0", "adaptive_clip=0.0", "torch_compile=False"],
            ),
            ("FADOPTScheduleFree", ["weight_decay=0.01", "fisher_clip=0.5", "adaptive_clip=0.0"]),
            ("FADOPTEMAMixScheduleFree", ["weight_decay=0.01", "fisher_clip=0.5", "alpha=4.0", "t_alpha_beta3=20"]),
            ("FADOPTNesterovScheduleFree", ["weight_decay=0.01", "fisher_clip=0.5", "adaptive_clip=0.0", "debias_beta3=True"]),
            ("FADOPTMARSScheduleFree", ["weight_decay=0.01", "fisher_clip=0.5", "gamma=0.05", "adaptive_clip=0.0"]),
        ],
    )
    def test_repo_owned_schedulefree_leaf_builds_and_uses_train_eval_helpers(self, mock_model_parameters, optimizer_type, optimizer_args):
        """Repo-owned schedule-free leaves should participate in train/eval handling via capability metadata."""
        config = OptimizerConfig(
            optimizer_type=optimizer_type,
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=optimizer_args,
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            mock_model_parameters,
        )

        assert optimizer_type in optimizer_name
        assert optimizer.param_groups[0]["weight_decay"] == 0.01
        assert is_schedulefree_optimizer(optimizer, config)
        train_fn, eval_fn = get_optimizer_train_eval_fn(optimizer, config)
        assert callable(train_fn)
        assert callable(eval_fn)


@pytest.mark.training
@pytest.mark.unit
class TestAbsorbedSchedulers:
    @pytest.mark.parametrize(
        "optimizer_type",
        [
            "ADOPTScheduleFree",
            "ADOPTEMAMixScheduleFree",
            "ADOPTNesterovScheduleFree",
            "ADOPTMARSScheduleFree",
            "ADOPTAOScheduleFree",
            "FADOPTScheduleFree",
            "FADOPTEMAMixScheduleFree",
            "FADOPTNesterovScheduleFree",
            "FADOPTMARSScheduleFree",
        ],
    )
    def test_repo_owned_schedulefree_leaf_uses_dummy_scheduler(self, mock_model_parameters, optimizer_type):
        """Schedule-free leaf optimizers should route through the dummy scheduler path."""
        optimizer_config, training_config, optimizer = _build_optimizer_and_training_config(
            mock_model_parameters,
            optimizer_type=optimizer_type,
            scheduler_config=SchedulerConfig(lr_scheduler="constant_with_warmup", lr_warmup_steps=5),
        )

        scheduler = get_scheduler_fix(
            optimizer_config.scheduler,
            optimizer_config,
            training_config,
            optimizer,
            num_processes=1,
        )

        assert scheduler.__class__.__name__ == "DummyScheduler"
        assert scheduler.optimizer is optimizer

    def test_cosine_warm_restarts_constructs_through_registry(self, mock_model_parameters):
        """Repo-owned warm-restart schedulers should build through the shared registry path."""
        optimizer_config, training_config, optimizer = _build_optimizer_and_training_config(
            mock_model_parameters,
            scheduler_config=SchedulerConfig(
                lr_scheduler="CosineAnnealingWarmRestarts",
                lr_scheduler_args=[
                    "gamma=0.9",
                    "min_lr=1e-6",
                    "warmup_steps=2",
                    "first_cycle_max_steps=8",
                ],
            ),
        )

        scheduler = get_scheduler_fix(
            optimizer_config.scheduler,
            optimizer_config,
            training_config,
            optimizer,
            num_processes=1,
        )

        assert isinstance(scheduler, CosineAnnealingWarmRestarts)
        assert optimizer.param_groups[0]["warmup_steps"] == 2
        assert optimizer.param_groups[0]["current_cycle_max_steps"] == 8

    def test_rex_warm_restarts_constructs_through_registry(self, mock_model_parameters):
        """Second repo-owned scheduler absorption should reuse the same scheduler registration flow."""
        optimizer_config, training_config, optimizer = _build_optimizer_and_training_config(
            mock_model_parameters,
            scheduler_config=SchedulerConfig(
                lr_scheduler="RexAnnealingWarmRestarts",
                lr_scheduler_args=[
                    "gamma=0.9",
                    "min_lr=1e-6",
                    "warmup_steps=1",
                    "first_cycle_max_steps=6",
                    "d=0.85",
                ],
            ),
        )

        scheduler = get_scheduler_fix(
            optimizer_config.scheduler,
            optimizer_config,
            training_config,
            optimizer,
            num_processes=1,
        )

        assert isinstance(scheduler, RexAnnealingWarmRestarts)
        assert scheduler.d == 0.85
        assert optimizer.param_groups[0]["warmup_steps"] == 1

    def test_builtin_cosineannealinglr_still_constructs_through_registry(self, mock_model_parameters):
        """The broader scheduler split should not disturb the built-in torch scheduler path."""
        optimizer_config, training_config, optimizer = _build_optimizer_and_training_config(
            mock_model_parameters,
            scheduler_config=SchedulerConfig(lr_scheduler="CosineAnnealingLR", lr_scheduler_args=["min_lr=1e-6"]),
        )

        scheduler = get_scheduler_fix(
            optimizer_config.scheduler,
            optimizer_config,
            training_config,
            optimizer,
            num_processes=2,
        )

        assert isinstance(scheduler, CosineAnnealingLR)
        assert scheduler.T_max == 50
