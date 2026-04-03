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
        pytest.importorskip("schedulefree")
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
        pytest.importorskip("schedulefree")
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
                "SGDSaI",
                1e-3,
                ["momentum=0.8", "weight_decay=0.01", "cautious=True"],
                "SGDSaI",
                {"momentum": 0.8, "weight_decay": 0.01, "cautious": True},
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
        if optimizer_type == "FCompassPlus":
            assert optimizer.use_lookahead is True
        if optimizer_type == "LaProp":
            assert optimizer.cautious is True
        if optimizer_type == "Lamb":
            assert optimizer.pre_norm is True
        if optimizer_type == "SGDSaI":
            assert optimizer.has_warmup is False
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

    @pytest.mark.parametrize(
        ("optimizer_type", "optimizer_args"),
        [
            ("ADOPTScheduleFree", ["weight_decay=0.01", "r=0.5", "weight_lr_power=1.5", "adaptive_clip=0.0", "debias_beta2=True"]),
            ("ADOPTEMAMixScheduleFree", ["weight_decay=0.01", "adaptive_clip=0.0", "alpha=4.0", "t_alpha_beta3=20"]),
            ("ADOPTNesterovScheduleFree", ["weight_decay=0.01", "adaptive_clip=0.0", "debias_beta3=True"]),
            ("ADOPTMARSScheduleFree", ["weight_decay=0.01", "gamma=0.05", "adaptive_clip=0.0"]),
            ("ADOPTAOScheduleFree", ["weight_decay=0.01", "state_precision='parameter'", "block_size=0", "adaptive_clip=0.0", "torch_compile=False"]),
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
