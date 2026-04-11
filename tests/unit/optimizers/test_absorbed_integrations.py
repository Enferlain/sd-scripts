import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from unittest.mock import patch

from library.config.dataclasses.optimizer import LearningRatesConfig, OptimizerConfig, SchedulerConfig
from library.config.dataclasses.training import TrainingConfig
from library.optimization.optimizer_factory import get_optimizer
from library.optimization.optimizer_utils import get_optimizer_train_eval_fn, is_schedulefree_optimizer
from library.optimization.scheduler import get_scheduler_fix
from library.optimization.schedulers import CosineAnnealingWarmRestarts, RexAnnealingWarmRestarts
from library.optimization.types import build_module_parameter_group


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
    class _FakeTorchAOCPUOffloadOptimizer:
        def __init__(
            self,
            params,
            optimizer_class=torch.optim.AdamW,
            *,
            offload_gradients=False,
            minimal_size=4096,
            **kwargs,
        ):
            self.param_groups = list(params)
            self.optimizer_class = optimizer_class
            self.offload_gradients = offload_gradients
            self.minimal_size = minimal_size
            self.kwargs = kwargs
            self.state = {}
            self.defaults = {}

        def add_param_group(self, param_group):
            self.param_groups.append(param_group)

        def load_state_dict(self, state_dict):
            self._loaded_state_dict = state_dict

        def state_dict(self):
            return {"offloaded": []}

        def zero_grad(self, set_to_none=True):
            self._zero_grad_called = set_to_none

        def step(self, closure=None):
            self._step_called = True
            return closure() if closure is not None else None

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

    def test_registered_cpu_offload_wrapper_builds_from_base_optimizer(self, mock_model_parameters):
        """CPU offload wrapper should rebuild the configured base optimizer through the shared wrapper path."""
        config = OptimizerConfig(
            optimizer_type="CPUOffloadOptimizer",
            learning_rates=LearningRatesConfig(base=3e-4),
            optimizer_args=[
                "base_optimizer_type=AdamW",
                "base_optimizer.weight_decay=0.01",
                "offload_gradients=True",
                "minimal_size=2048",
            ],
        )

        with (
            patch("library.optimization.wrappers.cpu_offload.get_available_devices", return_value=["cuda"]),
            patch(
                "library.optimization.wrappers.cpu_offload.TorchAOCPUOffloadOptimizer",
                self._FakeTorchAOCPUOffloadOptimizer,
            ),
        ):
            optimizer_name, _, optimizer = get_optimizer(
                config,
                config.learning_rates,
                config.scheduler,
                mock_model_parameters,
            )

        assert "CPUOffloadOptimizerWrapper" in optimizer_name
        assert str(optimizer) == "CPUOffloadOptimizer"
        assert optimizer.base_optimizer.optimizer_class.__name__ == "AdamW"
        assert optimizer.base_optimizer.kwargs["weight_decay"] == 0.01
        assert optimizer.base_optimizer.offload_gradients is True
        assert optimizer.base_optimizer.minimal_size == 2048
        assert optimizer.param_groups[0]["weight_decay"] == 0.01

    def test_registered_cpu_offload_wrapper_schedules_wrapper_directly(self, mock_model_parameters):
        """CPU offload wrapper should be scheduled directly rather than routing to a nested base optimizer."""
        optimizer_config = OptimizerConfig(
            optimizer_type="CPUOffloadOptimizer",
            learning_rates=LearningRatesConfig(base=3e-4),
            optimizer_args=[
                "base_optimizer_type=AdamW",
                "base_optimizer.weight_decay=0.01",
                "minimal_size=2048",
            ],
            scheduler=SchedulerConfig(lr_scheduler="constant_with_warmup", lr_warmup_steps=5),
        )
        training_config = TrainingConfig(max_train_steps=25)

        with (
            patch("library.optimization.wrappers.cpu_offload.get_available_devices", return_value=["cuda"]),
            patch(
                "library.optimization.wrappers.cpu_offload.TorchAOCPUOffloadOptimizer",
                self._FakeTorchAOCPUOffloadOptimizer,
            ),
        ):
            _, _, optimizer = get_optimizer(
                optimizer_config,
                optimizer_config.learning_rates,
                optimizer_config.scheduler,
                mock_model_parameters,
            )

            scheduler = get_scheduler_fix(
                optimizer_config.scheduler,
                optimizer_config,
                training_config,
                optimizer,
                num_processes=1,
            )

        assert scheduler.optimizer is optimizer

    def test_registered_cpu_offload_wrapper_requires_cuda_or_xpu(self, mock_model_parameters):
        """CPU offload wrapper should fail fast when no supported accelerator runtime is available."""
        config = OptimizerConfig(
            optimizer_type="CPUOffloadOptimizer",
            learning_rates=LearningRatesConfig(base=3e-4),
            optimizer_args=[
                "base_optimizer_type=AdamW",
                "minimal_size=2048",
            ],
        )

        with (
            patch("library.optimization.wrappers.cpu_offload.get_available_devices", return_value=["cpu"]),
            patch(
                "library.optimization.wrappers.cpu_offload.TorchAOCPUOffloadOptimizer",
                self._FakeTorchAOCPUOffloadOptimizer,
            ),
            pytest.raises(RuntimeError, match="CUDA or XPU"),
        ):
            get_optimizer(
                config,
                config.learning_rates,
                config.scheduler,
                mock_model_parameters,
            )

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
                "ABMOG",
                1e-4,
                [
                    "weight_decay=0.01",
                    "weight_decay_rate=0.99",
                    "adaptive=True",
                    "bcos=False",
                    "abm_order=3",
                    "abm_k=2",
                    "state_storage_dtype='float32'",
                    "state_storage_device='cpu'",
                ],
                "ABMOG",
                {
                    "weight_decay": 0.01,
                    "weight_decay_rate": 0.99,
                    "adaptive": True,
                    "bcos": False,
                    "abm_order": 3,
                    "abm_k": 2,
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
                "Compass8BitBNB",
                1e-4,
                [
                    "weight_decay=0.01",
                    "weight_decouple=True",
                    "clip=0.5",
                    "centralization=0.5",
                    "quantization_group_size=64",
                ],
                "Compass8BitBNB",
                {
                    "weight_decay": 0.01,
                    "weight_decouple": True,
                    "clip": 0.5,
                    "centralization": 0.5,
                    "group_size": 64,
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
                "CompassAO",
                1e-4,
                [
                    "weight_decay=0.01",
                    "weight_decouple=True",
                    "state_precision='parameter'",
                    "torch_compile=False",
                    "use_spam_clipping=False",
                ],
                "CompassAO",
                {
                    "weight_decay": 0.01,
                    "weight_decouple": True,
                    "use_spam_clipping": False,
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
                "GrokFastAdamW",
                3e-4,
                [
                    "weight_decay=0.01",
                    "weight_decouple=False",
                    "fixed_decay=True",
                    "grokfast=True",
                    "grokfast_alpha=0.95",
                    "grokfast_lamb=1.5",
                    "grokfast_after_step=1",
                    "eps=1e-7",
                ],
                "GrokFastAdamW",
                {
                    "weight_decay": 0.01,
                    "weight_decouple": False,
                    "fixed_decay": True,
                    "grokfast": True,
                    "grokfast_alpha": 0.95,
                    "grokfast_lamb": 1.5,
                    "grokfast_after_step": 1,
                    "eps": 1e-7,
                },
            ),
            (
                "Glyph",
                1e-4,
                [
                    "weight_decay=0.01",
                    "weight_decay_rate=0.99",
                    "amp=1.5",
                    "orthograd=True",
                    "adaptive_ema=True",
                    "atan2=True",
                    "cautious_min=0.25",
                    "stochastic_fp=False",
                ],
                "Glyph",
                {
                    "weight_decay": 0.01,
                    "weight_decay_rate": 0.99,
                    "amp": 1.5,
                    "orthograd": True,
                    "adaptive_ema": True,
                    "atan2": True,
                    "cautious_min": 0.25,
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
                "SingState",
                1e-4,
                [
                    "weight_decay=0.01",
                    "weight_decay_rate=0.99",
                    "spectral_clip=False",
                    "lowpass_grad=0.5",
                ],
                "SingState",
                {
                    "weight_decay": 0.01,
                    "weight_decay_rate": 0.99,
                    "spectral_clip": False,
                    "lowpass_grad": 0.5,
                },
            ),
            (
                "SCORN",
                1e-3,
                [
                    "focus_ratio=0.1",
                    "weight_decay=0.01",
                    "weight_decay_rate=0.99",
                    "amp=4.0",
                    "reset_interval=3",
                    "reset_increment=2",
                    "orthograd=True",
                    "spectral_update_scale=0.75",
                    "constrain=True",
                    "cautious_min=0.25",
                    "stochastic_fp=False",
                    "use_stable_spam_clipping=True",
                    "torch_compile=False",
                ],
                "SCORN",
                {
                    "focus_ratio": 0.1,
                    "weight_decay": 0.01,
                    "weight_decay_rate": 0.99,
                    "amp": 4.0,
                    "reset_interval": 3,
                    "reset_increment": 2,
                    "orthograd": True,
                    "spectral_update_scale": 0.75,
                    "constrain": True,
                    "cautious_min": 0.25,
                    "stochastic_fp": False,
                    "use_stable_spam_clipping": True,
                    "torch_compile": False,
                },
            ),
            (
                "SCORNMachina",
                6e-4,
                [
                    "focus_ratio=0.1",
                    "weight_decay=0.01",
                    "weight_decay_rate=0.99",
                    "amp=4.0",
                    "reset_interval=3",
                    "reset_increment=2",
                    "orthograd=True",
                    "orthograd_alpha=0.75",
                    "spectral_update_scale=0.5",
                    "constrain=True",
                    "cautious_min=0.25",
                    "stochastic_fp=False",
                    "use_stable_spam_clipping=True",
                    "eps=1e-8",
                    "eps2=1e-2",
                    "eps_floor=1e-16",
                    "use_adagc=True",
                    "adagc_warmup_steps=2",
                    "amsgrad=True",
                    "amsgrad_decay_rate=0.9",
                    "torch_compile=False",
                    "sync_chunk_size=16",
                    "state_storage_dtype='float32'",
                    "state_storage_device='cpu'",
                ],
                "SCORNMachina",
                {
                    "focus_ratio": 0.1,
                    "weight_decay": 0.01,
                    "weight_decay_rate": 0.99,
                    "amp": 4.0,
                    "reset_interval": 3,
                    "reset_increment": 2,
                    "orthograd": True,
                    "orthograd_alpha": 0.75,
                    "spectral_update_scale": 0.5,
                    "constrain": True,
                    "cautious_min": 0.25,
                    "stochastic_fp": False,
                    "use_stable_spam_clipping": True,
                    "eps": 1e-8,
                    "eps2": 1e-2,
                    "eps_floor": 1e-16,
                    "amsgrad": True,
                    "amsgrad_decay_rate": 0.9,
                    "torch_compile": False,
                    "sync_chunk_size": 16,
                },
            ),
            (
                "CAME",
                5e-5,
                [
                    "weight_decay=0.01",
                    "weight_decouple=False",
                    "fixed_decay=True",
                    "clip_threshold=0.75",
                    "ams_bound=True",
                    "eps1=1e-20",
                    "eps2=1e-12",
                    "update_strategy='grams'",
                    "sync_chunk_size=16",
                    "state_storage_dtype='float32'",
                    "state_storage_device='cpu'",
                    "cautious_weight_decay=True",
                ],
                "CAME",
                {
                    "weight_decay": 0.01,
                    "weight_decouple": False,
                    "fixed_decay": True,
                    "ams_bound": True,
                    "eps1": 1e-20,
                    "eps2": 1e-12,
                    "update_strategy": "grams",
                    "sync_chunk_size": 16,
                    "cautious_weight_decay": True,
                },
            ),
            (
                "BCOS",
                1e-4,
                [
                    "beta=0.95",
                    "beta2=0.98",
                    "eps=1e-8",
                    "weight_decay=0.02",
                    "mode='m'",
                    "decouple_wd=False",
                    "simple_cond=True",
                    "sync_chunk_size=16",
                    "state_storage_dtype='float32'",
                    "state_storage_device='cpu'",
                ],
                "BCOS",
                {
                    "beta": 0.95,
                    "beta2": 0.98,
                    "eps": 1e-8,
                    "wd": 0.02,
                    "sync_chunk_size": 16,
                    "state_storage_dtype": torch.float32,
                    "state_storage_device": "cpu",
                },
            ),
            (
                "OAGOpt",
                1e-4,
                [
                    "betas=(0.9, 0.95, 0.98)",
                    "weight_decay=0.01",
                    "weight_decay_rate=0.99",
                    "spectral_adaptive=False",
                    "spectral_clip_compile=False",
                    "spectral_clip_dtype='float32'",
                    "adaptive=False",
                    "input_norm=False",
                    "lowpass_grad=0.2",
                    "sim_match=True",
                    "cautious_min=0.2",
                    "sgd_nesterov=False",
                    "stochastic_fp=False",
                    "sync_chunk_size=16",
                    "state_storage_dtype='float32'",
                    "state_storage_device='cpu'",
                ],
                "OAGOpt",
                {
                    "betas": (0.9, 0.95, 0.98),
                    "weight_decay": 0.01,
                    "weight_decay_rate": 0.99,
                    "spectral_adaptive": False,
                    "spectral_clip_compile": False,
                    "spectral_clip_dtype": torch.float32,
                    "adaptive": False,
                    "input_norm": False,
                    "lowpass_grad": 0.2,
                    "sim_match": True,
                    "cautious_min": 0.2,
                    "sgd_nesterov": False,
                    "stochastic_fp": False,
                    "sync_chunk_size": 16,
                    "state_storage_dtype": torch.float32,
                    "state_storage_device": "cpu",
                },
            ),
            (
                "OCGOpt",
                1e-4,
                [
                    "betas=(0.9, 0.92, 0.97)",
                    "weight_decay=0.02",
                    "centralization=0.5",
                    "spectral_adaptive=False",
                    "spectral_clip_compile=False",
                    "spectral_clip_dtype='float32'",
                    "adaptive=False",
                    "input_norm=True",
                    "lowpass_grad=0.1",
                    "sim_match=True",
                    "cautious_min=0.1",
                    "stochastic_fp=False",
                    "kahan_summation=True",
                    "sync_chunk_size=16",
                    "state_storage_dtype='float32'",
                    "state_storage_device='cpu'",
                ],
                "OCGOpt",
                {
                    "betas": (0.9, 0.92, 0.97),
                    "weight_decay": 0.02,
                    "centralization": 0.5,
                    "spectral_adaptive": False,
                    "spectral_clip_compile": False,
                    "spectral_clip_dtype": torch.float32,
                    "adaptive": False,
                    "input_norm": True,
                    "lowpass_grad": 0.1,
                    "sim_match": True,
                    "cautious_min": 0.1,
                    "stochastic_fp": False,
                    "kahan_summation": True,
                    "sync_chunk_size": 16,
                    "state_storage_dtype": torch.float32,
                    "state_storage_device": "cpu",
                },
            ),
            (
                "SCGOpt",
                1e-4,
                [
                    "betas=(0.9, 0.92, 0.97)",
                    "weight_decay=0.02",
                    "centralization=0.5",
                    "spectral_clip=True",
                    "spectral_adaptive=False",
                    "spectral_clip_compile=False",
                    "spectral_clip_dtype='float32'",
                    "adaptive=False",
                    "use_sign=False",
                    "lowpass_grad=0.1",
                    "sim_match=True",
                    "cautious_min=0.1",
                    "stochastic_fp=False",
                ],
                "SCGOpt",
                {
                    "betas": (0.9, 0.92, 0.97),
                    "weight_decay": 0.02,
                    "centralization": 0.5,
                    "spectral_clip": True,
                    "spectral_adaptive": False,
                    "spectral_clip_compile": False,
                    "spectral_clip_dtype": torch.float32,
                    "adaptive": False,
                    "use_sign": False,
                    "lowpass_grad": 0.1,
                    "sim_match": True,
                    "cautious_min": 0.1,
                    "stochastic_fp": False,
                },
            ),
            (
                "CStableAdamW",
                1e-3,
                [
                    "weight_decay=0.01",
                    "weight_decouple=False",
                    "eps=1e-12",
                    "use_rms=True",
                    "use_atan2=True",
                    "atan2_a=1.1",
                    "atan2_b=0.9",
                    "cautious_factor=0.5",
                    "use_adopt=True",
                    "use_stable_spam_clipping=True",
                    "ssc_scale=0.8",
                    "ssc_gamma1=0.81",
                    "ssc_gamma2=0.9999",
                    "ssc_gamma3=0.99",
                    "ssc_eps_floor=1e-20",
                    "torch_compile=False",
                ],
                "CStableAdamW",
                {
                    "weight_decay": 0.01,
                    "weight_decouple": False,
                    "eps": 1e-12,
                    "use_rms": True,
                    "use_atan2": True,
                    "atan2_a": 1.1,
                    "atan2_b": 0.9,
                    "cautious_factor": 0.5,
                    "use_adopt": True,
                    "use_stable_spam_clipping": True,
                    "ssc_scale": 0.8,
                    "ssc_gamma1": 0.81,
                    "ssc_gamma2": 0.9999,
                    "ssc_gamma3": 0.99,
                    "ssc_eps_floor": 1e-20,
                    "torch_compile": False,
                },
            ),
            (
                "FFTDescent",
                1e-4,
                [
                    "beta=0.9",
                    "weight_decay=0.01",
                    "weight_decay_rate=0.99",
                    "spectral_clip=True",
                    "spectral_clip_compile=False",
                    "spectral_clip_dtype='float32'",
                    "spectral_min=-0.5",
                    "spectral_max=0.75",
                    "spectral_adaptive=True",
                    "lowpass_grad=0.5",
                    "sign_momentum=0.8",
                    "stochastic_fp=False",
                ],
                "FFTDescent",
                {
                    "beta": 0.9,
                    "weight_decay": 0.01,
                    "weight_decay_rate": 0.99,
                    "spectral_clip": True,
                    "spectral_clip_compile": False,
                    "spectral_clip_dtype": torch.float32,
                    "spectral_min": -0.5,
                    "spectral_max": 0.75,
                    "spectral_adaptive": True,
                    "lowpass_grad": 0.5,
                    "sign_momentum": 0.8,
                    "stochastic_fp": False,
                },
            ),
            (
                "FARMSCrop",
                1e-4,
                [
                    "betas=(0.9, 0.99)",
                    "weight_decay=0.01",
                    "centralization=0.5",
                    "diff_mult=1.25",
                    "momentum_beta=0.95",
                    "momentum_amp=3.0",
                    "eps=1e-7",
                    "eps2=0.02",
                    "eps_floor=1e-16",
                ],
                "FARMSCrop",
                {
                    "betas": (0.9, 0.99),
                    "weight_decay": 0.01,
                    "centralization": 0.5,
                    "diff_mult": 1.25,
                    "momentum_beta": 0.95,
                    "momentum_amp": 3.0,
                    "eps": 1e-7,
                    "eps2": 0.02,
                    "eps_floor": 1e-16,
                },
            ),
            (
                "FARMSCropV2",
                1e-4,
                [
                    "betas=(0.9, 0.99)",
                    "weight_decay=0.01",
                    "centralization=0.5",
                    "diff_mult=1.25",
                    "momentum_beta=0.95",
                    "momentum_lambda=0.35",
                    "clip=0.75",
                    "cautious=True",
                    "cautious_grad='approx_grad_nat'",
                    "eps=1e-7",
                    "eps2=0.02",
                    "eps_floor=1e-16",
                ],
                "FARMSCropV2",
                {
                    "betas": (0.9, 0.99),
                    "weight_decay": 0.01,
                    "centralization": 0.5,
                    "diff_mult": 1.25,
                    "momentum_beta": 0.95,
                    "momentum_lambda": 0.35,
                    "clip": 0.75,
                    "cautious": True,
                    "cautious_grad": "approx_grad_nat",
                    "eps": 1e-7,
                    "eps2": 0.02,
                    "eps_floor": 1e-16,
                },
            ),
            (
                "FMARSCrop",
                1e-4,
                [
                    "betas=(0.9, 0.99)",
                    "weight_decay=0.01",
                    "weight_decouple=True",
                    "centralization=0.5",
                    "moment_centralization=0.25",
                    "diff_mult=1.25",
                    "momentum_beta=0.95",
                    "momentum_lambda=0.35",
                    "gamma=0.01",
                    "clip=0.75",
                    "adaptive_clip=0.5",
                    "adaptive_clip_type='layer'",
                    "cautious=True",
                    "debias_beta2=False",
                    "eps=1e-7",
                    "eps2=0.02",
                    "eps_floor=1e-16",
                ],
                "FMARSCrop",
                {
                    "betas": (0.9, 0.99),
                    "weight_decay": 0.01,
                    "weight_decouple": True,
                    "centralization": 0.5,
                    "moment_centralization": 0.25,
                    "diff_mult": 1.25,
                    "momentum_beta": 0.95,
                    "momentum_lambda": 0.35,
                    "gamma": 0.01,
                    "clip": 0.75,
                    "adaptive_clip": 0.5,
                    "adaptive_clip_type": "layer",
                    "cautious": True,
                    "debias_beta2": False,
                    "eps": 1e-7,
                    "eps2": 0.02,
                    "eps_floor": 1e-16,
                },
            ),
            (
                "FMARSCropV2",
                1e-4,
                [
                    "betas=(0.9, 0.99)",
                    "weight_decay=0.01",
                    "centralization=0.5",
                    "moment_centralization=0.25",
                    "diff_mult=1.25",
                    "momentum_beta=0.95",
                    "momentum_lambda=0.35",
                    "gamma=0.01",
                    "clip=0.75",
                    "adaptive_clip=0.5",
                    "cautious=True",
                    "debias_beta2=False",
                    "eps=1e-7",
                    "eps2=0.02",
                    "eps_floor=1e-16",
                ],
                "FMARSCropV2",
                {
                    "betas": (0.9, 0.99),
                    "weight_decay": 0.01,
                    "centralization": 0.5,
                    "moment_centralization": 0.25,
                    "diff_mult": 1.25,
                    "momentum_beta": 0.95,
                    "momentum_lambda": 0.35,
                    "gamma": 0.01,
                    "clip": 0.75,
                    "adaptive_clip": 0.5,
                    "cautious": True,
                    "debias_beta2": False,
                    "eps": 1e-7,
                    "eps2": 0.02,
                    "eps_floor": 1e-16,
                },
            ),
            (
                "FMARSCropV2ExMachina",
                1e-4,
                [
                    "betas=(0.9, 0.99, 0.999)",
                    "weight_decay=0.01",
                    "weight_decouple=True",
                    "centralization=0.5",
                    "moment_centralization=0.25",
                    "diff_mult=1.25",
                    "momentum_lambda=0.35",
                    "gamma=0.01",
                    "clip=0.75",
                    "adaptive_clip=0.5",
                    "adaptive_clip_type='layer'",
                    "update_strategy='grams'",
                    "debias_beta1=True",
                    "debias_beta2=False",
                    "debias_beta3=True",
                    "eps=1e-7",
                    "eps2=0.02",
                    "eps_floor=1e-16",
                ],
                "FMARSCropV2ExMachina",
                {
                    "betas": (0.9, 0.99, 0.999),
                    "weight_decay": 0.01,
                    "weight_decouple": True,
                    "centralization": 0.5,
                    "moment_centralization": 0.25,
                    "diff_mult": 1.25,
                    "momentum_lambda": 0.35,
                    "gamma": 0.01,
                    "clip": 0.75,
                    "adaptive_clip": 0.5,
                    "adaptive_clip_type": "layer",
                    "update_strategy": "grams",
                    "debias_beta1": True,
                    "debias_beta2": False,
                    "debias_beta3": True,
                    "eps": 1e-7,
                    "eps2": 0.02,
                    "eps_floor": 1e-16,
                },
            ),
            (
                "FMARSCropV3",
                1e-4,
                [
                    "betas=(0.9, 0.95)",
                    "weight_decay=0.01",
                    "centralization=0.5",
                    "moment_centralization=0.25",
                    "diff_mult=1.25",
                    "momentum_lambda=1.5",
                    "gamma=0.01",
                    "clip_lambda=0.75",
                    "adaptive_clip=0.5",
                    "adaptive_clip_norm_type=False",
                    "cautious=True",
                    "eps=1e-7",
                    "eps2=0.02",
                    "eps_floor=1e-16",
                ],
                "FMARSCropV3",
                {
                    "betas": (0.9, 0.95),
                    "weight_decay": 0.01,
                    "centralization": 0.5,
                    "moment_centralization": 0.25,
                    "diff_mult": 1.25,
                    "momentum_lambda": 1.5,
                    "gamma": 0.01,
                    "clip_lambda": 0.75,
                    "adaptive_clip": 0.5,
                    "adaptive_clip_norm_type": False,
                    "cautious": True,
                    "eps": 1e-7,
                    "eps2": 0.02,
                    "eps_floor": 1e-16,
                },
            ),
            (
                "FMARSCropV3ExMachina",
                1e-4,
                [
                    "betas=(0.9, 0.95)",
                    "weight_decay=0.01",
                    "weight_decouple=True",
                    "centralization=0.5",
                    "moment_centralization=0.25",
                    "diff_mult=1.25",
                    "momentum_lambda=1.5",
                    "gamma=0.01",
                    "clip=0.75",
                    "adaptive_clip=0.5",
                    "adaptive_clip_type='layer'",
                    "update_strategy='both'",
                    "debias_beta1=True",
                    "debias_beta2=True",
                    "stable_update=True",
                    "atan2_denom=True",
                    "use_orthograd=True",
                    "eps=1e-7",
                    "eps2=0.02",
                    "eps_floor=1e-16",
                ],
                "FMARSCropV3ExMachina",
                {
                    "betas": (0.9, 0.95),
                    "weight_decay": 0.01,
                    "weight_decouple": True,
                    "centralization": 0.5,
                    "moment_centralization": 0.25,
                    "diff_mult": 1.25,
                    "momentum_lambda": 1.5,
                    "gamma": 0.01,
                    "clip": 0.75,
                    "adaptive_clip": 0.5,
                    "adaptive_clip_type": "layer",
                    "update_strategy": "both",
                    "debias_beta1": True,
                    "debias_beta2": True,
                    "stable_update": True,
                    "atan2_denom": True,
                    "use_orthograd": True,
                    "eps": 1e-7,
                    "eps2": 0.02,
                    "eps_floor": 1e-16,
                },
            ),
            (
                "FishMonger",
                1e-3,
                [
                    "betas=(0.8, 0.9, 0.99)",
                    "eps=1e-8",
                    "eps2=0.02",
                    "eps_floor=1e-16",
                    "weight_decay=0.01",
                    "clip=0.5",
                    "centralization=0.5",
                    "diff_amp=0.1",
                    "diff_amp_beta=0.95",
                ],
                "FishMonger",
                {
                    "betas": (0.8, 0.9, 0.99),
                    "eps": 1e-8,
                    "eps2": 0.02,
                    "eps_floor": 1e-16,
                    "weight_decay": 0.01,
                    "clip": 0.5,
                    "centralization": 0.5,
                    "diff_amp": 0.1,
                    "diff_amp_beta": 0.95,
                },
            ),
            (
                "FishMonger8BitBNB",
                1e-3,
                [
                    "betas=(0.8, 0.9, 0.99)",
                    "eps=1e-8",
                    "eps2=0.02",
                    "eps_floor=1e-16",
                    "weight_decay=0.01",
                    "clip=0.5",
                    "centralization=0.5",
                    "diff_amp=0.1",
                    "diff_amp_beta=0.95",
                    "quantization_group_size=64",
                ],
                "FishMonger8BitBNB",
                {
                    "betas": (0.8, 0.9, 0.99),
                    "eps": 1e-8,
                    "eps2": 0.02,
                    "eps_floor": 1e-16,
                    "weight_decay": 0.01,
                    "clip": 0.5,
                    "centralization": 0.5,
                    "diff_amp": 0.1,
                    "diff_amp_beta": 0.95,
                    "group_size": 64,
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
                "TALON",
                1e-4,
                [
                    "weight_decay=0.01",
                    "weight_decay_rate=0.99",
                    "denom_atan2=True",
                    "spectral_clip=False",
                    "signscale_power=1.5",
                ],
                "TALON",
                {
                    "weight_decay": 0.01,
                    "weight_decay_rate": 0.99,
                    "denom_atan2": True,
                    "spectral_clip": False,
                    "signscale_power": 1.5,
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
                "MomentusCaution",
                1e-4,
                [
                    "beta=0.85",
                    "momentum_beta=0.5",
                    "weight_decay=0.01",
                    "gamma_ratio=0.25",
                    "adaptive_clip=0.5",
                    "cautious=False",
                    "nesterov=True",
                ],
                "MomentusCaution",
                {
                    "beta": 0.85,
                    "momentum_beta": 0.5,
                    "weight_decay": 0.01,
                    "gamma_ratio": 0.25,
                    "adaptive_clip": 0.5,
                    "cautious": False,
                    "nesterov": True,
                },
            ),
            (
                "ProjectiveAdam",
                1e-4,
                [
                    "weight_decay=0.01",
                    "projection='hyperbolic'",
                    "input_norm=False",
                    "normuon=True",
                    "use_compile=False",
                    "ortho_dtype='float32'",
                    "stochastic_fp=False",
                    "sync_chunk_size=16",
                    "state_storage_dtype='float32'",
                    "state_storage_device='cpu'",
                ],
                "ProjectiveAdam",
                {
                    "weight_decay": 0.01,
                    "projection": "hyperbolic",
                    "input_norm": False,
                    "normuon": True,
                    "use_compile": False,
                    "ortho_dtype": torch.float32,
                    "stochastic_fp": False,
                    "sync_chunk_size": 16,
                    "state_storage_dtype": torch.float32,
                    "state_storage_device": "cpu",
                },
            ),
            (
                "REMASTER",
                1e-4,
                [
                    "weight_decay=0.01",
                    "weight_decay_rate=0.99",
                    "amp=3.0",
                    "reset_interval=2",
                    "reset_increment=1",
                    "orthograd=False",
                    "cautious_min=0.25",
                    "stochastic_fp=False",
                ],
                "REMASTER",
                {
                    "weight_decay": 0.01,
                    "weight_decay_rate": 0.99,
                    "amp": 3.0,
                    "reset_interval": 2,
                    "reset_increment": 1,
                    "orthograd": False,
                    "cautious_min": 0.25,
                    "stochastic_fp": False,
                },
            ),
            (
                "WiwiOpt",
                1e-4,
                [
                    "weight_decay=0.01",
                    "normuon=True",
                    "use_compile=False",
                    "ortho_dtype='float32'",
                    "stochastic_fp=False",
                    "dynamic_lr=True",
                    "dynamic_lr_boost=False",
                    "egd=True",
                    "egd_oja=True",
                ],
                "WiwiOpt",
                {
                    "weight_decay": 0.01,
                    "normuon": True,
                    "use_compile": False,
                    "ortho_dtype": torch.float32,
                    "stochastic_fp": False,
                    "dynamic_lr": True,
                    "dynamic_lr_boost": False,
                    "egd": True,
                    "egd_oja": True,
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
        if optimizer_type == "ABMOG":
            assert optimizer._init_lr == learning_rate
            assert optimizer.state_storage_dtype == torch.float32
            assert optimizer.state_storage_device == "cpu"
            assert str(optimizer) == "ABMOG"
        if optimizer_type == "CompassPlus":
            assert optimizer.use_lookahead is True
        if optimizer_type == "Compass8BitBNB":
            assert str(optimizer) == "Compass8BitBNB"
        if optimizer_type == "CompassAO":
            assert optimizer.block_size == 0
            assert optimizer.min_quant_size == 4096
            assert optimizer.state_precision == "parameter"
            assert optimizer.torch_compile is False
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
        if optimizer_type == "FFTDescent":
            assert optimizer._init_lr == learning_rate
            assert str(optimizer) == "FFTDescent"
        if optimizer_type == "FARMSCrop":
            assert optimizer.eps == 1e-7
            assert optimizer.eps2 == 0.02
            assert optimizer.eps_floor == 1e-16
            assert str(optimizer) == "FARMSCrop"
        if optimizer_type == "FARMSCropV2":
            assert optimizer.param_groups[0]["cautious"] is True
            assert optimizer.param_groups[0]["cautious_grad"] == "approx_grad_nat"
            assert str(optimizer) == "FARMSCropV2"
        if optimizer_type == "FMARSCrop":
            assert optimizer.param_groups[0]["weight_decouple"] is True
            assert optimizer.param_groups[0]["adaptive_clip_type"] == "layer"
            assert str(optimizer) == "FMARSCrop"
        if optimizer_type == "FMARSCropV2":
            assert optimizer.param_groups[0]["momentum_beta"] == 0.95
            assert optimizer.param_groups[0]["debias_beta2"] is False
            assert str(optimizer) == "FMARSCropV2"
        if optimizer_type == "FMARSCropV2ExMachina":
            assert optimizer.param_groups[0]["update_strategy"] == "grams"
            assert optimizer.param_groups[0]["debias_beta1"] is True
            assert optimizer.param_groups[0]["debias_beta3"] is True
            assert str(optimizer) == "FMARSCropV2ExMachina"
        if optimizer_type == "FMARSCropV3":
            assert optimizer.param_groups[0]["clip_lambda"] == 0.75
            assert optimizer.param_groups[0]["adaptive_clip_norm_type"] is False
            assert str(optimizer) == "FMARSCropV3"
        if optimizer_type == "FMARSCropV3ExMachina":
            assert optimizer.param_groups[0]["update_strategy"] == "both"
            assert optimizer.param_groups[0]["stable_update"] is True
            assert optimizer.param_groups[0]["atan2_denom"] is True
            assert optimizer.param_groups[0]["use_orthograd"] is True
            assert str(optimizer) == "FMARSCropV3ExMachina"
        if optimizer_type == "FishMonger":
            assert optimizer.eps == 1e-8
            assert optimizer.eps2 == 0.02
            assert optimizer.eps_floor == 1e-16
            assert str(optimizer) == "FishMonger"
        if optimizer_type == "FishMonger8BitBNB":
            assert optimizer.eps == 1e-8
            assert optimizer.eps2 == 0.02
            assert optimizer.eps_floor == 1e-16
            assert str(optimizer) == "FishMonger8BitBNB"
        if optimizer_type == "GOODDOG":
            assert optimizer._init_lr == learning_rate
        if optimizer_type == "Glyph":
            assert optimizer._init_lr == learning_rate
            assert str(optimizer) == "Glyph"
        if optimizer_type == "GrokFastAdamW":
            assert optimizer.param_groups[0]["lr"] == pytest.approx(learning_rate / 2.5)
        if optimizer_type == "SGDSaI":
            assert optimizer.has_warmup is False
        if optimizer_type == "Mythical":
            assert optimizer._init_lr == learning_rate
        if optimizer_type == "MomentusCaution":
            assert str(optimizer) == "MomentusCaution"
        if optimizer_type == "ProjectiveAdam":
            assert optimizer.state_storage_dtype == torch.float32
            assert optimizer.state_storage_device == "cpu"
        if optimizer_type == "REMASTER":
            assert optimizer._init_lr == learning_rate
            assert str(optimizer) == "REMASTER"
        if optimizer_type == "WiwiOpt":
            assert str(optimizer) == "WiwiOpt"
        if optimizer_type == "SCORN":
            assert optimizer._init_lr == learning_rate
        if optimizer_type == "SCORNMachina":
            assert optimizer._init_lr == learning_rate
            assert optimizer.use_adagc is True
            assert optimizer.state_storage_dtype == torch.float32
            assert optimizer.state_storage_device == "cpu"
        if optimizer_type == "CAME":
            assert optimizer.clip_threshold == 0.75
            assert optimizer.state_storage_dtype == torch.float32
            assert optimizer.state_storage_device == "cpu"
        if optimizer_type == "BCOS":
            assert optimizer.mode == "m"
            assert optimizer.decouple_wd is False
            assert optimizer.simple_cond is True
            assert optimizer.state_storage_dtype == torch.float32
            assert optimizer.state_storage_device == "cpu"
        if optimizer_type == "OAGOpt":
            assert optimizer.state_storage_dtype == torch.float32
            assert optimizer.state_storage_device == "cpu"
            assert str(optimizer) == "OAGOpt"
        if optimizer_type == "OCGOpt":
            assert optimizer.state_storage_dtype == torch.float32
            assert optimizer.state_storage_device == "cpu"
            assert optimizer._init_lr == learning_rate
        if optimizer_type == "SCGOpt":
            assert optimizer._init_lr == learning_rate
        if optimizer_type == "CStableAdamW":
            assert str(optimizer) == "CStableAdamW_with_SSC"
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
        if optimizer_type == "SingState":
            assert optimizer._init_lr == learning_rate
            assert str(optimizer) == "SingState"
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
        if optimizer_type == "TALON":
            assert optimizer._init_lr == learning_rate
            assert str(optimizer) == "TALON"
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

    def test_registered_compass8bitbnb_handles_runtime_support_expectations(self):
        """Compass8BitBNB should fail fast on unsupported CPU execution instead of crashing."""
        parameter = torch.nn.Parameter(torch.zeros(4, 4))
        parameters = [parameter]

        config = OptimizerConfig(
            optimizer_type="Compass8BitBNB",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=[
                "weight_decay=0.01",
                "weight_decouple=True",
                "clip=0.5",
                "centralization=0.5",
                "quantization_group_size=64",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            parameters,
        )

        assert "Compass8BitBNB" in optimizer_name
        parameter.grad = torch.arange(1, 17, dtype=parameter.dtype).view_as(parameter)

        if not torch.cuda.is_available():
            with pytest.raises(RuntimeError, match="requires CUDA-enabled bitsandbytes"):
                optimizer.step()
            return

        parameter = torch.nn.Parameter(torch.zeros(4, 4, device="cuda"))
        parameters = [parameter]
        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            parameters,
        )
        parameter.grad = torch.arange(1, 17, dtype=parameter.dtype, device=parameter.device).view_as(parameter)
        optimizer.step()

        state = optimizer.state[parameter]

        assert "Compass8BitBNB" in optimizer_name
        assert "ema" in state
        assert "ema_squared" in state
        assert isinstance(state["ema"], tuple)
        assert isinstance(state["ema_squared"], tuple)

    def test_registered_compassao_initializes_parameter_precision_state_on_first_step(self):
        """CompassAO should initialize repo-owned state cleanly when kept in parameter precision."""
        parameter = torch.nn.Parameter(torch.randn(8, 8))
        parameters = [parameter]

        config = OptimizerConfig(
            optimizer_type="CompassAO",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=[
                "weight_decay=0.01",
                "weight_decouple=True",
                "state_precision='parameter'",
                "torch_compile=False",
                "use_spam_clipping=False",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            parameters,
        )

        parameter.grad = torch.randn_like(parameter)
        optimizer.step()

        state = optimizer.state[parameter]

        assert "CompassAO" in optimizer_name
        assert "exp_avg" in state
        assert "exp_avg_sq" in state
        assert isinstance(state["exp_avg"], torch.Tensor)
        assert isinstance(state["exp_avg_sq"], torch.Tensor)
        assert state["exp_avg"].shape == parameter.shape

    def test_registered_compassao_quantized_state_requires_cuda(self):
        """CompassAO should fail fast when quantized state is requested for CPU parameters."""
        parameter = torch.nn.Parameter(torch.randn(64, 64))
        parameters = [parameter]

        config = OptimizerConfig(
            optimizer_type="CompassAO",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=[
                "weight_decay=0.01",
                "weight_decouple=True",
                "state_precision='q8bit'",
                "torch_compile=False",
                "use_spam_clipping=False",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            parameters,
        )

        assert "CompassAO" in optimizer_name
        parameter.grad = torch.randn_like(parameter)

        if torch.cuda.is_available():
            parameter = torch.nn.Parameter(torch.randn(64, 64, device="cuda"))
            parameters = [parameter]
            optimizer_name, _, optimizer = get_optimizer(
                config,
                config.learning_rates,
                config.scheduler,
                parameters,
            )
            parameter.grad = torch.randn_like(parameter)
            optimizer.step()
            state = optimizer.state[parameter]
            assert "exp_avg" in state
            return

        with pytest.raises(RuntimeError, match="quantized state requires CUDA"):
            optimizer.step()

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

    def test_registered_scorn_init_and_first_step_track_focus_and_reset_state(self, mock_model_parameters):
        """SCORN should tolerate init() and initialize focus/reset bookkeeping on the first optimization step."""
        config = OptimizerConfig(
            optimizer_type="SCORN",
            learning_rates=LearningRatesConfig(base=1e-3),
            optimizer_args=[
                "focus_ratio=0.1",
                "weight_decay=0.01",
                "amp=4.0",
                "reset_interval=2",
                "reset_increment=1",
                "orthograd=True",
                "spectral_update_scale=0.75",
                "constrain=True",
                "cautious_min=0.25",
                "stochastic_fp=False",
                "use_stable_spam_clipping=True",
                "torch_compile=False",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            mock_model_parameters,
        )

        optimizer.init()

        for parameter in optimizer.param_groups[0]["params"]:
            parameter.grad = torch.randn_like(parameter)

        optimizer.step()

        first_parameter = optimizer.param_groups[0]["params"][0]
        state = optimizer.state[first_parameter]

        assert "SCORN" in optimizer_name
        assert optimizer.param_groups[0]["step"] == 1
        assert "ema" in state
        assert "ema_squared" in state
        assert "pbar" in state
        assert "times_zero" in state
        assert "steps_since_reset" in state
        assert state["ema"].shape == first_parameter.shape
        assert state["ema_squared"].shape == first_parameter.shape
        assert state["pbar"].shape == first_parameter.shape
        assert state["times_zero"] == 0
        assert state["steps_since_reset"] == 2

    def test_registered_scornmachina_initializes_offloaded_state_and_adagc_tracking(self, mock_model_parameters):
        """SCORNMachina should initialize offloaded state buffers and AdaGC tracking on the first optimization step."""
        config = OptimizerConfig(
            optimizer_type="SCORNMachina",
            learning_rates=LearningRatesConfig(base=6e-4),
            optimizer_args=[
                "focus_ratio=0.1",
                "weight_decay=0.01",
                "amp=4.0",
                "reset_interval=2",
                "reset_increment=1",
                "orthograd=True",
                "orthograd_alpha=0.75",
                "spectral_update_scale=0.5",
                "constrain=True",
                "cautious_min=0.25",
                "stochastic_fp=False",
                "use_stable_spam_clipping=True",
                "eps=1e-8",
                "eps2=1e-2",
                "eps_floor=1e-16",
                "use_adagc=True",
                "adagc_warmup_steps=2",
                "amsgrad=True",
                "amsgrad_decay_rate=0.9",
                "torch_compile=False",
                "sync_chunk_size=16",
                "state_storage_dtype='float32'",
                "state_storage_device='cpu'",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            mock_model_parameters,
        )

        optimizer.init()

        for parameter in optimizer.param_groups[0]["params"]:
            parameter.grad = torch.randn_like(parameter)

        optimizer.step()

        first_parameter = optimizer.param_groups[0]["params"][0]
        state = optimizer.state[first_parameter]

        assert "SCORNMachina" in optimizer_name
        assert optimizer.param_groups[0]["step"] == 1
        assert optimizer._global_step == 1
        assert "ema" in state
        assert "ema_squared" in state
        assert "pbar" in state
        assert "times_zero" in state
        assert "steps_since_reset" in state
        assert "adagc_gamma" in state
        assert state["ema"].dtype == torch.float32
        assert state["ema_squared"].dtype == torch.float32
        assert state["pbar"].dtype == torch.float32
        assert state["ema"].device.type == "cpu"
        assert state["ema_squared"].device.type == "cpu"
        assert state["pbar"].device.type == "cpu"
        assert state["times_zero"] == 0
        assert state["steps_since_reset"] == 2

    def test_registered_came_initializes_factored_state_on_first_step(self):
        """CAME should initialize factored second-moment state and optional AMSBound storage on the first step."""
        parameter = torch.nn.Parameter(torch.zeros(4, 4))
        parameters = [parameter]

        config = OptimizerConfig(
            optimizer_type="CAME",
            learning_rates=LearningRatesConfig(base=5e-5),
            optimizer_args=[
                "weight_decay=0.01",
                "weight_decouple=False",
                "fixed_decay=True",
                "clip_threshold=0.75",
                "ams_bound=True",
                "eps1=1e-20",
                "eps2=1e-12",
                "update_strategy='grams'",
                "sync_chunk_size=16",
                "state_storage_dtype='float32'",
                "state_storage_device='cpu'",
                "cautious_weight_decay=True",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            parameters,
        )

        parameter.grad = torch.randn_like(parameter)
        optimizer.step()

        state = optimizer.state[parameter]

        assert "CAME" in optimizer_name
        assert optimizer.param_groups[0]["step"] == 1
        assert "exp_avg" in state
        assert "exp_avg_sq_row" in state
        assert "exp_avg_sq_col" in state
        assert "exp_avg_res_row" in state
        assert "exp_avg_res_col" in state
        assert "exp_avg_sq_hat" in state
        assert state["exp_avg"].dtype == torch.float32
        assert state["exp_avg"].device.type == "cpu"
        assert state["exp_avg_sq_row"].shape == (4,)
        assert state["exp_avg_sq_col"].shape == (4,)
        assert state["exp_avg_res_row"].shape == (4,)
        assert state["exp_avg_res_col"].shape == (4,)

    def test_registered_bcos_initializes_cpu_state_and_steps_without_cuda(self):
        """BCOS should keep CPU state-storage execution working even when CUDA is unavailable."""
        parameter = torch.nn.Parameter(torch.zeros(4, 4))
        parameters = [parameter]

        config = OptimizerConfig(
            optimizer_type="BCOS",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=[
                "beta=0.95",
                "beta2=0.98",
                "eps=1e-8",
                "weight_decay=0.02",
                "mode='m'",
                "decouple_wd=False",
                "simple_cond=True",
                "sync_chunk_size=16",
                "state_storage_dtype='float32'",
                "state_storage_device='cpu'",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            parameters,
        )

        parameter.grad = torch.randn_like(parameter)
        optimizer.step()

        state = optimizer.state[parameter]

        assert "BCOS" in optimizer_name
        assert optimizer.param_groups[0]["step"] == 1
        assert "m" in state
        assert "v" in state
        assert state["m"].dtype == torch.float32
        assert state["v"].dtype == torch.float32
        assert state["m"].device.type == "cpu"
        assert state["v"].device.type == "cpu"

    def test_registered_oagopt_initializes_offloaded_state_on_first_step(self):
        """OAGOpt should initialize its scalar state and value momentum on the first step."""
        parameter = torch.nn.Parameter(torch.zeros(4, 4))
        parameters = [parameter]

        config = OptimizerConfig(
            optimizer_type="OAGOpt",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=[
                "weight_decay=0.01",
                "spectral_clip_compile=False",
                "spectral_clip_dtype='float32'",
                "stochastic_fp=False",
                "sync_chunk_size=16",
                "state_storage_dtype='float32'",
                "state_storage_device='cpu'",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            parameters,
        )

        parameter.grad = torch.randn_like(parameter)
        optimizer.step()

        state = optimizer.state[parameter]

        assert "OAGOpt" in optimizer_name
        assert optimizer.param_groups[0]["step"] == 1
        assert "denom" in state
        assert "ratio" in state
        assert "value_momentum" in state
        assert state["denom"].dtype == torch.float32
        assert state["ratio"].dtype == torch.float32
        assert state["value_momentum"].dtype == torch.float32
        assert state["denom"].device.type == "cpu"
        assert state["ratio"].device.type == "cpu"
        assert state["value_momentum"].device.type == "cpu"

    def test_registered_ocgopt_initializes_offloaded_state_on_first_step(self):
        """OCGOpt should initialize its centralized momentum state on the first step."""
        parameter = torch.nn.Parameter(torch.zeros(4, 4))
        parameters = [parameter]

        config = OptimizerConfig(
            optimizer_type="OCGOpt",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=[
                "weight_decay=0.02",
                "centralization=0.5",
                "spectral_clip_compile=False",
                "spectral_clip_dtype='float32'",
                "stochastic_fp=False",
                "sync_chunk_size=16",
                "state_storage_dtype='float32'",
                "state_storage_device='cpu'",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            parameters,
        )

        parameter.grad = torch.randn_like(parameter)
        optimizer.step()

        state = optimizer.state[parameter]

        assert "OCGOpt" in optimizer_name
        assert optimizer.param_groups[0]["step"] == 1
        assert "value_momentum" in state
        assert "centralized_momentum" in state
        assert "kahan_comp" not in state
        assert state["value_momentum"].dtype == torch.float32
        assert state["centralized_momentum"].dtype == torch.float32
        assert state["value_momentum"].device.type == "cpu"
        assert state["centralized_momentum"].device.type == "cpu"

    def test_registered_scgopt_initializes_state_on_first_step(self):
        """SCGOpt should initialize its denominator and momentum state on the first step."""
        parameter = torch.nn.Parameter(torch.zeros(4, 4))
        parameters = [parameter]

        config = OptimizerConfig(
            optimizer_type="SCGOpt",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=[
                "use_sign=False",
                "spectral_clip=True",
                "spectral_clip_compile=False",
                "spectral_clip_dtype='float32'",
                "stochastic_fp=False",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            parameters,
        )

        parameter.grad = torch.randn_like(parameter)
        optimizer.step()

        state = optimizer.state[parameter]

        assert "SCGOpt" in optimizer_name
        assert optimizer.param_groups[0]["step"] == 1
        assert "denom" in state
        assert "value_momentum" in state
        assert "centralized_momentum" in state
        assert state["denom"].shape == parameter.shape

    def test_registered_projectiveadam_initializes_projection_and_normuon_state(self):
        """ProjectiveAdam should initialize projection EMA and NorMuon state on the first step."""
        parameter = torch.nn.Parameter(torch.zeros(4, 4))
        parameters = [parameter]

        config = OptimizerConfig(
            optimizer_type="ProjectiveAdam",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=[
                "weight_decay=0.01",
                "projection='hyperbolic'",
                "input_norm=False",
                "normuon=True",
                "use_compile=False",
                "ortho_dtype='float32'",
                "stochastic_fp=False",
                "sync_chunk_size=16",
                "state_storage_dtype='float32'",
                "state_storage_device='cpu'",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            parameters,
        )

        parameter.grad = torch.randn_like(parameter)
        optimizer.step()

        state = optimizer.state[parameter]

        assert "ProjectiveAdam" in optimizer_name
        assert state["step"] == 1
        assert "exp_avg_y" in state
        assert "exp_avg_z" in state
        assert "normuon_second_momentum" in state
        assert state["exp_avg_y"].dtype == torch.float32
        assert state["exp_avg_z"].dtype == torch.float32
        assert state["normuon_second_momentum"].dtype == torch.float32
        assert state["exp_avg_y"].device.type == "cpu"
        assert state["exp_avg_z"].device.type == "cpu"
        assert state["normuon_second_momentum"].device.type == "cpu"
        assert state["normuon_second_momentum"].shape == (4, 1)

    def test_registered_wiwiopt_initializes_dynamic_lr_and_oja_state(self):
        """WiwiOpt should initialize its dynamic-LR, NorMuon, and Oja state on the first step."""
        parameter = torch.nn.Parameter(torch.zeros(4, 4))
        parameters = [parameter]

        config = OptimizerConfig(
            optimizer_type="WiwiOpt",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=[
                "weight_decay=0.01",
                "normuon=True",
                "use_compile=False",
                "ortho_dtype='float32'",
                "stochastic_fp=False",
                "dynamic_lr=True",
                "dynamic_lr_boost=False",
                "egd=True",
                "egd_oja=True",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            parameters,
        )

        parameter.grad = torch.randn_like(parameter)
        optimizer.step()

        state = optimizer.state[parameter]

        assert "WiwiOpt" in optimizer_name
        assert state["step"] == 1
        assert "accum" in state
        assert "exp_avg" in state
        assert "exp_avg_sq_row" in state
        assert "exp_avg_sq_col" in state
        assert "delta_ema" in state
        assert "delta_norm_ema" in state
        assert "normuon_second_momentum" in state
        assert "oja_basis" in state
        assert state["accum"].shape == (4, 1)
        assert state["exp_avg_sq_row"].shape == (4,)
        assert state["exp_avg_sq_col"].shape == (4,)
        assert state["delta_norm_ema"].shape == (4, 1)
        assert state["normuon_second_momentum"].shape == (4, 1)
        assert state["oja_basis"].shape[1] == 4

    def test_registered_cstableadamw_initializes_adopt_and_ssc_state_on_first_step(self):
        """CStableAdamW should initialize ADOPT and Stable-SPAM state on the first optimization step."""
        parameter = torch.nn.Parameter(torch.zeros(4, 4))
        parameters = [parameter]

        config = OptimizerConfig(
            optimizer_type="CStableAdamW",
            learning_rates=LearningRatesConfig(base=1e-3),
            optimizer_args=[
                "weight_decay=0.01",
                "weight_decouple=False",
                "eps=1e-12",
                "use_rms=True",
                "use_atan2=True",
                "atan2_a=1.1",
                "atan2_b=0.9",
                "cautious_factor=0.5",
                "use_adopt=True",
                "use_stable_spam_clipping=True",
                "ssc_scale=0.8",
                "ssc_gamma1=0.81",
                "ssc_gamma2=0.9999",
                "ssc_gamma3=0.99",
                "ssc_eps_floor=1e-20",
                "torch_compile=False",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            parameters,
        )

        parameter.grad = torch.arange(1, 17, dtype=parameter.dtype).view_as(parameter)
        optimizer.step()

        state = optimizer.state[parameter]

        assert "CStableAdamW" in optimizer_name
        assert optimizer.param_groups[0]["step"] == 1
        assert state["steps"] == 1
        assert "exp_avg" in state
        assert "exp_avg_sq" in state
        assert "prev_grad" in state
        assert "ssc_m_norm_t" in state
        assert "ssc_v_norm_t" in state
        assert "ssc_m_max_t" in state
        assert torch.count_nonzero(state["exp_avg"]) > 0
        assert torch.count_nonzero(state["prev_grad"]) > 0
        assert torch.count_nonzero(state["exp_avg_sq"]) == 0
        assert state["ssc_m_norm_t"].item() > 0.0
        assert state["ssc_v_norm_t"].item() > 0.0
        assert state["ssc_m_max_t"].item() > 0.0

    def test_registered_fftdescent_initializes_momentum_state_on_first_step(self):
        """FFTDescent should initialize momentum and optional sign momentum state on the first step."""
        parameter = torch.nn.Parameter(torch.zeros(4, 4))
        parameters = [parameter]

        config = OptimizerConfig(
            optimizer_type="FFTDescent",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=[
                "beta=0.9",
                "weight_decay=0.01",
                "spectral_clip=True",
                "spectral_clip_compile=False",
                "spectral_clip_dtype='float32'",
                "sign_momentum=0.8",
                "stochastic_fp=False",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            parameters,
        )

        parameter.grad = torch.randn_like(parameter)
        optimizer.step()

        state = optimizer.state[parameter]

        assert "FFTDescent" in optimizer_name
        assert optimizer.param_groups[0]["step"] == 1
        assert "momentum" in state
        assert "sign_momentum" in state
        assert state["momentum"].shape == parameter.shape
        assert state["sign_momentum"].shape == parameter.shape

    def test_registered_fishmonger_initializes_fim_and_diff_state_on_first_step(self):
        """FishMonger should initialize its momentum, FIM, and differential-amplification state on the first step."""
        parameter = torch.nn.Parameter(torch.zeros(4, 4))
        parameters = [parameter]

        config = OptimizerConfig(
            optimizer_type="FishMonger",
            learning_rates=LearningRatesConfig(base=1e-3),
            optimizer_args=[
                "betas=(0.8, 0.9, 0.99)",
                "eps=1e-8",
                "eps2=0.02",
                "eps_floor=1e-16",
                "weight_decay=0.01",
                "clip=0.5",
                "centralization=0.5",
                "diff_amp=0.1",
                "diff_amp_beta=0.95",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            parameters,
        )

        parameter.grad = torch.arange(1, 17, dtype=parameter.dtype).view_as(parameter)
        optimizer.step()

        state = optimizer.state[parameter]

        assert "FishMonger" in optimizer_name
        assert optimizer.param_groups[0]["step"] == 1
        assert "momentum" in state
        assert "momentum_slow" in state
        assert "momentum_slow_squared" in state
        assert "fim" in state
        assert "ema_diff" in state
        assert "previous_grad" in state
        assert torch.count_nonzero(state["momentum"]) > 0
        assert torch.count_nonzero(state["momentum_slow"]) > 0
        assert torch.count_nonzero(state["fim"]) > 0

    def test_registered_fishmonger8bitbnb_handles_runtime_support_expectations(self):
        """FishMonger8BitBNB should fail fast on unsupported CPU execution instead of crashing."""
        parameter = torch.nn.Parameter(torch.zeros(4, 4))
        parameters = [parameter]

        config = OptimizerConfig(
            optimizer_type="FishMonger8BitBNB",
            learning_rates=LearningRatesConfig(base=1e-3),
            optimizer_args=[
                "betas=(0.8, 0.9, 0.99)",
                "eps=1e-8",
                "eps2=0.02",
                "eps_floor=1e-16",
                "weight_decay=0.01",
                "clip=0.5",
                "centralization=0.5",
                "diff_amp=0.1",
                "diff_amp_beta=0.95",
                "quantization_group_size=64",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            parameters,
        )

        assert "FishMonger8BitBNB" in optimizer_name
        parameter.grad = torch.arange(1, 17, dtype=parameter.dtype).view_as(parameter)

        if not torch.cuda.is_available():
            with pytest.raises(RuntimeError, match="requires CUDA-enabled bitsandbytes"):
                optimizer.step()
            return

        parameter = torch.nn.Parameter(torch.zeros(4, 4, device="cuda"))
        parameters = [parameter]
        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            parameters,
        )
        parameter.grad = torch.arange(1, 17, dtype=parameter.dtype, device=parameter.device).view_as(parameter)
        optimizer.step()

        state = optimizer.state[parameter]
        assert optimizer.param_groups[0]["step"] == 1
        assert "momentum" in state
        assert "momentum_slow" in state
        assert "momentum_slow_squared" in state
        assert "fim" in state
        assert "ema_diff" in state
        assert "previous_grad" in state
        assert isinstance(state["momentum"], tuple)
        assert isinstance(state["fim"], tuple)

    def test_registered_farmscrop_initializes_fisher_and_diff_state_on_first_step(self):
        """FARMSCrop should initialize FIM, momentum, and diff-history state on the first step."""
        parameter = torch.nn.Parameter(torch.zeros(4, 4))
        parameters = [parameter]

        config = OptimizerConfig(
            optimizer_type="FARMSCrop",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=[
                "betas=(0.9, 0.99)",
                "weight_decay=0.01",
                "centralization=0.5",
                "diff_mult=1.25",
                "momentum_beta=0.95",
                "momentum_amp=3.0",
                "eps=1e-7",
                "eps2=0.02",
                "eps_floor=1e-16",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            parameters,
        )

        parameter.grad = torch.arange(1, 17, dtype=parameter.dtype).view_as(parameter)
        optimizer.step()

        state = optimizer.state[parameter]

        assert "FARMSCrop" in optimizer_name
        assert optimizer.param_groups[0]["step"] == 1
        assert "fim" in state
        assert "momentum" in state
        assert "previous_grad" in state
        assert "grad_diff_fim" in state
        assert state["fim"].shape == parameter.shape
        assert state["previous_grad"].shape == parameter.shape

    def test_registered_farmscropv2_initializes_optional_diff_state_on_first_step(self):
        """FARMSCropV2 should initialize its FIM, momentum, and optional diff-history state on the first step."""
        parameter = torch.nn.Parameter(torch.zeros(4, 4))
        parameters = [parameter]

        config = OptimizerConfig(
            optimizer_type="FARMSCropV2",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=[
                "betas=(0.9, 0.99)",
                "weight_decay=0.01",
                "centralization=0.5",
                "diff_mult=1.25",
                "momentum_beta=0.95",
                "momentum_lambda=0.35",
                "clip=0.75",
                "cautious=True",
                "cautious_grad='approx_grad_nat'",
                "eps=1e-7",
                "eps2=0.02",
                "eps_floor=1e-16",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            parameters,
        )

        parameter.grad = torch.arange(1, 17, dtype=parameter.dtype).view_as(parameter)
        optimizer.step()

        state = optimizer.state[parameter]

        assert "FARMSCropV2" in optimizer_name
        assert optimizer.param_groups[0]["step"] == 1
        assert "fim" in state
        assert "momentum" in state
        assert "previous_grad" in state
        assert "grad_diff_fim" in state
        assert state["momentum"].shape == parameter.shape
        assert state["grad_diff_fim"].shape == parameter.shape

    def test_registered_fmarscrop_initializes_mars_and_diff_state_on_first_step(self):
        """FMARSCrop should initialize FIM, momentum, and MARS diff-history state on the first step."""
        parameter = torch.nn.Parameter(torch.zeros(4, 4))
        parameters = [parameter]

        config = OptimizerConfig(
            optimizer_type="FMARSCrop",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=[
                "betas=(0.9, 0.99)",
                "weight_decay=0.01",
                "weight_decouple=True",
                "centralization=0.5",
                "moment_centralization=0.25",
                "diff_mult=1.25",
                "momentum_beta=0.95",
                "momentum_lambda=0.35",
                "gamma=0.01",
                "clip=0.75",
                "adaptive_clip=0.5",
                "adaptive_clip_type='layer'",
                "cautious=True",
                "debias_beta2=False",
                "eps=1e-7",
                "eps2=0.02",
                "eps_floor=1e-16",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            parameters,
        )

        parameter.grad = torch.arange(1, 17, dtype=parameter.dtype).view_as(parameter)
        optimizer.step()

        state = optimizer.state[parameter]

        assert "FMARSCrop" in optimizer_name
        assert optimizer.param_groups[0]["step"] == 1
        assert "fim" in state
        assert "momentum" in state
        assert "prev_grad" in state
        assert "grad_diff_fim" in state
        assert state["fim"].shape == parameter.shape
        assert state["prev_grad"].shape == parameter.shape

    def test_registered_fmarscropv2_initializes_mars_and_diff_state_on_first_step(self):
        """FMARSCropV2 should initialize FIM, momentum, and optional diff-history state on the first step."""
        parameter = torch.nn.Parameter(torch.zeros(4, 4))
        parameters = [parameter]

        config = OptimizerConfig(
            optimizer_type="FMARSCropV2",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=[
                "betas=(0.9, 0.99)",
                "weight_decay=0.01",
                "centralization=0.5",
                "moment_centralization=0.25",
                "diff_mult=1.25",
                "momentum_beta=0.95",
                "momentum_lambda=0.35",
                "gamma=0.01",
                "clip=0.75",
                "adaptive_clip=0.5",
                "cautious=True",
                "debias_beta2=False",
                "eps=1e-7",
                "eps2=0.02",
                "eps_floor=1e-16",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            parameters,
        )

        parameter.grad = torch.arange(1, 17, dtype=parameter.dtype).view_as(parameter)
        optimizer.step()

        state = optimizer.state[parameter]

        assert "FMARSCropV2" in optimizer_name
        assert optimizer.param_groups[0]["step"] == 1
        assert "fim" in state
        assert "momentum" in state
        assert "prev_grad" in state
        assert "grad_diff_fim" in state
        assert state["momentum"].shape == parameter.shape
        assert state["grad_diff_fim"].shape == parameter.shape

    def test_registered_fmarscropv2exmachina_initializes_strategy_and_diff_state_on_first_step(self):
        """FMARSCropV2ExMachina should initialize FIM, momentum, and diff-history state on the first step."""
        parameter = torch.nn.Parameter(torch.zeros(4, 4))
        parameters = [parameter]

        config = OptimizerConfig(
            optimizer_type="FMARSCropV2ExMachina",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=[
                "betas=(0.9, 0.99, 0.999)",
                "weight_decay=0.01",
                "weight_decouple=True",
                "centralization=0.5",
                "moment_centralization=0.25",
                "diff_mult=1.25",
                "momentum_lambda=0.35",
                "gamma=0.01",
                "clip=0.75",
                "adaptive_clip=0.5",
                "adaptive_clip_type='layer'",
                "update_strategy='grams'",
                "debias_beta1=True",
                "debias_beta2=False",
                "debias_beta3=True",
                "eps=1e-7",
                "eps2=0.02",
                "eps_floor=1e-16",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            parameters,
        )

        parameter.grad = torch.arange(1, 17, dtype=parameter.dtype).view_as(parameter)
        optimizer.step()

        state = optimizer.state[parameter]

        assert "FMARSCropV2ExMachina" in optimizer_name
        assert optimizer.param_groups[0]["step"] == 1
        assert optimizer.param_groups[0]["update_strategy"] == "grams"
        assert "fim" in state
        assert "momentum" in state
        assert "prev_grad" in state
        assert "grad_diff_fim" in state
        assert state["momentum"].shape == parameter.shape
        assert state["grad_diff_fim"].shape == parameter.shape

    def test_registered_fmarscropv3_initializes_state_on_first_step(self):
        """FMARSCropV3 should initialize FIM, momentum, and diff-history state on the first step."""
        parameter = torch.nn.Parameter(torch.zeros(4, 4))
        parameters = [parameter]

        config = OptimizerConfig(
            optimizer_type="FMARSCropV3",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=[
                "betas=(0.9, 0.95)",
                "weight_decay=0.01",
                "centralization=0.5",
                "moment_centralization=0.25",
                "diff_mult=1.25",
                "momentum_lambda=1.5",
                "gamma=0.01",
                "clip_lambda=0.75",
                "adaptive_clip=0.5",
                "adaptive_clip_norm_type=False",
                "cautious=True",
                "eps=1e-7",
                "eps2=0.02",
                "eps_floor=1e-16",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            parameters,
        )

        parameter.grad = torch.arange(1, 17, dtype=parameter.dtype).view_as(parameter)
        optimizer.step()

        state = optimizer.state[parameter]

        assert "FMARSCropV3" in optimizer_name
        assert optimizer.param_groups[0]["step"] == 1
        assert "fim" in state
        assert "momentum" in state
        assert "prev_grad" in state
        assert "grad_diff_fim" in state
        assert state["momentum"].shape == parameter.shape
        assert state["grad_diff_fim"].shape == parameter.shape

    def test_registered_fmarscropv3exmachina_initializes_state_on_first_step(self):
        """FMARSCropV3ExMachina should initialize FIM, momentum, and diff-history state on the first step."""
        parameter = torch.nn.Parameter(torch.zeros(4, 4))
        parameters = [parameter]

        config = OptimizerConfig(
            optimizer_type="FMARSCropV3ExMachina",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=[
                "betas=(0.9, 0.95)",
                "weight_decay=0.01",
                "weight_decouple=True",
                "centralization=0.5",
                "moment_centralization=0.25",
                "diff_mult=1.25",
                "momentum_lambda=1.5",
                "gamma=0.01",
                "clip=0.75",
                "adaptive_clip=0.5",
                "adaptive_clip_type='layer'",
                "update_strategy='both'",
                "debias_beta1=True",
                "debias_beta2=True",
                "stable_update=True",
                "atan2_denom=True",
                "use_orthograd=True",
                "eps=1e-7",
                "eps2=0.02",
                "eps_floor=1e-16",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            parameters,
        )

        parameter.grad = torch.arange(1, 17, dtype=parameter.dtype).view_as(parameter)
        optimizer.step()

        state = optimizer.state[parameter]

        assert "FMARSCropV3ExMachina" in optimizer_name
        assert optimizer.param_groups[0]["step"] == 1
        assert optimizer.param_groups[0]["update_strategy"] == "both"
        assert "fim" in state
        assert "momentum" in state
        assert "prev_grad" in state
        assert "grad_diff_fim" in state
        assert state["momentum"].shape == parameter.shape
        assert state["grad_diff_fim"].shape == parameter.shape

    def test_registered_abmog_initializes_cpu_state_without_cuda(self):
        """ABMOG should initialize repo-owned offloaded state and run on CPU-only setups."""
        parameter = torch.nn.Parameter(torch.zeros(4, 4))
        parameters = [parameter]

        config = OptimizerConfig(
            optimizer_type="ABMOG",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=[
                "weight_decay=0.01",
                "weight_decay_rate=0.99",
                "adaptive=True",
                "bcos=False",
                "abm_order=3",
                "abm_k=2",
                "state_storage_dtype='float32'",
                "state_storage_device='cpu'",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            parameters,
        )

        parameter.grad = torch.arange(1, 17, dtype=parameter.dtype).view_as(parameter)
        optimizer.step()

        state = optimizer.state[parameter]

        assert "ABMOG" in optimizer_name
        assert optimizer.param_groups[0]["step"] == 1
        assert "value_momentum" in state
        assert "denom" in state
        assert "p_history" in state
        assert state["value_momentum"].device.type == "cpu"
        assert state["denom"].device.type == "cpu"

    def test_registered_glyph_initializes_ema_state_on_first_step(self):
        """Glyph should initialize EMA, squared EMA, and previous-grad state on the first step."""
        parameter = torch.nn.Parameter(torch.zeros(4, 4))
        parameters = [parameter]

        config = OptimizerConfig(
            optimizer_type="Glyph",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=[
                "weight_decay=0.01",
                "weight_decay_rate=0.99",
                "amp=1.5",
                "orthograd=True",
                "adaptive_ema=True",
                "atan2=True",
                "cautious_min=0.25",
                "stochastic_fp=False",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            parameters,
        )

        parameter.grad = torch.randn_like(parameter)
        optimizer.step()

        state = optimizer.state[parameter]

        assert "Glyph" in optimizer_name
        assert optimizer.param_groups[0]["step"] == 1
        assert "ema" in state
        assert "ema_squared" in state
        assert "prev_grad" in state
        assert state["ema"].shape == parameter.shape
        assert state["prev_grad"].shape == parameter.shape

    def test_registered_singstate_initializes_momentum_state_on_first_step(self):
        """SingState should initialize its sign-tracking momentum state on the first step."""
        parameter = torch.nn.Parameter(torch.zeros(4, 4))
        parameters = [parameter]

        config = OptimizerConfig(
            optimizer_type="SingState",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=[
                "weight_decay=0.01",
                "weight_decay_rate=0.99",
                "spectral_clip=False",
                "lowpass_grad=0.5",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            parameters,
        )

        parameter.grad = torch.randn_like(parameter)
        optimizer.step()

        state = optimizer.state[parameter]

        assert "SingState" in optimizer_name
        assert optimizer.param_groups[0]["step"] == 1
        assert "momentum" in state
        assert state["momentum"].shape == parameter.shape

    def test_registered_talon_initializes_multistage_momentum_state_on_first_step(self):
        """TALON should initialize value, denominator, and sign state on the first step."""
        parameter = torch.nn.Parameter(torch.zeros(4, 4))
        parameters = [parameter]

        config = OptimizerConfig(
            optimizer_type="TALON",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=[
                "weight_decay=0.01",
                "weight_decay_rate=0.99",
                "denom_atan2=True",
                "spectral_clip=False",
                "signscale_power=1.5",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            parameters,
        )

        parameter.grad = torch.randn_like(parameter)
        optimizer.step()

        state = optimizer.state[parameter]

        assert "TALON" in optimizer_name
        assert optimizer.param_groups[0]["step"] == 1
        assert "value_momentum" in state
        assert "stage2_emasq" in state
        assert "sign_momentum" in state
        assert state["value_momentum"].shape == parameter.shape
        assert state["sign_momentum"].shape == parameter.shape

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

    def test_registered_grokfastadamw_tracks_grok_state_after_warmup(self):
        """GrokFastAdamW should initialize the grok EMA on the first step and use it once warmup has elapsed."""
        parameter = torch.nn.Parameter(torch.zeros(4, 4))
        parameters = [parameter]

        config = OptimizerConfig(
            optimizer_type="GrokFastAdamW",
            learning_rates=LearningRatesConfig(base=3e-4),
            optimizer_args=[
                "weight_decay=0.01",
                "weight_decouple=False",
                "fixed_decay=True",
                "grokfast=True",
                "grokfast_alpha=0.95",
                "grokfast_lamb=1.5",
                "grokfast_after_step=1",
                "eps=1e-7",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            parameters,
        )

        first_grad = torch.ones_like(parameter)
        parameter.grad = first_grad.clone()
        optimizer.step()

        first_step_state = optimizer.state[parameter]["grok_exp_avg"].clone()

        second_grad = torch.full_like(parameter, 3.0)
        parameter.grad = second_grad.clone()
        optimizer.step()

        state = optimizer.state[parameter]

        assert "GrokFastAdamW" in optimizer_name
        assert optimizer.param_groups[0]["step"] == 2
        assert "exp_avg" in state
        assert "exp_avg_sq" in state
        assert "grok_exp_avg" in state
        assert torch.allclose(first_step_state, first_grad)
        assert torch.allclose(
            state["grok_exp_avg"],
            first_grad.lerp(second_grad, weight=1.0 - optimizer.param_groups[0]["grokfast_alpha"]),
        )
        assert torch.count_nonzero(state["exp_avg"]) > 0
        assert torch.count_nonzero(state["exp_avg_sq"]) > 0

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

    def test_registered_momentuscaution_initializes_running_state_on_first_step(self):
        """MomentusCaution should initialize momentum and gradient-history state on the first optimization step."""
        parameter = torch.nn.Parameter(torch.zeros(4, 4))
        parameters = [parameter]

        config = OptimizerConfig(
            optimizer_type="MomentusCaution",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=[
                "beta=0.85",
                "momentum_beta=0.5",
                "weight_decay=0.01",
                "gamma_ratio=0.25",
                "adaptive_clip=0.5",
                "cautious=False",
                "nesterov=True",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            parameters,
        )

        parameter.grad = torch.arange(1, 17, dtype=parameter.dtype).view_as(parameter)
        optimizer.step()

        state = optimizer.state[parameter]

        assert "MomentusCaution" in optimizer_name
        assert optimizer.param_groups[0]["step"] == 1
        assert "momentum" in state
        assert "prev_grad" in state
        assert "grad_momentum" in state
        assert state["momentum"].shape == parameter.shape
        assert state["prev_grad"].shape == parameter.shape

    def test_registered_remaster_initializes_running_state_on_first_step(self):
        """REMASTER should initialize EMA and squared-EMA state on the first optimization step."""
        parameter = torch.nn.Parameter(torch.zeros(4, 4))
        parameters = [parameter]

        config = OptimizerConfig(
            optimizer_type="REMASTER",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=[
                "weight_decay=0.01",
                "weight_decay_rate=0.99",
                "amp=3.0",
                "reset_interval=2",
                "reset_increment=1",
                "orthograd=False",
                "cautious_min=0.25",
                "stochastic_fp=False",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            parameters,
        )

        parameter.grad = torch.randn_like(parameter)
        optimizer.step()

        state = optimizer.state[parameter]

        assert "REMASTER" in optimizer_name
        assert optimizer.param_groups[0]["step"] == 1
        assert "ema" in state
        assert "ema_squared" in state
        assert "times_zero" in state
        assert "steps_since_reset" in state
        assert state["ema"].shape == parameter.shape
        assert state["ema_squared"].shape == parameter.shape

    def test_registered_adammini_builds_from_named_module_group_and_initializes_specialized_state(self):
        """AdamMini should preserve donor name-based grouping when built from module parameter groups."""

        class _ToyAdamMiniModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(8, 8)
                self.q_proj = nn.Linear(8, 8, bias=False)
                self.k_proj = nn.Linear(8, 8, bias=False)
                self.mlp = nn.Linear(8, 8, bias=False)

        module = _ToyAdamMiniModule()
        trainable_params = [build_module_parameter_group(module, lr=1e-3, label="toy_model")]

        config = OptimizerConfig(
            optimizer_type="AdamMini",
            learning_rates=LearningRatesConfig(base=1e-3),
            optimizer_args=[
                "weight_decay=0.01",
                "num_embeds=8",
                "num_heads=2",
                "num_query_groups=2",
            ],
        )

        optimizer_name, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            trainable_params,
        )

        named_groups = {group["name"]: group for group in optimizer.param_groups}
        assert "AdamMini" in optimizer_name
        assert str(optimizer) == "AdamMini"
        assert "embed.weight" in named_groups
        assert "q_proj.weight" in named_groups
        assert "mlp.weight" in named_groups

        for _, parameter in module.named_parameters():
            parameter.grad = torch.ones_like(parameter)

        optimizer.step()

        embed_state = optimizer.state[module.embed.weight]
        q_proj_state = optimizer.state[module.q_proj.weight]
        mlp_state = optimizer.state[module.mlp.weight]

        assert "m" in embed_state
        assert "v" in embed_state
        assert "head" in q_proj_state
        assert "v_mean" in q_proj_state
        assert "dimension" in mlp_state
        assert "reduced" in mlp_state

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
