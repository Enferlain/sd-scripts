from __future__ import annotations

import math
from functools import wraps
from typing import cast
from weakref import ref

from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer


class _BaseWarmRestartsScheduler(LRScheduler):
    """Shared warm-restart scheduler base used by absorbed repo schedulers."""

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        gamma: float,
        cycle_multiplier: float = 1.0,
        first_cycle_max_steps: int = 1,
        min_lr: float = 1e-6,
        warmup_steps: int = 0,
        last_epoch: int = -1,
    ) -> None:
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        if warmup_steps >= first_cycle_max_steps:
            raise ValueError(
                f"[-] warmup_steps must be smaller than first_cycle_max_steps. {warmup_steps} < {first_cycle_max_steps}"
            )

        self.cycle_multiplier = cycle_multiplier
        self.gamma = gamma

        if last_epoch == -1:
            self._setup_optimizer_state(
                optimizer=optimizer,
                warmup_steps=warmup_steps,
                first_cycle_max_steps=first_cycle_max_steps,
                min_lr=min_lr,
            )
        self._validate_optimizer_state(optimizer)
        self._patch_optimizer_step(optimizer)
        super().__init__(optimizer, last_epoch=last_epoch)

    @staticmethod
    def _patch_optimizer_step(optimizer: Optimizer) -> None:
        """Mirror PyTorch scheduler step tracking for custom scheduler classes."""
        if hasattr(optimizer.step, "_wrapped_by_lr_sched"):
            return

        def wrap_step(step_fn):
            optimizer_ref = ref(optimizer)
            func = step_fn.__func__

            @wraps(func)
            def wrapper(*args, **kwargs):
                current_optimizer = optimizer_ref()
                if current_optimizer is None:
                    raise RuntimeError("Optimizer reference expired while scheduler wrapper was active.")
                current_optimizer._opt_called = True  # type: ignore[attr-defined]
                return func.__get__(current_optimizer, current_optimizer.__class__)(*args, **kwargs)

            wrapper._wrapped_by_lr_sched = True  # type: ignore[attr-defined]
            return wrapper

        optimizer.step = wrap_step(optimizer.step)  # type: ignore[method-assign]

    @staticmethod
    def _resolve_group_min_lr(group_lr: float | Tensor, configured_min_lr: float) -> float:
        return configured_min_lr if configured_min_lr < group_lr else 0.0

    @classmethod
    def _setup_optimizer_state(
        cls,
        *,
        optimizer: Optimizer,
        warmup_steps: int,
        first_cycle_max_steps: int,
        min_lr: float,
    ) -> None:
        for group in optimizer.param_groups:
            lr = group["lr"]
            if isinstance(lr, Tensor):
                lr = float(lr.detach().item())

            group.setdefault("warmup_steps", warmup_steps)
            group.setdefault("current_cycle_max_steps", first_cycle_max_steps)
            group.setdefault("min_lr", cls._resolve_group_min_lr(lr, min_lr))
            group.setdefault("current_cycle", 0)
            group.setdefault("current_cycle_step", -1)
            group.setdefault("initial_lr", lr)
            group.setdefault("current_max_lr", lr)

    @staticmethod
    def _validate_optimizer_state(optimizer: Optimizer) -> None:
        required_keys = {
            "warmup_steps",
            "current_cycle_max_steps",
            "min_lr",
            "current_cycle",
            "current_cycle_step",
            "initial_lr",
            "current_max_lr",
        }
        for index, group in enumerate(optimizer.param_groups):
            for key in required_keys:
                if key not in group:
                    raise KeyError(f"param '{key}' is not specified in param_groups[{index}] when resuming an optimizer")
            if group["warmup_steps"] >= group["current_cycle_max_steps"]:
                raise ValueError(
                    "[-] warmup_steps must be smaller than first_cycle_max_steps. "
                    f"{group['warmup_steps']} < {group['current_cycle_max_steps']}"
                )

    def _advance_group(self, group: dict) -> dict:
        if group["current_cycle_step"] == -1:
            while group["current_cycle_step"] >= group["current_cycle_max_steps"]:
                group = self._roll_cycle(group)
        group["current_cycle_step"] += 1
        return self._roll_cycle(group)

    def _roll_cycle(self, group: dict) -> dict:
        if group["current_cycle_step"] < group["current_cycle_max_steps"]:
            return group

        group["current_cycle_step"] -= group["current_cycle_max_steps"]
        group["current_cycle"] += 1
        group["current_cycle_max_steps"] = round(
            (group["current_cycle_max_steps"] - group["warmup_steps"]) * self.cycle_multiplier
        ) + group["warmup_steps"]
        group["current_max_lr"] = group["initial_lr"] * (self.gamma ** group["current_cycle"])
        return group

    @staticmethod
    def _normalized_progress(group: dict) -> tuple[float, float]:
        normalized_step = cast(float, group["current_cycle_step"] - group["warmup_steps"])
        normalized_max_steps = cast(float, group["current_cycle_max_steps"] - group["warmup_steps"])
        return normalized_step, normalized_max_steps

    def _get_group_lr(self, group: dict) -> float:
        raise NotImplementedError

    def get_lr(self) -> list[float]:
        lrs: list[float] = []
        for index, group in enumerate(self.optimizer.param_groups):
            advanced_group = self._advance_group(group)
            self.optimizer.param_groups[index] = advanced_group
            lrs.append(self._get_group_lr(advanced_group))
        return lrs


class CosineAnnealingWarmRestarts(_BaseWarmRestartsScheduler):
    """Repo-owned absorption of the vendor cosine warm-restarts scheduler."""

    @staticmethod
    def _resolve_group_min_lr(group_lr: float | Tensor, configured_min_lr: float) -> float:
        return configured_min_lr

    def _get_group_lr(self, group: dict) -> float:
        if group["current_max_lr"] <= group["min_lr"]:
            return cast(float, group["min_lr"])

        lr_range = cast(float, group["current_max_lr"] - group["min_lr"])
        warmup_steps = cast(int, group["warmup_steps"])
        if group["current_cycle_step"] < warmup_steps:
            return lr_range * cast(float, group["current_cycle_step"]) / warmup_steps + cast(float, group["min_lr"])

        normalized_step, normalized_max_steps = self._normalized_progress(group)
        return lr_range * (1 + math.cos(math.pi * normalized_step / normalized_max_steps)) / 2.0 + cast(float, group["min_lr"])


class RexAnnealingWarmRestarts(_BaseWarmRestartsScheduler):
    """Repo-owned absorption of the vendor REX warm-restarts scheduler."""

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        gamma: float,
        cycle_multiplier: float = 1.0,
        first_cycle_max_steps: int = 1,
        min_lr: float = 1e-6,
        warmup_steps: int = 0,
        last_epoch: int = -1,
        d: float = 0.9,
    ) -> None:
        self.d = d
        super().__init__(
            optimizer,
            gamma=gamma,
            cycle_multiplier=cycle_multiplier,
            first_cycle_max_steps=first_cycle_max_steps,
            min_lr=min_lr,
            warmup_steps=warmup_steps,
            last_epoch=last_epoch,
        )

    def _get_group_lr(self, group: dict) -> float:
        if group["current_max_lr"] <= group["min_lr"]:
            return cast(float, group["min_lr"])

        lr_range = cast(float, group["current_max_lr"] - group["min_lr"])
        warmup_steps = cast(int, group["warmup_steps"])
        if group["current_cycle_step"] < warmup_steps:
            return lr_range * cast(float, group["current_cycle_step"]) / warmup_steps + cast(float, group["min_lr"])

        normalized_step, normalized_max_steps = self._normalized_progress(group)
        progress = normalized_step / normalized_max_steps
        divider = (1 - self.d) + (self.d * (1 - progress))
        return cast(float, group["min_lr"]) + lr_range * ((1 - progress) / divider)
