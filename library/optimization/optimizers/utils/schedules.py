import torch
from torch.nn import Parameter, ParameterList
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler


class CosineDecay:
    """Small cosine-decay helper used by some optimizer-local runtime features."""

    def __init__(self, death_rate: float, t_max: int, eta_min: float = 0.0, last_epoch: int = -1):
        self.sgd = torch.optim.SGD(ParameterList([Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, t_max + 1, eta_min, last_epoch)
        self.t_max = t_max
        self.eta_min = eta_min

    def step(self, current_step: int) -> None:
        self.cosine_stepper.step(current_step)

    def get_dr(self, current_step: int) -> float:
        if current_step >= self.t_max:
            return self.eta_min
        self.step(current_step)
        return self.sgd.param_groups[0]["lr"]


class SSCCosineDecay:
    """Cosine decay helper for stable-spam clipping warmup."""

    def __init__(self, death_rate: float, t_max: int, eta_min: float = 0.0, last_epoch: int = -1):
        self.sgd: Optimizer = SGD(ParameterList([Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper: LRScheduler = CosineAnnealingLR(self.sgd, t_max + 1, eta_min, last_epoch)
        self.t_max = t_max
        self.eta_min = eta_min

    def step(self, current_step: int) -> None:
        self.cosine_stepper.step(current_step)

    def get_death_rate(self, current_step: int) -> float:
        if current_step >= self.t_max:
            return self.eta_min
        self.step(current_step)
        return self.sgd.param_groups[0]["lr"]
