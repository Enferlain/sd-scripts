"""Elementwise losses: one error per element of a prediction against its target.

A training loss is composed in the strategy from calls cut where the choices
vary independently: an elementwise loss, the per-sample reduction, a
weighting and the weighted mean (:mod:`library.losses.reductions`). A
variant that computes differently, such as in another precision, is another
function.
"""

import torch


def l2_f64(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The squared error of each element, (prediction - target)², computed in float64.

    Today's l2 also adds float32's smallest normal number, about 1.2e-38, to
    every squared error. That changes no gradient and, in float64, no value
    above about 1e-22, so this one leaves it out; the reference step's
    numbers match either way.
    """
    difference = prediction.to(torch.float64) - target.to(torch.float64)
    return difference**2


__all__ = ["l2_f64"]
