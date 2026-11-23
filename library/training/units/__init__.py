"""Trainer optimization: optimization units, their resolution and their owner (D20, D21).

Named for optimization units, apart from ``library/optimization``, the
library of optimizers and schedulers it builds from. The plan derives from
the accepted strategy; candidates are resolved inside a preparation attempt;
prepared units publish with the attempt; and Trainer optimization owns every
unit at the owner seam, realizing the standard profile's contribution and
window end.
"""

from library.training.units.candidates import Member, UnitCandidate, resolve
from library.training.units.modes import ModuleModes
from library.training.units.owner import OptimizationSection, TrainerOptimization, UnitLease
from library.training.units.plan import OptimizationPlan, OptimizerPolicy, UnitDefinition, UnitRef
from library.training.units.derivation import DERIVATIVE, GRAD_MODE, UNITS, GradMode, TrainingDerivation
from library.training.units.prepared import TRAINER, Advancement, LiveUnit, Outcome, PreparedUnit, Progress

__all__ = [
    "DERIVATIVE",
    "GRAD_MODE",
    "TRAINER",
    "UNITS",
    "Advancement",
    "GradMode",
    "LiveUnit",
    "Member",
    "ModuleModes",
    "OptimizationPlan",
    "OptimizationSection",
    "OptimizerPolicy",
    "Outcome",
    "PreparedUnit",
    "Progress",
    "TrainerOptimization",
    "TrainingDerivation",
    "UnitCandidate",
    "UnitDefinition",
    "UnitLease",
    "UnitRef",
    "resolve",
]
