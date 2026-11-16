"""Training's words in the contract: the unit, the window and contribution, with the standard profile (D16-D18, D20).

``contract()`` is the core vocabulary extended with them, which the launcher
composes and tests use. The remaining words (advance, zero, trigger and
grant) come with the second pass of section 16.
"""

from library.iro.composition import Composition, Definition, core
from library.training.words.forms import ContributeForm, UnitForm, WindowForm
from library.training.words.operations import ADVANCE, CONTRIBUTE, STANDARD, UNIT_KIND, ContributeOp, ProfileAttr, UnitOp, WindowOp

AREA = "training"


def definitions() -> tuple[Definition, ...]:
    """Training's definitions, each with its definition version and written form."""
    return (
        Definition(UnitOp, AREA, form=UnitForm),
        Definition(WindowOp, AREA, form=WindowForm),
        Definition(ContributeOp, AREA, form=ContributeForm),
        Definition(ProfileAttr, AREA),
    )


def contract(base: Composition | None = None) -> Composition:
    """The core vocabulary with training's words."""
    return (base or core()).extend(definitions(), name="training")


__all__ = [
    "ADVANCE",
    "AREA",
    "CONTRIBUTE",
    "STANDARD",
    "UNIT_KIND",
    "ContributeOp",
    "ProfileAttr",
    "UnitOp",
    "WindowOp",
    "contract",
    "definitions",
]
