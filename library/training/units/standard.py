"""The standard profile's implementations: contribution and the end of a window (D17, D18).

Training's words call them, and a strategy pins them in its uses like any
other implementation. Each delegates to the unit's live state, which its
update lease hands it; Trainer optimization performs the mechanics. A
contribution differentiates its loss, so the calls whose results reach that
loss through inputs declared differentiable run with grad mode, and the
modules the unit trains run in train mode in them (D15, D20).

The window end takes the unit's update lease and one update lease on each
participant it trains, so it is declared for a unit of one participant;
units of several participants need a declaration whose lease inputs repeat,
and until then preparation refuses their window end as a call that
disagrees with its implementation.
"""

from library.iro.frontend import implementation
from library.iro.seams import Effect
from library.training.units.prepared import LiveUnit
from library.training.words import ADVANCE, CONTRIBUTE


@implementation(
    version=1,
    name=CONTRIBUTE,
    inputs=("owner_update lease", "retained loss"),
    effects=[Effect.DEVICE],
    differentiates=(1,),
)
def contribute(unit: LiveUnit, loss: object) -> None:
    unit.contribute(loss)  # ty: ignore[invalid-argument-type]


@implementation(version=1, name=ADVANCE, inputs=("owner_update lease", "owner_update lease"), effects=[Effect.DEVICE])
def advance(unit: LiveUnit, participant: object) -> None:
    unit.advance((participant,))


def installed() -> dict:
    """The standard profile's implementations by name, for the launcher's composition."""
    return {CONTRIBUTE: contribute, ADVANCE: advance}


__all__ = ["advance", "contribute", "installed"]
