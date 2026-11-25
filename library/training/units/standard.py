"""The standard profile's implementations: contribution and the end of a window (D17, D18).

Training's words call them, and a strategy names them in its uses like any
other implementation. Each delegates to the unit's live state, which its
update lease hands it; Trainer optimization performs the mechanics. A
contribution's loss is a sink of training's ``derivative`` relation: the
calls on a path from the unit's trained state into it run with grad mode,
and the modules the unit trains run in train mode in them (D15, D16, D20).
The loss may come with its mass, which the window divides by (D20).

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


@implementation(name=CONTRIBUTE, inputs=("owner_update lease", "retained loss"), effects=[Effect.DEVICE])
def contribute(unit: LiveUnit, loss: object) -> None:
    unit.contribute(loss)


@implementation(name=ADVANCE, inputs=("owner_update lease", "owner_update lease"), effects=[Effect.DEVICE])
def advance(unit: LiveUnit, participant: object) -> None:
    unit.advance((participant,))


def installed() -> dict:
    """The standard profile's implementations by name, for the launcher's composition."""
    return {CONTRIBUTE: contribute, ADVANCE: advance}


__all__ = ["advance", "contribute", "installed"]
