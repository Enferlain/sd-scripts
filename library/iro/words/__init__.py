"""Kuroiro's core words, by the vocabulary's areas (D16, D21).

Each area's module defines its operations with their operation contracts and
shape checks. Domain vocabularies, training first, add their own words under
sub-namespaces such as ``iro.train``.
"""

from xdsl.ir import Dialect

from library.iro.operation_contracts import ResourceUseAttr
from library.iro.words import access, calls, data, lifetimes, model, structure
from library.iro.words.types import TYPES

OPERATIONS = (*structure.OPERATIONS, *lifetimes.OPERATIONS, *data.OPERATIONS, *access.OPERATIONS, *calls.OPERATIONS, *model.OPERATIONS)

Kuroiro = Dialect("iro", list(OPERATIONS), [*TYPES, ResourceUseAttr])

__all__ = ["OPERATIONS", "Kuroiro"]
