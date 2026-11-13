"""The ``iro`` namespace authors import (D18, D21).

It starts with the implementation declaration; the activity reader and the
other decorators come with authoring (``tasks.md`` section 17).
"""

from library.iro.frontend.implementation import Declared, implementation

__all__ = ["Declared", "implementation"]
