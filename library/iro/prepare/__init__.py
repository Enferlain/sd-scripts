"""Preparation: from an accepted run to a prepared run for a target (D14, D16, D21).

Matching, effect checks, placement, fusion of local chains and device
admissions; the implementation manifest; the prepared form's printout and
its cold loading; and lowering, the form the runtime executes.
"""

from library.iro.prepare.lowered import Lowered, lower
from library.iro.prepare.prepared import Admission, Group, ManifestEntry, Prepared, code_identity, load_prepared, prepare
from library.iro.prepare.target import HOST, DeviceFacts, Target

__all__ = [
    "HOST",
    "Admission",
    "DeviceFacts",
    "Group",
    "Lowered",
    "ManifestEntry",
    "Prepared",
    "Target",
    "code_identity",
    "load_prepared",
    "lower",
    "prepare",
]
