"""EDM2 loss-weighting package."""

from library.losses.edm2.factory import create_edm2_modifier
from library.losses.edm2.edm2_modifier import EDM2LossModifier
from library.losses.edm2.validation import handle_conflicting_configuration

__all__ = ["EDM2LossModifier", "create_edm2_modifier", "handle_conflicting_configuration"]
