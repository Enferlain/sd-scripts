"""Repo-owned PEFT adapter method runtimes."""

from .boft import REGISTRATION as BOFT_REGISTRATION
from .dylora import REGISTRATION as DYLORA_REGISTRATION
from .loha import REGISTRATION as LOHA_REGISTRATION
from .locon import REGISTRATION as LOCON_REGISTRATION
from .lokr import REGISTRATION as LOKR_REGISTRATION
from .lora import REGISTRATION as LORA_REGISTRATION
from .oft import REGISTRATION as OFT_REGISTRATION


PEFT_METHOD_REGISTRATIONS = (
    BOFT_REGISTRATION,
    DYLORA_REGISTRATION,
    LOHA_REGISTRATION,
    LOCON_REGISTRATION,
    LOKR_REGISTRATION,
    LORA_REGISTRATION,
    OFT_REGISTRATION,
)

__all__ = [
    "PEFT_METHOD_REGISTRATIONS",
    "boft",
    "dylora",
    "loha",
    "locon",
    "lokr",
    "lora",
    "oft",
]
