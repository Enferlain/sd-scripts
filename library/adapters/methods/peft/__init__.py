"""Repo-owned PEFT adapter method runtimes."""

from .abba import REGISTRATION as ABBA_REGISTRATION
from .boft import REGISTRATION as BOFT_REGISTRATION
from .dylora import REGISTRATION as DYLORA_REGISTRATION
from .glora import REGISTRATION as GLORA_REGISTRATION
from .ia3 import REGISTRATION as IA3_REGISTRATION
from .loha import REGISTRATION as LOHA_REGISTRATION
from .locon import REGISTRATION as LOCON_REGISTRATION
from .lokr import REGISTRATION as LOKR_REGISTRATION
from .lora import REGISTRATION as LORA_REGISTRATION
from .oft import REGISTRATION as OFT_REGISTRATION
from .tlora import REGISTRATION as TLORA_REGISTRATION
from .vera import REGISTRATION as VERA_REGISTRATION


PEFT_METHOD_REGISTRATIONS = (
    ABBA_REGISTRATION,
    BOFT_REGISTRATION,
    DYLORA_REGISTRATION,
    GLORA_REGISTRATION,
    IA3_REGISTRATION,
    LOHA_REGISTRATION,
    LOCON_REGISTRATION,
    LOKR_REGISTRATION,
    LORA_REGISTRATION,
    OFT_REGISTRATION,
    TLORA_REGISTRATION,
    VERA_REGISTRATION,
)

__all__ = [
    "PEFT_METHOD_REGISTRATIONS",
    "abba",
    "boft",
    "dylora",
    "glora",
    "ia3",
    "loha",
    "locon",
    "lokr",
    "lora",
    "oft",
    "tlora",
    "vera",
]
