"""Repo-owned PEFT adapter method runtimes."""

from .loha import REGISTRATION as LOHA_REGISTRATION
from .lokr import REGISTRATION as LOKR_REGISTRATION
from .lora import REGISTRATION as LORA_REGISTRATION


PEFT_METHOD_REGISTRATIONS = (
    LOHA_REGISTRATION,
    LOKR_REGISTRATION,
    LORA_REGISTRATION,
)

__all__ = [
    "PEFT_METHOD_REGISTRATIONS",
    "dylora_deprecated",
    "loha",
    "lokr",
    "lora",
    "oft_deprecated",
]
