from dataclasses import dataclass

from library.config.dataclasses.peft import PeftConfig


@dataclass
class AdapterConfig:
    """Adapter training config rooted by adapter family."""

    peft: PeftConfig | None = None
