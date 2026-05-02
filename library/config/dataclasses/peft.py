from dataclasses import dataclass, field

from library.adapters.methods.peft.boft.config import PeftBoftConfig
from library.adapters.methods.peft.dylora.config import PeftDyloraConfig
from library.adapters.methods.peft.glora.config import PeftGloraConfig
from library.adapters.methods.peft.ia3.config import PeftIa3Config
from library.adapters.methods.peft.loha.config import PeftLohaConfig
from library.adapters.methods.peft.locon.config import PeftLoconConfig
from library.adapters.methods.peft.lokr.config import PeftLokrConfig
from library.adapters.methods.peft.lora.config import PeftLoraConfig
from library.adapters.methods.peft.oft.config import PeftOftConfig

VALID_PEFT_CONTINUE_MODES = ("strict", "initialize_from_artifact")


_DEFAULT_LEGACY_ORTHOGRAD_TARGETS = (
    "lora_down.weight",
    "lora_up.weight",
    "lora_down1.weight",
    "lora_up1.weight",
    "lora_down2.weight",
    "lora_up2.weight",
    "a1.weight",
    "a2.weight",
    "b1.weight",
    "b2.weight",
    "c1.weight",
)


@dataclass
class PeftConfig:
    """PEFT family config plus exactly one active method branch."""

    method: str | None = field(default=None, metadata={"help": "Legacy method shim; prefer adapter.peft.<method>"})
    continue_from: str | None = field(default=None, metadata={"help": "Continue from an existing adapter artifact"})
    continue_mode: str | None = field(
        default=None,
        metadata={"help": "Continuation intent: strict or initialize_from_artifact"},
    )

    boft: PeftBoftConfig | None = None
    dylora: PeftDyloraConfig | None = None
    glora: PeftGloraConfig | None = None
    ia3: PeftIa3Config | None = None
    lora: PeftLoraConfig | None = None
    loha: PeftLohaConfig | None = None
    locon: PeftLoconConfig | None = None
    lokr: PeftLokrConfig | None = None
    oft: PeftOftConfig | None = None

    scale_weight_norms: float | None = field(
        default=None, metadata={"help": "Scale weight norms to prevent exploding gradients (1.0 recommended)"}
    )
    # Left over on purpose as a legacy reference surface for later orthograd
    # follow-up. The active repo-owned PEFT path does not currently consume it.
    orthograd_targets: list[str] | None = field(
        default_factory=lambda: list(_DEFAULT_LEGACY_ORTHOGRAD_TARGETS),
        metadata={"help": "Parameter names to apply orthogonal gradient to"},
    )

    # Compatibility-only normalization shim. These fields are no longer the
    # forward PEFT config interface and should not be treated as equal support
    # surfaces in new configs.
    adapter_module: str | None = field(default=None, metadata={"help": "Legacy adapter selection shim"})
    adapter_args: list[str] | None = field(default=None, metadata={"help": "Legacy key=value adapter settings shim"})
    adapter_weights: str | None = field(default=None, metadata={"help": "Legacy continuation shim"})
    adapter_rank_from_weights: bool = field(default=False, metadata={"help": "Legacy strict-continuation shim"})
    base_weights: list[str] | None = field(default=None, metadata={"help": "Legacy pre-merge shim"})
    base_weights_multiplier: list[float] | None = field(default=None, metadata={"help": "Legacy pre-merge multiplier shim"})
    training_comment: str | None = field(default=None, metadata={"help": "Legacy metadata shim; prefer output.metadata.training_comment"})
