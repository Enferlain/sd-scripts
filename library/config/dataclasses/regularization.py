from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RegularizationConfig:
    noise_offset: Optional[float] = None
    noise_offset_random_strength: bool = False
    multires_noise_iterations: Optional[int] = None
    ip_noise_gamma: Optional[float] = None
    ip_noise_gamma_random_strength: bool = False
    multires_noise_discount: float = 0.3
    adaptive_noise_scale: Optional[float] = None
    zero_terminal_snr: bool = False

    def __post_init__(self):
        if self.adaptive_noise_scale is not None and self.noise_offset is None:
            raise ValueError("adaptive_noise_scale requires noise_offset")
