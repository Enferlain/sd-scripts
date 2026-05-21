"""Retired SDXL PEFT training entrypoint.

The active PEFT path now runs through the Hydra launcher and central metadata
backbone. This deprecated script is intentionally fenced instead of kept as a
partial compatibility copy.
"""

_DEPRECATED_SCRIPT_ERROR = (
    "scripts/_deprecated/sdxl_peft_copy.py is retired and no longer tracks the active "
    "metadata backbone. Use `uv run python train.py --config-name=presets/sdxl_peft` instead."
)


# These functions document the old public entrypoints; the module-level guard
# below intentionally prevents import or execution.
def train(*_args, **_kwargs):
    raise RuntimeError(_DEPRECATED_SCRIPT_ERROR)


def main(*_args, **_kwargs):
    raise RuntimeError(_DEPRECATED_SCRIPT_ERROR)


raise RuntimeError(_DEPRECATED_SCRIPT_ERROR)
