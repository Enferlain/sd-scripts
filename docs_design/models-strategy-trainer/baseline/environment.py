"""Print the interpreter, backend and installed packages of the current environment as JSON."""

import importlib.metadata
import json
import platform
import sys

import torch

gil = getattr(sys, "_is_gil_enabled", None)
print(
    json.dumps(
        {
            "python": sys.version,
            "platform": platform.platform(),
            "gil_enabled": gil() if gil is not None else True,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "torch_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
            "packages": sorted(f"{item.metadata['Name']}=={item.version}" for item in importlib.metadata.distributions()),
        },
        indent=1,
    )
)
