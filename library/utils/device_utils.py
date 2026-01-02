import functools
import gc
import torch

import contextlib

with contextlib.suppress(Exception):
    # intel gpu support for pytorch older than 2.5
    # ipex is not needed after pytorch 2.5
    import intel_extension_for_pytorch as ipex  # noqa: F401

try:
    HAS_CUDA = torch.cuda.is_available()
except Exception:
    HAS_CUDA = False

try:
    HAS_MPS = torch.backends.mps.is_available()
except Exception:
    HAS_MPS = False

try:
    HAS_XPU = torch.xpu.is_available()
except Exception:
    HAS_XPU = False


def clean_memory():
    """
    Cleans up memory by collecting garbage and emptying cache on available devices (CUDA, XPU, MPS).
    """
    gc.collect()
    if HAS_CUDA:
        torch.cuda.empty_cache()
    if HAS_XPU:
        torch.xpu.empty_cache()
    if HAS_MPS:
        torch.mps.empty_cache()


def clean_memory_on_device(device: str | torch.device | None):
    """
    Cleans memory on the specified device.

    This function collects garbage and empties the cache for the specified device type.

    Args:
        device (Optional[Union[str, torch.device]]): The device to clean memory on.
    """
    gc.collect()
    if device is None:
        return
    if isinstance(device, str):
        device = torch.device(device)
    # device may "cuda" or "cuda:0", so we need to check the type of device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()


def synchronize_device(device: str | torch.device | None):
    """
    Synchronizes the specified device.

    Args:
        device (Optional[Union[str, torch.device]]): The device to synchronize.
    """
    if device is None:
        return
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@functools.cache
def get_preferred_device() -> torch.device:
    """
    Gets the preferred device for the current environment.

    Checks for CUDA, XPU, and MPS availability in that order. Returns CPU if none are available.

    Note:
        Do not call this function from training scripts. Use accelerator.device instead.

    Returns:
        torch.device: The preferred device.
    """
    if HAS_CUDA:
        device = torch.device("cuda")
    elif HAS_XPU:
        device = torch.device("xpu")
    elif HAS_MPS:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"get_preferred_device() -> {device}")
    return device


def init_ipex():
    """
    Apply IPEX to CUDA hijacks using `library.ipex.ipex_init`.

    This function should run right after importing torch and before doing anything else.

    If xpu is not available, this function does nothing.
    """
    try:
        if HAS_XPU:
            from library.performance.ipex import ipex_init

            is_initialized, error_message = ipex_init()
            if not is_initialized:
                print("failed to initialize ipex:", error_message)
        else:
            return
    except Exception as e:
        print("failed to initialize ipex:", e)
