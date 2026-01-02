import hashlib
import os
import subprocess
from io import BytesIO
from typing import Literal

import safetensors.torch

# Supported hash algorithms
# blake3 requires optional 'blake3' package: pip install blake3
HASH_ALGORITHMS = ["md5", "sha1", "sha256", "sha512", "blake3"]
HashAlgorithm = Literal["md5", "sha1", "sha256", "sha512", "blake3"]

# Default algorithm (matches ComfyUI and ModelSpec standard)
DEFAULT_HASH_ALGORITHM = "sha256"


def calculate_hash(filename: str, algorithm: HashAlgorithm = DEFAULT_HASH_ALGORITHM) -> str:
    """
    Calculate file hash with configurable algorithm.

    Args:
        filename: Path to file to hash
        algorithm: Hash algorithm - 'md5', 'sha1', 'sha256', 'sha512', or 'blake3'
                   blake3 requires: pip install blake3

    Returns:
        Hex digest string, or error string if file not accessible

    Performance notes (6GB file on fast NVMe, modern 64-bit CPU):
        - blake3:  ~2-3 seconds (fastest, parallelized, saturates NVMe bandwidth)
        - sha512:  ~8-12 seconds (faster than sha256 on 64-bit due to native 64-bit ops)
        - sha256:  ~10-15 seconds (standard, has hardware acceleration on modern CPUs)
        - sha1:    ~10-15 seconds (no hardware accel, deprecated for security)
        - md5:     ~10-15 seconds (no hardware accel, deprecated for security)

    For maximum speed, use blake3. SHA-256 is recommended for compatibility with
    existing tools (A1111, ComfyUI, ModelSpec standard).
    """
    try:
        if algorithm == "blake3":
            try:
                import blake3

                hasher = blake3.blake3()
            except ImportError:
                raise ImportError("blake3 algorithm requires the blake3 package. Install with: pip install blake3") from None
        else:
            hasher = hashlib.new(algorithm)

        blksize = 1024 * 1024  # 1MB chunks

        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(blksize), b""):
                hasher.update(chunk)

        return hasher.hexdigest()

    except FileNotFoundError:
        return "NOFILE"
    except IsADirectoryError:
        return "IsADirectory"
    except PermissionError:
        return "IsADirectory"


def calculate_sha256(filename: str) -> str:
    """
    Calculate SHA-256 hash of a file.

    Legacy function - prefer calculate_hash(filename, 'sha256') for new code.
    Kept for backward compatibility with existing callers.
    """
    return calculate_hash(filename, "sha256")


def model_hash(filename):
    """
    Old model hash used by stable-diffusion-webui (partial file, first 8 chars).

    Args:
        filename: Path to the model file.

    Returns:
        The partial SHA256 hash or error string.
    """
    try:
        with open(filename, "rb") as file:
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return "NOFILE"
    except IsADirectoryError:  # Linux?
        return "IsADirectory"
    except PermissionError:  # Windows
        return "IsADirectory"


def precalculate_safetensors_hashes(tensors, metadata):
    """
    Precalculate the model hashes needed by sd-webui-additional-adapters to
    save time on indexing the model later.

    Args:
        tensors: The model tensors.
        metadata: The model metadata.

    Returns:
        tuple: (model_hash, legacy_hash)
    """

    # Because writing user metadata to the file can change the result of
    # sd_models.model_hash(), only retain the training metadata for purposes of
    # calculating the hash, as they are meant to be immutable
    metadata = {k: v for k, v in metadata.items() if k.startswith("ss_")}

    bytes = safetensors.torch.save(tensors, metadata)
    b = BytesIO(bytes)

    model_hash = addnet_hash_safetensors(b)
    legacy_hash = addnet_hash_legacy(b)
    return model_hash, legacy_hash


def addnet_hash_legacy(b):
    """
    Old model hash used by sd-webui-additional-adapters for .safetensors format files.

    Args:
        b: Bytes buffer of the safetensors file.

    Returns:
        The partial SHA256 hash.
    """
    m = hashlib.sha256()

    b.seek(0x100000)
    m.update(b.read(0x10000))
    return m.hexdigest()[0:8]


def addnet_hash_safetensors(b):
    """
    New model hash used by sd-webui-additional-adapters for .safetensors format files.

    Args:
        b: Bytes buffer of the safetensors file.

    Returns:
        The SHA256 hash.
    """
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def get_git_revision_hash() -> str:
    """
    Retrieves the current git revision hash.

    Returns:
        str: The current git revision hash or "(unknown)" if unavailable.
    """
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)).decode("ascii").strip()
    except Exception:
        return "(unknown)"
