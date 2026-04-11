import functools
import os
import shutil
from pathlib import Path

import torch


def _parse_version_key(version: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in version.split("."))
    except ValueError:
        return ()


def _check_msvc_layout(msvc_base_path: Path, version: str | None) -> bool:
    if not version:
        return False
    return all(
        path.exists()
        for path in (
            msvc_base_path / version / "bin" / "Hostx64" / "x64" / "cl.exe",
            msvc_base_path / version / "include" / "vcruntime.h",
            msvc_base_path / version / "lib" / "x64" / "vcruntime.lib",
        )
    )


def _find_latest_valid_msvc_version(msvc_base_path: Path) -> str | None:
    if not msvc_base_path.exists():
        return None

    versions = sorted(
        (entry.name for entry in msvc_base_path.iterdir() if entry.is_dir()),
        key=_parse_version_key,
        reverse=True,
    )
    for version in versions:
        if _check_msvc_layout(msvc_base_path, version):
            return version
    return None


def _discover_msvc_from_path() -> tuple[Path | None, str | None]:
    compiler_path = shutil.which("cl.exe") or shutil.which("cl")
    if not compiler_path:
        return None, None

    compiler = Path(compiler_path)
    for parent in compiler.parents:
        if parent.parent.name == "MSVC":
            msvc_base_path = parent.parent
            version = parent.name
            if _check_msvc_layout(msvc_base_path, version):
                return msvc_base_path, version
            break
    return None, None


def _discover_msvc_from_standard_locations() -> tuple[Path | None, str | None]:
    program_files_roots = (
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")),
        Path(os.environ.get("PROGRAMW6432", r"C:\Program Files")),
    )
    candidates: list[Path] = []
    for root in program_files_roots:
        visual_studio_root = root / "Microsoft Visual Studio"
        if visual_studio_root.exists():
            candidates.extend(visual_studio_root.glob("*/*/VC/Tools/MSVC"))

    for msvc_base_path in sorted(candidates, reverse=True):
        version = _find_latest_valid_msvc_version(msvc_base_path)
        if version is not None:
            return msvc_base_path, version
    return None, None


def _discover_msvc() -> tuple[Path | None, str | None]:
    vcinstall_dir = os.environ.get("VCINSTALLDIR")
    if vcinstall_dir:
        msvc_base_path = Path(vcinstall_dir) / "Tools" / "MSVC"
        version = os.environ.get("VCTOOLSVERSION") or _find_latest_valid_msvc_version(msvc_base_path)
        if _check_msvc_layout(msvc_base_path, version):
            return msvc_base_path, version

    msvc_base_path, version = _discover_msvc_from_path()
    if msvc_base_path is not None:
        return msvc_base_path, version

    return _discover_msvc_from_standard_locations()


def _check_windows_sdk_layout(winsdk_base_path: Path, version: str | None) -> bool:
    if not version:
        return False
    return all(
        path.exists()
        for path in (
            winsdk_base_path / "Include" / version / "ucrt" / "stdlib.h",
            winsdk_base_path / "Lib" / version / "ucrt" / "x64" / "ucrt.lib",
        )
    )


def _find_latest_valid_windows_sdk_version(winsdk_base_path: Path) -> str | None:
    include_path = winsdk_base_path / "Include"
    if not include_path.exists():
        return None

    versions = sorted(
        (entry.name.rstrip("\\/") for entry in include_path.iterdir() if entry.is_dir()),
        key=_parse_version_key,
        reverse=True,
    )
    for version in versions:
        if _check_windows_sdk_layout(winsdk_base_path, version):
            return version
    return None


def _discover_windows_sdk() -> tuple[Path | None, str | None]:
    windows_sdk_dir = os.environ.get("WINDOWSSDKDIR")
    if windows_sdk_dir:
        winsdk_base_path = Path(windows_sdk_dir)
        version = os.environ.get("WINDOWSSDKVERSION") or os.environ.get("WINDOWSSDKVER")
        version = version.rstrip("\\/") if version else None
        if not version:
            version = _find_latest_valid_windows_sdk_version(winsdk_base_path)
        if _check_windows_sdk_layout(winsdk_base_path, version):
            return winsdk_base_path, version

    default_sdk_root = Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")) / "Windows Kits" / "10"
    version = _find_latest_valid_windows_sdk_version(default_sdk_root)
    if version is not None:
        return default_sdk_root, version
    return None, None


@functools.lru_cache(maxsize=1)
def prepare_windows_compiler_env_for_torch_compile() -> None:
    """Fill in missing Windows MSVC/SDK env vars before Triton probes them."""
    if os.name != "nt":
        return

    msvc_base_path, msvc_version = _discover_msvc()
    if msvc_base_path is not None and msvc_version is not None:
        os.environ.setdefault("VCINSTALLDIR", str(msvc_base_path.parent.parent))
        os.environ.setdefault("VCToolsVersion", msvc_version)
        os.environ.setdefault("CC", str(msvc_base_path / msvc_version / "bin" / "Hostx64" / "x64" / "cl.exe"))

    winsdk_base_path, winsdk_version = _discover_windows_sdk()
    if winsdk_base_path is not None and winsdk_version is not None:
        os.environ.setdefault("WindowsSdkDir", str(winsdk_base_path))
        os.environ.setdefault("WindowsSDKVersion", f"{winsdk_version}\\")
        os.environ.setdefault("WindowsSDKVer", f"{winsdk_version}\\")


def can_use_compiled_cuda_helpers() -> bool:
    """Check whether the CUDA-backed compiled helper path is available."""
    prepare_windows_compiler_env_for_torch_compile()
    nvcc_path = shutil.which("nvcc")
    return bool(torch.cuda.is_available() and nvcc_path and os.access(nvcc_path, os.X_OK))
