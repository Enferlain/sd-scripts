"""Fresh-process runtime entry: fixture authoring is deliberately unavailable."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
import sysconfig

from dataclasses import replace
from importlib.abc import MetaPathFinder
from pathlib import Path


class NoAuthoring(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("xdsl_language.") and fullname.rsplit(".", 1)[-1].startswith(("fixtures", "qualification_cases", "test_")):
            raise ImportError("qualification child forbids authoring: " + fullname)
        return None


def main() -> None:
    before = sys._is_gil_enabled()
    sys.meta_path.insert(0, NoAuthoring())
    from .prepare import Artifact, Policy, load
    from .qualification_runtime import execute
    from .qualification_target import selected
    from .target import Runtime

    choices = json.loads(Path(sys.argv[2]).read_text())
    artifact = Artifact.from_json(Path(sys.argv[1]).read_text())
    installed = selected(wrong_score=choices.get("wrong_score", False))
    factory_calls = []

    def counted(binding):
        def initialize(value):
            factory_calls.append(binding.spec.kind)
            return binding.initialize(value)

        return replace(binding, initialize=initialize)

    installed = replace(installed, resources=tuple(counted(item) for item in installed.resources))
    report: dict
    if choices.get("corrupt"):
        try:
            plan = load(artifact, installed.manifest, Policy())
            Runtime(plan, installed, choices["initial"])
        except ValueError as failed:
            report = {"rejected": True, "reason": str(failed), "factory_calls": factory_calls}
        else:
            report = {"rejected": False, "factory_calls": factory_calls}
    else:
        plan = load(artifact, installed.manifest, Policy())
        _, report = asyncio.run(
            execute(
                plan,
                choices["initial"],
                prefix=choices.get("prefix", ""),
                seed=choices["seed"],
                mode=choices["mode"],
                erased=True,
                wrong_score=choices.get("wrong_score", False),
            )
        )
    try:
        importlib.import_module("xdsl_language.fixtures")
    except ImportError:
        authoring_blocked = True
    else:
        authoring_blocked = False
    import torch
    import xdsl

    report["environment"] = {
        "python": sys.version,
        "optimized": not __debug__,
        "build_gil_disabled": sysconfig.get_config_var("Py_GIL_DISABLED"),
        "forced_gil": os.getenv("PYTHON_GIL"),
        "gil_before": before,
        "gil_after": sys._is_gil_enabled(),
        "torch": torch.__version__,
        "xdsl": str(xdsl.__version__),
        "authoring_blocked": authoring_blocked,
        "authoring_loaded": any(name in sys.modules for name in ("xdsl_language.fixtures", "xdsl_language.qualification_cases")),
        "imports": sorted(sys.modules),
    }
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
