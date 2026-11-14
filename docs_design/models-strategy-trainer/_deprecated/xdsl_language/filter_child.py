"""Fresh cold entry, deliberately unable to import fixture construction."""

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
        if fullname.startswith("xdsl_language.") and fullname.rsplit(".", 1)[-1].startswith(
            ("filter_case", "fixtures", "qualification_cases", "test_")
        ):
            raise ImportError("cold filter child forbids authoring: " + fullname)
        return None


def main() -> None:
    before = sys._is_gil_enabled()
    sys.meta_path.insert(0, NoAuthoring())
    from xdsl.utils.exceptions import VerifyException
    from .filter_contracts import INITIAL
    from .filter_runtime import execute
    from .filter_target import installed
    from .prepare import Artifact, Policy, load
    from .target import Runtime

    artifact = Artifact.from_json(Path(sys.argv[1]).read_text())
    choices = json.loads(Path(sys.argv[2]).read_text())
    policy = Policy(workers=3, fuse_local=choices["fused"])
    selected = installed()
    factories = []
    report: dict

    def counted(binding):
        def initialize(value):
            factories.append(binding.spec.kind)
            return binding.initialize(value)

        return replace(binding, initialize=initialize)

    if choices.get("corrupt"):
        selected = replace(selected, resources=tuple(counted(item) for item in selected.resources))
        try:
            plan = load(artifact, selected.manifest, policy)
            Runtime(plan, selected, INITIAL)
        except (VerifyException, ValueError) as failed:
            report = {"rejected": True, "reason": str(failed), "factory_calls": factories}
        else:
            report = {"rejected": False, "factory_calls": factories}
    else:
        plan = load(artifact, selected.manifest, policy)
        owner, _, choices_observed = asyncio.run(
            execute(
                plan, choices["scenario"], erase_history=True, seed=choices.get("seed"), wrong_double=choices.get("wrong_double", False)
            )
        )
        report = {"owner": owner, "choices": choices_observed}
    blocked = []
    for name in ("filter_case", "fixtures", "qualification_cases"):
        try:
            importlib.import_module("xdsl_language." + name)
        except ImportError:
            blocked.append(name)
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
        "authoring_blocked": blocked,
        "authoring_loaded": any(
            name.startswith("xdsl_language.")
            and name.rsplit(".", 1)[-1].startswith(("filter_case", "fixtures", "qualification_cases", "test_"))
            for name in sys.modules
        ),
        "imports": sorted(sys.modules),
    }
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
