"""Cold retained-product execution without fixture construction or stand-ins."""

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


BLOCKED = ("product_case", "product_reference", "fixtures", "test_", "qualification")


class NoAuthoring(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("xdsl_language.") and fullname.rsplit(".", 1)[-1].startswith(BLOCKED):
            raise ImportError("cold product child forbids authoring: " + fullname)
        return None


def main() -> None:
    before = sys._is_gil_enabled()
    sys.meta_path.insert(0, NoAuthoring())
    from xdsl.utils.exceptions import VerifyException
    from .prepare import Artifact, Policy, load
    from .product_bindings import installed
    from .product_contracts import initial
    from .product_runtime import execute
    from .target import Runtime

    artifact = Artifact.from_json(Path(sys.argv[1]).read_text())
    options = json.loads(Path(sys.argv[2]).read_text())
    kind, derived = options["kind"], options["derived"]
    selected = installed(kind, derived=derived)
    factories = []
    report: dict
    if options.get("corrupt"):

        def counted(binding):
            def initialize(value):
                factories.append(binding.spec.kind)
                return binding.initialize(value)

            return replace(binding, initialize=initialize)

        selected = replace(selected, resources=tuple(counted(binding) for binding in selected.resources))
        try:
            plan = load(artifact, selected.manifest, Policy(fuse_local=True))
            Runtime(plan, selected, initial(kind))
        except (VerifyException, ValueError) as failed:
            report = {"rejected": True, "reason": str(failed), "factories": factories}
        else:
            report = {"rejected": False, "factories": factories}
    else:
        plan = load(artifact, selected.manifest, Policy(fuse_local=True))
        report = asyncio.run(execute(plan, kind, derived, options["scenario"], wrong=options.get("wrong", False)))
    blocked = []
    for name in ("product_case", "product_reference", "fixtures"):
        try:
            importlib.import_module("xdsl_language." + name)
        except ImportError:
            blocked.append(name)
    import torch
    import xdsl

    report["environment"] = {
        "python": sys.version,
        "torch": torch.__version__,
        "xdsl": str(xdsl.__version__),
        "gil_before": before,
        "gil_after": sys._is_gil_enabled(),
        "forced_gil": os.getenv("PYTHON_GIL"),
        "build_gil_disabled": sysconfig.get_config_var("Py_GIL_DISABLED"),
        "optimized": not __debug__,
        "blocked": blocked,
        "authoring_loaded": any(name.startswith("xdsl_language.") and name.rsplit(".", 1)[-1].startswith(BLOCKED) for name in sys.modules),
        "imports": sorted(sys.modules),
    }
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
