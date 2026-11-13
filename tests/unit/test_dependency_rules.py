"""D21's dependency rules, checked over the new system's modules (``tasks.md`` 13.2).

The rules make D15's direction exact. They apply to the modules in D21's
module map from their first line; today's code is never edited to follow
them. ``NEW_SYSTEM`` lists what exists of the new system so far, and each
section that adds a package adds it here.

Besides each layer's rule, new code never imports the code it replaces,
research code, or a ``_deprecated/`` folder, and no production module imports
the test-only accepted-run builder (D1).
"""

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Layer:
    """What one layer of D21's map may import from this repository and from third parties.

    ``third_party`` of ``None`` leaves third-party imports unrestricted.
    ``torch_only_in`` names the one package inside the layer that may load Torch.
    """

    package: str
    imports: tuple[str, ...]
    third_party: tuple[str, ...] | None = None
    torch_only_in: str | None = None


LIBRARIES = (
    "library.artifacts",
    "library.performance",
    "library.models",
    "library.conditioning",
    "library.losses",
    "library.timesteps",
    "library.objectives",
    "library.data",
    "library.optimization",
    "library.adapters",
    "library.pipelines",
    "library.evaluation",
)
FACTS = "library.observability.facts"

LAYERS = (
    Layer("library.iro", ("library.iro", FACTS, "library.base"), third_party=("xdsl",), torch_only_in="library.iro.devices"),
    Layer("library.runstate", ("library.runstate", "library.iro", FACTS, "library.base"), third_party=()),
    Layer("library.training", ("library.training", "library.runstate", "library.iro", *LIBRARIES, "library.observability", "library.base")),
    Layer(
        "library.strategies",
        ("library.strategies", "library.training", "library.runstate", "library.iro", *LIBRARIES, "library.observability", "library.base"),
    ),
    Layer("library.observability", ("library.observability", "library.base")),
    Layer("library.base", ("library.base",)),
)

# What exists of the new system: package directories, and test directories that are new code too.
NEW_SYSTEM = ("library/iro",)
NEW_TESTS = ("tests/unit/iro", "tests/unit/test_dependency_rules.py")

# Code the new system replaces or must not depend on (D21's coverage of today's code).
REPLACED = (
    "docs_design",
    "library.training.runners",
    "library.training.phases",
    "library.training.modes",
    "library.training.checkpointing",
    "library.training.execution",
    "library.training.diffusion",
    "library.training.noise_utils",
    "library.training.metadata",
    "library.training.sample_generation",
    "library.training.trainer_utils",
    "library.strategies.base",
    "library.strategies.factory",
    "library.strategies.shared",
    "library.config",
)
TEST_ONLY = "tests"


def modules(entry: str) -> list[Path]:
    location = ROOT / entry
    if location.is_file():
        return [location]
    return sorted(p for p in location.rglob("*.py") if "_deprecated" not in p.parts)


def dotted(file: Path) -> str:
    parts = file.relative_to(ROOT).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def imports(file: Path, name: str | None = None) -> list[tuple[int, str]]:
    """Every imported name, with its line, resolving relative imports against the module's name.

    ``from a.b import c`` gives ``a.b.c``, so importing a module from its
    package is checked as that module. Imports by a literal name through
    ``importlib.import_module`` or ``__import__`` count too.
    """
    tree = ast.parse(file.read_text(), filename=str(file))
    name = name or dotted(file)
    package = name if file.name == "__init__.py" else name.rpartition(".")[0]
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
                module = f"{base}.{node.module}" if node.module else base
            else:
                module = node.module or ""
            found.extend((node.lineno, f"{module}.{alias.name}" if alias.name != "*" else module) for alias in node.names)
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            called = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if called in ("import_module", "__import__"):
                found.append((node.lineno, node.args[0].value))
    return found


def within(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


def layer_of(module: str) -> Layer:
    matches = [layer for layer in LAYERS if within(module, layer.package)]
    assert matches, f"{module} belongs to no layer of D21's map"
    return max(matches, key=lambda layer: len(layer.package))


def is_standard(module: str) -> bool:
    return module.split(".")[0] in sys.stdlib_module_names or module == "__future__"


def problems_in(file: Path, *, production: bool, name: str | None = None) -> list[str]:
    name = name or dotted(file)
    layer = layer_of(name) if production else None
    found = []
    for line, module in imports(file, name):
        where = f"{name}:{line} imports {module}"
        if "_deprecated" in module.split("."):
            found.append(f"{where}: a _deprecated folder")
        if any(within(module, replaced) for replaced in REPLACED):
            found.append(f"{where}: code the new system replaces")
        if layer is None:
            continue
        if within(module, TEST_ONLY):
            found.append(f"{where}: test-only code")
        elif within(module, "library"):
            if not any(within(module, allowed) for allowed in layer.imports):
                found.append(f"{where}: {layer.package} may import only {', '.join(layer.imports)}")
        elif not is_standard(module):
            top = module.split(".")[0]
            if top == "torch" and layer.torch_only_in is not None:
                if not within(name, layer.torch_only_in):
                    found.append(f"{where}: Torch loads only in {layer.torch_only_in}")
            elif layer.third_party is not None and top not in layer.third_party:
                allowed = ", ".join(layer.third_party) or "no third-party package"
                found.append(f"{where}: {layer.package} may import only {allowed}")
    return found


@pytest.mark.parametrize("entry", NEW_SYSTEM)
def test_new_packages_follow_their_layers_rule(entry: str) -> None:
    files = modules(entry)
    assert files, f"{entry} has no modules"
    found = [problem for file in files for problem in problems_in(file, production=True)]
    assert found == [], "\n".join(found)


@pytest.mark.parametrize("entry", NEW_TESTS)
def test_new_tests_import_no_replaced_code(entry: str) -> None:
    found = [problem for file in modules(entry) for problem in problems_in(file, production=False)]
    assert found == [], "\n".join(found)


def test_no_library_module_imports_the_test_only_builder() -> None:
    found = [f"{file.relative_to(ROOT)}:{line}" for file in modules("library") for line, module in imports(file) if within(module, "tests")]
    assert found == []


def test_the_rules_catch_what_they_forbid(tmp_path: Path) -> None:
    """The checker itself: each kind of forbidden import is reported."""
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import torch\n"
        "import numpy\n"
        "from library.training.runners import trainer\n"
        "from library.models import sd\n"
        "from library.data._deprecated import old\n"
        "from tests.unit.iro.accepted_runs import accepted_run\n"
        "from . import text\n"
        "from library import iro\n"
        "import importlib\n"
        "importlib.import_module('torch.cuda')\n"
    )
    found = problems_in(probe, production=True, name="library.iro.probe")
    joined = "\n".join(found)
    assert "Torch loads only in library.iro.devices" in joined
    assert "may import only xdsl" in joined
    assert "code the new system replaces" in joined
    assert "library.iro may import only" in joined
    assert "a _deprecated folder" in joined
    assert "test-only code" in joined
    assert "library.iro.text" not in joined, "a relative import inside the package is allowed"
    assert "imports library.iro\n" not in joined + "\n", "importing the package from library is allowed"
    assert "imports torch.cuda" in joined, "a dynamic import by literal name is checked"


def test_a_replaced_module_imported_from_its_package_is_caught(tmp_path: Path) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text("from library.training import runners\nfrom library.strategies import factory\n")
    found = problems_in(probe, production=True, name="library.strategies.probe")
    assert len([f for f in found if "code the new system replaces" in f]) == 2
