import ast
import os
from pathlib import Path

import pytest

PKG_DIR = Path(__file__).resolve().parents[1] / "src" / "openresin"

# Modules with no import-time side effects, so a test can safely import them.
# The four pipeline scripts joined this list once their bodies moved into
# main(): before that, importing label.py opened a tkinter GUI and importing
# train.py started a training run.
IMPORTABLE = [
    "data_handling",
    "evaluate",
    "image_handling",
    "inference",
    "krisp_config",
    "label",
    "misc",
    "modelling",
    "nalira_config",
    "predict",
    "train",
    "user_interfacing",
]

# The stages pyproject.toml promises as console scripts. Each must expose a
# main() for `pip install -e .` to generate a working launcher.
STAGES = ["label", "train", "predict", "evaluate"]


def _parse_all():
    """Every module in the package, as {name: AST}."""
    return {
        path.stem: ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for path in sorted(PKG_DIR.glob("*.py"))
    }


def _bound_at_module_level(tree):
    """Names a module binds at module level.

    Covers defs, classes, assignments and imports, including those nested one
    level inside if/try/for/while; krisp_config and others define names that
    way, and treating them as missing would be a false positive.
    """
    names = set()

    def add_stmt(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])

    for node in tree.body:
        add_stmt(node)
        if isinstance(node, (ast.If, ast.Try, ast.For, ast.While)):
            for inner in ast.walk(node):
                add_stmt(inner)
    return names


def test_relative_imports_resolve():
    """Every `from .module import name` names something that exists.

    This is the check that would have caught save_image_file and
    get_sentinel_bands: both were deleted from their defining module while
    inference.py went on importing them, leaving the module unimportable and
    the whole predict path dead. Python only reports the first failure per
    module, so this collects all of them at once.
    """
    modules = _parse_all()
    bound = {name: _bound_at_module_level(tree) for name, tree in modules.items()}

    broken = []
    for module_name, tree in modules.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level != 1:
                continue
            if node.module is None:  # `from . import x`, so x is a module
                for alias in node.names:
                    if alias.name not in modules:
                        broken.append(
                            f"{module_name}.py:{node.lineno} "
                            f"no module .{alias.name}")
                continue
            if node.module not in modules:
                broken.append(
                    f"{module_name}.py:{node.lineno} no module .{node.module}")
                continue
            for alias in node.names:
                if alias.name not in bound[node.module]:
                    broken.append(
                        f"{module_name}.py:{node.lineno} "
                        f"{alias.name!r} not defined in .{node.module}")

    assert not broken, "broken intra-package imports:\n  " + "\n  ".join(broken)


def test_no_working_directory_manipulation():
    """No os.chdir anywhere in the package.

    Guards commit 852e7f8. Every stage used to chdir on import, which is what
    made the package impossible to import, test, or run from anywhere but one
    directory. A single reintroduced chdir undoes that for the whole package.
    """
    offenders = []
    for module_name, tree in _parse_all().items():
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "chdir"):
                offenders.append(f"{module_name}.py:{node.lineno}")

    assert not offenders, (
        "os.chdir reintroduced at:\n  " + "\n  ".join(offenders)
        + "\nDerive paths from the config anchors instead.")


@pytest.mark.parametrize("module_name", IMPORTABLE)
def test_imports_without_side_effects(module_name):
    """Importing a module must not run a pipeline or move the process.

    A changed working directory is the specific symptom being guarded: it is
    invisible until some later, unrelated path lookup resolves against the
    wrong root.
    """
    import importlib

    before = os.getcwd()
    importlib.import_module(f"openresin.{module_name}")
    assert os.getcwd() == before, (
        f"importing openresin.{module_name} changed the working directory "
        f"from {before} to {os.getcwd()}")


@pytest.mark.parametrize("module_name", STAGES)
def test_stage_exposes_main(module_name):
    """Every console script in pyproject.toml resolves to a callable main().

    `openresin-label = "openresin.label:main"` is a promise pip keeps by
    importing the module and calling that attribute, so a missing or renamed
    main() breaks the installed command while the module itself still imports
    cleanly. Nothing else would notice.
    """
    import importlib

    module = importlib.import_module(f"openresin.{module_name}")
    assert callable(getattr(module, "main", None)), (
        f"openresin.{module_name} has no callable main(), so its console "
        f"script in pyproject.toml would fail on invocation")


@pytest.mark.parametrize("module_name", STAGES)
def test_stage_help_does_not_run_the_pipeline(module_name):
    """--help must print usage and exit, without touching data or the cwd.

    This is the cheap end-to-end check that the argparse conversion actually
    took: a stage that still did its work at import time would run it here.
    """
    import importlib

    module = importlib.import_module(f"openresin.{module_name}")
    before = os.getcwd()

    with pytest.raises(SystemExit) as excinfo:
        module.build_parser().parse_args(["--help"])

    assert excinfo.value.code == 0
    assert os.getcwd() == before
