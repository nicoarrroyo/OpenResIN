import os
from pathlib import Path

import pytest

from openresin import krisp_config, nalira_config

CONFIGS = [krisp_config, nalira_config]
CONFIG_IDS = ["krisp_config", "nalira_config"]


@pytest.mark.parametrize("config", CONFIGS, ids=CONFIG_IDS)
def test_home_dir_is_repo_root(config):
    """HOME_DIR is the repo root, not src/ and not the package directory.

    The move to src/openresin/ left this one level short, pointing at src/, so
    DATA_DIR resolved to a src/data that has never existed. Nothing caught it
    because every stage chdir'd on import and rebuilt its paths from the cwd.
    """
    home = Path(config.HOME_DIR)
    package = Path(config.PKG_DIR)

    assert package == home / "src" / "openresin", (
        f"PKG_DIR {package} is not <repo>/src/openresin relative to "
        f"HOME_DIR {home}")
    assert (home / "pyproject.toml").is_file(), (
        f"HOME_DIR {home} has no pyproject.toml, so it is not the repo root")


@pytest.mark.parametrize("config", CONFIGS, ids=CONFIG_IDS)
def test_data_dir_exists(config):
    """DATA_DIR points somewhere real. Cheap, and it is the path that broke."""
    assert Path(config.DATA_DIR).is_dir(), (
        f"DATA_DIR does not exist: {config.DATA_DIR}")


def test_both_configs_agree_on_home_dir():
    """The two configs compute the same root independently.

    They duplicate the anchor until Phase 2 merges them, so this fails if one
    is edited and the other is not.
    """
    assert krisp_config.HOME_DIR == nalira_config.HOME_DIR


@pytest.mark.parametrize("config", CONFIGS, ids=CONFIG_IDS)
def test_paths_are_absolute(config):
    """A relative path here would silently re-acquire the cwd dependency the
    package just had removed."""
    relative = [
        name for name in dir(config)
        if name.endswith("_DIR") and not os.path.isabs(getattr(config, name))
    ]
    assert not relative, f"not absolute: {relative}"


@pytest.mark.parametrize("config", CONFIGS, ids=CONFIG_IDS)
def test_paths_survive_a_changed_working_directory(config, tmp_path, monkeypatch):
    """Re-importing from an unrelated cwd yields identical paths.

    This is the property the whole chdir removal exists to provide, so it is
    the one worth asserting directly.
    """
    import importlib

    expected = {name: getattr(config, name)
                for name in dir(config) if name.endswith("_DIR")}

    monkeypatch.chdir(tmp_path)
    reloaded = importlib.reload(config)

    assert {name: getattr(reloaded, name) for name in expected} == expected


def test_config_not_lazy():
    """Both configs (soon to be one) ship ready to run the pipeline as intended.

    Scaled-down runs can be configured by CLI flags. Not by editing the
    config files.
    """

    SHIPPED_VALUES = [
        (krisp_config, "SAVE_MODEL", True),
        (krisp_config, "EPOCHS", 150),
        (nalira_config, "N_IMAGES", -1),
        (nalira_config, "HIGH_RES", True),
        (nalira_config, "RES", "10m"),
        (nalira_config, "KNOWN_FEATURE_MASKING", True),
        (nalira_config, "CLOUD_MASKING", True),
        (nalira_config, "COMPOSITING", True),
        (nalira_config, "LABEL_DATA", True),
        (nalira_config, "SHOW_INDEX_PLOTS", False),
        (nalira_config, "SAVE_IMAGES", False),
    ]

    wrong = [
        f"{module.__name__.rpartition('.')[2]}.{name} "
        f"is {getattr(module, name)!r}, ships as {expected!r}"
        for module, name, expected in SHIPPED_VALUES
        if getattr(module, name) != expected
    ]
    assert not wrong, "config is scaled down:\n  " + "\n  ".join(wrong)
