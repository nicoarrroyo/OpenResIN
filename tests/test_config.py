import os
from pathlib import Path

from openresin import config


def test_home_dir_is_repo_root():
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


def test_data_dir_exists():
    """DATA_DIR points somewhere real. Cheap, and it is the path that broke."""
    assert Path(config.DATA_DIR).is_dir(), (
        f"DATA_DIR does not exist: {config.DATA_DIR}")


def test_paths_are_absolute():
    """A relative path here would silently re-acquire the cwd dependency the
    package just had removed."""
    relative = [
        name for name in dir(config)
        if name.endswith("_DIR") and not os.path.isabs(getattr(config, name))
    ]
    assert not relative, f"not absolute: {relative}"


def test_paths_survive_a_changed_working_directory(tmp_path, monkeypatch):
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
    """Single config ships ready to run the pipeline as intended.

    Scaled-down runs can be configured by CLI flags. Not by editing the
    config files.
    """

    shipped_values = [
        ("SAVE_MODEL", True),
        ("EPOCHS", 150),
        ("N_IMAGES", -1),
        ("HIGH_RES", True),
        ("RES", "10m"),
        ("KNOWN_FEATURE_MASKING", True),
        ("CLOUD_MASKING", True),
        ("COMPOSITING", True),
        ("LABEL_DATA", True),
        ("SHOW_INDEX_PLOTS", False),
        ("SAVE_IMAGES", False),
    ]

    wrong = [
        f"config.{name} "
        f"is {getattr(config, name)!r}, ships as {expected!r}"
        for name, expected in shipped_values
        if getattr(config, name) != expected
    ]
    assert not wrong, "config is scaled down:\n  " + "\n  ".join(wrong)
