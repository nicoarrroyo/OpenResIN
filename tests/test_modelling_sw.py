import json
import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from openresin import config as c
from openresin import label_sw
from openresin import labelling_sw
from openresin import modelling_sw


SCENES = (
    "S2C_MSIL2A_20260425T110621_N0512_R137_T31UCU_A.SAFE",
    "S2A_MSIL2A_20260427T110651_N0512_R137_T31UCU_B.SAFE",
    "S2B_MSIL2A_20260427T105619_N0512_R094_T31UCU_C.SAFE",
    "S2B_MSIL2A_20260430T110619_N0512_R137_T31UCU_D.SAFE",
)
TRAIN_IDS = (1, 3, 5, 7, 17, 19, 21, 23)
TEST_IDS = (41, 43, 45, 47)


def _features(shape, invalid=None):
    arrays = {}
    for position, name in enumerate(c.SW_FEATURES, start=1):
        rows, cols = np.indices(shape)
        arrays[name] = (
            position * 1000 + rows * 10 + cols).astype(np.float32)
    if invalid is not None:
        arrays["NDVI"][invalid] = np.nan
    return arrays


def _record(area_id, split, window, polygons, exclusions=None,
            completion=None):
    record = {
        "tile": "T31UCU",
        "area_id": area_id,
        "split": split,
        "window": list(window),
        "polygons": polygons,
        "exclusions": exclusions or [],
    }
    if completion is not None:
        record["completion"] = completion
    return record


def _rectangle(col0, row0, col1, row1, class_name="water", polygon_id=1):
    return {
        "id": polygon_id,
        "class": class_name,
        "vertices_scene": [
            [col0, row0], [col1, row0], [col1, row1], [col0, row1]
        ],
    }


def test_training_sampler_applies_both_caps_and_keeps_provenance(monkeypatch):
    monkeypatch.setattr(modelling_sw, "WATER_POLYGON_CAP", 4)
    monkeypatch.setattr(modelling_sw, "WATER_AREA_CAP", 5)
    window = [0, 5, 0, 5]
    features = _features((5, 5), invalid=(4, 4))
    record = _record(7, "train", window, [
        _rectangle(0, 0, 3, 2, polygon_id=11),
        _rectangle(2, 0, 5, 2, polygon_id=12),
        _rectangle(0, 3, 1, 4, "non-water", polygon_id=13),
    ])

    first, report = modelling_sw.sample_training_area(record, features)
    second, _ = modelling_sw.sample_training_area(record, features)

    assert np.array_equal(first["X"], second["X"])
    assert first["X"].dtype == np.float32
    assert first["X"].shape == (10, 6)
    assert np.array_equal(np.bincount(first["y"], minlength=2), [5, 5])
    for position, _name in enumerate(c.SW_FEATURES, start=1):
        assert np.array_equal(
            first["X"][:, position - 1],
            position * 1000 + first["row"] * 10 + first["col"])
    assert not np.any((first["row"] == 4) & (first["col"] == 4))
    assert set(first["polygon_index"]) >= {0, 1, 2}
    assert report["water"]["eligible"] == 10
    assert report["water"]["after_polygon_cap"] == 8
    assert report["water"]["selected"] == 5
    assert report["water"]["same_class_overlap_pixels"] == 2
    assert [item["polygon_index"] for item in report["water"]["polygons"]] \
        == [1, 2]
    assert sum(item["after_area_cap"]
               for item in report["water"]["polygons"]) == 5
    assert any(item["after_area_cap"] < item["after_polygon_cap"]
               for item in report["water"]["polygons"])


def test_test_sampler_uses_all_labels_and_explicit_background_provenance():
    window = [10, 13, 20, 23]
    features = _features((3, 3), invalid=(2, 2))
    record = _record(41, "test", window, [
        _rectangle(20, 10, 22, 11, polygon_id=1),
        _rectangle(20, 11, 21, 12, "non-water", polygon_id=2),
    ], exclusions=[
        {"vertices_scene": [[21, 11], [22, 11], [22, 12], [21, 12]]}
    ])

    dataset, report = modelling_sw.sample_test_area(record, features)

    assert dataset["X"].shape == (7, 6)
    assert report["eligible"] == 7
    assert set(dataset["y"]) == {0, 1}
    explicit = (dataset["row"] == 11) & (dataset["col"] == 20)
    background = (dataset["row"] == 12) & (dataset["col"] == 20)
    excluded = (dataset["row"] == 11) & (dataset["col"] == 21)
    assert dataset["polygon_index"][explicit].item() == 2
    assert dataset["polygon_index"][background].item() == 0
    assert not excluded.any()


def _write_reference_scene(root, scene_name, shape, transform, crs):
    band_dir = root / scene_name / "GRANULE" / "fixture" / "IMG_DATA" / "R10m"
    band_dir.mkdir(parents=True, exist_ok=True)
    band_path = band_dir / "fixture_B02_10m.tif"
    with rasterio.open(
            band_path, "w", driver="GTiff", height=shape[0], width=shape[1],
            count=1, dtype="uint16", transform=transform, crs=crs) as dst:
        dst.write(np.ones(shape, dtype=np.uint16), 1)
    return band_path


def _write_preparation_fixture(tmp_path, monkeypatch):
    input_dir = tmp_path / "inputs"
    source_root = tmp_path / "sat-images"
    input_dir.mkdir(parents=True)
    source_root.mkdir()
    monkeypatch.setattr(c, "SW_TILE_PX", 16)
    monkeypatch.setattr(c, "SW_CELL_PX", 2)
    monkeypatch.setattr(c, "SW_GRID_ROWS", 8)
    monkeypatch.setattr(c, "SW_GRID_COLS", 8)
    transform = Affine(10, 0, 300000, 0, -10, 5900040)
    for scene in SCENES:
        _write_reference_scene(
            source_root, scene, (16, 16), transform, "EPSG:32631")

    arrays = _features((16, 16))
    arrays["valid_count"] = np.ones((16, 16), dtype=np.int16)
    np.savez_compressed(input_dir / "features.npz", **arrays)
    provenance = {
        "tile": "T31UCU",
        "month": "2026-04",
        "source_scenes": sorted(SCENES),
        "feature_order": list(c.SW_FEATURES),
        "aggregation": "valid median within date, then median across dates",
        "masks": {
            "cloud_shadow_classes": list(c.SW_CLOUD_SHADOW_CLASSES),
            "nodata_value": c.SW_NODATA_VALUE,
            "sea_source": "/recorded/sea.geojson",
            "urban_source": "/recorded/urban.tif",
        },
        "crs": "EPSG:32631",
    }
    (input_dir / "features-provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8")

    area_entries = []
    feature_digest = labelling_sw.file_sha256(input_dir / "features.npz")
    provenance_digest = labelling_sw.file_sha256(
        input_dir / "features-provenance.json")
    for split, area_ids in (("train", TRAIN_IDS), ("test", TEST_IDS)):
        for area_id in area_ids:
            window = list(labelling_sw.grid_cell_window(area_id))
            row0, _, col0, _ = window
            polygons = [_rectangle(
                col0, row0, col0 + 1, row0 + 1, polygon_id=area_id)]
            annotations = labelling_sw.annotations_digest(
                "T31UCU", area_id, split, window, polygons, [])
            completion = {
                "schema_version": 1,
                "month": "2026-04",
                "features_sha256": feature_digest,
                "features_provenance_sha256": provenance_digest,
                "annotations_sha256": annotations,
            }
            area_record = _record(
                area_id, split, window, polygons, completion=completion)
            (input_dir / f"area-{area_id:03d}.json").write_text(
                json.dumps(area_record), encoding="utf-8")
            area_entries.append({
                "id": area_id, "split": split, "window": window})
    areas = {
        "tile": "T31UCU",
        "grid": {"cell_px": 2, "rows": 8, "cols": 8},
        "areas": area_entries,
    }
    (input_dir / "areas.json").write_text(
        json.dumps(areas), encoding="utf-8")
    contract = modelling_sw.PreparationContract(
        tile="T31UCU",
        month="2026-04",
        source_scenes=tuple(sorted(SCENES)),
        train_area_ids=TRAIN_IDS,
        test_area_ids=TEST_IDS,
    )
    return input_dir, source_root, contract


def test_prepare_roundtrip_validates_inputs_and_refuses_overwrite(
        tmp_path, monkeypatch):
    input_dir, source_root, contract = _write_preparation_fixture(
        tmp_path, monkeypatch)
    run_dir = tmp_path / "run"
    (input_dir / "area-009.json").write_text(
        "this unassigned area must not be read", encoding="utf-8")

    manifest = modelling_sw.prepare_datasets(
        input_dir, run_dir, source_root, contract=contract)

    assert sorted(path.name for path in run_dir.iterdir()) == [
        "prepare-complete.json", "v1-sampling.json", "v1-test.npz",
        "v1-train.npz"]
    with np.load(run_dir / "v1-train.npz", allow_pickle=False) as train:
        assert train["X"].shape == (16, 6)
        assert train["X"].dtype == np.float32
        assert train["y"].dtype == np.uint8
        assert np.array_equal(np.unique(train["area"]), TRAIN_IDS)
        train_keys = set(zip(train["row"].tolist(), train["col"].tolist()))
    with np.load(run_dir / "v1-test.npz", allow_pickle=False) as test:
        assert test["X"].shape == (16, 6)
        assert np.array_equal(np.unique(test["area"]), TEST_IDS)
        test_keys = set(zip(test["row"].tolist(), test["col"].tolist()))
        assert train_keys.isdisjoint(test_keys)
    assert manifest["feature_order"] == list(c.SW_FEATURES)
    assert manifest["grid_reference"]["transform"] == [
        10.0, 0.0, 300000.0, 0.0, -10.0, 5900040.0]
    assert manifest["datasets"]["train"]["rows"] == 16
    assert manifest["datasets"]["train"]["path"] == str(
        (run_dir / "v1-train.npz").resolve())
    assert manifest["model"]["requested"]["n_estimators"] == 100
    assert manifest["probability"]["threshold"] == 0.5
    assert all(item["area_file_sha256"]
               for item in manifest["areas"])

    with pytest.raises(FileExistsError, match="not empty"):
        modelling_sw.prepare_datasets(
            input_dir, run_dir, source_root, contract=contract)


def test_prepare_rejects_stale_completion_without_publishing(
        tmp_path, monkeypatch):
    input_dir, source_root, contract = _write_preparation_fixture(
        tmp_path, monkeypatch)
    area_path = input_dir / "area-001.json"
    record = json.loads(area_path.read_text(encoding="utf-8"))
    record["completion"]["features_sha256"] = "0" * 64
    area_path.write_text(json.dumps(record), encoding="utf-8")
    run_dir = tmp_path / "run"

    with pytest.raises(ValueError, match="area 001.*stale"):
        modelling_sw.prepare_datasets(
            input_dir, run_dir, source_root, contract=contract)

    assert not run_dir.exists()
    assert not any(path.name.startswith(".prepare-")
                   for path in tmp_path.iterdir())


def test_validate_rejects_missing_completion_and_wrong_window(
        tmp_path, monkeypatch):
    input_dir, source_root, contract = _write_preparation_fixture(
        tmp_path, monkeypatch)
    area_path = input_dir / "area-001.json"
    record = json.loads(area_path.read_text(encoding="utf-8"))
    record.pop("completion")
    area_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="area 001.*incomplete"):
        modelling_sw.validate_inputs(input_dir, source_root, contract)

    input_dir, source_root, contract = _write_preparation_fixture(
        tmp_path / "second", monkeypatch)
    areas_path = input_dir / "areas.json"
    areas = json.loads(areas_path.read_text(encoding="utf-8"))
    areas["areas"][0]["window"][0] += 1
    areas_path.write_text(json.dumps(areas), encoding="utf-8")

    with pytest.raises(ValueError, match="area 001 window"):
        modelling_sw.validate_inputs(input_dir, source_root, contract)

    input_dir, source_root, contract = _write_preparation_fixture(
        tmp_path / "third", monkeypatch)
    areas_path = input_dir / "areas.json"
    areas = json.loads(areas_path.read_text(encoding="utf-8"))
    areas["areas"][0]["split"] = "test"
    areas_path.write_text(json.dumps(areas), encoding="utf-8")

    with pytest.raises(ValueError, match="training membership"):
        modelling_sw.validate_inputs(input_dir, source_root, contract)


def test_test_changes_do_not_change_training_samples(tmp_path, monkeypatch):
    input_dir, source_root, contract = _write_preparation_fixture(
        tmp_path, monkeypatch)
    validated = modelling_sw.validate_inputs(
        input_dir, source_root, contract)
    first_train, first_test, _ = modelling_sw.build_datasets(
        validated, contract)
    validated.feature_windows[TEST_IDS[0]]["B02"] += 100
    second_train, second_test, _ = modelling_sw.build_datasets(
        validated, contract)

    for key in modelling_sw.DATASET_KEYS:
        assert np.array_equal(first_train[key], second_train[key])
    assert not np.array_equal(first_test["X"], second_test["X"])


def test_validate_rejects_wrong_feature_order_and_conflicting_grid(
        tmp_path, monkeypatch):
    input_dir, source_root, contract = _write_preparation_fixture(
        tmp_path, monkeypatch)
    provenance_path = input_dir / "features-provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["feature_order"] = list(reversed(c.SW_FEATURES))
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="feature_order"):
        modelling_sw.validate_inputs(input_dir, source_root, contract)

    provenance["feature_order"] = list(c.SW_FEATURES)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    bad_scene = source_root / SCENES[-1]
    band_path = next(bad_scene.rglob("*_B02_10m.tif"))
    band_path.unlink()
    _write_reference_scene(
        source_root, SCENES[-1], (16, 16),
        Affine(10, 0, 300010, 0, -10, 5900040), "EPSG:32631")

    with pytest.raises(ValueError, match="reference grid"):
        modelling_sw.validate_inputs(input_dir, source_root, contract)


def test_sampling_fails_when_nonwater_cannot_match_water():
    window = [0, 2, 0, 2]
    features = _features((2, 2))
    record = _record(1, "train", window, [
        _rectangle(0, 0, 2, 2),
    ])

    with pytest.raises(ValueError, match="non-water"):
        modelling_sw.sample_training_area(record, features)
