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


# %% Step 2: fit, scores, rasters and end-to-end (issue 10).

def _tiny_balanced_train():
    rng = np.random.default_rng(7)
    water = rng.normal(2000, 50, size=(6, 6)).astype(np.float32)
    land = rng.normal(1000, 50, size=(6, 6)).astype(np.float32)
    stacked = np.vstack((water, land))
    # Interleave water and land so both areas below stay balanced.
    order = np.array([0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11])
    return {
        "X": stacked[order],
        "y": np.tile(np.array([1, 0], dtype=np.uint8), 6),
        "area": np.repeat(np.array([1, 3], dtype=np.int32), 6),
        "row": np.arange(12, dtype=np.int32),
        "col": np.arange(12, dtype=np.int32),
        "polygon_index": np.ones(12, dtype=np.int32),
    }


def test_threshold_counts_exact_half_as_water():
    probabilities = np.array([0.49, 0.5, 0.5001, 0.0, 1.0], dtype=np.float64)
    binary = modelling_sw.apply_water_threshold(probabilities)

    assert binary.dtype == np.uint8
    assert binary.tolist() == [0, 1, 1, 0, 1]


def test_scores_report_null_with_reason_not_zero():
    no_predicted = modelling_sw.score_binary_predictions(
        [0, 1, 1], [0, 0, 0])
    assert no_predicted["confusion_matrix"] == [[1, 0], [2, 0]]
    assert no_predicted["precision"] is None
    assert no_predicted["precision"] != 0
    assert no_predicted["undefined_reasons"]["precision"] == \
        "no predicted water"
    assert no_predicted["recall"] == 0.0
    assert no_predicted["f1"] == 0.0

    no_truth = modelling_sw.score_binary_predictions(
        [0, 0, 0], [0, 0, 1])
    assert no_truth["recall"] is None
    assert no_truth["undefined_reasons"]["recall"] == "no true water"
    assert no_truth["precision"] == 0.0

    empty = modelling_sw.score_binary_predictions([0, 0], [0, 0])
    assert empty["precision"] is None
    assert empty["recall"] is None
    assert empty["f1"] is None
    assert empty["undefined_reasons"]["f1"] == "no true or predicted water"

    perfect = modelling_sw.score_binary_predictions(
        [0, 0, 1, 1], [0, 0, 1, 1])
    assert perfect["confusion_matrix"] == [[2, 0], [0, 2]]
    assert perfect["precision"] == 1.0
    assert perfect["recall"] == 1.0
    assert perfect["f1"] == 1.0

    json.dumps(no_predicted, allow_nan=False)
    json.dumps(empty, allow_nan=False)


def test_pooled_metrics_sum_counts_not_area_means():
    test_dataset = {
        "y": np.array([1, 0, 1, 0, 0, 0], dtype=np.uint8),
        "area": np.array([41, 41, 43, 43, 43, 43], dtype=np.int32),
    }
    predicted = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)

    scoring = modelling_sw.evaluate_test_predictions(
        test_dataset, predicted)

    assert scoring["per_area"]["41"]["f1"] == 1.0
    assert scoring["per_area"]["43"]["f1"] == 0.0
    assert scoring["pooled"]["confusion_matrix"] == [[4, 0], [1, 1]]
    assert scoring["pooled"]["f1"] == pytest.approx(2 / 3)
    assert scoring["confusion_convention"].startswith("rows true")


def test_tiny_forest_fits_reloads_and_shares_one_convention(tmp_path):
    train = _tiny_balanced_train()

    model = modelling_sw.fit_water_classifier(train)
    assert list(model.classes_.tolist()) == [0, 1]

    probabilities, binary = modelling_sw.predict_water_labels(
        model, train["X"])
    assert probabilities.dtype == np.float32
    assert binary.dtype == np.uint8
    assert np.array_equal(
        binary, (probabilities >= np.float32(0.5)).astype(np.uint8))
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))

    manifest_stub = {"model": {"requested": {
        "n_estimators": 100, "random_state": 202604,
        "class_weight": None}}}
    bundle_path = tmp_path / "v1-model.pkl"
    modelling_sw._save_model_bundle(
        bundle_path, model, "0" * 64, "1" * 64, manifest_stub)
    reloaded = modelling_sw._load_model_bundle(bundle_path)["model"]
    reloaded_probabilities, reloaded_binary = \
        modelling_sw.predict_water_labels(reloaded, train["X"])
    assert np.array_equal(probabilities, reloaded_probabilities)
    assert np.array_equal(binary, reloaded_binary)


def test_fit_rejects_single_class_and_unbalanced():
    balanced = _tiny_balanced_train()
    single = {**balanced,
              "y": np.zeros(12, dtype=np.uint8)}
    with pytest.raises(ValueError, match="single-class"):
        modelling_sw.fit_water_classifier(single)

    unbalanced = {**balanced,
                  "y": np.array([1, 1, 1, 1, 0, 0] * 2, dtype=np.uint8)}
    with pytest.raises(ValueError, match="unbalanced"):
        modelling_sw.fit_water_classifier(unbalanced)


def test_fit_rejects_nonfinite_features_and_wrong_manifest_order(
        tmp_path, monkeypatch):
    balanced = _tiny_balanced_train()
    nonfinite = {**balanced,
                 "X": balanced["X"].copy()}
    nonfinite["X"][0, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        modelling_sw.fit_water_classifier(nonfinite)

    input_dir, source_root, contract, run_dir, _ = _prepare_then_fit(
        tmp_path, monkeypatch)
    manifest_path = run_dir / "v1-sampling.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["feature_order"] = list(reversed(manifest["feature_order"]))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="feature_order"):
        modelling_sw.fit_run(input_dir, run_dir, source_root, contract)


def _prepare_then_fit(tmp_path, monkeypatch):
    input_dir, source_root, contract = _write_preparation_fixture(
        tmp_path / "fixture", monkeypatch)
    run_dir = tmp_path / "run"
    manifest = modelling_sw.prepare_datasets(
        input_dir, run_dir, source_root, contract=contract)
    return input_dir, source_root, contract, run_dir, manifest


def test_fit_uses_only_training_rows_and_leaves_manifest_stable(
        tmp_path, monkeypatch):
    input_dir, source_root, contract, run_dir, _ = _prepare_then_fit(
        tmp_path, monkeypatch)
    manifest_path = run_dir / "v1-sampling.json"
    before = labelling_sw.file_sha256(manifest_path)
    (run_dir / "v1-test.npz").unlink()

    modelling_sw.fit_run(input_dir, run_dir, source_root, contract)

    assert (run_dir / "v1-model.pkl").is_file()
    assert (run_dir / "fit-complete.json").is_file()
    assert labelling_sw.file_sha256(manifest_path) == before
    with pytest.raises(FileExistsError, match="already exists"):
        modelling_sw.fit_run(input_dir, run_dir, source_root, contract)


def test_fit_rejects_changed_training_data(tmp_path, monkeypatch):
    input_dir, source_root, contract, run_dir, _ = _prepare_then_fit(
        tmp_path, monkeypatch)
    with np.load(run_dir / "v1-train.npz", allow_pickle=False) as archive:
        tampered = {key: np.array(archive[key]) for key in archive.files}
    tampered["y"][0] = 1 - tampered["y"][0]
    np.savez_compressed(run_dir / "v1-train.npz", **tampered)

    with pytest.raises(ValueError, match="no longer matches"):
        modelling_sw.fit_run(input_dir, run_dir, source_root, contract)
    assert not (run_dir / "v1-model.pkl").exists()


def test_raster_roundtrip_keeps_offset_transform_and_nodata(tmp_path):
    transform = [10.0, 1.0, 300000.0, 0.5, -10.0, 5900040.0]
    grid_reference = {
        "shape": [16, 16], "crs": "EPSG:32631", "transform": transform}
    window = [10, 13, 20, 23]
    probability_grid = np.array([
        [0.1, 0.9, np.nan],
        [0.2, 0.5, 0.8],
        [np.nan, 0.0, 1.0],
    ], dtype=np.float32)
    binary_grid = np.array([
        [0, 1, 255],
        [0, 1, 1],
        [255, 0, 1],
    ], dtype=np.uint8)
    tags = {"tile": "T31UCU", "month": "2026-04"}

    probability_path = tmp_path / "prob.tif"
    binary_path = tmp_path / "binary.tif"
    modelling_sw.write_area_geotiffs(
        probability_path, binary_path,
        probability_grid, binary_grid, window, grid_reference, tags)

    for path, nodata, dtype in (
            (probability_path, -9999.0, "float32"),
            (binary_path, 255, "uint8")):
        with rasterio.open(path) as source:
            assert source.count == 1
            assert source.dtypes[0] == dtype
            assert source.nodata == nodata
            assert str(source.crs) == "EPSG:32631"
            assert (source.height, source.width) == (3, 3)
            assert list(source.transform)[:6] != transform
    with rasterio.open(binary_path) as source:
        assert np.array_equal(source.read(1), binary_grid)
    with rasterio.open(probability_path) as source:
        actual = source.read(1)
        assert actual[0, 0] == pytest.approx(0.1)
        assert actual[0, 2] == np.float32(-9999.0)
        assert actual[2, 2] == pytest.approx(1.0)


def test_excluded_valid_pixels_still_receive_predictions():
    window = [0, 3, 0, 3]
    features = _features((3, 3))
    record = _record(41, "test", window, [
        _rectangle(0, 0, 1, 1, polygon_id=1),
    ], exclusions=[
        {"vertices_scene": [[1, 1], [2, 1], [2, 2], [1, 2]]}
    ])
    label_mask, _, _ = label_sw.derive_area_label_state(
        record, features, window, True)
    assert label_mask[1, 1] == label_sw.LABEL_WITHHELD

    class _WaterModel:
        classes_ = np.array([0, 1])

        def predict_proba(self, matrix):
            return np.tile(np.array([[0.2, 0.8]]), (len(matrix), 1))

    probability_grid, binary_grid = modelling_sw.predict_area_grids(
        _WaterModel(), features)

    assert binary_grid[1, 1] == 1
    assert probability_grid[1, 1] == pytest.approx(0.8)

    test_dataset, _ = modelling_sw.sample_test_area(record, features)
    assert len(test_dataset["y"]) == 8


def test_all_invalid_window_never_calls_the_forest():
    features = {name: np.full((2, 2), np.nan, dtype=np.float32)
                for name in c.SW_FEATURES}

    class _ExplodingModel:
        def predict_proba(self, matrix):
            raise AssertionError("forest must not see an empty matrix")

    probability_grid, binary_grid = modelling_sw.predict_area_grids(
        _ExplodingModel(), features)

    assert np.all(~np.isfinite(probability_grid))
    assert np.array_equal(
        binary_grid, np.full((2, 2), 255, dtype=np.uint8))


def test_prepare_fit_evaluate_end_to_end_and_refuses_repeated_evaluate(
        tmp_path, monkeypatch):
    input_dir, source_root, contract, run_dir, _ = _prepare_then_fit(
        tmp_path, monkeypatch)
    manifest_path = run_dir / "v1-sampling.json"
    manifest_before = labelling_sw.file_sha256(manifest_path)

    modelling_sw.fit_run(input_dir, run_dir, source_root, contract)
    metrics = modelling_sw.evaluate_run(
        input_dir, run_dir, source_root, contract)

    assert labelling_sw.file_sha256(manifest_path) == manifest_before
    assert metrics["tile"] == "T31UCU"
    assert metrics["threshold"] == 0.5
    assert sorted(metrics["per_area"]) == ["41", "43", "45", "47"]
    assert metrics["pooled"]["rows"] == 16
    json.dumps(metrics, allow_nan=False)

    expected_files = {"v1-model.pkl", "fit-complete.json",
                      "v1-metrics.json", "evaluate-complete.json"}
    for area_id in (41, 43, 45, 47):
        expected_files |= {
            f"area-{area_id:03d}-water-probability.tif",
            f"area-{area_id:03d}-water-binary.tif",
            f"area-{area_id:03d}-overlay.png",
        }
    assert expected_files <= {path.name for path in run_dir.iterdir()}

    with np.load(run_dir / "v1-test.npz", allow_pickle=False) as test:
        test_rows = {key: np.array(test[key])
                     for key in modelling_sw.DATASET_KEYS}
    bundle = modelling_sw._load_model_bundle(run_dir / "v1-model.pkl")
    expected_probabilities, expected_binary = \
        modelling_sw.predict_water_labels(bundle["model"], test_rows["X"])
    scoring = modelling_sw.evaluate_test_predictions(
        test_rows, expected_binary)
    assert scoring["pooled"] == metrics["pooled"]

    for area_id in (41, 43, 45, 47):
        with rasterio.open(
                run_dir / f"area-{area_id:03d}-water-binary.tif") as source:
            raster = source.read(1)
            assert source.nodata == 255
        mask = test_rows["area"] == area_id
        manifest_areas = json.loads(manifest_path.read_text(
            encoding="utf-8"))["areas"]
        record_window = next(
            entry["window"] for entry in manifest_areas
            if entry["id"] == area_id)
        row0, _, col0, _ = record_window
        for row, col, expected in zip(
                test_rows["row"][mask].tolist(),
                test_rows["col"][mask].tolist(),
                expected_binary[mask].tolist()):
            assert raster[int(row) - row0, int(col) - col0] == int(expected)
        overlay = run_dir / f"area-{area_id:03d}-overlay.png"
        assert overlay.stat().st_size > 0

    metrics_sha_before = labelling_sw.file_sha256(
        run_dir / "v1-metrics.json")
    with pytest.raises(FileExistsError, match="already succeeded"):
        modelling_sw.evaluate_run(
            input_dir, run_dir, source_root, contract)
    assert labelling_sw.file_sha256(
        run_dir / "v1-metrics.json") == metrics_sha_before
