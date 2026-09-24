"""Reproducible pixel datasets and RF baseline for the V1 surface water."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import shutil
import subprocess
import tempfile

import numpy as np
import rasterio

from . import config as c
from . import label_sw
from . import labelling_sw


BASELINE_TILE = "T31UCU"
BASELINE_MONTH = "2026-04"
BASELINE_SOURCE_SCENES = (
    "S2A_MSIL2A_20260427T110651_N0512_R137_T31UCU_20260427T181157.SAFE",
    "S2B_MSIL2A_20260427T105619_N0512_R094_T31UCU_20260427T133501.SAFE",
    "S2B_MSIL2A_20260430T110619_N0512_R137_T31UCU_20260430T132541.SAFE",
    "S2C_MSIL2A_20260425T110621_N0512_R137_T31UCU_20260425T150617.SAFE",
)
BASELINE_TRAIN_AREAS = (2, 25, 200, 250, 275, 300, 325, 375)
BASELINE_TEST_AREAS = (225, 350, 400, 450)
EXPECTED_SCENE_DATES = ("20260425", "20260427", "20260427", "20260430")
EXPECTED_AGGREGATION = "valid median within date, then median across dates"
ROOT_SEED = 202604
WATER_POLYGON_CAP = 100
WATER_AREA_CAP = 1000
WATER_THRESHOLD = 0.5
PREPARATION_SCHEMA_VERSION = 1
MODEL_SCHEMA_VERSION = 1
METRICS_SCHEMA_VERSION = 1
MODEL_FILENAME = "v1-model.pkl"
METRICS_FILENAME = "v1-metrics.json"
FIT_COMPLETE_FILENAME = "fit-complete.json"
EVALUATE_COMPLETE_FILENAME = "evaluate-complete.json"
EVALUATE_FAILURES_FILENAME = "evaluate-failures.jsonl"
PROBABILITY_NODATA = -9999.0
BINARY_NODATA = 255
PREDICTION_BATCH_ROWS = 65536


@dataclass(frozen=True)
class PreparationContract:
    """Fixed identities which define one preparation run."""

    tile: str
    month: str
    source_scenes: tuple[str, ...]
    train_area_ids: tuple[int, ...]
    test_area_ids: tuple[int, ...]


BASELINE_CONTRACT = PreparationContract(
    tile=BASELINE_TILE,
    month=BASELINE_MONTH,
    source_scenes=BASELINE_SOURCE_SCENES,
    train_area_ids=BASELINE_TRAIN_AREAS,
    test_area_ids=BASELINE_TEST_AREAS,
)


@dataclass
class ValidatedInputs:
    """Validated metadata, annotations and copied feature windows."""

    provenance: dict
    assignments: list[dict]
    area_records: dict[int, dict]
    feature_windows: dict[int, dict[str, np.ndarray]]
    input_digests: dict
    grid_reference: dict
    area_metadata: list[dict]


DATASET_KEYS = ("X", "y", "area", "row", "col", "polygon_index")


def _load_json_snapshot(path, description):
    try:
        payload = Path(path).read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"cannot read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object")
    digest = hashlib.sha256(payload).hexdigest()
    return value, digest


def _require_equal(name, actual, expected):
    if actual != expected:
        raise ValueError(f"{name} is {actual!r}, expected {expected!r}")


def _validate_provenance(input_dir, contract):
    path = input_dir / "features-provenance.json"
    provenance, digest = _load_json_snapshot(path, "feature provenance")
    _require_equal("provenance tile", provenance.get("tile"), contract.tile)
    _require_equal(
        "provenance month", provenance.get("month"), contract.month)
    _require_equal(
        "provenance source_scenes", provenance.get("source_scenes"),
        list(contract.source_scenes))
    _require_equal(
        "provenance feature_order", provenance.get("feature_order"),
        list(c.SW_FEATURES))
    _require_equal(
        "provenance aggregation", provenance.get("aggregation"),
        EXPECTED_AGGREGATION)

    scene_dates = sorted(scene.split("_")[2][:8]
                         for scene in contract.source_scenes)
    _require_equal("source scene dates", scene_dates,
                   list(EXPECTED_SCENE_DATES))
    masks = provenance.get("masks")
    if not isinstance(masks, dict):
        raise ValueError("provenance masks must be an object")
    _require_equal(
        "provenance cloud_shadow_classes",
        masks.get("cloud_shadow_classes"),
        list(c.SW_CLOUD_SHADOW_CLASSES))
    _require_equal(
        "provenance nodata_value", masks.get("nodata_value"),
        c.SW_NODATA_VALUE)
    for source_name in ("sea_source", "urban_source"):
        if not isinstance(masks.get(source_name), str) \
                or not masks[source_name].strip():
            raise ValueError(
                f"provenance {source_name} must record its source path")
    if not isinstance(provenance.get("crs"), str) \
            or not provenance["crs"].strip():
        raise ValueError("provenance crs must be a non-empty string")
    return provenance, digest


def _validate_assignments(input_dir, contract):
    path = input_dir / "areas.json"
    assignment_record, digest = _load_json_snapshot(path, "area assignment")
    _require_equal("areas tile", assignment_record.get("tile"), contract.tile)
    expected_grid = {
        "cell_px": c.SW_CELL_PX,
        "rows": c.SW_GRID_ROWS,
        "cols": c.SW_GRID_COLS,
    }
    _require_equal(
        "areas grid", assignment_record.get("grid"), expected_grid)
    entries = assignment_record.get("areas")
    if not isinstance(entries, list):
        raise ValueError("areas must be a list")

    by_id = {}
    split_ids = {"train": [], "test": []}
    for position, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"area assignment {position} must be an object")
        area_id = entry.get("id")
        split = entry.get("split")
        if area_id in by_id:
            raise ValueError(f"duplicate area assignment {area_id!r}")
        if split not in split_ids:
            raise ValueError(
                f"area {area_id!r} has unknown split {split!r}")
        expected_window = list(labelling_sw.grid_cell_window(area_id))
        _require_equal(
            f"area {area_id:03d} window", entry.get("window"),
            expected_window)
        by_id[area_id] = {
            "id": area_id, "split": split, "window": expected_window}
        split_ids[split].append(area_id)

    expected_ids = set(contract.train_area_ids) | set(contract.test_area_ids)
    if set(by_id) != expected_ids:
        missing = sorted(expected_ids - set(by_id))
        extra = sorted(set(by_id) - expected_ids)
        raise ValueError(
            f"areas membership mismatch: missing {missing}, extra {extra}")
    if set(split_ids["train"]) != set(contract.train_area_ids):
        raise ValueError(
            f"training membership is {sorted(split_ids['train'])}, expected "
            f"{sorted(contract.train_area_ids)}")
    if set(split_ids["test"]) != set(contract.test_area_ids):
        raise ValueError(
            f"test membership is {sorted(split_ids['test'])}, expected "
            f"{sorted(contract.test_area_ids)}")
    labelling_sw.validate_areas(
        list(contract.train_area_ids), list(contract.test_area_ids))
    assignments = [by_id[area_id] for area_id in sorted(by_id)]
    return assignments, digest


def _reference_band_path(source_image_root, scene_name):
    scene_dir = source_image_root / scene_name
    try:
        image_dir = Path(labelling_sw.granule_img_data(
            str(scene_dir), "R10m"))
    except OSError as exc:
        raise ValueError(
            f"cannot locate 10 m imagery for {scene_name}: {exc}") from exc
    matches = sorted(
        path for path in image_dir.iterdir()
        if path.name.endswith(("_B02_10m.jp2", "_B02_10m.tif")))
    if len(matches) != 1:
        raise ValueError(
            f"expected one 10 m B02 reference for {scene_name}, found "
            f"{len(matches)}")
    return matches[0]


def _grid_metadata(path):
    try:
        with rasterio.open(path) as source:
            return {
                "shape": [source.height, source.width],
                "crs": str(source.crs),
                "transform": [float(value) for value in source.transform[:6]],
            }
    except (OSError, rasterio.errors.RasterioError) as exc:
        raise ValueError(f"cannot read reference raster {path}: {exc}") from exc


def _validate_grid_reference(source_image_root, contract, provenance,
                             expected_shape):
    references = []
    common_grid = None
    for scene_name in contract.source_scenes:
        path = _reference_band_path(source_image_root, scene_name)
        metadata = _grid_metadata(path)
        if metadata["shape"] != list(expected_shape):
            raise ValueError(
                f"reference grid for {scene_name} has shape "
                f"{metadata['shape']}, expected {list(expected_shape)}")
        if metadata["crs"] != provenance["crs"]:
            raise ValueError(
                f"reference grid for {scene_name} has CRS "
                f"{metadata['crs']!r}, expected {provenance['crs']!r}")
        if common_grid is None:
            common_grid = metadata
        elif metadata != common_grid:
            raise ValueError(
                f"reference grid for {scene_name} conflicts with the other "
                "recorded scenes")
        references.append({
            "scene": scene_name,
            "band": "B02",
            "path": str(path.resolve()),
        })
    return {**common_grid, "references": references,
            "provenance_limitation": (
                "The feature archive does not store its affine transform; "
                "the grid was recovered from the recorded source rasters.")}


def _load_feature_windows(input_dir, assignments):
    features_path = input_dir / "features.npz"
    expected_shape = (c.SW_TILE_PX, c.SW_TILE_PX)
    windows = {entry["id"]: {} for entry in assignments}
    try:
        with np.load(features_path, allow_pickle=False) as archive:
            for name in c.SW_FEATURES:
                if name not in archive:
                    raise ValueError(f"features archive is missing {name}")
                full_tile = archive[name]
                if not np.issubdtype(full_tile.dtype, np.number):
                    raise ValueError(
                        f"feature {name} has non-numeric dtype "
                        f"{full_tile.dtype}")
                if full_tile.shape != expected_shape:
                    raise ValueError(
                        f"feature {name} is {full_tile.shape}, expected "
                        f"{expected_shape}")
                for entry in assignments:
                    row0, row1, col0, col1 = entry["window"]
                    windows[entry["id"]][name] = np.array(
                        full_tile[row0:row1, col0:col1],
                        dtype=np.float32, copy=True)
                del full_tile
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(
                ("features archive", "feature ")):
            raise
        raise ValueError(f"cannot read features archive: {exc}") from exc
    return windows, expected_shape


def _load_active_area_records(input_dir, assignments, contract,
                              features_digest, provenance_digest):
    records = {}
    metadata = []
    digests = {}
    for entry in assignments:
        area_id = entry["id"]
        path = input_dir / f"area-{area_id:03d}.json"
        try:
            snapshot = labelling_sw.load_area_record_snapshot(path)
        except ValueError as exc:
            raise ValueError(f"area {area_id:03d}: {exc}") from exc
        record = snapshot["record"]
        raw_digest = snapshot["sha256"]
        state, reason, active = label_sw.check_completion_active(
            record, contract.month, contract.tile, area_id, entry["split"],
            entry["window"], features_digest, provenance_digest)
        if not active:
            raise ValueError(
                f"area {area_id:03d} completion is {state}: {reason}")
        annotation_digest = labelling_sw.annotations_digest(
            record["tile"], area_id, record["split"], record["window"],
            record["polygons"], record.get("exclusions", []))
        records[area_id] = record
        digests[str(path.resolve())] = raw_digest
        metadata.append({
            "id": area_id,
            "split": entry["split"],
            "window": list(entry["window"]),
            "area_file": str(path.resolve()),
            "area_file_sha256": raw_digest,
            "annotations_sha256": annotation_digest,
            "polygon_index_map": [
                {
                    "polygon_index": position,
                    "class": polygon["class"],
                    "stored_id": polygon.get("id"),
                }
                for position, polygon in enumerate(
                    record["polygons"], start=1)
            ],
        })
    return records, metadata, digests


def validate_inputs(input_dir, source_image_root,
                    contract=BASELINE_CONTRACT):
    """Validate and load one immutable preparation snapshot."""
    input_dir = Path(input_dir).resolve()
    source_image_root = Path(source_image_root).resolve()
    provenance, provenance_digest = _validate_provenance(
        input_dir, contract)
    assignments, areas_digest = _validate_assignments(input_dir, contract)

    features_path = input_dir / "features.npz"
    provenance_path = input_dir / "features-provenance.json"
    areas_path = input_dir / "areas.json"
    for path in (features_path, provenance_path, areas_path):
        if not path.is_file():
            raise ValueError(f"missing required input {path}")
    features_digest = labelling_sw.file_sha256(features_path)
    feature_windows, expected_shape = _load_feature_windows(
        input_dir, assignments)
    grid_reference = _validate_grid_reference(
        source_image_root, contract, provenance, expected_shape)
    area_records, area_metadata, area_digests = _load_active_area_records(
        input_dir, assignments, contract, features_digest,
        provenance_digest)
    input_digests = {
        str(features_path): features_digest,
        str(provenance_path): provenance_digest,
        str(areas_path): areas_digest,
        **area_digests,
    }
    return ValidatedInputs(
        provenance=provenance,
        assignments=assignments,
        area_records=area_records,
        feature_windows=feature_windows,
        input_digests=input_digests,
        grid_reference=grid_reference,
        area_metadata=area_metadata,
    )


def _polygon_ownership(record, label_mask, label_code, class_name):
    owner = np.zeros(label_mask.shape, dtype=np.int32)
    assigned = np.zeros(label_mask.shape, dtype=bool)
    polygon_rows = []
    overlap_total = 0
    for position, polygon in enumerate(record["polygons"], start=1):
        if polygon["class"] != class_name:
            continue
        footprint = labelling_sw.rasterize_scene_polygons(
            [{"vertices_scene": polygon["vertices_scene"]}],
            record["window"])
        eligible = footprint & (label_mask == label_code)
        overlap = eligible & assigned
        owned = eligible & ~assigned
        owner[owned] = position
        assigned |= owned
        overlap_count = int(overlap.sum())
        overlap_total += overlap_count
        polygon_rows.append({
            "polygon_index": position,
            "stored_id": polygon.get("id"),
            "eligible_owned": int(owned.sum()),
            "same_class_overlap_removed": overlap_count,
        })
    return owner, polygon_rows, overlap_total


def _dataset_for_pixels(record, features_window, flat_pixels, labels,
                        polygon_indices):
    row0, _, col0, _ = record["window"]
    local_rows, local_cols = np.unravel_index(
        np.asarray(flat_pixels, dtype=np.int64),
        next(iter(features_window.values())).shape)
    x_columns = []
    for name in c.SW_FEATURES:
        x_columns.append(
            features_window[name].reshape(-1)[flat_pixels])
    x_values = np.column_stack(x_columns).astype(np.float32, copy=False)
    return {
        "X": x_values,
        "y": np.asarray(labels, dtype=np.uint8),
        "area": np.full(len(flat_pixels), record["area_id"],
                        dtype=np.int32),
        "row": (local_rows + row0).astype(np.int32),
        "col": (local_cols + col0).astype(np.int32),
        "polygon_index": np.asarray(polygon_indices, dtype=np.int32),
    }


def _sort_dataset(dataset):
    order = np.lexsort((dataset["col"], dataset["row"], dataset["area"]))
    return {key: value[order] for key, value in dataset.items()}


def _combine_datasets(datasets):
    if not datasets:
        raise ValueError("cannot combine an empty dataset list")
    combined = {
        key: np.concatenate([dataset[key] for dataset in datasets], axis=0)
        for key in DATASET_KEYS
    }
    return _sort_dataset(combined)


def _derive_active_mask(record, features_window):
    label_mask, counts, _ = label_sw.derive_area_label_state(
        record, features_window, record["window"], completion_active=True)
    return label_mask, counts


def _select_capped_water(record, water_owner, polygon_rows, generator):
    """Apply the polygon cap, then the area cap, to owned water pixels."""
    capped_pixels = []
    capped_polygon_indices = []
    for polygon in polygon_rows:
        polygon_index = polygon["polygon_index"]
        candidates = np.flatnonzero(water_owner.reshape(-1) == polygon_index)
        if len(candidates) > WATER_POLYGON_CAP:
            selected = generator.choice(
                candidates, size=WATER_POLYGON_CAP, replace=False)
        else:
            selected = candidates
        polygon["after_polygon_cap"] = int(len(selected))
        capped_pixels.extend(np.asarray(selected, dtype=np.int64).tolist())
        capped_polygon_indices.extend([polygon_index] * len(selected))

    if not capped_pixels:
        raise ValueError(
            f"area {record['area_id']:03d} has no eligible water samples")
    capped_pixels = np.asarray(capped_pixels, dtype=np.int64)
    capped_polygon_indices = np.asarray(
        capped_polygon_indices, dtype=np.int32)
    if len(capped_pixels) > WATER_AREA_CAP:
        keep = generator.choice(
            len(capped_pixels), size=WATER_AREA_CAP, replace=False)
        water_pixels = capped_pixels[keep]
        water_polygon_indices = capped_polygon_indices[keep]
    else:
        water_pixels = capped_pixels
        water_polygon_indices = capped_polygon_indices

    for polygon in polygon_rows:
        polygon["after_area_cap"] = int(np.count_nonzero(
            water_polygon_indices == polygon["polygon_index"]))
    return (water_pixels, water_polygon_indices,
            int(len(capped_pixels)))


def _sample_matching_nonwater(record, label_mask, nonwater_owner,
                              water_count, generator):
    """Draw the fixed 1:1 non-water sample from the same area."""
    nonwater_candidates = np.flatnonzero(
        label_mask.reshape(-1) == label_sw.LABEL_NONWATER)
    if len(nonwater_candidates) < water_count:
        raise ValueError(
            f"area {record['area_id']:03d} has {water_count} selected "
            f"water pixels but only {len(nonwater_candidates)} eligible "
            "non-water pixels")
    nonwater_pixels = generator.choice(
        nonwater_candidates, size=water_count, replace=False)
    nonwater_polygon_indices = nonwater_owner.reshape(-1)[nonwater_pixels]
    return nonwater_pixels, nonwater_polygon_indices, nonwater_candidates


def _training_sampling_report(record, label_mask, label_counts,
                              polygon_rows, overlap_count,
                              after_polygon_cap, water_count,
                              nonwater_eligible):
    return {
        "area_id": record["area_id"],
        "label_counts": label_counts,
        "water": {
            "eligible": int((label_mask == label_sw.LABEL_WATER).sum()),
            "after_polygon_cap": after_polygon_cap,
            "selected": water_count,
            "same_class_overlap_pixels": overlap_count,
            "zero_eligible_polygons": sum(
                item["eligible_owned"] == 0 for item in polygon_rows),
            "polygons": polygon_rows,
        },
        "nonwater": {
            "eligible": nonwater_eligible,
            "selected": water_count,
        },
    }


def sample_training_area(record, features_window):
    """Sample one training area with the fixed two-level water caps."""
    label_mask, label_counts = _derive_active_mask(record, features_window)
    water_owner, polygon_rows, overlap_count = _polygon_ownership(
        record, label_mask, label_sw.LABEL_WATER, "water")
    nonwater_owner, _, _ = _polygon_ownership(
        record, label_mask, label_sw.LABEL_NONWATER, "non-water")
    generator = np.random.default_rng([ROOT_SEED, record["area_id"]])
    water_pixels, water_polygon_indices, after_polygon_cap = \
        _select_capped_water(
            record, water_owner, polygon_rows, generator)
    nonwater_pixels, nonwater_polygon_indices, nonwater_candidates = \
        _sample_matching_nonwater(
            record, label_mask, nonwater_owner, len(water_pixels), generator)

    flat_pixels = np.concatenate((water_pixels, nonwater_pixels))
    labels = np.concatenate((
        np.ones(len(water_pixels), dtype=np.uint8),
        np.zeros(len(nonwater_pixels), dtype=np.uint8),
    ))
    polygon_indices = np.concatenate((
        water_polygon_indices, nonwater_polygon_indices))
    dataset = _sort_dataset(_dataset_for_pixels(
        record, features_window, flat_pixels, labels, polygon_indices))
    if len(np.unique(np.column_stack((dataset["row"], dataset["col"])),
                     axis=0)) != len(dataset["y"]):
        raise AssertionError("training samples contain duplicate pixels")
    report = _training_sampling_report(
        record, label_mask, label_counts, polygon_rows, overlap_count,
        after_polygon_cap, len(water_pixels), len(nonwater_candidates))
    return dataset, report


def sample_test_area(record, features_window):
    """Select every eligible labelled test pixel without RNG draws."""
    label_mask, label_counts = _derive_active_mask(record, features_window)
    water_owner, _, water_overlap = _polygon_ownership(
        record, label_mask, label_sw.LABEL_WATER, "water")
    nonwater_owner, _, nonwater_overlap = _polygon_ownership(
        record, label_mask, label_sw.LABEL_NONWATER, "non-water")
    eligible = (label_mask == label_sw.LABEL_WATER) \
        | (label_mask == label_sw.LABEL_NONWATER)
    flat_pixels = np.flatnonzero(eligible.reshape(-1))
    if len(flat_pixels) == 0:
        raise ValueError(
            f"test area {record['area_id']:03d} has no eligible pixels")
    flat_labels = label_mask.reshape(-1)[flat_pixels]
    labels = (flat_labels == label_sw.LABEL_WATER).astype(np.uint8)
    water_flat = water_owner.reshape(-1)[flat_pixels]
    nonwater_flat = nonwater_owner.reshape(-1)[flat_pixels]
    polygon_indices = np.where(labels == 1, water_flat, nonwater_flat)
    dataset = _sort_dataset(_dataset_for_pixels(
        record, features_window, flat_pixels, labels, polygon_indices))
    report = {
        "area_id": record["area_id"],
        "eligible": int(len(flat_pixels)),
        "water": int((labels == 1).sum()),
        "nonwater": int((labels == 0).sum()),
        "label_counts": label_counts,
        "same_class_overlap_pixels": {
            "water": water_overlap, "nonwater": nonwater_overlap},
    }
    return dataset, report


def build_datasets(validated, contract=BASELINE_CONTRACT):
    """Build stable train and test arrays from validated inputs."""
    train_parts = []
    test_parts = []
    sampling_reports = {"train": [], "test": []}
    for area_id in sorted(contract.train_area_ids):
        dataset, report = sample_training_area(
            validated.area_records[area_id],
            validated.feature_windows[area_id])
        train_parts.append(dataset)
        sampling_reports["train"].append(report)
    for area_id in sorted(contract.test_area_ids):
        dataset, report = sample_test_area(
            validated.area_records[area_id],
            validated.feature_windows[area_id])
        test_parts.append(dataset)
        sampling_reports["test"].append(report)
    train = _combine_datasets(train_parts)
    test = _combine_datasets(test_parts)

    train_keys = set(zip(train["row"].tolist(), train["col"].tolist()))
    test_keys = set(zip(test["row"].tolist(), test["col"].tolist()))
    overlap = train_keys & test_keys
    if overlap:
        raise ValueError(
            f"training and test datasets share {len(overlap)} pixels")
    for area_id in contract.train_area_ids:
        area_labels = train["y"][train["area"] == area_id]
        counts = np.bincount(area_labels, minlength=2)
        if counts[0] != counts[1] or counts[0] == 0:
            raise AssertionError(
                f"training area {area_id:03d} is not non-empty and balanced")
    return train, test, sampling_reports


def _save_dataset(path, dataset):
    np.savez_compressed(path, **dataset)
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(DATASET_KEYS):
            raise AssertionError(f"saved dataset keys changed in {path}")
        row_count = len(saved["y"])
        if any(len(saved[key]) != row_count for key in DATASET_KEYS):
            raise AssertionError(f"saved dataset rows are misaligned in {path}")


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _resolved_model_parameters():
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required to freeze the model settings; "
            "install the package with 'pip install -e .'") from exc
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=ROOT_SEED,
        class_weight=None,
    )
    return model.get_params(deep=False)


def _implementation_identity():
    package_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in (
            "config.py", "label_sw.py", "labelling_sw.py",
            "modelling_sw.py", "train_sw.py"):
        path = package_dir / name
        if path.is_file():
            digest.update(name.encode("utf-8"))
            digest.update(path.read_bytes())
    repo_root = package_dir.parent.parent
    revision = None
    dirty = None
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root,
            capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root,
            capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "git_revision": revision,
        "git_dirty": dirty,
        "preparation_source_sha256": digest.hexdigest(),
    }


def _dataset_summary(path, dataset, published_path=None):
    if published_path is None:
        published_path = path
    return {
        "path": str(Path(published_path).resolve()),
        "sha256": labelling_sw.file_sha256(path),
        "rows": int(len(dataset["y"])),
        "shape": list(dataset["X"].shape),
        "dtypes": {key: str(value.dtype)
                   for key, value in dataset.items()},
    }


def _build_manifest(validated, train, test, reports, contract,
                    train_path, test_path, published_run_dir):
    return {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "phase": "prepared",
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "tile": contract.tile,
        "month": contract.month,
        "source_scenes": list(contract.source_scenes),
        "feature_order": list(c.SW_FEATURES),
        "aggregation": validated.provenance["aggregation"],
        "masks": validated.provenance["masks"],
        "validity": "all six saved feature values are finite",
        "label_mapping": {
            "source_codes": {
                "unusable": label_sw.LABEL_UNUSABLE,
                "withheld": label_sw.LABEL_WITHHELD,
                "water": label_sw.LABEL_WATER,
                "non-water": label_sw.LABEL_NONWATER,
            },
            "model_targets": {"non-water": 0, "water": 1},
        },
        "split": {
            "train": list(contract.train_area_ids),
            "test": list(contract.test_area_ids),
        },
        "grid_reference": validated.grid_reference,
        "areas": validated.area_metadata,
        "inputs": {
            "paths_and_sha256": validated.input_digests,
        },
        "sampling": {
            "root_seed": ROOT_SEED,
            "area_generator": (
                f"numpy.random.default_rng([{ROOT_SEED}, area_id])"),
            "without_replacement": True,
            "water_polygon_cap": WATER_POLYGON_CAP,
            "water_area_cap": WATER_AREA_CAP,
            "nonwater_rule": "match selected water count within each area",
            "test_rule": "all eligible labelled pixels; no RNG draws",
            "reports": reports,
        },
        "probability": {
            "meaning": "uncalibrated RF water probability from balanced training",
            "threshold": WATER_THRESHOLD,
            "tie_rule": "water when probability >= 0.5",
        },
        "model": {
            "requested": {
                "type": "sklearn.ensemble.RandomForestClassifier",
                "n_estimators": 100,
                "random_state": ROOT_SEED,
                "class_weight": None,
            },
            "resolved_parameters": _resolved_model_parameters(),
        },
        "datasets": {
            "train": _dataset_summary(
                train_path, train, published_run_dir / "v1-train.npz"),
            "test": _dataset_summary(
                test_path, test, published_run_dir / "v1-test.npz"),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": _package_version("scikit-learn"),
            "rasterio": rasterio.__version__,
        },
        "implementation": _implementation_identity(),
        "limitations": [
            "Spatial caps reduce dominance but do not remove correlation.",
            "The frozen split prevents pixel overlap but does not establish "
            "statistical independence.",
            "The balanced-training probability is not calibrated to natural "
            "water prevalence.",
            "Completion can include missed water in the derived non-water "
            "background.",
            "A water body spanning train and test still requires visual "
            "inspection.",
        ],
    }


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _recheck_input_digests(input_digests, phase="preparation"):
    for path_text, expected_digest in input_digests.items():
        path = Path(path_text)
        try:
            actual_digest = labelling_sw.file_sha256(path)
        except OSError as exc:
            raise ValueError(
                f"input changed or disappeared during {phase}: "
                f"{path}: {exc}") from exc
        if actual_digest != expected_digest:
            raise ValueError(
                f"input changed during {phase}: {path}")


def _recheck_grid_reference(grid_reference, phase="preparation"):
    expected = {
        key: grid_reference[key] for key in ("shape", "crs", "transform")}
    for reference in grid_reference["references"]:
        actual = _grid_metadata(Path(reference["path"]))
        if actual != expected:
            raise ValueError(
                f"source reference grid changed during {phase}: "
                f"{reference['path']}")


def prepare_datasets(input_dir, run_dir, source_image_root,
                     contract=BASELINE_CONTRACT):
    """Validate, sample and atomically publish the preparation artifacts."""
    input_dir = Path(input_dir).resolve()
    run_dir = Path(run_dir).resolve()
    if run_dir == input_dir:
        raise ValueError("run directory must be separate from input directory")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_dir}")

    validated = validate_inputs(input_dir, source_image_root, contract)
    train, test, reports = build_datasets(validated, contract)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(
        prefix=".prepare-", dir=run_dir.parent))
    try:
        train_path = temporary_dir / "v1-train.npz"
        test_path = temporary_dir / "v1-test.npz"
        _save_dataset(train_path, train)
        _save_dataset(test_path, test)
        manifest = _build_manifest(
            validated, train, test, reports, contract,
            train_path, test_path, run_dir)
        _write_json(temporary_dir / "v1-sampling.json", manifest)
        _write_json(temporary_dir / "prepare-complete.json", {
            "schema_version": PREPARATION_SCHEMA_VERSION,
            "phase": "prepared",
            "manifest_sha256": labelling_sw.file_sha256(
                temporary_dir / "v1-sampling.json"),
        })
        _recheck_input_digests(validated.input_digests)
        _recheck_grid_reference(validated.grid_reference)
        if run_dir.exists():
            os.rmdir(run_dir)
        os.replace(temporary_dir, run_dir)
        temporary_dir = None
        return manifest
    finally:
        if temporary_dir is not None and temporary_dir.exists():
            shutil.rmtree(temporary_dir)


# %% Fit the fixed 100-tree forest without touching held-out test data.

def _load_run_manifest(run_dir):
    """Read one run's frozen sampling manifest and its digest."""
    path = Path(run_dir) / "v1-sampling.json"
    try:
        payload = path.read_bytes()
        manifest = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"cannot read run manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"run manifest {path} must contain a JSON object")
    return manifest, hashlib.sha256(payload).hexdigest()


def _check_manifest_contract(manifest, contract):
    """Require the frozen manifest to describe this baseline exactly."""
    _require_equal("manifest tile", manifest.get("tile"), contract.tile)
    _require_equal("manifest month", manifest.get("month"), contract.month)
    _require_equal(
        "manifest feature_order", manifest.get("feature_order"),
        list(c.SW_FEATURES))
    _require_equal(
        "manifest source_scenes", manifest.get("source_scenes"),
        list(contract.source_scenes))
    split = manifest.get("split") or {}
    _require_equal(
        "manifest train split", sorted(split.get("train", [])),
        sorted(contract.train_area_ids))
    _require_equal(
        "manifest test split", sorted(split.get("test", [])),
        sorted(contract.test_area_ids))
    probability = manifest.get("probability") or {}
    _require_equal(
        "manifest threshold", probability.get("threshold"), WATER_THRESHOLD)
    requested = (manifest.get("model") or {}).get("requested") or {}
    _require_equal(
        "manifest n_estimators", requested.get("n_estimators"), 100)
    _require_equal(
        "manifest random_state", requested.get("random_state"), ROOT_SEED)
    if requested.get("class_weight") is not None:
        raise ValueError(
            f"manifest class_weight is {requested.get('class_weight')!r}, "
            "expected None")


def _load_npz_dataset(path):
    """Load one saved dataset with key, alignment and dtype checks."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(DATASET_KEYS):
                raise ValueError(
                    f"dataset {path} holds {sorted(archive.files)}, "
                    f"expected {sorted(DATASET_KEYS)}")
            dataset = {key: np.array(archive[key]) for key in DATASET_KEYS}
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"cannot read dataset {path}: {exc}") from exc
    row_count = len(dataset["y"])
    for key in DATASET_KEYS:
        if len(dataset[key]) != row_count:
            raise ValueError(f"dataset {path} has misaligned rows")
        if dataset[key].dtype.kind == "O":
            raise ValueError(f"dataset {path} key {key} is an object array")
    return dataset


def _validate_dataset_content(dataset, dataset_name):
    """Require finite features, binary targets and integer provenance."""
    if dataset["X"].ndim != 2 or dataset["X"].shape[1] != len(c.SW_FEATURES):
        raise ValueError(
            f"{dataset_name} X is {dataset['X'].shape}, expected (N, 6)")
    if dataset["X"].dtype != np.float32:
        raise ValueError(
            f"{dataset_name} X dtype is {dataset['X'].dtype}, "
            "expected float32")
    if dataset["y"].dtype != np.uint8:
        raise ValueError(
            f"{dataset_name} y dtype is {dataset['y'].dtype}, "
            "expected uint8")
    if not np.all(np.isfinite(dataset["X"])):
        raise ValueError(f"{dataset_name} X must be finite")
    unique_targets = set(np.unique(dataset["y"]).tolist())
    if not unique_targets <= {0, 1}:
        raise ValueError(
            f"{dataset_name} targets are {sorted(unique_targets)}, "
            "expected a subset of [0, 1]")
    for key in ("area", "row", "col", "polygon_index"):
        if dataset[key].dtype.kind not in "iu":
            raise ValueError(
                f"{dataset_name} {key} dtype is {dataset[key].dtype}, "
                "expected an integer dtype")


def _require_exact_areas(dataset, expected_area_ids, dataset_name):
    """Require the frozen split membership before fitting or scoring."""
    actual_areas = sorted(np.unique(dataset["area"]).tolist())
    if actual_areas != sorted(expected_area_ids):
        raise ValueError(
            f"{dataset_name} areas are {actual_areas}, expected "
            f"{sorted(expected_area_ids)}")


def _validate_training_balance(dataset):
    """Require two non-empty balanced classes in every training area."""
    for area_id in sorted(np.unique(dataset["area"]).tolist()):
        area_labels = dataset["y"][dataset["area"] == area_id]
        counts = np.bincount(area_labels, minlength=2)
        if counts[0] == 0 or counts[1] == 0:
            raise ValueError(
                f"training area {int(area_id):03d} is single-class: "
                f"counts [non-water={int(counts[0])}, "
                f"water={int(counts[1])}]")
        if counts[0] != counts[1]:
            raise ValueError(
                f"training area {int(area_id):03d} is unbalanced: "
                f"counts [non-water={int(counts[0])}, "
                f"water={int(counts[1])}]")
    pixel_keys = list(zip(
        dataset["row"].tolist(), dataset["col"].tolist()))
    if len(set(pixel_keys)) != len(pixel_keys):
        raise ValueError("training dataset contains duplicate pixels")


def fit_water_classifier(train_dataset, contract=None):
    """Fit the fixed forest from training rows only.

    Takes the training dataset alone. The test dataset must never be
    passed here; the fit action does not load it. An optional contract
    adds an exact-membership check; without one, every present area
    must still be non-empty and balanced.
    """
    _validate_dataset_content(train_dataset, "training dataset")
    if contract is not None:
        _require_exact_areas(
            train_dataset, contract.train_area_ids, "training dataset")
    _validate_training_balance(train_dataset)
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required to fit the baseline; "
            "install the package with 'pip install -e .'") from exc
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=ROOT_SEED,
        class_weight=None,
    )
    model.fit(train_dataset["X"], train_dataset["y"])
    if list(model.classes_.tolist()) != [0, 1]:
        raise ValueError(
            f"fitted classes are {model.classes_.tolist()}, expected [0, 1]")
    return model


def _save_model_bundle(path, model, manifest_sha256, train_sha256,
                       manifest):
    """Pickle the fitted forest with its frozen run identities."""
    bundle = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model": model,
        "feature_order": list(c.SW_FEATURES),
        "threshold": WATER_THRESHOLD,
        "tie_rule": "water when probability >= 0.5",
        "manifest_sha256": manifest_sha256,
        "train_dataset_sha256": train_sha256,
        "resolved_parameters": model.get_params(deep=False),
        "fitted_classes": [int(value) for value in model.classes_.tolist()],
        "model_requested": (manifest.get("model") or {}).get("requested"),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": _package_version("scikit-learn"),
            "rasterio": rasterio.__version__,
        },
        "implementation": _implementation_identity(),
    }
    with open(path, "wb") as handle:
        pickle.dump(bundle, handle, protocol=4)
    with open(path, "rb") as handle:
        reloaded = pickle.load(handle)
    if list(reloaded["model"].classes_.tolist()) != [0, 1]:
        raise AssertionError(f"saved model classes changed in {path}")
    return bundle


def _load_model_bundle(path):
    """Load one run's fitted forest bundle with basic checks."""
    try:
        with open(path, "rb") as handle:
            bundle = pickle.load(handle)
    except (OSError, ValueError, pickle.UnpicklingError) as exc:
        raise ValueError(f"cannot read model {path}: {exc}") from exc
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"model {path} is not a baseline bundle")
    model = bundle["model"]
    if list(model.classes_.tolist()) != [0, 1]:
        raise ValueError(
            f"model {path} classes are {model.classes_.tolist()}, "
            "expected [0, 1]")
    if list(bundle.get("feature_order", [])) != list(c.SW_FEATURES):
        raise ValueError(f"model {path} has the wrong feature order")
    if bundle.get("threshold") != WATER_THRESHOLD:
        raise ValueError(f"model {path} has the wrong threshold")
    return bundle


def fit_run(input_dir, run_dir, source_image_root,
            contract=BASELINE_CONTRACT):
    """Fit the baseline forest from the frozen training dataset.

    Never opens the test dataset; a missing test file does not stop the
    fit. Refuses to overwrite a successful model.
    """
    input_dir = Path(input_dir).resolve()
    run_dir = Path(run_dir).resolve()
    source_image_root = Path(source_image_root).resolve()
    prepare_marker = run_dir / "prepare-complete.json"
    manifest_path = run_dir / "v1-sampling.json"
    train_path = run_dir / "v1-train.npz"
    model_path = run_dir / MODEL_FILENAME
    fit_marker_path = run_dir / FIT_COMPLETE_FILENAME
    for path in (prepare_marker, manifest_path, train_path):
        if not path.is_file():
            raise ValueError(
                f"run is not prepared: missing {path}; "
                "run prepare first")
    if model_path.exists() or fit_marker_path.exists():
        raise FileExistsError(
            f"fit output already exists in {run_dir}; "
            "do not overwrite a successful fit")

    manifest, manifest_sha256 = _load_run_manifest(run_dir)
    _check_manifest_contract(manifest, contract)
    _recheck_input_digests(
        manifest["inputs"]["paths_and_sha256"], phase="fitting")
    _recheck_grid_reference(manifest["grid_reference"], phase="fitting")

    expected_train_sha256 = manifest["datasets"]["train"]["sha256"]
    actual_train_sha256 = labelling_sw.file_sha256(train_path)
    if actual_train_sha256 != expected_train_sha256:
        raise ValueError(
            f"training dataset changed: {train_path} no longer matches "
            "the frozen manifest")
    train_dataset = _load_npz_dataset(train_path)

    model = fit_water_classifier(train_dataset, contract)

    temporary_dir = Path(tempfile.mkdtemp(
        prefix=".fit-", dir=run_dir.parent))
    try:
        staged_model = temporary_dir / MODEL_FILENAME
        bundle = _save_model_bundle(
            staged_model, model, manifest_sha256, actual_train_sha256,
            manifest)
        staged_marker = temporary_dir / FIT_COMPLETE_FILENAME
        _write_json(staged_marker, {
            "schema_version": MODEL_SCHEMA_VERSION,
            "phase": "fitted",
            "manifest_sha256": manifest_sha256,
            "train_dataset_sha256": actual_train_sha256,
            "model_sha256": labelling_sw.file_sha256(staged_model),
            "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
            "resolved_parameters": bundle["resolved_parameters"],
        })
        if model_path.exists() or fit_marker_path.exists():
            raise FileExistsError(
                f"fit output appeared during fitting in {run_dir}; "
                "refusing to overwrite it")
        os.replace(staged_model, model_path)
        os.replace(staged_marker, fit_marker_path)
        temporary_dir = None
        return bundle
    finally:
        if temporary_dir is not None and temporary_dir.exists():
            shutil.rmtree(temporary_dir)


# %% One prediction convention for evaluation and raster export.

def water_probabilities(model, features):
    """Return uncalibrated water probability as float32.

    Selects the water column through classes_, never by position.
    """
    classes = [int(value) for value in list(model.classes_.tolist())]
    if classes != [0, 1]:
        raise ValueError(
            f"model classes are {classes}, expected [0, 1]")
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != len(c.SW_FEATURES):
        raise ValueError(
            f"feature matrix is {matrix.shape}, expected (N, 6)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("feature matrix must be finite for prediction")
    water_column = classes.index(1)
    probabilities = model.predict_proba(matrix)[:, water_column]
    return np.asarray(probabilities, dtype=np.float32)


def apply_water_threshold(probabilities):
    """Apply the fixed >= 0.5 rule with exact ties counted as water."""
    values = np.asarray(probabilities, dtype=np.float32)
    return (values >= np.float32(WATER_THRESHOLD)).astype(np.uint8)


def predict_water_labels(model, features):
    """Predict probabilities and binary labels with one convention."""
    probabilities = water_probabilities(model, features)
    return probabilities, apply_water_threshold(probabilities)


# %% Held-out scores with explicit undefined-value handling.

def _score_from_counts(true_negative, false_positive,
                       false_negative, true_positive):
    """Score one confusion table without inventing undefined values."""
    support_water = true_positive + false_negative
    support_nonwater = true_negative + false_positive
    predicted_water = true_positive + false_positive
    predicted_nonwater = true_negative + false_negative
    if predicted_water > 0:
        precision: float | None = true_positive / predicted_water
        precision_reason = None
    else:
        precision = None
        precision_reason = "no predicted water"
    if support_water > 0:
        recall: float | None = true_positive / support_water
        recall_reason = None
    else:
        recall = None
        recall_reason = "no true water"
    f1_denominator = 2 * true_positive + false_positive + false_negative
    if f1_denominator > 0:
        f1: float | None = 2 * true_positive / f1_denominator
        f1_reason = None
    else:
        f1 = None
        f1_reason = "no true or predicted water"
    return {
        "confusion_matrix": [
            [int(true_negative), int(false_positive)],
            [int(false_negative), int(true_positive)],
        ],
        "support": {
            "water": int(support_water),
            "non_water": int(support_nonwater),
        },
        "predicted": {
            "water": int(predicted_water),
            "non_water": int(predicted_nonwater),
        },
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "undefined_reasons": {
            "precision": precision_reason,
            "recall": recall_reason,
            "f1": f1_reason,
        },
    }


def score_binary_predictions(true_labels, predicted_labels):
    """Score aligned binary labels with the [[TN, FP], [FN, TP]] table."""
    truth = np.asarray(true_labels, dtype=np.uint8).reshape(-1)
    predicted = np.asarray(predicted_labels, dtype=np.uint8).reshape(-1)
    if truth.shape != predicted.shape:
        raise ValueError(
            f"truth {truth.shape} and predictions {predicted.shape} "
            "have different lengths")
    if set(np.unique(truth).tolist()) - {0, 1}:
        raise ValueError("true labels must be binary 0/1")
    if set(np.unique(predicted).tolist()) - {0, 1}:
        raise ValueError("predicted labels must be binary 0/1")
    true_negative = int(np.count_nonzero((truth == 0) & (predicted == 0)))
    false_positive = int(np.count_nonzero((truth == 0) & (predicted == 1)))
    false_negative = int(np.count_nonzero((truth == 1) & (predicted == 0)))
    true_positive = int(np.count_nonzero((truth == 1) & (predicted == 1)))
    return _score_from_counts(
        true_negative, false_positive, false_negative, true_positive)


def evaluate_test_predictions(test_dataset, predicted_labels):
    """Score each test area and the pooled counts summed across areas."""
    truth = np.asarray(test_dataset["y"], dtype=np.uint8).reshape(-1)
    predicted = np.asarray(predicted_labels, dtype=np.uint8).reshape(-1)
    if truth.shape != predicted.shape:
        raise ValueError("test predictions do not match the test dataset")
    areas = np.asarray(test_dataset["area"]).reshape(-1)
    per_area = {}
    pooled_counts = [0, 0, 0, 0]
    for area_id in sorted(np.unique(areas).tolist()):
        mask = areas == area_id
        scores = score_binary_predictions(truth[mask], predicted[mask])
        per_area[str(int(area_id))] = {
            "area_id": int(area_id),
            "rows": int(mask.sum()),
            **scores,
        }
        table = scores["confusion_matrix"]
        pooled_counts[0] += table[0][0]
        pooled_counts[1] += table[0][1]
        pooled_counts[2] += table[1][0]
        pooled_counts[3] += table[1][1]
    pooled = {
        "rows": int(len(truth)),
        **_score_from_counts(*pooled_counts),
    }
    return {
        "per_area": per_area,
        "pooled": pooled,
        "confusion_convention": (
            "rows true [0, 1], columns predicted [0, 1]: "
            "[[TN, FP], [FN, TP]]"),
    }


# %% Area GeoTIFFs and one prediction overlay per test area.

def _base_transform_and_crs(grid_reference):
    """Rebuild the frozen 10 m affine transform and CRS."""
    from rasterio.transform import Affine

    values = grid_reference["transform"]
    base = Affine(*[float(value) for value in values])
    crs = grid_reference["crs"]
    if not isinstance(crs, str) or not crs.strip():
        raise ValueError("grid reference CRS must be a non-empty string")
    return base, crs


def _area_window_transform(base_transform, window):
    """Return the georeferencing for one area crop window."""
    from rasterio import windows as rio_windows
    from rasterio.windows import Window as RioWindow

    row0, row1, col0, col1 = (int(window[0]), int(window[1]),
                              int(window[2]), int(window[3]))
    return rio_windows.transform(
        RioWindow(col0, row0, col1 - col0, row1 - row0), base_transform)


def _stack_valid_features(feature_window):
    """Stack one window's six features and mark the finite pixels."""
    first_shape = next(iter(feature_window.values())).shape
    height, width = int(first_shape[0]), int(first_shape[1])
    for name in c.SW_FEATURES:
        if name not in feature_window:
            raise ValueError(f"features are missing {name}")
        if feature_window[name].shape != (height, width):
            raise ValueError(
                f"{name} window is {feature_window[name].shape}, "
                f"expected {(height, width)}")
    valid = np.ones((height, width), dtype=bool)
    for name in c.SW_FEATURES:
        valid &= np.isfinite(feature_window[name])
    columns = [feature_window[name][valid] for name in c.SW_FEATURES]
    if valid.any():
        stacked = np.column_stack(columns).astype(np.float32, copy=False)
    else:
        stacked = np.empty((0, len(c.SW_FEATURES)), dtype=np.float32)
    return valid, stacked


def predict_area_grids(model, feature_window):
    """Predict every feature-valid pixel in one window.

    Excluded-but-valid pixels still receive predictions because
    exclusions suppress labels, not inference. Invalid pixels stay
    NoData. An all-invalid window returns all-NoData without calling
    the forest.
    """
    valid, stacked = _stack_valid_features(feature_window)
    probability_grid = np.full(
        valid.shape, np.nan, dtype=np.float32)
    binary_grid = np.full(
        valid.shape, BINARY_NODATA, dtype=np.uint8)
    if not valid.any():
        return probability_grid, binary_grid
    probabilities = np.empty(
        (int(valid.sum()),), dtype=np.float32)
    binaries = np.empty((int(valid.sum()),), dtype=np.uint8)
    for start in range(0, int(valid.sum()), PREDICTION_BATCH_ROWS):
        stop = min(start + PREDICTION_BATCH_ROWS,
                   int(valid.sum()))
        batch_probability, batch_binary = predict_water_labels(
            model, stacked[start:stop])
        probabilities[start:stop] = batch_probability
        binaries[start:stop] = batch_binary
    probability_grid[valid] = probabilities
    binary_grid[valid] = binaries
    return probability_grid, binary_grid


def _geotiff_tags(manifest, manifest_sha256, model_sha256, area_id, window):
    """Describe one export with frozen run identities only."""
    masks = manifest.get("masks", {})
    return {
        "tile": str(manifest.get("tile", "")),
        "month": str(manifest.get("month", "")),
        "area_id": str(int(area_id)),
        "window": ",".join(str(int(value)) for value in window),
        "source_scenes": ";".join(manifest.get("source_scenes", [])),
        "feature_order": ",".join(manifest.get("feature_order", [])),
        "masks": json.dumps(masks, sort_keys=True),
        "aggregation": str(manifest.get("aggregation", "")),
        "validity": str(manifest.get("validity", "")),
        "threshold": str(WATER_THRESHOLD),
        "tie_rule": "water when probability >= 0.5",
        "probability_meaning": (
            "uncalibrated RF water probability from balanced training"),
        "manifest_sha256": str(manifest_sha256),
        "model_sha256": str(model_sha256),
    }


def write_area_geotiffs(probability_path, binary_path,
                        probability_grid, binary_grid,
                        window, grid_reference, tags):
    """Write one probability/binary pair and verify them by read-back."""
    base_transform, crs = _base_transform_and_crs(grid_reference)
    area_transform = _area_window_transform(base_transform, window)
    height, width = probability_grid.shape
    if binary_grid.shape != (height, width):
        raise ValueError("probability and binary grids have different shapes")

    probability_write = np.where(
        np.isfinite(probability_grid),
        probability_grid.astype(np.float32),
        np.float32(PROBABILITY_NODATA)).astype(np.float32)
    if np.any((probability_write != np.float32(PROBABILITY_NODATA))
              & ((probability_write < 0.0) | (probability_write > 1.0))):
        raise ValueError("probability pixels left the [0, 1] range")

    def _write_one(path, array, dtype, nodata):
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": dtype,
            "crs": crs,
            "transform": area_transform,
            "nodata": nodata,
            "compress": "deflate",
        }
        with rasterio.open(path, "w", **profile) as destination:
            destination.write(array, 1)
            destination.update_tags(**{key: str(value)
                                       for key, value in tags.items()})

    _write_one(probability_path, probability_write, "float32",
               PROBABILITY_NODATA)
    _write_one(binary_path, binary_grid, "uint8", BINARY_NODATA)

    for path, expected, dtype, nodata in (
            (probability_path, probability_write, "float32",
             PROBABILITY_NODATA),
            (binary_path, binary_grid, "uint8", BINARY_NODATA)):
        with rasterio.open(path) as source:
            if (source.height, source.width) != (height, width):
                raise ValueError(f"raster {path} has the wrong dimensions")
            if str(source.crs) != str(crs):
                raise ValueError(f"raster {path} has the wrong CRS")
            if list(source.transform)[:6] != list(area_transform)[:6]:
                raise ValueError(
                    f"raster {path} has the wrong transform")
            if source.count != 1 or source.dtypes[0] != dtype:
                raise ValueError(f"raster {path} has the wrong band layout")
            if source.nodata != nodata:
                raise ValueError(f"raster {path} has the wrong nodata")
            actual = source.read(1)
            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                raise ValueError(
                    f"raster {path} read back with the wrong array type")
            if dtype == "float32":
                same_values = np.where(
                    expected == np.float32(PROBABILITY_NODATA),
                    actual == np.float32(PROBABILITY_NODATA),
                    actual == expected)
                if not np.all(same_values):
                    raise ValueError(
                        f"raster {path} pixels differ from the prediction")
            elif not np.array_equal(actual, expected):
                raise ValueError(
                    f"raster {path} pixels differ from the prediction")


def write_prediction_overlay(path, area_id, month, ndwi_window,
                             binary_grid, true_grid):
    """Write one inspected view of NDWI, prediction and test labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    height, width = binary_grid.shape
    if ndwi_window.shape != (height, width):
        raise ValueError("overlay NDWI and prediction have different shapes")
    if true_grid.shape != (height, width):
        raise ValueError("overlay truth and prediction have different shapes")

    display_ndwi = labelling_sw.colorise_ndwi(ndwi_window)
    prediction_display = np.full(
        (height, width, 3), 200, dtype=np.uint8)
    prediction_display[binary_grid == 1] = (30, 144, 255)
    prediction_display[binary_grid == 0] = (245, 235, 220)
    prediction_display[binary_grid == BINARY_NODATA] = (0, 0, 0)
    truth_display = np.full((height, width, 3), 90, dtype=np.uint8)
    truth_display[true_grid == 1] = (30, 144, 255)
    truth_display[true_grid == 0] = (255, 140, 0)
    truth_display[true_grid == BINARY_NODATA] = (0, 0, 0)

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(display_ndwi)
    axes[0].set_title("monthly NDWI ([-0.5, 0.5], NoData black)")
    axes[1].imshow(prediction_display)
    axes[1].set_title("predicted water (blue), land (cream)")
    axes[2].imshow(truth_display)
    axes[2].set_title("held-out labels (blue/orange)")
    figure.suptitle(f"area {int(area_id):03d} — {month} — "
                    "uncalibrated RF water probability >= 0.5")
    legend_items = [
        (patches.Patch(color="#1e90ff"), "water / predicted water"),
        (patches.Patch(color="#ff8c00"), "labelled non-water"),
        (patches.Patch(color="#f5ebdc"), "predicted non-water"),
        (patches.Patch(color="#000000"), "NoData / unevaluated"),
    ]
    labels = [text for _, text in legend_items]
    handles = [handle for handle, _ in legend_items]
    figure.legend(handles, labels, loc="lower center", ncol=4)
    for axis in axes:
        axis.set_axis_off()
    figure.tight_layout(rect=(0, 0.06, 1, 0.90))
    figure.savefig(path, dpi=150)
    plt.close(figure)
    if not Path(path).is_file() or Path(path).stat().st_size == 0:
        raise ValueError(f"overlay {path} was not written")


def _metrics_core(metrics):
    """Return the comparable core of a metrics record for retry checks."""
    return {
        "tile": metrics.get("tile"),
        "month": metrics.get("month"),
        "identities": metrics.get("identities"),
        "feature_order": metrics.get("feature_order"),
        "threshold": metrics.get("threshold"),
        "per_area": metrics.get("per_area"),
        "pooled": metrics.get("pooled"),
    }


def _publish_staged_file(staged_path, final_path, compare_core=None):
    """Move one staged file into the run, preserving an identical result.

    Returns "published" when the file is newly moved and "preserved"
    when an identical file already exists. Any differing existing file
    raises instead of overwriting a successful result.
    """
    final_path = Path(final_path)
    if not final_path.exists():
        os.replace(staged_path, final_path)
        return "published"
    if compare_core is None:
        raise FileExistsError(
            f"refusing to overwrite successful output {final_path}")
    try:
        with open(final_path, encoding="utf-8") as handle:
            existing = json.load(handle)
        with open(staged_path, encoding="utf-8") as handle:
            staged = json.load(handle)
    except (OSError, ValueError) as exc:
        raise FileExistsError(
            f"refusing to overwrite {final_path}: "
            f"existing file cannot be compared ({exc})") from exc
    if compare_core(existing) != compare_core(staged):
        raise FileExistsError(
            f"refusing to overwrite {final_path} with different content; "
            "use a new run directory for changed inputs or code")
    try:
        os.remove(staged_path)
    except OSError:
        pass
    return "preserved"


def _publish_staged_binary(staged_path, final_path):
    """Preserve an identical raster/overlay or publish a missing one."""
    final_path = Path(final_path)
    if not final_path.exists():
        os.replace(staged_path, final_path)
        return "published"
    if labelling_sw.file_sha256(staged_path) != \
            labelling_sw.file_sha256(final_path):
        raise FileExistsError(
            f"refusing to overwrite {final_path} with different bytes; "
            "use a new run directory for changed inputs or code")
    try:
        os.remove(staged_path)
    except OSError:
        pass
    return "preserved"


def _record_evaluation_failure(run_dir, metrics_path, error):
    """Keep an export failure beside any scores already published."""
    record = {
        "failed_at_utc": datetime.now(timezone.utc).isoformat(),
        "error_type": type(error).__name__,
        "error": str(error),
        "metrics_sha256": (
            labelling_sw.file_sha256(metrics_path)
            if metrics_path.is_file() else None),
    }
    with open(run_dir / EVALUATE_FAILURES_FILENAME, "a",
              encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")


def evaluate_run(input_dir, run_dir, source_image_root,
                 contract=BASELINE_CONTRACT):
    """Evaluate the saved forest once on the frozen test dataset.

    Reports per-area and pooled water scores, writes four
    probability/binary GeoTIFF pairs plus one overlay per test area,
    and records exposure and retry history. Re-running after an
    export failure preserves identical metrics and finishes the
    missing files; it never overwrites a differing successful result.
    """
    input_dir = Path(input_dir).resolve()
    run_dir = Path(run_dir).resolve()
    source_image_root = Path(source_image_root).resolve()
    manifest_path = run_dir / "v1-sampling.json"
    model_path = run_dir / MODEL_FILENAME
    test_path = run_dir / "v1-test.npz"
    metrics_path = run_dir / METRICS_FILENAME
    evaluate_marker_path = run_dir / EVALUATE_COMPLETE_FILENAME
    for path in (manifest_path, model_path, test_path,
                 run_dir / "prepare-complete.json",
                 run_dir / FIT_COMPLETE_FILENAME):
        if not path.is_file():
            raise ValueError(
                f"run is not fitted: missing {path}; "
                "run prepare then fit first")
    if evaluate_marker_path.exists():
        raise FileExistsError(
            f"evaluation already succeeded in {run_dir}; "
            "do not overwrite a successful evaluation")

    manifest, manifest_sha256 = _load_run_manifest(run_dir)
    _check_manifest_contract(manifest, contract)
    _recheck_input_digests(
        manifest["inputs"]["paths_and_sha256"], phase="evaluation")
    _recheck_grid_reference(manifest["grid_reference"], phase="evaluation")

    model_bundle = _load_model_bundle(model_path)
    if model_bundle.get("manifest_sha256") != manifest_sha256:
        raise ValueError(
            f"model {model_path} was fitted from a different manifest "
            "revision")
    model_sha256 = labelling_sw.file_sha256(model_path)
    expected_test_sha256 = manifest["datasets"]["test"]["sha256"]
    actual_test_sha256 = labelling_sw.file_sha256(test_path)
    if actual_test_sha256 != expected_test_sha256:
        raise ValueError(
            f"test dataset changed: {test_path} no longer matches "
            "the frozen manifest")
    test_dataset = _load_npz_dataset(test_path)
    _validate_dataset_content(test_dataset, "test dataset")
    _require_exact_areas(
        test_dataset, contract.test_area_ids, "test dataset")

    model = model_bundle["model"]
    test_probabilities, test_predictions = predict_water_labels(
        model, test_dataset["X"])
    scoring = evaluate_test_predictions(test_dataset, test_predictions)

    test_assignments = [
        {"id": area["id"], "window": list(area["window"])}
        for area in manifest["areas"]
        if area["id"] in contract.test_area_ids
    ]
    if sorted(entry["id"] for entry in test_assignments) != \
            sorted(contract.test_area_ids):
        raise ValueError("manifest areas do not hold the four test windows")
    feature_windows, _ = _load_feature_windows(input_dir, test_assignments)
    window_by_id = {entry["id"]: list(entry["window"])
                    for entry in test_assignments}

    area_grids = {}
    for area_id in sorted(contract.test_area_ids):
        probability_grid, binary_grid = predict_area_grids(
            model, feature_windows[area_id])
        area_grids[area_id] = (probability_grid, binary_grid)

    for area_id in sorted(contract.test_area_ids):
        window = window_by_id[area_id]
        row0, _, col0, _ = window
        _, binary_grid = area_grids[area_id]
        mask = test_dataset["area"] == area_id
        for row, col, expected in zip(
                test_dataset["row"][mask].tolist(),
                test_dataset["col"][mask].tolist(),
                test_predictions[mask].tolist()):
            local_row = int(row) - int(row0)
            local_col = int(col) - int(col0)
            if binary_grid[local_row, local_col] != int(expected):
                raise AssertionError(
                    f"area {area_id:03d} raster disagrees with its "
                    "evaluation prediction")

    temporary_dir = Path(tempfile.mkdtemp(
        prefix=".evaluate-", dir=run_dir.parent))
    scores_published = False
    try:
        staged_metrics = temporary_dir / METRICS_FILENAME
        metrics = {
            "schema_version": METRICS_SCHEMA_VERSION,
            "phase": "evaluated",
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "evaluation_status": (
                "retry-after-partial-export"
                if metrics_path.exists() else "initial"),
            "held_out_exposure": (
                "Held-out predictions and scores were first generated "
                "in this evaluation. Engineering and file-integrity "
                "checks do not count as exposure; any later bug fix "
                "that changes sampling, labels, features, predictions "
                "or scoring must call its result test-exposed."),
            "tile": contract.tile,
            "month": contract.month,
            "identities": {
                "manifest_sha256": manifest_sha256,
                "model_sha256": model_sha256,
                "train_dataset_sha256":
                    manifest["datasets"]["train"]["sha256"],
                "test_dataset_sha256": actual_test_sha256,
            },
            "feature_order": list(c.SW_FEATURES),
            "threshold": WATER_THRESHOLD,
            "tie_rule": "water when probability >= 0.5",
            "probability_meaning": (
                "uncalibrated RF water probability from balanced training"),
            "confusion_convention": scoring["confusion_convention"],
            "per_area": scoring["per_area"],
            "pooled": scoring["pooled"],
            "limitations": list(manifest.get("limitations", [])),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scikit_learn": _package_version("scikit-learn"),
                "rasterio": rasterio.__version__,
            },
            "implementation": _implementation_identity(),
        }
        _write_json(staged_metrics, metrics)
        publish_states = {}
        publish_states[str(metrics_path)] = _publish_staged_file(
            staged_metrics, metrics_path, compare_core=_metrics_core)
        scores_published = True

        staged_artifacts = []
        for area_id in sorted(contract.test_area_ids):
            window = window_by_id[area_id]
            row0, _, col0, _ = window
            probability_grid, binary_grid = area_grids[area_id]
            tags = _geotiff_tags(
                manifest, manifest_sha256, model_sha256, area_id, window)
            staged_probability = temporary_dir / (
                f"area-{area_id:03d}-water-probability.tif")
            staged_binary = temporary_dir / (
                f"area-{area_id:03d}-water-binary.tif")
            write_area_geotiffs(
                staged_probability, staged_binary,
                probability_grid, binary_grid,
                window, manifest["grid_reference"], tags)
            staged_artifacts.append((
                staged_probability,
                run_dir / f"area-{area_id:03d}-water-probability.tif",
                "binary"))
            staged_artifacts.append((
                staged_binary,
                run_dir / f"area-{area_id:03d}-water-binary.tif",
                "binary"))

            true_grid = np.full(
                probability_grid.shape, BINARY_NODATA, dtype=np.uint8)
            mask = test_dataset["area"] == area_id
            for row, col, label in zip(
                    test_dataset["row"][mask].tolist(),
                    test_dataset["col"][mask].tolist(),
                    test_dataset["y"][mask].tolist()):
                true_grid[int(row) - int(row0),
                          int(col) - int(col0)] = int(label)
            staged_overlay = temporary_dir / (
                f"area-{area_id:03d}-overlay.png")
            write_prediction_overlay(
                staged_overlay, area_id, contract.month,
                feature_windows[area_id]["NDWI"], binary_grid, true_grid)
            staged_artifacts.append((
                staged_overlay,
                run_dir / f"area-{area_id:03d}-overlay.png",
                "binary"))

        for staged_path, final_path, _kind in staged_artifacts:
            publish_states[str(final_path)] = _publish_staged_binary(
                staged_path, final_path)
        retried = any(
            state == "preserved" for state in publish_states.values())
        _write_json(temporary_dir / EVALUATE_COMPLETE_FILENAME, {
            "schema_version": METRICS_SCHEMA_VERSION,
            "phase": "evaluated",
            "manifest_sha256": manifest_sha256,
            "model_sha256": model_sha256,
            "test_dataset_sha256": actual_test_sha256,
            "metrics_sha256": labelling_sw.file_sha256(metrics_path),
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "retried_partial_export": bool(retried),
            "published": publish_states,
        })
        os.replace(
            temporary_dir / EVALUATE_COMPLETE_FILENAME,
            evaluate_marker_path)
        temporary_dir = None
        with open(metrics_path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as error:
        if scores_published:
            _record_evaluation_failure(run_dir, metrics_path, error)
        raise
    finally:
        if temporary_dir is not None and temporary_dir.exists():
            shutil.rmtree(temporary_dir)
