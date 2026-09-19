"""Prepare reproducible pixel datasets for the V1 surface-water baseline."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
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


def _recheck_input_digests(input_digests):
    for path_text, expected_digest in input_digests.items():
        path = Path(path_text)
        try:
            actual_digest = labelling_sw.file_sha256(path)
        except OSError as exc:
            raise ValueError(
                f"input changed or disappeared during preparation: "
                f"{path}: {exc}") from exc
        if actual_digest != expected_digest:
            raise ValueError(
                f"input changed during preparation: {path}")


def _recheck_grid_reference(grid_reference):
    expected = {
        key: grid_reference[key] for key in ("shape", "crs", "transform")}
    for reference in grid_reference["references"]:
        actual = _grid_metadata(Path(reference["path"]))
        if actual != expected:
            raise ValueError(
                "source reference grid changed during preparation: "
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
