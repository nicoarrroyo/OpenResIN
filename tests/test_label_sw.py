import json
import os

import numpy as np

from openresin import label_sw


def test_find_known_feature_masks_finds_nested_urban_tif(
        tmp_path, monkeypatch):
    """The documented CEH archive keeps the urban TIFF four levels deep."""
    urban_path = (
        tmp_path
        / "masks"
        / "urban-areas"
        / "CEH_GBLandCover_2024_10m"
        / "data"
        / "4dd9df19-8df5-41a0-9829-8f6114e28db1"
        / "gblcm2024_10m.tif"
    )
    urban_path.parent.mkdir(parents=True)
    urban_path.touch()
    monkeypatch.setattr(label_sw.c, "DATA_DIR", str(tmp_path))

    _, found_urban_path = label_sw._find_known_feature_masks()

    assert found_urban_path == str(urban_path)


def test_main_passes_requested_month_to_overview(tmp_path, monkeypatch):
    """The overview provenance must name the month selected by the CLI."""
    scene = (
        tmp_path
        / "S2B_MSIL2A_20270315T110619_N0512_R137_T31UCU_X.SAFE"
    )
    calls = []
    monkeypatch.setattr(
        label_sw.sw, "discover_scenes", lambda _sat_images_dir: [str(scene)])
    monkeypatch.setattr(
        label_sw, "_create_navigation_overview",
        lambda out_dir, scenes, device, month: calls.append(month))
    monkeypatch.setattr(
        label_sw, "_create_monthly_features",
        lambda out_dir, scenes, device, month: None)

    result = label_sw.main([
        "--month", "2027-03",
        "--out-dir", str(tmp_path / "outputs"),
    ])

    assert result == 0
    assert calls == ["2027-03"]


def test_prepare_annotation_chips_adds_two_stage_ndwi_composite(
        tmp_path, monkeypatch, capsys):
    """NDWI follows date-first aggregation and sits after the TCI composite."""
    scenes = [
        "/S2A_MSIL2A_20260427T100000_N0000_R001_T31UCU_A.SAFE",
        "/S2B_MSIL2A_20260427T110000_N0000_R002_T31UCU_B.SAFE",
        "/S2C_MSIL2A_20260430T100000_N0000_R001_T31UCU_C.SAFE",
    ]
    tci_values = dict(zip(scenes, (10.0, 30.0, 100.0)))
    band_values = {
        (scenes[0], "B03"): 75.0,
        (scenes[0], "B08"): 25.0,
        (scenes[1], "B03"): 25.0,
        (scenes[1], "B08"): 75.0,
        (scenes[2], "B03"): 100.0,
        (scenes[2], "B08"): 0.0,
    }
    monkeypatch.setattr(
        label_sw.sw, "read_tci_window",
        lambda scene, _window: np.full((2, 2, 3), tci_values[scene]))
    monkeypatch.setattr(
        label_sw.sw, "read_band_window",
        lambda scene, band, _window: np.full(
            (2, 2), band_values[(scene, band)], dtype=np.float32))

    chips = label_sw._prepare_annotation_chips(
        scenes, (0, 2, 0, 2), str(tmp_path), "2026-04", "T31UCU")

    assert list(chips) == [
        "Composite", "NDWI (raw)", "20260427", "20260430"]
    assert np.all(chips["Composite"] == 60)
    assert np.all(chips["20260427"] == 20)
    assert np.all(chips["20260430"] == 100)
    assert chips["NDWI (raw)"].shape == (2, 2, 3)
    assert chips["NDWI (raw)"].dtype == np.uint8
    assert np.all(chips["NDWI (raw)"] == [5, 48, 97])
    assert "lacks cloud, sea and urban masking" in capsys.readouterr().out


def _write_matching_archive(out_dir, scenes, ndwi, sea_path, urban_path,
                            provenance_overrides=None):
    """Save a synthetic features archive with consistent provenance."""
    provenance = {
        "tile": "T31UCU",
        "month": "2026-04",
        "source_scenes": sorted(os.path.basename(s) for s in scenes),
        "feature_order": list(label_sw.c.SW_FEATURES),
        "aggregation": "valid median within date, then median across dates",
        "masks": {
            "cloud_shadow_classes": list(
                label_sw.c.SW_CLOUD_SHADOW_CLASSES),
            "nodata_value": label_sw.c.SW_NODATA_VALUE,
            "sea_source": sea_path,
            "urban_source": urban_path,
        },
    }
    provenance.update(provenance_overrides or {})
    np.savez_compressed(
        os.path.join(out_dir, "features.npz"), NDWI=ndwi)
    with open(os.path.join(out_dir, "features-provenance.json"), "w",
              encoding="utf-8") as handle:
        json.dump(provenance, handle)
    return provenance


def _tiny_tile_setup(tmp_path, monkeypatch):
    """Shrink the tile to 4 px and fix the mask lookup for fixtures."""
    monkeypatch.setattr(label_sw.c, "SW_TILE_PX", 4)
    monkeypatch.setattr(
        label_sw, "_find_known_feature_masks",
        lambda: ("sea.geojson", "urban.tif"))
    return ["/S2A_MSIL2A_20260427T100000_N0000_R001_T31UCU_A.SAFE"]


def test_prepare_annotation_chips_prefers_masked_archive_ndwi(
        tmp_path, monkeypatch, capsys):
    """A current archive supplies the cropped masked NDWI, labelled as such."""
    scenes = _tiny_tile_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        label_sw.sw, "read_tci_window",
        lambda _scene, _window: np.full((2, 2, 3), 60.0))
    ndwi = np.array([
        [0.50, -0.50, 0.00, 0.13],
        [0.25, np.nan, -0.25, 0.40],
        [-0.40, 0.10, 0.13, -0.10],
        [0.00, 0.30, -0.30, np.nan],
    ], dtype=np.float32)
    _write_matching_archive(
        str(tmp_path), scenes, ndwi, "sea.geojson", "urban.tif")

    chips = label_sw._prepare_annotation_chips(
        scenes, (1, 3, 0, 2), str(tmp_path), "2026-04", "T31UCU")

    assert list(chips)[:2] == ["Composite", "NDWI (masked)"]
    expected = label_sw.sw.colorise_ndwi(ndwi[1:3, 0:2])
    assert np.array_equal(chips["NDWI (masked)"], expected)
    assert np.array_equal(chips["NDWI (masked)"][0, 1], [0, 0, 0])
    assert "NDWI fallback" not in capsys.readouterr().out


def test_load_masked_ndwi_window_crops_edge_cells(
        tmp_path, monkeypatch):
    """Windows on the tile edge clip exactly inside the saved array."""
    scenes = _tiny_tile_setup(tmp_path, monkeypatch)
    ndwi = np.arange(16, dtype=np.float32).reshape(4, 4) / 16
    _write_matching_archive(
        str(tmp_path), scenes, ndwi, "sea.geojson", "urban.tif")

    cell, reason = label_sw._load_masked_ndwi_window(
        str(tmp_path), "2026-04", "T31UCU", scenes, (2, 4, 2, 4))

    assert reason is None
    assert np.array_equal(cell, ndwi[2:4, 2:4])


def test_load_masked_ndwi_window_names_each_stale_field(
        tmp_path, monkeypatch):
    """Each provenance mismatch reports its own field, not a bare refusal."""
    scenes = _tiny_tile_setup(tmp_path, monkeypatch)
    ndwi = np.zeros((4, 4), dtype=np.float32)
    stale_cases = [
        ({"month": "2026-05"}, "2026-04", scenes, "stale: provenance month"),
        ({"source_scenes": ["other.SAFE"]}, "2026-04", scenes,
         "stale: provenance source_scenes"),
        ({"feature_order": ["B02"]}, "2026-04", scenes,
         "stale: provenance feature_order"),
        ({"masks": {}}, "2026-04", scenes,
         "stale: provenance cloud_shadow_classes"),
    ]
    for overrides, month, call_scenes, expected in stale_cases:
        _write_matching_archive(
            str(tmp_path), scenes, ndwi, "sea.geojson", "urban.tif",
            provenance_overrides=overrides)

        cell, reason = label_sw._load_masked_ndwi_window(
            str(tmp_path), month, "T31UCU", call_scenes, (0, 2, 0, 2))

        assert cell is None
        assert reason.startswith(expected)


def test_load_masked_ndwi_window_rejects_missing_archive(tmp_path):
    """No files on disk is the absent fallback reason."""
    cell, reason = label_sw._load_masked_ndwi_window(
        str(tmp_path), "2026-04", "T31UCU", ["scene.SAFE"], (0, 2, 0, 2))

    assert cell is None
    assert reason.startswith("absent:")


def test_load_masked_ndwi_window_rejects_corrupt_archive(
        tmp_path, monkeypatch):
    """Garbage bytes read as corrupt, never as water labels."""
    scenes = _tiny_tile_setup(tmp_path, monkeypatch)
    _write_matching_archive(
        str(tmp_path), scenes, np.zeros((4, 4), dtype=np.float32),
        "sea.geojson", "urban.tif")
    with open(os.path.join(str(tmp_path), "features.npz"), "wb") as handle:
        handle.write(b"not a zip archive")

    cell, reason = label_sw._load_masked_ndwi_window(
        str(tmp_path), "2026-04", "T31UCU", scenes, (0, 2, 0, 2))

    assert cell is None
    assert reason.startswith("corrupt:")


def test_load_masked_ndwi_window_rejects_wrong_shape(
        tmp_path, monkeypatch):
    """An NDWI array that does not cover the tile cannot supply a window."""
    scenes = _tiny_tile_setup(tmp_path, monkeypatch)
    _write_matching_archive(
        str(tmp_path), scenes, np.zeros((3, 3), dtype=np.float32),
        "sea.geojson", "urban.tif")

    cell, reason = label_sw._load_masked_ndwi_window(
        str(tmp_path), "2026-04", "T31UCU", scenes, (0, 2, 0, 2))

    assert cell is None
    assert reason.startswith("wrong-shape:")


def test_load_masked_ndwi_window_requires_both_masks_on_disk(
        tmp_path, monkeypatch):
    """Provenance agreeing that no masks exist still refuses the masked label."""
    scenes = _tiny_tile_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        label_sw, "_find_known_feature_masks", lambda: (None, None))
    _write_matching_archive(
        str(tmp_path), scenes, np.zeros((4, 4), dtype=np.float32),
        None, None)

    cell, reason = label_sw._load_masked_ndwi_window(
        str(tmp_path), "2026-04", "T31UCU", scenes, (0, 2, 0, 2))

    assert cell is None
    assert "mask source is missing" in reason


def _write_full_features(out_dir, scenes, values, sea_path, urban_path):
    """Save all six features on a tiny tile with matching provenance."""
    provenance = {
        "tile": "T31UCU",
        "month": "2026-04",
        "source_scenes": sorted(os.path.basename(s) for s in scenes),
        "feature_order": list(label_sw.c.SW_FEATURES),
        "aggregation": "valid median within date, then median across dates",
        "masks": {
            "cloud_shadow_classes": list(
                label_sw.c.SW_CLOUD_SHADOW_CLASSES),
            "nodata_value": label_sw.c.SW_NODATA_VALUE,
            "sea_source": sea_path,
            "urban_source": urban_path,
        },
    }
    arrays = {name: np.array(values, dtype=np.float32)
              for name in label_sw.c.SW_FEATURES}
    np.savez_compressed(os.path.join(out_dir, "features.npz"), **arrays)
    with open(os.path.join(out_dir, "features-provenance.json"), "w",
              encoding="utf-8") as handle:
        json.dump(provenance, handle)
    return provenance


def _tiny_features_setup(tmp_path, monkeypatch, values=None):
    """Fix a 4 px tile and save uniform valid features for fixtures."""
    scenes = _tiny_tile_setup(tmp_path, monkeypatch)
    if values is None:
        values = np.ones((4, 4), dtype=np.float32)
    _write_full_features(
        str(tmp_path), scenes, values, "sea.geojson", "urban.tif")
    return scenes


def _area_record(window, polygons, exclusions=None, completion=None):
    record = {"tile": "T31UCU", "area_id": 1, "split": "train",
              "window": list(window), "polygons": polygons,
              "exclusions": exclusions or []}
    if completion is not None:
        record["completion"] = completion
    return record


def test_derive_incomplete_uses_only_explicit_polygons(tmp_path, monkeypatch):
    """Incomplete areas leave other usable pixels unreviewed, not non-water."""
    monkeypatch.setattr(label_sw.c, "SW_TILE_PX", 4)
    window = [0, 4, 0, 4]
    features = {name: np.ones((4, 4), dtype=np.float32)
                for name in label_sw.c.SW_FEATURES}
    features["B02"][0, 0] = np.nan  # invalid even under the water polygon
    record = _area_record(window, [
        {"id": 1, "class": "water",
         "vertices_scene": [[0, 0], [2, 0], [2, 2], [0, 2]]},
        {"id": 2, "class": "non-water",
         "vertices_scene": [[2, 2], [3, 2], [3, 3], [2, 3]]},
    ])

    mask, counts, info = label_sw.derive_area_label_state(
        record, features, window, False)

    assert mask[0, 0] == label_sw.LABEL_UNUSABLE
    assert mask[0, 1] == label_sw.LABEL_WATER
    assert mask[2, 2] == label_sw.LABEL_NONWATER
    assert mask[3, 3] == label_sw.LABEL_WITHHELD
    assert counts["water"] + counts["non-water"] + counts["unusable"] \
        + counts["withheld"] == 16
    assert info["completion_active"] is False
    assert info["split"] == "train"


def test_derive_complete_turns_background_to_nonwater(tmp_path, monkeypatch):
    """A valid completion labels usable ground, but keeps exclusions back."""
    monkeypatch.setattr(label_sw.c, "SW_TILE_PX", 4)
    window = [0, 4, 0, 4]
    features = {name: np.ones((4, 4), dtype=np.float32)
                for name in label_sw.c.SW_FEATURES}
    record = _area_record(window, [
        {"id": 1, "class": "water",
         "vertices_scene": [[0, 0], [2, 0], [2, 2], [0, 2]]},
        {"id": 2, "class": "non-water",
         "vertices_scene": [[3, 3], [4, 3], [4, 4], [3, 4]]},
    ], exclusions=[
        {"vertices_scene": [[0, 3], [1, 3], [1, 4], [0, 4]]},
    ])

    mask, counts, _ = label_sw.derive_area_label_state(
        record, features, window, True)

    assert mask[0, 1] == label_sw.LABEL_WATER
    assert mask[3, 3] == label_sw.LABEL_NONWATER
    assert mask[1, 3] == label_sw.LABEL_NONWATER
    assert mask[3, 0] == label_sw.LABEL_WITHHELD
    assert mask[0, 3] == label_sw.LABEL_NONWATER
    assert counts["withheld"] == 1
    assert counts["unusable"] == 0


def test_derive_exclusion_beats_water_and_rejects_overlap(monkeypatch):
    """Exclusions withhold even water pixels; class overlaps raise."""
    monkeypatch.setattr(label_sw.c, "SW_TILE_PX", 4)
    window = [0, 4, 0, 4]
    features = {name: np.ones((4, 4), dtype=np.float32)
                for name in label_sw.c.SW_FEATURES}
    water = {"id": 1, "class": "water",
             "vertices_scene": [[0, 0], [2, 0], [2, 2], [0, 2]]}
    excluded = _area_record(window, [water], exclusions=[
        {"vertices_scene": [[0, 0], [1, 0], [1, 1], [0, 1]]}])
    mask, _, _ = label_sw.derive_area_label_state(
        excluded, features, window, True)
    assert mask[0, 0] == label_sw.LABEL_WITHHELD
    assert mask[0, 1] == label_sw.LABEL_WATER

    clashing = _area_record(window, [
        water,
        {"id": 2, "class": "non-water",
         "vertices_scene": [[0, 0], [2, 0], [2, 2], [0, 2]]},
    ])
    try:
        label_sw.derive_area_label_state(clashing, features, window, False)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("expected an overlap error")


def test_derive_uses_final_validity_not_valid_count(monkeypatch):
    """A sea/urban-masked pixel with a positive count stays unusable."""
    monkeypatch.setattr(label_sw.c, "SW_TILE_PX", 4)
    window = [0, 4, 0, 4]
    features = {name: np.ones((4, 4), dtype=np.float32)
                for name in label_sw.c.SW_FEATURES}
    features["NDVI"][1, 1] = np.nan  # post-mask gap after compositing
    record = _area_record(window, [
        {"id": 1, "class": "water",
         "vertices_scene": [[0, 0], [4, 0], [4, 4], [0, 4]]},
    ])

    mask, counts, _ = label_sw.derive_area_label_state(
        record, features, window, True)

    assert mask[1, 1] == label_sw.LABEL_UNUSABLE
    assert counts["unusable"] == 1
    assert counts["water"] == 15


def test_check_completion_reports_stale_reasons(tmp_path, monkeypatch):
    """Month, archive, provenance and annotation edits each invalidate."""
    scenes = _tiny_features_setup(tmp_path, monkeypatch)
    features_digest, provenance_digest, _ = \
        label_sw.compute_feature_digests(str(tmp_path))
    window = [0, 4, 0, 4]
    polygons = [{"id": 1, "class": "water",
                 "vertices_scene": [[0, 0], [2, 0], [2, 2], [0, 2]]}]
    annotations = label_sw.sw.annotations_digest(
        "T31UCU", 1, "train", window, polygons, [])
    record = _area_record(window, polygons, [], {
        "schema_version": 1, "month": "2026-04",
        "features_sha256": features_digest,
        "features_provenance_sha256": provenance_digest,
        "annotations_sha256": annotations})

    state, _, active = label_sw.check_completion_active(
        record, "2026-04", "T31UCU", 1, "train", window,
        features_digest, provenance_digest)
    assert (state, active) == ("complete", True)

    state, reason, active = label_sw.check_completion_active(
        record, "2026-05", "T31UCU", 1, "train", window,
        features_digest, provenance_digest)
    assert state == "stale" and not active and "month" in reason

    state, _, active = label_sw.check_completion_active(
        record, "2026-04", "T31UCU", 1, "train", window,
        "0" * 64, provenance_digest)
    assert state == "stale" and not active

    edited = _area_record(window, [
        {"id": 1, "class": "water",
         "vertices_scene": [[0, 0], [3, 0], [3, 3], [0, 3]]},
    ], [], record["completion"])
    state, reason, active = label_sw.check_completion_active(
        edited, "2026-04", "T31UCU", 1, "train", window,
        features_digest, provenance_digest)
    assert state == "stale" and not active and "annotations" in reason

    assert label_sw.check_completion_active(
        _area_record(window, polygons), "2026-04", "T31UCU", 1,
        "train", window, features_digest,
        provenance_digest)[0] == "incomplete"


def test_check_completion_available_needs_all_features(
        tmp_path, monkeypatch):
    """NDWI alone is not enough; every classifier feature must be readable."""
    scenes = _tiny_tile_setup(tmp_path, monkeypatch)
    _write_matching_archive(
        str(tmp_path), scenes, np.ones((4, 4), dtype=np.float32),
        "sea.geojson", "urban.tif")

    available, reason = label_sw.check_completion_available(
        str(tmp_path), "2026-04", "T31UCU", scenes, (0, 2, 0, 2))

    assert available is False
    assert "B02" in reason or "features" in reason


def _annotate_setup(tmp_path, monkeypatch, polygons):
    """Point the area workflow at a tiny out-dir with valid features."""
    scenes = _tiny_features_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(label_sw.c, "SW_TILE_PX", 4)
    monkeypatch.setattr(
        label_sw.sw, "grid_cell_window", lambda _cell: (0, 4, 0, 4))
    monkeypatch.setattr(
        label_sw, "_lookup_area_tile_and_split",
        lambda _out_dir, _cell, _tile: ("T31UCU", "train"))
    monkeypatch.setattr(
        label_sw, "_prepare_annotation_chips",
        lambda _scenes, _window, _out_dir, _month, _tile: {
            "Composite": np.zeros((4, 4, 3), dtype=np.uint8)})
    path = os.path.join(str(tmp_path), "area-001.json")
    label_sw.sw.save_area_record(path, _area_record(
        [0, 4, 0, 4], polygons))
    return scenes, path


def _area_polygons_in_editor_format(polygons_scene):
    """Convert saved scene polygons to the editor's area format."""
    converted = []
    for polygon in polygons_scene:
        converted.append({
            "class": polygon["class"],
            "vertices": [[x, y] for x, y in polygon["vertices_scene"]],
        })
    return converted


def test_annotate_grid_area_finish_keeps_valid_completion(tmp_path,
                                                          monkeypatch):
    """A no-op Finish never rewrites the file or drops the decision."""
    polygons = [{"id": 1, "class": "water",
                 "vertices_scene": [[0, 0], [2, 0], [2, 2], [0, 2]]}]
    scenes, path = _annotate_setup(tmp_path, monkeypatch, polygons)
    features_digest, provenance_digest, _ = \
        label_sw.compute_feature_digests(str(tmp_path))
    annotations = label_sw.sw.annotations_digest(
        "T31UCU", 1, "train", [0, 4, 0, 4], polygons, [])
    completed = _area_record([0, 4, 0, 4], polygons, [], {
        "schema_version": 1, "month": "2026-04",
        "features_sha256": features_digest,
        "features_provenance_sha256": provenance_digest,
        "annotations_sha256": annotations})
    label_sw.sw.save_area_record(path, completed)
    before = open(path, encoding="utf-8").read()
    editor_polygons = _area_polygons_in_editor_format(polygons)
    monkeypatch.setattr(
        label_sw.sw, "annotate_reviewed_area",
        lambda *args, **kwargs: ([], editor_polygons, [], [], "finish"))

    result = label_sw._annotate_grid_area(
        str(tmp_path), scenes, 1, "2026-04")

    assert result == 0
    assert open(path, encoding="utf-8").read() == before


def test_annotate_grid_area_edit_drops_completion(tmp_path, monkeypatch):
    """An actual polygon edit removes the decision but keeps the drawings."""
    polygons = [{"id": 1, "class": "water",
                 "vertices_scene": [[0, 0], [2, 0], [2, 2], [0, 2]]}]
    scenes, path = _annotate_setup(tmp_path, monkeypatch, polygons)
    features_digest, provenance_digest, _ = \
        label_sw.compute_feature_digests(str(tmp_path))
    annotations = label_sw.sw.annotations_digest(
        "T31UCU", 1, "train", [0, 4, 0, 4], polygons, [])
    label_sw.sw.save_area_record(path, _area_record(
        [0, 4, 0, 4], polygons, [], {
            "schema_version": 1, "month": "2026-04",
            "features_sha256": features_digest,
            "features_provenance_sha256": provenance_digest,
            "annotations_sha256": annotations}))
    edited_scene = [[0, 0], [3, 0], [3, 3], [0, 3]]
    monkeypatch.setattr(
        label_sw.sw, "annotate_reviewed_area",
        lambda *args, **kwargs: (
            [], [{"class": "water",
                  "vertices": [list(p) for p in edited_scene]}],
            [], [], "finish"))

    result = label_sw._annotate_grid_area(
        str(tmp_path), scenes, 1, "2026-04")

    assert result == 0
    saved = label_sw.sw.load_area_record(path)
    assert "completion" not in saved
    assert saved["polygons"][0]["vertices_scene"] == edited_scene


def test_annotate_grid_area_complete_and_reopen(tmp_path, monkeypatch):
    """Complete records digests; Reopen removes the decision only."""
    polygons = [{"id": 1, "class": "water",
                 "vertices_scene": [[0, 0], [2, 0], [2, 2], [0, 2]]}]
    scenes, path = _annotate_setup(tmp_path, monkeypatch, polygons)
    editor_polygons = _area_polygons_in_editor_format(polygons)
    monkeypatch.setattr(
        label_sw.sw, "annotate_reviewed_area",
        lambda *args, **kwargs: ([], editor_polygons, [], [], "complete"))

    assert label_sw._annotate_grid_area(
        str(tmp_path), scenes, 1, "2026-04") == 0
    completed = label_sw.sw.load_area_record(path)
    assert completed["completion"]["month"] == "2026-04"
    assert completed["polygons"] == polygons
    features_digest, provenance_digest, _ = \
        label_sw.compute_feature_digests(str(tmp_path))
    assert completed["completion"]["features_sha256"] == features_digest
    assert completed["completion"]["features_provenance_sha256"] == \
        provenance_digest

    monkeypatch.setattr(
        label_sw.sw, "annotate_reviewed_area",
        lambda *args, **kwargs: ([], editor_polygons, [], [], "reopen"))
    assert label_sw._annotate_grid_area(
        str(tmp_path), scenes, 1, "2026-04") == 0
    reopened = label_sw.sw.load_area_record(path)
    assert "completion" not in reopened
    assert reopened["polygons"] == polygons


def test_annotate_grid_area_rejects_malformed_record(tmp_path, monkeypatch):
    """A contradictory file is reported, never silently overwritten."""
    scenes = _tiny_features_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(label_sw.c, "SW_TILE_PX", 4)
    monkeypatch.setattr(
        label_sw.sw, "grid_cell_window", lambda _cell: (0, 4, 0, 4))
    monkeypatch.setattr(
        label_sw, "_lookup_area_tile_and_split",
        lambda _out_dir, _cell, _tile: ("T31UCU", "train"))
    monkeypatch.setattr(
        label_sw, "_prepare_annotation_chips",
        lambda *_args, **_kwargs: {
            "Composite": np.zeros((4, 4, 3), dtype=np.uint8)})
    path = os.path.join(str(tmp_path), "area-001.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"tile": "T31UCU"}, handle)
    before = open(path, encoding="utf-8").read()

    result = label_sw._annotate_grid_area(
        str(tmp_path), scenes, 1, "2026-04")

    assert result == 2
    assert open(path, encoding="utf-8").read() == before
