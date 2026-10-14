import json
import os

import numpy as np
import pytest

from openresin import label_sw

from builders import area_record, feature_provenance


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


@pytest.mark.parametrize("extra_args, expected_code", [
    ([], 1),  # no matching scenes
    # A lone --train-areas is a usage error. Exit code 2 shows the parser
    # caught it before the empty discovery could return 1.
    (["--train-areas", "1", "2", "3", "4", "5", "6", "7", "8"], 2),
])
def test_main_fails_before_processing(tmp_path, monkeypatch, extra_args,
                                      expected_code):
    """No scenes, or unpaired split flags, fail before any processing."""
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("processing must not run after a failure")

    monkeypatch.setattr(label_sw.sw, "discover_scenes", lambda _dir: [])
    for name in ("_create_navigation_overview", "_create_monthly_features",
                 "_freeze_area_split", "_annotate_grid_area"):
        monkeypatch.setattr(label_sw, name, fail_if_called)

    try:
        result = label_sw.main([
            "--month", "2026-04",
            "--out-dir", str(tmp_path / "outputs"),
            *extra_args,
        ])
    except SystemExit as exc:  # argparse reports usage errors this way
        result = exc.code

    assert result == expected_code


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


def _write_features(out_dir, scenes, arrays, sea_source="sea.geojson",
                    urban_source="urban.tif", **provenance_overrides):
    """Save a synthetic features archive with consistent provenance."""
    np.savez_compressed(os.path.join(out_dir, "features.npz"), **arrays)
    provenance = feature_provenance(
        scenes, sea_source, urban_source, **provenance_overrides)
    with open(os.path.join(out_dir, "features-provenance.json"), "w",
              encoding="utf-8") as handle:
        json.dump(provenance, handle)


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
    _write_features(str(tmp_path), scenes, {"NDWI": ndwi})

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
    _write_features(str(tmp_path), scenes, {"NDWI": ndwi})

    cell, reason = label_sw._load_masked_ndwi_window(
        str(tmp_path), "2026-04", "T31UCU", scenes, (2, 4, 2, 4))

    assert reason is None
    assert np.array_equal(cell, ndwi[2:4, 2:4])


def _no_archive(_out_dir, _scenes, _monkeypatch):
    pass


def _corrupt_archive(out_dir, scenes, _monkeypatch):
    _write_features(out_dir, scenes, {"NDWI": np.zeros((4, 4), np.float32)})
    with open(os.path.join(out_dir, "features.npz"), "wb") as handle:
        handle.write(b"not a zip archive")


def _small_archive(out_dir, scenes, _monkeypatch):
    _write_features(out_dir, scenes, {"NDWI": np.zeros((3, 3), np.float32)})


def _masks_missing_on_disk(out_dir, scenes, monkeypatch):
    # The provenance agrees that no masks exist; the masked label is still
    # refused, because masking did not happen.
    monkeypatch.setattr(
        label_sw, "_find_known_feature_masks", lambda: (None, None))
    _write_features(out_dir, scenes, {"NDWI": np.zeros((4, 4), np.float32)},
                    sea_source=None, urban_source=None)


def _stale(**overrides):
    def write(out_dir, scenes, _monkeypatch):
        _write_features(out_dir, scenes,
                        {"NDWI": np.zeros((4, 4), np.float32)}, **overrides)
    return write


@pytest.mark.parametrize("write_archive, expected_reason", [
    (_no_archive, "absent:"),
    (_corrupt_archive, "corrupt:"),
    (_small_archive, "wrong-shape:"),
    (_masks_missing_on_disk, "stale: a mask source is missing"),
    (_stale(month="2026-05"), "stale: provenance month"),
    (_stale(source_scenes=["other.SAFE"]), "stale: provenance source_scenes"),
    (_stale(feature_order=["B02"]), "stale: provenance feature_order"),
    (_stale(masks={}), "stale: provenance cloud_shadow_classes"),
], ids=["absent", "corrupt", "wrong-shape", "masks-missing", "month",
        "source-scenes", "feature-order", "cloud-classes"])
def test_load_masked_ndwi_window_refuses_unusable_archive(
        tmp_path, monkeypatch, write_archive, expected_reason):
    """Each unusable archive gives no NDWI and a reason naming the cause."""
    scenes = _tiny_tile_setup(tmp_path, monkeypatch)
    write_archive(str(tmp_path), scenes, monkeypatch)

    cell, reason = label_sw._load_masked_ndwi_window(
        str(tmp_path), "2026-04", "T31UCU", scenes, (0, 2, 0, 2))

    assert cell is None
    assert reason.startswith(expected_reason)


def _tiny_features_setup(tmp_path, monkeypatch):
    """Fix a 4 px tile and save uniform valid features for fixtures."""
    scenes = _tiny_tile_setup(tmp_path, monkeypatch)
    _write_features(str(tmp_path), scenes, {
        name: np.ones((4, 4), dtype=np.float32)
        for name in label_sw.c.SW_FEATURES})
    return scenes


def test_derive_incomplete_uses_only_explicit_polygons(monkeypatch):
    """Incomplete areas leave other usable pixels unreviewed, not non-water."""
    monkeypatch.setattr(label_sw.c, "SW_TILE_PX", 4)
    window = [0, 4, 0, 4]
    features = {name: np.ones((4, 4), dtype=np.float32)
                for name in label_sw.c.SW_FEATURES}
    features["B02"][0, 0] = np.nan  # invalid even under the water polygon
    record = area_record(window, [
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


def test_derive_complete_turns_background_to_nonwater(monkeypatch):
    """A valid completion labels usable ground, but keeps exclusions back."""
    monkeypatch.setattr(label_sw.c, "SW_TILE_PX", 4)
    window = [0, 4, 0, 4]
    features = {name: np.ones((4, 4), dtype=np.float32)
                for name in label_sw.c.SW_FEATURES}
    record = area_record(window, [
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
    excluded = area_record(window, [water], exclusions=[
        {"vertices_scene": [[0, 0], [1, 0], [1, 1], [0, 1]]}])
    mask, _, _ = label_sw.derive_area_label_state(
        excluded, features, window, True)
    assert mask[0, 0] == label_sw.LABEL_WITHHELD
    assert mask[0, 1] == label_sw.LABEL_WATER

    clashing = area_record(window, [
        water,
        {"id": 2, "class": "non-water",
         "vertices_scene": [[0, 0], [2, 0], [2, 2], [0, 2]]},
    ])
    with pytest.raises(ValueError, match="overlap"):
        label_sw.derive_area_label_state(clashing, features, window, False)


def test_derive_nan_in_any_feature_is_unusable(monkeypatch):
    """A NaN in one feature makes the pixel unusable, even under a polygon.

    This is the case of a sea- or urban-masked pixel that still has a
    positive valid-date count: validity is all features finite.
    """
    monkeypatch.setattr(label_sw.c, "SW_TILE_PX", 4)
    window = [0, 4, 0, 4]
    features = {name: np.ones((4, 4), dtype=np.float32)
                for name in label_sw.c.SW_FEATURES}
    features["NDVI"][1, 1] = np.nan  # post-mask gap after compositing
    record = area_record(window, [
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
    record = area_record(window, polygons, [], {
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

    edited = area_record(window, [
        {"id": 1, "class": "water",
         "vertices_scene": [[0, 0], [3, 0], [3, 3], [0, 3]]},
    ], [], record["completion"])
    state, reason, active = label_sw.check_completion_active(
        edited, "2026-04", "T31UCU", 1, "train", window,
        features_digest, provenance_digest)
    assert state == "stale" and not active and "annotations" in reason

    assert label_sw.check_completion_active(
        area_record(window, polygons), "2026-04", "T31UCU", 1,
        "train", window, features_digest,
        provenance_digest)[0] == "incomplete"


def test_check_completion_available_needs_all_features(
        tmp_path, monkeypatch):
    """NDWI alone is not enough; every classifier feature must be readable."""
    scenes = _tiny_tile_setup(tmp_path, monkeypatch)
    _write_features(str(tmp_path), scenes,
                    {"NDWI": np.ones((4, 4), dtype=np.float32)})

    available, reason = label_sw.check_completion_available(
        str(tmp_path), "2026-04", "T31UCU", scenes, (0, 2, 0, 2))

    assert available is False
    assert reason.startswith("corrupt: no B02")


WATER = [{"id": 1, "class": "water",
          "vertices_scene": [[0, 0], [2, 0], [2, 2], [0, 2]]}]


def _annotate_setup(tmp_path, monkeypatch, completed=False):
    """Point the area workflow at a tiny out-dir with valid features.

    Saves area 001 with one water polygon, and with an active completion
    when completed is True.
    """
    scenes = _tiny_features_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        label_sw.sw, "grid_cell_window", lambda _cell: (0, 4, 0, 4))
    monkeypatch.setattr(
        label_sw, "_lookup_area_tile_and_split",
        lambda _out_dir, _cell, _tile: ("T31UCU", "train"))
    monkeypatch.setattr(
        label_sw, "_prepare_annotation_chips",
        lambda _scenes, _window, _out_dir, _month, _tile: {
            "Composite": np.zeros((4, 4, 3), dtype=np.uint8)})
    completion = None
    if completed:
        features_digest, provenance_digest, _ = \
            label_sw.compute_feature_digests(str(tmp_path))
        completion = {
            "schema_version": 1, "month": "2026-04",
            "features_sha256": features_digest,
            "features_provenance_sha256": provenance_digest,
            "annotations_sha256": label_sw.sw.annotations_digest(
                "T31UCU", 1, "train", [0, 4, 0, 4], WATER, [])}
    path = os.path.join(str(tmp_path), "area-001.json")
    label_sw.sw.save_area_record(
        path, area_record([0, 4, 0, 4], WATER, [], completion))
    return scenes, path


def _editor_returns(monkeypatch, polygons_scene, action):
    """Make the editor return these saved polygons and this button action.

    The editor reports polygons in area pixels; the window starts at the
    scene origin here, so they equal the scene coordinates.
    """
    kept = [{"class": polygon["class"],
             "vertices": [list(point) for point in polygon["vertices_scene"]]}
            for polygon in polygons_scene]
    monkeypatch.setattr(
        label_sw.sw, "annotate_reviewed_area",
        lambda *args, **kwargs: ([], kept, [], [], action))


EDITED = [{"id": 1, "class": "water",
           "vertices_scene": [[0, 0], [3, 0], [3, 3], [0, 3]]}]


@pytest.mark.parametrize("returned_polygons", [WATER, EDITED],
                         ids=["no-op", "edited"])
def test_annotate_grid_area_finish_keeps_completion_only_without_edits(
        tmp_path, monkeypatch, returned_polygons):
    """A no-op Finish leaves the file as it was; an edit drops the decision
    but keeps the new drawing."""
    scenes, path = _annotate_setup(tmp_path, monkeypatch, completed=True)
    before = open(path, encoding="utf-8").read()
    _editor_returns(monkeypatch, returned_polygons, "finish")

    result = label_sw._annotate_grid_area(
        str(tmp_path), scenes, 1, "2026-04")

    assert result == 0
    if returned_polygons is WATER:
        assert open(path, encoding="utf-8").read() == before
    else:
        saved = label_sw.sw.load_area_record(path)
        assert "completion" not in saved
        assert saved["polygons"] == EDITED


def test_annotate_grid_area_complete_and_reopen(tmp_path, monkeypatch):
    """Complete records digests; Reopen removes the decision only."""
    scenes, path = _annotate_setup(tmp_path, monkeypatch)
    _editor_returns(monkeypatch, WATER, "complete")

    assert label_sw._annotate_grid_area(
        str(tmp_path), scenes, 1, "2026-04") == 0
    completed = label_sw.sw.load_area_record(path)
    assert completed["completion"]["month"] == "2026-04"
    assert completed["polygons"] == WATER
    features_digest, provenance_digest, _ = \
        label_sw.compute_feature_digests(str(tmp_path))
    assert completed["completion"]["features_sha256"] == features_digest
    assert completed["completion"]["features_provenance_sha256"] == \
        provenance_digest

    _editor_returns(monkeypatch, WATER, "reopen")
    assert label_sw._annotate_grid_area(
        str(tmp_path), scenes, 1, "2026-04") == 0
    reopened = label_sw.sw.load_area_record(path)
    assert "completion" not in reopened
    assert reopened["polygons"] == WATER


def test_annotate_grid_area_rejects_malformed_record(tmp_path, monkeypatch):
    """A contradictory file is reported, never silently overwritten."""
    scenes, path = _annotate_setup(tmp_path, monkeypatch)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"tile": "T31UCU"}, handle)
    before = open(path, encoding="utf-8").read()

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("the editor must not open on a bad record")

    monkeypatch.setattr(label_sw.sw, "annotate_reviewed_area", fail_if_called)

    result = label_sw._annotate_grid_area(
        str(tmp_path), scenes, 1, "2026-04")

    assert result == 2
    assert open(path, encoding="utf-8").read() == before
