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
        "composite", "NDWI (raw)", "20260427", "20260430"]
    assert np.all(chips["composite"] == 60)
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

    assert list(chips)[:2] == ["composite", "NDWI (masked)"]
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
