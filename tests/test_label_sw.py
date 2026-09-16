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


def test_prepare_annotation_chips_adds_two_stage_ndwi_composite(monkeypatch):
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

    chips = label_sw._prepare_annotation_chips(scenes, (0, 2, 0, 2))

    assert list(chips) == ["composite", "NDWI", "20260427", "20260430"]
    assert np.all(chips["composite"] == 60)
    assert np.all(chips["20260427"] == 20)
    assert np.all(chips["20260430"] == 100)
    assert chips["NDWI"].shape == (2, 2, 3)
    assert chips["NDWI"].dtype == np.uint8
    assert np.all(chips["NDWI"] == [104, 170, 207])
