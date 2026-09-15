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
