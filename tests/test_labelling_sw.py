import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import rasterio

from openresin import labelling_sw as sw


def test_grid_cell_windows():
    """Row-major 500 px cells; the last row and column clip to the tile."""
    assert sw.grid_cell_window(1) == (0, 500, 0, 500)
    assert sw.grid_cell_window(22) == (0, 500, 10500, 10980)  # row end
    assert sw.grid_cell_window(23) == (500, 1000, 0, 500)  # second row
    assert sw.grid_cell_window(484) == (10500, 10980, 10500, 10980)


def test_grid_rejects_bad_ids():
    """IDs outside 1-484 mean nothing without the tile context."""
    for bad in (0, 485, -1, 1.5, "1", None):
        with pytest.raises(ValueError):
            sw.grid_cell_window(bad)


def _tci(value, shape=(2, 2)):
    return np.full((3,) + shape, value, dtype=np.float32)


def test_composite_takes_valid_medians_within_then_across_dates():
    """Median within each date, then across dates; NaN is ignored, and a
    pixel with no valid date stays NoData."""
    nan = np.nan
    cases = [
        # Skewed values tell a median from a mean (43.3 for these).
        ({"d1": [_tci(10.0), _tci(20.0), _tci(100.0)]}, 20.0),
        ({"d1": [_tci(10.0)], "d2": [_tci(20.0)], "d3": [_tci(100.0)]},
         20.0),
        # A masked date does not pull a valid one.
        ({"d1": [_tci(nan)], "d2": [_tci(80.0)]}, 80.0),
    ]
    for dated, expected in cases:
        out, valid = sw.composite_scenes(dated)
        assert np.allclose(out, expected)
        assert valid.all()

    out, valid = sw.composite_scenes({"d1": [_tci(nan)], "d2": [_tci(nan)]})
    assert np.isnan(out).all()
    assert not valid.any()


def test_mask_invalid_covers_cloud_shadow_and_nodata():
    """Classes 1-3 and all-bands-zero pixels become NaN; clear stays."""
    tci = np.arange(15, dtype=np.uint8).reshape(3, 1, 5)
    mask = np.array([[0, 1, 2, 3, 0]])
    bands = [np.array([[5, 5, 5, 5, 0]], dtype=np.float32),
             np.array([[5, 5, 5, 5, 0]], dtype=np.float32)]
    out = sw.mask_invalid(tci, mask, bands)
    assert out[0, 0, 0] == 0.0  # clear, non-zero bands: kept
    assert np.isnan(out[:, 0, 1:4]).all()  # thick cloud, thin cloud, shadow
    assert np.isnan(out[:, 0, 4]).all()  # clear but all bands zero: NoData


def test_discover_scenes_rejects_files(tmp_path):
    """A scene-named regular file is not imagery and must be skipped."""
    good = tmp_path / "S2B_MSIL2A_20260430T110619_N0512_R137_T31UCU_X.SAFE"
    good.mkdir()
    (tmp_path / "S2C_MSIL2A_20260425T110621_N0512_R137_T31UCU_Y.SAFE").touch()
    (tmp_path / "notes.txt").touch()
    assert sw.discover_scenes(str(tmp_path)) == [str(good)]


def test_provenance_names_sources_and_settings():
    """The composite is uninterpretable without its provenance record."""
    from openresin import config as c
    prov = sw.build_provenance(
        ["x/S2B_MSIL2A_20260430T110619_N0512_R137_T31UCU_X.SAFE"],
        month="2026-04", inference_device="cuda")
    assert prov["tile"] == "T31UCU"
    assert prov["month"] == "2026-04"
    assert len(prov["source_scenes"]) == 1
    assert prov["inference"]["patch_size"] == c.SW_OCM_PATCH_SIZE
    assert prov["grid"]["ids"] == "1-484 row-major"


def test_validate_areas_needs_eight_four_and_separated_splits():
    """Eight training and four test cells, none touching across splits.

    A grid boundary is not independence: a test cell that shares an edge or
    a corner with a training cell is rejected, and so is a duplicate.
    """
    train_ids = [100, 102, 104, 106, 300, 302, 304, 306]
    test_ids = [200, 202, 400, 402]
    assert sw.validate_areas(train_ids, test_ids) == {
        "train": train_ids, "test": test_ids}

    bad_splits = [
        (train_ids[:-1], test_ids),  # seven training areas
        (train_ids, test_ids[:-1]),  # three test areas
        (train_ids, [101, 202, 400, 402]),  # 101 is east of 100
        (train_ids, [123, 202, 400, 402]),  # 123 is diagonal to 100
    ]
    for train, test in bad_splits:
        with pytest.raises(ValueError):
            sw.validate_areas(train, test)


def test_freeze_and_load_areas_roundtrip(tmp_path):
    """Frozen areas persist IDs, splits and 10 m windows."""
    path = str(tmp_path / "areas.json")
    record = sw.freeze_areas(
        path, "T31UCU",
        {"train": [1, 2, 3, 4, 5, 6, 7, 8],
         "test": [100, 102, 104, 106]})
    assert sw.load_areas(path) == record
    first = record["areas"][0]
    assert (first["id"], first["split"]) == (1, "train")
    assert first["window"] == [0, 500, 0, 500]


def _feature_dicts(value):
    from openresin import config as c
    return {name: np.full((2, 2), value, dtype=np.float32)
            for name in c.SW_FEATURES}


def test_monthly_features_median_and_count():
    """Within-date medians feed the across-date median; counts track valid dates."""
    nan_dicts = _feature_dicts(np.nan)
    dated = {"2026-04-25": [nan_dicts],
             "2026-04-27": [_feature_dicts(10.0), _feature_dicts(20.0)],
             "2026-04-30": [_feature_dicts(100.0)]}
    features, valid_count = sw.monthly_features(dated)
    assert np.allclose(features["B04"], 57.5)  # median of 15 and 100
    assert np.allclose(features["NDWI"], 57.5)
    assert (valid_count == 2).all()

    # Skewed values distinguish median from mean: within-date medians
    # are 2, 4 and 50, so the monthly median is 4 (their mean is 18.67).
    def _single(value):
        from openresin import config as c
        return {name: np.full((1, 1), value, dtype=np.float32)
                for name in c.SW_FEATURES}

    skewed = {"2026-04-25": [_single(1.0), _single(2.0), _single(100.0)],
              "2026-04-27": [_single(4.0)],
              "2026-04-30": [_single(50.0)]}
    skewed_features, skewed_count = sw.monthly_features(skewed)
    assert np.allclose(skewed_features["B04"], 4.0)
    assert np.allclose(skewed_features["NDWI"], 4.0)
    assert (skewed_count == 3).all()


def test_cached_medians_are_exact_and_bound_memory(tmp_path, monkeypatch):
    """Cached blocks match independent whole-array medians, including NaNs."""
    from openresin import label_sw
    rng = np.random.default_rng(17)
    dated, cached, expected = {}, {}, {}
    for date, count in (("25", 1), ("27", 2), ("30", 1)):
        dated[date], cached[date] = [], []
        for index in range(count):
            layers = {name: rng.normal(size=(17, 19)).astype(np.float32)
                      for name in sw.c.SW_FEATURES}
            for values in layers.values():
                values[rng.random(values.shape) < 0.2] = np.nan
                values[0, 0] = np.nan
            folder = tmp_path / f"{date}-{index}"
            folder.mkdir()
            dated[date].append(layers)
            cached[date].append(label_sw._save_feature_arrays(layers, folder))
    with np.errstate(invalid="ignore"), pytest.warns(RuntimeWarning):
        for name in sw.c.SW_FEATURES:
            daily = [np.nanmedian(np.stack([scene[name] for scene in scenes]),
                                  axis=0) for scenes in dated.values()]
            expected[name] = np.nanmedian(np.stack(daily), axis=0)
            if name == "B02":
                expected_count = np.isfinite(daily).sum(axis=0)
    monkeypatch.setattr(sw.c, "SW_CELL_PX", 5)
    original_load, original_stack, maps = np.load, np.stack, []

    def load(*args, **kwargs):
        assert sum(not array._mmap.closed for array in maps) < 4
        array = original_load(*args, **kwargs)
        maps.append(array)
        return array

    def stack(arrays, *args, **kwargs):
        assert all(max(array.shape) <= 5 for array in arrays)
        return original_stack(arrays, *args, **kwargs)

    monkeypatch.setattr(sw.np, "load", load)
    monkeypatch.setattr(sw.np, "stack", stack)
    actual, count = sw.monthly_features(cached)
    assert all(array._mmap.closed for array in maps)
    for name in sw.c.SW_FEATURES:
        assert actual[name].dtype == np.float32
        np.testing.assert_array_equal(actual[name], expected[name])
    assert count.dtype == np.int32
    np.testing.assert_array_equal(count, expected_count)


def test_monthly_medians_reject_mismatched_shapes():
    dated = {"25": [{"B02": np.ones((2, 2), np.float32)}],
             "30": [{"B02": np.ones((3, 2), np.float32)}]}
    with pytest.raises(ValueError, match="share a 2D grid"):
        sw.monthly_features(dated)


def test_indices_values_and_nodata():
    """NDWI and NDVI follow their formulas; NaN and zero-sum stay NoData."""
    scene = {"green": np.array([[100.0, np.nan]]),
             "nir": np.array([[50.0, np.nan]]),
             "red": np.array([[25.0, 25.0]])}
    indices = sw.scene_indices(scene)
    assert np.isclose(indices["NDWI"][0, 0], (100 - 50) / (100 + 50))
    assert np.isclose(indices["NDVI"][0, 0], (50 - 25) / (50 + 25))
    assert np.isnan(indices["NDWI"][0, 1])
    assert np.isnan(indices["NDVI"][0, 1])

    # 5 and -5 sum to zero with a non-zero difference: without the
    # zero-sum guard this pixel would be infinite, not NoData.
    ndwi = sw.calculate_ndwi(np.array([[5.0]], np.float32),
                             np.array([[-5.0]], np.float32))
    assert ndwi.dtype == np.float32
    assert np.isnan(ndwi[0, 0])


def _write_spectral_scene(scene_dir, green_dn=4000, swir_dn=2000):
    """Write aligned tiny rasters and metadata with real BOA offsets."""
    transform = rasterio.Affine(10, 0, 300000, 0, -10, 5900040)
    img_data = scene_dir / "GRANULE" / "fixture" / "IMG_DATA"
    for band in ("B03", "B11", "B12", "B05", "B06", "B07", "B8A"):
        resolution = 10 if band == "B03" else 20
        size = 8 if resolution == 10 else 4
        folder = img_data / f"R{resolution}m"
        folder.mkdir(parents=True, exist_ok=True)
        value = green_dn if band == "B03" else swir_dn
        if band == "B12":
            value = 500  # negative reflectance must survive
        with rasterio.open(
                folder / f"fixture_{band}_{resolution}m.jp2", "w",
                driver="GTiff", height=size, width=size, count=1,
                dtype="uint16", crs="EPSG:32631",
                transform=transform * rasterio.Affine.scale(resolution / 10)
                ) as destination:
            destination.write(np.full((size, size), value, np.uint16), 1)
    offsets = "".join(
        f'<BOA_ADD_OFFSET band_id="{band_id}">-1000</BOA_ADD_OFFSET>'
        for band_id in (2, 4, 5, 6, 8, 11, 12))
    (scene_dir / "MTD_MSIL2A.xml").write_text(
        '<metadata xmlns="urn:fixture"><BOA_QUANTIFICATION_VALUE>'
        f'10000</BOA_QUANTIFICATION_VALUE>{offsets}</metadata>')
    return {key: np.full((8, 8), green_dn, np.float32)
            for key in ("blue", "green", "red", "nir")}


def test_added_features_use_offsets_and_mask_native_cloud_footprints(
        tmp_path, monkeypatch):
    scene_dir = tmp_path / "scene.SAFE"
    scene = _write_spectral_scene(scene_dir)
    monkeypatch.setattr(sw.c, "SW_CELL_PX", 4)
    cloud = np.zeros((8, 8), np.uint8)
    cloud[2, 2] = 1  # whole parent footprint must be excluded
    scene["green"][0, 0] = 0  # destination NoData
    band_path = (scene_dir / "GRANULE/fixture/IMG_DATA/R20m/"
                 "fixture_B11_20m.jp2")
    with rasterio.open(band_path, "r+") as destination:
        values = destination.read(1)
        values[0, 3] = 0  # native NoData must be masked before conversion
        destination.write(values, 1)

    features, metadata = sw.added_scene_features(scene_dir, scene, cloud)

    assert metadata["offsets"]["B03"] == -1000
    assert metadata["metadata_sha256"] == sw.file_sha256(
        scene_dir / "MTD_MSIL2A.xml")
    assert features["MNDWI"][7, 7] == pytest.approx(0.5)
    assert features["B11"][7, 7] == pytest.approx(0.1)
    assert features["B12"][7, 7] == pytest.approx(-0.05)
    for values in features.values():
        assert values.dtype == np.float32
        assert np.isnan(values[0, 0])
        assert np.isnan(values[2:4, 2:4]).all()
    assert np.isnan(features["B11"][0:2, 6:8]).all()

    # Chunk edges must not change the interpolation result.
    monkeypatch.setattr(sw.c, "SW_CELL_PX", 8)
    full_features, _ = sw.added_scene_features(scene_dir, scene, cloud)
    for name in features:
        np.testing.assert_array_equal(features[name], full_features[name])


def test_resampling_rejects_misaligned_native_grid(tmp_path):
    scene_dir = tmp_path / "scene.SAFE"
    _write_spectral_scene(scene_dir)
    img_data = scene_dir / "GRANULE/fixture/IMG_DATA"
    with rasterio.open(img_data / "R20m/fixture_B11_20m.jp2", "r+") as band:
        band.transform = rasterio.Affine(20, 0, 300010, 0, -20, 5900040)
    with rasterio.open(img_data / "R20m/fixture_B11_20m.jp2") as band, \
            rasterio.open(img_data / "R10m/fixture_B03_10m.jp2") as reference:
        with pytest.raises(ValueError, match="does not align"):
            sw.resample_20m_band(
                band, reference, (0, 8, 0, 8), np.zeros((8, 8), np.uint8))


def test_monthly_builder_saves_all_features_with_date_balanced_indices(
        tmp_path, monkeypatch):
    from openresin import label_sw
    import weakref

    scenes = []
    scene_arrays = {}
    for date, suffix, green, swir in (
            ("20260427", "A", 4000, 2000),
            ("20260427", "B", 4000, 4000),
            ("20260430", "C", 4000, 7000)):
        scene_dir = tmp_path / (
            f"S2A_MSIL2A_{date}T110651_N0512_R137_T31UCU_{suffix}.SAFE")
        scene = _write_spectral_scene(scene_dir, green, swir)
        scene["meta"] = {"crs": "EPSG:32631"}
        # NoData in an old feature must mask the added monthly features.
        scene["blue"][0, 0] = np.nan
        scene_arrays[str(scene_dir)] = scene
        scenes.append(str(scene_dir))
    references = []
    save_arrays = label_sw._save_feature_arrays

    def save(features, folder):
        references.extend(weakref.ref(values) for values in features.values())
        return save_arrays(features, folder)

    def read(path):
        assert all(reference() is None for reference in references)
        return scene_arrays[path]

    monkeypatch.setattr(label_sw, "_save_feature_arrays", save)
    monkeypatch.setattr(sw, "read_scene_10m", read)
    monkeypatch.setattr(sw.c, "SW_CELL_PX", 4)
    monkeypatch.setattr(sw, "predict_cloud_mask",
                        lambda *args, **kwargs: np.zeros((8, 8), np.uint8))
    monkeypatch.setattr(label_sw, "_find_known_feature_masks",
                        lambda: (None, None))
    out_dir = tmp_path / "features"

    label_sw._create_monthly_features(str(out_dir), scenes, "cpu", "2026-04")

    assert all(reference() is None for reference in references)
    assert not list(out_dir.glob(".feature-build-*"))
    with np.load(out_dir / "features.npz") as archive:
        assert set(archive.files) == set(sw.c.SW_FEATURES) | {"valid_count"}
        # Daily MNDWI medians: (0.5 + 0)/2 and -1/3.
        assert archive["MNDWI"][7, 7] == pytest.approx((0.25 - 1 / 3) / 2)
        assert archive["B11"][7, 7] == pytest.approx(0.4)
        assert archive["B03"][7, 7] == 4000  # original DN kept
        assert archive["B12"][7, 7] == pytest.approx(-0.05)
        assert archive["valid_count"][7, 7] == 2
        for name in ("MNDWI", "B11", "B12", "B05", "B06", "B07", "B8A"):
            assert np.isnan(archive[name][0, 0])
    import json
    provenance = json.loads((out_dir / "features-provenance.json").read_text())
    assert provenance["feature_order"] == list(sw.c.SW_FEATURES)
    assert len(provenance["spectral_method"]["scene_metadata"]) == 3


def test_failed_feature_build_preserves_archive_and_cleans_cache(
        tmp_path, monkeypatch):
    from openresin import label_sw
    scene_dir = tmp_path / (
        "S2A_MSIL2A_20260427T110651_N0512_R137_T31UCU_A.SAFE")
    scene = _write_spectral_scene(scene_dir)
    scene["meta"] = {"crs": "EPSG:32631"}
    monkeypatch.setattr(sw, "read_scene_10m", lambda _path: scene)
    monkeypatch.setattr(sw, "predict_cloud_mask",
                        lambda *args, **kwargs: np.zeros((8, 8), np.uint8))
    monkeypatch.setattr(label_sw, "_find_known_feature_masks",
                        lambda: (None, None))

    def fail(*_args, **_kwargs):
        raise OSError("verification failure")

    monkeypatch.setattr(label_sw.np, "savez_compressed", fail)
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    (out_dir / "features.npz").write_bytes(b"earlier archive")
    with pytest.raises(OSError, match="verification"):
        label_sw._create_monthly_features(
            out_dir, [str(scene_dir)], "cpu", "2026-04")
    assert (out_dir / "features.npz").read_bytes() == b"earlier archive"
    assert not (out_dir / "features-provenance.json").exists()
    assert not list(out_dir.glob(".feature-build-*"))


def test_read_band_window_reads_requested_10m_pixels_as_float32(tmp_path):
    """Band quicklooks read only the requested source window."""
    image_dir = (
        tmp_path / "scene.SAFE" / "GRANULE" / "tile" / "IMG_DATA" / "R10m"
    )
    image_dir.mkdir(parents=True)
    band_path = image_dir / "tile_B03_10m.jp2"
    values = np.arange(20, dtype=np.uint16).reshape(4, 5)
    with rasterio.open(
            band_path, "w", driver="GTiff", height=4, width=5, count=1,
            dtype=values.dtype,
            transform=rasterio.Affine(10, 0, 0, 0, -10, 40)) as dst:
        dst.write(values, 1)

    band = sw.read_band_window(str(tmp_path / "scene.SAFE"), "B03",
                               (1, 3, 2, 5))

    assert band.dtype == np.float32
    assert np.array_equal(band, values[1:3, 2:5])


def test_colorise_ndwi_palette_limits_and_nodata():
    """Land is red, water blue, zero neutral and NoData black. Values clip
    at the [-0.5, 0.5] display limits, so faint water (0.13) is still blue."""
    ndwi = np.array([[-0.5, 0.0, 0.13, 0.5, -0.9, 0.9, np.nan]],
                    dtype=np.float32)

    rgb = sw.colorise_ndwi(ndwi)

    assert rgb.shape == (1, 7, 3)
    assert rgb.dtype == np.uint8
    assert rgb[0, 0, 0] > rgb[0, 0, 2]
    assert np.ptp(rgb[0, 1].astype(np.int16)) <= 1
    assert rgb[0, 2, 2] > rgb[0, 2, 0]
    assert rgb[0, 3, 2] > rgb[0, 3, 0]
    assert np.array_equal(rgb[0, 0], rgb[0, 4])
    assert np.array_equal(rgb[0, 3], rgb[0, 5])
    assert np.array_equal(rgb[0, 6], [0, 0, 0])


SCROLL_OFFSET = 10  # the fake canvas is scrolled by this many pixels
DISPLAY_SCALE = 4  # 2x2 chips on a 1920x1080 screen hit the scale cap


def _install_fake_tk(monkeypatch):
    """Replace tkinter with mocks and record the editor's callbacks.

    The tests drive the editor through its buttons and canvas clicks and
    check only what it returns, not how it draws.
    """
    button_commands = {}
    canvas_bindings = {}

    root = MagicMock()
    root.winfo_screenwidth.return_value = 1920
    root.winfo_screenheight.return_value = 1080
    canvas = MagicMock()
    canvas.canvasx.side_effect = lambda value: value + SCROLL_OFFSET
    canvas.canvasy.side_effect = lambda value: value + SCROLL_OFFSET
    canvas.bind.side_effect = lambda seq, func: canvas_bindings.__setitem__(
        seq, func)

    def make_button(_parent, text, command):
        button_commands[text] = command
        return MagicMock()

    fake_tk = SimpleNamespace(
        Tk=lambda: root,
        Canvas=lambda *_args, **_kwargs: canvas,
        Frame=lambda *_args, **_kwargs: MagicMock(),
        Button=make_button,
        Label=lambda *_args, **_kwargs: MagicMock(),
        Scrollbar=lambda *_args, **_kwargs: MagicMock(),
        TclError=Exception,
        LEFT="left", RIGHT="right", BOTTOM="bottom", X="x", Y="y",
        BOTH="both", HORIZONTAL="horizontal", VERTICAL="vertical",
        SUNKEN="sunken", W="w",
    )
    from PIL import ImageTk
    monkeypatch.setattr(ImageTk, "PhotoImage", lambda _image: object())
    monkeypatch.setitem(sys.modules, "tkinter", fake_tk)

    def click(image_x, image_y):
        """Click where the image pixel appears on the scaled, scrolled canvas."""
        canvas_bindings["<ButtonPress-1>"](SimpleNamespace(
            x=image_x * DISPLAY_SCALE - SCROLL_OFFSET,
            y=image_y * DISPLAY_SCALE - SCROLL_OFFSET))

    return SimpleNamespace(root=root, press=lambda text: button_commands[text](),
                           click=click)


def _tiny_chips():
    return {
        "composite": np.zeros((2, 2, 3), dtype=np.uint8),
        "NDWI": np.ones((2, 2, 3), dtype=np.uint8),
    }


def test_annotator_saves_clicked_shapes_in_image_pixels(monkeypatch):
    """Clicks on the scaled, scrolled view are saved as image-pixel vertices
    with the chosen class; exclusions are saved apart; Undo drops the newest
    polygon."""
    gui = _install_fake_tk(monkeypatch)

    def run_session():
        for point in [(4.0, 4.0), (6.0, 4.0), (4.0, 6.0)]:
            gui.click(*point)
        gui.click(5.0, 5.0)  # near the first vertex: auto-closes as water
        gui.press("Auto-close: on")  # now clicks near the start are vertices
        for point in [(10.0, 10.0), (12.0, 10.0), (10.0, 12.0), (11.0, 11.0)]:
            gui.click(*point)
        gui.press("Close as non-water")
        for point in [(20.0, 20.0), (22.0, 20.0), (20.0, 22.0)]:
            gui.click(*point)
        gui.press("Close as exclusion")
        for point in [(30.0, 30.0), (32.0, 30.0), (30.0, 32.0)]:
            gui.click(*point)
        gui.press("Close as water")
        gui.press("Undo polygon")

    gui.root.mainloop.side_effect = run_session

    (new_polygons, kept, new_exclusions, kept_exclusions,
     action) = sw.annotate_reviewed_area(_tiny_chips())

    assert new_polygons == [
        {"class": "water",
         "vertices": [[4.0, 4.0], [6.0, 4.0], [4.0, 6.0]]},
        {"class": "non-water",
         "vertices": [[10.0, 10.0], [12.0, 10.0], [10.0, 12.0],
                      [11.0, 11.0]]},
    ]
    assert new_exclusions == [
        {"vertices": [[20.0, 20.0], [22.0, 20.0], [20.0, 22.0]]}]
    assert kept == []
    assert kept_exclusions == []
    assert action == "finish"


@pytest.mark.parametrize("undo_presses, expected_kept", [
    (1, "saved"),  # the first Undo removes this session's polygon
    (2, "none"),  # the second removes the saved polygon
])
def test_annotator_undo_reaches_saved_polygons_last(
        monkeypatch, undo_presses, expected_kept):
    """Undo works newest first, and never edits the caller's list."""
    gui = _install_fake_tk(monkeypatch)
    saved = [{"class": "water",
              "vertices": [[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]]}]

    def run_session():
        for point in [(4.0, 4.0), (6.0, 4.0), (4.0, 6.0)]:
            gui.click(*point)
        gui.press("Close as water")
        for _ in range(undo_presses):
            gui.press("Undo polygon")

    gui.root.mainloop.side_effect = run_session

    new_polygons, kept, _, _, _ = sw.annotate_reviewed_area(
        _tiny_chips(), saved)

    assert new_polygons == []
    assert kept == (saved if expected_kept == "saved" else [])
    assert len(saved) == 1


def test_mask_scene_bands_masks_every_band():
    """Cloud and all-zero nodata pixels become NaN in all four bands.

    Regression test: the nodata test must run before any NaN is
    written, otherwise every band after the first keeps its zeros
    while the valid-date count claims NoData.
    """
    scene = {"blue": np.array([[0.0, 5.0, 7.0]]),
             "green": np.array([[0.0, 5.0, 7.0]]),
             "red": np.array([[0.0, 5.0, 7.0]]),
             "nir": np.array([[0.0, 5.0, 7.0]])}
    cloud_mask = np.array([[0, 1, 0]])  # middle pixel is cloud
    masked = sw.mask_scene_bands(scene, cloud_mask)
    for key in ("blue", "green", "red", "nir"):
        assert np.isnan(masked[key][0, 0])  # nodata wedge
        assert np.isnan(masked[key][0, 1])  # cloud
        assert masked[key][0, 2] == 7.0  # clear pixel untouched


def test_mask_known_features_skips_missing_files(capsys):
    """With no mask sources shipped, the masks skip loudly rather than
    crashing, so the run record stays honest."""
    arrays = {"B04": np.ones((2, 2), dtype=np.float32)}
    sw.mask_known_features(arrays, {}, None, None)
    out = capsys.readouterr().out
    assert "skipping sea masking" in out
    assert "skipping urban masking" in out
    assert (arrays["B04"] == 1).all()


def test_mask_known_features_masks_sea_and_urban(tmp_path):
    """Sea outside the land boundary and urban classes 20/21 become NaN."""
    import json

    from rasterio.transform import Affine

    transform = Affine(1, 0, 0, 0, -1, 4)
    meta = {"transform": transform, "width": 4, "height": 4,
            "crs": "EPSG:4326"}
    land = {"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [
            [[0, 0], [2, 0], [2, 4], [0, 4], [0, 0]]]}}]}
    boundaries_path = tmp_path / "land.geojson"
    boundaries_path.write_text(json.dumps(land), encoding="utf-8")
    urban_values = np.full((4, 4), 10, dtype=np.uint8)
    urban_values[0, 0] = 20
    urban_values[1, 1] = 21
    urban_path = tmp_path / "urban.tif"
    with rasterio.open(
            urban_path, "w", driver="GTiff", height=4, width=4, count=1,
            dtype="uint8", transform=transform, crs="EPSG:4326") as dst:
        dst.write(urban_values, 1)

    from openresin import config as c
    arrays = {name: np.full((4, 4), 5.0, dtype=np.float32)
              for name in c.SW_FEATURES}
    for name in c.SW_FEATURES:
        arrays[name][3, 0] = np.nan  # pre-existing gap stays NaN

    sw.mask_known_features(arrays, meta, str(boundaries_path), str(urban_path))

    for name in c.SW_FEATURES:
        assert np.isnan(arrays[name][:, 2:]).all()  # sea, right half
        assert np.isnan(arrays[name][0, 0])  # urban class 20 on land
        assert np.isnan(arrays[name][1, 1])  # urban class 21 on land
        assert arrays[name][2, 0] == 5.0  # valid land untouched
        assert arrays[name][0, 1] == 5.0  # valid land untouched
        assert np.isnan(arrays[name][3, 0])  # pre-existing NaN stays NaN


def _area_record_polygons():
    return [
        {"id": 1, "class": "water",
         "vertices_scene": [[0, 0], [2, 0], [2, 2], [0, 2]]},
        {"id": 2, "class": "non-water",
         "vertices_scene": [[2, 2], [4, 2], [4, 4], [2, 4]]},
    ]


def test_annotations_digest_tracks_geometry_not_ids():
    """Display ids do not change the revision; vertex moves, order swaps
    and exclusion edits do, so a completion must be reviewed again."""
    polygons = _area_record_polygons()
    exclusions = [{"vertices_scene": [[0, 3], [1, 3], [1, 4]]}]

    def digest(polygons, exclusions):
        return sw.annotations_digest(
            "T31UCU", 1, "train", [0, 4, 0, 4], polygons, exclusions)

    base = digest(polygons, exclusions)
    renumbered = [dict(polygon, id=polygon["id"] + 8) for polygon in polygons]
    moved = [dict(polygons[0], vertices_scene=[[0, 0], [2, 0], [2, 2],
                                               [0, 2.5]]), polygons[1]]
    assert digest(renumbered, exclusions) == base
    assert len(base) == 64
    assert digest(moved, exclusions) != base
    assert digest([polygons[1], polygons[0]], exclusions) != base
    assert digest(polygons, []) != base


def test_load_area_record_legacy_defaults(tmp_path):
    """Legacy files without exclusions or completion stay incomplete."""
    import json

    path = str(tmp_path / "area-001.json")
    legacy = {"tile": "T31UCU", "area_id": 1, "split": "train",
              "window": [0, 4, 0, 4],
              "polygons": _area_record_polygons()}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(legacy, handle)
    record = sw.load_area_record(path)
    assert record["exclusions"] == []
    assert "completion" not in record
    assert record["polygons"] == legacy["polygons"]


def test_load_area_record_rejects_malformed(tmp_path):
    """Bad classes, vertices, exclusions and completions fail loudly."""
    import json

    bad_cases = [
        {"tile": "T31UCU", "area_id": 1, "split": "train",
         "window": [0, 4, 0, 4],
         "polygons": [{"class": "cloud", "vertices_scene": [[0, 0]]}]},
        {"tile": "T31UCU", "area_id": 1, "split": "train",
         "window": [0, 4, 0, 4],
         "polygons": [{"class": "water", "vertices_scene": [[0, 0]]}]},
        {"tile": "T31UCU", "area_id": 1, "split": "train",
         "window": [0, 4, 0, 4], "polygons": [],
         "exclusions": [{"class": "water",
                         "vertices_scene": [[0, 0], [1, 0], [1, 1]]}]},
        {"tile": "T31UCU", "area_id": 1, "split": "train",
         "window": [0, 4, 0, 4], "polygons": [],
         "completion": {"schema_version": 1, "month": "April",
                        "features_sha256": "x" * 64,
                        "features_provenance_sha256": "y" * 64,
                        "annotations_sha256": "z" * 64}},
    ]
    for number, bad in enumerate(bad_cases):
        path = str(tmp_path / f"bad-{number}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(bad, handle)
        with pytest.raises(ValueError):
            sw.load_area_record(path)


def test_save_area_record_roundtrip_with_completion(tmp_path):
    """Exclusions and the completion decision survive an atomic save."""
    path = str(tmp_path / "area-001.json")
    polygons = _area_record_polygons()
    exclusions = [{"vertices_scene": [[0, 3], [1, 3], [1, 4]]}]
    annotations = sw.annotations_digest(
        "T31UCU", 1, "train", [0, 4, 0, 4], polygons, exclusions)
    record = {"tile": "T31UCU", "area_id": 1, "split": "train",
              "window": [0, 4, 0, 4], "polygons": polygons,
              "exclusions": exclusions,
              "completion": {"schema_version": 1, "month": "2026-04",
                             "features_sha256": "a" * 64,
                             "features_provenance_sha256": "b" * 64,
                             "annotations_sha256": annotations}}
    sw.save_area_record(path, record)
    assert sw.load_area_record(path) == record
    leftovers = [name for name in __import__("os").listdir(str(tmp_path))
                 if name.startswith(".area-")]
    assert leftovers == []


def test_save_area_record_cleans_temp_on_failure(tmp_path, monkeypatch):
    """A failed replace leaves the old file and no temporary file."""
    import os

    path = str(tmp_path / "area-001.json")
    first = {"tile": "T31UCU", "area_id": 1, "split": "train",
             "window": [0, 4, 0, 4], "polygons": []}
    sw.save_area_record(path, first)

    def fail_replace(_src, _dst):
        raise OSError("disk is full")

    monkeypatch.setattr(os, "replace", fail_replace)
    second = {"tile": "T31UCU", "area_id": 1, "split": "train",
              "window": [0, 4, 0, 4],
              "polygons": _area_record_polygons()}
    with pytest.raises(OSError):
        sw.save_area_record(path, second)
    assert sw.load_area_record(path)["polygons"] == []
    leftovers = [name for name in os.listdir(str(tmp_path))
                 if name.startswith(".area-")]
    assert leftovers == []


def test_rasterize_clips_to_window_and_keeps_fractional_edges():
    """Outside-window geometry is ignored; fractional edges are kept.

    A pixel is labelled when its centre lies inside the polygon. An edge at
    1.6 covers the centre at 1.5, which a rounded-down vertex would lose.
    """
    window = [0, 4, 0, 4]
    inside = [{"vertices_scene": [[0, 0], [2, 0], [2, 2], [0, 2]]}]
    outside = [{"vertices_scene": [[10, 10], [12, 10], [12, 12], [10, 12]]}]
    fractional = [{"vertices_scene": [[0, 0], [1.6, 0],
                                      [1.6, 1.6], [0, 1.6]]}]
    inside_mask = sw.rasterize_scene_polygons(inside, window)
    assert inside_mask.shape == (4, 4)
    assert inside_mask[:2, :2].all()
    assert not inside_mask[2:, 2:].any()
    assert not sw.rasterize_scene_polygons(outside, window).any()
    fractional_mask = sw.rasterize_scene_polygons(fractional, window)
    assert fractional_mask[:2, :2].all()
    assert fractional_mask.sum() == 4
    assert sw.rasterize_scene_polygons([], window).shape == (4, 4)
