import io
import sys
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock

import numpy as np
import pytest
import rasterio

from openresin import labelling_sw as sw


def test_grid_first_cell_is_full():
    """Cell 1 is the north-west 500 x 500 px window."""
    assert sw.grid_cell_window(1) == (0, 500, 0, 500)


def test_grid_last_cell_is_clipped():
    """Cell 484 is the south-east corner, clipped to the tile edge."""
    assert sw.grid_cell_window(484) == (10500, 10980, 10500, 10980)


def test_grid_row_end_and_second_row():
    """Cell 22 ends the first row; cell 23 starts the second."""
    assert sw.grid_cell_window(22) == (0, 500, 10500, 10980)
    assert sw.grid_cell_window(23) == (500, 1000, 0, 500)


def test_grid_rejects_bad_ids():
    """IDs outside 1-484 mean nothing without the tile context."""
    for bad in (0, 485, -1, 1.5, "1", None):
        with pytest.raises(ValueError):
            sw.grid_cell_window(bad)


def _tci(value, shape=(2, 2)):
    return np.full((3,) + shape, value, dtype=np.float32)


def test_same_day_median_averages_two_values():
    """Two valid same-day acquisitions average, per the agreed operator."""
    out, valid = sw.composite_scenes({"2026-04-27": [_tci(10.0), _tci(20.0)]})
    assert np.allclose(out, 15.0)
    assert valid.all()


def test_single_acquisition_passes_through():
    """One valid date is sufficient for V1 and is used directly."""
    out, valid = sw.composite_scenes({"2026-04-30": [_tci(42.0)]})
    assert np.allclose(out, 42.0)
    assert valid.all()


def test_no_valid_date_is_nodata():
    """Zero valid dates means NoData, not zero: NaN out, invalid flag."""
    bad = _tci(np.nan)
    out, valid = sw.composite_scenes({"2026-04-25": [bad]})
    assert np.isnan(out).all()
    assert not valid.any()


def test_cloudy_date_does_not_pull_median():
    """A NaN (masked) date is ignored where another date is valid."""
    out, valid = sw.composite_scenes({
        "2026-04-25": [_tci(np.nan)],
        "2026-04-30": [_tci(100.0)],
    })
    assert np.allclose(out, 100.0)
    assert valid.all()


def test_mask_invalid_covers_cloud_shadow_and_nodata():
    """Classes 1-3 and all-bands-zero pixels become NaN; clear stays."""
    tci = np.arange(12, dtype=np.uint8).reshape(3, 2, 2)
    mask = np.array([[0, 1], [2, 3]])
    bands = [np.array([[5, 5], [0, 5]], dtype=np.float32),
             np.array([[5, 5], [0, 5]], dtype=np.float32)]
    out = sw.mask_invalid(tci, mask, bands)
    assert out[0, 0, 0] == 0.0  # clear, non-zero bands: kept
    assert np.isnan(out[:, 0, 1]).all()  # cloud
    assert np.isnan(out[:, 1, 0]).all()  # shadow + nodata zeros
    assert np.isnan(out[:, 1, 1]).all()  # shadow


def test_nodata_does_not_survive_composite():
    """The west-edge wedge: NaN (masked nodata) on one date does not
    darken a valid other date, and NaN on every date stays NoData."""
    valid = _tci(80.0)
    masked = np.full((3, 2, 2), np.nan, dtype=np.float32)
    out, v = sw.composite_scenes({"d1": [masked], "d2": [valid]})
    assert np.allclose(out, 80.0)
    assert v.all()
    out, v = sw.composite_scenes({"d1": [masked], "d2": [masked]})
    assert np.isnan(out).all()
    assert not v.any()


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


def test_validate_areas_accepts_separated_windows():
    """Four training and two test cells, none touching across splits."""
    assignment = sw.validate_areas([100, 102, 300, 302], [200, 400])
    assert assignment == {"train": [100, 102, 300, 302],
                          "test": [200, 400]}


def test_validate_areas_rejects_wrong_counts():
    """The design fixes the split at four training and two test areas."""
    with pytest.raises(ValueError):
        sw.validate_areas([100, 102, 300], [200, 400])
    with pytest.raises(ValueError):
        sw.validate_areas([100, 102, 300, 302], [200])


def test_validate_areas_rejects_duplicates_and_neighbours():
    """One cell cannot serve twice, and a grid boundary is not
    independence: neighbouring opposite-split cells are rejected."""
    with pytest.raises(ValueError):
        sw.validate_areas([100, 102, 300, 302], [100, 400])
    with pytest.raises(ValueError):  # 101 neighbours 100, even diagonally
        sw.validate_areas([100, 200, 300, 400], [101, 402])
    with pytest.raises(ValueError):  # 122 is diagonal to 100
        sw.validate_areas([100, 200, 300, 400], [122, 402])


def test_freeze_and_load_areas_roundtrip(tmp_path):
    """Frozen areas persist IDs, splits and 10 m windows."""
    path = str(tmp_path / "areas.json")
    record = sw.freeze_areas(
        path, "T31UCU", {"train": [1, 2, 3, 4], "test": [10, 11]})
    assert sw.load_areas(path) == record
    first = record["areas"][0]
    assert (first["id"], first["split"]) == (1, "train")
    assert first["window"] == [0, 500, 0, 500]


def test_save_polygons_rejects_unknown_class(tmp_path):
    """A polygon whose class is neither water nor non-water is a
    labelling error, caught at save time rather than at sampling."""
    with pytest.raises(ValueError):
        sw.save_polygons(str(tmp_path / "a.json"), "T31UCU", 1, "train",
                         [0, 500, 0, 500],
                         [{"id": 1, "class": "cloud",
                           "vertices_scene": [[0, 0]]}])


def test_save_polygons_roundtrip(tmp_path):
    """Saved polygons reload with scene coordinates intact."""
    import json

    path = str(tmp_path / "area-001.json")
    polygons = [{"id": 1, "class": "water",
                 "vertices_scene": [[10, 20], [30, 20], [30, 40]]}]
    sw.save_polygons(path, "T31UCU", 1, "train", [0, 500, 0, 500],
                     polygons)
    with open(path, encoding="utf-8") as handle:
        assert json.load(handle)["polygons"] == polygons


def _feature_dicts(value):
    from openresin import config as c
    return {name: np.full((2, 2), value, dtype=np.float32)
            for name in c.SW_FEATURES}


def test_monthly_features_median_and_count():
    """Two same-day values average, the across-date median follows,
    and the valid-date count tracks NaN dates."""
    nan_dicts = _feature_dicts(np.nan)
    dated = {"2026-04-25": [nan_dicts],
             "2026-04-27": [_feature_dicts(10.0), _feature_dicts(20.0)],
             "2026-04-30": [_feature_dicts(100.0)]}
    features, valid_count = sw.monthly_features(dated)
    assert np.allclose(features["B04"], 57.5)  # median of 15 and 100
    assert np.allclose(features["NDWI"], 57.5)
    assert (valid_count == 2).all()


def test_monthly_features_skips_single_acquisition_date_median(monkeypatch):
    """Only repeated dates reach the date-level median calculation."""
    real_nanmedian = np.nanmedian
    stack_sizes = []

    def record_stack_size(values, axis):
        stack_sizes.append(values.shape[0])
        return real_nanmedian(values, axis=axis)

    monkeypatch.setattr(sw.np, "nanmedian", record_stack_size)
    dated = {"2026-04-25": [_feature_dicts(10.0)],
             "2026-04-27": [_feature_dicts(20.0), _feature_dicts(30.0)],
             "2026-04-30": [_feature_dicts(40.0)]}

    sw.monthly_features(dated)

    feature_count = len(_feature_dicts(0.0))
    assert stack_sizes == [2] * feature_count + [3] * feature_count


def test_monthly_features_keeps_redirected_output_minimal(capsys):
    """Redirected output contains one durable completion message."""
    dated = {"2026-04-25": [_feature_dicts(10.0)],
             "2026-04-30": [_feature_dicts(20.0)]}

    sw.monthly_features(dated)

    assert capsys.readouterr().out == "  medians complete\n"


def test_monthly_features_rewrites_progress_in_interactive_terminal(
        monkeypatch):
    """Interactive feature progress occupies one line per median phase."""
    class InteractiveOutput(io.StringIO):
        def isatty(self):
            return True

    terminal = InteractiveOutput()
    monkeypatch.setattr("sys.stdout", terminal)
    dated = {"2026-04-25": [_feature_dicts(10.0)],
             "2026-04-27": [_feature_dicts(20.0), _feature_dicts(30.0)],
             "2026-04-30": [_feature_dicts(40.0)]}

    sw.monthly_features(dated)

    output = terminal.getvalue()
    assert "\r\033[K  date median | 2026-04-27 | NDVI" in output
    assert "date median | 2026-04-25" not in output
    assert "date median | 2026-04-30" not in output
    assert "\r\033[K  monthly median | NDVI" in output
    assert output.endswith("\r\033[K  medians complete\n")
    assert output.count("\n") == 1


def test_scene_indices_inherit_nodata():
    """Masked (NaN) bands yield NaN indices, not numbers."""
    scene = {"green": np.array([[100.0, np.nan]]),
             "nir": np.array([[50.0, np.nan]]),
             "red": np.array([[25.0, 25.0]])}
    indices = sw.scene_indices(scene)
    assert np.isclose(indices["NDWI"][0, 0], (100 - 50) / (100 + 50))
    assert np.isclose(indices["NDVI"][0, 0], (50 - 25) / (50 + 25))
    assert np.isnan(indices["NDWI"][0, 1])
    assert np.isnan(indices["NDVI"][0, 1])


def test_calculate_ndwi_handles_values_zero_sum_and_nodata():
    """NDWI is float32, while zero-sum and missing pixels stay NoData."""
    green = np.array([[75.0, 0.0, np.nan]], dtype=np.float32)
    nir = np.array([[25.0, 0.0, 10.0]], dtype=np.float32)

    ndwi = sw.calculate_ndwi(green, nir)

    assert ndwi.shape == (1, 3)
    assert ndwi.dtype == np.float32
    assert np.isclose(ndwi[0, 0], 0.5)
    assert np.isnan(ndwi[0, 1])
    assert np.isnan(ndwi[0, 2])


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


def test_colorize_ndwi_uses_diverging_water_palette_and_black_nodata():
    """Land is red, water blue, zero neutral, and NoData black."""
    ndwi = np.array([[-1.0, 0.0, 1.0, np.nan]], dtype=np.float32)

    rgb = sw.colorize_ndwi(ndwi)

    assert rgb.shape == (1, 4, 3)
    assert rgb.dtype == np.uint8
    assert rgb[0, 0, 0] > rgb[0, 0, 2]
    assert np.ptp(rgb[0, 1].astype(np.int16)) <= 1
    assert rgb[0, 2, 2] > rgb[0, 2, 0]
    assert np.array_equal(rgb[0, 3], [0, 0, 0])


def test_annotate_area_reuses_background_canvas_item(monkeypatch):
    """Switching chips updates one background without covering polygons."""
    button_commands = {}
    root = MagicMock()
    canvas = MagicMock()
    canvas.create_image.return_value = 17
    root.mainloop.side_effect = lambda: button_commands["NDWI"]()

    def make_button(_parent, text, command):
        button_commands[text] = command
        return MagicMock()

    fake_tk = SimpleNamespace(
        Tk=lambda: root,
        Canvas=lambda *_args, **_kwargs: canvas,
        Frame=lambda *_args, **_kwargs: MagicMock(),
        Button=make_button,
        Label=lambda *_args, **_kwargs: MagicMock(),
        LEFT="left",
        X="x",
        SUNKEN="sunken",
        W="w",
    )
    from PIL import ImageTk
    monkeypatch.setattr(ImageTk, "PhotoImage", lambda _image: object())
    monkeypatch.setitem(sys.modules, "tkinter", fake_tk)
    chips = {
        "composite": np.zeros((2, 2, 3), dtype=np.uint8),
        "NDWI": np.ones((2, 2, 3), dtype=np.uint8),
    }

    assert sw.annotate_area(chips) == []
    assert canvas.create_image.call_count == 1
    canvas.itemconfig.assert_called_once_with(17, image=ANY)


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
