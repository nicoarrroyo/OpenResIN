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
    """Eight training and four test cells, none touching across splits."""
    train_ids = [100, 102, 104, 106, 300, 302, 304, 306]
    test_ids = [200, 202, 400, 402]
    assignment = sw.validate_areas(train_ids, test_ids)
    assert assignment == {"train": train_ids, "test": test_ids}


def test_validate_areas_rejects_wrong_counts():
    """The design fixes the split at eight training and four test areas."""
    with pytest.raises(ValueError):
        sw.validate_areas([100, 102, 104, 106, 300, 302, 304],
                          [200, 202, 400, 402])
    with pytest.raises(ValueError):
        sw.validate_areas([100, 102, 104, 106, 300, 302, 304, 306],
                          [200, 202, 400])


def test_validate_areas_rejects_duplicates_and_neighbours():
    """One cell cannot serve twice, and a grid boundary is not
    independence: neighbouring opposite-split cells are rejected."""
    with pytest.raises(ValueError):
        sw.validate_areas([100, 102, 104, 106, 300, 302, 304, 306],
                          [100, 202, 400, 402])
    with pytest.raises(ValueError):  # 101 neighbours 100, even diagonally
        sw.validate_areas([100, 102, 104, 106, 300, 302, 304, 306],
                          [101, 202, 400, 402])
    with pytest.raises(ValueError):  # 122 is diagonal to 100
        sw.validate_areas([100, 102, 104, 106, 300, 302, 304, 306],
                          [122, 202, 400, 402])


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


def test_colorise_ndwi_uses_diverging_water_palette_and_black_nodata():
    """Land is red, water blue, zero neutral, and NoData black."""
    ndwi = np.array([[-1.0, 0.0, 1.0, np.nan]], dtype=np.float32)

    rgb = sw.colorise_ndwi(ndwi)

    assert rgb.shape == (1, 4, 3)
    assert rgb.dtype == np.uint8
    assert rgb[0, 0, 0] > rgb[0, 0, 2]
    assert np.ptp(rgb[0, 1].astype(np.int16)) <= 1
    assert rgb[0, 2, 2] > rgb[0, 2, 0]
    assert np.array_equal(rgb[0, 3], [0, 0, 0])


def test_colorise_ndwi_resolves_weak_water_at_display_limits():
    """The tighter [-0.5, 0.5] range keeps faint coastal water visibly blue."""
    ndwi = np.array([[-0.5, 0.0, 0.13, 0.5, -0.9, 0.9]], dtype=np.float32)

    rgb = sw.colorise_ndwi(ndwi)

    assert rgb[0, 0, 0] > rgb[0, 0, 2]
    assert np.ptp(rgb[0, 1].astype(np.int16)) <= 1
    assert rgb[0, 2, 2] > rgb[0, 2, 0]
    assert rgb[0, 3, 2] > rgb[0, 3, 0]
    assert np.array_equal(rgb[0, 0], rgb[0, 4])
    assert np.array_equal(rgb[0, 3], rgb[0, 5])


def test_annotate_area_reuses_background_canvas_item(monkeypatch):
    """Enlarged view reuses one background and stores scrolled clicks in image pixels."""
    button_commands = {}
    canvas_bindings = {}
    root_bindings = {}
    canvas_kwargs = {}
    scrollbar_calls = []
    scroll_offset = 10
    screen_width, screen_height = 1920, 1080

    root = MagicMock()
    root.winfo_screenwidth.return_value = screen_width
    root.winfo_screenheight.return_value = screen_height
    canvas = MagicMock()
    canvas.create_image.return_value = 17
    canvas.canvasx.side_effect = lambda value: value + scroll_offset
    canvas.canvasy.side_effect = lambda value: value + scroll_offset
    canvas.bind.side_effect = lambda seq, func: canvas_bindings.__setitem__(
        seq, func)
    root.bind.side_effect = lambda seq, func: root_bindings.__setitem__(
        seq, func)

    def make_button(_parent, text, command):
        button_commands[text] = command
        return MagicMock()

    def make_canvas(*_args, **kwargs):
        canvas_kwargs.update(kwargs)
        return canvas

    def make_scrollbar(*args, **kwargs):
        scrollbar_calls.append((args, kwargs))
        return MagicMock()

    fake_tk = SimpleNamespace(
        Tk=lambda: root,
        Canvas=make_canvas,
        Frame=lambda *_args, **_kwargs: MagicMock(),
        Button=make_button,
        Label=lambda *_args, **_kwargs: MagicMock(),
        Scrollbar=make_scrollbar,
        TclError=Exception,
        LEFT="left",
        RIGHT="right",
        BOTTOM="bottom",
        X="x",
        Y="y",
        BOTH="both",
        HORIZONTAL="horizontal",
        VERTICAL="vertical",
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

    # Tiny 2x2 chips on 1920x1080 fit far above the cap, so scale is 4
    # and the scaled 8x8 canvas fits the viewport without scrolling.
    expected_scale = 4
    expected_scaled = 2 * expected_scale

    def run_session():
        button_commands["NDWI"]()
        wanted_image = [(4.0, 4.0), (6.0, 4.0), (4.0, 6.0)]
        for image_x, image_y in wanted_image:
            event_x = image_x * expected_scale - scroll_offset
            event_y = image_y * expected_scale - scroll_offset
            canvas_bindings["<ButtonPress-1>"](
                SimpleNamespace(x=event_x, y=event_y))
        button_commands["Close as water"]()

    root.mainloop.side_effect = run_session

    new_polygons, kept = sw.annotate_area(chips)
    assert canvas.create_image.call_count == 1
    canvas.itemconfig.assert_called_once_with(17, image=ANY)
    root.state.assert_called_with("zoomed")
    scrollregions = [
        call.kwargs.get("scrollregion")
        for call in canvas.config.call_args_list
        if "scrollregion" in call.kwargs
    ]
    assert (0, 0, expected_scaled, expected_scaled) in scrollregions
    assert canvas_kwargs["width"] == expected_scaled
    assert canvas_kwargs["height"] == expected_scaled
    assert len(scrollbar_calls) == 2
    assert new_polygons == [{
        "class": "water",
        "vertices": [[4.0, 4.0], [6.0, 4.0], [4.0, 6.0]],
    }]
    assert kept == []


def test_annotate_area_undo_last_polygon(monkeypatch):
    """Undo polygon drops the newest closed boundary and its outline."""
    button_commands = {}
    canvas_bindings = {}
    root_bindings = {}
    scroll_offset = 10
    expected_scale = 4

    root = MagicMock()
    root.winfo_screenwidth.return_value = 1920
    root.winfo_screenheight.return_value = 1080
    canvas = MagicMock()
    canvas.create_image.return_value = 17
    canvas.canvasx.side_effect = lambda value: value + scroll_offset
    canvas.canvasy.side_effect = lambda value: value + scroll_offset
    canvas.create_polygon.side_effect = [101, 102]
    canvas.bind.side_effect = lambda seq, func: canvas_bindings.__setitem__(
        seq, func)
    root.bind.side_effect = lambda seq, func: root_bindings.__setitem__(
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
        LEFT="left",
        RIGHT="right",
        BOTTOM="bottom",
        X="x",
        Y="y",
        BOTH="both",
        HORIZONTAL="horizontal",
        VERTICAL="vertical",
        SUNKEN="sunken",
        W="w",
    )
    from PIL import ImageTk
    monkeypatch.setattr(ImageTk, "PhotoImage", lambda _image: object())
    monkeypatch.setitem(sys.modules, "tkinter", fake_tk)

    def click(image_x, image_y):
        canvas_bindings["<ButtonPress-1>"](SimpleNamespace(
            x=image_x * expected_scale - scroll_offset,
            y=image_y * expected_scale - scroll_offset))

    def run_session():
        button_commands["Undo polygon"]()
        for point in [(4.0, 4.0), (6.0, 4.0), (4.0, 6.0)]:
            click(*point)
        button_commands["Close as water"]()
        for point in [(10.0, 10.0), (12.0, 10.0), (10.0, 12.0)]:
            click(*point)
        button_commands["Close as non-water"]()
        button_commands["Undo polygon"]()

    root.mainloop.side_effect = run_session
    chips = {
        "composite": np.zeros((2, 2, 3), dtype=np.uint8),
        "NDWI": np.ones((2, 2, 3), dtype=np.uint8),
    }

    new_polygons, kept = sw.annotate_area(chips)
    assert new_polygons == [{
        "class": "water",
        "vertices": [[4.0, 4.0], [6.0, 4.0], [4.0, 6.0]],
    }]
    assert kept == []
    canvas.delete.assert_any_call(102)


def test_annotate_area_undo_saved_polygon(monkeypatch):
    """Undo after reopen drops the session polygon first, then the saved one."""
    button_commands = {}
    canvas_bindings = {}
    root_bindings = {}
    scroll_offset = 10
    expected_scale = 4

    root = MagicMock()
    root.winfo_screenwidth.return_value = 1920
    root.winfo_screenheight.return_value = 1080
    canvas = MagicMock()
    canvas.create_image.return_value = 17
    canvas.canvasx.side_effect = lambda value: value + scroll_offset
    canvas.canvasy.side_effect = lambda value: value + scroll_offset
    canvas.create_polygon.side_effect = [301, 302]
    canvas.bind.side_effect = lambda seq, func: canvas_bindings.__setitem__(
        seq, func)
    root.bind.side_effect = lambda seq, func: root_bindings.__setitem__(
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
        LEFT="left",
        RIGHT="right",
        BOTTOM="bottom",
        X="x",
        Y="y",
        BOTH="both",
        HORIZONTAL="horizontal",
        VERTICAL="vertical",
        SUNKEN="sunken",
        W="w",
    )
    from PIL import ImageTk
    monkeypatch.setattr(ImageTk, "PhotoImage", lambda _image: object())
    monkeypatch.setitem(sys.modules, "tkinter", fake_tk)

    def run_session():
        for image_x, image_y in [(4.0, 4.0), (6.0, 4.0), (4.0, 6.0)]:
            canvas_bindings["<ButtonPress-1>"](SimpleNamespace(
                x=image_x * expected_scale - scroll_offset,
                y=image_y * expected_scale - scroll_offset))
        button_commands["Close as water"]()
        button_commands["Undo polygon"]()
        button_commands["Undo polygon"]()

    root.mainloop.side_effect = run_session
    chips = {
        "composite": np.zeros((2, 2, 3), dtype=np.uint8),
        "NDWI": np.ones((2, 2, 3), dtype=np.uint8),
    }
    existing = [{"class": "water",
                 "vertices": [[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]]}]

    new_polygons, kept = sw.annotate_area(chips, existing)
    assert new_polygons == []
    assert kept == []
    assert len(existing) == 1
    deleted = [call.args[0] for call in canvas.delete.call_args_list]
    assert deleted.index(302) < deleted.index(301)


def test_annotate_area_tab_toggles_composite_and_ndwi(monkeypatch):
    """Tab flips between the composite and NDWI chips, in order."""
    button_commands = {}
    root_bindings = {}

    root = MagicMock()
    root.winfo_screenwidth.return_value = 1920
    root.winfo_screenheight.return_value = 1080
    canvas = MagicMock()
    canvas.create_image.return_value = 17
    canvas.bind.side_effect = lambda *args: None
    root.bind.side_effect = lambda seq, func: root_bindings.__setitem__(
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
        LEFT="left",
        RIGHT="right",
        BOTTOM="bottom",
        X="x",
        Y="y",
        BOTH="both",
        HORIZONTAL="horizontal",
        VERTICAL="vertical",
        SUNKEN="sunken",
        W="w",
    )
    from PIL import ImageTk
    monkeypatch.setattr(
        ImageTk, "PhotoImage",
        lambda image: float(np.mean(np.asarray(image))))
    monkeypatch.setitem(sys.modules, "tkinter", fake_tk)

    def run_session():
        root_bindings["<Tab>"](SimpleNamespace())
        root_bindings["<Tab>"](SimpleNamespace())

    root.mainloop.side_effect = run_session
    chips = {
        "composite": np.zeros((2, 2, 3), dtype=np.uint8),
        "NDWI": np.ones((2, 2, 3), dtype=np.uint8),
    }

    sw.annotate_area(chips)
    shown = [call.kwargs["image"]
             for call in canvas.itemconfig.call_args_list]
    assert shown == [1.0, 0.0]


def test_annotate_area_auto_close_toggle(monkeypatch):
    """Auto-close defaults on; the button turns click-to-close off."""
    button_commands = {}
    canvas_bindings = {}
    root_bindings = {}
    scroll_offset = 10
    expected_scale = 4

    root = MagicMock()
    root.winfo_screenwidth.return_value = 1920
    root.winfo_screenheight.return_value = 1080
    canvas = MagicMock()
    canvas.create_image.return_value = 17
    canvas.canvasx.side_effect = lambda value: value + scroll_offset
    canvas.canvasy.side_effect = lambda value: value + scroll_offset
    canvas.bind.side_effect = lambda seq, func: canvas_bindings.__setitem__(
        seq, func)
    root.bind.side_effect = lambda seq, func: root_bindings.__setitem__(
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
        LEFT="left",
        RIGHT="right",
        BOTTOM="bottom",
        X="x",
        Y="y",
        BOTH="both",
        HORIZONTAL="horizontal",
        VERTICAL="vertical",
        SUNKEN="sunken",
        W="w",
    )
    from PIL import ImageTk
    monkeypatch.setattr(ImageTk, "PhotoImage", lambda _image: object())
    monkeypatch.setitem(sys.modules, "tkinter", fake_tk)

    def click(image_x, image_y):
        canvas_bindings["<ButtonPress-1>"](SimpleNamespace(
            x=image_x * expected_scale - scroll_offset,
            y=image_y * expected_scale - scroll_offset))

    def run_session():
        for point in [(4.0, 4.0), (6.0, 4.0), (4.0, 6.0)]:
            click(*point)
        click(5.0, 5.0)
        button_commands["Auto-close: on"]()
        for point in [(10.0, 10.0), (12.0, 10.0), (10.0, 12.0)]:
            click(*point)
        click(11.0, 11.0)
        button_commands["Close as non-water"]()

    root.mainloop.side_effect = run_session
    chips = {
        "composite": np.zeros((2, 2, 3), dtype=np.uint8),
        "NDWI": np.ones((2, 2, 3), dtype=np.uint8),
    }

    new_polygons, kept = sw.annotate_area(chips)
    assert new_polygons == [
        {"class": "water",
         "vertices": [[4.0, 4.0], [6.0, 4.0], [4.0, 6.0]]},
        {"class": "non-water",
         "vertices": [[10.0, 10.0], [12.0, 10.0], [10.0, 12.0],
                      [11.0, 11.0]]},
    ]
    assert kept == []


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
