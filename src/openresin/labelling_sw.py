"""Prepare imagery and polygon labels for the V1 surface-water classifier.

The 60 m composite is only a navigation aid. Classifier features use the
separate 10 m workflow.
"""

import json
import os
import sys
import warnings

import numpy as np
import rasterio

from . import config as c
from . import image_handling as image_do


# %% 1. Discover scenes and build the 60 m overview
def discover_scenes(sat_images_dir):
    """Return sorted Sentinel-2 scene directories from one directory."""
    scenes = []
    for name in sorted(os.listdir(sat_images_dir)):
        path = os.path.join(sat_images_dir, name)
        if (name.endswith(".SAFE")
                and len(name.split("_")) == 7
                and os.path.isdir(path)):
            scenes.append(path)
    return scenes


def _granule_img_data(scene_dir, res):
    """Return one scene's IMG_DATA directory for the requested resolution."""
    granule = os.path.join(scene_dir, "GRANULE")
    subdirs = [d for d in os.listdir(granule)
               if os.path.isdir(os.path.join(granule, d))]
    if len(subdirs) != 1:
        raise FileNotFoundError(
            f"expected one granule in {granule}, found {len(subdirs)}")
    return os.path.join(granule, subdirs[0], "IMG_DATA", res)


def read_scene_60m(scene_dir):
    """Return red, green, nir, tci, meta, and meta10 for one 60 m scene."""
    img_60m = _granule_img_data(scene_dir, "R60m")
    tci_names = [f for f in sorted(os.listdir(img_60m))
                 if f.endswith("_TCI_60m.jp2")]
    if not tci_names:
        raise FileNotFoundError(f"no 60 m TCI in {img_60m}")
    prefix = tci_names[0].replace("_TCI_60m.jp2", "")

    def band_path(band):
        path = os.path.join(img_60m, f"{prefix}_{band}_60m.jp2")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing band file {path}")
        return path

    # B8A is the 60 m NIR; B08 exists only at 10 m.
    scene = {
        "red": image_do.image_to_array(band_path("B04")).astype(np.float32),
        "green": image_do.image_to_array(
            band_path("B03")).astype(np.float32),
        "nir": image_do.image_to_array(band_path("B8A")).astype(np.float32),
    }
    with rasterio.open(os.path.join(img_60m, f"{prefix}_TCI_60m.jp2")) as src:
        scene["tci"] = src.read()
        scene["meta"] = src.meta.copy()

    img_10m = _granule_img_data(scene_dir, "R10m")
    b04_names = [f for f in sorted(os.listdir(img_10m))
                 if f.endswith("_B04_10m.jp2")]
    if not b04_names:
        raise FileNotFoundError(f"no 10 m B04 in {img_10m}")
    with rasterio.open(os.path.join(img_10m, b04_names[0])) as src10:
        scene["meta10"] = src10.meta.copy()

    return scene


def predict_cloud_mask(red, green, nir, inference_device=None,
                       inference_dtype=None):
    """Run OmniCloudMask on explicit red, green, and NIR arrays.

    CUDA failure falls back to CPU. The 60 m caller uses the result only for
    navigation because 60 m is outside OmniCloudMask's documented range.
    """
    from omnicloudmask import predict_from_array

    if inference_device is None:
        inference_device = c.SW_OCM_DEVICE
    if inference_dtype is None:
        inference_dtype = c.SW_OCM_DTYPE
    stacked = np.stack((red, green, nir))
    try:
        return predict_from_array(
            stacked,
            patch_size=c.SW_OCM_PATCH_SIZE,
            patch_overlap=c.SW_OCM_PATCH_OVERLAP,
            batch_size=c.SW_OCM_BATCH_SIZE,
            inference_device=inference_device,
            inference_dtype=inference_dtype)[0]
    except Exception as exc:
        if inference_device == "cpu":
            raise
        print(f"WARNING: {inference_device} inference failed ({exc}); "
              "falling back to CPU (slower)")
        return predict_from_array(
            stacked,
            patch_size=c.SW_OCM_PATCH_SIZE,
            patch_overlap=c.SW_OCM_PATCH_OVERLAP,
            batch_size=c.SW_OCM_BATCH_SIZE,
            inference_device="cpu")[0]
    finally:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def mask_invalid(tci, cloud_mask, bands):
    """Set cloud, shadow, and all-band-zero TCI pixels to NaN."""
    masked = tci.astype(np.float32)
    unusable = np.isin(cloud_mask, c.SW_CLOUD_SHADOW_CLASSES)
    nodata = np.ones(cloud_mask.shape, dtype=bool)
    for band in bands:
        nodata &= (band == c.SW_NODATA_VALUE)
    masked[:, unusable | nodata] = np.nan
    return masked


def composite_scenes(masked_by_date):
    """Take the valid median within each date, then across dates."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="All-NaN slice encountered",
            category=RuntimeWarning)

        median_by_date = []
        for date_arrays in masked_by_date.values():
            stacked_date = np.stack(date_arrays, axis=0)
            date_median = np.nanmedian(stacked_date, axis=0)
            median_by_date.append(date_median)

        stacked_dates = np.stack(median_by_date, axis=0)
        composite = np.nanmedian(stacked_dates, axis=0)

    validity = np.isfinite(composite).all(axis=0)
    return composite, validity


# %% Grid definition and overview provenance
def grid_cell_window(cell_id):
    """Return one row-major grid cell as an exclusive 10 m pixel window."""
    n_cells = c.SW_GRID_ROWS * c.SW_GRID_COLS
    if not isinstance(cell_id, (int, np.integer)) \
            or not 1 <= cell_id <= n_cells:
        raise ValueError(f"cell_id must be an integer 1-{n_cells}, "
                         f"got {cell_id!r}")
    row, col = divmod(int(cell_id) - 1, c.SW_GRID_COLS)
    row_start = row * c.SW_CELL_PX
    col_start = col * c.SW_CELL_PX
    return (row_start, min(row_start + c.SW_CELL_PX, c.SW_TILE_PX),
            col_start, min(col_start + c.SW_CELL_PX, c.SW_TILE_PX))


def cell_to_display(meta10, meta60, cell_id):
    """Map a 10 m grid cell onto the 60 m overview using georeferencing."""
    row_start, row_end, col_start, col_end = grid_cell_window(cell_id)
    to_60m = ~meta60["transform"]
    x0, y0 = meta10["transform"] * (col_start, row_start)
    x1, y1 = meta10["transform"] * (col_end, row_end)
    disp_col_start, disp_row_start = to_60m * (x0, y0)
    disp_col_end, disp_row_end = to_60m * (x1, y1)
    return (disp_row_start, disp_row_end, disp_col_start, disp_col_end)


def build_provenance(scene_dirs, month, inference_device=None):
    """Describe the scenes and settings used to build an overview."""
    if inference_device is None:
        inference_device = c.SW_OCM_DEVICE
    scene_names = sorted(os.path.basename(d) for d in scene_dirs)
    tile = scene_names[0].split("_")[5] if scene_names else None
    n_cells = c.SW_GRID_ROWS * c.SW_GRID_COLS
    return {
        "tile": tile,
        "month": month,
        "source_scenes": scene_names,
        "grid": {"tile_size_px": c.SW_TILE_PX, "cell_px": c.SW_CELL_PX,
                 "rows": c.SW_GRID_ROWS, "cols": c.SW_GRID_COLS,
                 "ids": f"1-{n_cells} row-major"},
        "aggregation": "valid median within date, then median across dates",
        "mask": {"cloud_shadow_classes": list(c.SW_CLOUD_SHADOW_CLASSES),
                 "nodata_value": c.SW_NODATA_VALUE},
        "inference": {"patch_size": c.SW_OCM_PATCH_SIZE,
                      "patch_overlap": c.SW_OCM_PATCH_OVERLAP,
                      "batch_size": c.SW_OCM_BATCH_SIZE,
                      "device": inference_device,
                      "dtype": c.SW_OCM_DTYPE},
    }


# %% 2. Read and mask the 10 m classifier bands
def read_scene_10m(scene_dir):
    """Return blue, green, red, nir, tci, and meta for one 10 m scene."""
    img_10m = _granule_img_data(scene_dir, "R10m")
    tci_names = [f for f in sorted(os.listdir(img_10m))
                 if f.endswith("_TCI_10m.jp2")]
    if not tci_names:
        raise FileNotFoundError(f"no 10 m TCI in {img_10m}")
    prefix = tci_names[0].replace("_TCI_10m.jp2", "")

    def band_path(band):
        path = os.path.join(img_10m, f"{prefix}_{band}_10m.jp2")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing band file {path}")
        return path

    # B08 is the 10 m NIR; the 60 m reader uses B8A instead.
    scene = {
        "blue": image_do.image_to_array(
            band_path("B02")).astype(np.float32),
        "green": image_do.image_to_array(
            band_path("B03")).astype(np.float32),
        "red": image_do.image_to_array(
            band_path("B04")).astype(np.float32),
        "nir": image_do.image_to_array(
            band_path("B08")).astype(np.float32),
    }
    with rasterio.open(os.path.join(img_10m, f"{prefix}_TCI_10m.jp2")) as src:
        scene["tci"] = src.read()
        scene["meta"] = src.meta.copy()
    return scene


def mask_scene_bands(scene, cloud_mask):
    """Set cloud, shadow, and all-band-zero pixels to NaN in every band.

    Find all-band-zero pixels before writing any NaNs. Otherwise the first
    edited band prevents the remaining zero comparisons from matching.
    """
    names = ("blue", "green", "red", "nir")
    invalid = np.ones(cloud_mask.shape, dtype=bool)
    for key in names:
        invalid &= (scene[key] == c.SW_NODATA_VALUE)
    cloudy = np.isin(cloud_mask, c.SW_CLOUD_SHADOW_CLASSES)
    masked_bands = {}
    for key in names:
        band = scene[key].astype(np.float32)
        band[cloudy | invalid] = np.nan
        masked_bands[key] = band
    return masked_bands


# %% 3. Calculate spectral indices
def calculate_ndwi(green, nir):
    """Calculate float32 NDWI, preserving zero-sum pixels as NoData."""
    denominator = green + nir
    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = (green - nir) / denominator
    ndwi = ndwi.astype(np.float32)
    ndwi[denominator == 0] = np.nan
    return ndwi


def colorise_ndwi(ndwi, vmin=-0.5, vmax=0.5):
    """Map NDWI to red land, neutral white, blue water, and black NoData.

    Limits stay centred on zero and clip outside values; stored NDWI
    values are never altered, only their display colours.
    """
    import matplotlib

    norm_ndwi = np.clip((ndwi - vmin) / (vmax - vmin), 0.0, 1.0)
    rgba = matplotlib.colormaps["RdBu"](norm_ndwi)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    rgb[~np.isfinite(ndwi)] = 0
    return rgb


def scene_indices(scene):
    """Calculate NDWI and NDVI from one scene's masked bands."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (scene["nir"] - scene["red"]) / (scene["nir"] + scene["red"])
    return {"NDWI": calculate_ndwi(scene["green"], scene["nir"]),
            "NDVI": ndvi.astype(np.float32)}


# %% 4. Calculate monthly features
def monthly_features(features_by_date):
    """Take the valid median within each date, then across dates.

    Also return the number of valid dates at each pixel. One valid date is
    sufficient; zero valid dates means NoData.
    """
    feature_names = list(c.SW_FEATURES)
    interactive_output = sys.stdout.isatty()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="All-NaN slice encountered",
            category=RuntimeWarning)

        median_features_by_date = []
        for date, date_features in features_by_date.items():
            if len(date_features) == 1:
                median_features_by_date.append(date_features[0])
                continue

            date_median = {}
            for feature_name in feature_names:
                if interactive_output:
                    print(
                        f"\r\033[K  date median | {date} | {feature_name}",
                        end="", flush=True)
                acquisitions = []
                for acquisition in date_features:
                    acquisitions.append(acquisition[feature_name])
                stacked_acquisitions = np.stack(acquisitions, axis=0)
                date_median[feature_name] = np.nanmedian(
                    stacked_acquisitions, axis=0)
            median_features_by_date.append(date_median)

        monthly_median = {}
        for feature_name in feature_names:
            if interactive_output:
                print(
                    f"\r\033[K  monthly median | {feature_name}",
                    end="", flush=True)
            date_arrays = []
            for date_features in median_features_by_date:
                date_arrays.append(date_features[feature_name])
            stacked_dates = np.stack(date_arrays, axis=0)
            monthly_median[feature_name] = np.nanmedian(
                stacked_dates, axis=0)

        if interactive_output:
            print("\r\033[K  medians complete", flush=True)
        else:
            print("  medians complete")

    first_feature = feature_names[0]
    valid_count = np.zeros_like(monthly_median[first_feature], dtype=np.int32)
    for date_features in median_features_by_date:
        date_is_valid = np.isfinite(date_features[first_feature])
        valid_count += date_is_valid.astype(np.int32)

    return monthly_median, valid_count


# %% 5. Mask known non-water features
def mask_known_features(arrays, meta, boundaries_path, urban_path):
    """Mask sea and urban areas with NaN, leaving valid water features."""
    if boundaries_path is not None:
        for key in arrays:
            image_do.known_feature_mask(
                arrays[key], meta, boundaries_path, "sea", fill=np.nan)
    else:
        print("no sea-boundary file: skipping sea masking")
    if urban_path is not None:
        for key in arrays:
            image_do.mask_urban_areas(
                arrays[key], meta, urban_path, fill=np.nan)
    else:
        print("no urban-area file: skipping urban masking")
    return arrays


# %% 6. Validate and save the area split
def validate_areas(train_ids, test_ids):
    """Require four training and two test cells that do not touch, even diagonally."""
    n_cells = c.SW_GRID_ROWS * c.SW_GRID_COLS
    if len(train_ids) != 4 or len(test_ids) != 2:
        raise ValueError("need exactly 4 training and 2 test areas, got "
                         f"{len(train_ids)} and {len(test_ids)}")
    for cell_id in list(train_ids) + list(test_ids):
        if not isinstance(cell_id, (int, np.integer)) \
                or not 1 <= cell_id <= n_cells:
            raise ValueError(f"area id must be an integer 1-{n_cells}, "
                             f"got {cell_id!r}")
    if len(set(train_ids) | set(test_ids)) != 6:
        raise ValueError("training and test areas must be six distinct cells")

    def neighbours(cell_id):
        row, col = divmod(int(cell_id) - 1, c.SW_GRID_COLS)
        for drow in (-1, 0, 1):
            for dcol in (-1, 0, 1):
                other = (row + drow, col + dcol)
                if 0 <= other[0] < c.SW_GRID_ROWS \
                        and 0 <= other[1] < c.SW_GRID_COLS:
                    yield other[0] * c.SW_GRID_COLS + other[1] + 1

    clashes = []
    for train_id in train_ids:
        neighbouring_ids = set(neighbours(train_id))
        for test_id in test_ids:
            if test_id in neighbouring_ids:
                clashes.append((train_id, test_id))
    if clashes:
        raise ValueError(
            "training and test areas must not neighbour each other, "
            f"got adjacent pairs {clashes}")
    return {"train": [int(i) for i in train_ids],
            "test": [int(i) for i in test_ids]}


def freeze_areas(path, tile, assignment):
    """Save validated area IDs, splits, and 10 m windows."""

    record = {"tile": tile, "grid": {"cell_px": c.SW_CELL_PX,
                                     "rows": c.SW_GRID_ROWS,
                                     "cols": c.SW_GRID_COLS},
              "areas": []}
    for split in ("train", "test"):
        for cell_id in assignment[split]:
            row0, row1, col0, col1 = grid_cell_window(cell_id)
            record["areas"].append({"id": cell_id, "split": split,
                                    "window": [row0, row1, col0, col1]})
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    return record


def load_areas(path):
    """Read a saved area assignment."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_polygons(path, tile, area_id, split, window, polygons):
    """Save water/non-water polygons in 10 m scene coordinates."""

    for polygon in polygons:
        if polygon["class"] not in ("water", "non-water"):
            raise ValueError("polygon class must be 'water' or "
                             f"'non-water', got {polygon['class']!r}")
    record = {"tile": tile, "area_id": area_id, "split": split,
              "window": list(window), "polygons": polygons}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    return record


# %% 7. Read and annotate one area
def read_band_window(scene_dir, band, window):
    """Read one 10 m band area as a 2D float32 array."""
    from rasterio.windows import Window as RioWindow

    img_10m = _granule_img_data(scene_dir, "R10m")
    names = [f for f in sorted(os.listdir(img_10m))
             if f.endswith(f"_{band}_10m.jp2")]
    if not names:
        raise FileNotFoundError(f"no 10 m {band} in {img_10m}")
    row0, row1, col0, col1 = window
    rio_window = RioWindow(col0, row0, col1 - col0, row1 - row0)
    with rasterio.open(os.path.join(img_10m, names[0])) as src:
        return src.read(1, window=rio_window).astype(np.float32)


def read_tci_window(scene_dir, window):
    """Read one 10 m TCI area without loading the whole tile."""
    from rasterio.windows import Window as RioWindow

    img_10m = _granule_img_data(scene_dir, "R10m")
    names = [f for f in sorted(os.listdir(img_10m))
             if f.endswith("_TCI_10m.jp2")]
    if not names:
        raise FileNotFoundError(f"no 10 m TCI in {img_10m}")
    row0, row1, col0, col1 = window
    rio_window = RioWindow(col0, row0, col1 - col0, row1 - row0)
    with rasterio.open(os.path.join(img_10m, names[0])) as src:
        return np.transpose(src.read(window=rio_window), (1, 2, 0))


def annotate_area(chips, existing=None):
    """Draw polygons while switching between the composite and dated chips.

    Saved polygons are shown and can be removed with Undo, newest first.
    Return (new_polygons, kept_existing): polygons drawn this session and
    the loaded polygons left after any undo. Finish saves kept + new.
    """
    import tkinter as tk
    from PIL import Image, ImageTk

    chip_names = list(chips.keys())
    height, width = chips[chip_names[0]].shape[:2]

    # Display margins leave room for decorations, buttons, and labels.
    WIDTH_MARGIN = 120
    HEIGHT_MARGIN = 200

    root = tk.Tk()
    root.title("Draw water and non-water polygons")
    try:
        root.state("zoomed")
    except tk.TclError:
        pass

    # Auto-fit integer enlarge: sharp NEAREST pixels, at least 2x, scroll if needed.
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    fit_width = (screen_width - WIDTH_MARGIN) // width
    fit_height = (screen_height - HEIGHT_MARGIN) // height
    scale = min(fit_width, fit_height)
    scale = max(2, min(scale, 4))

    scaled_width = width * scale
    scaled_height = height * scale
    viewport_width = screen_width - WIDTH_MARGIN
    viewport_height = screen_height - HEIGHT_MARGIN
    canvas_width = min(scaled_width, viewport_width)
    canvas_height = min(scaled_height, viewport_height)
    closing_distance_display = 8 * scale

    canvas_frame = tk.Frame(root)
    canvas_frame.pack(expand=True, fill=tk.BOTH)
    canvas = tk.Canvas(canvas_frame, width=canvas_width, height=canvas_height)
    horizontal_scroll = tk.Scrollbar(
        canvas_frame, orient=tk.HORIZONTAL, command=canvas.xview)
    vertical_scroll = tk.Scrollbar(
        canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
    canvas.config(xscrollcommand=horizontal_scroll.set,
                  yscrollcommand=vertical_scroll.set)
    canvas.config(scrollregion=(0, 0, scaled_width, scaled_height))
    horizontal_scroll.pack(side=tk.BOTTOM, fill=tk.X)
    vertical_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
    photo_images = {}
    for chip_name in chip_names:
        enlarged = Image.fromarray(chips[chip_name]).resize(
            (scaled_width, scaled_height), Image.NEAREST)
        photo_images[chip_name] = ImageTk.PhotoImage(enlarged)
    image_item = canvas.create_image(
        0, 0, anchor="nw", image=photo_images[chip_names[0]])

    new_polygons = []
    polygon_outlines = []
    kept_existing = list(existing or [])
    kept_outlines = []
    auto_close_enabled = True
    current_chip_name = chip_names[0]
    current_vertices = []
    vertex_markers = []
    edge_lines = []
    preview_line = None
    status_label = None
    colors_by_class = {
        "water": "dodgerblue",
        "non-water": "darkorange",
    }

    def set_status(message):
        status_label.config(text=message)

    def flatten_vertices(vertices):
        coordinates = []
        for x, y in vertices:
            coordinates.extend((x * scale, y * scale))
        return coordinates

    def draw_polygon_outline(polygon_class, vertices):
        return canvas.create_polygon(
            flatten_vertices(vertices),
            outline=colors_by_class.get(polygon_class, "white"),
            width=2,
            fill="")

    for polygon in kept_existing:
        kept_outlines.append(draw_polygon_outline(
            polygon["class"], polygon["vertices"]))

    def redraw_preview(event=None):
        nonlocal preview_line
        if preview_line is not None:
            canvas.delete(preview_line)
            preview_line = None
        if current_vertices and event is not None:
            previous_x, previous_y = current_vertices[-1]
            cursor_x = canvas.canvasx(event.x)
            cursor_y = canvas.canvasy(event.y)
            preview_line = canvas.create_line(
                previous_x * scale,
                previous_y * scale,
                cursor_x,
                cursor_y,
                fill="yellow",
                dash=(4, 2))

    def on_click(event):
        display_x = canvas.canvasx(event.x)
        display_y = canvas.canvasy(event.y)
        if current_vertices and auto_close_enabled:
            first_x, first_y = current_vertices[0]
            near_first_vertex = (
                abs(display_x - first_x * scale)
                <= closing_distance_display
                and abs(display_y - first_y * scale)
                <= closing_distance_display
            )
            if near_first_vertex and len(current_vertices) >= 3:
                close_as("water")
                return

        image_x = display_x / scale
        image_y = display_y / scale
        current_vertices.append((float(image_x), float(image_y)))
        vertex_markers.append(canvas.create_oval(
            display_x - 2, display_y - 2, display_x + 2, display_y + 2,
            fill="yellow", outline=""))
        if len(current_vertices) > 1:
            previous_x, previous_y = current_vertices[-2]
            edge_lines.append(canvas.create_line(
                previous_x * scale,
                previous_y * scale,
                display_x,
                display_y,
                fill="yellow",
                width=2))
        set_status(f"{len(current_vertices)} vertices "
                    "(Close as water / non-water to close)")

    def clear_drawing():
        nonlocal preview_line
        for item in vertex_markers + edge_lines:
            canvas.delete(item)
        if preview_line is not None:
            canvas.delete(preview_line)
            preview_line = None
        del current_vertices[:]
        del vertex_markers[:]
        del edge_lines[:]

    def close_as(polygon_class):
        if len(current_vertices) < 3:
            set_status("need at least 3 vertices before closing")
            return

        vertices = []
        for x, y in current_vertices:
            vertices.append([x, y])
        new_polygons.append({
            "class": polygon_class,
            "vertices": vertices,
        })
        polygon_outlines.append(
            draw_polygon_outline(polygon_class, current_vertices))
        clear_drawing()

        water_count = 0
        for polygon in new_polygons:
            if polygon["class"] == "water":
                water_count += 1
        set_status(
            f"saved {polygon_class} polygon {len(new_polygons)} "
            f"({water_count} water)")

    def cancel_shape():
        clear_drawing()
        set_status("shape cancelled")

    def undo_point():
        if not current_vertices:
            return
        current_vertices.pop()
        canvas.delete(vertex_markers.pop())
        if edge_lines:
            canvas.delete(edge_lines.pop())
        set_status(f"{len(current_vertices)} vertices")

    def undo_last_polygon():
        if new_polygons:
            new_polygons.pop()
            canvas.delete(polygon_outlines.pop())
            water_count = 0
            for polygon in new_polygons:
                if polygon["class"] == "water":
                    water_count += 1
            set_status(
                f"removed last polygon ({len(new_polygons)} new left, "
                f"{water_count} water)")
            return
        if kept_existing:
            kept_existing.pop()
            canvas.delete(kept_outlines.pop())
            set_status(
                f"removed saved polygon ({len(kept_existing)} saved left); "
                "Finish will save the change")
            return
        set_status("nothing to undo")

    def switch_chip(chip_name):
        nonlocal current_chip_name
        current_chip_name = chip_name
        canvas.itemconfig(image_item, image=photo_images[chip_name])
        set_status(f"viewing {chip_name}; {len(new_polygons)} new polygons")

    def toggle_composite_ndwi():
        if len(chip_names) < 2:
            return
        if current_chip_name == chip_names[0]:
            switch_chip(chip_names[1])
        else:
            switch_chip(chip_names[0])

    def toggle_auto_close():
        nonlocal auto_close_enabled
        auto_close_enabled = not auto_close_enabled
        auto_close_button.config(
            text="Auto-close: on" if auto_close_enabled
            else "Auto-close: off")
        set_status("auto-close on" if auto_close_enabled else "auto-close off")

    def finish_labelling():
        root.destroy()

    canvas.bind("<ButtonPress-1>", on_click)
    canvas.bind("<Motion>", redraw_preview)
    root.bind("<Tab>", lambda _event: toggle_composite_ndwi())
    root.bind("<Escape>", lambda _event: cancel_shape())
    root.bind("<Shift_L>", lambda _event: toggle_auto_close())

    buttons = tk.Frame(root)
    buttons.pack(fill=tk.X, pady=6)
    for chip_name in chip_names:
        tk.Button(
            buttons,
            text=chip_name,
            command=lambda name=chip_name: switch_chip(name)).pack(
                side=tk.LEFT, padx=4)
    tk.Button(buttons, text="Close as water",
              command=lambda: close_as("water")).pack(side=tk.LEFT, padx=4)
    tk.Button(buttons, text="Close as non-water",
              command=lambda: close_as("non-water")).pack(
                  side=tk.LEFT, padx=4)
    tk.Button(buttons, text="Undo point", command=undo_point).pack(
        side=tk.LEFT, padx=4)
    tk.Button(buttons, text="Undo polygon", command=undo_last_polygon).pack(
        side=tk.LEFT, padx=4)
    auto_close_button = tk.Button(
        buttons, text="Auto-close: on", command=toggle_auto_close)
    auto_close_button.pack(side=tk.LEFT, padx=4)
    tk.Button(buttons, text="Finish", command=finish_labelling).pack(
        side=tk.LEFT, padx=4, expand=True, fill=tk.X)

    if "NDWI (masked)" in chip_names:
        source_note = ("NDWI is masked (cloud, sea, urban); "
                       "TCI composite and dated chips are raw window reads")
    elif "NDWI (raw)" in chip_names:
        source_note = ("NDWI is raw (no cloud, sea or urban masking); "
                       "TCI composite and dated chips are raw window reads")
    else:
        source_note = "TCI composite and dated chips are raw window reads"
    source_label = tk.Label(root, text=source_note, anchor=tk.W)
    source_label.pack(fill=tk.X, padx=2)

    status_label = tk.Label(root, text="", bd=1, relief=tk.SUNKEN, anchor=tk.W)
    status_label.pack(fill=tk.X, padx=2, pady=2)
    set_status("click polygon vertices; flip dates to check stability")
    root.protocol("WM_DELETE_WINDOW", finish_labelling)
    root.mainloop()
    return new_polygons, kept_existing
