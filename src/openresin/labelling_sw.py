"""Monthly surface-water classification track (V1 random forest).

Replacement-track code for the water/non-water classifier agreed after the
3 September 2026 call (issue 04). The old patch/CNN path in labelling.py
stays untouched until the V1 is proven. It holds the 60 m navigation
overview, the monthly 10 m feature extraction, and polygon annotation
persistence for the water/non-water labels.

The 60 m overview is a navigation aid, not the classifier input and not
its validity mask. Direct inference at 60 m is outside OmniCloudMask's
documented 10-50 m range; the trial showed it tracks visible clouds, which
is all the overview needs.

Numbers live in config.py (SW_ settings); nothing here is tuned per run.
"""

import os
import warnings

import numpy as np
import rasterio

from . import config as c
from . import image_handling as image_do


def discover_scenes(sat_images_dir):
    """Sorted scene directories, skipping anything that is not a scene.

    Step 1 of openresin-label-sw (overview inputs).

    This stays separate from ui_do.list_folders because that helper is
    looser: unsorted, no directory check (a scene-named file passes),
    coupled to c.N_IMAGES, and it exits on multi-tile input.

    Parameters
    ----------
    sat_images_dir : str
        Directory holding extracted Sentinel-2 scenes.

    Returns
    -------
    list of str
        Full paths to directories whose names are seven
        underscore-separated fields ending in .SAFE. Scene-named regular
        files are rejected, not read.
    """
    scenes = []
    for name in sorted(os.listdir(sat_images_dir)):
        path = os.path.join(sat_images_dir, name)
        if (name.endswith(".SAFE")
                and len(name.split("_")) == 7
                and os.path.isdir(path)):
            scenes.append(path)
    return scenes


def _granule_img_data(scene_dir, res):
    """IMG_DATA directory for one resolution, following the single
    granule below GRANULE/. Step 1 of openresin-label-sw."""
    granule = os.path.join(scene_dir, "GRANULE")
    subdirs = [d for d in os.listdir(granule)
               if os.path.isdir(os.path.join(granule, d))]
    if len(subdirs) != 1:
        raise FileNotFoundError(
            f"expected one granule in {granule}, found {len(subdirs)}")
    return os.path.join(granule, subdirs[0], "IMG_DATA", res)


def read_scene_60m(scene_dir):
    """Read one scene's 60 m red/green/NIR bands, TCI and metadata.

    Step 1 of openresin-label-sw (overview inputs).

    Band values come through image_to_array, the same numerical reader
    the old path uses, so values keep their meaning. Raises
    FileNotFoundError if a band or TCI file is missing rather than
    silently continuing with a partial scene.

    Returns a plain dict with keys red, green, nir (float32 2D arrays),
    tci (uint8 (3, H, W) display image), meta (60 m TCI metadata) and
    meta10 (10 m B04 metadata used to map the fixed grid).
    """
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
    """OmniCloudMask clear/cloud/shadow mask for one scene at 60 m.

    Step 1 of openresin-label-sw (overview inputs).

    This stays separate from two_mask_clouds because that step assumes
    the old band order, masks arrays in place, and asks interactive
    questions on failure. Here the bands arrive explicitly and a failed
    device falls back to CPU with a printed warning, so the overview
    still runs on machines without CUDA.
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
    """TCI with cloud, shadow and nodata pixels set to NaN.

    Step 1 of openresin-label-sw (overview inputs).

    Parameters
    ----------
    tci : np.ndarray, uint8 (3, H, W)
    cloud_mask : np.ndarray, 2D, OmniCloudMask classes
    bands : sequence of 2D arrays
        The input bands; pixels that are zero in all of them are
        unimaged swath edge. OmniCloudMask labels these clear, so they
        need this explicit mask or zeros leak into the composite.

    Returns
    -------
    np.ndarray, float32 (3, H, W), NaN where unusable.
    """
    masked = tci.astype(np.float32)
    unusable = np.isin(cloud_mask, c.SW_CLOUD_SHADOW_CLASSES)
    nodata = np.ones(cloud_mask.shape, dtype=bool)
    for band in bands:
        nodata &= (band == c.SW_NODATA_VALUE)
    masked[:, unusable | nodata] = np.nan
    return masked


def composite_scenes(masked_by_date):
    """Median composite over dates from masked TCI arrays.

    Step 1 of openresin-label-sw (overview inputs).

    Parameters
    ----------
    masked_by_date : dict date -> list of (3, H, W) float32 arrays
        One list entry per acquisition; dates with two acquisitions
        (e.g. 27 April) hold two arrays.

    Returns
    -------
    composite : (3, H, W) float32, NaN where no date was valid
    validity : 2D bool array, True where at least one date was valid

    Within each date the median of valid acquisitions is taken (two
    values average, one passes through), then the median across dates.
    All-NaN locations yield NaN by construction; that is the explicit
    NoData path, not a warning to silence, so only that expected
    RuntimeWarning is filtered, narrowly.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="All-NaN slice encountered",
            category=RuntimeWarning)
        per_date = [np.nanmedian(np.stack(arrs, axis=0), axis=0)
                    for arrs in masked_by_date.values()]
        composite = np.nanmedian(np.stack(list(per_date), axis=0), axis=0)
    validity = np.isfinite(composite).all(axis=0)
    return composite, validity


def grid_cell_window(cell_id):
    """10 m pixel window for a numbered grid cell.

    Step 6 of openresin-label-sw (area freezing).

    IDs are stable row-major numbers 1-484 (row 1 is the north edge)
    and mean nothing without the tile.

    Returns
    -------
    (row_start, row_end, col_start, col_end) in 10 m pixels, where the
    end is exclusive. Full cells are 500 x 500; cells in the last row
    or column are clipped to the tile edge (480 px).
    """
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
    """Map a grid cell's 10 m window onto 60 m overview pixels.

    Step 1 of openresin-label-sw (overview preview).

    Cell bounds go through geographic coordinates, so a cell keeps its
    exact fractional overview position (500 10 m px = 83.33 60 m px)
    instead of accumulating rounding error from cell to cell.
    Returns (row_start, row_end, col_start, col_end) as floats.
    """
    row_start, row_end, col_start, col_end = grid_cell_window(cell_id)
    to_60m = ~meta60["transform"]
    x0, y0 = meta10["transform"] * (col_start, row_start)
    x1, y1 = meta10["transform"] * (col_end, row_end)
    disp_col_start, disp_row_start = to_60m * (x0, y0)
    disp_col_end, disp_row_end = to_60m * (x1, y1)
    return (disp_row_start, disp_row_end, disp_col_start, disp_col_end)


def build_provenance(scene_dirs, month, inference_device=None):
    """Record what went into an overview composite.

    Step 1 of openresin-label-sw (overview inputs).

    Returns a plain dict: tile, month, source scene folder names,
    grid identity, aggregation operators, mask and inference settings.
    Callers persist it next to the composite; nothing about the
    composite is interpretable without it.
    """
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


def read_scene_10m(scene_dir):
    """Read one scene's 10 m classifier bands, TCI and metadata.

    Step 2 of openresin-label-sw (feature inputs).

    Same contract as read_scene_60m: values through image_to_array,
    FileNotFoundError on a missing file. Returns a plain dict with
    keys blue, green, red, nir (float32 2D), tci (uint8 (3, H, W))
    and meta (10 m metadata).
    """
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
    """Cloud/shadow and nodata masking for the four 10 m bands.

    Step 2 of openresin-label-sw (feature inputs).

    Returns a new dict of float32 arrays with unusable pixels as NaN.
    The nodata test (zero in every band) must run on the unmasked
    bands: once NaN is written, == 0 stops matching, so masking one
    band after another would silently keep zeros in every band after
    the first while the valid-date count (from the first) claims
    NoData. That exact mismatch is what this single joint mask avoids.
    """
    names = ("blue", "green", "red", "nir")
    invalid = np.ones(cloud_mask.shape, dtype=bool)
    for key in names:
        invalid &= (scene[key] == c.SW_NODATA_VALUE)
    cloudy = np.isin(cloud_mask, c.SW_CLOUD_SHADOW_CLASSES)
    masked = {}
    for key in names:
        arr = scene[key].astype(np.float32)
        arr[cloudy | invalid] = np.nan
        masked[key] = arr
    return masked


def scene_indices(scene):
    """NDWI and NDVI from one scene's masked bands, float32.

    Step 3 of openresin-label-sw.

    Assumes cloud/shadow/nodata pixels are already NaN, so the indices
    inherit invalidity instead of computing with masked values.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = (scene["green"] - scene["nir"]) / (scene["green"] + scene["nir"])
        ndvi = (scene["nir"] - scene["red"]) / (scene["nir"] + scene["red"])
    return {"NDWI": ndwi.astype(np.float32),
            "NDVI": ndvi.astype(np.float32)}


def monthly_features(dated_features):
    """Monthly median of the six classifier features.

    Step 4 of openresin-label-sw.

    Parameters
    ----------
    dated_features : dict date -> list of dicts
        One dict per acquisition mapping each of c.SW_FEATURES to a
        masked 2D float32 array. Dates with two acquisitions hold two
        dicts; masked pixels are NaN.

    Returns
    -------
    features : dict name -> 2D float32, NaN where no date was valid
    valid_count : 2D int array, valid dates per pixel

    Same operator as the overview composite: median of valid
    same-day values, then median across dates. One valid date is
    sufficient for V1; zero means NoData.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="All-NaN slice encountered",
            category=RuntimeWarning)
        per_date = []
        for dicts in dated_features.values():
            per_date.append({name: np.nanmedian(
                np.stack([d[name] for d in dicts], axis=0), axis=0)
                for name in c.SW_FEATURES})
        names = list(c.SW_FEATURES)
        features = {name: np.nanmedian(
            np.stack([d[name] for d in per_date], axis=0), axis=0)
            for name in names}
    valid_count = np.zeros_like(features[names[0]], dtype=np.int32)
    for one in per_date:
        valid_count += np.isfinite(one[names[0]]).astype(np.int32)
    return features, valid_count


def mask_known_features(arrays, meta, boundaries_path, urban_path):
    """Sea/ocean and urban masking with NaN fill, in place.

    Step 5 of openresin-label-sw.

    Rivers and known reservoirs are deliberately NOT masked: they are
    valid water training examples under the V1 contract. A missing
    source file skips that mask with a message instead of failing,
    and the skip is recorded in the run provenance by the caller.
    """
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


def validate_areas(train_ids, test_ids):
    """Check six frozen windows: four training, two test.

    Step 6 of openresin-label-sw (area freezing).

    IDs must be distinct cells 1-484, and no training cell may touch a
    test cell (8-neighbourhood): a grid boundary does not establish
    independence, so adjacent opposite-split cells are rejected rather
    than warned about. Keeping a whole water body in one split stays
    the annotator's job; geometry cannot check it.
    """
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

    clashes = [(t, s) for t in train_ids for s in test_ids
               if s in set(neighbours(t))]
    if clashes:
        raise ValueError(
            "training and test areas must not neighbour each other, "
            f"got adjacent pairs {clashes}")
    return {"train": [int(i) for i in train_ids],
            "test": [int(i) for i in test_ids]}


def freeze_areas(path, tile, assignment):
    """Persist validated area IDs, splits and 10 m windows as JSON.

    Step 6 of openresin-label-sw (area freezing)."""
    import json

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
    """Read back a frozen areas file.

    Step 6 of openresin-label-sw (area freezing)."""
    import json

    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_polygons(path, tile, area_id, split, window, polygons):
    """Persist drawn polygons as the authoritative annotation.

    Step 7 of openresin-label-sw (annotation).

    polygons is a list of dicts with keys id (int, 1-based in draw
    order), class ("water" or "non-water") and vertices_scene (list of
    [x, y] in 10 m scene pixels). Pixels outside polygons stay
    unlabelled; no full-scene label mask is written.
    """
    import json

    for polygon in polygons:
        if polygon["class"] not in ("water", "non-water"):
            raise ValueError("polygon class must be 'water' or "
                             f"'non-water', got {polygon['class']!r}")
    record = {"tile": tile, "area_id": area_id, "split": split,
              "window": list(window), "polygons": polygons}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    return record


def read_tci_window(scene_dir, window):
    """One area's TCI chip from a 10 m scene without reading the tile.

    Step 7 of openresin-label-sw (annotation).

    window is (row_start, row_end, col_start, col_end) in scene pixels.
    Returns a uint8 (H, W, 3) display array.
    """
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
    """Draw water/non-water polygons on one area chip.

    Step 7 of openresin-label-sw (annotation).

    Drawing mechanics are ported from the parked prompt_roi tool; the
    coordinate mapping is fixed (that tool scaled both axes by the
    image height, which is wrong for non-square views).

    Parameters
    ----------
    chips : dict label -> uint8 (H, W, 3) display arrays, all one shape
        The composite view plus one entry per acquisition date. The
        annotator flips between them to check a polygon stays the same
        class on every usable date; polygons persist across flips.
    existing : list of polygon dicts, optional
        Previously saved polygons with chip-pixel vertices, overlaid
        read-only so work can resume. Same shape as save_polygons
        records but with "vertices" in chip pixels.

    Returns
    -------
    list of {"class": "water" | "non-water", "vertices": [[x, y], ...]}
        Newly drawn polygons in chip pixels as floats.
    """
    import tkinter as tk
    from PIL import Image, ImageTk

    labels = list(chips.keys())
    height, width = chips[labels[0]].shape[:2]
    CLOSE_RADIUS = 8

    root = tk.Tk()
    root.title("Draw water and non-water polygons")
    canvas = tk.Canvas(root, width=width, height=height)
    canvas.pack()
    photos = {key: ImageTk.PhotoImage(Image.fromarray(chips[key]))
              for key in labels}
    canvas.create_image(0, 0, anchor="nw", image=photos[labels[0]])

    drawn = []
    current_points = []
    vertex_markers = []
    edge_lines = []
    preview_line = None
    status_label = None

    def set_status(message):
        status_label.config(text=message)

    colors = {"water": "dodgerblue", "non-water": "darkorange"}
    for polygon in existing or []:
        flat = [coord for vertex in polygon["vertices"] for coord in vertex]
        canvas.create_polygon(
            flat, outline=colors.get(polygon["class"], "white"),
            width=2, fill="")

    def redraw_preview(event=None):
        nonlocal preview_line
        if preview_line is not None:
            canvas.delete(preview_line)
            preview_line = None
        if current_points and event is not None:
            x0, y0 = current_points[-1]
            preview_line = canvas.create_line(
                x0, y0, event.x, event.y, fill="yellow", dash=(4, 2))

    def on_click(event):
        if current_points:
            x0, y0 = current_points[0]
            if abs(event.x - x0) <= CLOSE_RADIUS \
                    and abs(event.y - y0) <= CLOSE_RADIUS \
                    and len(current_points) >= 3:
                close_as("water")
                return
        current_points.append((float(event.x), float(event.y)))
        vertex_markers.append(canvas.create_oval(
            event.x - 2, event.y - 2, event.x + 2, event.y + 2,
            fill="yellow", outline=""))
        if len(current_points) > 1:
            edge_lines.append(canvas.create_line(
                current_points[-2][0], current_points[-2][1],
                event.x, event.y, fill="yellow", width=2))
        set_status(f"{len(current_points)} vertices "
                   "(w: close as water, n: close as non-water)")

    def clear_drawing():
        nonlocal preview_line
        for item in vertex_markers + edge_lines:
            canvas.delete(item)
        if preview_line is not None:
            canvas.delete(preview_line)
            preview_line = None
        del current_points[:]
        del vertex_markers[:]
        del edge_lines[:]

    def close_as(cls):
        if len(current_points) < 3:
            set_status("need at least 3 vertices before closing")
            return
        drawn.append({"class": cls,
                      "vertices": [[x, y] for x, y in current_points]})
        flat = [coord for point in current_points for coord in point]
        canvas.create_polygon(flat, outline=colors[cls], width=2, fill="")
        clear_drawing()
        set_status(f"saved {cls} polygon {len(drawn)} "
                   f"({sum(p['class'] == 'water' for p in drawn)} water)")

    def cancel_shape():
        clear_drawing()
        set_status("shape cancelled")

    def undo_point():
        if not current_points:
            return
        current_points.pop()
        canvas.delete(vertex_markers.pop())
        if edge_lines:
            canvas.delete(edge_lines.pop())
        set_status(f"{len(current_points)} vertices")

    def switch_chip(key):
        canvas.create_image(0, 0, anchor="nw", image=photos[key])
        set_status(f"viewing {key}; {len(drawn)} new polygons")

    def finish():
        root.destroy()

    canvas.bind("<ButtonPress-1>", on_click)
    canvas.bind("<Motion>", redraw_preview)
    root.bind("w", lambda event: close_as("water"))
    root.bind("n", lambda event: close_as("non-water"))
    root.bind("<Escape>", lambda event: cancel_shape())
    root.bind("<BackSpace>", lambda event: undo_point())

    buttons = tk.Frame(root)
    buttons.pack(fill=tk.X, pady=6)
    for key in labels:
        tk.Button(buttons, text=key,
                  command=lambda key=key: switch_chip(key)).pack(
                      side=tk.LEFT, padx=4)
    tk.Button(buttons, text="Close as water",
              command=lambda: close_as("water")).pack(side=tk.LEFT, padx=4)
    tk.Button(buttons, text="Close as non-water",
              command=lambda: close_as("non-water")).pack(
                  side=tk.LEFT, padx=4)
    tk.Button(buttons, text="Undo point", command=undo_point).pack(
        side=tk.LEFT, padx=4)
    tk.Button(buttons, text="Finish", command=finish).pack(
        side=tk.LEFT, padx=4, expand=True, fill=tk.X)

    status_label = tk.Label(root, text="", bd=1, relief=tk.SUNKEN, anchor=tk.W)
    status_label.pack(fill=tk.X, padx=2, pady=2)
    set_status("click polygon vertices; flip dates to check stability")
    root.protocol("WM_DELETE_WINDOW", finish)
    root.mainloop()
    return drawn
