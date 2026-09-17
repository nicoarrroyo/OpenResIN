"""Run the V1 surface-water feature and polygon-labelling workflow."""

import argparse
import glob
import json
import os
import warnings

import numpy as np

from . import config as c
from . import labelling_sw as sw


def build_parser():
    parser = argparse.ArgumentParser(
        prog="openresin-label-sw",
        description=("Build monthly water/non-water features and draw "
                     "label polygons. Usual order: build the features, "
                     "inspect the overview, annotate the 12 chosen cells, "
                     "then freeze their training/test split."))

    parser.add_argument(
        "--month", default="2026-04",
        help="calendar month as YYYY-MM (default: %(default)s)")
    parser.add_argument(
        "--out-dir", default=os.path.join(c.OUTPUTS_DIR, "label-water"),
        help="directory for features, areas and polygons "
             "(default: %(default)s)")
    parser.add_argument(
        "--device", default=c.SW_OCM_DEVICE, choices=("cuda", "cpu"),
        help="OmniCloudMask inference device (default: %(default)s)")
    parser.add_argument(
        "--train-areas", type=int, nargs=8, default=None, metavar="CELL",
        help="eight training area IDs (1-484); requires --test-areas")
    parser.add_argument(
        "--test-areas", type=int, nargs=4, default=None, metavar="CELL",
        help="four test area IDs (1-484); cannot touch training cells, "
             "including diagonally")
    parser.add_argument(
        "--annotate", type=int, default=None, metavar="CELL",
        help="open one area, including any polygons already saved there")

    return parser


# %% Shared helpers
def _scene_month(scene_dir):
    return os.path.basename(scene_dir).split("_")[2][:6]


def _scene_date(scene_dir):
    return os.path.basename(scene_dir).split("_")[2][:8]


def _scene_tile(scene_dir):
    return os.path.basename(scene_dir).split("_")[5]


def _find_first_file(patterns):
    for pattern in patterns:
        matching_files = sorted(glob.glob(pattern, recursive=True))
        if matching_files:
            return matching_files[0]
    return None


def _print_step(number, title):
    print("----------")
    print(f"| STEP {number} | {title}")
    print("----------")


# %% 1. Build the navigation overview
def _save_grid_preview(composite, metadata_10m, metadata_60m, path):
    """Save the overview composite with its numbered grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    display_image = np.nan_to_num(
        np.transpose(composite, (1, 2, 0)), nan=255).astype(np.uint8)
    n_cells = c.SW_GRID_ROWS * c.SW_GRID_COLS

    figure, axis = plt.subplots(figsize=(10, 10))
    axis.imshow(display_image)
    for cell_id in range(1, n_cells + 1):
        row_start, row_end, col_start, col_end = sw.cell_to_display(
            metadata_10m, metadata_60m, cell_id)
        axis.add_patch(patches.Rectangle(
            (col_start, row_start),
            col_end - col_start,
            row_end - row_start,
            linewidth=0.4,
            edgecolor="yellow",
            facecolor="none"))
        axis.text(
            (col_start + col_end) / 2,
            (row_start + row_end) / 2,
            str(cell_id),
            color="yellow",
            fontsize=4,
            ha="center",
            va="center")

    axis.set_title("Masked TCI median with fixed numbered grid")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def _create_navigation_overview(out_dir, scenes, device, month):
    """Build the 60 m navigation preview unless its cache is current."""
    overview_dir = os.path.join(out_dir, "overview")
    provenance_path = os.path.join(overview_dir, "provenance.json")
    preview_path = os.path.join(overview_dir, "composite-grid.png")
    os.makedirs(overview_dir, exist_ok=True)

    _print_step(1, "navigation overview")

    current_provenance = sw.build_provenance(
        scenes, month=month, inference_device=device)
    if os.path.isfile(provenance_path) and os.path.isfile(preview_path):
        with open(provenance_path, encoding="utf-8") as handle:
            saved_provenance = json.load(handle)
        if saved_provenance == current_provenance:
            print("  overview matches inputs and settings; reusing it")
            return

    masked_tci_by_date = {}
    metadata_10m = None
    metadata_60m = None

    for scene_dir in scenes:
        date = _scene_date(scene_dir)
        scene = sw.read_scene_60m(scene_dir)
        cloud_mask = sw.predict_cloud_mask(
            scene["red"],
            scene["green"],
            scene["nir"],
            inference_device=device)
        masked_tci = sw.mask_invalid(
            scene["tci"],
            cloud_mask,
            (scene["red"], scene["green"], scene["nir"]))
        masked_tci_by_date.setdefault(date, []).append(masked_tci)
        metadata_10m = scene["meta10"]
        metadata_60m = scene["meta"]

    composite, _ = sw.composite_scenes(masked_tci_by_date)
    _save_grid_preview(composite, metadata_10m, metadata_60m, preview_path)

    with open(provenance_path, "w", encoding="utf-8") as handle:
        json.dump(current_provenance, handle, indent=2)
    print(f"  wrote {preview_path}")


# %% 2-5. Build the monthly classifier features
def _find_known_feature_masks():
    masks_dir = os.path.join(c.DATA_DIR, "masks")
    boundaries_path = _find_first_file([
        os.path.join(masks_dir, "boundaries", "*.shp"),
        os.path.join(masks_dir, "boundaries", "*.gpkg"),
        os.path.join(masks_dir, "boundaries", "*.geojson"),
    ])
    urban_path = _find_first_file([
        os.path.join(masks_dir, "urban-areas", "**", "*.tif"),
        os.path.join(masks_dir, "urban-areas", "**", "*.tiff"),
        os.path.join(masks_dir, "urban-areas", "**", "*.jp2"),
    ])
    return boundaries_path, urban_path


def _create_monthly_features(out_dir, scenes, device, month):
    """Build and save the six 10 m features for one month."""
    features_by_date = {}
    image_metadata = None

    for scene_dir in scenes:
        _print_step(2, "scene bands and cloud masking")

        scene = sw.read_scene_10m(scene_dir)
        cloud_mask = sw.predict_cloud_mask(
            scene["red"],
            scene["green"],
            scene["nir"],
            inference_device=device)
        masked_bands = sw.mask_scene_bands(scene, cloud_mask)

        _print_step(3, "water indices")
        print(f"  indices for {os.path.basename(scene_dir)[:22]}")
        indices = sw.scene_indices(masked_bands)

        scene_features = {
            "B02": masked_bands["blue"],
            "B03": masked_bands["green"],
            "B04": masked_bands["red"],
            "B08": masked_bands["nir"],
            **indices,
        }
        date = _scene_date(scene_dir)
        features_by_date.setdefault(date, []).append(scene_features)
        image_metadata = scene["meta"]
        del scene

    _print_step(4, "monthly median features")
    features, valid_count = sw.monthly_features(features_by_date)
    del features_by_date

    _print_step(5, "sea and urban masking")
    print("  rivers and known reservoirs stay: they are valid water")
    boundaries_path, urban_path = _find_known_feature_masks()
    sw.mask_known_features(
        features, image_metadata, boundaries_path, urban_path)

    os.makedirs(out_dir, exist_ok=True)
    features_path = os.path.join(out_dir, "features.npz")
    np.savez_compressed(
        features_path, valid_count=valid_count, **features)

    provenance = {
        "tile": _scene_tile(scenes[0]),
        "month": month,
        "source_scenes": sorted(os.path.basename(s) for s in scenes),
        "feature_order": list(c.SW_FEATURES),
        "aggregation": "valid median within date, then median across dates",
        "masks": {
            "cloud_shadow_classes": list(c.SW_CLOUD_SHADOW_CLASSES),
            "nodata_value": c.SW_NODATA_VALUE,
            "sea_source": boundaries_path,
            "urban_source": urban_path,
        },
        "crs": str(image_metadata["crs"]),
    }
    provenance_path = os.path.join(out_dir, "features-provenance.json")
    with open(provenance_path, "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)

    valid_fraction = 100 * float((valid_count > 0).mean())
    print(f"  monthly features valid on at least one date: "
          f"{valid_fraction:.2f}%")
    return image_metadata


# %% 6. Freeze the training and test areas
def _freeze_area_split(out_dir, tile, assignment):
    _print_step(6, "freezing areas")
    areas_path = os.path.join(out_dir, "areas.json")
    os.makedirs(out_dir, exist_ok=True)
    sw.freeze_areas(areas_path, tile, assignment)
    print(f"  froze {assignment} to {areas_path}")


def _lookup_area_tile_and_split(out_dir, area_id, default_tile):
    """Return the saved tile and split for one area, if assigned."""
    areas_path = os.path.join(out_dir, "areas.json")
    if not os.path.isfile(areas_path):
        return default_tile, "unassigned"

    saved_areas = sw.load_areas(areas_path)
    for area in saved_areas["areas"]:
        if area["id"] == area_id:
            return saved_areas["tile"], area["split"]
    return saved_areas["tile"], "unassigned"


# %% 7. Annotate one area
def _load_masked_ndwi_window(out_dir, month, tile, scenes, window):
    """Read the saved masked monthly NDWI for one grid window, if usable.

    Loads the archive without modifying it. Return None with a reason for
    missing, stale, or corrupt archives so callers can fall back to a regular
    NDWI window calculated on demand. Includes the reason why it was rejected for
    the annotation window to display.

    """
    row0, row1, col0, col1 = window
    features_path = os.path.join(out_dir, "features.npz")
    provenance_path = os.path.join(out_dir, "features-provenance.json")
    if not os.path.isfile(features_path):
        return None, f"absent: no {features_path}"
    if not os.path.isfile(provenance_path):
        return None, f"absent: no {provenance_path}"

    try:
        with open(provenance_path, encoding="utf-8") as handle:
            provenance = json.load(handle)
    except (OSError, ValueError) as exc:
        return None, f"unreadable provenance {provenance_path}: {exc}"

    masks = provenance.get("masks") or {}
    expected_sea, expected_urban = _find_known_feature_masks()
    checks = [
        ("month", provenance.get("month"), month),
        ("tile", provenance.get("tile"), tile),
        ("source_scenes", provenance.get("source_scenes"),
         sorted(os.path.basename(s) for s in scenes)),
        ("feature_order", provenance.get("feature_order"),
         list(c.SW_FEATURES)),
        ("aggregation", provenance.get("aggregation"),
         "valid median within date, then median across dates"),
        ("cloud_shadow_classes", masks.get("cloud_shadow_classes"),
         list(c.SW_CLOUD_SHADOW_CLASSES)),
        ("nodata_value", masks.get("nodata_value"), c.SW_NODATA_VALUE),
        ("sea_source", masks.get("sea_source"), expected_sea),
        ("urban_source", masks.get("urban_source"), expected_urban),
    ]
    for name, saved, current in checks:
        if saved != current:
            return None, (f"stale: provenance {name} is {saved!r}, "
                           f"expected {current!r}")
    if expected_sea is None or expected_urban is None:
        return None, ("stale: a mask source is missing on disk "
                      f"(sea={expected_sea!r}, urban={expected_urban!r})")

    try:
        with np.load(features_path) as archive:
            if "NDWI" not in archive:
                return None, f"corrupt: no NDWI array in {features_path}"
            full_tile = archive["NDWI"]
            expected_shape = (c.SW_TILE_PX, c.SW_TILE_PX)
            if full_tile.shape != expected_shape:
                return None, (f"wrong-shape: NDWI is {full_tile.shape}, "
                               f"expected {expected_shape}")
            cell = full_tile[row0:row1, col0:col1].astype(np.float32)
    except (OSError, ValueError, KeyError) as exc:
        return None, f"corrupt: cannot read NDWI from {features_path}: {exc}"
    return cell, None


def _raw_ndwi_composite(scenes, window):
    """Aggregate unmasked B03/B08 window reads into a monthly NDWI."""
    ndwi_by_date = {}
    for scene_dir in scenes:
        date = _scene_date(scene_dir)
        green = sw.read_band_window(scene_dir, "B03", window)
        nir = sw.read_band_window(scene_dir, "B08", window)
        ndwi_by_date.setdefault(date, []).append(
            sw.calculate_ndwi(green, nir))

    median_ndwi_by_date = {}
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="All-NaN slice encountered",
            category=RuntimeWarning)

        for date, date_chips in ndwi_by_date.items():
            if len(date_chips) == 1:
                median_ndwi_by_date[date] = date_chips[0]
            else:
                stacked_chips = np.stack(date_chips, axis=0)
                median_ndwi_by_date[date] = np.nanmedian(
                    stacked_chips, axis=0)

        all_ndwi_dates = np.stack(
            list(median_ndwi_by_date.values()), axis=0)
        return np.nanmedian(all_ndwi_dates, axis=0)


def _prepare_annotation_chips(scenes, window, out_dir, month, tile):
    """Prepare TCI chips and a labelled masked-first monthly NDWI."""
    chips_by_date = {}
    for scene_dir in scenes:
        date = _scene_date(scene_dir)
        chip = sw.read_tci_window(scene_dir, window).astype(np.float32)
        chips_by_date.setdefault(date, []).append(chip)

    median_chips_by_date = {}
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="All-NaN slice encountered",
            category=RuntimeWarning)

        for date, date_chips in chips_by_date.items():
            if len(date_chips) == 1:
                median_chips_by_date[date] = date_chips[0]
            else:
                stacked_chips = np.stack(date_chips, axis=0)
                median_chips_by_date[date] = np.nanmedian(
                    stacked_chips, axis=0)

        all_dates = np.stack(list(median_chips_by_date.values()), axis=0)
        composite = np.nanmedian(all_dates, axis=0)

    masked_ndwi, reason = _load_masked_ndwi_window(
        out_dir, month, tile, scenes, window)
    if masked_ndwi is not None:
        ndwi_composite = masked_ndwi
        ndwi_key = "NDWI (masked)"
    else:
        ndwi_composite = _raw_ndwi_composite(scenes, window)
        ndwi_key = "NDWI (raw)"
        print(f"  NDWI fallback ({reason}); this view lacks cloud, sea "
              "and urban masking, and the TCI chips are raw window reads")

    display_chips = {
        "composite": composite,
        ndwi_key: sw.colorise_ndwi(ndwi_composite),
        **median_chips_by_date,
    }
    for name, chip in display_chips.items():
        display_chips[name] = np.nan_to_num(
            chip, nan=0).astype(np.uint8)
    return display_chips


def _load_saved_polygons(path, row_offset, col_offset):
    """Load scene coordinates and convert them to area coordinates."""
    if not os.path.isfile(path):
        return []

    with open(path, encoding="utf-8") as handle:
        saved = json.load(handle)

    area_polygons = []
    for polygon in saved["polygons"]:
        area_vertices = []
        for scene_x, scene_y in polygon["vertices_scene"]:
            area_vertices.append([
                scene_x - col_offset,
                scene_y - row_offset,
            ])
        area_polygons.append({
            "class": polygon["class"],
            "vertices": area_vertices,
        })
    return area_polygons


def _move_polygons_to_scene(polygons, row_offset, col_offset):
    """Convert polygon vertices from area coordinates to scene coordinates."""
    scene_polygons = []
    for polygon in polygons:
        scene_vertices = []
        for area_x, area_y in polygon["vertices"]:
            scene_vertices.append([
                area_x + col_offset,
                area_y + row_offset,
            ])
        scene_polygons.append({
            "class": polygon["class"],
            "vertices_scene": scene_vertices,
        })
    return scene_polygons


def _annotate_grid_area(out_dir, scenes, area_id, month):
    """Open one grid area, then save kept plus newly drawn polygons."""
    window = sw.grid_cell_window(area_id)
    row_start, _, col_start, _ = window
    default_tile = _scene_tile(scenes[0])
    tile, split = _lookup_area_tile_and_split(
        out_dir, area_id, default_tile)

    display_chips = _prepare_annotation_chips(
        scenes, window, out_dir, month, tile)
    polygons_path = os.path.join(out_dir, f"area-{area_id:03d}.json")
    saved_polygons = _load_saved_polygons(
        polygons_path, row_start, col_start)
    if saved_polygons:
        print(f"  resuming with {len(saved_polygons)} saved polygons")

    _print_step(7, "annotating")
    print(f"  cell {area_id} ({split}); "
          "close the window or press Finish when done")
    new_polygons, kept_polygons = sw.annotate_area(
        display_chips, saved_polygons)
    removed = len(saved_polygons) - len(kept_polygons)
    if removed:
        print(f"  removed {removed} saved polygon(s) via Undo")

    all_area_polygons = kept_polygons + new_polygons
    scene_polygons = _move_polygons_to_scene(
        all_area_polygons, row_start, col_start)
    numbered_polygons = []
    for polygon_number, polygon in enumerate(scene_polygons, start=1):
        numbered_polygons.append({"id": polygon_number, **polygon})

    sw.save_polygons(
        polygons_path,
        tile,
        area_id,
        split,
        window,
        numbered_polygons)
    print(f"  saved {len(numbered_polygons)} polygons to {polygons_path}")
    return 0


# %% Command-line workflow
def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # Training and test areas form one assignment and must be supplied together.
    if (args.train_areas is None) != (args.test_areas is None):
        parser.error("--train-areas and --test-areas go together")

    # Validate before any expensive image processing.
    area_assignment = None
    if args.train_areas is not None:
        try:
            area_assignment = sw.validate_areas(
                args.train_areas, args.test_areas)
        except ValueError as exc:
            print(f"bad split: {exc}")
            return 2
        print(f"split validates: {area_assignment}")

    sat_images_dir = os.path.join(c.DATA_DIR, "sat-images")
    requested_month = args.month.replace("-", "")
    scenes = []
    for scene_dir in sw.discover_scenes(sat_images_dir):
        if _scene_month(scene_dir) == requested_month:
            scenes.append(scene_dir)

    if not scenes:
        print(f"no scenes for month {args.month} in {sat_images_dir}")
        return 1
    print(f"using {len(scenes)} scenes for {args.month}")
    tile = _scene_tile(scenes[0])

    # Mode 1: save the training/test assignment.
    if area_assignment is not None:
        _freeze_area_split(args.out_dir, tile, area_assignment)

    # Mode 2: build the overview and monthly features.
    if args.annotate is None and area_assignment is None:
        _create_navigation_overview(
            args.out_dir, scenes, args.device, args.month)
        _create_monthly_features(
            args.out_dir, scenes, args.device, args.month)
        print("inspect the overview preview, draw polygons with "
              "--annotate CELL, then freeze the split")
        return 0

    # Mode 3: draw or resume polygons in one area.
    if args.annotate is not None:
        return _annotate_grid_area(
            args.out_dir, scenes, args.annotate, args.month)

    print("pass --annotate CELL to draw polygons in one area")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
