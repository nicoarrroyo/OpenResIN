"""Water/non-water labelling for the surface-water classifier (V1).
"""

import argparse
import glob
import json
import os

import numpy as np

from . import config as c
from . import labelling_sw as sw


def build_parser():
    parser = argparse.ArgumentParser(
        prog="openresin-label-sw",
        description=("Build monthly water/non-water features and draw "
                      "label polygons. Usual order: run once with no area "
                      "flags to build features, inspect the overview "
                      "preview, draw polygons with --annotate CELL in each "
                      "of the six chosen cells, then freeze the split. "
                      "Training and test areas must not neighbour each "
                      "other (not even diagonally): a grid boundary does "
                      "not establish independence, so adjacent "
                      "opposite-split cells are rejected."))

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
        "--train-areas", type=int, nargs=4, default=None, metavar="CELL",
        help="four training area IDs (1-484); needs --test-areas, and no "
             "training cell may neighbour a test cell, not even "
             "diagonally (default: %(default)s)")
    parser.add_argument(
        "--test-areas", type=int, nargs=2, default=None, metavar="CELL",
        help="two test area IDs (1-484); must not neighbour any training "
             "cell (default: %(default)s)")
    parser.add_argument(
        "--annotate", type=int, default=None, metavar="CELL",
        help="open the annotation window for this area ID; works on any "
             "cell without freezing first, and resumes saved polygons")

    return parser


def _month_stamp(scene_dir):
    return os.path.basename(scene_dir).split("_")[2][:6]


def _first_match(patterns):
    for pattern in patterns:
        found = sorted(glob.glob(pattern))
        if found:
            return found[0]
    return None


def _save_grid_png(composite, meta10, meta60, path):
    """Overview composite with the fixed numbered grid overlaid, the
    image the annotator picks areas from."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    shown = np.nan_to_num(np.transpose(composite, (1, 2, 0)),
                          nan=255).astype(np.uint8)
    n_cells = c.SW_GRID_ROWS * c.SW_GRID_COLS
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(shown)
    for cell_id in range(1, n_cells + 1):
        row0, row1, col0, col1 = sw.cell_to_display(meta10, meta60, cell_id)
        ax.add_patch(patches.Rectangle(
            (col0, row0), col1 - col0, row1 - row0,
            linewidth=0.4, edgecolor="yellow", facecolor="none"))
        ax.text((col0 + col1) / 2, (row0 + row1) / 2, str(cell_id),
                color="yellow", fontsize=4, ha="center", va="center")
    ax.set_title("Masked TCI median with fixed numbered grid")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def _overview(out_dir, scenes, device):
    """Step 1: 60 m navigation composite, rebuilt only when inputs or
    settings changed since the cached run. Only the preview image and
    the provenance are kept; the composite array itself is recomputed
    when needed and never stored."""
    cache_dir = os.path.join(out_dir, "overview")
    os.makedirs(cache_dir, exist_ok=True)
    provenance_path = os.path.join(cache_dir, "provenance.json")
    preview_path = os.path.join(cache_dir, "composite-grid.png")

    print("----------")
    print("| STEP 1 | navigation overview")
    print("----------")
    fresh = sw.build_provenance(scenes, "overview", device)
    if os.path.isfile(provenance_path) and os.path.isfile(preview_path):
        with open(provenance_path, encoding="utf-8") as handle:
            if json.load(handle) == fresh:
                print("  overview matches inputs and settings; reusing it")
                return

    masked_by_date = {}
    meta10 = meta60 = None
    for scene in scenes:
        date = os.path.basename(scene).split("_")[2][:8]
        one = sw.read_scene_60m(scene)
        mask = sw.predict_cloud_mask(
            one["red"], one["green"], one["nir"],
            inference_device=device)
        masked_by_date.setdefault(date, []).append(sw.mask_invalid(
            one["tci"], mask, (one["red"], one["green"], one["nir"])))
        meta10, meta60 = one["meta10"], one["meta"]

    composite, _ = sw.composite_scenes(masked_by_date)
    _save_grid_png(composite, meta10, meta60, preview_path)
    with open(provenance_path, "w", encoding="utf-8") as handle:
        json.dump(fresh, handle, indent=2)
    print(f"  wrote {preview_path}")


def _features(out_dir, scenes, device, month):
    """Steps 2-5: 10 m bands and cloud masking, indices, monthly median
    and sea/urban masking."""
    dated = {}
    meta = None
    for scene in scenes:
        date = os.path.basename(scene).split("_")[2][:8]
        print("----------")
        print("| STEP 2 | scene bands and cloud masking")
        print("----------")
        one = sw.read_scene_10m(scene)
        mask = sw.predict_cloud_mask(
            one["red"], one["green"], one["nir"],
            inference_device=device)
        bands = sw.mask_scene_bands(one, mask)
        print("----------")
        print("| STEP 3 | water indices")
        print("----------")
        print(f"  indices for {os.path.basename(scene)[:22]}")
        indices = sw.scene_indices(bands)
        dated.setdefault(date, []).append(
            {"B02": bands["blue"], "B03": bands["green"],
             "B04": bands["red"], "B08": bands["nir"], **indices})
        meta = one["meta"]
        del one

    print("----------")
    print("| STEP 4 | monthly median features")
    print("----------")
    features, valid_count = sw.monthly_features(dated)
    del dated

    print("----------")
    print("| STEP 5 | sea and urban masking")
    print("----------")
    print("  rivers and known reservoirs stay: they are valid water")
    masks_dir = os.path.join(c.DATA_DIR, "masks")
    boundaries = _first_match([
        os.path.join(masks_dir, "boundaries", "*.shp"),
        os.path.join(masks_dir, "boundaries", "*.gpkg"),
        os.path.join(masks_dir, "boundaries", "*.geojson")])
    urban = _first_match([
        os.path.join(masks_dir, "urban-areas", "*.tif"),
        os.path.join(masks_dir, "urban-areas", "*.tiff"),
        os.path.join(masks_dir, "urban-areas", "*.jp2")])
    sw.mask_known_features(features, meta, boundaries, urban)

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "features.npz"),
                        valid_count=valid_count, **features)
    provenance = {
        "tile": os.path.basename(scenes[0]).split("_")[5],
        "month": month,
        "source_scenes": sorted(os.path.basename(s) for s in scenes),
        "feature_order": list(c.SW_FEATURES),
        "aggregation": "valid median within date, then median across dates",
        "masks": {"cloud_shadow_classes": list(c.SW_CLOUD_SHADOW_CLASSES),
                  "nodata_value": c.SW_NODATA_VALUE,
                  "sea_source": boundaries,
                  "urban_source": urban},
        "crs": str(meta["crs"]),
    }
    with open(os.path.join(out_dir, "features-provenance.json"), "w",
              encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
    valid_frac = 100 * float((valid_count > 0).mean())
    print(f"  monthly features valid on at least one date: "
          f"{valid_frac:.2f}%")
    return meta


def _annotate(out_dir, scenes, area_id):
    """Draw polygons in one grid cell, resumable.

    Needs no frozen split: the cell window comes straight from the
    grid, and the split is looked up from areas.json when it exists
    ("unassigned" otherwise, refreshed on the next save after
    freezing)."""
    window = sw.grid_cell_window(area_id)  # also rejects bad IDs
    row0, _, col0, _ = window

    areas_path = os.path.join(out_dir, "areas.json")
    split = "unassigned"
    tile = os.path.basename(scenes[0]).split("_")[5]
    if os.path.isfile(areas_path):
        areas = sw.load_areas(areas_path)
        tile = areas["tile"]
        match = [a for a in areas["areas"] if a["id"] == area_id]
        if match:
            split = match[0]["split"]

    date_chips = {}
    for scene in scenes:
        date = os.path.basename(scene).split("_")[2][:8]
        chip = sw.read_tci_window(scene, window).astype(np.float32)
        date_chips.setdefault(date, []).append(chip)
    with np.errstate(invalid="ignore"):
        chips = {"composite": np.nanmedian(
            np.stack([np.nanmedian(np.stack(arrs, axis=0), axis=0)
                      for arrs in date_chips.values()], axis=0), axis=0)}
    for date, arrs in date_chips.items():
        chips[date] = arrs[0] if len(arrs) == 1 else np.nanmedian(
            np.stack(arrs, axis=0), axis=0)
    chips = {key: np.nan_to_num(arr, nan=0).astype(np.uint8)
             for key, arr in chips.items()}

    polygons_path = os.path.join(out_dir, f"area-{area_id:03d}.json")
    existing = []
    if os.path.isfile(polygons_path):
        with open(polygons_path, encoding="utf-8") as handle:
            saved = json.load(handle)
        existing = [{"class": p["class"],
                     "vertices": [[x - col0, y - row0]
                                  for x, y in p["vertices_scene"]]}
                    for p in saved["polygons"]]
        print(f"  resuming with {len(existing)} saved polygons")

    print("----------")
    print("| STEP 7 | annotating")
    print("----------")
    print(f"  cell {area_id} ({split}); "
          "close the window or press Finish when done")
    drawn = sw.annotate_area(chips, existing)
    kept = [{"class": p["class"],
             "vertices_scene": [[x + col0, y + row0]
                                for x, y in p["vertices"]]}
            for p in existing]
    new = [{"class": p["class"],
            "vertices_scene": [[x + col0, y + row0]
                               for x, y in p["vertices"]]}
           for p in drawn]
    merged = [{"id": number + 1, **polygon}
              for number, polygon in enumerate(kept + new)]
    sw.save_polygons(polygons_path, tile, area_id, split, window, merged)
    print(f"  saved {len(merged)} polygons to {polygons_path}")
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.train_areas is None) != (args.test_areas is None):
        parser.error("--train-areas and --test-areas go together")

    # Validate the split before anything expensive: a rejected
    # assignment must not cost a full median run first.
    assignment = None
    if args.train_areas is not None:
        try:
            assignment = sw.validate_areas(args.train_areas,
                                           args.test_areas)
        except ValueError as exc:
            print(f"bad split: {exc}")
            return 2
        print(f"split validates: {assignment}")

    sat_images_dir = os.path.join(c.DATA_DIR, "sat-images")
    month_key = args.month.replace("-", "")
    scenes = [s for s in sw.discover_scenes(sat_images_dir)
              if _month_stamp(s) == month_key]
    if not scenes:
        print(f"no scenes for month {args.month} in {sat_images_dir}")
        return 1
    print(f"using {len(scenes)} scenes for {args.month}")
    tile = os.path.basename(scenes[0]).split("_")[5]

    if assignment is not None:
        print("----------")
        print("| STEP 6 | freezing areas")
        print("----------")
        areas_path = os.path.join(args.out_dir, "areas.json")
        os.makedirs(args.out_dir, exist_ok=True)
        sw.freeze_areas(areas_path, tile, assignment)
        print(f"  froze {assignment} to {areas_path}")

    if args.annotate is None and assignment is None:
        _overview(args.out_dir, scenes, args.device)
        _features(args.out_dir, scenes, args.device, args.month)
        print("inspect the overview preview, draw polygons with "
              "--annotate CELL, then freeze the split")
        return 0

    if args.annotate is not None:
        return _annotate(args.out_dir, scenes, args.annotate)
    print("pass --annotate CELL to draw polygons in one area")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
