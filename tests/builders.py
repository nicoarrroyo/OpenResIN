"""Builders shared by the surface-water test files.

Each returns plain dictionaries in the on-disk formats that label-sw writes
and train-sw reads, so the tests in different files describe the same
hand-off the same way.
These are basically copies of the real code, made to produce fake data to
test against in the suite.
"""
import os

from openresin import config as c


def feature_provenance(scenes, sea_source, urban_source, **overrides):
    """A features-provenance.json record that matches the given scenes."""
    provenance = {
        "tile": "T31UCU",
        "month": "2026-04",
        "source_scenes": sorted(os.path.basename(s) for s in scenes),
        "feature_order": list(c.SW_FEATURES),
        "aggregation": "valid median within date, then median across dates",
        "masks": {
            "cloud_shadow_classes": list(c.SW_CLOUD_SHADOW_CLASSES),
            "nodata_value": c.SW_NODATA_VALUE,
            "sea_source": sea_source,
            "urban_source": urban_source,
        },
    }
    provenance.update(overrides)
    return provenance


def area_record(window, polygons, exclusions=None, completion=None,
                area_id=1, split="train"):
    """An area-NNN.json record; completion is added only when given."""
    record = {
        "tile": "T31UCU",
        "area_id": area_id,
        "split": split,
        "window": list(window),
        "polygons": polygons,
        "exclusions": exclusions or [],
    }
    if completion is not None:
        record["completion"] = completion
    return record


def rectangle(col0, row0, col1, row1, class_name="water", polygon_id=1):
    """An axis-aligned polygon in scene pixel coordinates."""
    return {
        "id": polygon_id,
        "class": class_name,
        "vertices_scene": [
            [col0, row0], [col1, row0], [col1, row1], [col0, row1]
        ],
    }
