import os

# --- Surface-water overview ---
SW_TILE_PX = 10980       # 10 m pixels per tile side (T31UCU scenes)
SW_CELL_PX = 500         # grid cell size in 10 m pixels (5 x 5 km)
SW_GRID_ROWS = 22
SW_GRID_COLS = 22
SW_NODATA_VALUE = 0      # zero in every input band means unimaged swath edge
SW_CLOUD_SHADOW_CLASSES = (1, 2, 3)  # OmniCloudMask classes masked as unusable
SW_OCM_PATCH_SIZE = 1000
SW_OCM_PATCH_OVERLAP = 300
SW_OCM_BATCH_SIZE = 4
SW_OCM_DTYPE = "bf16"
SW_OCM_DEVICE = "cuda"   # falls back to CPU with a warning when unavailable
SW_FEATURES = ("B02", "B03", "B04", "B08", "NDWI", "NDVI", "MNDWI",
               "B11", "B12", "B05", "B06", "B07", "B8A")  # classifier order

# --- Paths ---
# Anchored on this file's location, never on the working directory, so the
# stages resolve the same paths no matter where they are launched from.
PKG_DIR = os.path.dirname(os.path.abspath(__file__))    # src/openresin
HOME_DIR = os.path.dirname(os.path.dirname(PKG_DIR))    # repo root
DATA_DIR = os.path.join(HOME_DIR, "data")
OUTPUTS_DIR = os.path.join(HOME_DIR, "outputs")
