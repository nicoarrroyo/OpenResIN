import os

# --- Processing Settings ---
N_CHUNKS                = 5000      # number of chunks into which image is split
HIGH_RES                = True      # use finer 10m spatial resolution (slower)
RES = "10m" if HIGH_RES else "60m"  # derived; see label.apply_overrides
KNOWN_FEATURE_MASKING   = True
CLOUD_MASKING           = True
COMPOSITING             = True
SHOW_INDEX_PLOTS        = False
SAVE_IMAGES             = False
LABEL_DATA              = True
DATA_FILE_NAME_SUFFIX   = str(N_CHUNKS) + "chunks.csv"  # derived, as RES above
TITLE_SIZE              = 5         # title size of plots
PLOT_SIZE               = (3, 3)    # larger increases detail and pixels
PLOT_SIZE_CHUNKS        = (4, 4)
N_IMAGES                = -1        # number of images to use (-1 for all)

# --- Paths ---
# Anchored on this file's location, never on the working directory, so the
# stages resolve the same paths no matter where they are launched from.
PKG_DIR = os.path.dirname(os.path.abspath(__file__))    # src/openresin
HOME_DIR = os.path.dirname(os.path.dirname(PKG_DIR))    # repo root
DATA_DIR = os.path.join(HOME_DIR, "data")
SEED_LABELS_DIR = os.path.join(DATA_DIR, "seed-labels")
OUTPUTS_DIR = os.path.join(HOME_DIR, "outputs")
LABELS_DIR = os.path.join(OUTPUTS_DIR, "labels")
PATCHES_DIR = os.path.join(OUTPUTS_DIR, "patches")
MODELS_DIR = os.path.join(HOME_DIR, "models")

# --- Constants ---
BAND_MAP_H = { # high-res version of band names
    "green": "B03",
    "red": "B04",
    "nir": "B08"
}
BAND_MAP_L = { # low-res version of band names
    "green": "B03",
    "red": "B04",
    "nir": "B8A"
}

# -- Model Settings --
MODEL_TYPE      = "ndwi"        # "ndwi" or "tci"
EPOCHS          = 150
LEARNING_RATE   = 0.001
DROPOUT_RATE    = 0.2
SAVE_MODEL      = True
SHOW_PLOTS      = False         # accuracy/loss curves after training

# -- Dataset Parameters --
IMG_HEIGHT      = int(157 / 5)  # must match BOX_SIZE in data_handling
IMG_WIDTH       = int(157 / 5)
BATCH_SIZE      = 256
VALIDATION_SPLIT = 0.2
BUFFER_SIZE     = 1000
RANDOM_SEED     = 123
CLASS_NAMES     = ["land", "reservoirs", "sea", "water-bodies"]

# -- Default tile --
# No-flag fallback for `predict` and `evaluate`. Will be replaced with
# an automated tile search. For now, the goal is just to get it to run.
DEFAULT_FOLDER  = "S2C_MSIL2A_20250301T111031_N0511_R137_T31UCU_20250301T152054.SAFE"
