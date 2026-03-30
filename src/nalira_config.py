import os

# --- Processing Settings ---
N_CHUNKS                = 5000      # number of chunks into which images are split

HIGH_RES                = False     # use finer 10m spatial resolution (slower)

RES = "10m" if HIGH_RES else "20m"

KNOWN_FEATURE_MASKING   = False

CLOUD_MASKING           = False

COMPOSITING             = True

SHOW_INDEX_PLOTS        = True

SAVE_IMAGES             = False

LABEL_DATA              = True

DATA_FILE_NAME_SUFFIX   = str(N_CHUNKS) + "chunks.csv"

TITLE_SIZE              = 5         # title size of plots

PLOT_SIZE               = (3, 3)    # larger plots increase detail and pixel count

PLOT_SIZE_CHUNKS        = (4, 4)

N_IMAGES                = 1         # number of the found images to use (-1 for all)
# --- Processing Settings ---

# --- Paths ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

HOME_DIR = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(HOME_DIR, "data")
# --- Paths ---

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
# --- Constants ---import os

# -- Processing Settings --
LP_MODE                 = False     # low-power mode (changes order of operations)
N_CHUNKS                = 5000      # number of chunks into which images are split
HIGH_RES                = False     # use finer 10m spatial resolution (slower)
RES = "10m" if HIGH_RES else "60m"
KNOWN_FEATURE_MASKING   = False
CLOUD_MASKING           = False
COMPOSITING             = True
SHOW_INDEX_PLOTS        = True
SAVE_IMAGES             = False
LABEL_DATA              = True
# DATA_FILE_NAME          = "responses_" + str(N_CHUNKS) + "_chunks.csv"
DATA_FILE_NAME_SUFFIX   = str(N_CHUNKS) + "chunks.csv"

TITLE_SIZE              = 5         # title size of plots
PLOT_SIZE               = (3, 3)    # larger plots increase detail and pixel count
PLOT_SIZE_CHUNKS        = (4, 4)

N_IMAGES                = -1         # number of the found images to use (-1 for all)

# -- Paths --
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(HOME_DIR, "data")

# -- Constants --
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
