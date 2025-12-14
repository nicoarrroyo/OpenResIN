import os

# -- Processing Settings --
DPI                     = 3000      # 3000 for full resolution, below 1000, images become fuzzy
N_CHUNKS                = 5000      # number of chunks into which images are split
HIGH_RES                = True      # use finer 10m spatial resolution (slower)
KNOWN_FEATURE_MASKING   = True
CLOUD_MASKING           = True
SHOW_INDEX_PLOTS        = True
SAVE_IMAGES             = False
LABEL_DATA              = True
DATA_FILE_NAME          = "responses_" + str(N_CHUNKS) + "_chunks.csv"

TITLE_SIZE              = 8         # title size of plots
LABEL_SIZE              = 4         # label size (axes) of plotsd
PLOT_SIZE               = (5, 5)    # larger plots increase detail and pixel count
PLOT_SIZE_CHUNKS        = (7, 7)

# -- Paths --
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(DATA_DIR, "data") 

# -- Constants --
BAND_MAP = {
    "green": "B03",
    "red": "B04",
    "nir": "B08"
}
