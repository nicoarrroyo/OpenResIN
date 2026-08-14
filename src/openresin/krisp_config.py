import os

# -- Model Settings --
MODEL_TYPE      = "ndwi"    # "ndwi" or "tci"
EPOCHS          = 150
LEARNING_RATE   = 0.001
DROPOUT_RATE    = 0.2
SAVE_MODEL      = False
SHOW_PLOTS      = False     # accuracy/loss curves after training

# -- Dataset Parameters --
IMG_HEIGHT      = int(157 / 5)  # must match BOX_SIZE in data_handling
IMG_WIDTH       = int(157 / 5)
BATCH_SIZE      = 256
VALIDATION_SPLIT = 0.2
RANDOM_SEED     = 123
CLASS_NAMES     = ["land", "reservoirs", "sea", "water bodies"]

# -- Default tile --
# The scene predict and evaluate both fall back to when --folder is not given.
# This will, in future, be replaced with an automated search for a tile. For now,
# the goal is just to get it to run.
DEFAULT_FOLDER  = "S2C_MSIL2A_20250301T111031_N0511_R137_T31UCU_20250301T152054.SAFE"

# -- Paths --
# Anchored on this file's location, never on the working directory, so the
# stages resolve the same paths no matter where they are launched from.
PKG_DIR         = os.path.dirname(os.path.abspath(__file__))    # src/openresin
HOME_DIR        = os.path.dirname(os.path.dirname(PKG_DIR))     # repo root
DATA_DIR        = os.path.join(HOME_DIR, "data")

# Trained models are written here by krisp_trainer (now called train) and
# read back by krisp (now called inference). Both names will be definitively
# updated later on - TODO
# Single source of truth: both sides must import this rather than rebuilding
# the path, which is how they came to disagree in the first place.
MODELS_DIR      = os.path.join(HOME_DIR, "models")
