import os

# -- Model Settings --
MODEL_TYPE      = "ndwi"    # "ndwi" or "tci"
EPOCHS          = 500
LEARNING_RATE   = 0.001
DROPOUT_RATE    = 0.2
SAVE_MODEL      = False

# -- Dataset Parameters --
IMG_HEIGHT      = int(157 / 5)  # must match BOX_SIZE in data_handling
IMG_WIDTH       = int(157 / 5)
BATCH_SIZE      = 256
VALIDATION_SPLIT = 0.2
RANDOM_SEED     = 123
CLASS_NAMES     = ["land", "reservoirs", "sea", "water bodies"]

# -- Paths --
SRC_DIR         = os.path.dirname(os.path.abspath(__file__))
HOME_DIR        = os.path.dirname(SRC_DIR)
DATA_DIR        = os.path.join(HOME_DIR, "data")

# Trained models are written here by krisp_trainer (now called train) and
# read back by krisp (now called inference). Both names will be definitively
# updated later on - TODO
# Single source of truth: both sides must import this rather than rebuilding
# the path, which is how they came to disagree in the first place.
MODELS_DIR      = os.path.join(HOME_DIR, "models")
