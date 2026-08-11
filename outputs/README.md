# Outputs Directory
This directory holds everything the the pipeline considers an output, which includes the user-labelled training data coordinates + images, and the trained model. The programs you use should be the ones putting things in this directory; it exists for the code to search for and find the correct things. You should never need to put anything in here yourself, and you should never need to edit anything in here by hand. Every script that produces a file writes it into one of the sub-folders below, creating that folder if it does not already exist.

The distinction this directory exists to enforce is:

- `data/` is what **you** put in (satellite imagery, masking layers, and the hand-labelled seed data that ships with the repository).
- `outputs/` is what the **programs** put out (labels from your own labelling sessions, segmented training images, and model predictions).

Nothing in here is tracked by git except this README (see `.gitignore` in the repository root). This is done so that a fresh clone doesn't contain any large model files or other people's training data and so that your own training data doesn't get automatically added to staging or pushed by accident. Also, the sub-folders below appear the first time you run something that writes to them.

## Directory Overview
```
outputs/
├── labels/         # Label coordinates from your own labelling sessions
├── patches/        # Segmented training images, one folder per class
│   ├── land/
│   ├── reservoirs/
│   ├── sea/
│   └── water-bodies/
└── predictions/    # Model predictions across a whole tile
```

## `labels`
NALIRA writes the coordinates you draw in the labelling GUI here, as a CSV named after the tile and the chunk count, e.g. `T31UCU-5000chunks.csv`. One row per chunk, recording how many reservoirs and water bodies you marked in that chunk and the bounding box coordinates for each. NOTE: the naming convention for the `.csv` file may change, but the code must be changed first / accordingly. 

**Relationship to `data/seed-labels/`:** the repository ships with a hand-labelled seed file so that a fresh clone can produce training images without sitting through a labelling session first. The seed is copied here and appended to, but the original file is never edited, only read. This should preserve the idea that two different machines can clone the same repo and get the same results on the first run. 

## `patches`
NALIRA's segmentation step cuts the labelled regions out of the NDWI array and saves them here as 8-bit greyscale PNGs, sorted into one folder per class: `reservoirs`, `water-bodies`, `land`, and `sea`. These are the images the model is trained on.

- **Format**: 8-bit greyscale PNG (mode `L`)
- **Naming**: `{tile}-{index}.png`, e.g. `T31UCU-0000.png`
- **Example count**: a full labelling session at 5000 chunks produces thousands of these, which is the main reason this directory is not tracked

Pixels that were `NaN` in the source NDWI array are written as mid-grey (128) so they can be told apart from genuine zero-valued pixels. The number and type of classes are subject to change. The stacking of bands is also subject to change for training. 

## `predictions`
KRISP-Y writes one CSV per prediction run, named `P_{chunks}_{epochs}_{tile}.csv`, where the number of chunks the tile was split into, the number of epochs the model was trained for, and the tile number field are what make up the name. Each row is a mini-chunk, its predicted class, and the model's confidence. 

## Current Status
NALIRA writes to `labels/` and `patches/` as described above. KRISP-Y will write the model's predictions to `predictions/`, but this is currently being developed. Right now, KRISP-Y writes its predictions to a `.csv` in the satellite image folder which it uses to make its predictions. 
