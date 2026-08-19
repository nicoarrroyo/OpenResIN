# Outputs Directory
This directory holds everything the pipeline considers an output, which includes the user-labelled training data coordinates + images, the mini-chunks cut for prediction, and the model's predictions on a scene. Trained models are the one exception: they go to `models/` at the repository root. The commands you use should be the ones putting things in this directory, not you; it exists for the code to search for and find the correct things. You should never need to put anything in here yourself, and you should never need to edit anything in here by hand. Every script that produces a file writes it into one of the sub-folders below, creating that folder if it does not already exist.

The distinction this directory exists to enforce is:

- `data/` is what **you** put in (satellite imagery, masking layers, and the hand-labelled seed data that ships with the repository).
- `outputs/` is what the **programs** put out (labels from your own labelling sessions, segmented training images, and model predictions).

Nothing in here is tracked by git except this README (see `.gitignore` in the repository root). This is done so that a fresh clone doesn't contain any large model files or other people's training data and so that your own training data doesn't get automatically added to staging or pushed by accident. Also, the sub-folders below appear the first time you run something that writes to them.

## Directory Overview
```
outputs/
├── chunks/         # Mini-chunk PNGs cut for prediction, one folder per scene
├── labels/         # Label coordinates from your own labelling sessions
├── patches/        # Segmented training images, one folder per class
│   ├── land/
│   ├── reservoirs/
│   ├── sea/
│   └── water-bodies/
└── predictions/    # Model predictions across a whole tile
```

## `labels/`
`openresin-label` writes the coordinates you draw in the labelling GUI here, as a CSV named after the tile and the chunk count, e.g. `T31UCU-5000chunks.csv`. One row per chunk, recording how many reservoirs and water bodies you marked in that chunk and the bounding box coordinates for each. 

> [!NOTE]
> The naming convention for the `.csv` file may change, but the code must (and would) be changed first / accordingly. 

**Relationship to `data/seed-labels/`:** the repository ships with a hand-labelled seed file so that a fresh clone can produce training images without sitting through a labelling session first. The seed is copied here and appended to, but the original file is never edited, only read. This should preserve the idea that two different machines can clone the same repo and get the same results on the first run. 

## `patches/`
NALIRA's segmentation step cuts the labelled regions out of the NDWI array and saves them here as 8-bit greyscale PNGs, sorted into one folder per class: `reservoirs`, `water-bodies`, `land`, and `sea`. These are the images the model is trained on.

- **Format**: 8-bit greyscale PNG (mode `L`)
- **Naming**: `{tile}-{date}-{index}.png`, e.g. `T31UCU-20250331-0000.png`
- **Example count**: a full labelling session at 5000 chunks produces thousands of these, which is the main reason this directory is not tracked

Pixels that were `NaN` in the source NDWI array are written as mid-grey (128) so they can be told apart from genuine zero-valued pixels. The number and type of classes are subject to change. The stacking of bands is also subject to change for training. 

## `predictions/`
`openresin-predict` writes one CSV per prediction run, named `P_{chunks}_{epochs}_{tile}.csv`, where the number of chunks the tile was split into, the number of epochs the model was trained for, and the tile number field are what make up the name. Each row is a mini-chunk, its predicted class, and the model's confidence. 

## `chunks/`
`openresin-predict` cuts the scene it is about to classify into mini-chunks and writes them here as 8-bit greyscale PNGs, one folder per scene, and one sub-folder per normalisation setting: `chunks/{scene}.SAFE/ndwi_{multiplier}/`. The multiplier in that folder name is the fraction of the scene maximum used as the top of the greyscale range, so two runs at different settings do not overwrite each other.

- **Format**: 8-bit greyscale PNG (mode `L`), written to match the training patches in `patches/`
- **Naming**: `ndwi chunk {chunk} minichunk {minichunk}.png`
- **Example count**: a full scene at 5000 chunks produces 122,500 files, which is why a rebuild asks for confirmation before it starts

These are inputs to the model rather than results, and they are kept because rebuilding them is the slow part of a prediction run. `openresin-predict` reuses an existing set and only regenerates it if you pass `--rebuild-inputs`. Delete a scene's folder if you want it rebuilt without the flag. Note that the prediction step currently finds these files by reading the directory and parsing the chunk and mini-chunk numbers back out of the filenames, so the naming above is load-bearing.

## Current Status
`openresin-label` writes to `labels/` and `patches/`, and `openresin-predict` writes to `chunks/` and `predictions/`, all as described above.
