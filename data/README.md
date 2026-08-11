# Data Directory
This directory contains all input and already-existing data for the machine learning pipeline. There should be three folders in this directory. These folders are the place where you (the user) will place the data that the program will then use to train the ROI recognition model, deploy the model, or aid in data labelling. This README explains the purpose and expected data format for each of these folders. 

## Directory Overview
```
data/
├── masks/                   # Masking files organized by category
│   ├── boundaries/          # Boundary masks
│   ├── known-reservoirs/    # Known reservoir masks
│   ├── rivers/              # River masks
│   ├── terrain/             # Terrain masks
│   └── urban-areas/         # Urban area masks
├── seed-labels/             # Hand-labelled seed data, shipped with the repository
└── sat-images/              # Sentinel-2 imagery you download and extract yourself
```

Everything the pipeline *produces* goes in `outputs/` at the repository root, not here. See the README in that directory.

## `masks`
This folder contains several sub-folders. Each sub-folder will expect a type of file that the program will automatically search for and use for masking. For more information, visit the README file in the `masks` directory.
- **Format**: Depends (.shp + .shx + .dbf + .prj)
- **Naming**: Should match corresponding satellite image region
- **Example**: `region_2024_boundaries.shp`

## `seed-labels`
This contains label data that has already been labelled by hand, and it is the one folder in `data/` whose contents are tracked by git rather than supplied by you. It serves as a public starting point: a fresh clone can produce training images straight away, without sitting through a labelling session first.

NALIRA reads these files but never writes to them. Labels you create yourself are written to `outputs/labels/` instead, so your work never appears as a modification to a tracked file.
- **Format**: Comma-separated values (`.csv`), one row per chunk
- **Naming**: `{tile}-{chunks}chunks.csv`
- **Example**: `T31UCU-5000chunks.csv`

## `sat-images`
This folder expects Sentinel 2 satellite image folders, which can be downloaded from the Copernicus Browser as .zip folders. Simply unzip the folder here and the program should find everything automatically. Please take care to not change any of the names of the files or sub-folders inside the image folder, as the program uses specific, standardised naming rules to find the necessary files. For more information, visit the README file in the `sat-images` directory.
- **File Format**: JPEG 2000 (.jp2)
- **Folder Format**: Standard Archive Format for Europe (SAFE) format specification (see ESA SentiWiki)
- **Naming**: Standardised ESA Sentinel folder naming scheme.
- **Example**: `S2B_MSIL2A_20240719T110619_N0510_R137_T31UCU_20240719T142134.SAFE`
