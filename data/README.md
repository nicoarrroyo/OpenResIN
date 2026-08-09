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
├── training-data/           # Already hand-labelled training data
└── sat-images/              # Preprocessed data (generated)
```

## `masks`
This folder contains several sub-folders. Each sub-folder will expect a type of file that the program will automatically search for and use for masking. For more information, visit the README file in the `masks` directory.
- **Format**: Depends (.shp + .shx + .dbf + .prj)
- **Naming**: Should match corresponding satellite image region
- **Example**: `region_2024_boundaries.shp`

## `training-data`
This contains training data that has already been labelled by hand. As the program runs, if the user labels more data, more files are placed somewhere else. This data serves as a public starting point. 

## `sat-images`
This folder expects Sentinel 2 satellite image folders, which can be downloaded from the Copernicus Browser as .zip folders. Simply unzip the folder here and the program should find everything automatically. Please take care to not change any of the names of the files or sub-folders inside the image folder, as the program uses specific, standardised naming rules to find the necessary files. For more information, visit the README file in the `sentinel_2` directory.
- **File Format**: GeoTIFF (.tif)
- **Folder Format**: Standard Archive Format for Europe (SAFE) format specification (see ESA SentiWiki)
- **Naming**: Standardised ESA Sentinel folder naming scheme.
- **Example**: `S2B_MSIL2A_20240719T110619_N0510_R137_T31UCU_20240719T142134.SAFE`
