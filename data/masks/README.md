# Masks Directory
This directory is built to contain all the masking information requested by `nalira` (data-labelling portion of `OpenResIN`). The example file names shown below are the exact files used in the development of the program. Currently, these are the only files compatible with `nalira`, however in future, this should be more flexible. 

## Boundaries
- **File Source**: [UK GOV](https://www.data.gov.uk/dataset/2e17269d-10b9-4e43-b67b-57f9b02bd0f8/countries-december-2021-boundaries-uk-buc)
- **File Type**: Geographic JavaScript Objection Notation (`.geojson`) file
- **File Path**: `boundaries` -> `Regions_December_2024_Boundaries_EN_BSC-6948965129330885393.geojson`

To restrict study to one country, we use a BSC (Boundary, Super-Resolution, Clipped) `.geojson` file which contains information about the outline of England. This has the significant added benefit of masking out the sea as well. The sea can make up a large part of a satellite image, and since there are often clouds covering parts of the sea as well, small segments can be confused with reservoirs. 

## Known Reservoirs
- **File Source**: [UK GOV](https://www.data.gov.uk/dataset/aa1e16e8-eded-4a60-8d1d-0df920c319b6/inventory-of-reservoirs-amounting-to-90-of-total-uk-storage)
- **File Type**: Shapefile (`.shp`)
- **File Path**: `known-reservoirs` -> `LRR_EW_202307_v1` -> `SHP` -> `LRR_ENG_20230601_WGS84.shp`

The data from the UK Government here is given in a shapefile, however the location of each reservoir is given only by 1 pixel, rather than the outline of the reservoir itself. Because of this, each reservoir mask would only be 1 pixel in size. Therefore, an abtitrary buffer size of 50m was chosen and applied to each of these points so that each known reservoir is identified with a large white (white in `nalira` but can be any `NaN` or 0-value plot colour) dot in its centre. This completely eliminates some reservoirs, while some others are still largely visible, only with a dot in the centre.

## Rivers and Streams
- **File Source**: [Ordnance Survey](https://www.ordnancesurvey.co.uk/products/os-open-rivers)
- **File Type**: Shapefile (`.shp`)
- **File Path**: `rivers` -> `data` -> `WatercourseLink.shp`

This data is given as points in this shapefile, which means that each point that shows a river is only 1 pixel in size. Therefore, for this masking, a buffer width of 10m is applied, which ensures that almost all rivers and streams are completely masked out. There are, of coure, larger rivers that are not entirely masked, but there is still a streak down the middle of the river that makes it obvious where the river is. 

## Urban Areas
- **File Source**: [UK CEH](https://catalogue.ceh.ac.uk/documents/5af9e97d-9f33-495d-8323-e57734388533?_gl=1*4k1tkl*_ga*NTgzNDcxODg0LjE3NTUwOTY2NjE.*_ga_27CMQ4NHKV*czE3NTUwOTY2NjEkbzEkZzAkdDE3NTUwOTY2NjEkajYwJGwwJGgw)
- **File Type**: Tagged Image Format `.tif` file in a `data` folder in a main folder
- **File Name**: `urban-areas` -> `CEH_GBLandCover_2024_10m` -> `data` -> `4dd9df19-8df5-41a0-9829-8f6114e28db1` -> `gblcm2024_10m.tif`

The main folder also contains a supporting documents folder, a `.DS_Store` file, and a `readme.html` file. This data seems to mask out some known reservoirs as well. This is a known issue. This is also the least easily accessible data.

## High-Slope Terrain (NOT IMPLEMENTED YET)
- **File Source**: [UK GOV](https://environment.data.gov.uk/dataset/13787b9a-26a4-4775-8523-806d13af58fc) or [Ordnance Survey](https://www.ordnancesurvey.co.uk/products/os-terrain-50#get)
- **File Type**: `.zip` folder in a sub-folder in a `data` folder in a folder
- **File Name**: `terrain` -> `terr_50_cesh_gb` -> `data` -> e.g. `hp` -> `hp40_OST50CONT_2025_0529.zip`

Implementation of high-slope area masking is planned, but not complete yet.

## Special Case (Clouds)
Cloud masking is already implemented in `OpenResIN`, however there is no need for the user to place any specific files in any specific directory. `OpenResIN` uses the [OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask) Python library (can be installed and found hosted on GitHub), which is a deep learning solution to cloud and cloud shadow segmentation.

## Notes
- The OmniCloudMask, since it is built on deep learning principles, relies heavily on using a discrete Graphics Processing Unit (GPU), specifically NVIDIA GPUs (using the proprietary NVIDIA software for machine learning, CUDA). `OpenResIN` will try to deploy OmniCloudMask using CUDA, but if an NVIDIA GPU is not detected on your system, it will shift to using the CPU, which is significantly slower (modern NVIDIA GPU inference time ~= 30 seconds, modern CPU inference time ~= 1 hour) and therefore not recommended. 
