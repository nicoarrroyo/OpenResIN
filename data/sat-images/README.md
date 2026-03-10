# Sentinel 2 Directory
This section will describe the Sentinel 2 folder naming structure and how / where to put your downloaded Sentinel 2 images for IPDMP to run as intended. Any example folder names added throughout this description will be referenced like so: 

> **`EXAMPLE FOLDER`**

and any example file names will be referenced like so:

> **`EXAMPLE_FILE.jp2`**

The only step required by the user is to extract the relevant folder in the correct place. There is also some more documentation regarding the specific Sentinel 2 folder naming convention included, but if everything is set up correctly, that information shouldn't be necessary.

## Sentinel 2 Folder Naming
In this section I will briefly explain the Sentinel 2 image folder aming convention. Upon downloading a selected image from the [Copernicus Browser](https://browser.dataspace.copernicus.eu/?zoom=5&lat=50.16282&lng=20.78613&demSource3D=%22MAPZEN%22&cloudCoverage=30&dateMode=SINGLE) the file on your computer will probably be roughly 1 GB in size and it should be packaged as a `.zip` folder. This should be extracted and the target extraction location should be **THIS** directory (the one you're in right now!), which is where IPDMP will search for the images. If you want IPDMP to composite several images over each other, extract other folders **OF THE SAME TILE** into the  Once this extraction is complete, a folder will be in this directory and it will look something like this:

> **`S2C_MSIL2A_20250331T110651_N0511_R137_T31UCU_20250331T143812.SAFE`**

This folder name follows a strict convention, where information is separated by underscores (`_`). Using the example above, the sections are defined as follows:

- **`S2C`**: The satellite name (e.g., Sentinel-2C). This could also be `S2A` or `S2B`.
- **`MSIL2A`**: Describes the instrument and product level. `MSI` stands for Multi-Spectral Instrument, and `L2A` indicates Product Level 2A (post-processed).
- **`20250331T110651`**: The "datatake start sensing time" formatted as `YYYYMMDDTHHMMSS`, with the `T` in the middle separating date and time.
- **`N0511`**: The Processing Baseline Number.
- **`R137`**: The Relative Orbit Number (ranging from 001 to 143), indicating the specific orbit track.
- **`T31UCU`**: The Tile Number Field, which describes the specific 110km x 110km area on Earth covered by the image.
- **`20250331T143812.SAFE`**: The Product Discriminator and format extension. The `.SAFE` suffix denotes the Standard Archive Format for Europe.

IPDMP uses these folder names and strict naming conventions to navigate the image folders, so it is imperative that the folders are not renamed, otherwise the satellite images will not be found and the program will not be able to run properly. 

## Example Image Location
In this subsection, I will use a sample image to demonstrate the path IPDMP takes to reach a given file. 

### Image File
> **`T31UCU_20250331T110651_B04_10m.jp2`**

Using the convention defined above, we can identify this image as one of the Tile Number Field **`T31UCU`** (which is in East England), taken on the 31st of March 2025 at 11h 06m 51s. Additional information not included above is the band number **`B04`**, which is the red band, and the spatial resolution **`10m`**, which essentially means that each pixel covers a 10m x 10m area (maximum resolution for Sentinel 2).

This satellite image is contained in a folder which contains all the 10m resolution images:

### Resolution Folder
> **`R10m`**

This "resolution folder" is itself contained in a folder which contains all possible resolutions for Sentinel 2 (**`10m`**, **`20m`**, **`60m`**), where the **`10m`** folder is used for all labelling and index calculation, and the **`60m`**, resolution folder is used for troubleshooting the program because it takes less time to open the smaller files. The **`20m`** images are used only if there is a band that is necessary for an index calculation which Sentinel 2 does not have at a **`10m`** resolution. This resolution parent folder is called:

### Images Folder
> **`IMG_DATA`**

This folder is in the same directory as two other `DATA` folders called `AUX_DATA` and `QI_DATA`, which stand for Auxilliary and Quality Indicator Data respectively. These other folders are no longer used by IPDMP directly, although they may be used by OmniCloudMask or other masking processes. 

### Intermittent Processed Level Folder
All three of these folders are in yet another folder. 
> **`L2A_T31UCU_A002967_20250331T111534`**

This folder doesn't quite follow the same convention as the rest of the data, but fortunately it is the only item in its directory, so it can be searched for iteratively instead of by using naming conventions. This **`L2A`** folder is in a final folder: 

### Initial Folder
> **`GRANULE`**

This folder is contained in the main parent folder from which we were able to extract all the naming information (the one starting with **`S2C`** in this case), meaning we have finally reached the end of this very long branching process. This **`GRANULE`** folder is in the same directory as a couple other files and folders which, similarly to the `AUX_DATA` and `QI_DATA` folders are not explicitly used by IPDMP but should not be deleted regardless. 

### Summary and File Tree
To summarise, the image file is contained in a resolution folder, which is contained in an images folder, which is contained in the intermittent folder, which is contained in the initial folder, which is contained in the main parent folder. To visualise, use the file tree below: 

- _`DATASTRIP/`_
  - _`DS_2CPS_20250331T143812_S20250331T111534/`_
    - _`...`_
- **`GRANULE/`**
  - **`L2A_T31UCU_A002967_20250331T111534/`**
    - _`AUX_DATA/`_
      - _`...`_
    - **`IMG_DATA/`**
      - **`R10m/`**
        - **`T31UCU_20250331T110651_B04_10m.jp2`**
    - _`QI_DATA/`_
      - _`...`_
- _`HTML/`_
  - _`UserProduct_index.html`_
    - _`...`_
- _`rep_info/`_
  - _`S2_PDI_Level-2A_Datastrip_Metadata.xsd`_
  - _`...`_
- _`INSPIRE.xml`_
- _`manifest.safe`_
- _`MTD_MSIL2A.xml`_

It is also important to note that as you run the different scripts in this project (NALIRA, KRISP-Y, KRISP_trainer, etc.) different images, files, or folders will be generated and placed in the parent folder (starting with **`S2C`**). This is normal and intended behaviour. 

## Notes
hi :) no notes here :)




