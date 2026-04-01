""" Navigable Automated Labelling Interface for Regions of Attention (NALIRA)

Description:
NALIRA processes Sentinel 2 imagery to generate labelled data for the 
Keras Reservoir Identification Sequential Platform (KRISP) as part of the 
overarching Individual Project Data to Model Pipeline (IPDMP). It extracts 
water body information from satellite imagery and provides a UI for data 
labelling. The purpose of NALIRA is to create training and test data for KRISP.

Workflow:
1. Data Ingestion:
    - Reads Sentinel 2 image folders and locates necessary image bands.

2. Known Feature Masking:
    - Sea: Can use region shapefile boundaries
    - Rivers and streams: Uses dedicated river shapefile with 100m buffer
    - Large reservoirs: Uses Northing and Easting information in CSV file
    - Urban areas: Uses .tif rasterized for masking
    - Areas with large slopes: 

3. Cloud Masking: 
    - OmniCloudMask, using red and green Sentinel 2 bands

4. Index Calculation:
    - Compute necessary spectral indices
        - Normalized Difference Water Index (NDWI)
        - Normalized Difference Vegetation Index (NDVI)
        - Enhanced Vegetation Index (EVI)

5. Compositing:
    - A set of Spectral-Temporal Metrics (STMs) computed for all pixels. 
        - These metrics are based on the temporal median, and 25th and 75th 
        percentiles of NDWI, NDVI, and/or EVI. 
        - The NDVI and/or EVI can be used to differentiate vegetation water 
        content from surface water bodies. 
    - Optional data visualization at this stage. 

7. Training Data Polygons:
    - Preliminary data preparation steps, including ensuring file content 
    validity and presence, as well as locating and opening the necessary True 
    Colour Image (TCI) for data labelling. 
    - Provides a Tkinter GUI for manual region of interest (ROI)  labelling via 
    rectangle selection.
    - Uses chunk-based processing; saves the quantity of water reservoirs and 
    water bodies, labelled ROI coordinates, and chunk numbers to a CSV file. 

Outputs:
    - Labelled data in CSV format, with chunk IDs, counts of water bodies, and 
    their coordinates.
    - Python list containing each calculated water index.
"""
# %% Start
import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) # fix working directory
import numpy as np
import nalira_operations as operation
import user_interfacing as ui_do
from misc import pre_run_checks

import nalira_config as c

folders_path = os.path.join(c.HOME_DIR, "data", "sat-images")
folders = ui_do.list_folders(folders_path)

LP_MODE = pre_run_checks()
if LP_MODE:
    ui_do.alert_user(
        warning=("Pre-run checks found that your machine lacks the supported "
                 "hardware to accelerate the regular NALIRA workflow."), 
        consequence=("The program wants to switch to the low-power mode "
                     "(LP_mode) branch, where expensive operations like cloud "
                     "masking and percentile calculations will be carried out "
                     "on a chunk-by-chunk basis. Data segmentation in LP_MODE "
                     "is not supported yet, but your responses will be saved."), 
        solution="Accept the switch to LP_MODE.")
    image_arrays_list = []
ui_do.confirm_continue_or_exit()

ui_do.table_print(
    n_chunks=c.N_CHUNKS, n_images=c.N_IMAGES, high_res=c.HIGH_RES, 
    known_feature_masking=c.KNOWN_FEATURE_MASKING, 
    cloud_masking=c.CLOUD_MASKING, 
    compositing=c.COMPOSITING, 
    show_plots=c.SHOW_INDEX_PLOTS, 
    save_images=c.SAVE_IMAGES, 
    labelling=c.LABEL_DATA, 
    low_power=LP_MODE)

    # %% 1. Create Image Arrays
tci_array = np.empty([1,1]); tci_60_array = np.empty([1,1])
index_arrays = {"ndwi": [], "ndvi": []}
for folder_num, folder in enumerate(folders):
    print("\n===============")
    print(f"|| IMG {folder_num+1} / {len(folders)} ||")
    print("===============")
    print("----------")
    print("| STEP 1 |")
    print("----------")
    [image_arrays, 
     image_metadata, 
     prefix, 
     tci_array, 
     tci_60_array
     ] = operation.one_create_image_arrays(
         folders_path, 
         folder, 
         tci_60_array # for checking if a tci has been opened yet
         )
    if LP_MODE:
        image_arrays_list.append(image_arrays)
    
    # %% 2. Mask Known Features
    print("----------")
    print("| STEP 2 |")
    print("----------")
    if c.KNOWN_FEATURE_MASKING:
        image_arrays = operation.two_mask_known_feature(
            image_arrays, 
            image_metadata)
        if LP_MODE: # overwrite previous image arrays if features get masked
            image_arrays_list[folder_num] = image_arrays
    elif not c.KNOWN_FEATURE_MASKING:
        print("skipping known feature masking")
    
    # %% 3. Mask Clouds (Omnicloudmask)
    print("----------")
    print("| STEP 3 |")
    print("----------")
    if not LP_MODE:
        if c.CLOUD_MASKING:
            image_arrays = operation.three_mask_clouds(image_arrays)
        elif not c.CLOUD_MASKING:
            print("skipping cloud masking")
    elif LP_MODE:
        print("skipping cloud masking (done during labelling)")
    
    # %% 4. Calculate Spectral Indices
    print("----------")
    print("| STEP 4 |")
    print("----------")
    if not LP_MODE:
        indices = operation.four_compute_indices(image_arrays)
        for key in index_arrays:
            index_arrays[key].append(indices[key])
    elif LP_MODE:
        print("skipping spectral index calculation (done during labelling)")

# %% 5. Composite Images (and plot)
print("----------")
print("| STEP 5 |")
print("----------")
if not LP_MODE:
    if c.COMPOSITING:
        stms = operation.five_composite(index_arrays)
        labelling_array = stms["ndwi"]["median"] # TODO replace with full stm
    elif not c.COMPOSITING:
        labelling_array = operation.five_mean(index_arrays)["ndwi"]
elif LP_MODE:
    labelling_array = image_arrays_list # known features get masked in LP_MODE
    print("skipping image compositing (done during labelling)")

if c.SHOW_INDEX_PLOTS and not LP_MODE:
    operation.fiveb_plot(labelling_array, folders_path)
else:
    print("skipping water index image display")

# %% 6. Prepare Labelling Data
print("----------")
print("| STEP 6 |")
print("----------")
if c.LABEL_DATA:
    [break_flag, 
     i, 
     data_file_path, 
     data_correction, 
     invalid_rows, 
     lines, 
     last_chunk, 
     labelling_path
     ] = operation.six_prepare_data(
         folders, 
         prefix
         )
else:
    print("skipping data preparation")

# %% 7. Label Data
print("----------")
print("| STEP 7 |")
print("----------")

if c.LABEL_DATA:
    index_chunks = operation.seven_label_data(
        LP_MODE,
        i, 
        labelling_array, 
        tci_array, 
        tci_60_array, 
        data_file_path, 
        data_correction, 
        invalid_rows, 
        lines, 
        last_chunk
        )
else:
    print("skipping data labelling")

# %% 8. Save Labelling Results
print("----------")
print("| STEP 8 |")
print("----------")
if not LP_MODE:
    operation.eight_segment_data(
        data_file_path, 
        index_chunks, 
        labelling_path, 
        prefix
        )
elif LP_MODE:
    print("skipping data segmentation (cannot be done in LP MODE)")
