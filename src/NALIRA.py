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
os.environ["CUDA_HOME"] = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2"
import numpy as np
import pipeline_operations as operation
import user_interfacing as ui_do
import config_NALIRA as c

ui_do.table_print(n_chunks=c.N_CHUNKS, n_images=c.N_IMAGES,
                  high_res=c.HIGH_RES, 
                  cloud_masking=c.CLOUD_MASKING, 
                  known_feature_masking=c.KNOWN_FEATURE_MASKING, 
                  show_plots=c.SHOW_INDEX_PLOTS, save_images=c.SAVE_IMAGES, 
                  labelling=c.LABEL_DATA)

index_arrays = {"ndwi": [], "ndvi": []}
tci_array = np.empty([1,1]); tci_60_array = np.empty([1,1])

folders_path = os.path.join(c.HOME_DIR, "data", "sat-images")
folders = ui_do.list_folders(folders_path)

    # %% 1. Create Image Arrays
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
     images_path, 
     tci_array, 
     tci_60_array
     ] = operation.one_create_image_arrays(
         folders_path, 
         folder, 
         tci_60_array # for checking if a tci has been opened yet
         )
    
    # %% 2. Mask Known Features
    print("----------")
    print("| STEP 2 |")
    print("----------")
    if c.KNOWN_FEATURE_MASKING:
        image_arrays = operation.two_mask_known_feature(
            image_arrays, 
            image_metadata)
    else:
        print("skipping known feature masking")
    
    # %% 3. Mask Clouds (Omnicloudmask)
    print("----------")
    print("| STEP 3 |")
    print("----------")
    if c.CLOUD_MASKING:
        image_arrays = operation.three_mask_clouds(image_arrays)
    else:
        print("skipping cloud masking")
    
    # %% 4. Calculate Spectral Indices
    print("----------")
    print("| STEP 4 |")
    print("----------")
    indices = operation.four_compute_indices(image_arrays)
    for key in index_arrays:
        index_arrays[key].append(indices[key])

# %% 5. Composite Images (and plot)
print("----------")
print("| STEP 5 |")
print("----------")
stms = operation.five_composite(index_arrays)
ndwi_mean = stms["ndwi"]["median"] # temporary. will replace with full stm

if c.SHOW_INDEX_PLOTS:
    operation.fiveb_plot(ndwi_mean, folders_path)
else:
    print("skipping water index image display")

# %% 6. Prepare Labelling Data
print("----------")
print("| STEP 6 |")
print("----------")
if c.LABEL_DATA:
    [index_chunks, 
     tci_chunks, 
     break_flag, 
     i, 
     data_file_path, 
     data_correction, 
     invalid_rows, 
     lines, 
     last_chunk, 
     labelling_path
     ] = operation.six_prepare_data(
         ndwi_mean, 
         tci_array, 
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
    operation.seven_label_data(
         i, 
         index_chunks, 
         ndwi_mean, 
         tci_chunks, 
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
operation.eight_segment_data(
    data_file_path, 
    index_chunks, 
    labelling_path, 
    prefix)
