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
# %%% Standard Libraries
import time
import os
import csv
import datetime as dt

# %%% Third Party Libraries
import tensorflow as tf
import numpy as np
from PIL import Image
from omnicloudmask import predict_from_array
import rasterio

# %%% Local Libraries
from data_handling import rewrite, blank_entry_check, check_file_permission
from data_handling import extract_coords, change_to_folder, create_tf_example
from data_handling import hash_tfrecord

from image_handling import image_to_array, known_feature_mask, mask_urban_areas
from image_handling import plot_indices, plot_chunks#, save_training_file

from misc import get_sentinel_bands, split_array, combine_sort_unique

from user_interfacing import table_print, prompt_roi, list_folders
from user_interfacing import confirm_continue_or_exit

import config as c
from objects import sentinel_image


table_print(n_chunks=c.N_CHUNKS, high_res=c.HIGH_RES, 
            cloud_masking=c.CLOUD_MASKING, 
            known_feature_masking=c.KNOWN_FEATURE_MASKING, 
            show_plots=c.SHOW_INDEX_PLOTS, save_images=c.SAVE_IMAGES, 
            labelling=c.LABEL_DATA)

ndwi_arrays_list = []
# ndvi_arrays_list = []
# evi_arrays_list = []
# evi2_arrays_list = []

def one_create_image_arrays(folder_path):
    # %%%% 1.1 Establishing Paths
    """Most Sentinel 2 files that come packaged in a satellite image 
    folder follow naming conventions that use information contained in the 
    title of the folder. This information can be used to easily navigate 
    through the folder's contents."""
    file_paths = []
    folder_path = os.path.join(folders_path, folder)
    images_path = os.path.join(folder_path, "GRANULE")
    
    # %%%%% 1.1.1 Subfolder iterative search
    """This folder has a strange naming convention that doesn't quite apply 
    to the other folders, so it's difficult to find a rule that would work 
    for any Sentinel 2 image. The easier way of finding this folder is by 
    searching for any available directories in the GRANULE folder, and if 
    there is more than one, then alert the user and exit, otherwise go into 
    that one directory because it will be the one we're looking for."""
    subdirs = [d for d in os.listdir(images_path) 
               if os.path.isdir(os.path.join(images_path, d))]
    if len(subdirs) == 1:
        images_path = os.path.join(images_path, subdirs[0])
    else:
        print("Too many subdirectories in 'GRANULE':", len(subdirs))
        confirm_continue_or_exit()
    
    # %%%%% 1.1.2 Resolution selection and file name deconstruction
    """Low resolution should only be used for troubleshooting as it does 
    not produce usable training data. High resolution uses the 10m spatial 
    resolution images but processing time is significantly longer. To save 
    some space, when splitting the folder name into its parts, some of the 
    splitting functions outputs are suppressed. In order, the variables not 
    assigned are: sentinel number, instrument and product level, 
    processing baseline number, relative orbit number, and product 
    discriminator and format."""
    if c.HIGH_RES:
        res = "10m"
        path_10 = os.path.join(images_path, "IMG_DATA", "R10m")
    else:
        res = "60m"
        path_60 = os.path.join(images_path, "IMG_DATA", "R60m")
    
    (_, _, datatake_start_sensing_time, _, _, 
     tile_number_field, _) = folder.split("_")
    
    prefix = f"{tile_number_field}_{datatake_start_sensing_time}"
    bands = get_sentinel_bands(sat_number, HIGH_RES)
    
    for band in bands:
        if c.HIGH_RES:
            file_paths.append(
            os.path.join(path_10, f"{prefix}_B{band}_10m.jp2")
            )
        else:
            file_paths.append(
            os.path.join(path_60, f"{prefix}_B{band}_60m.jp2")
            )
    
    # %%%% 1.2 Opening and Converting Images
    """This operation takes a long time because the image files are so big. 
    The difference in duration for this operation between using and not 
    using HIGH_RES is a factor of about 20, but, again, not using HIGH_RES 
    results in unusable images."""
    try:
        with rasterio.open(file_paths[0]) as src:
            image_metadata = src.meta.copy()
    except Exception as e:
        print("failed raster metadata pull")
        print("error: ", e)
        confirm_continue_or_exit()
    
    image_arrays = image_to_array(file_paths)
    
    if c.CLOUD_MASKING:
        image_arrays_clouds = image_arrays
    
    print("step 1 complete! finished at {dt.datetime.now().time()}")
    return
def two_mask_known_feature():
    return
def three_mask_clouds():
    return
def four_compute_indices():
    return
def five_composite():
    return
def six_prepare_data():
    return
def seven_label_data():
    return
def eight_segment_data():
    return


# %%% i. Find the Relevant Folders
folders_path = os.path.join(c.HOME_DIR, "data", "sentinel_2")
folders = list_folders(folders_path)

# %%% 1. Opening Images and Creating Image Arrays
for i, folder in enumerate(folders):
    print("===============")
    print(f"| IMAGE {i+1} / {len(folders)} |")
    print("===============")
    print("==========")
    print("| STEP 1 |")
    print("==========")
    print("opening images and creating image arrays")
    
    
    
    # %%% 2. Known Feature Masking
    print("==========")
    print("| STEP 2 |")
    print("==========")
    if c.KNOWN_FEATURE_MASKING:
        print("masking out known features")
        
        masking_path = os.path.join(c.HOME_DIR, "data", "masks")
        
        rivers_data = os.path.join(
            masking_path, 
            "rivers", 
            "data", 
            "WatercourseLink.shp"
            )
        
        boundaries_data = os.path.join( # for masking the sea
            masking_path, 
            "boundaries", 
            ("Regions_December_2024_Boundaries_EN_BSC_"
            "-6948965129330885393.geojson")
            )
        
        known_reservoirs_data = os.path.join(
            masking_path, 
            "known reservoirs", 
            "LRR _EW_202307_v1", 
            "SHP", 
            "LRR_ENG_20230601_WGS84.shp" # WGS84 is more accurate than OSGB35
            )
        
        urban_areas_data = os.path.join( # REMEMBER TO CITE SOURCE FROM README
            masking_path, 
            "urban areas", 
            "CEH_GBLandCover_2024_10m", 
            "data", 
            "4dd9df19-8df5-41a0-9829-8f6114e28db1", 
            "gblcm2024_10m.tif"
            )
        
        for i in range(len(image_arrays)):
            image_arrays[i] = known_feature_mask(
                image_arrays[i], 
                image_metadata, 
                rivers_data, 
                feature_type="rivers", 
                buffer_metres=20
                )
            image_arrays[i] = known_feature_mask(
                image_arrays[i], 
                image_metadata, 
                boundaries_data, 
                feature_type="sea"
                )
            image_arrays[i] = known_feature_mask(
                image_arrays[i], 
                image_metadata, 
                known_reservoirs_data, 
                feature_type="known reservoirs", 
                buffer_metres=50
                )
            image_arrays[i] = mask_urban_areas( # different process (.tif)
                image_arrays[i], 
                image_metadata, 
                urban_areas_data
                )
        
        print("step 2 complete! finished at {dt.datetime.now().time()}")
    else:
        print("skipping known feature masking")

    # %%% 3. Masking Clouds
    print("==========")
    print("| STEP 3 |")
    print("==========")
    if c.CLOUD_MASKING:
        if not c.HIGH_RES:
            print(("WARNING: high-resolution setting is disabled. "
            "cloud masking may not be accurate"))
            confirm_continue_or_exit()
        
        print("masking clouds")

        input_array = np.stack((
            image_arrays_clouds[2], # red
            image_arrays_clouds[0], # green
            image_arrays_clouds[1] # nir
            ))
        
        try:
            pred_mask_2d = predict_from_array(input_array, 
                                              mosaic_device="cuda")[0]
        except Exception as e:
            print("WARNING: CUDA call failed, using CPU")
            print("error:", e)
            confirm_continue_or_exit()
            pred_mask_2d = predict_from_array(input_array, 
                                              mosaic_device="cpu")[0]
        
        combined_mask = (
            (pred_mask_2d == 1) | 
            (pred_mask_2d == 2) | 
            (pred_mask_2d == 3)
            )
        
        for i in range(len(image_arrays)):
            # float is used as it supports NaN
            image_arrays[i] = image_arrays[i].astype(np.float32)
            image_arrays[i][combined_mask] = np.nan
        
        print("step 3 complete! finished at {dt.datetime.now().time()}")
    else:
        print("skipping cloud masking")
    
    # %%% 4. Index Calculation
    print("==========")
    print("| STEP 4 |")
    print("==========")
    print("index calculation start")
    
    # %%%% 4.1 Image Array Type Conversion
    print("converting image array types")
    # first convert to int
    # np.uint16 type is bad for algebraic operations!
    for i, image_array in enumerate(image_arrays):
        image_arrays[i] = image_array.astype(np.float32)
    green, nir, red = image_arrays
    
    # %%%% 4.2 Calculating Indices
    print("populating index arrays")
    np.seterr(divide="ignore", invalid="ignore")
    ndwi = (green - nir) / (green + nir)
    ndwi_arrays_list.append(ndwi)
    
    # ndvi = ((nir - red) / (nir + red))
    # ndvi_arrays_list.append(ndvi)
    
    # gain factor g, aerosol resistance coefficient c1 & c2
    # evi_num = g * (nir - red)
    # evi_den = (nir + (c1 * red) - (c2 * blue) + l)
    # evi = evi_num / evi_den
    # evi_arrays_list.append(evi)
    
    # evi2 = 2.4 * (nir - red) / (nir + red + 1) # 2-band evi can be useful
    # evi2_arrays_list.append(evi2)
    
    print("step 4 complete! finished at {dt.datetime.now().time()}")

# %%% 5. Spectral Temporal Metrics
print("==========")
print("| STEP 5 |")
print("==========")
print("temporal image compositing start")

# %%%% 5.1 Preparation for Compositing
ndwi_stack = np.stack(ndwi_arrays_list)
ndwi_mean = np.nanmean(ndwi_stack, axis=0)
globals()["ndwi_mean"] = ndwi_mean
#ndwi_sd = np.nanstd(ndwi_stack, axis=0)

# %%%% 5.2 Compositing Scenes Together
#ndwi_composite = np.stack([ndwi_mean, ndwi_sd], axis=-1)
#globals()["ndwi_comp"] = ndwi_composite

# %%%% 5.3 Displaying Index
if c.SHOW_INDEX_PLOTS:
    if c.SAVE_IMAGES:
        print("saving and displaying water index images")
    else:
        print("displaying water index images")
    plot_indices(ndwi_mean, c.PLOT_SIZE, c.DPI, c.SAVE_IMAGES, 
    folder_path, res)
    print(f"step 5 complete! finished at {dt.datetime.now().time()}")
else:
    print("not displaying water index images")

# %%% 6. Data Preparation
print("==========")
print("| STEP 6 |")
print("==========")
print("data preparation start")

# %%%% 6.1 Preparing True Colour Image
"""Load the TCI file at selected resolution, convert to array, resize for 
GUI labelling."""
if c.LABEL_DATA:
    print(f"opening {res} resolution true colour image")
    
    tci_path = os.path.join(images_path, "IMG_DATA", f"R{res}")
    tci_file_name = prefix + f"_TCI_{res}.jp2"
    tci_array = image_to_array(os.path.join(tci_path, tci_file_name))
    
    tci_60_path = os.path.join(folder_path, "GRANULE", subdirs[0], 
                               "IMG_DATA", "R60m")
    tci_60_file_name = prefix + "_TCI_60m.jp2"
    with Image.open(os.path.join(tci_60_path, tci_60_file_name)) as img:
        size = (img.width//10, img.height//10)
        tci_60_array = np.array(img.resize(size))

# %%%% 6.2 Creating Chunks from Satellite Imagery
"""Split the NDWI array into N_CHUNKS equal segments for batch ROI 
labelling and parallel processing."""
print(f"creating {c.N_CHUNKS} chunks from satellite imagery")
index_chunks = split_array(array=ndwi_mean, n_chunks=c.N_CHUNKS)
global_max = 0
if c.LABEL_DATA:
    tci_chunks = split_array(array=tci_array, n_chunks=c.N_CHUNKS)

# %%%% 6.3 Preparing File for Labelling
"""Initialise or validate the CSV file by enforcing a header, removing 
blank lines, and ensuring sequential chunk indices"""
break_flag = False

# check for a responses folder
data_folder_found = False
for folder in folders:
    if data_folder_found:
        break
    
    folder_path = os.path.join(c.HOME_DIR, "data", "sentinel_2", folder)
    if os.path.exists(os.path.join(folder_path, "training data")):
        data_folder_found = True
        labelling_path = os.path.join(folder_path, "training data")

if not data_folder_found:
    labelling_path = os.path.join(folder_path, "training data")
    change_to_folder(labelling_path) # create the folder
    os.chdir(c.HOME_DIR) # always go back to initial home folder
    # THIS IS BAD!! PLEASE FIND A WAY TO CHANGE THIS!!

lines = []
header = ("chunk,reservoirs,water-bodies,reservoir-"
"coordinates,,,,,water-body-coordinates\n")
data_file = os.path.join(labelling_path, c.DATA_FILE_NAME)
blank_entry_check(file=data_file) # remove all blank entries

# %%%%% 6.3.1 File validity check
"""This section is about checking that the contents of the file are sound. 
This means checking that, for example, the file exists, or that the entry 
for chunk 241 is after the entry for chunk 240, or any other problem that 
may arise. As with every data handling operation, file permission is 
checked directly before this step, as well as any blank entries. Although 
this may seem an excessive step to perform every time, it is necessary to 
ensure that the file data is exactly as it should be. 
All these checks are carried out in 'read-only' mode unless the user 
specifies otherwise. This is to make sure that the data is not accidentally 
overwritten at any point, again, unless the user is sure this is the 
intended behaviour."""
print("preparing file for labelling")
while True:
    # file will always exist due to blank_entry_check call
    with open(data_file, "r") as file:
        lines = file.readlines()
    try:
        # Validate file data
        for i in range(1, len(lines) - 1):
            current_chunk = int(lines[i].split(",")[0])
            next_chunk = int(lines[i+1].split(",")[0])
            if next_chunk - current_chunk != 1:
                print(f"error in line {i + 2}, "
                      f"expected chunk {current_chunk + 1}")
                raise ValueError("File validity error")
        last_chunk = int(lines[-1].split(",")[0])
        break
    except (ValueError, IndexError) as e:
        print(f"error - file with invalid data: {e}")
        print("type 'quit' to exit, or 'new' for a fresh file")
        ans = input("press enter to retry: ").strip().lower()
        if ans.lower() == 'quit':
            print("quitting")
            #return ndwi_mean
        if ans.lower() == 'new':
            print("creating new file")
            with open(data_file, "w") as file:
                file.write(header)
                file.write("0, 1, 0\n") # dummy file to start up
            continue

i = last_chunk + 1 # from this point on, "i" is off-limits as a counter

# %%%%% 6.3.2 Data completion check
"""Once file validity has been verified, this step is for ensuring that 
the data in the file is complete. While the previous step (6.3.1) was 
mostly intended for checking that the chunks are in the correct order, 
this step additionally checks that the chunks that are supposed to have 
data, i.e. a chunk is noted as containing a reservoir, that there are 
coordinates outlining that reservoir. If this is not the case, the 
'data_correction' mode is activated, in which the user is prompted to 
essentially fill in the coordinates that should exist in the place where 
a chunk is supposed to contain some water body."""
# find chunks with invalid or incomplete reservoir coordinate data
reservoir_rows = []
body_rows = []
invalid_rows = []
data_correction = False

with open(data_file, "r") as file:
    lines = file.readlines() # reread lines in case of changes
    globals()["lines"] = lines
for j in range(1, len(lines)): # starting from the "headers" line
    # check for reservoirs without coordinates
    num_of_reservoirs = int(lines[j].split(",")[1])
    try: # try to access coordinates
        res_coord = lines[j].split(",")[3]
        res_has_coords = bool(res_coord[0] == "[")
    except: # if unable to access, they do not exist
        res_has_coords = False
    if num_of_reservoirs != 0 and not res_has_coords:
        reservoir_rows.append(j-1)
        data_correction = True
    elif num_of_reservoirs == 0 and res_has_coords:
        invalid_rows.append(j-1)
        data_correction = True
    
    # check for non-reservoir water bodies without coordinates
    num_of_bodies = int(lines[j].split(",")[2])
    try: # try to access coordinates
        body_coord = lines[j].split(",")[8]
        body_has_coords = bool(body_coord[0] == "[")
    except: # if unable to access, they do not exist
        body_has_coords = False
    if num_of_bodies != 0 and not body_has_coords:
        body_rows.append(j-1)
        data_correction = True
    elif num_of_bodies == 0 and body_has_coords:
        invalid_rows.append(j-1)
        data_correction = True
invalid_rows = combine_sort_unique(reservoir_rows, body_rows, invalid_rows)

if data_correction:
    print(f"found {len(invalid_rows)} chunks containing "
           "incomplete, missing, or incorrect coordinate data")
    i = invalid_rows[0]
    invalid_rows_index = 0
print(f"step 6 complete! finished at {dt.datetime.now().time()}")

# %%% 7. Data Labelling
print("==========")
print("| STEP 7 |")
print("==========")
if not c.LABEL_DATA:
    print("not labelling data")
else:
    print("data labelling start")
    
    # %%%% 7.1 Outputting Images
    print("outputting images...")
    
    while i < len(index_chunks):
        if break_flag:
            break
        plot_chunks(ndwi_mean, index_chunks, c.PLOT_SIZE_CHUNKS, i, 
                    c.TITLE_SIZE, c.LABEL_SIZE, tci_chunks, tci_60_array)
        max_index = [0, 0]
        max_index[0] = round(np.nanmax(index_chunks[i]), 2)
        print(f"MAX ADJUSTED NDWI: {max_index[0]}", end=" | ")
        max_index[1] = round(np.nanmax(index_chunks[i]), 2)
        print(f"MAX NDWI: {max_index[1]}")
        
        # %%%% 7.2 User Labelling
        blank_entry_check(file=data_file)
        if data_correction:
            print((
                "this chunk "
                f"({invalid_rows_index+1}/{len(invalid_rows)})"
                " should contain "
                f"{int(lines[i+1].split(',')[1])} reservoirs and "
                f"{int(lines[i+1].split(',')[2])} non-reservoir "
                "water bodies"
                ))
        n_reservoirs = input("how many reservoirs? ").strip().lower()
        n_bodies = ""
        entry_list = []
        while True:
            blank_entry_check(file=data_file)
            back_flag = False
            try:
                # %%%%% 7.2.1 Regular integer response
                """Parse integer input for reservoirs and bodies, enfore 
                maximum limits, and prompt ROI drawing and collect 
                coordinates."""
                # handle number of reservoirs entry
                n_reservoirs = int(n_reservoirs)
                entry_list = [i,n_reservoirs,""]
                while n_reservoirs > 5: # NOTE add user input type check
                    print("maximum of 5 reservoirs")
                    n_reservoirs = input("how many "
                                         "reservoirs? ").strip().lower()
                if n_reservoirs != 0:
                    print("please draw a square around the reservoir(s)", 
                          flush=True)
                    chunk_coords = prompt_roi(tci_chunks[i], n_reservoirs)
                    for coord in chunk_coords:
                        entry_list.append(coord)
                while len(entry_list) < 8:
                    entry_list.append("")
                 
                # handle number of non-reservoir water bodies entry
                n_bodies = input("how many non-reservoir "
                                 "water bodies? ").strip().lower()
                n_bodies = int(n_bodies)
                entry_list[2] = n_bodies
                if n_bodies != 0:
                    print("please draw a square around the water bodies", 
                          flush=True)
                    chunk_coords = prompt_roi(tci_chunks[i], n_bodies)
                    for coord in chunk_coords:
                        entry_list.append(coord)
                i += 1
                print("generating next chunk...", flush=True)
                break # exit loop and continue to next chunk
            
            # handle non-integer responses
            except:
                n_reservoirs = str(n_reservoirs)
                n_bodies = str(n_bodies)
                if "break" in n_bodies or "break" in n_reservoirs:
                    # %%%%% 7.2.2 Non-integer response: "break"
                    """nico!! remember to add a description!"""
                    print("taking a break")
                    break_flag = True
                    break
                if "back" in n_bodies or "back" in n_reservoirs:
                    # %%%%% 7.2.3 Non-integer response: "back"
                    """nico!! remember to add a description!"""
                    back_flag = True
                    if data_correction:
                        print("cannot use 'back' during data correction")
                        break
                    try:
                        n_backs = int(n_reservoirs.split(" ")[1])
                    except:
                        n_backs = 1
                    i -= n_backs
                    check_file_permission(file_name=data_file)
                    with open(data_file, mode="r") as re: # read
                        rows = list(csv.reader(re))
                    for j in range(n_backs):
                        rows.pop() # remove the last "n_backs" rows
                    with open(data_file, mode="w") as wr: # write
                        rewrite(write_file=wr, rows=rows)
                    break
                # %%%%% 7.2.4 Non-integer response: error
                """nico!! remember to add a description!"""
                print("error: non-integer response."
                      "\ntype 'break' to save and quit"
                      "\ntype 'back' to go to previous chunk")
                n_reservoirs = input("how many "
                                     "reservoirs? ").strip().lower()
        
        # %%%% 7.3 Saving Results
        """nico!! remember to add a description!"""
        if break_flag:
            break
        if not break_flag and not back_flag:
            check_file_permission(file_name=data_file)
            csv_entry = ""
            first_csv_entry = True
            for entry in entry_list:
                if first_csv_entry:
                    csv_entry = f"{entry}"
                elif not first_csv_entry:
                    csv_entry = f"{csv_entry},{entry}"
                first_csv_entry = False
            if data_correction: # add coordinates to data
                lines[i] = f"{csv_entry}\n"
                with open(data_file, mode="w") as wr: # write
                    for j in range(len(lines)):
                        current_entry = lines[j]
                        wr.write(f"{current_entry}")
                invalid_rows_index += 1
                if invalid_rows_index >= len(invalid_rows):
                    i = last_chunk + 1
                    data_correction = False
                else:
                    i = invalid_rows[invalid_rows_index]
            else: # convert entry_list to a string for csv
                with open(data_file, mode="a") as ap: # append
                    ap.write(f"\n{csv_entry}")
    print(f"step 7 complete! finished at {dt.datetime.now().time()}")

# %%% 8. Data Segmentation
print("==========")
print("| STEP 8 |")
print("==========")

# %%%% 8.1 Extract Reservoir and Water Body Coordinates
"""nico!! remember to add a description!"""
if not c.HIGH_RES:
    print("high resolution setting must be activated for data segmentation")
    print("exiting program")
    #return ndwi_mean
print("data segmentation start")

res_rows = []
res_coords = []

body_rows = []
body_coords = []

land_rows = []
land_coords = []
none_coord = "[50.0 50.0 55.0 55.0]" # replicate res and bod coords format
# allows it to be passed as an argument of extract_coords

sea_rows = []
sea_coords = []

with open(data_file, "r") as file:
    lines = file.readlines()

for i in range(1, len(lines)):
    lines[i] = lines[i].split(",")
    if int(lines[i][1]) > 0: # if there is a reservoir
        res_rows.append(lines[i])
        if int(res_rows[-1][1]) > 1:
            for j in range(3, 3+int(res_rows[-1][1])):
                res_coords.append((i, extract_coords(res_rows[-1][j], 
                                                     create_box_flag=True)))
        elif int(res_rows[-1][1]) == 1:
            res_coords.append((i, extract_coords(res_rows[-1][3], 
                                                 create_box_flag=True)))
    
    # if there is a water body
    if int(lines[i][2]) > 0:
        body_rows.append(lines[i])
        first_coords = extract_coords(body_rows[-1][8], 
                                      create_box_flag=False)
        # and the water body is not the sea
        if first_coords[0] != 0 and first_coords[-1] != 157:
            if int(body_rows[-1][2]) > 1:
                for j in range(8, 8+int(body_rows[-1][2])):
                    this_coord = extract_coords(body_rows[-1][j], 
                                                create_box_flag=True)
                    body_coords.append((i, this_coord))
            elif int(body_rows[-1][2]) == 1:
                this_coord = extract_coords(body_rows[-1][8], 
                                              create_box_flag=True)
                body_coords.append((i, this_coord))
        else:# if it IS the sea, save a minichunk of it too
            sea_rows.append(lines[i])
            sea_coords.append((i, extract_coords(none_coord, 
                                              create_box_flag=True)))
    
    # if it's just land, save a minichunk of it too
    if int(lines[i][1]) == 0 and int(lines[i][2]) == 0:
        land_rows.append(lines[i])
        land_coords.append((i, extract_coords(none_coord, 
                                              create_box_flag=True)))
    
# %%%% 8.2 Isolate and Save an Image of Each Reservoir and Water Body
"""nico!! remember to add a description! 0.4*max to bring down the ceiling 
of ndwi so that reservoir and water bodies are better highlighted"""
valid_chunks = [chunk
                for chunk in index_chunks
                if not np.all(np.isnan(chunk))]
if valid_chunks:
    global_min = min(np.nanmin(chunk) for chunk in valid_chunks)
    global_max = 0.4*max(np.nanmax(chunk) for chunk in valid_chunks)
else:
    global_min = np.nan
    print("Warning: All NDWI chunks contained only NaN values.")

# %%%% 8.3 Create a Single TFRecord File to Store Training Data
class_names = ["reservoirs", "water bodies", "land", "sea"]
class_map = {name: i for i, name in enumerate(class_names)}

tfrecord_filename = "training_data.tfrecord"

# check for file name already existing and increment file name
counter = 1
base_name, extension = tfrecord_filename.split(".")
while os.path.exists(os.path.join(labelling_path, tfrecord_filename)):
    counter += 1
    tfrecord_filename = f"{base_name}_{counter}.{extension}"

tfrecord_file_location = os.path.join(labelling_path, tfrecord_filename)
with tf.io.TFRecordWriter(tfrecord_file_location) as tf_writer:
    all_coords = [(res_coords, "reservoirs"), 
                  (body_coords, "water bodies"), 
                  (land_coords, "land"), 
                  (sea_coords, "sea")]
    
    for coords_list, class_name in all_coords:
        print(f"{len(coords_list)} examples for class {class_name}")
        class_index = class_map[class_name]
        
        for i in range(len(coords_list)):
            chunk_n = int(coords_list[i][0]) - 1
            coordinates = coords_list[i][1]
            
            rgb_data = get_rgb_data(
                data=valid_chunks, 
                chunk_n=chunk_n, 
                coordinates=coordinates, 
                g_min=global_min, g_max=global_max)
            
            tf_example = create_tf_example(
                image_array=rgb_data, 
                class_index=class_index, 
                class_name_str=class_name)
            
            tf_writer.write(tf_example.SerializeToString())

existing_records = []
for path in os.listdir(labelling_path):
    if ".tfrecord" in path:
        existing_records.append(path)

hash1 = hash_tfrecord(tfrecord_file_location)
for record in existing_records:
    hash2 = hash_tfrecord(os.path.join(labelling_path, record))
    duplicate = bool(hash1 == hash2)
    if duplicate:
        print("found duplicate tf record data")
        try:
            new_save_location = os.path.join(labelling_path, 
                                             f"DUPLICATE_{record}")
            os.rename(tfrecord_file_location, new_save_location)
            print(f"tf record file renamed to DUPLICATE_{record}")
            tfrecord_filename = f"DUPLICATE_{record}"
            tfrecord_file_location = os.path.join(labelling_path, 
                                                  tfrecord_filename)
        except FileNotFoundError:
            print("tf record file not found")
        except PermissionError:
            print("permission denied. unable to rename the tf record file")
        break

print(f"wrote all available training data to {tfrecord_filename}")

print(f"step 8 complete! finished at {dt.datetime.now().time()}")
