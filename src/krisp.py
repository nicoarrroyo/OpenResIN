# %% 0. Start
""" KRISP
Keras Reservoir Identification Sequential Platform
"""
# %%% i. Import External Libraries
import time
MAIN_START_TIME = time.monotonic()
import os
import numpy as np
import re # "regular expressions" for parsing filenames
import sys
import math

from tensorflow import keras
import tensorflow as tf
import rasterio
from omnicloudmask import predict_from_array

# %%% ii. Import Internal Functions
from data_handling import change_to_folder, extract_chunk_details
from data_handling import sort_prediction_results, sort_file_names
from data_handling import check_positive_int

from image_handling import image_to_array, plot_indices, save_image_file
from image_handling import known_feature_mask, mask_urban_areas

from misc import get_sentinel_bands, split_array

from user_interfacing import start_spinner, end_spinner, list_folders
from user_interfacing import confirm_continue_or_exit

# %%% Directory, Plot, and Model Configuration Properties
dpi = 3000 # 3000 for full resolution, below 1000, images become fuzzy
n_chunks = 5000 # number of chunks into which images are split
high_res = True # use finer 10m spatial resolution (slower)
cloud_masking = True
show_index_plots = True
save_images = False
response_time = 0.0

title_size = 8
label_size = 4
plot_size = (5, 5) # larger plots increase detail and pixel count
plot_size_chunks = (6, 6)

HOME = os.path.dirname(os.getcwd()) # HOME path is one level up from the cwd

class_names = ["land", "reservoirs", "sea", "water bodies"]
batch_size = 512

# %% Big guy
def run_model(folder, n_chunks, model_name, max_multiplier, 
              start_chunk, n_chunk_preds):
    
    start_chunk = check_positive_int(
        var=start_chunk, 
        description="chunk to start on")
    
    n_chunk_preds = check_positive_int(
        var=n_chunk_preds, 
        description="number of chunks to make predictions on")
    
    n_files = n_chunk_preds * 25
    start_file = start_chunk * 25
    # %%% 0. Check for Pre-existing Files
    print("==========")
    print("| STEP 0 |")
    print("==========")
    stop_event, thread = start_spinner(message="checking for "
                                       "pre-existing files")
    start_time = time.monotonic()
    satellite = "Sentinel 2"
    path = os.path.join(HOME, "data", satellite, folder)
    
    # %%%% 0.1 Chunk Check!
    test_data_path = os.path.join(path, "test data", f"ndwi_{max_multiplier}")
    existing_files = []
    real_n_chunks = math.floor(math.sqrt(n_chunks)) ** 2 - 1
    n_mini_chunks = 25
    mc_per_len = int(np.sqrt(n_mini_chunks)) # mini-chunks per length
    # important note! ensure this matches the IMG_HEIGHT division in trainer
    # as well as the BOX_SIZE division in data_handling
    generate_chunks = False
    
    # %%%%% 0.1.1 Extract current folder contents
    try:
        all_items = os.listdir(test_data_path)
        # Filter for files only, ignore subdirectories
        for item in all_items:
            if os.path.isfile(os.path.join(test_data_path, item)):
                existing_files.append(item)
        existing_files[1] # try to access any item
        # this will induce an intended error in the event that a directory 
        # does exist but it is unpopulated
    except:
        end_spinner(stop_event, thread)
        while True:
            print("test data directory does not exist")
            print("WARNING: creating and saving many images may take " 
                  "very long! The console may freeze of crash, but "
                  "progress should continue regardless")
            print("to check progress, go to the download directory.")
            user_input = input("do you want to recalculate NDWI and fill in "
                               "the remaining chunks? this may add/overwrite "
                               "files (y/n): ").strip().lower()
            if user_input in ["y", "yes"]:
                generate_chunks = True
                print("starting chunk generation process")
                # create the directory
                change_to_folder(test_data_path)
                os.chdir(HOME)
                break
            elif user_input in ["n", "no"]:
                generate_chunks = False
                print("without chunks, the script cannot run")
                sys.exit(0)
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
    
    # Proceed only if the directory was accessible
    if len(existing_files) > 0:
    
        # %%%%% 0.1.2 Find latest saved chunk
        max_chunk_index = -1
        start_chunk_index = 0
        
        filename_pattern = re.compile(r"ndwi chunk (\d+) minichunk \d+\.png")
        
        for filename in existing_files:
            match = filename_pattern.match(filename)
            if match:
                # Extract the captured chunk index (group 1) and convert to int
                chunk_index = int(match.group(1))
                max_chunk_index = max(max_chunk_index, chunk_index)
        end_spinner(stop_event, thread)
        print(f"Target directory contains {len(existing_files)} file(s).")
        if max_chunk_index != -1:
            start_chunk_index = max_chunk_index + 1
            chunks_rem = real_n_chunks - max_chunk_index
            percent_rem = round(100 * (chunks_rem / real_n_chunks), 2)
            if chunks_rem > 0:
                print(f"{max_chunk_index} chunks saved in "
                      f"'ndwi_{max_multiplier}'.")
                print(f"{chunks_rem} chunks remaining, "
                      f"{percent_rem}% remaining")
        else:
            print("No files matching the 'ndwi chunk i minichunk j.png' "
                  f"pattern found in 'ndwi_{max_multiplier}'.")
            chunks_rem = real_n_chunks - max_chunk_index
        
        # %%%%% 0.1.3 Ask user if they want to continue
        while True:
            if chunks_rem > 0:
                print("WARNING: creating and saving many images may take " 
                      "very long! The console may freeze or crash, but "
                      "progress should continue regardless")
                print("to check progress, go to the download directory.")
                user_input = input("do you want to recalculate NDWI and fill "
                                   f"in the remaining {chunks_rem} chunks? "
                                   "this may add/overwrite files "
                                   "(y/n): ").strip().lower()
                if user_input in ["y", "yes"]:
                    generate_chunks = True
                    break
                elif user_input in ["n", "no"]:
                    generate_chunks = False
                    break
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")
            else:
                print("disabling chunk generation")
                generate_chunks = False
                break
    
    end_spinner(stop_event, thread)
    time_taken = time.monotonic() - start_time
    print(f"step 0 complete! time taken: {time_taken:.2f} seconds")

    # %%% 1. Load Sentinel 2 Image File
    if generate_chunks:
        ndwi_arrays_list = []
        # ndvi_arrays_list = []
        # evi_arrays_list = []
        # evi2_arrays_list = []
        global response_time
        
        # %%% i. Find the Relevant Folders
        folders_path = os.path.join(HOME, "data", satellite)
        folders = list_folders(folders_path)
        # %%% 1. Opening Images and Creating Image Arrays
        for folder in folders:
            print("==========")
            print("| STEP 1 |")
            print("==========")
            print("opening images and creating image arrays")
            start_time = time.monotonic()
            
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
                response_time_start = time.monotonic()
                confirm_continue_or_exit()
                response_time += time.monotonic() - response_time_start
                continue
            
            # %%%%% 1.1.2 Resolution selection and file name deconstruction
            """Low resolution should only be used for troubleshooting as it does 
            not produce usable training data. High resolution uses the 10m spatial 
            resolution images but processing time is significantly longer."""
            if high_res:
                res = "10m"
                path_10 = os.path.join(images_path, "IMG_DATA", "R10m")
            else:
                res = "60m"
                path_60 = os.path.join(images_path, "IMG_DATA", "R60m")
            
            (sentinel_name, instrument_and_product_level, 
            datatake_start_sensing_time, processing_baseline_number, 
            relative_orbit_number, tile_number_field, 
            product_discriminator_and_format) = folder.split("_")
            
            prefix = (f"{tile_number_field}_{datatake_start_sensing_time}")
            bands = get_sentinel_bands(sentinel_n=2, high_res=high_res)
            
            for band in bands:
                if high_res:
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
            using high_res is a factor of about 20, but, again, not using high_res 
            results in unusable images."""
            try:
                with rasterio.open(file_paths[0]) as src:
                    image_metadata = src.meta.copy()
            except:
                print("failed raster metadata pull")
                response_time_start = time.monotonic()
                confirm_continue_or_exit()
                response_time += time.monotonic() - response_time_start
            
            image_arrays = image_to_array(file_paths)
            
            if cloud_masking:
                image_arrays_clouds = image_arrays
            
            time_taken = time.monotonic() - start_time
            print(f"step 1 complete! time taken: {time_taken:.2f} seconds")
            
            # %%% 2. Known Feature Masking
            print("==========")
            print("| STEP 2 |")
            print("==========")
            print("masking out known features")
            start_time = time.monotonic()
            
            masking_path = os.path.join(HOME, "data", "Masking")
            
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
                    buffer_metres=50
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
            
            time_taken = time.monotonic() - start_time
            print(f"step 2 complete! time taken: {time_taken:.2f} seconds")
        
            # %%% 3. Masking Clouds
            print("==========")
            print("| STEP 3 |")
            print("==========")
            if cloud_masking:
                if not high_res:
                    print(("WARNING: high-resolution setting is disabled. "
                    "cloud masking may not be accurate"))
                    response_time_start = time.monotonic()
                    confirm_continue_or_exit()
                    response_time += time.monotonic() - response_time_start
                
                print("masking clouds")
                start_time = time.monotonic()

                input_array = np.stack((
                    image_arrays_clouds[2], # red
                    image_arrays_clouds[0], # green
                    image_arrays_clouds[1] # nir
                    ))
                
                try:
                    pred_mask_2d = predict_from_array(input_array, 
                                                      mosaic_device="cuda")[0]
                except:
                    print("WARNING: CUDA call failed, using CPU")
                    response_time_start = time.monotonic()
                    confirm_continue_or_exit()
                    response_time += time.monotonic() - response_time_start
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
                
                time_taken = time.monotonic() - start_time
                print("step 3 complete! time taken: "
                f"{time_taken:.2f} seconds")
            else:
                print("skipping cloud masking")
            
            # %%% 4. Index Calculation
            print("==========")
            print("| STEP 4 |")
            print("==========")
            start_time = time.monotonic()
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
            ndwi = ((green - nir) / (green + nir))
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
            
            time_taken = time.monotonic() - start_time
            print(f"step 4 complete! time taken: {time_taken:.2f} seconds")
        
        # %%% 5. Spectral Temporal Metrics
        print("==========")
        print("| STEP 5 |")
        print("==========")
        start_time = time.monotonic()
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
        if show_index_plots:
            if save_images:
                print("saving and displaying water index images")
            else:
                print("displaying water index images")
            start_time = time.monotonic()
            plot_indices(ndwi_mean, plot_size, dpi, save_images, 
            folder_path, res)
            time_taken = time.monotonic() - start_time
            print(f"step 5 complete! time taken: {time_taken:.2f} seconds")
        else:
            print("not displaying water index images")
        
    # %%% 4. Save Satellite Image Chunks
        """nico!! remember to add a description!"""
        print("==========")
        print("| STEP 6 |")
        print("==========")
        # %%%% 6.1 Create Chunks
        stop_event, thread = start_spinner(message=f"creating {n_chunks} "
                                           "chunks from satellite imagery")
        start_time = time.monotonic()
        
        ndwi_chunks = split_array(array=ndwi_mean, n_chunks=n_chunks)
        global_min = min(np.nanmin(chunk) for chunk in ndwi_chunks)
        global_max = max_multiplier*max(np.nanmax(chunk) for \
                                        chunk in ndwi_chunks)
        
        end_spinner(stop_event, thread)
        
        # %%%% 6.2 Create and Save Mini-Chunks
        print("saving chunks as image files")
        change_to_folder(test_data_path)
                
        for i, chunk in enumerate(ndwi_chunks):
            if i > real_n_chunks:
                print("WARNING: Exceeded expected number of chunks "
                      f"({real_n_chunks}). Stopping.")
                break
            if not generate_chunks and i < start_chunk_index:
                continue # Skip chunks already processed
            
            chunk_height, chunk_width = chunk.shape
            mini_chunk_h = chunk_height / mc_per_len
            mini_chunk_w = chunk_width / mc_per_len
            
            uly_s = np.linspace(0, chunk_height - mini_chunk_h, mc_per_len)
            ulx_s = np.linspace(0, chunk_width - mini_chunk_w, mc_per_len)
            
            mc_idx = 0 # mini-chunk index
            for j, ulx in enumerate(ulx_s):
                for k, uly in enumerate(uly_s):
                    image_name = (f"ndwi chunk {i} minichunk {mc_idx}.png")
                    mini_chunk_coord = [
                        float(ulx),                 # ulx
                        float(uly),                 # uly
                        float(ulx + mini_chunk_w),  # lrx
                        float(uly + mini_chunk_h)   # lry
                    ]
                    save_image_file(data=chunk, 
                                    image_name=image_name, 
                                    normalise=True, 
                                    coordinates=mini_chunk_coord, 
                                    g_max=global_max, g_min=global_min, 
                                    dupe_check=False)
                    mc_idx += 1
        time_taken = time.monotonic() - start_time
        print(f"step 6 complete! time taken: {time_taken:.2f} seconds")
    else:
        print("============")
        print("| STEP 1-5 |")
        print("============")
        print("chunk generation disabled, skipping steps 1-4")
    # %%% 7. Load and Deploy Model
    print("==========")
    print("| STEP 7 |")
    print("==========")
    # %%%% 7.1 Load Essential Info & Prepare File List
    print("loading model and preparing file list")
    start_time = time.monotonic()
    results_list = []
    
    height = int(157 / mc_per_len)
    width = int(157 / mc_per_len)
    
    models_path = os.path.join(HOME, "data", "saved_models")
    model_names = os.listdir(models_path)
    found_model = False
    for name in model_names:
        if model_name in name:
            model_path = os.path.join(models_path, model_name)
            found_model = True
            break
    
    if not found_model:
        print("could not find the requested model")
        print("these are the options:")
        for i in range(len(model_names)):
            print(f"[{i}]: {model_names[i]}")
        while True:
            model_choice = input("please select the index of the"
                              " model you would like to use: ")
            try:
                model_choice = int(model_choice)
                if model_choice > len(model_names) or model_choice < 0:
                    print("not a possible model number")
                else:
                    model_name = model_names[model_choice]
                    model_path = os.path.join(models_path, model_name)
                    break
            except Exception as e:
                print(e)
                print("please input an integer")
        
    model = keras.models.load_model(model_path)
    
    all_file_names = os.listdir(test_data_path)
    all_file_names = sort_file_names(all_file_names)
    
    selected_file_names = all_file_names[start_file:(start_file+n_files)]
    del all_file_names # save memory
        
    # %%%% 7.2 Make Predictions using Batch Processing
    stop_event, thread = start_spinner(message="preparing for predictions on "
                               f"{n_files} files "
                               f"({n_chunk_preds} chunks)")
    # --- Create tf.data Pipeline ---
    all_file_paths = [
        os.path.join(test_data_path, fname) 
        for fname in selected_file_names
        ]
    path_ds = tf.data.Dataset.from_tensor_slices(all_file_paths)
    
    # Define the loading/preprocessing function
    def load_and_preprocess_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [height, width])
        return img, path # Return path as well to identify errors
    
    # Map, batch, prefetch
    image_ds = path_ds.map(load_and_preprocess_image, 
                           num_parallel_calls=tf.data.AUTOTUNE)
    image_batch_ds = image_ds.batch(batch_size)
    image_batch_ds = image_batch_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    end_spinner(stop_event, thread)
    
    # --- Run Prediction ---
    # Pass only the image tensor part of the dataset to predict
    all_predictions = model.predict(
        image_batch_ds.map(lambda img, path: img), 
        verbose=1
        )
    
    # --- Process Results ---
    stop_event, thread = start_spinner(message="processing results")
    filename_pattern = re.compile(r"chunk\s+(\d+)\s+minichunk\s+(\d+)")
    for i, prediction in enumerate(all_predictions):
        file_name = selected_file_names[i] # Get corresponding filename

        # Apply softmax to get probabilities because the model outputs logits
        score = tf.nn.softmax(prediction)
        predicted_class_index = np.argmax(score)
        predicted_class_name = class_names[predicted_class_index].upper()

        confidence = (100 * np.max(score)).astype(np.float32)

        # Parse filename using optimized function
        file_name = selected_file_names[i]
        chunk_num, minichunk_num = extract_chunk_details(file_name, 
                                                         filename_pattern)
        
        result = [chunk_num, minichunk_num, predicted_class_name, confidence]
        results_list.append(result)
    end_spinner(stop_event, thread)
    
    time_taken = time.monotonic() - start_time
    print(f"step 7 complete! time taken: {round(time_taken, 2)} seconds")
    
    # %%% 8. Return
    sorted_results_list = sort_prediction_results(results_list)
    return sorted_results_list

# %% Run the big guy
if __name__ == "__main__":
    results = run_model(
        folder=("S2C_MSIL2A_20250301T111031_N0511_R137_"
                "T31UCU_20250301T152054.SAFE"), 
        n_chunks=5000, # number of chunks to split the image into
        model_name="ndwi model epochs-1000.keras", 
        max_multiplier=0.41, # multiply max value of ndwi
        start_chunk=0, 
        n_chunk_preds=1000
        )
    
    # %% Final
    TOTAL_TIME = time.monotonic() - MAIN_START_TIME
    print(f"total processing time: {round(TOTAL_TIME, 2)} seconds", flush=True)
