# %% Standard Libraries
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import csv
import datetime as dt
import sys
import time

# %% Common Third Party Libraries
import numpy as np
from PIL import Image

# %% Local Libraries
import data_handling as data_do
import image_handling as image_do
import misc
import user_interfacing as ui_do

import nalira_config as c

# %% 1. Creating image arrays (iterative)
def one_create_image_arrays(folders_path, folder, tci_60_array):
    """
    Navigates Sentinel-2 directory structures to locate, open, and convert 
    image files into arrays.
    
    Parses folder naming conventions to locate the 'GRANULE' directory. 
    Extracts high-resolution (10m) band images and True Colour Images (TCI), 
    converting them into NumPy arrays. Low resolution is only used for 
    troubleshooting. TCI arrays are also resized to aid GUI labelling.
    
    Args:
        folders_path (str): The base directory containing the satellite 
        imagery folders.
        folder (str): The specific Sentinel-2 folder name being processed.
        tci_60_array (np.ndarray): An existing 60m resolution TCI array, or an 
        empty array.
        
    Returns:
        list: Contains [image_arrays, image_metadata, 
                        images_path, tci_array, tci_60_array].
    """
    print(f"step 1 beginning at {dt.datetime.now().time():%H:%M:%S}")
    
    # 1.1 Establishing Paths
    file_paths = []
    folder_path = os.path.join(folders_path, folder)
    images_path = os.path.join(folder_path, "GRANULE")
    
    # 1.1.1 Subfolder iterative search
    subdirs = [d for d in os.listdir(images_path) 
               if os.path.isdir(os.path.join(images_path, d))]
    if len(subdirs) == 1:
        images_path = os.path.join(images_path, subdirs[0], "IMG_DATA")
    else:
        print("Too many subdirectories in 'GRANULE':", len(subdirs))
        ui_do.confirm_continue_or_exit()
    
    # 1.1.2 Resolution selection and file name deconstruction
    (_, _, datatake_start_sensing_time, _, _, 
     tile_number_field, _) = folder.split("_")
    
    prefix = f"{tile_number_field}_{datatake_start_sensing_time}"
    bands = data_do.get_sen2_bands(c.HIGH_RES)
    
    for band in bands:
        file_paths.append(
            os.path.join(
                images_path, 
                f"R{c.RES}", 
                f"{prefix}_B{band}_{c.RES}.jp2"
                )
            )
    
    # 1.2.1 Opening and Converting Band Images
    image_metadata = "TEMPORARY VALUE"
    if c.KNOWN_FEATURE_MASKING:
        import rasterio
        try:
            with rasterio.open(file_paths[0]) as src:
                image_metadata = src.meta.copy()
        except Exception as e:
            print("failed raster metadata pull")
            print("error: ", e)
            ui_do.confirm_continue_or_exit()
    
    
    print(f"opening {c.RES} resolution band images")
    image_arrays = image_do.image_to_array(file_paths)
    
    # 1.2.2 Opening and Converting True Colour Images
    if not tci_60_array.any():
        tci_array = np.empty([1,1])
        tci_60_array = np.empty([1,1])
    if c.LABEL_DATA:
        print(f"opening {c.RES} resolution true colour image")
        
        tci_path = os.path.join(images_path, f"R{c.RES}")
        tci_file_name = prefix + f"_TCI_{c.RES}.jp2"
        tci_array = image_do.image_to_array(os.path.join(tci_path, tci_file_name))
        
        tci_60_path = os.path.join(images_path, f"R{c.RES}")
        tci_60_file_name = prefix + f"_TCI_{c.RES}.jp2"
        with Image.open(os.path.join(tci_60_path, tci_60_file_name)) as img:
            size = (img.width//10, img.height//10)
            tci_60_array = np.array(img.resize(size))
    
    print(f"step 1 complete! finished at {dt.datetime.now().time():%H:%M:%S}")
    return [image_arrays, 
            image_metadata, 
            images_path, 
            tci_array, 
            tci_60_array]

# %% 2. Masking out known features (iterative)
def two_mask_known_feature(image_arrays, image_metadata):
    """
    Applies geographic masks to the image arrays to remove known features.
    
    Uses shapefiles and GeoTIFFs to mask out rivers, the sea, known reservoirs, 
    and urban areas from the satellite imagery. This prevents these features 
    from causing false positives during the water body identification training.
    
    Args:
        image_arrays (list): List of image band arrays (Green, NIR, Red).
        image_metadata (dict): Metadata associated with the rasterized images.
        
    Returns:
        list: The modified image_arrays with known features masked out.
    """
    print(f"step 2 beginning at {dt.datetime.now().time():%H:%M:%S}")
    print("masking out known features")
    
    masking_path = os.path.join(c.DATA_DIR, "masks")
    
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
        "known-reservoirs", 
        "LRR_EW_202307_v1", 
        "SHP", 
        "LRR_ENG_20230601_WGS84.shp" # WGS84 is more accurate than OSGB35
        )
    urban_areas_data = os.path.join( # REMEMBER TO CITE SOURCE FROM README
        masking_path, 
        "urban-areas", 
        "CEH_GBLandCover_2024_10m", 
        "data", 
        "4dd9df19-8df5-41a0-9829-8f6114e28db1", 
        "gblcm2024_10m.tif"
        )
    
    for i in range(len(image_arrays)):
        try:
            image_arrays[i] = image_do.known_feature_mask(
                image_arrays[i], 
                image_metadata, 
                rivers_data, 
                feature_type="rivers", 
                buffer_metres=20
                )
        except Exception as e:
            print(f"FAILURE: river masking (due to error: {e})")
            print("TRYING: skip step. User must source the file.")
            ui_do.confirm_continue_or_exit()
        try:
            image_arrays[i] = image_do.known_feature_mask(
                image_arrays[i], 
                image_metadata, 
                boundaries_data, 
                feature_type="sea"
                )
        except Exception as e:
            print(f"FAILURE: sea masking (due to error: {e})")
            print("TRYING: skip step. User must source the file.")
            ui_do.confirm_continue_or_exit()
        try:
            image_arrays[i] = image_do.known_feature_mask(
                image_arrays[i], 
                image_metadata, 
                known_reservoirs_data, 
                feature_type="known reservoirs", 
                buffer_metres=50
                )
        except Exception as e:
            print(f"FAILURE: known reservoir masking (due to error: {e})")
            print("TRYING: skip step. User must source the file.")
            ui_do.confirm_continue_or_exit()
        try:
            image_arrays[i] = image_do.mask_urban_areas( # different process (.tif)
                image_arrays[i], 
                image_metadata, 
                urban_areas_data
                )
        except Exception as e:
            print(f"FAILURE: urban area masking (due to error: {e})")
            print("TRYING: skip step. User must source the file.")
            ui_do.confirm_continue_or_exit()
    
    print(f"step 2 complete! finished at {dt.datetime.now().time():%H:%M:%S}")
    return image_arrays

# %% 3. Masking out clouds (OmniCloudMask) (iterative)
def three_mask_clouds(image_arrays, patch_size=1000, patch_overlap=300, 
                      batch_size=4, inference_device="cuda", 
                      inference_dtype="bf16", LP_MODE=False):
    
    from omnicloudmask import predict_from_array
    if not c.HIGH_RES and not LP_MODE:
        print(("WARNING: high-resolution setting is disabled. "
        "cloud masking may not be accurate"))
        ui_do.confirm_continue_or_exit()
    
    print(f"step 3 beginning at {dt.datetime.now().time():%H:%M:%S}")
    print("masking clouds")
    input_array = np.stack(
        (
        image_arrays[2], # red
        image_arrays[0], # green
        image_arrays[1]  # nir
        ))
    
    try:
        pred_mask_2d = predict_from_array(
            input_array, 
            patch_size=patch_size, 
            patch_overlap=patch_overlap, 
            batch_size=batch_size, 
            inference_device=inference_device, 
            inference_dtype=inference_dtype
            )[0]
    except Exception as e:
        print(f"FAILURE: call to CUDA (due to error: {e})")
        print("TRYING: use CPU for inference (slower)")
        ui_do.confirm_continue_or_exit()
        pred_mask_2d = predict_from_array(
            input_array, 
            inference_device="cpu")[0]
    
    combined_mask = (
        (pred_mask_2d == 1) | 
        (pred_mask_2d == 2) | 
        (pred_mask_2d == 3)
        )
    
    for i in range(len(image_arrays)):
        # float is used as it supports NaN
        image_arrays[i] = image_arrays[i].astype(np.float32)
        image_arrays[i][combined_mask] = np.nan
    
    print(f"step 3 complete! finished at {dt.datetime.now().time():%H:%M:%S}")
    return image_arrays

# %% 4. Compute water indices (iterative)
def four_compute_indices(image_arrays):
    
    print(f"step 4 beginning at {dt.datetime.now().time():%H:%M:%S}")
        
    print("converting image array types")
    # first convert to float32 (np.uint16 type is bad for algebraic operations)!
    for i, image_array in enumerate(image_arrays):
        image_arrays[i] = image_array.astype(np.float32)
    green, nir, red = image_arrays
    
    # 4.2 Calculating Indices
    print("populating index arrays")
    np.seterr(divide="ignore", invalid="ignore")
    
    ndwi = (green - nir) / (green + nir)
    ndvi = ((nir - red) / (nir + red))
    
    # gain factor g, aerosol resistance coefficient c1 & c2
    # evi_num = g * (nir - red)
    # evi_den = (nir + (c1 * red) - (c2 * blue) + l)
    # evi = evi_num / evi_den
    # evi_arrays_list.append(evi)
    
    # evi2 = 2.4 * (nir - red) / (nir + red + 1) # 2-band evi can be useful
    # evi2_arrays_list.append(evi2)
    
    indices = {"ndwi":ndwi, "ndvi":ndvi}
    
    print(f"step 4 complete! finished at {dt.datetime.now().time():%H:%M:%S}")
    return indices

# %% 5. Image compositing (and plotting)
def five_composite(index_arrays):
    print(f"step 5 beginning at {dt.datetime.now().time():%H:%M:%S}")
    try: # start by checking for cuda install
        import cupy; del cupy;
        use_cuda = True
    except:
        use_cuda = False
    
    print("compositing all images together")
    stms = {}
    
    for index_name, arrays_list in index_arrays.items():
        shapes = [a.shape for a in arrays_list]
        if len(set(shapes)) > 1:
            ui_do.alert_user(
                warning=f"Shape mismatch in {index_name} arrays: {shapes}", 
                consequence="Image compositing will fail for this index", 
                solution="Skipping this index (check your array dimensions)")
            continue
        
        stack = np.stack(arrays_list)
        
        if use_cuda:
            q = np.array([25., 50., 75.], dtype=np.float32)
            p25, median, p75 = data_do.gpu_nanpercentile(stack, q)
            
            import cupy as cp
            stack_gpu = cp.asarray(stack)
            mean = cp.asnumpy(cp.nanmean(stack_gpu, axis=0))
            del stack_gpu
            cp.get_default_memory_pool().free_all_blocks()
        else:
            p25, median, p75 = np.percentile(stack, [25, 50, 75], axis=0)
            mean = np.nanmean(stack, axis=0)
        
        stms[index_name] = ({
            "p25"   : p25, 
            "median": median, 
            "p75"   : p75, 
            "mean"  : mean
            })
    
    print(f"step 5 complete! finished at {dt.datetime.now().time():%H:%M:%S}")
    return stms

def five_mean(index_arrays):
    print(f"step 5 beginning at {dt.datetime.now().time():%H:%M:%S}")
    print("calculating basic mean of all images")
    mean = {}
    
    for index_name, arrays_list in index_arrays.items():
        shapes = [a.shape for a in arrays_list]
        if len(set(shapes)) > 1:
            print(f"FAILURE: shape mismatch in {index_name} arrays: {shapes}")
            print("TRYING: skip this index")
            continue
        
        stack = np.stack(arrays_list)
        mean[index_name] = np.nanmean(stack, axis=0)
    
    print(f"step 5 complete! finished at {dt.datetime.now().time():%H:%M:%S}")
    return mean

def fiveb_plot(labelling_array, folder_path):
    if c.SAVE_IMAGES:
        print(f"step 5b beginning at {dt.datetime.now().time():%H:%M:%S}")
        print("saving and displaying water index images")
    else:
        print("displaying water index images")
    image_do.plot_indices(labelling_array, c.PLOT_SIZE, c.SAVE_IMAGES, 
                          folder_path, c.RES)
    print(f"step 5b complete! finished at {dt.datetime.now().time():%H:%M:%S}")
    return

# %% 6. Data preparation
def six_prepare_data(folders, prefix):
    """
    Initializes and validates the labelling CSV file to ensure data integrity 
    and sequential ordering.
    
    Creates or loads the training data CSV, removes blank entries, and enforces 
    structural validity (e.g., verifying chunks are in sequential order). It 
    also performs a completion check to flag any chunks that claim to contain 
    water bodies but are missing coordinate data, activating a 
    'data_correction' mode to prompt the user to fix them.
    
    Args:
        folders (list): List of available data folders.
        prefix (str): Prefix string derived from the satellite image name.
        
    Returns:
        list: State variables including break_flag, starting index (i), 
        paths, correction flags, and invalid rows required for the labelling 
        loop.
    """
    print(f"step 6 beginning at {dt.datetime.now().time():%H:%M:%S}")
    
    # 6.1 Preparing File for Labelling
    break_flag = False
    
    # check for a responses folder
    data_folder_found = False
    for folder in folders:
        if data_folder_found:
            break
        
        if os.path.exists(os.path.join(c.DATA_DIR, "training-data")):
            data_folder_found = True
            labelling_path = os.path.join(c.DATA_DIR, "training-data")
    
    if not data_folder_found:
        labelling_path = os.path.join(c.DATA_DIR, "training-data")
        os.makedirs(labelling_path, exist_ok=True)
    
    data_file_name_prefix = f"{prefix.split('_')[0]}"
    data_file_name = f"{data_file_name_prefix}-{c.DATA_FILE_NAME_SUFFIX}"
    data_file_path = os.path.join(labelling_path, data_file_name)
    data_do.blank_entry_check(file=data_file_path) # remove all blank entries
    
    # 6.1.1 File validity check
    print("preparing file for labelling")
    
    header = ("chunk,reservoirs,water-bodies,reservoir-"
    "coordinates,,,,,water-body-coordinates\n")
    while True:
        # file should always exist due to blank_entry_check call
        with open(data_file_path, "r") as file:
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
                sys.exit()
            if ans.lower() == 'new':
                print("creating new file")
                with open(data_file_path, "w") as file:
                    file.write(header)
                    file.write("0, 1, 0\n") # dummy file to start up
                continue

    i = last_chunk + 1 # from this point on, "i" is off-limits as a counter

    # 6.1.2 Data completion check
    # find chunks with invalid or incomplete reservoir coordinate data
    reservoir_rows = []
    body_rows = []
    invalid_rows = []
    data_correction = False

    with open(data_file_path, "r") as file:
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
    invalid_rows = misc.combine_sort_unique(reservoir_rows, body_rows, invalid_rows)

    if data_correction:
        print(f"found {len(invalid_rows)} chunks containing "
               "incomplete, missing, or incorrect coordinate data")
        i = invalid_rows[0]
    print(f"step 6 complete! finished at {dt.datetime.now().time():%H:%M:%S}")
    return [break_flag, i, data_file_path, data_correction, 
            invalid_rows, lines, last_chunk, labelling_path]

# %%
def lp_chunk_processing(imgs, i):
    """
    Processes one element of a numpy array (according to NALIRA pipeline).
    
    Takes a list of a list of numpy arrays, each item in the list being one 
    array for each band of a Sentinel 2 image, and returns a single array with 
    one of the elements (a single chunk) having undergone cloud-masking, index 
    calculation, and STM compositing. This is equivalent to applying the 
    regular NALIRA workflow on a single chunk, rather than applying it to an 
    entire array. 
    
    Parameters
    ----------
    img_chunks_list : list of lists of a numpy arrays (a mess)
        It's a mess. Suppose we have two images. img_chunks_list has two items. 
        Each item is a list with three items. Each of those items is a list, 
        representing the bands green, nir, red, with 4900 items each. Each of 
        THOSE items is a numpy array representing a single chunk of a single 
        band of a single image. 
    i : int
        Counter of current chunk.
    
    Returns
    -------
    labelling_array : numpy array (double check?)
        Input into the rest of step 7.
    
    """
    start_time = time.monotonic()
    
    index_arrays = {"ndwi": [], "ndvi": []}
    
    for img in imgs:
        current_chunk = [img[band][i] for band in range(len(img))]
        
        masked_chunk = three_mask_clouds(
            current_chunk, 
            patch_size=75, 
            patch_overlap=64, 
            batch_size=4, 
            inference_device="cuda", # might as well attempt
            inference_dtype="float32", 
            LP_MODE=True)
        print("cloud masking complete")
        
        indices = four_compute_indices(masked_chunk)
        for key in index_arrays:
            index_arrays[key].append(indices[key])
        print("index calculation complete")
    
    stms = five_composite(index_arrays)
    labelling_array = stms["ndwi"]["mean"] # TODO replace with full stm
    
    time_taken = round(time.monotonic() - start_time, 1)
    print(f"chunk processing completed in {time_taken}")
    
    return labelling_array

# %% 7. Data labelling
def seven_label_data(LP_MODE, i, labelling_array, tci_array, tci_60_array, 
                     data_file_path, data_correction, invalid_rows, lines, 
                     last_chunk):
    """
    Provides an interactive GUI loop for users to label regions of interest 
    (ROIs) within image chunks.
    
    Splits the aggregated NDWI and TCI arrays into smaller segments for batch 
    ROI labelling. Iterates through these chunks, prompting the user to specify 
    the number of reservoirs/water bodies and draw bounding boxes. Handles 
    regular inputs, navigation commands ('back', 'break'), and saves the 
    extracted coordinates to the CSV file.
    
    Parameters
    ----------
    LP_MODE : bool
        Flag for switching to low-power mode of NALIRA (chunk-wise operations).
    i : int
        The current chunk index.
    labelling_array : np.ndarray or list of np.ndarray
        The processed STM/NDWI array or a list of unprocessed arrays (for LP).
    tci_array : np.ndarray
        The True Colour Image array.
    tci_60_array : np.ndarray
        The resized 60m TCI array for context.
    data_file_path : str
        Path to the labelling CSV file.
    data_correction : bool
        Flag indicating if the user is correcting incomplete data.
    invalid_rows : list
        List of row indices that require data correction.
    lines : The current lines of the CSV file
        DESCRIPTION.
    last_chunk : int
        The integer index of the final chunk.
    
    Returns
    -------
    index_chunks : list
        The generated list of split NDWI chunks.
    
    """
    
    # ### 7. Data Labelling
    print(f"step 7 beginning at {dt.datetime.now().time():%H:%M:%S}")
    print("data labelling start")
    break_flag = False
    
    # #### 7.1 Creating Chunks from Satellite Imagery
    print(f"creating {c.N_CHUNKS} chunks from satellite imagery")
    
    tci_chunks = misc.split_array(array=tci_array, n_chunks=c.N_CHUNKS)
    if not LP_MODE:
        index_chunks = misc.split_array(array=labelling_array, n_chunks=c.N_CHUNKS)
    elif LP_MODE:
        # TODO create a LP_MODE flag in split_array and move this logic there
        img_chunks_list = [] # will be shape: [n_images][n_bands][n_chunks][h][w]
        for img in labelling_array:
            img_chunks_list_band = []
            for band in img:
                img_chunks_list_band.append(misc.split_array(array=band, n_chunks=c.N_CHUNKS))
            img_chunks_list.append(img_chunks_list_band)
    
    # #### 7.2 Outputting Images
    print("outputting images")
    invalid_rows_index = 0
    
    while i < round(np.floor(np.sqrt(c.N_CHUNKS)))**2: # 4900 for N_CHUNKS=5000
        if break_flag:
            break
        
        if LP_MODE: # TODO these variable names are misleading
            index_chunks = lp_chunk_processing(img_chunks_list, i)
            labelling_array = index_chunks
        
        image_do.plot_chunks(labelling_array, index_chunks, c.PLOT_SIZE_CHUNKS, 
                             i, tci_chunks, tci_60_array, LP_MODE)
        # TODO broken max values in LP MODE (at least it works though!)
        max_index = [0, 0]
        max_index[0] = round(np.nanmax(index_chunks[i]), 2)
        print(f"MAX ADJUSTED NDWI: {max_index[0]}", end=" | ")
        max_index[1] = round(np.nanmax(index_chunks[i]), 2)
        print(f"MAX NDWI: {max_index[1]}")
        
        # #### 7.3 User Labelling
        data_do.blank_entry_check(file=data_file_path)
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
            data_do.blank_entry_check(file=data_file_path)
            back_flag = False
            try:
                # #### 7.3.1 Regular integer response
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
                    chunk_coords = ui_do.prompt_roi(tci_chunks[i], n_reservoirs)
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
                    chunk_coords = ui_do.prompt_roi(tci_chunks[i], n_bodies)
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
                    # ##### 7.3.2 Non-integer response: "break"
                    print("taking a break")
                    break_flag = True
                    break
                if "back" in n_bodies or "back" in n_reservoirs:
                    # ##### 7.3.3 Non-integer response: "back"
                    back_flag = True
                    if data_correction:
                        print("cannot use 'back' during data correction")
                        break
                    try:
                        n_backs = int(n_reservoirs.split(" ")[1])
                    except:
                        n_backs = 1
                    i -= n_backs
                    data_do.check_file_permission(file_name=data_file_path)
                    with open(data_file_path, mode="r") as re: # read
                        rows = list(csv.reader(re))
                    for j in range(n_backs):
                        rows.pop() # remove the last "n_backs" rows
                    with open(data_file_path, mode="w") as wr: # write
                        data_do.rewrite(write_file=wr, rows=rows)
                    break
                # ##### 7.3.4 Non-integer response: error
                print("error: non-integer response."
                      "\ntype 'break' to save and quit"
                      "\ntype 'back' to go to previous chunk")
                n_reservoirs = input("how many "
                                     "reservoirs? ").strip().lower()
        
        # #### 7.4 Saving Results
        if break_flag:
            break
        if not break_flag and not back_flag:
            data_do.check_file_permission(file_name=data_file_path)
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
                with open(data_file_path, mode="w") as wr: # write
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
                with open(data_file_path, mode="a") as ap: # append
                    ap.write(f"\n{csv_entry}")
    print(f"step 7 complete! finished at {dt.datetime.now().time():%H:%M:%S}")
    return index_chunks

# %% 8. Data segmentation
def eight_segment_data(data_file_path, index_chunks, labelling_path, prefix):
    """
    Segments labelled chunks into individual training images and saves them 
    by class.
    
    Reads coordinate data from the CSV file and extracts localized patches from 
    the NDWI chunks for four classes: reservoirs, water-bodies, land, and sea. 
    Unlabelled land/sea chunks use a fixed centre-crop. Computes global 
    normalization bounds to improve feature contrast, replaces NaN values with 
    a distinguishable mid-grey, and saves the patches as 8-bit greyscale PNGs.
    
    Args:
        data_file_path (str): Path to the labelled CSV file containing 
        coordinates.
        index_chunks (list): List of split NDWI chunk arrays.
        labelling_path (str): Directory where the segmented class folders 
        will be saved.
        prefix (str): Image prefix used for naming the saved PNG files.
    """
    # #### 8.1 Extract Reservoir and Water Body Coordinates
    if not c.HIGH_RES:
        print("high resolution setting must be activated for data segmentation")
        print("exiting program")
        return
    print(f"step 8 beginning at {dt.datetime.now().time():%H:%M:%S}")
    print("data segmentation start")

    res_rows  = []; res_coords  = []
    body_rows = []; body_coords = []
    land_rows = []; land_coords = []
    sea_rows  = []; sea_coords  = []
    
    # placeholder coord – centred crop used for unlabelled land / sea chunks
    none_coord = "[50.0 50.0 55.0 55.0]"
    
    with open(data_file_path, "r") as file:
        lines = file.readlines()

    for i in range(1, len(lines)):
        lines[i] = lines[i].split(",")
        
        # ---- reservoirs ----
        if int(lines[i][1]) > 0:
            res_rows.append(lines[i])
            n = int(res_rows[-1][1])
            col_range = range(3, 3 + n) if n > 1 else [3]
            for j in col_range:
                res_coords.append(
                    (i, data_do.extract_coords(res_rows[-1][j],
                                               create_box_flag=True)))
        
        # ---- water bodies / sea ----
        if int(lines[i][2]) > 0:
            body_rows.append(lines[i])
            first_coords = data_do.extract_coords(body_rows[-1][8],
                                                  create_box_flag=False)
            is_sea = (first_coords[0] == 0 and first_coords[-1] == 157)
            if not is_sea:
                n = int(body_rows[-1][2])
                col_range = range(8, 8 + n) if n > 1 else [8]
                for j in col_range:
                    body_coords.append(
                        (i, data_do.extract_coords(body_rows[-1][j],
                                                   create_box_flag=True)))
            else:
                sea_rows.append(lines[i])
                sea_coords.append(
                    (i, data_do.extract_coords(none_coord,
                                               create_box_flag=True)))
        
        # ---- land ----
        if int(lines[i][1]) == 0 and int(lines[i][2]) == 0:
            land_rows.append(lines[i])
            land_coords.append(
                (i, data_do.extract_coords(none_coord,
                                           create_box_flag=True)))

    # #### 8.2 Compute global NDWI normalisation bounds
    valid_chunks = [chunk for chunk in index_chunks
                    if not np.all(np.isnan(chunk))]
    if valid_chunks:
        global_min = min(np.nanmin(chunk) for chunk in valid_chunks)
        global_max = 0.4 * max(np.nanmax(chunk) for chunk in valid_chunks)
    else:
        global_min = np.nan
        global_max = 0.0
        print("WARNING: All NDWI chunks contained only NaN values.")

    # #### 8.3 Save PNG Training Images per Class
    # --- class definitions ---
    all_coords = [
        (res_coords,  "reservoirs"),
        (body_coords, "water-bodies"),
        (land_coords, "land"),
        (sea_coords,  "sea"),
    ]

    saved_counts = {}

    for coords_list, class_name in all_coords:
        print(f"{len(coords_list)} examples for class '{class_name}'")

        # create class sub-folder
        class_dir = os.path.join(labelling_path, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # find the highest existing image index to avoid overwriting
        existing = [f for f in os.listdir(class_dir) if f.endswith(".png")]
        if existing:
            indices = []
            for name in existing:
                try:
                    indices.append(int(os.path.splitext(name)[0].split("_")[-1]))
                except ValueError:
                    pass
            start_index = max(indices) + 1 if indices else 0
        else:
            start_index = 0

        saved = 0
        for idx, (chunk_n_raw, coordinates) in enumerate(coords_list):
            chunk_n = int(chunk_n_raw) - 1

            # bounds-check against valid_chunks length
            if chunk_n < 0 or chunk_n >= len(valid_chunks):
                print(f"skipping out-of-range chunk {chunk_n} "
                      f"for class '{class_name}'")
                continue

            # extract normalised NDWI patch (float32, shape H×W)
            ndwi_patch = image_do.get_ndwi_patch(
                data=valid_chunks,
                chunk_n=chunk_n,
                coordinates=coordinates,
                g_min=global_min,
                g_max=global_max,
            )

            if ndwi_patch is None or ndwi_patch.size == 0:
                print(f"skipping empty patch (chunk {chunk_n}, "
                      f"class '{class_name}')")
                continue
            
            # normalise to uint8 [0, 255]
            patch_clipped = np.clip(ndwi_patch, global_min, global_max)
            if global_max != global_min:
                patch_norm = ((patch_clipped - global_min)
                              / (global_max - global_min) * 255)
            else:
                patch_norm = np.zeros_like(patch_clipped)
            patch_uint8 = patch_norm.astype(np.uint8)

            # replace NaN-derived zeros with mid-grey so they are visually
            # distinguishable from true zero-valued pixels
            nan_mask = np.isnan(ndwi_patch)
            patch_uint8[nan_mask] = 128
            
            img = Image.fromarray(patch_uint8, mode="L") # greyscale PNG
            img_tile = str(prefix.split('_')[0])
            img_date = str(prefix.split('_')[1].split('T')[0])
            img_index = start_index + saved
            file_name = f"{img_tile}-{img_date}-{img_index:04d}.png"
            img.save(os.path.join(class_dir, file_name))
            saved += 1

        saved_counts[class_name] = saved
        print(f"saved {saved} images to "
              f"{os.path.relpath(class_dir, labelling_path)}/")

    total = sum(saved_counts.values())
    print(f"saved {total} PNG training images across "
          f"{len(all_coords)} classes")
    
    print(f"step 8 complete! finished at {dt.datetime.now().time():%H:%M:%S}")
    return