# %% 0. Start
""" KRISP
Keras Reservoir Identification Sequential Platform
"""
# %%% i. Import External Libraries
import time
import math
import os
import re  # "regular expressions" for parsing filenames

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # filter TF outputs

import numpy as np
import tensorflow as tf
from tensorflow import keras

# %%% ii. Import Internal Functions
from . import config as c
from . import labelling # purely for the functions
from .data_handling import (
    check_positive_int,
    ensure_folder,
    extract_chunk_details,
    sort_file_names,
    sort_prediction_results,
)
from .image_handling import save_image_file
from .misc import split_array
from .user_interfacing import (
    end_spinner,
    start_spinner,
    confirm_continue_or_exit
)

# %% Big guy
def run_model(folder, n_chunks, model_name, max_multiplier=0.41,
              start_chunk=0, n_chunk_preds=None, rebuild_inputs=False):

    start_chunk = check_positive_int(
        var=start_chunk,
        description="chunk to start on")

    if n_chunk_preds is not None:
        n_chunk_preds = check_positive_int(
            var=n_chunk_preds,
            description="number of chunks to make predictions on")

    # %%% 0. Check for Pre-existing Files
    print("==========")
    print("| STEP 0 |")
    print("==========")
    start_time = time.monotonic()

    # %%%% 0.1 Chunk Check!
    test_data_path = os.path.join(c.CHUNKS_DIR, folder,
                                  f"ndwi_{max_multiplier}")
    real_n_chunks = math.floor(math.sqrt(n_chunks)) ** 2 - 1
    n_mini_chunks = 25
    mc_per_len = int(np.sqrt(n_mini_chunks)) # mini-chunks per length
    # important note! ensure this matches the IMG_HEIGHT division in trainer
    # as well as the BOX_SIZE division in data_handling
    # real_n_chunks is the highest chunk index, so a full set is one more.
    expected_files = (real_n_chunks + 1) * n_mini_chunks

    generate_chunks = False
    if rebuild_inputs or not os.path.isdir(test_data_path):
        generate_chunks = True
    else:
        existing_files = [item for item in os.listdir(test_data_path)
                          if os.path.isfile(os.path.join(test_data_path, item))]
        if len(existing_files) < expected_files:
            generate_chunks = True

    if generate_chunks:
        print(f"writing {expected_files} mini-chunk PNGs to '{test_data_path}'")
        print("this takes a long time, and the console may look frozen while "
              "it runs. check the output dir for progress.")
        confirm_continue_or_exit()
        ensure_folder(test_data_path)
    else:
        print(f"using {len(existing_files)} cached mini-chunk PNGs in "
              f"'{test_data_path}'")

    time_taken = time.monotonic() - start_time
    print(f"step 0 complete! time taken: {time_taken:.2f} seconds")

    # %%% 1-6. Scene preparation like labelling does it
    if generate_chunks:
        folders_path = os.path.join(c.DATA_DIR, "sat-images")

        # %%%% 1. Create Image Arrays
        # one_create_image_arrays also opens the two TCI images for the
        # labelling GUI, which inference has no use for. LABEL_DATA is read
        # off the config module, so it is toggled around the call and put back.
        label_data_setting = c.LABEL_DATA
        c.LABEL_DATA = False
        try:
            (image_arrays,
             image_metadata,
             _prefix,
             _tci_array,
             _tci_60_array) = labelling.one_create_image_arrays(
                 folders_path,
                 folder,
                 np.empty([1, 1])
                 )
        finally:
            c.LABEL_DATA = label_data_setting

        # %%%% 2. Mask Clouds (Omnicloudmask)
        if c.CLOUD_MASKING:
            image_arrays = labelling.two_mask_clouds(image_arrays)
        else:
            print("skipping cloud masking")

        # %%%% 3. Calculate Spectral Indices
        ndwi = labelling.three_compute_indices(image_arrays)["ndwi"]

        # %%%% Compositing skipped, inference on a single tile, no step 4
        print("step 4 skipped - no compositing on a single tile")

        # %%%% 5. Mask Known Features
        if c.KNOWN_FEATURE_MASKING:
            ndwi = labelling.five_mask_known_feature(ndwi, image_metadata)
        else:
            print("skipping known feature masking")

        # %%%% 6. Save Satellite Image Chunks
        print("==========")
        print("| STEP 6 |")
        print("==========")
        # %%%% 6.1 Create Chunks
        stop_event, thread = start_spinner(message=f"creating {n_chunks} "
                                           "chunks from satellite imagery")
        start_time = time.monotonic()

        ndwi_chunks = split_array(array=ndwi, n_chunks=n_chunks)
        # bounds are taken over the chunks that still hold data
        valid_chunks = [chunk for chunk in ndwi_chunks
                        if not np.all(np.isnan(chunk))]
        if valid_chunks:
            global_min = min(np.nanmin(chunk) for chunk in valid_chunks)
            global_max = max_multiplier * max(np.nanmax(chunk)
                                              for chunk in valid_chunks)
        else:
            global_min = np.nan
            global_max = 0.0
            print("WARNING: every chunk was masked out; all NDWI is NaN")

        end_spinner(stop_event, thread)

        # %%%% 6.2 Create and Save Mini-Chunks
        print("saving chunks as image files")
        ensure_folder(test_data_path)

        for i, chunk in enumerate(ndwi_chunks):
            if i > real_n_chunks:
                print("WARNING: Exceeded expected number of chunks "
                      f"({real_n_chunks}). Stopping.")
                break

            chunk_height, chunk_width = chunk.shape
            mini_chunk_h = chunk_height / mc_per_len
            mini_chunk_w = chunk_width / mc_per_len

            uly_s = np.linspace(0, chunk_height - mini_chunk_h, mc_per_len)
            ulx_s = np.linspace(0, chunk_width - mini_chunk_w, mc_per_len)

            mc_idx = 0 # mini-chunk index
            for j, ulx in enumerate(ulx_s):
                for k, uly in enumerate(uly_s):
                    # full path, not a bare name: the writer no longer runs
                    # with test_data_path as the working directory
                    image_name = os.path.join(
                        test_data_path,
                        f"ndwi chunk {i} minichunk {mc_idx}.png")
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
        print("| STEP 1-6 |")
        print("============")
        print("chunk generation disabled, skipping steps 1-6")
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

    model_names = os.listdir(c.MODELS_DIR)
    found_model = False
    for name in model_names:
        if model_name in name:
            model_path = os.path.join(c.MODELS_DIR, model_name)
            found_model = True
            break

    if not found_model:
        raise FileNotFoundError(
            f"could not find model matching '{model_name}' in {c.MODELS_DIR}. "
            f"Available models: {model_names}")

    model = keras.models.load_model(model_path)

    all_file_names = os.listdir(test_data_path)
    all_file_names = sort_file_names(all_file_names)

    start_file = start_chunk * 25
    if n_chunk_preds is not None:
        n_files = n_chunk_preds * 25
        selected_file_names = all_file_names[start_file:(start_file + n_files)]
    else:
        selected_file_names = all_file_names[start_file:]
        n_files = len(selected_file_names)
        n_chunk_preds = n_files // 25
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
    image_batch_ds = image_ds.batch(c.BATCH_SIZE)
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
        predicted_class_name = c.CLASS_NAMES[predicted_class_index].upper()

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
