# %% 0. Start
""" KRISP-Y
Keras Reservoir Identification Sequential Platform - Yielding of data
"""
# %%% i. Import External Libraries
import time
MAIN_START_TIME = time.monotonic()
import os
import datetime
import zoneinfo as zf
import math
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from collections import Counter

# %%% ii. Import Internal Functions
from KRISP import run_model
from data_handling import check_file_permission, blank_entry_check
from data_handling import deduplicate_by_max_confidence
from image_handling import image_to_array
from misc import convert_seconds_to_hms
from user_interfacing import start_spinner, end_spinner, confirm_continue_or_exit

# %%% iii. Directory Management
HOME = os.path.dirname(os.getcwd()) # HOME path is one level up from the cwd

n_chunks = 5000 # do not change!!
confidence_threshold = 40 # do not change!! these are calculated

res_precision = 0.1027 # do not change!! these are calculated
bod_precision = 0.130
land_precision = 0.964
sea_precision = 0.960

res_recall = 0.8952 # do not change!! these are calculated
bod_recall = 0.909
land_recall = 0.748
sea_recall = 0.985

# %% prelim
stop_event, thread = start_spinner(message="pre-run preparation")
# "##" = downloaded, "###" = fully predicted
####folder = ("S2C_MSIL2A_20250301T111031_N0511_R137_T31UCU_20250301T152054.SAFE")
####folder = ("S2C_MSIL2A_20250318T105821_N0511_R094_T30UYC_20250318T151218.SAFE")
####folder = ("S2A_MSIL2A_20250320T105751_N0511_R094_T31UCT_20250320T151414.SAFE")
####folder = ("S2A_MSIL2A_20250330T105651_N0511_R094_T30UYC_20250330T161414.SAFE")
####folder = ("S2C_MSIL2A_20250331T110651_N0511_R137_T30UXC_20250331T143812.SAFE")
#folder = ("S2C_MSIL2A_20250331T110651_N0511_R137_T31UCU_20250331T143812.SAFE")

folder = ("S2C_MSIL2A_20250301T111031_N0511_R137_T31UCU_20250301T152054.SAFE")
folder_path = os.path.join(HOME, "data", "Sentinel 2", folder)

(sentinel_name, instrument_and_product_level, datatake_start_sensing_time, 
 processing_baseline_number, relative_orbit_number, tile_number_field, 
 product_discriminator_and_format) = folder.split("_")

real_n_chunks = math.floor(math.sqrt(n_chunks)) ** 2 - 1
model_epochs = 150 # 150 for ndwi, 151 for tci
n_chunk_preds = 5000 # can be bigger than n_chunks
if model_epochs == 151:
    model_name=f"tci model epochs-{model_epochs}.keras"
else:
    model_name=f"ndwi model epochs-{model_epochs}.keras"
save_map = True
res = 10 # options: 10, 20, 60

# file format: P_(chunks)_(minichunks)_(epochs)_(tile number)
# P for predictions
predictions_file = f"P_{n_chunks}_{model_epochs}_{tile_number_field}.csv"

minichunk_header = ",minichunks,"
chunk_header = "chunk," + ",".join(map(str, range(25)))

#os.chdir(os.path.join(HOME, "Sentinel 2", folder))
predictions_file_path = os.path.join(folder_path, predictions_file)

# %% find biggest chunk
check_file_permission(predictions_file_path)
blank_entry_check(predictions_file_path)

with open(predictions_file_path, mode="r") as file:
    lines = file.readlines()

biggest_chunk = 0
for i, line in enumerate(lines):
    if i < 2:
        continue # skip first couple rows for header
    try:
        biggest_chunk = max(biggest_chunk, int(line.split(",")[0])) + 1
    except:
        continue

if n_chunk_preds > real_n_chunks - biggest_chunk:
    n_chunk_preds = real_n_chunks - biggest_chunk

if n_chunk_preds == 0:
    end_spinner(stop_event, thread)
    print("this image is complete! commencing data processing")
    # %% data processing
    with open (predictions_file_path, mode="r") as file:
        predictions = file.readlines()
    
    # %%% collecting and printing prediction results
    res_predictions = []
    bod_predictions = []
    land_predictions = []
    sea_predictions = []
    for i, prediction in enumerate(predictions):
        prediction = predictions[i].split(",")
        for j, cell in enumerate(prediction):
            try:
                confidence = float(cell.split(" ")[-1])
            except:
                confidence = 100
            if "reservoir" in cell.strip().lower():
                if confidence >= confidence_threshold:
                    res_predictions.append([i, confidence])
            if "water" in cell.strip().lower():
                if confidence >= confidence_threshold:
                    bod_predictions.append([i, confidence])
            if "land" in cell.strip().lower():
                if confidence >= confidence_threshold:
                    land_predictions.append([i, confidence])
            if "sea" in cell.strip().lower():
                if confidence >= confidence_threshold:
                    sea_predictions.append([i, confidence])
    
    res_estimate = int(len(res_predictions) * res_precision / res_recall)
    bod_estimate = int(len(bod_predictions) * bod_precision / bod_recall)
    land_estimate = int(len(land_predictions) * land_precision / land_recall)
    sea_estimate = int(len(sea_predictions) * sea_precision / sea_recall)
    
    print("congratulations!")
    print("krisp, with help from nalira and krispette, and everyone else, "
          "have found...")
    print(f"{res_estimate} reservoirs in east england!")
    print(f"{bod_estimate} non-reservoir water bodies in east england!")
    print(f"{land_estimate} land minichunks in east england!")
    print(f"{sea_estimate} sea minichunks in east england!")
    
    stop_event, thread = start_spinner(message="creating map")
    sorted_res = sorted(res_predictions, reverse=True, key=lambda row: row[1])
    sorted_res = sorted_res[:res_estimate]
    sorted_bod = sorted(bod_predictions, reverse=True, key=lambda row: row[1])
    sorted_bod = sorted_bod[:bod_estimate]
    sorted_land = sorted(land_predictions, reverse=True, key=lambda row: row[1])
    sorted_land = sorted_land[:land_estimate]
    sorted_sea = sorted(sea_predictions, reverse=True, key=lambda row: row[1])
    sorted_sea = sorted_sea[:sea_estimate]
    
    # deduplication
    deduplicated_sorted_res = deduplicate_by_max_confidence(sorted_res)
    deduplicated_sorted_bod = deduplicate_by_max_confidence(sorted_bod)
    deduplicated_sorted_land = deduplicate_by_max_confidence(sorted_land)
    deduplicated_sorted_sea = deduplicate_by_max_confidence(sorted_sea)
    
    # %%% creating the reservoir map
    granule_path = os.path.join(folder_path, "GRANULE")
    subdirs = [d for d in os.listdir(granule_path) 
               if os.path.isdir(os.path.join(granule_path, d))]
    prefix = (f"{tile_number_field}_{datatake_start_sensing_time}")
    
    
    map_image = image_to_array(os.path.join(granule_path, 
                                            subdirs[0], "IMG_DATA", 
                                            f"R{res}m", 
                                            f"{prefix}_TCI_{res}m.jp2"))
    plt.figure(figsize=(6, 6))
    plt.imshow(map_image)
    ax = plt.gca()
    
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(left=False, bottom=False, 
                   labelleft=False, labelbottom=False)
    
    cmap = cm.get_cmap("coolwarm")
    all_confidences = [r[1] for r in sorted_res]
    norm = plt.Normalize(min(all_confidences), max(all_confidences))
    
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # blue is lowest confidence, red is highest confidence
    
    # calculate chunk geometry
    chunks_per_side = int(np.sqrt(n_chunks))
    side_length = map_image.shape[0]
    chunk_length = side_length / chunks_per_side
    
    # %%%% plotting reservoir map
    for res_item in deduplicated_sorted_res:      
        chunk = int(res_item[0])
        chunk_col = chunk % chunks_per_side
        chunk_ulx = chunk_col * chunk_length
        
        chunk_row = chunk // chunks_per_side
        chunk_uly = chunk_row * chunk_length
        
        marker_color = cmap(norm(res_item[1]))
        ax.plot(chunk_ulx, chunk_uly, marker="o", color=marker_color, ms=2)
    end_spinner(stop_event, thread)
    
    # %%%% saving reservoir map
    if save_map:
        stop_event, thread = start_spinner(message="saving map")
        plot_name_base = "the_map"
        counter = 0
        plot_name = f"{plot_name_base}.jpg"
        while os.path.exists(plot_name):
            counter += 1
            plot_name = f"{plot_name_base}_{counter}.jpg"
        plot_save_location = os.path.join(folder_path, plot_name)
        plt.savefig(plot_save_location, dpi=3000, bbox_inches="tight")
        end_spinner(stop_event, thread)
        print(f"map saved as {plot_name}")
    
    plt.show()
    
    # %%% creating confidence distribution plot
    stop_event, thread = start_spinner(message="plotting confidence "
                                       "distribution")
    # Extract confidence values and round them to the nearest integer
    confidences = [(round(res[1] / 2)) * 2 for res in deduplicated_sorted_res]
    
    # Count occurrences of each confidence level
    confidence_counts = Counter(confidences)
    
    # Sort by confidence level
    sorted_confidences = sorted(confidence_counts.items())
    
    # Extract data for plotting
    x_vals, y_vals = zip(*sorted_confidences)
    
    # Plot the histogram
    plt.figure(figsize=(5, 3))
    plt.bar(x_vals, y_vals, color='royalblue', edgecolor='black')
    plt.xlabel("Confidence Level (%)")
    plt.ylabel("Number of Reservoirs")
    plt.title("Distribution of Confidence Levels for Predicted Reservoirs")
    plt.xticks(range(min(x_vals), max(x_vals) + 5, 5))
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    end_spinner(stop_event, thread)
    plt.show()
    # Compute the average confidence
    average_confidence = sum(res[1] for res in sorted_res) / len(sorted_res)
    
    print(f"Average Confidence Level: {average_confidence:.2f}%")
    
    # %%% create everything bagel map
    stop_event, thread = start_spinner(message="creating everything bagel plot")
    deduplicated_sorted_classes = [
        deduplicated_sorted_res, 
        deduplicated_sorted_bod, 
        deduplicated_sorted_land, 
        deduplicated_sorted_sea
        ]
    colours = ["b", "g", "y", "c"]
    
    plt.figure(figsize=(6, 6))
    plt.imshow(map_image)
    ax = plt.gca()
    
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(left=False, bottom=False, 
                   labelleft=False, labelbottom=False)
    end_spinner(stop_event, thread)
    
    # %%%% plotting everything bagel map
    for i, deduplicated_sorted_class in enumerate(deduplicated_sorted_classes):
        stop_event, thread = start_spinner(message=f"plotting class {i+1}")
        for element in deduplicated_sorted_class:
            chunk = int(element[0])
            chunk_col = chunk % chunks_per_side
            chunk_ulx = chunk_col * chunk_length
            
            chunk_row = chunk // chunks_per_side
            chunk_uly = chunk_row * chunk_length
            
            marker_color = colours[i]
            ax.plot(chunk_ulx, chunk_uly, marker="o", color=marker_color, ms=1)
        end_spinner(stop_event, thread)
    
    # %%%% saving everything bagel map
    if save_map:
        stop_event, thread = start_spinner(message="saving 'everything bagel' map")
        plot_name_base = "the_everything_bagel"
        counter = 0
        plot_name = f"{plot_name_base}.jpg"
        while os.path.exists(plot_name):
            counter += 1
            plot_name = f"{plot_name_base}_{counter}.jpg"
        plot_save_location = os.path.join(folder_path, plot_name)
        plt.savefig(plot_save_location, dpi=3000, bbox_inches="tight")
        end_spinner(stop_event, thread)
        print(f"map saved as {plot_name}")
    end_spinner(stop_event, thread)
    plt.show()
    
    sys.exit(0)

# %% yield expected duration of run
n_files = n_chunk_preds * 25
# duration relationship for the dell xps 9315 (personal pc)
duration = (0.00045 * n_files) + 6.62
h, m, s = convert_seconds_to_hms(1.1 * duration)
est_duration = datetime.timedelta(
    hours=h, 
    minutes=m, 
    seconds=s)

time_format = "%H:%M:%S %B %d %Y"
start_time_obj = datetime.datetime.now(zf.ZoneInfo("Europe/Rome"))
est_end_time = start_time_obj + est_duration

start_str = start_time_obj.strftime(time_format)
est_end_str = est_end_time.strftime(time_format)
end_spinner(stop_event, thread)

# %% pre-run update
# note: these numbers are estimates for reference only
pre_completion = round(100 * biggest_chunk / real_n_chunks, 2)
post_completion = round(100 * (biggest_chunk + n_chunk_preds) / real_n_chunks, 2)

print(f"\n=== PRE-RUN CHECK == MODEL EPOCHS {model_epochs} ===")
print(f"COMPLETED SO FAR: {pre_completion}%")
print(f"chunks {biggest_chunk}/{real_n_chunks} | "
      f"files {biggest_chunk * 25}/{real_n_chunks * 25} |")

print(f"\nREMAINING: {round(100 - pre_completion, 2)}%")
print(f"chunks {real_n_chunks - biggest_chunk} | "
      f"files {(real_n_chunks - biggest_chunk) * 25} |")

print("\nTO BE COMPLETED THIS RUN: "
      f"{round(post_completion - pre_completion, 2)}%")
print(f"chunks {n_chunk_preds} | files {n_files} | ")

print(f"\nSTARTING AT: {start_str}")
print(f"EXPECTED DURATION: {h} hours, {m} minutes, {s} seconds")
print(f"EXPECTED TO END AT: {est_end_str}")
print(f"=== PRE-RUN CHECK == MODEL EPOCHS {model_epochs} ===\n")

confirm_continue_or_exit()

# %% yield predictions
run_start_time = time.monotonic()
print("\n=== KRISP RUN START ===")
the_results = run_model(
    folder=folder, 
    n_chunks=5000, 
    model_name=model_name, 
    max_multiplier=0.41, 
    start_chunk=biggest_chunk, 
    n_chunk_preds=int(n_chunk_preds)
    )
print("=== KRISP RUN COMPLETE ===\n")

# %% write the results
stop_event, thread = start_spinner(message="aftercare")
# os.chdir(os.path.join(HOME, "Sentinel 2", folder))
check_file_permission(predictions_file_path)
blank_entry_check(predictions_file_path)

if biggest_chunk < 1:
    with open(predictions_file_path, mode="a") as ap:
        ap.write(minichunk_header)
        ap.write(f"\n{chunk_header}")

with open(predictions_file_path, mode="a") as ap:
    for result in the_results:
        chunk_num, minichunk_num, label, confidence = result
        base_entry = f"{label} {str(confidence)},"
        if minichunk_num == 0:
            ap.write(f"\n{str(chunk_num)},{base_entry}")
        else:
            ap.write(f"{base_entry}")

check_file_permission(predictions_file_path)
blank_entry_check(predictions_file_path)

# %% post-run update
# note: these numbers are estimates for reference only
h, m, s = convert_seconds_to_hms(time.monotonic() - run_start_time)
end_time_obj = datetime.datetime.now(zf.ZoneInfo("Europe/London"))
end_str = end_time_obj.strftime(time_format)
end_spinner(stop_event, thread)

print(f"\n=== POST-RUN UPDATE == MODEL EPOCHS {model_epochs} ===")
print(f"COMPLETED SO FAR: {post_completion}%")
print(f"chunks {biggest_chunk + n_chunk_preds}/{real_n_chunks} | "
      f"files {(biggest_chunk + n_chunk_preds) * 25}/{real_n_chunks * 25} |")

print("\nCOMPLETED THIS RUN: "
      f"{round(post_completion - pre_completion, 2)}%")
print(f"chunks {n_chunk_preds} | files {n_files} | ")

print(f"\nREMAINING: {round(100 - post_completion, 2)}%")
print(f"chunks {real_n_chunks - biggest_chunk - n_chunk_preds} | "
      f"files {(real_n_chunks - biggest_chunk - n_chunk_preds) * 25} |")

print(f"\nSTARTED AT: {start_str}")
print(f"ACTUAL DURATION: {h} hours, {m} minutes, {s} seconds")
print(f"ENDED AT: {end_str}")

print(f"=== POST-RUN UPDATE == MODEL EPOCHS {model_epochs} ===")
