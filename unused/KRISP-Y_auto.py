# %% 0. Start
# %%% i. Import External Libraries
import time
MAIN_START_TIME = time.monotonic()
import os
import datetime
import zoneinfo as zf
import math

# %%% ii. Import Internal Functions
from KRISP import run_model
from data_handling import check_file_permission, blank_entry_check
from misc import convert_seconds_to_hms
from user_interfacing import start_spinner, end_spinner

# %%% iii. Directory Management
try: # personal pc mode
    HOME = os.path.join("C:\\", "Users", "nicol", "Documents", "UoM", "YEAR 3", 
                        "Individual Project", "Downloads")
    os.chdir(HOME)
except: # uni mode
    HOME = os.path.join("C:\\", "Users", "c55626na", "OneDrive - "
                        "The University of Manchester", "Individual Project")
    os.chdir(HOME)

class_names = ["land", "reservoirs", "water bodies"]
n_chunks = 5000 # do not change!!

# %% prelim
stop_event, thread = start_spinner(message="pre-run preparation")
# =============================================================================
# folders = [
#     ("S2C_MSIL2A_20250301T111031_N0511_R137_T31UCU_20250301T152054.SAFE"), 
#     ("S2C_MSIL2A_20250318T105821_N0511_R094_T30UYC_20250318T151218.SAFE"), 
#     ("S2A_MSIL2A_20250320T105751_N0511_R094_T31UCT_20250320T151414.SAFE"), 
#     ("S2A_MSIL2A_20250330T105651_N0511_R094_T30UYC_20250330T161414.SAFE"), 
#     ("S2C_MSIL2A_20250331T110651_N0511_R137_T30UXC_20250331T143812.SAFE")
#     ]
# =============================================================================
# "#" = to download, "##" = downloaded
# "###" = fully predicted, "####" fully predicted with every model
# all have to be retrainded on balanced set
##folder = ("S2C_MSIL2A_20250301T111031_N0511_R137_T31UCU_20250301T152054.SAFE")
##folder = ("S2C_MSIL2A_20250318T105821_N0511_R094_T30UYC_20250318T151218.SAFE")
##folder = ("S2A_MSIL2A_20250320T105751_N0511_R094_T31UCT_20250320T151414.SAFE")
##folder = ("S2A_MSIL2A_20250330T105651_N0511_R094_T30UYC_20250330T161414.SAFE")
##folder = ("S2C_MSIL2A_20250331T110651_N0511_R137_T30UXC_20250331T143812.SAFE")
folder = ("S2C_MSIL2A_20250331T110651_N0511_R137_T31UCU_20250331T143812.SAFE")

(sentinel_name, instrument_and_product_level, datatake_start_sensing_time, 
 processing_baseline_number, relative_orbit_number, tile_number_field, 
 product_discriminator_and_format) = folder.split("_")

real_n_chunks = math.floor(math.sqrt(n_chunks)) ** 2 - 1
# model_epoch_options = [50, 55, 60, 65, 70]
# model_epoch_options = [50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70]
model_epoch_options = [50, 75, 100, 125, 150]

for model_epochs in model_epoch_options:
    n_chunk_preds = 5000 # has to be reset every time
    
    # file format: P_(chunks)_(minichunks)_(epochs)_(tile number)
    # P for predictions
    predictions_file = f"P_{n_chunks}_{model_epochs}_{tile_number_field}.csv"
    
    minichunk_header = ",minichunks,"
    chunk_header = "chunk," + ",".join(map(str, range(25)))
    
    os.chdir(os.path.join(HOME, "Sentinel 2", folder))
    
    # %% find biggest chunk
    check_file_permission(predictions_file)
    blank_entry_check(predictions_file)

    with open(predictions_file, mode="r") as file:
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
        print("this image is complete! exiting program")
        continue
    
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
    start_time_obj = datetime.datetime.now(zf.ZoneInfo("Europe/London"))
    est_end_time = start_time_obj + est_duration
    
    start_str = start_time_obj.strftime(time_format)
    est_end_str = est_end_time.strftime(time_format)
    end_spinner(stop_event, thread)
    
    # %% pre-run update
    # note: these numbers are estimates for reference only
    pre_completion = round(100 * biggest_chunk / real_n_chunks, 2)
    post_completion = round(
        100 * (biggest_chunk + n_chunk_preds) / real_n_chunks, 
        2
        )
    
    print(f"\n=== PRE-RUN CHECK | MODEL EPOCHS {model_epochs} ===")
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
    print(f"=== PRE-RUN CHECK | MODEL EPOCHS {model_epochs} ===\n")
    
    # %% yield predictions
    run_start_time = time.monotonic()
    print(f"\n=== KRISP RUN START | MODEL EPOCHS {model_epochs} ===")
    the_results = run_model(
        folder=folder, 
        n_chunks=5000, 
        model_name=f"ndwi model epochs-{model_epochs}.keras", 
        max_multiplier=0.41, 
        start_chunk=biggest_chunk, 
        n_chunk_preds=int(n_chunk_preds)
        )
    print(f"=== KRISP RUN COMPLETE | MODEL EPOCHS {model_epochs} ===\n")
    
    # %% write the results
    stop_event, thread = start_spinner(message="aftercare")
    os.chdir(os.path.join(HOME, "Sentinel 2", folder))
    check_file_permission(predictions_file)
    blank_entry_check(predictions_file)
    
    if biggest_chunk < 1:
        with open(predictions_file, mode="a") as ap:
            ap.write(minichunk_header)
            ap.write(f"\n{chunk_header}")
    
    with open(predictions_file, mode="a") as ap:
        for result in the_results:
            chunk_num, minichunk_num, label, confidence = result
            base_entry = f"{label} {str(confidence)},"
            if minichunk_num == 0:
                ap.write(f"\n{str(chunk_num)},{base_entry}")
            else:
                ap.write(f"{base_entry}")
    
    check_file_permission(predictions_file)
    blank_entry_check(predictions_file)
    
    # %% post-run update
    # note: these numbers are estimates for reference only
    h, m, s = convert_seconds_to_hms(time.monotonic() - run_start_time)
    end_time_obj = datetime.datetime.now(zf.ZoneInfo("Europe/London"))
    end_str = end_time_obj.strftime(time_format)
    end_spinner(stop_event, thread)
    
    print(f"\n=== POST-RUN UPDATE | MODEL EPOCHS {model_epochs} ===")
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
    
    print(f"=== POST-RUN UPDATE | MODEL EPOCHS {model_epochs} ===")

# %% end
time_taken = time.monotonic() - MAIN_START_TIME
print(f"total processing time: {round(time_taken, 2)} seconds")
