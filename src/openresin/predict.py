# %% 0. Start
""" KRISP-Y
Keras Reservoir Identification Sequential Platform - Yielding of data
"""
# %%% i. Import External Libraries
import argparse
import datetime
import math
import os
import time
import zoneinfo as zf

# %%% ii. Import Internal Functions
from . import config as c
from .data_handling import (
    blank_entry_check,
    check_file_permission,
)
from .inference import run_model
from .misc import convert_seconds_to_hms
from .user_interfacing import confirm_continue_or_exit, end_spinner, start_spinner

# Tiles I have worked through, kept as a record of what is downloaded and how
# far each one got. "##" = downloaded, "###" = fully predicted.
####folder = ("S2C_MSIL2A_20250301T111031_N0511_R137_T31UCU_20250301T152054.SAFE")
####folder = ("S2C_MSIL2A_20250318T105821_N0511_R094_T30UYC_20250318T151218.SAFE")
####folder = ("S2A_MSIL2A_20250320T105751_N0511_R094_T31UCT_20250320T151414.SAFE")
####folder = ("S2A_MSIL2A_20250330T105651_N0511_R094_T30UYC_20250330T161414.SAFE")
####folder = ("S2C_MSIL2A_20250331T110651_N0511_R137_T30UXC_20250331T143812.SAFE")
#folder = ("S2C_MSIL2A_20250331T110651_N0511_R137_T31UCU_20250331T143812.SAFE")
# Shared with evaluate.py via the config
DEFAULT_FOLDER = c.DEFAULT_FOLDER


def build_parser():
    parser = argparse.ArgumentParser(
        prog="openresin-predict",
        description=("Run the trained KRISP model over a tile and append the "
                     "predictions to its CSV. Resumes from the last chunk "
                     "already written."))

    parser.add_argument(
        "--folder", default=DEFAULT_FOLDER,
        help="the .SAFE folder under data/sat-images to predict over")
    # Defaults to what train.py would have saved, so the two cannot drift:
    # train writes "{MODEL_TYPE} model epochs-{EPOCHS}.keras" and this reads
    # it back. Hardcoding 150 here is how they came to disagree.
    parser.add_argument(
        "--model-epochs", type=int, default=c.EPOCHS,
        help="epoch count in the model filename (default: %(default)s)")
    parser.add_argument(
        "--model-type", default=c.MODEL_TYPE, choices=("ndwi", "tci"),
        help="model type in the model filename (default: %(default)s)")
    parser.add_argument(
        "--n-chunk-preds", type=int, default=5000,
        help="chunks to predict this run; clamped to what is left "
             "(default: %(default)s)")

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    MAIN_START_TIME = time.monotonic()
    folder = args.folder
    model_epochs = args.model_epochs
    n_chunk_preds = args.n_chunk_preds

    # %%% iii. Directory Management
    n_chunks = 5000 # do not change!!

    # %% prelim
    stop_event, thread = start_spinner(message="pre-run preparation")

    (sentinel_name, instrument_and_product_level, datatake_start_sensing_time,
     processing_baseline_number, relative_orbit_number, tile_number_field,
     product_discriminator_and_format) = folder.split("_")

    real_n_chunks = math.floor(math.sqrt(n_chunks)) ** 2 - 1
    # Same filename train.py writes: "{MODEL_TYPE} model epochs-{EPOCHS}.keras".
    model_name = f"{args.model_type} model epochs-{model_epochs}.keras"

    # file format: P_(chunks)_(minichunks)_(epochs)_(tile number)
    # P for predictions
    predictions_file = f"P_{n_chunks}_{model_epochs}_{tile_number_field}.csv"

    minichunk_header = ",minichunks,"
    chunk_header = "chunk," + ",".join(map(str, range(25)))

    os.makedirs(c.PREDICTIONS_DIR, exist_ok=True)
    predictions_file_path = os.path.join(c.PREDICTIONS_DIR, predictions_file)

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

    n_chunk_preds = min(n_chunk_preds, real_n_chunks - biggest_chunk)

    if n_chunk_preds == 0:
        end_spinner(stop_event, thread)
        print("this image is already complete; nothing to predict")
        return 0

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
    check_file_permission(predictions_file_path)
    blank_entry_check(predictions_file_path)

    if biggest_chunk < 1:
        with open(predictions_file_path, mode="a") as ap:
            ap.write(minichunk_header)
            ap.write(f"\n{chunk_header}")

    with open(predictions_file_path, mode="a") as ap:
        for result in the_results:
            chunk_num, minichunk_num, label, confidence = result
            base_entry = f"{label} {confidence!s}," # apparently, !s is the same as str()
            if minichunk_num == 0:
                ap.write(f"\n{chunk_num!s},{base_entry}")
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
