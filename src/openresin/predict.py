# %% 0. Start
""" KRISP-Y
Keras Reservoir Identification Sequential Platform - Yielding of data
"""
# %%% i. Import External Libraries
import argparse
import os
import time

# %%% ii. Import Internal Functions
from . import config as c
from .data_handling import (
    blank_entry_check,
    check_file_permission,
)
from .inference import run_model
from .misc import convert_seconds_to_hms
from .user_interfacing import end_spinner, start_spinner


def build_parser():
    parser = argparse.ArgumentParser(
        prog="openresin-predict",
        description=("Run the trained KRISP model over a Sentinel-2 tile and "
                     "write predictions to CSV under outputs/predictions/."))

    parser.add_argument(
        "--folder", type=str, default=None,
        help="the .SAFE folder under data/sat-images to predict over "
             "(default: the first one found there)")
    parser.add_argument(
        "--model-epochs", type=int, default=c.EPOCHS,
        help="epoch count in the model filename (default: %(default)s)")
    parser.add_argument(
        "--model-type", default=c.MODEL_TYPE, choices=("ndwi", "tci"),
        help="model type in the model filename (default: %(default)s)")
    parser.add_argument(
        "--rebuild-inputs", action="store_true",
        help="force regeneration of mini-chunk PNGs even if they already exist")
    parser.add_argument(
        "--n-chunk-preds", type=int, default=None,
        help="predict only the first N chunks, for a smoke test on a weaker "
             "machine (default: the whole scene)")

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    folder = args.folder or c.default_folder()
    if folder is None:
        print("no .SAFE scene found under data/sat-images")
        print("put a Sentinel-2 product there, or name one with --folder")
        return 1

    scene_path = os.path.join(c.DATA_DIR, "sat-images", folder)
    if not os.path.isdir(scene_path):
        print(f"no such scene: {scene_path}")
        print("--folder takes the name of a .SAFE directory that exists "
              "under data/sat-images")
        return 1

    (sentinel_name, instrument_and_product_level, datatake_start_sensing_time,
     processing_baseline_number, relative_orbit_number, tile_number_field,
     product_discriminator_and_format) = folder.split("_")

    model_name = f"{args.model_type} model epochs-{args.model_epochs}.keras"
    model_path = os.path.join(c.MODELS_DIR, model_name)
    if not os.path.isfile(model_path):
        print(f"no such model: {model_path}")
        print("--model-type and --model-epochs name the file train.py wrote")
        return 1

    predictions_file = (f"P_{c.N_CHUNKS}_{args.model_epochs}"
                        f"_{tile_number_field}.csv")

    os.makedirs(c.PREDICTIONS_DIR, exist_ok=True)
    predictions_file_path = os.path.join(c.PREDICTIONS_DIR, predictions_file)

    # The CSV is written whole, not appended to, so an earlier one for this
    # tile is replaced. Say so: ten of the scenes here share tile T31UCU.
    if os.path.isfile(predictions_file_path):
        print(f"overwriting the existing {predictions_file}")
    if args.n_chunk_preds is not None:
        print(f"partial run: the first {args.n_chunk_preds} chunks only, and "
              "the CSV will hold only those")

    run_start_time = time.monotonic()
    print("\n=== KRISP RUN START ===")
    the_results = run_model(
        folder=folder,
        n_chunks=c.N_CHUNKS,
        model_name=model_name,
        max_multiplier=0.41,
        n_chunk_preds=args.n_chunk_preds,
        rebuild_inputs=args.rebuild_inputs
    )
    print("=== KRISP RUN COMPLETE ===\n")

    # %% write the results
    stop_event, thread = start_spinner(message="writing predictions CSV")
    check_file_permission(predictions_file_path)

    minichunk_header = ",minichunks,"
    chunk_header = "chunk," + ",".join(map(str, range(25)))

    with open(predictions_file_path, mode="w") as ap:
        ap.write(minichunk_header)
        ap.write(f"\n{chunk_header}")
        for result in the_results:
            chunk_num, minichunk_num, label, confidence = result
            base_entry = f"{label} {confidence!s}," # apparently, !s is the same as str()
            if minichunk_num == 0:
                ap.write(f"\n{chunk_num!s},{base_entry}")
            else:
                ap.write(f"{base_entry}")

    check_file_permission(predictions_file_path)
    blank_entry_check(predictions_file_path)
    end_spinner(stop_event, thread)

    h, m, s = convert_seconds_to_hms(time.monotonic() - run_start_time)
    print(f"predictions written to {predictions_file_path}")
    print(f"total time taken: {h} hours, {m} minutes, {s} seconds")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
