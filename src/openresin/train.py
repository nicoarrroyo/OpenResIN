""" KRISP Trainer
Keras Reservoir Identification Sequential Platform - Trainer
"""
import argparse
import os
import time

import matplotlib.pyplot as plt

from . import krisp_config as c
from . import modelling as operation
from .user_interfacing import end_spinner, list_folders, start_spinner, table_print


def build_parser():
    parser = argparse.ArgumentParser(
        prog="openresin-train",
        description="Train the KRISP classifier on labelled Sentinel-2 patches.")

    parser.add_argument(
        "--epochs", type=int, default=c.EPOCHS,
        help="training epochs (default: %(default)s)")
    parser.add_argument(
        "--batch-size", type=int, default=c.BATCH_SIZE,
        help="samples per gradient update (default: %(default)s)")
    parser.add_argument(
        "--learning-rate", type=float, default=c.LEARNING_RATE,
        help="Adam learning rate (default: %(default)s)")
    parser.add_argument(
        "--dropout-rate", type=float, default=c.DROPOUT_RATE,
        help="dropout before the dense layers (default: %(default)s)")
    parser.add_argument(
        "--save-model", action=argparse.BooleanOptionalAction,
        default=c.SAVE_MODEL,
        help="write the trained model to models/ (default: %(default)s)")

    parser.add_argument(
        "--show-plots", action=argparse.BooleanOptionalAction,
        default=c.SHOW_PLOTS,
        help="plot accuracy and loss curves when training finishes "
             "(default: %(default)s)")

    # --folder is not a setting, it is which data to run against, so it has no
    # config constant and is read straight off args below.
    parser.add_argument(
        "--folder", default=None,
        help="the .SAFE folder under data/sat-images holding the training "
             "data (default: the first one found)")

    return parser


def apply_overrides(args):
    """Write the parsed flags back onto krisp_config.

    modelling.py reads these constants directly rather than taking them as
    arguments, so a per-run override has to land on the module itself.
    """
    c.EPOCHS = args.epochs
    c.BATCH_SIZE = args.batch_size
    c.LEARNING_RATE = args.learning_rate
    c.DROPOUT_RATE = args.dropout_rate
    c.SAVE_MODEL = args.save_model
    c.SHOW_PLOTS = args.show_plots


def main(argv=None):
    args = build_parser().parse_args(argv)
    apply_overrides(args)


    # Without --folder, list_folders returns every .SAFE tile it recognises
    # and we take the first. Same convention as epoch_pathfinder.py, which
    # trains off the same data.
    # Upcoming change is adopting compositing like NALIRA.
    folders_path = os.path.join(c.DATA_DIR, "sat-images")
    folder = args.folder if args.folder else list_folders(folders_path)[0]

    MAIN_START_TIME = time.monotonic()

    # %% 1. Validate paths
    print("----------")
    print("| STEP 1 |")
    print("----------")
    training_data_path = os.path.join(
        folders_path, folder,
        "training data", "training_data_1.tfrecord")
    model_save_dir = c.MODELS_DIR

    if not os.path.exists(training_data_path):
        print(f"error: training data not found at {training_data_path}")
        return 1
    print("paths validated")

    # %% 2. Pre-run summary
    print("----------")
    print("| STEP 2 |")
    print("----------")
    table_print(
        model_type=c.MODEL_TYPE, epochs=c.EPOCHS,
        img_size=f"{c.IMG_HEIGHT}x{c.IMG_WIDTH}",
        batch_size=c.BATCH_SIZE, val_split=c.VALIDATION_SPLIT,
        dropout=c.DROPOUT_RATE, save_model=c.SAVE_MODEL
    )

    # %% 3. Load dataset
    print("----------")
    print("| STEP 3 |")
    print("----------")
    train_ds, val_ds, dataset_size = operation.three_load_dataset(training_data_path)
    num_classes = len(c.CLASS_NAMES)

    # %% 4. Build model
    print("----------")
    print("| STEP 4 |")
    print("----------")
    stop_event, thread = start_spinner(message="building and compiling model")
    model = operation.four_build_model(num_classes)
    end_spinner(stop_event, thread)

    # %% 5. Train
    print("----------")
    print("| STEP 5 |")
    print("----------")
    print(f"training for {c.EPOCHS} epochs")
    history = operation.five_train(model, train_ds, val_ds)

    # %% 6. Save
    print("----------")
    print("| STEP 6 |")
    print("----------")
    operation.six_save_model(model, history, model_save_dir)

    # %% 7. Visualize
    print("----------")
    print("| STEP 7 |")
    print("----------")
    if history and c.SHOW_PLOTS:
        acc     = history.history["accuracy"]
        val_acc = history.history["val_accuracy"]
        loss    = history.history["loss"]
        val_loss= history.history["val_loss"]
        epochs_range = range(c.EPOCHS)

        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc,     label="Train", linewidth=1)
        plt.plot(epochs_range, val_acc, label="Val",   linewidth=1)
        plt.legend(loc="lower right", fontsize=5)
        plt.title(f"{c.MODEL_TYPE.upper()} Accuracy", fontsize=10)
        plt.xlabel("Epoch", fontsize=8); plt.ylabel("Accuracy", fontsize=8)
        plt.ylim([max(0, min(min(acc), min(val_acc)) - 0.1), 1.05])

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss,     label="Train", linewidth=1)
        plt.plot(epochs_range, val_loss, label="Val",   linewidth=1)
        plt.legend(loc="upper right", fontsize=5)
        plt.title(f"{c.MODEL_TYPE.upper()} Loss", fontsize=10)
        plt.xlabel("Epoch", fontsize=8); plt.ylabel("Loss", fontsize=8)
        max_loss = max(max(loss), max(val_loss))
        plt.ylim([-0.05, max(1.0, max_loss + 0.1)])

        plt.tight_layout()
        plt.show()

    TOTAL_TIME = time.monotonic() - MAIN_START_TIME
    print(f"total time: {TOTAL_TIME:.2f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
