""" KRISP Trainer
Keras Reservoir Identification Sequential Platform - Trainer
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import time
import sys
import matplotlib.pyplot as plt

import krisp_config as c
import krisp_operations as operation
from user_interfacing import start_spinner, end_spinner, list_folders
folder = list_folders(c.DATA_DIR)

MAIN_START_TIME = time.monotonic()

# %% 1. Validate paths
print("----------")
print("| STEP 1 |")
print("----------")
training_data_path = os.path.join(
    c.DATA_DIR, "sat-images", c.SENTINEL_FOLDER, 
    "training data", "training_data_1.tfrecord")
model_save_dir = os.path.join(c.DATA_DIR, "saved-models")

if not os.path.exists(training_data_path):
    print(f"error: training data not found at {training_data_path}")
    sys.exit(1)
print("paths validated")

# %% 2. Pre-run summary
print("----------")
print("| STEP 2 |")
print("----------")
from user_interfacing import table_print
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
if history:
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