# %% 0.Start
""" eepy
epoch pathfinder python
"""
print("=== Script Start (Epoch Comparison) ===")
# %%% i. Import External Libraries
import time
import pathlib
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd # Added for easier results handling
import config_NALIRA as c
import user_interfacing as ui_do

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

MAIN_START_TIME = time.monotonic()
print("Imports complete.")

folders_path = os.path.join(c.HOME_DIR, "data", "sat-images")
folder = ui_do.list_folders(folders_path)[0]

# %% 1. Configuration
print("=== 1. Configuring Parameters ===")
# --- Core Settings ---
# =============================================================================
# BASE_PROJECT_DIR = os.path.join("C:\\", "Users", "nicol", "Documents", "PAPER",
#                                 "YEAR 3", "Individual Project", "data") # Adjust if needed
# =============================================================================

training_data_dir = os.path.join(c.DATA_DIR, "training-data")



# =============================================================================
# SENTINEL_FOLDER = ("S2C_MSIL2A_20250301T111031_N0511_R137_T31UCU_"
#                    "20250301T152054.SAFE") # Adjust if needed
# =============================================================================
# =============================================================================
# DATA_BASE_PATH = os.path.join(BASE_PROJECT_DIR, "Sentinel 2",
#                               SENTINEL_FOLDER, "training data")
# =============================================================================

# --- Training Parameters ---
LEARNING_RATE = 0.001 # Adam optimizer default, but can be specified #
EPOCH_SETTINGS = list(range(100, 171, 10)) # List of epochs to test
NUM_REPEATS = 10 # Number of times to repeat training for each epoch setting

# --- Model Parameters ---
DROPOUT_RATE = 0.2

# --- Test Image ---
# =============================================================================
# TEST_IMAGE_SUBDIR = MODEL_TYPE # Subdirectory within DATA_BASE_PATH
# =============================================================================
# =============================================================================
# TEST_IMAGE_NAME = f"{MODEL_TYPE} chunk 1 reservoir 1.png"
# =============================================================================

# --- Dataset Parameters ---
IMG_HEIGHT = 31 # int(157/5) # must adjust this for the actual image size!!!!
IMG_WIDTH = 31 # int(157/5)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 123 # For reproducibility of splits

print(f"Image size: ({IMG_HEIGHT}, {IMG_WIDTH})")
print(f"Batch size: {BATCH_SIZE}") #
print(f"Validation split: {VALIDATION_SPLIT}")
print(f"Epoch Settings: {EPOCH_SETTINGS}")
print(f"Repeats per Setting: {NUM_REPEATS}")


# %% 2. Prepare Paths and Directories
print("=== 2. Preparing Paths ===")
data_dir = os.path.join(DATA_BASE_PATH, MODEL_TYPE)
test_image_path = os.path.join(DATA_BASE_PATH, TEST_IMAGE_SUBDIR,
                               TEST_IMAGE_NAME)


# --- Path Validation ---
if not os.path.isdir(data_dir):
    print(f"Error: Data directory not found at {data_dir}")
    print("Please check the configuration for BASE_PROJECT_DIR, "
          "SENTINEL_FOLDER, and DATA_DIR_NAME.")
    sys.exit(1)
else:
    print("Using nominal data directory.")

if not os.path.exists(test_image_path):
    print(f"WARNING: Test image not found at {test_image_path}. Prediction "
           "step will be skipped.")
else:
    print(f"Using nominal test image: {TEST_IMAGE_NAME}")

data_dir_pathlib = pathlib.Path(data_dir)
try:
    image_count = len(list(data_dir_pathlib.glob('*/*.png')))
    print(f"Found {image_count} images in nominal data directory.")
    if image_count == 0:
        print("WARNING: No images found. Check the data directory structure "
              "and image format.")
except Exception as e:
    print(f"Error counting images: {e}")
    image_count = 0 # Assume zero if listing fails

if image_count < BATCH_SIZE:
    print(f"WARNING: Total image count ({image_count}) is less than the batch "
           f"size ({BATCH_SIZE}). This might cause issues during training.")

# %% 3. Prepare the Dataset (Done ONCE)
print("=== 3. Loading and Preparing Dataset ===")
# Load data using a Keras utility - This only needs to be done once
try:
    train_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir_pathlib,
      validation_split=VALIDATION_SPLIT,
      subset="training",
      seed=RANDOM_SEED,
      image_size=(IMG_HEIGHT, IMG_WIDTH),
      batch_size=BATCH_SIZE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir_pathlib,
      validation_split=VALIDATION_SPLIT,
      subset="validation",
      seed=RANDOM_SEED,
      image_size=(IMG_HEIGHT, IMG_WIDTH),
      batch_size=BATCH_SIZE)
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Check if the directory structure matches "
          "'data_dir/class_a/image.png', 'data_dir/class_b/image.png'.")
    sys.exit(1)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Found classes: {class_names}")
if num_classes <= 1:
    print("Error: Need at least two classes for classification.")
    sys.exit(1)

# Configure the dataset for performance
print("configuring dataset performance")
AUTOTUNE = tf.data.AUTOTUNE
SHUFFLE_BUFFER_SIZE = max(1000, image_count) # Adjust buffer size based on image count

# Cache, shuffle, prefetch the datasets - done once
train_ds = train_ds.cache().shuffle(buffer_size=SHUFFLE_BUFFER_SIZE, seed=RANDOM_SEED).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("complete!")

# %% 4. Define Model Building Function
print("=== 4. Defining Model Building Function ===")

# Data augmentation layers (defined once)
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ],
  name="data_augmentation",
)

# --- NEW: Function to build and compile a fresh model ---
def build_compile_model(num_classes, learning_rate, dropout_rate):
    """Builds and compiles a new instance of the CNN model."""
    model = Sequential([
      data_augmentation, # Apply data augmentation as the first layer
      layers.Rescaling(1./255), # Rescale pixel values
      # Conv Blocks
      layers.Conv2D(16, kernel_size=3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      # Dropout and Dense Layers
      layers.Dropout(dropout_rate), # Use passed dropout rate
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes, name="outputs") # Output layer (logits)
    ], name=f"{MODEL_TYPE}_classifier_{int(time.time())}") # Add timestamp for unique name

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), # Use passed LR
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

# %% 5. Training Loop Experiment
print("=== 5. Starting Training Experiment ===")

# --- NEW: Import tqdm ---
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    print("Info: 'tqdm' library not found. Progress bars for repeats will not be shown.")
    print("Install it using: pip install tqdm")
    TQDM_AVAILABLE = False

# --- Store results ---
results_data = {
    'epochs_tested': [],
    'avg_train_acc': [],
    'avg_val_acc': [],
    'avg_train_loss': [],
    'avg_val_loss': []
}

# --- Outer loop for epoch settings ---
for epochs_to_run in EPOCH_SETTINGS:
    print(f"\n--- Testing with {epochs_to_run} Epochs ---")
    # Store results for the repeats of this epoch setting
    repeat_train_acc = []
    repeat_val_acc = []
    repeat_train_loss = []
    repeat_val_loss = []

    # --- Inner loop for repeats - NOW WITH TQDM ---
    # Determine the iterable based on whether tqdm is available
    repeat_iterable = range(NUM_REPEATS)
    if TQDM_AVAILABLE:
        repeat_iterable = tqdm(range(NUM_REPEATS),
                               desc=f"Epochs {epochs_to_run} Progress", # Description for the bar
                               unit="repeat", # Label for each step
                               ncols=100) # Optional: Set fixed width for the bar

    for i in repeat_iterable: # Iterate using tqdm or plain range
        # Optional: Clear description after loop finishes if using tqdm
        # if TQDM_AVAILABLE:
        #     repeat_iterable.set_description(f"Epochs {epochs_to_run} Repeat {i+1}/{NUM_REPEATS}")

        # Build and compile a fresh model for each repeat
        # print("Building and compiling new model instance...") # Keep or remove this print
        model = build_compile_model(num_classes, LEARNING_RATE, DROPOUT_RATE)

        # Train the model
        # print(f"Starting training for {epochs_to_run} epochs...") # Removed this - tqdm shows progress
        try:
            history = model.fit(
              train_ds,
              validation_data=val_ds,
              epochs=epochs_to_run,
              verbose=0 # Keep verbose=0 for model.fit
            )
            # print("Training complete for this repeat.") # Removed - tqdm shows completion

            # Extract final metrics
            final_train_acc = history.history['accuracy'][-1]
            final_val_acc = history.history['val_accuracy'][-1]
            final_train_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]

            repeat_train_acc.append(final_train_acc)
            repeat_val_acc.append(final_val_acc)
            repeat_train_loss.append(final_train_loss)
            repeat_val_loss.append(final_val_loss)

        except Exception as e:
            print(f"\nError occurred during training repeat {i+1} for {epochs_to_run} epochs: {e}")
            # Add placeholder for failed run
            repeat_train_acc.append(np.nan)
            repeat_val_acc.append(np.nan)
            repeat_train_loss.append(np.nan)
            repeat_val_loss.append(np.nan)
            # If using tqdm, ensure the bar closes properly even on error
            if TQDM_AVAILABLE and isinstance(repeat_iterable, tqdm):
                 repeat_iterable.close() # May not be necessary depending on context

    # --- Averaging after repeats ---
    avg_tr_acc = np.nanmean(repeat_train_acc)
    avg_v_acc = np.nanmean(repeat_val_acc)
    avg_tr_loss = np.nanmean(repeat_train_loss)
    avg_v_loss = np.nanmean(repeat_val_loss)

    print(f"--- Results for {epochs_to_run} Epochs (Avg over {NUM_REPEATS} repeats) ---")
    print(f"  Avg Train Accuracy: {avg_tr_acc:.4f}")
    print(f"  Avg Val Accuracy:   {avg_v_acc:.4f}")
    print(f"  Avg Train Loss:     {avg_tr_loss:.4f}")
    print(f"  Avg Val Loss:       {avg_v_loss:.4f}")

    # Store averaged results
    results_data['epochs_tested'].append(epochs_to_run)
    results_data['avg_train_acc'].append(avg_tr_acc)
    results_data['avg_val_acc'].append(avg_v_acc)
    results_data['avg_train_loss'].append(avg_tr_loss)
    results_data['avg_val_loss'].append(avg_v_loss)

# Convert results to a DataFrame for easier handling/saving (optional)
results_df = pd.DataFrame(results_data)
print("\n=== Experiment Results Summary ===")
print(results_df)
# Optionally save results_df to a CSV file
# results_df.to_csv(f"{MODEL_TYPE}_epoch_comparison_results.csv", index=False)
# print("Results saved to CSV.")


# %% 6. Visualize Experiment Results
print("=== 6. Visualizing Experiment Results ===")

plt.figure(figsize=(6, 4)) # Adjusted figure size for 4 plots
x_ticks_to_show = results_data['epochs_tested'][::2]

# Plot Average Accuracy vs Epochs Tested
plt.subplot(1, 2, 1)
plt.plot(results_data['epochs_tested'], results_data['avg_train_acc'], 
         marker='o', linestyle='-', label='Avg Training Accuracy', markersize=4)
plt.plot(results_data['epochs_tested'], results_data['avg_val_acc'], 
         marker='s', linestyle='--', label='Avg Validation Accuracy', markersize=4)
plt.xlabel("Number of Epochs Trained", fontsize=8)
plt.ylabel("Average Accuracy", fontsize=8)
plt.title("Average Accuracy vs. Epochs Trained", fontsize=10)
plt.legend(fontsize=5)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(x_ticks_to_show) # Ensure ticks match tested epochs
min_acc = min(min(results_data['avg_train_acc']), min(results_data['avg_val_acc']))
plt.ylim([max(0, min_acc - 0.1), 1.05]) # Start below min accuracy, max 1.05

# Plot Average Loss vs Epochs Tested
plt.subplot(1, 2, 2)
plt.plot(results_data['epochs_tested'], results_data['avg_train_loss'], 
         marker='o', linestyle='-', label='Avg Training Loss', markersize=4)
plt.plot(results_data['epochs_tested'], results_data['avg_val_loss'], 
         marker='s', linestyle='--', label='Avg Validation Loss', markersize=4)
plt.xlabel("Number of Epochs Trained", fontsize=8)
plt.ylabel("Average Loss", fontsize=8)
plt.title("Average Loss vs. Epochs Trained", fontsize=10)
plt.legend(fontsize=5)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(x_ticks_to_show) # Ensure ticks match tested epochs
# Dynamically set ylim based on data
max_loss = max(max(results_data['avg_train_loss']), max(results_data['avg_val_loss']))
# Start below 0, go above max loss
plt.ylim([-0.05, max(1.0, max_loss + 0.1)])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# %% 7. Predict on New Data (Using the LAST trained model)
print("=== 7. Predicting on New Data (Using Last Model) ===")
# Note: This uses the model instance from the very last training run (150 epochs, 3rd repeat)
# If you need to predict with the "best" model based on results_df, you'd need to
# retrain a model with the optimal epoch setting found, or save models during the loop.

if 'model' in locals() and model is not None: # Check if model exists
    if os.path.exists(test_image_path):
        print(f"Loading test image: {TEST_IMAGE_NAME}")
        # (Visualization code omitted for brevity - kept from original if desired)

        try:
            img = tf.keras.utils.load_img(
                test_image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
            )
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Create a batch

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            predicted_class_index = np.argmax(score)
            predicted_class_name = class_names[predicted_class_index]
            confidence = 100 * np.max(score)

            print(
                "Prediction: This image most likely belongs to "
                f"'{predicted_class_name}' "
                f"with a {confidence:.2f}% confidence."
            )
            # Optional: Add success/failure check based on name

        except FileNotFoundError:
            print(f"Error: Test image file not found at {TEST_IMAGE_NAME}")
        except Exception as e:
            print(f"An error occurred during prediction: {e}")

    else:
        print("Skipping prediction because test image was not found at: "
              f"{TEST_IMAGE_NAME}")
else:
    print("Skipping prediction as no model was successfully trained in the last iteration.")


# %% 8. Save Model (Optional) - Disabled for this experiment script
# if SAVE_MODEL and history: # ... code removed ...
print("=== 8. Saving Model (Disabled for this experiment) ===")


# %% 9. Final Summary
print("=== 9. Script End ===")
TOTAL_TIME = time.monotonic() - MAIN_START_TIME
print(f"Total processing time: {TOTAL_TIME:.2f} seconds")
print("================")
