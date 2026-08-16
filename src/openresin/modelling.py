import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # filter TF outputs

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers

from . import config as c


# %% 3. Load and prepare dataset
def three_load_dataset(patches_path):
    """Load the labelled patch PNGs from NALIRA, split into train and val.

    The directory layout *is* the label: keras takes one class per sub-folder
    of patches_path, in alphabetical order. The discovered names are returned
    so the caller can check them against config rather than trusting the tree.

    Args:
        patches_path (str): Directory holding one sub-folder per class.

    Returns:
        tuple: (train_ds, val_ds, class_names).
    """
    loader_args = dict(
        validation_split=c.VALIDATION_SPLIT,
        seed=c.RANDOM_SEED,
        image_size=(c.IMG_HEIGHT, c.IMG_WIDTH),
        batch_size=c.BATCH_SIZE,
        color_mode="rgb", # patches are greyscale, the model takes 3 channels
    )

    train_ds = keras.utils.image_dataset_from_directory(
        patches_path, subset="training", **loader_args)
    val_ds = keras.utils.image_dataset_from_directory(
        patches_path, subset="validation", **loader_args)

    class_names = train_ds.class_names

    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


# %% 4. Build model
def four_build_model(num_classes):
    data_augmentation = Sequential([
        layers.RandomFlip("horizontal",
                          input_shape=(c.IMG_HEIGHT, c.IMG_WIDTH, 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")

    model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(c.DROPOUT_RATE),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, name="outputs"),
    ], name=f"{c.MODEL_TYPE}_classifier")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=c.LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    return model


# %% 5. Train
def five_train(model, train_ds, val_ds):
    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=c.EPOCHS,
            verbose=0
        )
        print("training complete")
        return history
    except Exception as e:
        print(f"training failed: {e}")
        return None


# %% 6. Save model
def six_save_model(model, history, save_dir):
    import datetime
    if not c.SAVE_MODEL or not history:
        print("model saving skipped")
        return

    os.makedirs(save_dir, exist_ok=True)
    base_path = os.path.join(
        save_dir, f"{c.MODEL_TYPE} model epochs-{c.EPOCHS}.keras")

    if os.path.exists(base_path):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(base_path)
        save_path = f"{base}_{timestamp}{ext}"
        print("file exists, saving versioned copy")
    else:
        save_path = base_path

    try:
        model.save(save_path)
        print(f"model saved to: {save_path}")
    except Exception as e:
        print(f"save failed: {e}")
