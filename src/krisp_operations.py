import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import krisp_config as c

# %% 3. Load and prepare dataset
def three_load_dataset(training_data_path):
    img_features = {
        'height':    tf.io.FixedLenFeature([], tf.int64),
        'width':     tf.io.FixedLenFeature([], tf.int64),
        'depth':     tf.io.FixedLenFeature([], tf.int64),
        'label':     tf.io.FixedLenFeature([], tf.int64),
        'label_text':tf.io.FixedLenFeature([], tf.string),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    
    def parse_img(example_proto):
        features = tf.io.parse_single_example(example_proto, img_features)
        img = tf.io.decode_png(features["image_raw"], channels=3)
        img = tf.reshape(img, [
            tf.cast(features["height"], tf.int32),
            tf.cast(features["width"],  tf.int32),
            tf.cast(features["depth"],  tf.int32)
        ])
        img = tf.image.resize(img, [c.IMG_HEIGHT, c.IMG_WIDTH])
        return img, features["label"]
    
    raw = tf.data.TFRecordDataset(training_data_path)
    dataset_size = sum(1 for _ in raw)
    
    shuffled = raw.shuffle(dataset_size, seed=c.RANDOM_SEED, 
                           reshuffle_each_iteration=True)
    val_size  = int(dataset_size * c.VALIDATION_SPLIT)
    
    val_ds   = (shuffled.take(val_size)
                        .map(parse_img, num_parallel_calls=tf.data.AUTOTUNE)
                        .batch(c.BATCH_SIZE)
                        .cache()
                        .prefetch(tf.data.AUTOTUNE))
    
    train_ds = (shuffled.skip(val_size)
                        .map(parse_img, num_parallel_calls=tf.data.AUTOTUNE)
                        .batch(c.BATCH_SIZE)
                        .cache()
                        .prefetch(tf.data.AUTOTUNE))
    
    print(f"dataset loaded: {dataset_size} records, "
          f"val={val_size}, train={dataset_size - val_size}")
    return train_ds, val_ds, dataset_size


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
    import os, datetime
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