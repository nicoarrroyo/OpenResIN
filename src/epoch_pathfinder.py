from krisp_operations import three_load_dataset, four_build_model
import krisp_config as c
import numpy as np
import matplotlib.pyplot as plt
import user_interfacing as ui_do
import os

EPOCH_SETTINGS = list(range(50, 175, 25))
NUM_REPEATS    = 2

folders_path = os.path.join(c.HOME_DIR, "data", "sat-images")
folders = ui_do.list_folders(folders_path)[0]

training_data_path = ...  # same path logic as trainer

train_ds, val_ds, _ = three_load_dataset(training_data_path)
num_classes = len(c.CLASS_NAMES)

results = {}
for epochs in EPOCH_SETTINGS:
    val_accs = []
    for _ in range(NUM_REPEATS):
        model = four_build_model(num_classes)
        history = model.fit(train_ds, validation_data=val_ds,
                           epochs=epochs, verbose=0)
        val_accs.append(history.history["val_accuracy"][-1])
    results[epochs] = np.mean(val_accs)
    print(f"epochs={epochs:3d}  avg_val_acc={results[epochs]:.4f}")

# plot
plt.plot(list(results.keys()), list(results.values()), marker="o")
plt.xlabel("Epochs"); plt.ylabel("Avg Val Accuracy")
plt.title("Epoch Pathfinder"); plt.grid(True)
plt.show()