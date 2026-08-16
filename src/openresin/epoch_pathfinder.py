import matplotlib.pyplot as plt
import numpy as np

from . import config as c
from .modelling import four_build_model, three_load_dataset

EPOCH_SETTINGS = list(range(50, 175, 25))
NUM_REPEATS    = 2

# same source as the trainer: the patch PNGs NALIRA wrote
train_ds, val_ds, class_names = three_load_dataset(c.PATCHES_DIR)
num_classes = len(class_names)

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
