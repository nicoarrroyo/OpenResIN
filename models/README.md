# Models Directory
This is an output data folder, meaning you don't need to put anything in here yourself. Any models trained using the `krisp_trainer` script will be automatically saved here. Then, when the model is deployed with the `krispy` script, it will be searched for here as well. Models will be saved as TensorFlow Keras models (e.g. `ndwi model epochs-100.keras`).

Checkpoints are not tracked by git, so this directory arrives empty on a fresh clone and you will need to train a model (or be given one) before `krispy` has anything to load. Both scripts resolve this location through `MODELS_DIR` in `src/krisp_config.py`; if you need to move it, change it there rather than in the scripts.
