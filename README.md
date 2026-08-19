# Open-source Reservoir Identifier and Navigator (OpenResIN)

OpenResIN is a project for identifying small water reservoirs in Sentinel-2 satellite imagery. It is a four-stage pipeline: label a scene by hand, train a classification model on the labelled image patches, run that model across a whole tile, and score the result.

Each stage of this pipeline is a console script, and each stage hands the next one files on disk. These files are all ordinary PNGs and CSVs, so you can open them and look at them at any point.

> [!important]
> The pipeline is not fully final; Labelling and training are relatively stable/sound. Inference and evaluation do run but are largely provisional and known to have significant methodological limitations.

## Installation

Python 3.11 or newer. Developed and tested on 3.13.

**1. Clone the repository.**

```bash
git clone https://github.com/nicoarrroyo/OpenResIN.git
cd OpenResIN
```

**2. Create a virtual environment.** Strongly recommended and is the only tested method.

```bash
python -m venv .venv
```

Activate it with `.\.venv\Scripts\Activate.ps1` on Windows PowerShell, or `source .venv/bin/activate` on Linux and macOS.

**3. Install the package.** This installs every dependency and puts the four pipeline stages on your PATH as console scripts.

```bash
pip install -e .
```

`tkinter` may need to be installed manually on Linux.

**4. Install a CUDA build of PyTorch.** Do not skip this if you have a CUDA-compatible GPU (NVIDIA). `torch` arrives as a dependency of `omnicloudmask`, and the wheel pip takes from PyPI is CPU-only: cloud masking will start, fail to find CUDA, and offer to fall back to the CPU. Fix it with this:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

Pick the index matching your driver from the selector at [pytorch.org](https://pytorch.org/get-started/locally/). `cu130` is the build this project was developed against. Verify with:

```bash
python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
```

That should output `13.0 True`.

**5. Install `cupy` for GPU compositing.** Optional but strongly recommended; only useful with a CUDA-capable GPU. It accelerates the percentile calculation in the compositing step, which is slow and unreliable on the CPU.

```bash
pip install -e ".[gpu]"
```

This will install `cupy-cuda12x`. If your CUDA is a different generation, install the matching `cupy` package yourself: see the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).

**A note on the shipped configuration.** `src/openresin/config.py` ships at full scale, sized for the development machine. Do not edit it to run on something smaller: every stage takes command-line flags for that, and they are listed below.

## Usage

Four commands, run in order. Each one reads what the previous one wrote:

```
data/sat-images/*.SAFE
        |
        |  openresin-label
        v
outputs/patches/<class>/*.png
        |
        |  openresin-train
        v
models/*.keras
        |
        |  openresin-predict  (also writes outputs/chunks/)
        v
outputs/predictions/*.csv
        |
        |  openresin-evaluate
        v
metrics
```

Every command takes `-h` and `--help` to list its flags and their defaults.

### 1. Set up the data directory

The repository ships with the code and sample seed labels. The user supplies the imagery and the masking files themselves. Sentinel-2 L2A scenes in `.SAFE` format are extracted into `data/sat-images/`, and the masking files go under `data/masks/` in their labelled subdirectories. See [`data/README.md`](data/README.md) for much more detail.

### 2. Label the imagery (`openresin-label`)

Do some image labelling on 3 images without any cloud masking, for example.
```bash
openresin-label --n-images 3 --no-cloud-masking
```

`openresin-label` reads every `.SAFE` scene under `data/sat-images/`, masks the clouds out of each one, and composites them into a single clean image to label against. A Tkinter window then opens for you to draw bounding boxes around reservoirs, water bodies, land and sea.

If you make a mistake or would like to return to a previous chunk, type `back` (use `back n` to go back `n` chunks). Once you've finished your labelling sessions, type `break` to take a break. The script will then segment the labelled regions and write them as PNGs into `outputs/patches/<class>/`. Those patches are the training set.

Label coordinates from your own session go to `outputs/labels/`, never to the tracked seed set in `data/seed-labels/`.

### 3. Train the model (`openresin-train`)

Train a model for 50 epochs and see the results, for example.

```bash
openresin-train --epochs 50 --show-plots
```

`openresin-train` loads the patch tree, splits it into training and validation sets, builds the Keras model, and fits it. The model is written to `models/` as `{model type} model epochs-{epochs}.keras`. This filename is used to identify the model for inference.

There is also `epoch_pathfinder.py`, which is an experimental script, not part of the pipeline (yet). No need to run it for now.

### 4. Run predictions (`openresin-predict`): provisional

Conduct inference with a 50-epoch-trained model over 2000 chunks, for example.

```bash
openresin-predict --model-epochs 50 --n-chunk-preds 2000
```

`openresin-predict` prepares the scene, cuts it into mini-chunk PNGs under `outputs/chunks/`, classifies them in batches, and writes one row per prediction to a CSV in `outputs/predictions/`.

The `--model-epochs` must match the epoch count you trained with. It defaults to the config value of 150, so a model trained with `--epochs 50` will not be found unless you say so. `--model-type` works the same way.


> [!important]
> **Pending redesign.** See *Project Status* below.
> 
> The output format, the input representation and the decision rule are all stopgaps.

### 5. Assess accuracy (`openresin-evaluate`): provisional

```bash
openresin-evaluate --model-epochs 50
```

`openresin-evaluate` reads the prediction CSV, compares it against the labelled chunks in `data/seed-labels/`, and prints a confusion matrix with the derived metrics.

- `--model-epochs` must match the predictions file you want scored, exactly as it does for `openresin-predict`.
- `--confidence-threshold` sets the minimum confidence for a prediction to count. Defaults to 40.
- `--folder` selects the scene, defaulting to the first found.

> [!important] 
> **Pending redesign.** See *Project Status* below.
>
> This stage is the most provisional in the pipeilne. It is kept because it is the tool that produced the figures in the original dissertation. Do not cite its output.

## Repository Structure

This repository is organised into several key directories:

- `data/`: Inputs. Seed lables that ship with the repo. The data in `sat-images/` and `masks/` must be sourced by the user; see [`data/README.md`](data/README.md) for more information.
	- `seed-labels/`: Public, hand-labelled seed label data. Tracked by git, and read-only as far as the pipeline is concerned.
    	- `T31UCU-5000chunks.csv`: Labelled chunk coordinates for tile T31UCU, used as a starting point so a fresh clone can train without labelling first.
    - `masks/`: Directory for user to organise masking files.
        - `boundaries/`: Country boundary masks.
        - `known-reservoirs/`: Masks for known reservoirs.
        - `rivers/`: River masks.
        - `terrain/`: Terrain level / gradient masks.
        - `urban-areas/`: Urban area (cities, towns, etc.) masks.
    - `sat-images/`: Directory for user-downloaded Sentinel-2 `.SAFE` scenes.

- `outputs/`: Everything the pipeline generates. Not tracked by git, arrives empty on a fresh clone.
    - `labels/`: Label coordinates from your own labelling sessions, kept separate from the tracked seed set.
    - `patches/`: Segmented training images, one subdirectory per class.
    - `chunks/`: Mini-chunk PNGs cut for prediction, one subdirectory per scene.
    - `predictions/`: Model predictions across a whole tile.

- `models/`: Trained models. Not tracked by git, arrives empty.

- `src/openresin/`: The source package.
    - `label.py` (`openresin-label`): Orchestrates the labelling stage.
    - `labelling.py`: The eight labelling steps that `label.py` orchestrates.
    - `train.py` (`openresin-train`): Orchestrates training.
    - `modelling.py`: Dataset loading, model building, training and saving.
    - `predict.py` (`openresin-predict`): Runs predictions over a whole tile.
    - `inference.py`: The core prediction logic that `predict.py` drives.
    - `evaluate.py` (`openresin-evaluate`): Confusion matrix and metrics.
    - `config.py`: Settings and path anchors for every stage.
    - `data_handling.py`: Loading, preprocessing and managing data.
    - `image_handling.py`: Image manipulation and processing.
    - `user_interfacing.py`: Prompts, warnings and progress reporting.
    - `misc.py`: Miscellaneous utilities.
    - `epoch_pathfinder.py`: Epoch sweep helper. Not part of the pipeline; see the note under *Train the model*.

- `tests/`: Test suite. Run with `python -m pytest tests`.
    - `test_config.py`: Checks the config path anchors survive a changed working directory.
    - `test_imports.py`: Checks every module imports without side effects and that each stage exposes a `main()`.
    - More tests for the labelling and training stages are planned.

## Project Status and Current Limitations

Labelling and training are the stable parts of the pipeline: `openresin-label` and `openresin-train` run end to end and produce outputs as intended. The actual model for `openresin-train` may not stay as a Keras Sequential classifier, but the scaffolding of the stages themselves is intentional.

Prediction and evaluation are provisional: `openresin-predict` and `openresin-evaluate` both run and both produce output, but the methodology behind them is not settled and is expected to be replaced. The code is left in place so the pipeline can be executed end to end, and so that a reader can see what is currently being done before deciding what should be done instead.

Specific known limitations:
- Compositing on the CPU is slow and unreliable: The percentile calculation falls back to `numpy` when `cupy` is absent. On a full tile this has crashed on the machines it was tried on. A CUDA-capable GPU is the supported path.
- Cloud masking on the CPU is extremely time-consuming: For this reason, using a CUDA-capable GPU is the strongly suggested path. If you want to use a CPU, be prepared to wait approximately 1 hour per image.
- `LP_MODE` does not support data segmentation: The low-power path through labelling skips segmentation entirely, so it cannot produce training patches. This is unlikely to change.

## License

Apache 2.0: see [LICENSE](LICENSE).

## Contact

Nicolas Arroyo, nicolas.renato.arroyo@gmail.com, he/him
