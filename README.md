# Open-source Reservoir Identifier and Navigator (OpenResIN)

OpenResIN is a project for identifying small water reservoirs in Sentinel-2 satellite imagery. Its existing patch-based pipeline has four stages: label a scene by hand, train a classification model on the labelled image patches, run that model across a whole tile, and score the result. The experimental surface-water path has separate commands for feature building and polygon annotation (`openresin-label-sw`) and for preparing pixel datasets (`openresin-train-sw prepare`). It does not use the existing patch trainer.

Each stage of the existing four-stage pipeline is a console script, and each stage hands the next one ordinary PNG or CSV files on disk. The surface-water preparation command instead writes numerical NPZ datasets with a readable JSON manifest.

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

**3. Install the package.** This installs the dependencies, including scikit-learn, and puts the four existing pipeline stages plus the two experimental surface-water commands on your PATH as console scripts.

```bash
pip install -e .
```

`tkinter` may need to be installed manually on Linux through your package manager.

**4. Install a CUDA build of PyTorch.** Do not skip this if you have a CUDA-compatible GPU (NVIDIA). `torch` arrives as a dependency of `omnicloudmask`, and the wheel pip takes from PyPI is CPU-only: cloud masking will start, fail to find CUDA, and offer to fall back to the CPU. Fix it with this:

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

The uninstall is needed because if `torch` is already installed, pip reports "requirement already satisfied". `torchvision` comes along because the two are built as a pair and have to match.

Pick the index matching your driver from the selector at [pytorch.org](https://pytorch.org/get-started/locally/). Run `nvidia-smi` to see which CUDA version your driver reports. `cu130` is the build this project was developed against. Verify with:

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

The four existing pipeline commands run in order. Each one reads what the previous one wrote:

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

### Experimental surface-water labelling (`openresin-label-sw`)

This command is part of a separate, incomplete path toward a monthly pixel-level water/non-water classifier. With Sentinel-2 L2A `.SAFE` scenes in `data/sat-images/` and the masks described in [`data/README.md`](data/README.md), for example:

```bash
openresin-label-sw --month 2026-04
openresin-label-sw --month 2026-04 --annotate 25
```

The first command builds a numbered navigation preview and six monthly feature arrays under `outputs/label-water/`. Inspect the preview and choose a suitable cell number before using `--annotate`; the second command opens that cell for water and non-water polygon labels and saves them as `area-025.json` in the same output directory. `--train-areas` and `--test-areas` now accept eight training and four test areas, expanded from the original four-training/two-test plan; an existing `areas.json` is not changed automatically. Run `openresin-label-sw --help` for the exact flags. The annotation NDWI chip prefers the saved masked monthly NDWI (`NDWI (masked)`) when its provenance matches the month, scenes and masks, and otherwise falls back to a raw B03/B08 window calculation (`NDWI (raw)`) with a console warning; the TCI composite and dated chips stay raw window reads in both cases. NDWI colours use a `[-0.5, 0.5]` diverging display only, centred on zero with non-finite values shown black. This command does **not** train a random forest, and `openresin-train` still trains the older patch model.

### Prepare the surface-water baseline (`openresin-train-sw prepare`)

Preparation is a separate, non-fitting checkpoint. It requires the frozen T31UCU April 2026 split, all twelve active completion decisions, the saved six-feature archive, and the four recorded `.SAFE` source scenes. Choose a new run directory; preparation refuses a populated destination and never edits the feature, split, or annotation inputs.

```bash
openresin-train-sw prepare \
  --input-dir outputs/label-water \
  --run-dir outputs/label-water/runs/2026-04-v1 \
  --source-image-root data/sat-images
```

The command validates the recorded scene identities and grids, completion digests, exact 8/4 split, feature order and final six-feature validity. It then writes `v1-train.npz`, `v1-test.npz`, `v1-sampling.json`, and `prepare-complete.json`. Training uses a deterministic 100-pixel water cap per polygon followed by a 1,000-pixel water cap per area, then matches the selected water count with non-water pixels from the same area. Test data contain every eligible labelled pixel at natural prevalence. The NPZ files retain area, scene row/column, and polygon position for every sample.

`fit` and `evaluate` are not implemented in this checkpoint. `openresin-train-sw prepare` does not fit a model, calculate scores, or show held-out predictions. Keep the prepared directory unchanged for the later phases; if preparation fails, correct the reported input problem and use a new or empty run directory. Do not use `openresin-train` for these datasets: that command remains the older Keras patch trainer.

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

Train a model for 50 epochs, without accuracy and loss curves plotted post-training, for example.
```bash
openresin-train --epochs 50 --no-show-plots
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

Find performance metrics for a 50-epoch-trained model, for example.
```bash
openresin-evaluate --model-epochs 50
```

`openresin-evaluate` reads the prediction CSV, compares it against the labelled chunks in `data/seed-labels/`, and prints a confusion matrix with the derived metrics. Its `--model-epochs` has to match the predictions file you want scored, exactly as it does for `openresin-predict`.

> [!important] 
> **Pending redesign.** See *Project Status* below.
>
> This stage is the most provisional in the pipeline. It is kept because it is the tool that produced the figures in the original dissertation. Do not cite its output.

## Repository Structure

```
OpenResIN/
├── data/                 # Inputs supplied by the user (except seed labels).
│   ├── masks/            # Masking layers, organised by category
│   ├── sat-images/       # Sentinel-2 .SAFE scenes you download and extract
│   └── seed-labels/      # Hand-labelled seed data, git-tracked, read-only
├── models/               # Trained models. Not tracked by git, arrives empty
├── outputs/              # Everything the pipeline generates. Not tracked by git
│   ├── chunks/           # Mini-chunk PNGs cut for prediction, one folder per scene
│   ├── label-water/      # Surface-water features, annotations and prepared runs
│   ├── labels/           # Label coordinates from your own labelling sessions
│   ├── patches/          # Segmented training images, one folder per class
│   └── predictions/      # Model predictions across a whole tile
├── src/openresin/        # The source package
└── tests/                # Test suite, run with `python -m pytest tests`
```

Each of `data/` and `outputs/` has its own README with the detail: see [`data/README.md`](data/README.md) and [`outputs/README.md`](outputs/README.md).

### The source package

```
src/openresin/
├── label.py              # openresin-label. Orchestrates the labelling stage
├── labelling.py          # The labelling steps that label.py orchestrates
├── train.py              # openresin-train. Orchestrates training
├── modelling.py          # Dataset loading, model building, training, and saving
├── train_sw.py           # Surface-water preparation command
├── modelling_sw.py       # Surface-water validation and deterministic sampling
├── predict.py            # openresin-predict. Runs predictions over a whole tile
├── inference.py          # The core prediction logic that predict.py drives
├── evaluate.py           # openresin-evaluate. Confusion matrix and metrics
├── config.py             # Settings and path anchors for every stage
├── data_handling.py      # Loading, preprocessing and managing data
├── image_handling.py     # Image manipulation and processing
├── user_interfacing.py   # Prompts, warnings and progress reporting
├── misc.py               # Miscellaneous utilities
└── epoch_pathfinder.py   # Experimental epoch sweep. Not part of the pipeline
```

The test suite checks configuration and import contracts, the surface-water feature and annotation workflow, completion and label-state semantics, and deterministic preparation of the random-forest datasets. The new random-forest fit and evaluation phases do not have tests yet because those phases are not implemented in this checkpoint.

## Project Status and Current Limitations

Labelling and training are the stable parts of the pipeline: `openresin-label` and `openresin-train` run end to end and produce outputs as intended. The actual model for `openresin-train` may not stay as a Keras Sequential classifier, but the scaffolding of the stages themselves is intentional.

`openresin-label-sw` and `openresin-train-sw` are experimental and separate from that four-stage path. Monthly features and polygon labels now connect to deterministic prepared pixel datasets. Random-forest fitting, held-out evaluation, and raster export are the next checkpoint and are not yet implemented.

Prediction and evaluation are provisional: `openresin-predict` and `openresin-evaluate` both run and both produce output, but the methodology behind them is not settled and is expected to be replaced. The code is left in place so the pipeline can be executed end to end, and so that a reader can see what is currently being done before deciding what should be done instead.

Specific known limitations:
- Compositing on the CPU is slow and unreliable: The percentile calculation falls back to `numpy` when `cupy` is absent. On a full tile this has crashed on the machines it was tried on. A CUDA-capable GPU is the supported path.
- Cloud masking on the CPU is extremely time-consuming: For this reason, using a CUDA-capable GPU is the strongly suggested path. If you want to use a CPU, be prepared to wait approximately 1 hour per image.
- `LP_MODE` does not support data segmentation: The low-power path through labelling skips segmentation entirely, so it cannot produce training patches. This is unlikely to change.

## License

Apache 2.0: see [LICENSE](LICENSE).

## Contact

Nicolas Arroyo, nicolas.renato.arroyo@gmail.com, he/him
