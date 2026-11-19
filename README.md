# Open-source Reservoir Identifier and Navigator (OpenResIN)

OpenResIN finds surface water in Sentinel-2 satellite imagery. A random forest classifies each 10 m pixel as water or non-water from monthly features. The pipeline has two commands:

- `openresin-label-sw` builds the monthly features and lets you draw water and non-water polygons by hand.
- `openresin-train-sw` prepares the datasets (`prepare`), fits the forest (`fit`) and scores it on the test areas (`evaluate`).

> [!important]
> This is the first version (V1). It is fixed to one tile (T31UCU) and one month (April 2026). A scene-based second version is in development and will change the commands.

## Installation

You need Python 3.11 or newer. The project is developed on Python 3.13.

```bash
git clone https://github.com/nicoarrroyo/OpenResIN.git
cd OpenResIN
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -e .
```

On Linux you may need to install `tkinter` with your package manager.

**If you have an NVIDIA GPU, install a CUDA build of PyTorch.** pip installs a CPU-only build, and cloud masking on a CPU takes about one hour per scene. Pick the index for your driver at [pytorch.org](https://pytorch.org/get-started/locally/); `cu130` is the build used in development. The uninstall is necessary, because otherwise pip reports "requirement already satisfied" and changes nothing.

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
```

The last command should print `13.0 True`.

## Input data

The repository ships only the code. You supply:

- **Sentinel-2 L2A scenes**, extracted as `.SAFE` folders into `data/sat-images/`.
- **The sea and urban masks**, in `data/masks/boundaries/` and `data/masks/urban-areas/`. See [`data/README.md`](data/README.md).
- **The area labels**: `areas.json` and the twelve `area-XXX.json` files. Seed labels are being created currently; email me for a copy of the current version (find email below).

## Usage

Run these commands from the repository root (pass `--help` for options); outputs go to `outputs/label-water/` by default.

**1. Build the features.**

```bash
openresin-label-sw --month 2026-04
```

This masks clouds, the sea and urban areas, takes the monthly median of thirteen features per pixel, and writes `features.npz` and a numbered grid preview of the tile. Allow about 30 GiB of free disk space. If the command is interrupted and leaves a `.feature-build-*` folder behind, delete it before rerunning.

**2. Add the labels and confirm each area.** Copy `areas.json` and the `area-XXX.json` files into `outputs/label-water/`. A completed area is tied to one exact `features.npz`, so you must confirm each of the twelve areas again against yours. For example, for area 025:

```bash
openresin-label-sw --month 2026-04 --annotate 25
```

Check the polygons, click **Complete area**, then **Confirm completion**. To label a new area, choose a cell number from the grid preview and open it in the same way.

**3. Prepare, fit and evaluate.** Use a new run folder for each attempt: no step overwrites a finished output.

```bash
openresin-train-sw prepare  --run-dir outputs/label-water/runs/2026-04
openresin-train-sw fit      --run-dir outputs/label-water/runs/2026-04
openresin-train-sw evaluate --run-dir outputs/label-water/runs/2026-04
```

- `prepare` checks the four scenes, the frozen 8/4 split and the completed areas, then samples the training and test pixels.
- `fit` fits a 100-tree random forest on the training pixels. It never reads the test set.
- `evaluate` calls a pixel water at a probability of 0.75 or more. It prints precision, recall and F1 per test area and for all four together, and writes probability and binary GeoTIFFs and an overlay image for each test area.

If `fit` fails, start again in a new run folder. If `evaluate` fails while it writes the GeoTIFFs, run it again on the same run folder; the scores do not change.

## Repository structure

```
OpenResIN/
├── data/                 # Inputs you supply. Git tracks only the READMEs
│   ├── masks/            # Sea and urban masks (three other folders are unused for now)
│   └── sat-images/       # Sentinel-2 .SAFE scenes you download and extract
├── outputs/              # Everything the commands write. Git tracks only the README
│   └── label-water/      # Features, area labels and one folder per run
├── src/openresin/        # The source package
└── tests/                # Test suite
```

```
src/openresin/
├── label_sw.py           # openresin-label-sw. Feature build and annotation command
├── labelling_sw.py       # Scene reading, cloud masking, features, annotation window
├── train_sw.py           # openresin-train-sw. prepare, fit and evaluate command
├── modelling_sw.py       # Checks, sampling, random forest, scores and GeoTIFF export
├── image_handling.py     # Band reading and the sea and urban masks
└── config.py             # Fixed settings and path anchors
```

## Limitations

- **Fixed scope.** V1 only runs on tile T31UCU for April 2026.
- **Test-exposed scores.** The V1 settings were chosen after looking at results on the test areas, so its scores (water F1 84.12%) are exploratory, not independent validation.
- **Spatial correlation.** Nearby pixels may still be spatially correlated, even though training and test areas do not touch or share pixels.
- **Uncalibrated probability.** The probability GeoTIFF is the forest's vote share after 1:5 water-to-non-water training, not a calibrated probability of water.
- **Missing masks.** If a mask folder is empty, `openresin-label-sw` prints a warning and continues without that mask, which changes the features.
- **Reproducibility across machines.** Running feature extraction on a different GPU, PyTorch, or OmniCloudMask version may change the cloud masks, and therefore the features and scores.

## License

Apache 2.0: see [LICENSE](LICENSE).

## Contact

Nicolas Arroyo, nicolas.renato.arroyo@gmail.com, he/him
