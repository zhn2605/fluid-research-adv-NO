# fluid-research-adv-NO

Fluid prediction using adversarial neural operators. A UNet is trained on PIV
velocity data three ways and compared:

- baseline (MSE loss)
- PITA, physics-informed loss with PDE discovery (`models/PITA.py`)
- adv-NO, adversarial + perceptual loss, arXiv:2509.08752 (`models/adv_NO.py`)

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m ipykernel install --user --name fluid-adv-no --display-name "Python (fluid-adv-NO)"
.venv/bin/jupyter lab
```

Tested on an RTX 4060 laptop GPU (8 GB). adv-NO downloads the VGG-19 weights
(~550 MB) on first run.

Data goes in `.data/` (`Re_240.h5`, `Re_1280.h5`, `Re_2520.h5`, gitignored).

## Files

| File | Purpose |
|---|---|
| `DeepLearningBenchmark_Framework_UNetExample.ipynb` | train the three models and save runs |
| `DeepLearning_Analysis.ipynb` | compare saved runs (figures A-D) |
| `load_and_process_data.ipynb` | data preparation |
| `make_poster_figures.py` | poster figures, written to `figures/poster/` |

## Outputs

Each run saves a rollout `.h5` and model `.pt` to `outputs/<model>/`, named
`<timestamp>_<model>_<epochs>epoch_bs<batch>_<speed>_<tag>`. The analysis
notebook uses the newest run per model unless `RUN_SELECT` is set to part of a
filename.

## adv-NO memory

`adv_max_frames` limits how many frames the discriminator and VGG see per step.
At batch 32 on 8 GB, 16 frames peaked at ~5.1 GB and 32+ ran out of memory. If
you run out of memory, lower `batch_size` first. The runs used for the poster
all used batch size 16.
