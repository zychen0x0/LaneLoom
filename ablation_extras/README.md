# Ablation and analysis add-ons

This folder is self-contained. It registers four small subclasses of existing
LaneLoom modules, adds configs that inherit the main temporal config, and
provides two inference-only analysis scripts. No file outside this folder is
modified.

## Layout

```text
ablation_extras/
├── modules.py                 # registered ablation subclasses
├── configs/                   # one config per experiment, all inherit e0
├── tools/
│   ├── _runtime.py            # model + validation loader builder
│   ├── e3_benchmark_fps.py    # speed, batch size 1
│   ├── e4_q0_coverage.py      # initial-query coverage of GT lanes
│   └── e6_shape_modes.py      # shape modes per frame (labels only, CPU)
├── run_experiment.sh          # train + evaluate final epoch + collect results
├── run_analysis.sh            # run e3 / e4 / e6 + collect results
├── tests/                     # CPU unit tests for the add-ons
└── return_back/               # created on first run; results to send back
```

## Experiments

All training runs share one protocol (`configs/e0_full_50e.py`): the main
temporal config, 50 epochs, two GPUs with eight samples each, 500-iteration
linear warm-up, cosine annealing, seed 2026. Validation inside training runs
every 10 epochs, and the final-epoch checkpoint is evaluated.

Reduced protocol (`QUARTER=1`): the same schedule on every fourth training
frame, about a quarter of the iterations; evaluation still uses the full
validation split. Rows compared with each other must use the same protocol,
so a reduced ablation set needs its own E0 run (`QUARTER=1 ... e0`).

| ID | Config | Single change relative to E0 |
|---|---|---|
| e0 | `e0_full_50e.py` | none (full model under the shared protocol) |
| e1 | `e1_straight_templates_50e.py` | CQG templates fixed to straight lines; shape losses off |
| e2a | `e2a_history_image_only_50e.py` | LiDAR branch keeps only the current sweep |
| e2b | `e2b_history_lidar_only_50e.py` | MTFF samples only the current image |
| e2c | `e2c_mean_instead_of_tconv_50e.py` | masked temporal mean instead of TConv (optional) |
| e5 | `e5_three_decoder_layers_50e.py` | three decoder layers (optional) |

## Commands

Run from any directory; the scripts locate the repository themselves.

```bash
# quick check first (~10 min): 1 epoch on every 400th frame
SMOKE=1 bash ablation_extras/run_experiment.sh e1

# training experiments (defaults: GPUS=0,1, random port)
bash ablation_extras/run_experiment.sh e0
GPUS=2,3 PORT=29622 bash ablation_extras/run_experiment.sh e1

# reduced protocol (every 4th training frame, full validation)
QUARTER=1 bash ablation_extras/run_experiment.sh e0
QUARTER=1 GPUS=2,3 PORT=29622 bash ablation_extras/run_experiment.sh e1

# inference-only analyses (default GPU=0)
bash ablation_extras/run_analysis.sh e3 CONFIG CHECKPOINT mf
bash ablation_extras/run_analysis.sh e4 CONFIG CHECKPOINT full
bash ablation_extras/run_analysis.sh e6 CONFIG none train     # no GPU needed
```

E4 also scores straight longitudinal lines at the same 101 lateral seeds, which
is the initial query geometry of e1, so the coverage comparison with e1 needs
no training.

Each command ends with a SUCCESS or FAILED line and the path of the result
file or folder under `ablation_extras/return_back/`.

## Tests

```bash
PYTHONPATH="$(dirname "$PWD")" python -m pytest ablation_extras/tests -q
```

The tests run on CPU. When `spconv` or `mmdet3d` is missing, stand-ins are
used for those two imports only; all LaneLoom classes under test are real.
