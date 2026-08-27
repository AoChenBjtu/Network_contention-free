# Network contention-free sparse recovery

TensorFlow 1.x implementations for sparse signal recovery experiments. The
repository contains model-based LISTA and LAMP networks together with two
black-box baselines: a fully connected DNN and a 15-stage fully connected
ResNet.

## Models

- `LISTA.py`: 15-layer LISTA experiment.
- `LAMP.py`: 15-layer LAMP experiment.
- `DNN.py`: 15-layer fully connected DNN baseline.
- `ResNet.py`: 15-stage residual MLP baseline with layer-wise NMSE reporting.
- `tools/`: problem generators, network builders, shrinkage functions, and
  the shared training/testing pipeline.

## Environment

The experiment code targets Python 3, NumPy, and TensorFlow 1.12/1.13. Some
optional problem generators also use SciPy and h5py.

## Running an experiment

Run a model entry point from the repository root, for example:

```bash
python ResNet.py
```

ResNet trains stages 1 through 15 end-to-end using the shared greedy extension
and refinement schedule. Its final checkpoint is written to
`ResNet15_stage_aligned_gated.npz`, and per-stage test NMSE is written to
`ResNet15_layer_nmse.csv`. Generated checkpoints and local experiment outputs
are excluded from version control.
