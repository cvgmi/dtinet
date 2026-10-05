# DTINet

DTINet implements ManifoldNet primitives for SPD(3)-valued images (e.g. diffusion tensor
imaging), with pluggable metrics on the SPD manifold:

- **LCM** — log-Cholesky metric: everything reduces to Euclidean operations in log-Cholesky
  coordinates.
- **LEM** — log-Euclidean metric: Euclidean operations in (isometrically scaled) matrix-log
  coordinates.
- **AIM** — affine-invariant metric: Riemannian log/exp maps and (recursive and iterative)
  Fréchet means, following [cvgmi/manifold-net-dmri](https://github.com/cvgmi/manifold-net-dmri).

Theoretical basis: Chakraborty et al., *ManifoldNet: A Deep Neural Network for
Manifold-valued Data with Applications*, TPAMI 2022.

The installable package lives entirely under `src/dtinet/`; study-specific cohort
construction, cluster scripts, and protected data are intentionally kept outside this public
repository.

## Layout

```
src/dtinet/
  geometry/    # metric mathematics (SPD linear algebra, coordinate maps, AIM log/exp)
  layers/      # ManifoldNet layers, dispatching on the metric
  models.py    # ManifoldNetClassifier
  data.py      # synthetic SPD classification dataset
  train.py     # portable training interface for real DTI cohorts
examples/
  train_synthetic.py  # minimal two-class classification example
```

## Installation

```bash
uv sync
```

The package exposes the training command as `uv run train`.

## Quickstart (synthetic data)

```bash
uv run python examples/train_synthetic.py --metric lcm
```

The example trains a `ManifoldNetClassifier` on a synthetic two-class SPD(3) classification
task — fields of isotropic versus anisotropic diffusion tensors — and doubles as a smoke test,
exiting nonzero unless validation accuracy reaches 0.9. Use `--metric lem` or `--metric aim`
and `--device cuda` to exercise the other metrics on GPU.

## Data organization

Place local data under the repository's ignored `data/` directory. A typical
layout is:

```text
data/
├── training/
│   └── ledger.json
├── tensors/
│   ├── scan-001.nii.gz
│   └── scan-002.nii.gz
└── masks/
    ├── scan-001.nii.gz
    └── scan-002.nii.gz
```

`data/` is ignored by Git except for its `.gitkeep`; never commit protected scans,
subject identifiers, generated checkpoints, or real cohort ledgers.

Each ledger record points to files **relative to the configured data root**:

```json
{
  "image_id": "scan-001",
  "subject_id": "subject-001",
  "label": 0,
  "group": "Control",
  "split": "train",
  "tensor_path": "tensors/scan-001.nii.gz",
  "mask_path": "masks/scan-001.nii.gz"
}
```

Absolute paths and paths that escape the data root are rejected. The default data
root is `data/`; override it in YAML or at runtime:

```yaml
data:
  root: data
  ledger: training/ledger.json
```

```bash
uv run train --config configs/dtinetlc.example.yaml \
  --data-root /path/to/local/data
```

## Training

Copy the example configuration to a local, ignored YAML file and adjust it:

```bash
cp configs/dtinetlc.example.yaml configs/local.yaml
uv run train --config configs/local.yaml --validate-only
uv run train --config configs/local.yaml
```

The ledger must assign all scans from a subject to one split. Training writes
checkpoints, history, metrics, and scan- and subject-level predictions under the
configured output directory.
