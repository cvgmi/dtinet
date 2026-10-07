<h1 align="center">dtinet</h1>

<p align="center">
  <b>ManifoldNet for SPD(3)-valued images, with pluggable Riemannian metrics</b><br>
  A clean PyTorch reimplementation of ManifoldNet's building blocks for fields of 3×3
  symmetric positive-definite matrices, such as diffusion tensor images.
</p>

<p align="center">
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-3776AB">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C">
  <img alt="License: BSD-3-Clause" src="https://img.shields.io/badge/license-BSD--3--Clause-blue">
</p>

---

## Overview

ManifoldNet ([Chakraborty et al., TPAMI 2022](#references)) builds convolution-like networks
for manifold-valued data out of **weighted Fréchet means** (wFM): every layer outputs points on
the manifold, and a final **invariant layer** turns them into Euclidean features. `dtinet`
implements these primitives for 3D images whose voxels are SPD(3) matrices, under three
metrics:

| Metric | Key | Geometry | How layers are computed |
|---|---|---|---|
| **Log-Cholesky** | `lcm` | Flat chart via the Cholesky factor ([Lin, 2019](#references)) | Euclidean operations in log-Cholesky coordinates |
| **Log-Euclidean** | `lem` | Flat chart via the matrix logarithm ([Arsigny et al., 2006](#references)) | Euclidean operations in (isometrically scaled) matrix-log coordinates |
| **Affine-invariant** | `aim` | Curved, affine-invariant ([Pennec et al., 2006](#references)) | Riemannian log/exp maps and recursive Fréchet means, following the [reference implementation](https://github.com/cvgmi/manifold-net-dmri) |

Under `lcm` and `lem` a wFM is an ordinary weighted average in coordinates, so a wFM layer
becomes a single convex-weighted `conv3d`: fast, exact, and memory-light. `aim` keeps the full
Riemannian geometry at higher cost.

Tensors are stored in Voigt order `(xx, xy, yy, xz, yz, zz)` with shape
`(batch, channels, 6, D, H, W)`, and every layer accepts an optional binary mask so that
background voxels never contaminate a mean.

## Installation

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/cvgmi/dtinet.git
cd dtinet
uv sync
```

On Linux, `uv sync` installs CUDA-enabled PyTorch wheels, so one environment serves CPU and
GPU machines.

## Quick start

The primitives compose like ordinary PyTorch layers:

```python
import torch
from dtinet.geometry.lcm import exp_cholesky_coords
from dtinet.layers import (
    FrechetBatchNorm3d, InvariantReadout, SPDToCoords, WeightedFrechetMean3d,
)

# a batch of random SPD fields: (batch=2, channels=1, 6, 16, 16, 16)
x = exp_cholesky_coords(0.3 * torch.randn(2, 1, 16, 16, 16, 6)).movedim(-1, 2)
mask = torch.ones(2, 1, 16, 16, 16)

to_coords = SPDToCoords("lcm")                       # SPD -> log-Cholesky coordinates
wfm = WeightedFrechetMean3d(1, 8, kernel_size=3, metric="lcm", stride=2, padding=1)
bn = FrechetBatchNorm3d(8, metric="lcm")
readout = InvariantReadout("lcm")                    # distances to the Fréchet mean

z, mask = wfm(to_coords(x, mask), mask)              # wFM layers also downsample the mask
features = readout(bn(z, mask), mask)                # (2, 8, 8, 8, 8) invariant field
```

A complete classifier is available as `experiments.models.ManifoldNetClassifier`:

```python
from experiments.models import ManifoldNetClassifier

model = ManifoldNetClassifier(
    num_classes=2, num_layers=2, num_channels=[4, 8], metric="lcm",
    activation="relu", kernel_size=3, padding=1,
)
logits = model(x, torch.ones(2, 1, 16, 16, 16))      # (2, 2)
```

## What's inside

**Geometry** (`dtinet.geometry`)

| Module | Contents |
|---|---|
| `spd` | Voigt ↔ matrix conversion; matrix log, exp, square root and powers with an exact spectral backward pass that stays correct at repeated eigenvalues |
| `lcm`, `lem` | Coordinate maps and their inverses |
| `aim` | Log/exp maps, distances, parallel transport, recursive and Karcher weighted Fréchet means |

**Layers** (`dtinet.layers`): each dispatches on `metric=`

| Layer | Role | From |
|---|---|---|
| `SPDToCoords` | SPD field → metric coordinates (`lcm`, `lem`) | ManifoldNet |
| `WeightedFrechetMean3d` | Convex-weighted Fréchet mean over a 3D window: the manifold "convolution" | ManifoldNet |
| `InvariantReadout` | Distance of each voxel to the sample's Fréchet mean: a metric-invariant scalar field | ManifoldNet |
| `FrechetBatchNorm3d` | Riemannian batch normalization | [Brooks et al., 2019](#references) |
| `ManifoldResidualBlock3d` | Residual block whose skip connection is a geodesic interpolation | extension |
| `BiMap3dLC` | Voxelwise congruence `P ↦ W P Wᵀ` | [Huang & Van Gool, 2017](#references) |

**Models** (`experiments.models`), selected in a config with `model.architecture`

- `manifoldnet` — the ManifoldNet stack: wFM layers, Fréchet batch norm, invariant readout,
  linear head. All metrics.
- `manifold_resnet` — a residual ManifoldNet for `lcm`/`lem` (see
  [below](#beyond-the-original-manifoldnet)), optionally with BiMap layers.

## Example: synthetic tensor fields

`data/synth/` is a small, fully deterministic example dataset: 8³ fields of isotropic vs.
anisotropic tensors (512 samples), stored in the same on-disk format as real data, so it
exercises the whole pipeline and doubles as a smoke test.

```bash
uv run python data/synth/generate.py                                     # ~ a few MB
uv run python -m experiments.train --config data/synth/configs/lcm.yaml   # or lem.yaml, aim.yaml
uv run python data/synth/check.py runs/synth/lcm                         # PASS at test accuracy ≥ 0.9
```

Typical test accuracy is 0.97–1.00 for every metric. The configs assume a GPU; set
`experiment.device: cpu` otherwise. AIM features start near the identity, so its accuracy
stays at chance for the first epochs before converging, which is why its config trains
longer at a higher learning rate.

## Training on your own data

The trainer reads a **ledger**: a JSON file listing every scan, its label, and its
train/val/test split. Tasks are binary.

<details>
<summary><b>1. Lay out the data</b></summary>

```text
data/my-dataset/                 # Git-ignored
├── training/ledger.json
├── tensors/scan-001.nii.gz      # (X, Y, Z, 6) float32, Voigt order, image (voxel) frame
└── masks/scan-001.nii.gz        # binary brain mask on the same grid
```

Express tensors in the image frame. Tools such as MRtrix write tensors in scanner coordinates;
on oblique acquisitions rotate them with `Rᵀ D R`, where `R` holds the unit-length columns of
the affine.
</details>

<details>
<summary><b>2. Write the ledger</b></summary>

One record per scan, with paths relative to the dataset folder. All scans of a subject must
share one split. `data/synth/generate.py` is a complete reference for the top-level fields
(`label_map`, `split_summary`, `ledger_sha256`, …).

```json
{
  "image_id": "scan-001",
  "subject_id": "subject-001",
  "label": 0,
  "group": "control",
  "split": "train",
  "tensor_path": "tensors/scan-001.nii.gz",
  "mask_path": "masks/scan-001.nii.gz"
}
```
</details>

<details>
<summary><b>3. Configure, validate, train</b></summary>

Start from the annotated [`experiments/configs/example.yaml`](experiments/configs/example.yaml):

```bash
uv run python -m experiments.train --config my-config.yaml --validate-only
uv run python -m experiments.train --config my-config.yaml
```

Runs write checkpoints, per-epoch history, final metrics, and scan- and subject-level
predictions to `experiment.output_dir`. Resume with `--resume path/to/last.pt`.
</details>

Useful options:

| Option | Purpose |
|---|---|
| `data.tensor_scale` | Rescale tensors on load. Diffusion tensors in mm²/s (~10⁻³) make log-Cholesky orientation coordinates ~70× smaller than the diagonal ones; `1000` converts to µm²/ms. |
| `data.flip_axes` | Random reflections of the listed image axes, applied correctly to the tensor components. |
| `data.cache_in_memory` | Decompress each volume once and keep it in RAM, so training needs about one CPU. |
| `training.selection_metric` | Checkpoint selection on validation `balanced_accuracy` or `roc_auc`. |
| `training.grad_clip`, `training.lr_schedule` | Gradient-norm clipping; cosine or constant learning rate after warmup. |

## Beyond the original ManifoldNet

`dtinet` follows the paper's core construction (wFM layers and an invariant final layer) but
departs from it where that made networks easier to train or the code more general. Each
departure is opt-in or reduces exactly to the original behaviour.

**Metrics and normalization**
- **Coordinate metrics.** Besides the affine-invariant metric, `lcm` and `lem` turn every wFM
  into an exact, convex-weighted convolution in coordinates, which is much faster and uses less memory.
- **Fréchet batch normalization** ([Brooks et al., 2019](#references)) centers each channel at
  its batch Fréchet mean and rescales geodesic dispersion. The AIM version transports tangent
  vectors directly between means rather than through the identity (both are isometries).
- **Coordinate-wise activations** (`relu`, `elu`, …) are available for `lcm`/`lem`; the paper
  relies on the nonlinearity of the wFM itself.

**Trainability**
- **Symmetry-breaking initialization.** Constant pre-softmax wFM weights make every output
  channel the same uniform filter, and channels never diverge. Weights are drawn from
  `N(0, init_std²)`; `init_std=0` restores uniform filters.
- **Residual blocks** (`ManifoldResidualBlock3d`). The skip connection is the geodesic
  interpolation `(1 − a)·x + a·F(x)` with a learned per-channel weight `a`, so every operation
  is still a Fréchet mean.
- **BiMap layers** (`BiMap3dLC`, [Huang & Van Gool, 2017](#references)). wFM layers and
  coordinate-wise activations never mix a tensor's components with each other; a voxelwise
  congruence `P ↦ W P Wᵀ` does, while keeping every feature SPD. It is computed through the
  Cholesky factor with modified Gram–Schmidt, always in float32, with `W = I + Δ` so that
  weight decay pulls it toward the identity.
- **Richer readout.** `manifold_resnet` pools the mean *and* spread of each channel's distance
  field, not just the mean.

**Practicalities**
- **Masks everywhere.** Means renormalize over valid voxels, so background never leaks into
  features.
- **Robust linear algebra.** The spectral backward pass is exact at repeated eigenvalues
  (e.g. isotropic tensors), and batched CUDA eigensolves are chunked around a cuSOLVER limit
  (see below).

## Repository layout

```text
src/dtinet/          the library: geometry/ and layers/ (primitives only)
experiments/         models.py, train.py, configs/example.yaml, tests/
data/synth/          example dataset: generator, configs, smoke-test check
tests/               library tests
scripts/             diagnose_eigh_cuda.py
```

`experiments` is a top-level package rather than part of the installed library, so run the
examples and the trainer from the repository root.

## Development

```bash
uv run pytest            # library tests and model/trainer tests
uvx ruff check .         # lint
```

> [!NOTE]
> `torch.linalg.eigh` on CUDA fails for batches of 2¹⁶ or more matrices (cuSOLVER
> `syevBatched`), which large SPD fields easily exceed. `dtinet` chunks CUDA eigensolves
> transparently and falls back to the CPU if needed; `scripts/diagnose_eigh_cuda.py` checks a
> given GPU.

## References

- R. Chakraborty, J. Bouza, J. H. Manton, B. C. Vemuri. *ManifoldNet: A Deep Neural Network
  for Manifold-valued Data with Applications.* IEEE TPAMI, 2022.
- Z. Lin. *Riemannian Geometry of Symmetric Positive Definite Matrices via Cholesky
  Decomposition.* SIAM J. Matrix Anal. Appl., 2019.
- V. Arsigny, P. Fillard, X. Pennec, N. Ayache. *Log-Euclidean Metrics for Fast and Simple
  Calculus on Diffusion Tensors.* Magn. Reson. Med., 2006.
- X. Pennec, P. Fillard, N. Ayache. *A Riemannian Framework for Tensor Computing.* IJCV, 2006.
- D. Brooks, O. Schwander, F. Barbaresco, J.-Y. Schneider, M. Cord. *Riemannian Batch
  Normalization for SPD Neural Networks.* NeurIPS, 2019.
- Z. Huang, L. Van Gool. *A Riemannian Network for SPD Matrix Learning.* AAAI, 2017.

## Citation and license

If you use `dtinet`, please cite it via [`CITATION.cff`](CITATION.cff) (GitHub's
"Cite this repository" button) along with the ManifoldNet paper. Released under the
[BSD 3-Clause License](LICENSE).
