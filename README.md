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

## Installation

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

On Linux, `uv sync` installs the CUDA-enabled PyTorch wheels, so the same environment works
on CPU and GPU nodes — no extra setup needed.

## Repository layout

```
src/dtinet/             # the library: primitives only
  geometry/             #   metric mathematics (SPD linear algebra, coordinate maps, AIM log/exp)
  layers/               #   ManifoldNet layers (wFM, Fréchet batch norm, readouts, residual
                        #   block, BiMap), dispatching on the metric where applicable
experiments/            # code built on the primitives
  models.py             #   assembled classifiers (see "Models" below)
  train.py              #   config- and ledger-driven trainer
  configs/example.yaml  #   annotated example configuration
  tests/                #   model and trainer tests
data/
  synth/                # example dataset: generator, configs, smoke-test check (tracked)
  <your-dataset>/       # your own datasets (Git-ignored)
scripts/
  diagnose_eigh_cuda.py # GPU diagnostic for the cuSOLVER batched-eigh workaround
```

Everything under `experiments/` and `data/` is run from the repository root.

## Example dataset: synthetic tensor fields

A two-class task (fields of isotropic vs. anisotropic tensors) in the same on-disk format as
real data, so it exercises the full pipeline and doubles as a smoke test. Generate it once
(deterministic, a few MB), then train with any metric:

```bash
uv run python data/synth/generate.py
uv run python -m experiments.train --config data/synth/configs/lcm.yaml   # or lem.yaml, aim.yaml
uv run python data/synth/check.py runs/synth/lcm                          # PASS if test accuracy >= 0.9
```

The configs default to `device: cuda`; set `experiment.device: cpu` to run without a GPU.
AIM starts flat: its features begin near the identity, so accuracy sits at chance for the
first few epochs before converging. This is expected, which is why `aim.yaml` trains for
50 epochs at a higher learning rate with gradient clipping.

## Models

`experiments/models.py` assembles the primitives into classifiers, selected in a config with
`model.architecture`:

- `manifoldnet` — the original ManifoldNet stack: stride-2 weighted Fréchet mean (wFM)
  layers with Fréchet batch normalization, an invariant readout, and a linear head. All
  metrics.
- `manifold_resnet` — a residual ManifoldNet for coordinate metrics: a full-resolution stem,
  stages of a stride-2 wFM layer followed by residual blocks whose skip connection is a
  geodesic interpolation, and a readout pooling the mean and spread of each channel's
  distances. With `bimap: true` (LCM only), a voxelwise congruence `P -> W P W^T` after
  every normalization lets the network combine tensor components, which wFM layers and
  coordinate-wise activations cannot.
- `coord_resnet` — a Euclidean 3D ResNet with the same topology on the metric coordinates;
  a control without manifold constraints.

## Training on your own DTI data

1. **Organize the data** in its own folder under `data/` (Git-ignored, so protected scans
   and identifiers are never committed):

   ```text
   data/my-cohort/
   ├── training/
   │   └── ledger.json
   ├── tensors/          # one SPD(3) tensor volume per scan, NIfTI (X, Y, Z, 6)
   │   └── scan-001.nii.gz
   └── masks/
       └── scan-001.nii.gz
   ```

   Tensors are stored in the library's Voigt order (xx, xy, yy, xz, yz, zz) and should be
   expressed in the image (voxel) frame. Tools such as MRtrix write tensors in scanner
   coordinates; on oblique acquisitions rotate them with `R^T D R`, where `R` holds the
   unit-length columns of the affine.

2. **Write a ledger**: one JSON record per scan, paths relative to the dataset folder.
   Every scan of a subject must belong to exactly one of `train`/`val`/`test`. Tasks are
   binary: `label_map` maps exactly two group names to 0 and 1. `data/synth/generate.py` is
   a complete reference for the ledger's top-level fields.

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

3. **Configure**: copy `experiments/configs/example.yaml` (for example into the dataset
   folder or the Git-ignored `configs/`) and adjust. Notable options:

   - `model.architecture` and `model.metric` (set `activation: null` for `aim`).
   - `data.tensor_scale`: tensors fitted from b-values in s/mm² come out in mm²/s (~1e-3 in
     brain), where the log-Cholesky off-diagonal (orientation) coordinates are ~70× smaller
     than the diagonal ones; use `1000` to train in µm²/ms.
   - `data.flip_axes`: random reflections of the listed image axes during training (e.g.
     `[0]` for left-right), applied correctly to the tensors; requires image-frame tensors.
   - `training.selection_metric` (`balanced_accuracy` or `roc_auc`) and
     `training.grad_clip`.

4. **Validate, then train**:

   ```bash
   uv run python -m experiments.train --config my-config.yaml --validate-only
   uv run python -m experiments.train --config my-config.yaml
   ```

   Use `--data-root /path/to/dataset` to override the configured root and `--resume
   path/to/checkpoint.pt` to continue an interrupted run. Training writes checkpoints,
   history, metrics, and scan- and subject-level predictions under the configured
   `experiment.output_dir` (`runs/`, Git-ignored). A resumed run does not replay the
   random flip augmentation exactly as an uninterrupted run would.

## Development

```bash
uv run pytest                 # library tests (tests/) and model/trainer tests (experiments/tests/)
uvx ruff check .              # lint
```

CUDA note: `torch.linalg.eigh` on CUDA fails for batches of 2¹⁶ or more matrices
(cuSOLVER `syevBatched` limit), which large SPD fields easily exceed. The library works
around this transparently by chunking CUDA eigensolves (with a CPU fallback); see
`src/dtinet/geometry/spd.py` and `scripts/diagnose_eigh_cuda.py`.
