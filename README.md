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

## Synthetic demo

A self-contained two-class task (fields of isotropic vs. anisotropic tensors) that doubles
as a smoke test: it exits nonzero unless validation accuracy reaches 0.9.

```bash
uv run python examples/train_synthetic.py --metric lcm
uv run python examples/train_synthetic.py --metric lem
uv run python examples/train_synthetic.py --metric aim --epochs 50 --lr 1e-2
```

Add `--device cuda` on a GPU machine (it is the default when CUDA is available).

Reference results: LCM and LEM reach val acc ≥ 0.99 within the default 10 epochs (~10 s on
CPU, faster on GPU). AIM converges more slowly per step and from a flat start (features
begin near the identity, so early gradients are tiny) — expect chance-level accuracy for the
first few epochs, then rapid convergence to 1.00; use ~50 epochs and `--lr 1e-2` as above.

## Training on your own DTI data

1. **Organize the data** under a data root (default `./data`, Git-ignored — never commit
   protected scans or identifiers):

   ```text
   data/
   ├── training/
   │   └── ledger.json
   ├── tensors/          # one SPD(3) tensor volume per scan, NIfTI
   │   └── scan-001.nii.gz
   └── masks/
       └── scan-001.nii.gz
   ```

2. **Write a ledger**: one JSON record per scan, paths relative to the data root. Every
   scan of a subject must belong to exactly one of `train`/`val`/`test`.

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

3. **Configure**: copy the example config (Git-ignored once renamed) and adjust. Choose the
   metric under `model.metric` (`lcm`, `lem`, or `aim`; set `activation: null` for `aim`).

   ```bash
   cp configs/dtinetlc.example.yaml configs/local.yaml
   ```

4. **Validate, then train**:

   ```bash
   uv run train --config configs/local.yaml --validate-only
   uv run train --config configs/local.yaml
   ```

   Use `--data-root /path/to/data` to override the configured root and `--resume
   path/to/checkpoint.pt` to continue an interrupted run. Training writes checkpoints,
   history, metrics, and scan- and subject-level predictions under the configured
   `experiment.output_dir`.

## Layout

```
src/dtinet/
  geometry/    # metric mathematics (SPD linear algebra, coordinate maps, AIM log/exp)
  layers/      # ManifoldNet layers, dispatching on the metric
  models.py    # ManifoldNetClassifier (metric=...); DTINetLC legacy alias
  data.py      # synthetic SPD classification dataset
  train.py     # config-driven training interface for real DTI cohorts
examples/
  train_synthetic.py  # synthetic demo / smoke test
scripts/
  diagnose_eigh_cuda.py  # GPU diagnostic for the cuSOLVER batched-eigh workaround
```

## Development

```bash
uv run pytest tests/          # 33 tests: geometry identities, masking, eigh backward
uvx ruff check src/ tests/    # lint
```

CUDA note: `torch.linalg.eigh` on CUDA fails for batches of 2¹⁶ or more matrices
(cuSOLVER `syevBatched` limit), which large SPD fields easily exceed. The library works
around this transparently by chunking CUDA eigensolves (with a CPU fallback); see
`src/dtinet/geometry/spd.py` and `scripts/diagnose_eigh_cuda.py`.
