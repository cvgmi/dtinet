#!/usr/bin/env python
"""
Generate the synthetic SPD(3) classification dataset in the trainer's ledger format.

Two classes of ``grid``^3 fields in which every voxel is an i.i.d. draw from a
class-conditional distribution on SPD(3), so a classifier must aggregate spatial statistics
rather than rely on any single voxel:

- ``isotropic`` (label 0): ``P = s^2 expm(noise S)`` with ``S`` a random symmetric matrix
  (i.i.d. standard normal entries); no preferred orientation.
- ``anisotropic`` (label 1): ``P = s^2 R diag(e^a, e^{-a/2}, e^{-a/2}) R^T`` with a random
  rotation ``R`` per voxel, perturbed as ``P <- P^{1/2} expm(noise S) P^{1/2}``.

In both classes the scale ``s = exp(scale_jitter z)``, ``z ~ N(0, 1)``, is drawn per voxel,
so scale alone is not discriminative. Sample ``i`` has label ``i % 2`` and is drawn from a
generator seeded with ``(seed, i)``, so the dataset is fully deterministic.

Writes, under the dataset directory (``data/synth`` by default):

- ``tensors/sample-XXXX.nii.gz``: float32 (X, Y, Z, 6) in Voigt order (xx, xy, yy, xz, yz, zz)
- ``masks/sample-XXXX.nii.gz``: all-ones uint8 masks
- ``training/ledger.json``: one record per sample with train/val/test splits

Run from the repository root: ``uv run python data/synth/generate.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dtinet.geometry.spd import matrix_to_voigt6, sym_expm, sym_sqrtm  # noqa: E402
from experiments.train import canonical_hash, validate_ledger  # noqa: E402

LABEL_MAP = {"isotropic": 0, "anisotropic": 1}


def rand_symmetric(n: int, generator: torch.Generator) -> torch.Tensor:
    """Symmetric (n, 3, 3) matrices with i.i.d. standard normal upper-triangular entries."""
    s = torch.zeros(n, 3, 3)
    idx = torch.triu_indices(3, 3)
    s[:, idx[0], idx[1]] = torch.randn(n, 6, generator=generator)
    return s + s.triu(diagonal=1).transpose(-1, -2)


def rand_rotation(n: int, generator: torch.Generator) -> torch.Tensor:
    """Haar-random (n, 3, 3) rotations via QR of Gaussian matrices, determinant +1."""
    q, r = torch.linalg.qr(torch.randn(n, 3, 3, generator=generator))
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)
    neg = torch.linalg.det(q) < 0
    q[neg, :, 0] = -q[neg, :, 0]
    return q


def sample_field(index: int, args: argparse.Namespace) -> tuple[np.ndarray, int]:
    """Return the (X, Y, Z, 6) float32 Voigt field and label of sample ``index``."""
    g = torch.Generator()
    g.manual_seed((args.seed + 1) * 1_000_003 + index)
    label = index % 2
    n = args.grid**3
    if label == 0:
        p = sym_expm(args.noise * rand_symmetric(n, g))
    else:
        a = args.anisotropy
        d = torch.tensor([[a, -a / 2, -a / 2]]).exp().expand(n, 3)
        rot = rand_rotation(n, g)
        p = (rot * d.unsqueeze(-2)) @ rot.transpose(-1, -2)
        p_sqrt = sym_sqrtm(p)
        p = p_sqrt @ sym_expm(args.noise * rand_symmetric(n, g)) @ p_sqrt
    s2 = torch.exp(args.scale_jitter * torch.randn(n, 1, 1, generator=g)).square()
    voigt = matrix_to_voigt6(s2 * p).float().numpy()
    return voigt.reshape(args.grid, args.grid, args.grid, 6), label


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--noise", type=float, default=0.3)
    parser.add_argument("--anisotropy", type=float, default=1.0)
    parser.add_argument("--scale-jitter", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.125)
    parser.add_argument("--test-fraction", type=float, default=0.125)
    args = parser.parse_args()

    for sub in ("tensors", "masks", "training"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    # contiguous blocks of even length keep every split exactly class-balanced
    n_val = 2 * round(args.val_fraction * args.num_samples / 2)
    n_test = 2 * round(args.test_fraction * args.num_samples / 2)
    n_train = args.num_samples - n_val - n_test

    mask = nib.Nifti1Image(np.ones((args.grid,) * 3, dtype=np.uint8), np.eye(4))
    records = {}
    for index in range(args.num_samples):
        field, label = sample_field(index, args)
        sample_id = f"sample-{index:04d}"
        split = "train" if index < n_train else "val" if index < n_train + n_val else "test"
        nib.save(nib.Nifti1Image(field, np.eye(4)), args.out / "tensors" / f"{sample_id}.nii.gz")
        nib.save(mask, args.out / "masks" / f"{sample_id}.nii.gz")
        records[sample_id] = {
            "image_id": sample_id,
            "subject_id": sample_id,
            "label": label,
            "group": next(k for k, v in LABEL_MAP.items() if v == label),
            "split": split,
            "tensor_path": f"tensors/{sample_id}.nii.gz",
            "mask_path": f"masks/{sample_id}.nii.gz",
        }

    split_subjects = {s: sorted(r["subject_id"] for r in records.values() if r["split"] == s)
                      for s in ("train", "val", "test")}
    ledger = {
        "schema_version": 1,
        "task": "synthetic_isotropic_vs_anisotropic",
        "label_map": LABEL_MAP,
        "split_summary": {
            s: {"scans": len(ids),
                "groups": {g: sum(records[i]["group"] == g for i in ids) for g in LABEL_MAP}}
            for s, ids in split_subjects.items()
        },
        "split_subjects": split_subjects,
        "records": records,
        "generator": {k: v for k, v in vars(args).items() if k != "out"},
    }
    ledger["ledger_sha256"] = canonical_hash(ledger)
    validate_ledger(ledger)
    (args.out / "training" / "ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print(f"Wrote {args.num_samples} samples to {args.out}: "
          + ", ".join(f"{s}={v['scans']}" for s, v in ledger["split_summary"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
