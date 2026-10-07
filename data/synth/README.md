# Synthetic SPD(3) fields: isotropic vs. anisotropic

An example dataset in the same on-disk format as real data (NIfTI tensors and masks plus a
trainer ledger), used to demonstrate the pipeline and as an end-to-end smoke test.

Each sample is an 8x8x8 field in which every voxel is an i.i.d. draw from a
class-conditional distribution: random isotropic-ish tensors (label 0) or strongly
anisotropic tensors with a random orientation per voxel (label 1), with per-voxel scale
jitter so scale alone is not discriminative. See `generate.py` for the exact definition.
512 samples split 384 / 64 / 64 (train / val / test), class-balanced, fully deterministic.

From the repository root:

```bash
uv run python data/synth/generate.py                                    # writes tensors/, masks/, training/
uv run python -m experiments.train --config data/synth/configs/lcm.yaml   # or lem.yaml, aim.yaml
uv run python data/synth/check.py runs/synth/lcm                        # PASS if test accuracy >= 0.9
```

Generated files are not tracked; regenerate them with `generate.py`.
