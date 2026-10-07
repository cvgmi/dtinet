"""Tests for the assembled coordinate-metric (LCM/LEM) classifiers and trainer helpers."""

import pytest
import torch

from dtinet.geometry.lcm import exp_cholesky_coords
from experiments.models import ManifoldNetClassifier


def _spd_field(batch: int, size: int, generator: torch.Generator) -> torch.Tensor:
    z = 0.3 * torch.randn(batch, 1, size, size, size, 6, generator=generator)
    return exp_cholesky_coords(z).movedim(-1, 2)


@pytest.mark.parametrize("metric", ["lcm", "lem"])
def test_channels_differ_at_initialization(metric):
    # constant pre-softmax weights make every channel the same uniform filter, collapsing the
    # readout to a single repeated feature per sample
    torch.manual_seed(0)
    x = _spd_field(4, 16, torch.Generator().manual_seed(1))
    mask = torch.ones(4, 1, 16, 16, 16)
    model = ManifoldNetClassifier(
        2, 3, [4, 8, 8], metric=metric, activation="elu", kernel_size=3, stride=2, padding=1
    )
    features = {}
    model.head.register_forward_hook(lambda module, args, out: features.update(x=args[0]))
    model(x, mask)
    spread = features["x"].max(dim=1).values - features["x"].min(dim=1).values
    assert (spread > 1e-3).all()


def test_zero_init_std_reproduces_uniform_filters():
    model = ManifoldNetClassifier(2, 2, [4, 4], metric="lcm", kernel_size=3, init_std=0.0)
    for layer in model.encoder:
        if hasattr(layer, "constrain_weight"):
            weight = layer.constrain_weight()
            assert torch.allclose(weight, torch.full_like(weight, 1.0 / weight[0].numel()))


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_flip_sign_rule_matches_reflection(axis):
    from dtinet.geometry.spd import matrix_to_voigt6, voigt6_to_matrix
    from experiments.train import VOIGT6_OFFDIAG_INVOLVING

    torch.manual_seed(axis)
    a = torch.randn(10, 3, 3, dtype=torch.float64)
    m = a @ a.transpose(-1, -2)
    reflection = torch.eye(3, dtype=torch.float64)
    reflection[axis, axis] = -1
    expected = matrix_to_voigt6(reflection @ m @ reflection)
    flipped = matrix_to_voigt6(m).clone()
    flipped[..., VOIGT6_OFFDIAG_INVOLVING[axis]] *= -1
    assert torch.allclose(flipped, expected)
    assert torch.allclose(voigt6_to_matrix(flipped), reflection @ m @ reflection)


@pytest.mark.parametrize("bimap", [False, True])
def test_manifold_resnet_respects_mask_and_trains(bimap):
    from experiments.models import ManifoldResNetClassifier

    torch.manual_seed(0)
    model = ManifoldResNetClassifier(2, 4, [8, 16], blocks_per_stage=1, bimap=bimap)
    model.eval()
    generator = torch.Generator().manual_seed(3)
    x = _spd_field(2, 12, generator)
    mask = torch.zeros(2, 1, 12, 12, 12)
    mask[:, :, 2:10, 2:10, 2:10] = 1
    out = model(x, mask)
    assert out.shape == (2, 2) and torch.isfinite(out).all()

    # values outside the mask must not change the prediction
    noise = _spd_field(2, 12, generator)
    x_outside = torch.where(mask.unsqueeze(2).bool(), x, noise)
    assert torch.allclose(model(x_outside, mask), out, atol=1e-5)

    # every parameter group receives a finite gradient in training mode
    model.train()
    model(x, mask).sum().backward()
    for name, parameter in model.named_parameters():
        if "running" in name:
            continue
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("kernel_size", [0, 2, 4])
def test_manifold_resnet_rejects_invalid_kernels(kernel_size):
    from experiments.models import ManifoldResNetClassifier

    with pytest.raises(ValueError, match="odd"):
        ManifoldResNetClassifier(2, 4, [8], kernel_size=kernel_size)


def test_cached_dataset_matches_uncached_and_is_not_mutated(tmp_path):
    import nibabel as nib
    import numpy as np

    from experiments.train import DTILedgerDataset

    rng = np.random.default_rng(0)
    records = {}
    for i in range(2):
        a = rng.standard_normal((6, 5, 4, 3, 3))
        m = a @ np.swapaxes(a, -1, -2) + np.eye(3)
        voigt = np.stack([m[..., 0, 0], m[..., 0, 1], m[..., 1, 1], m[..., 0, 2],
                          m[..., 1, 2], m[..., 2, 2]], axis=-1).astype(np.float32)
        nib.save(nib.Nifti1Image(voigt, np.eye(4)), tmp_path / f"t{i}.nii.gz")
        mask = nib.Nifti1Image(np.ones((6, 5, 4), np.uint8), np.eye(4))
        nib.save(mask, tmp_path / f"m{i}.nii.gz")
        records[f"s{i}"] = {"image_id": f"s{i}", "subject_id": f"s{i}", "label": i,
                            "group": "g", "split": "train", "tensor_path": f"t{i}.nii.gz",
                            "mask_path": f"m{i}.nii.gz"}
    ledger = {"records": records}
    args = (ledger, "train", tmp_path, (6, 5, 4), 1.0, 1000.0)
    plain = DTILedgerDataset(*args)
    cache = {}
    cached = DTILedgerDataset(*args, (0, 1, 2), cache=cache)
    reference = [plain[i]["tensor"].clone() for i in range(2)]
    torch.manual_seed(0)
    for _ in range(4):  # repeated random flips must never alter the cached arrays
        for i in range(2):
            cached[i]
    unflipped = DTILedgerDataset(*args, cache=cache)
    for i in range(2):
        assert torch.equal(unflipped[i]["tensor"], reference[i])
    assert len(cache) == 2
