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


@pytest.mark.parametrize(
    "architecture", ["manifold_resnet", "manifold_resnet_bimap", "coord_resnet"]
)
def test_resnet_models_respect_mask_and_train(architecture):
    from experiments.models import CoordResNetClassifier, ManifoldResNetClassifier

    torch.manual_seed(0)
    if architecture == "coord_resnet":
        model = CoordResNetClassifier(2, 4, [8, 16], blocks_per_stage=1)
    else:
        model = ManifoldResNetClassifier(
            2, 4, [8, 16], blocks_per_stage=1, bimap=architecture.endswith("bimap")
        )
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
def test_resnets_reject_invalid_kernels(kernel_size):
    from experiments.models import CoordResNetClassifier, ManifoldResNetClassifier

    for cls in (ManifoldResNetClassifier, CoordResNetClassifier):
        with pytest.raises(ValueError, match="odd"):
            cls(2, 4, [8], kernel_size=kernel_size)
