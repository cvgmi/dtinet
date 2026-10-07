"""End-to-end training smoke test of the AIM ManifoldNet classifier."""

import torch
import torch.nn as nn

from dtinet.geometry.spd import matrix_to_voigt6, sym_expm, sym_logm
from experiments.models import ManifoldNetClassifier


def rand_sym(*batch_shape, scale=1.0, seed=None):
    """
    Random symmetric matrices of shape (*batch_shape, 3, 3).
    """
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    a = torch.randn(*batch_shape, 3, 3, generator=g)
    return scale * (a + a.transpose(-1, -2)) / 2


def _synthetic_spd_volume(label, D=4, seed=0):
    """
    One synthetic SPD volume of shape (1, 1, 6, D, D, D) for a separable two-class problem.
    """
    g = torch.Generator().manual_seed(seed)
    if label == 0:
        s = torch.randn(3, 3, generator=g)
        s = (s + s.T) / 2
        base = sym_expm(0.3 * s)
    else:
        q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g))
        base = q @ torch.diag(torch.tensor([torch.e, torch.e**-0.5, torch.e**-0.5])) @ q.T
    log_base = sym_logm(base)
    noise = rand_sym(D, D, D, scale=0.05, seed=seed + 1)
    voxels = sym_expm(log_base + noise)
    return matrix_to_voigt6(voxels).movedim(-1, 0).unsqueeze(0).unsqueeze(0)


def test_manifoldnet_aim_training_smoke():
    torch.manual_seed(0)
    model = ManifoldNetClassifier(
        num_classes=2, num_layers=1, num_channels=[2], metric="aim", kernel_size=3, padding=1
    )
    x = torch.cat([_synthetic_spd_volume(i % 2, seed=10 * i) for i in range(4)], dim=0)
    labels = torch.tensor([0, 1, 0, 1])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    model.train()
    for _ in range(3):
        optimizer.zero_grad()
        logits = model(x)
        assert torch.isfinite(logits).all()
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss)
        loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[1] < losses[0]
