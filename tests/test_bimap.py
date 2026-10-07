"""Tests for the voxelwise BiMap (congruence) layer under the log-Cholesky metric."""

import pytest
import torch


def test_bimap_matches_congruence_and_identity():
    from dtinet.geometry.lcm import exp_cholesky_coords, log_cholesky_coords
    from dtinet.geometry.spd import voigt6_to_matrix
    from dtinet.layers import BiMap3dLC

    torch.manual_seed(0)
    z = 0.5 * torch.randn(2, 3, 6, 4, 4, 4, dtype=torch.float64)
    layer = BiMap3dLC(3).double()
    with torch.no_grad():
        layer.delta.zero_()
    assert torch.allclose(layer(z), z, atol=1e-10)

    with torch.no_grad():
        layer.delta.copy_(torch.randn(3, 3, 3, dtype=torch.float64))
    out = voigt6_to_matrix(exp_cholesky_coords(layer(z).movedim(2, -1)))
    p = voigt6_to_matrix(exp_cholesky_coords(z.movedim(2, -1)))
    w = layer.weight.view(1, 3, 1, 1, 1, 3, 3)
    assert torch.allclose(out, w @ p @ w.transpose(-1, -2), atol=1e-8)
    # round trip stays in log-Cholesky coordinates of an SPD field
    assert torch.allclose(
        log_cholesky_coords(exp_cholesky_coords(layer(z).movedim(2, -1))),
        layer(z).movedim(2, -1),
        atol=1e-8,
    )


def test_bimap_ill_conditioned_weight_stays_accurate():
    from dtinet.geometry.lcm import exp_cholesky_coords
    from dtinet.geometry.spd import voigt6_to_matrix
    from dtinet.layers import BiMap3dLC

    torch.manual_seed(1)
    z = 0.5 * torch.randn(1, 1, 6, 3, 3, 3)
    layer = BiMap3dLC(1)
    target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1e-3]])
    with torch.no_grad():
        layer.delta.copy_((target - torch.eye(3)).unsqueeze(0))
    out = voigt6_to_matrix(exp_cholesky_coords(layer(z).movedim(2, -1)).double())
    p = voigt6_to_matrix(exp_cholesky_coords(z.double().movedim(2, -1)))
    w = target.double()
    expected = w @ p @ w.T
    assert torch.allclose(out, expected, rtol=1e-3, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="autocast float16 needs CUDA")
def test_bimap_is_finite_under_autocast():
    from dtinet.layers import BiMap3dLC

    z = torch.zeros(1, 1, 6, 2, 2, 2, device="cuda", requires_grad=True)
    with torch.no_grad():
        z[:, :, 2] = -10.0  # tiny l22: underflows in float16
    layer = BiMap3dLC(1).cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        out = layer(z)
    out.sum().backward()
    assert torch.isfinite(out).all() and torch.isfinite(z.grad).all()
