"""Regression tests for the audit-found defects:

1. Spectral-function backward at degenerate eigenvalues must use the Loewner
   divided-difference limit f'(lambda), not zero (previously: gradients along
   eigenvector directions vanished at isotropic matrices such as the identity).
2. Normalization by tiny-but-positive masses must divide by the true mass, not
   an epsilon floor (Karcher weights, masked convolution windows).
"""

import torch

from dtinet.geometry.aim import karcher_wfm
from dtinet.geometry.spd import sym_expm, sym_logm, sym_powm, sym_sqrtm
from dtinet.layers.wfm import WeightedFrechetMean3dCoords


def _sym(x: torch.Tensor) -> torch.Tensor:
    return (x + x.transpose(-1, -2)) / 2


def test_logm_gradient_at_identity_is_full_jacobian_transpose():
    # d/dM <G, log M> at M = I is f'(1) * G = G for symmetric G; the old
    # degeneracy-zeroing backward returned only the diagonal of G.
    g = _sym(torch.randn(3, 3, dtype=torch.float64))
    m = torch.eye(3, dtype=torch.float64).requires_grad_(True)

    grad = torch.autograd.grad(sym_logm(m), m, grad_outputs=g)[0]

    torch.testing.assert_close(grad, g, rtol=1e-9, atol=1e-12)


def test_expm_gradient_at_zero_is_full_jacobian_transpose():
    g = _sym(torch.randn(3, 3, dtype=torch.float64))
    m = torch.zeros(3, 3, dtype=torch.float64).requires_grad_(True)

    grad = torch.autograd.grad(sym_expm(m), m, grad_outputs=g)[0]

    # f'(0) = exp(0) = 1
    torch.testing.assert_close(grad, g, rtol=1e-9, atol=1e-12)


def test_spectral_backward_matches_finite_differences_at_degenerate_eigenvalues():
    torch.manual_seed(0)
    base = torch.diag(torch.tensor([2.0, 2.0, 1.0], dtype=torch.float64))
    g = _sym(torch.randn(3, 3, dtype=torch.float64))
    e = _sym(torch.randn(3, 3, dtype=torch.float64))
    h = 1e-6

    for fn in (sym_logm, sym_sqrtm, lambda m: sym_powm(m, 2.0)):
        m = base.clone().requires_grad_(True)
        grad = torch.autograd.grad(fn(m), m, grad_outputs=g)[0]
        fd = ((g * (fn(base + h * e) - fn(base - h * e))).sum() / (2 * h)).item()

        assert abs((grad * e).sum().item() - fd) < 1e-6 * max(1.0, abs(fd)), fn


def test_karcher_wfm_invariant_to_tiny_weight_scales():
    eye = torch.eye(3, dtype=torch.float64)
    x = torch.stack([eye, 4 * eye, 16 * eye])  # (K, 3, 3)
    w = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)

    m1 = karcher_wfm(x, w, dim=-3)
    m2 = karcher_wfm(x, w * 1e-18, dim=-3)

    torch.testing.assert_close(m2, m1, rtol=1e-6, atol=1e-9)


def test_powm_tensor_exponent_receives_gradient():
    torch.manual_seed(2)
    a = torch.randn(3, 3, dtype=torch.float64)
    base = _sym(a @ a + torch.eye(3, dtype=torch.float64))
    g = _sym(torch.randn(3, 3, dtype=torch.float64))
    h = 1e-6

    p = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    grad = torch.autograd.grad(sym_powm(base, p), p, grad_outputs=g)[0]
    fd = ((g * (sym_powm(base, p.detach() + h) - sym_powm(base, p.detach() - h))).sum() / (2 * h))

    torch.testing.assert_close(grad, fd, rtol=1e-6, atol=1e-8)


def test_masked_conv_normalizes_tiny_surviving_mass():
    # softmax mass on the single surviving voxel is ~e^-20, far below float32 eps;
    # the output must still be the correct convex combination (the input value).
    layer = WeightedFrechetMean3dCoords(1, 1, kernel_size=3, padding=1).double()
    with torch.no_grad():
        layer.weight.fill_(0.0)
        layer.weight[0, 0, 0, 0, 0] = -20.0  # corner voxel: tiny softmax share

    x = torch.full((1, 1, 2, 3, 3, 3), 3.0, dtype=torch.float64)
    mask = torch.zeros(1, 1, 3, 3, 3, dtype=torch.float64)
    mask[0, 0, 0, 0, 0] = 1.0

    y, out_mask = layer(x, mask)

    assert bool(out_mask[0, 0, 0, 0, 0])
    torch.testing.assert_close(
        y[0, 0, :, 0, 0, 0], torch.full((2,), 3.0, dtype=torch.float64), rtol=1e-6, atol=1e-9
    )
