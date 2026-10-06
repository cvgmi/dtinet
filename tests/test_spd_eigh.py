"""Tests for the cuSOLVER batch-limit workaround in dtinet.geometry.spd."""

import torch

from dtinet.geometry.spd import _eigh_slabs, _linalg_eigh, sym_sqrtm


def _rand_spd(batch_shape: tuple[int, ...], seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(*batch_shape, 3, 3, generator=g) * 0.15
    return f @ f.transpose(-1, -2) + torch.eye(3) * 0.5


def test_eigh_slabs_match_reference_with_tiny_chunk():
    m = _rand_spd((7,), seed=1)
    ref_evals, ref_evecs = torch.linalg.eigh(m)

    evals, evecs = _eigh_slabs(m, chunk=3)

    # eigenvectors are only defined up to sign; compare the reconstructed matrix
    recon = (evecs * evals.unsqueeze(-2)) @ evecs.transpose(-1, -2)
    torch.testing.assert_close(evals, ref_evals)
    torch.testing.assert_close(recon, m, rtol=1e-5, atol=1e-6)
    assert evecs.shape == m.shape


def test_eigh_slabs_reconstructs_multidim_batches():
    m = _rand_spd((4, 5), seed=2)
    evals, evecs = _eigh_slabs(m.reshape(-1, 3, 3), chunk=4)

    assert evals.shape == (20, 3)
    assert evecs.shape == (20, 3, 3)
    recon = (evecs * evals.unsqueeze(-2)) @ evecs.transpose(-1, -2)
    torch.testing.assert_close(recon, m.reshape(-1, 3, 3), rtol=1e-5, atol=1e-6)


def test_linalg_eigh_matches_torch_on_cpu():
    m = _rand_spd((2, 11), seed=3)
    evals, evecs = _linalg_eigh(m)
    ref_evals, _ = torch.linalg.eigh(m)

    torch.testing.assert_close(evals, ref_evals)
    recon = (evecs * evals.unsqueeze(-2)) @ evecs.transpose(-1, -2)
    torch.testing.assert_close(recon, m, rtol=1e-5, atol=1e-6)


def test_sym_sqrtm_backward_still_finite_after_patch():
    m = _rand_spd((5,), seed=4).requires_grad_(True)
    sym_sqrtm(m).sum().backward()

    assert torch.isfinite(m.grad).all()
    assert float(m.grad.norm()) > 0
