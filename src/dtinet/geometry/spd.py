"""
Batched linear algebra helpers for SPD(3) matrices in Voigt-6 representation.

The Voigt-6 representation stores the unique entries of a symmetric 3x3 matrix in the order
(xx, xy, yy, xz, yz, zz). All matrix-valued helpers operate on the trailing dimensions, so they
work for arbitrary batch shapes (..., 3, 3).
"""

import torch

# index order shared by the whole library: (xx, xy, yy, xz, yz, zz)
VOIGT6_DIAG = (0, 2, 5)
VOIGT6_OFFDIAG = (1, 3, 4)

#: Eigenvalue gap below which the eigh backward treats a pair as degenerate.
EIGH_GAP_EPS = 1e-12


class _SafeEigh(torch.autograd.Function):
    """
    Symmetric eigendecomposition with a degeneracy-safe backward pass.

    The standard spectral gradient couples eigenvector gradients through 1/(λ_i − λ_j) terms,
    which are singular at repeated eigenvalues (e.g. the identity matrix, or any isotropic
    tangent vector). This wrapper zeroes the coupling terms for eigenvalue pairs closer than
    ``EIGH_GAP_EPS``: for functions that are symmetric within the degenerate subspace (all
    spectral functions used here) those terms vanish in the limit anyway, and gradients stay
    finite everywhere else.
    """

    @staticmethod
    def forward(ctx, m):
        evals, evecs = torch.linalg.eigh(m)
        ctx.save_for_backward(evals, evecs)
        return evals, evecs

    @staticmethod
    def backward(ctx, grad_evals, grad_evecs):
        evals, evecs = ctx.saved_tensors
        # F_ij = 1/(λ_j − λ_i), zeroed on the diagonal and for near-degenerate pairs
        gap = evals.unsqueeze(-2) - evals.unsqueeze(-1)
        f = torch.where(gap.abs() < EIGH_GAP_EPS, torch.zeros_like(gap), gap.reciprocal())
        grad = grad_evals.diag_embed() + f * (evecs.transpose(-1, -2) @ grad_evecs)
        grad = evecs @ grad @ evecs.transpose(-1, -2)
        grad = (grad + grad.transpose(-1, -2)) / 2
        return grad


def voigt6_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a Voigt-6 field to symmetric matrices.

    Parameters
    ----------
    x : torch.Tensor
        Input of shape (..., 6) with entries ordered (xx, xy, yy, xz, yz, zz).

    Returns
    -------
    torch.Tensor
        Symmetric matrices of shape (..., 3, 3).

    """
    xx, xy, yy, xz, yz, zz = x.unbind(dim=-1)
    return torch.stack(
        (
            torch.stack((xx, xy, xz), dim=-1),
            torch.stack((xy, yy, yz), dim=-1),
            torch.stack((xz, yz, zz), dim=-1),
        ),
        dim=-2,
    )


def matrix_to_voigt6(m: torch.Tensor, offdiag_scale: float = 1.0) -> torch.Tensor:
    """
    Convert symmetric matrices to a Voigt-6 field.

    Parameters
    ----------
    m : torch.Tensor
        Symmetric matrices of shape (..., 3, 3).
    offdiag_scale : float, optional
        Scale applied to the off-diagonal entries. Use sqrt(2) for an isometric
        Frobenius-preserving encoding.

    Returns
    -------
    torch.Tensor
        Output of shape (..., 6) with entries ordered (xx, xy, yy, xz, yz, zz).

    """
    return torch.stack(
        (
            m[..., 0, 0],
            offdiag_scale * m[..., 0, 1],
            m[..., 1, 1],
            offdiag_scale * m[..., 0, 2],
            offdiag_scale * m[..., 1, 2],
            m[..., 2, 2],
        ),
        dim=-1,
    )


def _sym_eigh_func(m: torch.Tensor, fn, clamp_min: float | None) -> torch.Tensor:
    m = (m + m.transpose(-1, -2)) / 2
    evals, evecs = _SafeEigh.apply(m)
    if clamp_min is not None:
        evals = evals.clamp_min(clamp_min)
    evals = fn(evals)
    return (evecs * evals.unsqueeze(-2)) @ evecs.transpose(-1, -2)


def sym_powm(m: torch.Tensor, p: float | torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Symmetric matrix power of SPD matrices, M^p, via symmetric eigendecomposition.

    Eigenvalues are clamped to ``eps`` for numerical safety. If ``p`` is a tensor, it must be
    broadcastable against the batch shape of ``m`` (i.e., ``m.shape[:-2]``).
    """
    if isinstance(p, torch.Tensor):
        return _sym_eigh_func(m, lambda e: e.pow(p.unsqueeze(-1)), clamp_min=eps)
    return _sym_eigh_func(m, lambda e: e.pow(p), clamp_min=eps)


def sym_logm(m: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Matrix logarithm of SPD matrices via symmetric eigendecomposition.

    Eigenvalues are clamped to ``eps`` for numerical safety.
    """
    return _sym_eigh_func(m, torch.log, clamp_min=eps)


def sym_expm(m: torch.Tensor) -> torch.Tensor:
    """
    Matrix exponential of symmetric matrices via symmetric eigendecomposition.
    """
    return _sym_eigh_func(m, torch.exp, clamp_min=None)


def sym_sqrtm(m: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Symmetric positive definite square root of SPD matrices.
    """
    return _sym_eigh_func(m, torch.sqrt, clamp_min=eps)


def sym_inv_sqrtm(m: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Inverse symmetric positive definite square root of SPD matrices.
    """
    return _sym_eigh_func(m, lambda e: e.rsqrt(), clamp_min=eps)
