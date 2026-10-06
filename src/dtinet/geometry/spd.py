"""
Batched linear algebra helpers for SPD(3) matrices in Voigt-6 representation.

The Voigt-6 representation stores the unique entries of a symmetric 3x3 matrix in the order
(xx, xy, yy, xz, yz, zz). All matrix-valued helpers operate on the trailing dimensions, so they
work for arbitrary batch shapes (..., 3, 3).
"""

import warnings

import torch

# index order shared by the whole library: (xx, xy, yy, xz, yz, zz)
VOIGT6_DIAG = (0, 2, 5)
VOIGT6_OFFDIAG = (1, 3, 4)

#: Eigenvalue gap (relative to the spectral scale) below which the Loewner
#: divided difference in the spectral-function backward uses the limit f'(λ).
EIGH_GAP_EPS = 1e-12

#: Batch count at which cuSOLVER's batched symmetric eigensolver
#: (cusolverDnXsyevBatched) fails with CUSOLVER_STATUS_INTERNAL_ERROR
#: (verified on L4 / torch 2.12.1+cu130: 2**16 - 1 works, 2**16 fails).
_CUSOLVER_MAX_BATCH = 65535

#: Chunk size for CUDA eigh calls: stays well under the cuSOLVER batch limit and
#: bounds the transient solver workspace (~0.25 MB per 3x3 matrix on L4).
_EIGH_CUDA_CHUNK = 32768


def _eigh_slabs(flat: torch.Tensor, chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Eigendecomposition of a flattened batch (B, n, n) computed in slabs.

    Slabs that fail on the accelerator (e.g. cuSOLVER internal errors) are retried
    on the CPU and moved back, so a single bad batch cannot abort training.
    """
    evals, evecs = [], []
    for slab in flat.split(chunk):
        try:
            e, v = torch.linalg.eigh(slab)
        except RuntimeError:
            warnings.warn(
                "CUDA eigh failed on a slab; retrying on CPU "
                "(this will slow training down considerably)",
                RuntimeWarning,
                stacklevel=2,
            )
            e, v = torch.linalg.eigh(slab.cpu())
            e, v = e.to(flat.device), v.to(flat.device)
        evals.append(e)
        evecs.append(v)
    return torch.cat(evals), torch.cat(evecs)


def _linalg_eigh(m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    ``torch.linalg.eigh`` with a workaround for cuSOLVER's batched-size limit.

    CUDA batches larger than ``_EIGH_CUDA_CHUNK`` are split into slabs below the
    2**16 syevBatched limit; smaller batches are computed directly, falling back
    to slab-wise (and ultimately CPU) evaluation if the solver errors.
    """
    if not m.is_cuda:
        return torch.linalg.eigh(m)
    n = m.shape[-1]
    flat = m.reshape(-1, n, n)
    if flat.shape[0] <= _EIGH_CUDA_CHUNK:
        try:
            return torch.linalg.eigh(m)
        except RuntimeError:
            pass
    evals, evecs = _eigh_slabs(flat, _EIGH_CUDA_CHUNK)
    return evals.reshape(*m.shape[:-1]), evecs.reshape(*m.shape)


class _SpectralFunc(torch.autograd.Function):
    """
    Spectral function f(M) = U f(Λ) Uᵀ of a symmetric matrix, with the exact backward.

    The gradient of a spectral function involves the Loewner (divided-difference) matrix
    K_ij = (f(λ_i) − f(λ_j)) / (λ_i − λ_j), whose diagonal and degenerate limit is f'(λ_i):
    ``grad = U (K ⊙ Uᵀ G U) Uᵀ``. Note that the coupling between (near-)degenerate
    eigenpairs is f'(λ_i), NOT zero: zeroing it (the naive degeneracy guard) silently
    discards the eigenvector-direction derivative — e.g. it makes the gradient of log(M)
    vanish along off-diagonal directions at M = I, which is exactly where identity padding
    and isotropic tensors live. The divided-difference form is finite everywhere f is
    differentiable and needs no special-casing beyond the limit value.
    """

    @staticmethod
    def forward(ctx, m, fn, fn_prime, clamp_min, p_exp):
        evals, evecs = _linalg_eigh(m)
        if clamp_min is not None:
            clamped = evals < clamp_min
            evals = evals.clamp_min(clamp_min)
        else:
            clamped = torch.zeros_like(evals, dtype=torch.bool)
        f_evals = fn(evals)
        # clamp_min is (locally) constant in the clamped region, so f' vanishes there
        fp_evals = torch.where(clamped, torch.zeros_like(evals), fn_prime(evals))
        ctx.save_for_backward(evals, f_evals, fp_evals, evecs)
        ctx.p_exp = p_exp
        return (evecs * f_evals.unsqueeze(-2)) @ evecs.transpose(-1, -2)

    @staticmethod
    def backward(ctx, grad_out):
        evals, f_evals, fp_evals, evecs = ctx.saved_tensors
        gap = evals.unsqueeze(-2) - evals.unsqueeze(-1)  # (..., i, j) = λ_i − λ_j
        f_gap = f_evals.unsqueeze(-2) - f_evals.unsqueeze(-1)
        # scale-aware degeneracy threshold: eigenvalue gaps below this carry no
        # resolvable information at the working precision, so use the limit f'(λ_i)
        scale = evals.abs().amax(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(1.0)
        degenerate = gap.abs() < EIGH_GAP_EPS * scale
        gap_safe = torch.where(degenerate, torch.ones_like(gap), gap)
        loewner = torch.where(degenerate, fp_evals.unsqueeze(-1).expand_as(gap), f_gap / gap_safe)
        g_eig = evecs.transpose(-1, -2) @ grad_out @ evecs
        grad = evecs @ (loewner * g_eig) @ evecs.transpose(-1, -2)
        grad = (grad + grad.transpose(-1, -2)) / 2

        grad_p = None
        if ctx.p_exp is not None:
            # f(λ) = λ^p varies with p as ∂f/∂p = λ^p ln λ, diagonal in the eigenbasis
            grad_p = (g_eig.diagonal(dim1=-2, dim2=-1) * f_evals * torch.log(evals))
            grad_p = grad_p.sum(dim=-1, keepdim=True)
            # reduce broadcast dimensions back to the shape of the p argument
            while grad_p.dim() > ctx.p_exp.dim():
                grad_p = grad_p.sum(dim=0)
            for i, size in enumerate(ctx.p_exp.shape):
                if size == 1 and grad_p.shape[i] > 1:
                    grad_p = grad_p.sum(dim=i, keepdim=True)
        return grad, None, None, None, grad_p


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


def _sym_eigh_func(
    m: torch.Tensor, fn, fn_prime, clamp_min: float | None, p_exp: torch.Tensor | None = None
) -> torch.Tensor:
    m = (m + m.transpose(-1, -2)) / 2
    return _SpectralFunc.apply(m, fn, fn_prime, clamp_min, p_exp)


def sym_powm(m: torch.Tensor, p: float | torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Symmetric matrix power of SPD matrices, M^p, via symmetric eigendecomposition.

    Eigenvalues are clamped to ``eps`` for numerical safety. If ``p`` is a tensor, it must be
    broadcastable against the batch shape of ``m`` (i.e., ``m.shape[:-2]``).
    """
    if isinstance(p, torch.Tensor):
        e = p.unsqueeze(-1)
        return _sym_eigh_func(
            m, lambda x: x.pow(e), lambda x: e * x.pow(e - 1), clamp_min=eps, p_exp=e
        )
    return _sym_eigh_func(m, lambda x: x.pow(p), lambda x: p * x.pow(p - 1), clamp_min=eps)


def sym_logm(m: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Matrix logarithm of SPD matrices via symmetric eigendecomposition.

    Eigenvalues are clamped to ``eps`` for numerical safety.
    """
    return _sym_eigh_func(m, torch.log, torch.reciprocal, clamp_min=eps)


def sym_expm(m: torch.Tensor) -> torch.Tensor:
    """
    Matrix exponential of symmetric matrices via symmetric eigendecomposition.
    """
    return _sym_eigh_func(m, torch.exp, torch.exp, clamp_min=None)


def sym_sqrtm(m: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Symmetric positive definite square root of SPD matrices.
    """
    return _sym_eigh_func(m, torch.sqrt, lambda x: 0.5 * x.rsqrt(), clamp_min=eps)


def sym_inv_sqrtm(m: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Inverse symmetric positive definite square root of SPD matrices.
    """
    return _sym_eigh_func(
        m, torch.rsqrt, lambda x: -0.5 * x.pow(-1.5), clamp_min=eps
    )
