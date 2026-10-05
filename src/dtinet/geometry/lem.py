"""
Log-Euclidean metric (LEM) coordinates for SPD(3).

The log-Euclidean map sends an SPD matrix P to its matrix logarithm log(P), a symmetric matrix.
Encoding log(P) with the Frobenius-isometric Voigt scaling (off-diagonal entries scaled by
sqrt(2)) yields Euclidean coordinates in which geodesics, Fréchet means, and distances under
the LEM are ordinary Euclidean operations.
"""

import math

import torch

from dtinet.geometry.spd import matrix_to_voigt6, sym_expm, sym_logm, voigt6_to_matrix

SQRT2 = math.sqrt(2.0)


def log_euclidean_coords(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Log-Euclidean coordinates of an SPD(3) Voigt-6 field.

    Parameters
    ----------
    x : torch.Tensor
        SPD(3) field of shape (..., 6) with entries ordered (xx, xy, yy, xz, yz, zz).
    eps : float, optional
        Eigenvalue clamp used in the matrix logarithm.

    Returns
    -------
    torch.Tensor
        Log-Euclidean coordinates of shape (..., 6): the isometric Voigt-6 encoding of log(P)
        (off-diagonal entries scaled by sqrt(2)).

    """
    return matrix_to_voigt6(sym_logm(voigt6_to_matrix(x), eps=eps), offdiag_scale=SQRT2)


def exp_euclidean_coords(z: torch.Tensor) -> torch.Tensor:
    """
    Inverse of :func:`log_euclidean_coords`.

    Parameters
    ----------
    z : torch.Tensor
        Log-Euclidean coordinates of shape (..., 6).

    Returns
    -------
    torch.Tensor
        SPD(3) field of shape (..., 6) with entries ordered (xx, xy, yy, xz, yz, zz).

    """
    scale = z.new_tensor((1.0, 1.0 / SQRT2, 1.0, 1.0 / SQRT2, 1.0 / SQRT2, 1.0))
    return matrix_to_voigt6(sym_expm(voigt6_to_matrix(z * scale)))
