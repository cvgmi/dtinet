"""
Log-Cholesky metric (LCM) coordinates for SPD(3).

The log-Cholesky map sends an SPD matrix P = L Lᵀ (L lower triangular with positive diagonal)
to the Euclidean coordinates (log L_00, log L_11, log L_22, L_10, L_21, L_20). Under the LCM,
geodesics, Fréchet means, and distances in these coordinates are ordinary Euclidean operations.
"""

import torch


def log_cholesky_coords(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Log-Cholesky coordinates of an SPD(3) Voigt-6 field.

    Parameters
    ----------
    x : torch.Tensor
        SPD(3) field of shape (..., 6) with entries ordered (xx, xy, yy, xz, yz, zz).
    eps : float, optional
        Value used to clamp the Cholesky diagonal for numerical safety.

    Returns
    -------
    torch.Tensor
        Log-Cholesky coordinates of shape (..., 6), ordered
        (log l00, log l11, log l22, l10, l21, l20).

    """
    xx, xy, yy, xz, yz, zz = x.unbind(dim=-1)

    # entries of the Cholesky factor are known in closed form. there is no need to call
    # torch.linalg.cholesky
    l00 = torch.sqrt(xx.clamp_min(eps))
    l10 = xy / l00
    l20 = xz / l00
    l11 = torch.sqrt((yy - l10.square()).clamp_min(eps))
    l21 = (yz - l20 * l10) / l11
    l22 = torch.sqrt((zz - l20.square() - l21.square()).clamp_min(eps))

    return torch.stack(
        (
            torch.log(l00),
            torch.log(l11),
            torch.log(l22),
            l10,
            l21,
            l20,
        ),
        dim=-1,
    )


def exp_cholesky_coords(z: torch.Tensor) -> torch.Tensor:
    """
    Inverse of :func:`log_cholesky_coords`.

    Parameters
    ----------
    z : torch.Tensor
        Log-Cholesky coordinates of shape (..., 6), ordered
        (log l00, log l11, log l22, l10, l21, l20).

    Returns
    -------
    torch.Tensor
        SPD(3) field of shape (..., 6) with entries ordered (xx, xy, yy, xz, yz, zz).

    """
    log_l00, log_l11, log_l22, l10, l21, l20 = z.unbind(dim=-1)
    l00 = torch.exp(log_l00)
    l11 = torch.exp(log_l11)
    l22 = torch.exp(log_l22)

    return torch.stack(
        (
            l00.square(),
            l10 * l00,
            l10.square() + l11.square(),
            l20 * l00,
            l20 * l10 + l21 * l11,
            l20.square() + l21.square() + l22.square(),
        ),
        dim=-1,
    )
