"""
Voxelwise bilinear (congruence) maps for coordinate-metric feature maps.

Weighted Fréchet means apply one kernel to all coordinates and coordinate-wise activations
act on each coordinate alone, so a coordinate-metric ManifoldNet cannot combine the
components of a tensor (e.g. into anisotropy or orientation) before its readout. The SPDNet
BiMap layer, ``P -> W P W^T`` with a learned full-rank ``W``, maps SPD matrices to SPD
matrices and mixes all components; applied per channel and voxel it acts like a 1x1x1
convolution on the manifold. Under LCM it is computed as log-Cholesky coordinates ->
SPD -> congruence -> log-Cholesky coordinates, all in closed form.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# floor on squared row norms; keeps the logarithms finite for (near-)singular W
_TINY = 1e-30



class BiMap3dLC(nn.Module):
    """
    Per-channel voxelwise congruence ``P -> W_c P W_c^T`` on a log-Cholesky coordinate field.

    Parameters
    ----------
    channels : int
        Number of channels; each has its own 3x3 matrix ``W_c``.
    init_noise : float, optional
        Standard deviation of the perturbation added to the identity at initialization.
    checkpoint : bool, optional
        Recompute the map in the backward pass instead of storing its (..., 3, 3)
        intermediates, which dominate memory on full-resolution fields.

    """

    def __init__(self, channels: int, init_noise: float = 0.1, checkpoint: bool = True):
        super().__init__()
        self.checkpoint = checkpoint
        # W = I + delta, so weight decay pulls the map toward the identity rather than toward
        # the singular zero matrix
        self.delta = nn.Parameter(init_noise * torch.randn(channels, 3, 3))

    @property
    def weight(self) -> torch.Tensor:
        """Congruence matrices ``W_c`` of shape (C, 3, 3)."""
        return torch.eye(3, device=self.delta.device, dtype=self.delta.dtype) + self.delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Log-Cholesky coordinate field of shape (B, C, 6, D, H, W).

        Returns
        -------
        torch.Tensor
            Log-Cholesky coordinates of the mapped field, shape (B, C, 6, D, H, W).

        """
        if self.checkpoint and torch.is_grad_enabled():
            return checkpoint(self._map, x, self.weight, use_reentrant=False).to(x.dtype)
        return self._map(x, self.weight).to(x.dtype)

    @staticmethod
    def _map(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # P = L L^T from log-Cholesky coordinates, so W P W^T = G G^T with G = W L. W is
        # shared across voxels, so G is one (3x3) @ (3xN) product per channel. The Cholesky
        # factor of G G^T is then read off G's rows by modified Gram-Schmidt (the LQ
        # decomposition of G), which avoids forming G G^T and the cancellation-prone
        # subtractive Cholesky formulas. Always float32: in float16 the norm floors vanish.
        dtype = torch.promote_types(x.dtype, torch.float32)
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.to(dtype)
            weight = weight.to(dtype)
            log_l00, log_l11, log_l22, l10, l21, l20 = x.unbind(dim=2)
            zero = torch.zeros_like(l10)
            rows = (
                torch.stack((log_l00.exp(), zero, zero), dim=2),
                torch.stack((l10, log_l11.exp(), zero), dim=2),
                torch.stack((l20, l21, log_l22.exp()), dim=2),
            )
            lower = torch.stack(rows, dim=2)  # (B, C, 3 rows, 3 cols, D, H, W)
            g0, g1, g2 = torch.einsum("cik,bckj...->bcij...", weight, lower).unbind(dim=2)

            def dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return (a * b).sum(dim=2)

            def norm(a: torch.Tensor) -> torch.Tensor:
                return torch.sqrt(dot(a, a).clamp_min(_TINY))

            n00 = norm(g0)
            q0 = g0 / n00.unsqueeze(2)
            n10 = dot(g1, q0)
            r1 = g1 - n10.unsqueeze(2) * q0
            n11 = norm(r1)
            q1 = r1 / n11.unsqueeze(2)
            n20 = dot(g2, q0)
            r2 = g2 - n20.unsqueeze(2) * q0
            n21 = dot(r2, q1)
            r2 = r2 - n21.unsqueeze(2) * q1
            n22 = norm(r2)
            return torch.stack(
                (n00.log(), n11.log(), n22.log(), n10, n21, n20), dim=2
            )
