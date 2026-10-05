"""
Coordinate maps from SPD(3)-valued fields to Euclidean coordinate fields.

These layers implement the first step of ManifoldNet for coordinate metrics (LCM, LEM): map
each voxel's SPD matrix to its Euclidean coordinates, after which all subsequent primitives are
ordinary Euclidean operations.
"""

import torch
import torch.nn as nn

from dtinet.geometry import IDENTITY_VOIGT6, is_coord_metric
from dtinet.geometry.lcm import log_cholesky_coords
from dtinet.geometry.lem import log_euclidean_coords

_COORD_MAPS = {
    "lcm": log_cholesky_coords,
    "lem": log_euclidean_coords,
}


class SPDToCoords(nn.Module):
    """
    Coordinate map for SPD(3)-valued fields under a coordinate metric.

    Parameters
    ----------
    metric : {"lcm", "lem"}
        Coordinate metric. "lcm" uses log-Cholesky coordinates; "lem" uses (isometrically
        scaled) log-Euclidean coordinates.

    """

    def __init__(self, metric: str = "lcm"):
        super().__init__()
        if not is_coord_metric(metric):
            raise ValueError(
                f"Metric {metric!r} has no global coordinate chart. "
                f"Supported coordinate metrics: {', '.join(_COORD_MAPS)}."
            )
        self.metric = metric

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input Voigt-encoded SPD(3) field of shape (B, C, 6, D, H, W).
        mask : torch.Tensor or None, optional
            Binary spatial mask of shape (B, 1, D, H, W).

        Returns
        -------
        torch.Tensor
            Output coordinate field of shape (B, C, 6, D, H, W).

        """
        if mask is not None:
            # set all voxels outside the mask to be the identity matrix
            mask = mask.unsqueeze(2)
            identity = x.new_tensor(IDENTITY_VOIGT6)
            identity = identity.view(1, 1, 6, 1, 1, 1)
            x = torch.where(mask.bool(), x, identity)

        # move the Voigt axis to the end for the geometry functions
        out = _COORD_MAPS[self.metric](x.movedim(2, -1)).movedim(-1, 2)

        if mask is not None:
            out = out * mask

        return out


class SPD3ToLC(SPDToCoords):
    """
    Log-Cholesky coordinate map SPD(3) → 𝐑⁶. Legacy alias for ``SPDToCoords(metric="lcm")``.
    """

    def __init__(self):
        super().__init__(metric="lcm")
