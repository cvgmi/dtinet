"""
Invariant readout layers for SPD(3)-valued 3D lattice data.

The readout converts an SPD-valued feature map to a scalar distance field by computing, per
sample, the geodesic distance of each fiber to the masked Fréchet mean over all fibers
(channels and spatial positions). For coordinate metrics (LCM, LEM) this is a squared
Euclidean distance in coordinates; for the affine-invariant metric ("aim") it uses the
squared affine-invariant distance (see :mod:`dtinet.layers.aim`).

``InvariantReadout`` dispatches on the ``metric`` argument.
"""

import torch
import torch.nn as nn

from dtinet.geometry import check_metric, is_coord_metric


class BaseInvariantReadout(nn.Module):
    """
    Base class for invariant readout layers.
    """

    metric: str


class InvariantReadoutCoords(BaseInvariantReadout):
    """
    Invariant readout for coordinate-metric feature maps.

    Computes, per sample, the masked Euclidean mean over all fibers (channels and spatial
    positions) — the Fréchet mean under a coordinate metric — and outputs the squared
    Euclidean distance in coordinates from each fiber to it. This converts the coordinate
    field to a scalar squared-distance field.

    Parameters
    ----------
    metric : {"lcm", "lem"}, optional
        Coordinate metric.

    """

    def __init__(self, metric: str = "lcm"):
        super().__init__()
        if not is_coord_metric(metric):
            raise ValueError(f"{type(self).__name__} requires a coordinate metric, got {metric!r}.")
        self.metric = metric

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input coordinate feature map of shape (B, C, F, D, H, W).
        mask : torch.Tensor or None, optional
            Binary spatial mask of shape (B, 1, D, H, W).

        Returns
        -------
        torch.Tensor
            Output squared-distance field of shape (B, C, D, H, W), zero outside the mask.

        """
        B, C, F, D, H, W = x.shape

        if mask is None:
            # one Euclidean mean per sample over all channels and spatial positions
            mean = x.mean(dim=(1, 3, 4, 5))
        else:
            weights = mask.to(x.dtype)
            denom = C * weights.sum(dim=(2, 3, 4))
            mean = (x * weights.unsqueeze(2)).sum(dim=(1, 3, 4, 5))
            mean = mean / denom.clamp_min(torch.finfo(x.dtype).eps).view(B, 1)

        out = (x - mean.view(B, 1, F, 1, 1, 1)).square().sum(dim=2)

        if mask is not None:
            # also handles the empty-mask case: all distances are zeroed out
            out = out * mask.to(x.dtype)

        return out


class InvariantReadoutLC(InvariantReadoutCoords):
    """
    Invariant readout for log-Cholesky feature maps.

    Legacy alias for :class:`InvariantReadoutCoords` with ``metric="lcm"``, keeping the
    original no-argument constructor.
    """

    def __init__(self):
        super().__init__(metric="lcm")


class InvariantReadout:
    """
    Invariant readout, dispatching on the metric.

    Returns an :class:`InvariantReadoutCoords` instance for coordinate metrics ("lcm", "lem")
    and an ``InvariantReadoutAIM`` instance for the affine-invariant metric ("aim").
    """

    def __new__(cls, metric: str = "lcm", **kwargs) -> BaseInvariantReadout:
        check_metric(metric)
        if is_coord_metric(metric):
            return InvariantReadoutCoords(metric=metric, **kwargs)
        from dtinet.layers.aim import InvariantReadoutAIM

        return InvariantReadoutAIM(**kwargs)

