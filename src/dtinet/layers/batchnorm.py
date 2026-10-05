"""
Fréchet batch normalization layers for SPD(3)-valued 3D lattice data.

Analogous to Euclidean batch normalization, each channel is centered at its batch Fréchet mean
and scaled by its geodesic standard deviation, then translated to a learned Fréchet mean. For
coordinate metrics (LCM, LEM) these are Euclidean operations in coordinates; for the
affine-invariant metric ("aim") they use log/exp maps (see :mod:`dtinet.layers.aim`).

``FrechetBatchNorm3d`` dispatches on the ``metric`` argument.
"""

import torch
import torch.nn as nn

from dtinet.geometry import check_metric, is_coord_metric


class BaseFrechetBatchNorm3d(nn.Module):
    """
    Base class for Fréchet batch normalization layers.
    """

    metric: str


class FrechetBatchNorm3dCoords(BaseFrechetBatchNorm3d):
    """
    Fréchet batch normalization for coordinate-metric feature maps.

    Under a coordinate metric (LCM or LEM), the Fréchet mean and variance are the Euclidean mean
    and squared distance in coordinates. This layer normalizes each channel by its batch Fréchet
    mean and geodesic variance, then translates it to a learned Fréchet mean.

    Parameters
    ----------
    num_channels : int
        Number of input channels.
    metric : {"lcm", "lem"}, optional
        Coordinate metric.
    num_coordinates : int, optional
        Number of coordinates per SPD matrix.
    eps : float, optional
        Value added to the variance for numerical stability.
    momentum : float, optional
        Momentum used to update running Fréchet statistics.

    """

    def __init__(
        self,
        num_channels: int,
        metric: str = "lcm",
        num_coordinates: int = 6,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        super().__init__()

        if not is_coord_metric(metric):
            raise ValueError(f"{type(self).__name__} requires a coordinate metric, got {metric!r}.")

        self.num_channels = num_channels
        self.metric = metric
        self.num_coordinates = num_coordinates
        self.eps = eps
        self.momentum = momentum

        # learned target Fréchet mean and positive geodesic scale for each channel
        self.bias = nn.Parameter(torch.zeros(num_channels, num_coordinates))
        self.log_scale = nn.Parameter(torch.zeros(num_channels))
        self.register_buffer("running_mean", torch.zeros(num_channels, num_coordinates))
        self.register_buffer("running_var", torch.ones(num_channels))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input coordinate field of shape (B, C, F, D, H, W).
        mask : torch.Tensor or None, optional
            Binary spatial mask of shape (B, 1, D, H, W).

        Returns
        -------
        torch.Tensor
            Output feature map of shape (B, C, F, D, H, W).

        """
        B, C, F, D, H, W = x.shape

        if self.training:
            # Euclidean statistics in coordinates are Fréchet statistics under a coordinate metric
            if mask is None:
                N = B * D * H * W
                batch_mean = x.mean(dim=(0, 3, 4, 5))
                centered = x - batch_mean.view(1, C, F, 1, 1, 1)
                # sum over coordinates gives one scalar Fréchet variance per channel
                batch_var = centered.square().sum(dim=2)
                batch_var = batch_var.mean(dim=(0, 2, 3, 4))
            else:
                mask = mask.to(dtype=x.dtype)
                N = int(mask.sum().item())
                if N > 0:
                    # exclude invalid voxels from both the mean and variance
                    batch_mean = (x * mask.unsqueeze(2)).sum(dim=(0, 3, 4, 5)) / N
                    centered = x - batch_mean.view(1, C, F, 1, 1, 1)
                    batch_var = centered.square().sum(dim=2)
                    batch_var = (batch_var * mask).sum(dim=(0, 2, 3, 4)) / N
                else:
                    # an empty mask has no batch statistics, so retain the running estimates
                    batch_mean = self.running_mean
                    batch_var = self.running_var

            if mask is None or N > 0:
                # keep biased batch variance for this pass; debias only the running estimate
                running_var_update = batch_var.detach()
                if N > 1:
                    running_var_update = running_var_update * N / (N - 1)
                self.running_mean.mul_(1 - self.momentum).add_(batch_mean.detach() * self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(running_var_update * self.momentum)

            mean = batch_mean
            var = batch_var

        else:
            mean = self.running_mean
            var = self.running_var

        # normalize geodesic dispersion, then translate to the learned Fréchet mean
        out = x - mean.view(1, C, F, 1, 1, 1)
        out = out * (self.log_scale.exp() / torch.sqrt(var + self.eps)).view(1, C, 1, 1, 1, 1)
        out = out + self.bias.view(1, C, F, 1, 1, 1)

        if mask is not None:
            # zeros are a computational placeholder; downstream layers must propagate the mask
            out = out * mask.unsqueeze(2)

        return out


class FrechetBatchNorm3d:
    """
    Fréchet batch normalization, dispatching on the metric.

    Returns a :class:`FrechetBatchNorm3dCoords` instance for coordinate metrics ("lcm", "lem")
    and a ``FrechetBatchNorm3dAIM`` instance for the affine-invariant metric ("aim").
    """

    def __new__(cls, num_channels: int, metric: str = "lcm", **kwargs) -> BaseFrechetBatchNorm3d:
        check_metric(metric)
        if is_coord_metric(metric):
            return FrechetBatchNorm3dCoords(num_channels, metric=metric, **kwargs)
        from dtinet.layers.aim import FrechetBatchNorm3dAIM

        return FrechetBatchNorm3dAIM(num_channels, **kwargs)


class FrechetBatchNorm3dLC(FrechetBatchNorm3dCoords):
    """
    Fréchet batch normalization for log-Cholesky feature maps.

    Legacy alias for :class:`FrechetBatchNorm3dCoords` with ``metric="lcm"``, keeping the
    original constructor signature ``(num_channels, num_coordinates=6, eps=1e-5,
    momentum=0.1)``.
    """

    def __init__(
        self,
        num_channels: int,
        num_coordinates: int = 6,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        super().__init__(
            num_channels,
            metric="lcm",
            num_coordinates=num_coordinates,
            eps=eps,
            momentum=momentum,
        )
