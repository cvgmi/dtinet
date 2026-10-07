"""
Residual blocks for coordinate-metric (LCM, LEM) feature maps.

A ResNet block adds a learned correction to its input, ``x + F(x)``. The manifold analogue
used here replaces the sum with a weighted Fréchet mean of the input and the block output,
``(1 - a) x + a F(x)`` with a per-channel weight ``a`` in (0, 1). Under a coordinate metric this
is the geodesic interpolation between the two fields, so every operation remains a Fréchet
mean, and the identity path keeps gradients well conditioned in deep stacks.
"""

import torch
import torch.nn as nn

from dtinet.geometry import is_coord_metric
from dtinet.layers.batchnorm import FrechetBatchNorm3dCoords
from dtinet.layers.bimap import BiMap3dLC
from dtinet.layers.wfm import WeightedFrechetMean3dCoords


def masked(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero a (B, C, F, D, H, W) field outside a (B, 1, D, H, W) mask."""
    return x * mask.unsqueeze(2).to(x.dtype)


class ManifoldResidualBlock3d(nn.Module):
    """
    Two stride-1 weighted Fréchet mean layers with a geodesic-interpolation skip connection.

    Computes ``y = M(BN(wFM(act(M(BN(wFM(x)))))))``, with ``M`` an optional BiMap, and returns
    ``act((1 - a) x + a y)`` restricted to the input mask, where ``a = sigmoid(mix_logit)`` is
    learned per channel.

    Parameters
    ----------
    channels : int
        Number of input and output channels.
    metric : {"lcm", "lem"}, optional
        Coordinate metric.
    kernel_size : int, optional
        Size of the cubic kernels (odd, so stride-1 layers preserve the grid).
    activation : torch.nn.Module or None, optional
        Activation module; None disables activations.
    init_mix : float, optional
        Initial interpolation weight ``a`` given to the block output.
    init_std : float, optional
        Initialization scale of the pre-softmax wFM weights.
    bimap : bool, optional
        Apply a per-channel voxelwise congruence (:class:`BiMap3dLC`) after each batch
        normalization, so the block can combine tensor components. LCM only.

    """

    def __init__(
        self,
        channels: int,
        metric: str = "lcm",
        kernel_size: int = 3,
        activation: nn.Module | None = None,
        init_mix: float = 0.5,
        init_std: float = 1.0,
        bimap: bool = False,
    ):
        super().__init__()
        if not is_coord_metric(metric):
            raise ValueError(f"{type(self).__name__} requires a coordinate metric, got {metric!r}.")
        if kernel_size % 2 != 1:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        if bimap and metric != "lcm":
            raise ValueError(f"bimap is implemented for metric='lcm' only, got {metric!r}")
        if not 0 < init_mix < 1:
            raise ValueError(f"init_mix must be in (0, 1), got {init_mix}")
        padding = kernel_size // 2
        self.conv1 = WeightedFrechetMean3dCoords(
            channels, channels, kernel_size, metric=metric, padding=padding, init_std=init_std
        )
        self.bn1 = FrechetBatchNorm3dCoords(channels, metric=metric)
        self.conv2 = WeightedFrechetMean3dCoords(
            channels, channels, kernel_size, metric=metric, padding=padding, init_std=init_std
        )
        self.bn2 = FrechetBatchNorm3dCoords(channels, metric=metric)
        self.bimap1 = BiMap3dLC(channels) if bimap else nn.Identity()
        self.bimap2 = BiMap3dLC(channels) if bimap else nn.Identity()
        self.activation = activation if activation is not None else nn.Identity()
        logit = torch.logit(torch.tensor(float(init_mix)))
        self.mix_logit = nn.Parameter(torch.full((channels,), float(logit)))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input coordinate field of shape (B, C, F, D, H, W), zero outside the mask.
        mask : torch.Tensor
            Binary spatial mask of shape (B, 1, D, H, W). Stride-1 layers would dilate it, so
            the block keeps the input mask fixed.

        Returns
        -------
        torch.Tensor
            Output coordinate field of shape (B, C, F, D, H, W), zero outside the mask.

        """
        y, _ = self.conv1(x, mask)
        y = masked(self.activation(self.bimap1(self.bn1(masked(y, mask), mask))), mask)
        y, _ = self.conv2(y, mask)
        y = masked(self.bimap2(self.bn2(masked(y, mask), mask)), mask)
        a = torch.sigmoid(self.mix_logit).view(1, -1, 1, 1, 1, 1)
        return masked(self.activation((1 - a) * x + a * y), mask)
