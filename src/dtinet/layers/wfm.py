"""
Weighted Fréchet mean (wFM) layers for SPD(3)-valued 3D lattice data.

The wFM layer is the ManifoldNet analogue of a convolution: each output voxel is the weighted
Fréchet mean of the SPD matrices in the corresponding input window. For coordinate metrics
(LCM, LEM) the wFM is an ordinary Euclidean weighted average in coordinates, implemented as an
efficient Conv3d. For the affine-invariant metric ("aim"), the wFM requires Riemannian
machinery and is implemented in :mod:`dtinet.layers.aim`.

``WeightedFrechetMean3d`` dispatches on the ``metric`` argument and returns an instance of the
appropriate implementation.
"""

import torch
import torch.nn as nn

from dtinet.geometry import check_metric, is_coord_metric


class BaseWeightedFrechetMean3d(nn.Module):
    """
    Base class for weighted Fréchet mean layers.

    Implementations consume and produce fields of shape (B, C, 6, D, H, W): coordinate fields
    for coordinate metrics, Voigt-encoded SPD fields for "aim".
    """

    metric: str


class WeightedFrechetMean3dCoords(BaseWeightedFrechetMean3d):
    """
    Weighted Fréchet mean layer for SPD-valued 3D lattice data under a coordinate metric.

    Under a coordinate metric (LCM or LEM), the weighted Fréchet mean (wFM) becomes an ordinary
    Euclidean weighted average in coordinates. Therefore, for this particular case, we can
    forgo iterative estimation of the wFM in exchange for an even more efficient Conv3d
    operation. The only caveat is that the Conv3d operation must:

    1. Ensure weights satisfy a convexity constraint.
    2. Ensure weights are shared across coordinates.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Size of the cubic kernel.
    metric : {"lcm", "lem"}, optional
        Coordinate metric.
    stride : int, optional
        Stride of cubic kernel.
    padding : int, optional
        Padding applied to the input.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        metric: str = "lcm",
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()

        if not is_coord_metric(metric):
            raise ValueError(f"{type(self).__name__} requires a coordinate metric, got {metric!r}.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.metric = metric
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(
            torch.zeros(
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
                self.kernel_size,
            )
        )

    def constrain_weight(self) -> torch.Tensor:
        """
        Constrain weights to be non-negative and sum to one.
        """
        w = self.weight.view(self.out_channels, -1)
        w = nn.functional.softmax(w, dim=-1)
        return w.view_as(self.weight)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input coordinate field of shape (B, C_in, F, D_in, H_in, W_in).
        mask : torch.Tensor or None, optional
            Binary spatial mask of shape (B, 1, D_in, H_in, W_in).

        Returns
        -------
        torch.Tensor
            Output feature map of shape (B, C_out, F, D_out, H_out, W_out).
        tuple[torch.Tensor, torch.Tensor]
            Output feature map and propagated binary mask of shape (B, 1, D_out, H_out, W_out).

        """
        B, C, F, D, H, W = x.shape

        provided_mask = mask is not None

        if mask is None and self.padding > 0:
            mask = torch.ones(B, 1, D, H, W, device=x.device, dtype=x.dtype)

        # fold the coordinate axis into the batch axis so the same Conv3d kernel is applied
        # independently to every coordinate
        x = x.permute(0, 2, 1, 3, 4, 5).reshape(B * F, C, D, H, W)

        weight = self.constrain_weight()

        if mask is not None:
            # replicate the mask across coordinates and fold it in the same way as x so each
            # coordinate sees the same valid support
            mask = (
                mask.unsqueeze(1)
                .expand(B, F, 1, D, H, W)
                .reshape(B * F, 1, D, H, W)
                .to(dtype=x.dtype)
            )
            # remove invalid samples from the weighted numerator
            x = x * mask

        y = nn.functional.conv3d(
            x,
            weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
        )
        D_out, H_out, W_out = y.shape[-3:]

        if mask is not None:
            # renormalize surviving convex weights over valid samples only
            denom = nn.functional.conv3d(
                mask,
                weight.sum(dim=1, keepdim=True),
                bias=None,
                stride=self.stride,
                padding=self.padding,
            )
            valid = denom > 0
            y = y / denom.clamp_min(torch.finfo(y.dtype).eps)
            y = torch.where(valid, y, torch.zeros_like(y))
            # construct output mask to be propagated
            denom = denom.reshape(B, F, self.out_channels, D_out, H_out, W_out)
            out_mask = denom[:, 0, :1] > 0

        y = y.reshape(B, F, self.out_channels, D_out, H_out, W_out).permute(0, 2, 1, 3, 4, 5)

        if not provided_mask:
            return y

        return y, out_mask


class WeightedFrechetMean3d:
    """
    Weighted Fréchet mean layer, dispatching on the metric.

    Returns a :class:`WeightedFrechetMean3dCoords` instance for coordinate metrics ("lcm",
    "lem") and a ``WeightedFrechetMean3dAIM`` instance for the affine-invariant metric
    ("aim"). Accepts the same constructor arguments as the metric-specific implementations.
    """

    def __new__(
        cls,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        metric: str = "lcm",
        **kwargs,
    ) -> BaseWeightedFrechetMean3d:
        check_metric(metric)
        if is_coord_metric(metric):
            return WeightedFrechetMean3dCoords(
                in_channels, out_channels, kernel_size, metric=metric, **kwargs
            )
        from dtinet.layers.aim import WeightedFrechetMean3dAIM

        return WeightedFrechetMean3dAIM(in_channels, out_channels, kernel_size, **kwargs)


class WeightedFrechetMean3dLC(WeightedFrechetMean3dCoords):
    """
    Weighted Fréchet mean layer for log-Cholesky feature maps.

    Legacy alias for :class:`WeightedFrechetMean3dCoords` with ``metric="lcm"``, keeping the
    original constructor signature ``(in_channels, out_channels, kernel_size, stride=1,
    padding=0)``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, metric="lcm", stride=stride, padding=padding
        )
