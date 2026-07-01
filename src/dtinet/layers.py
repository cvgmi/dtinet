import torch
import torch.nn as nn


class SPD3ToLC(nn.Module):
    """
    Log-Cholesky coordinate map SPD(3) → 𝐑⁶.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input Voigt-encoded SPD(3) field of shape (B, C, 6, D, H, W).

        Returns
        -------
        torch.Tensor
            Output log-Cholesky factor field of shape (B, C, 6, D, H, W).

        """
        xx, xy, yy, xz, yz, zz = x.unbind(dim=2)

        # entries of the Cholesky factor are known in closed form. there is no need to call
        # torch.linalg.cholesky
        l00 = torch.sqrt(xx)
        l10 = xy / l00
        l20 = xz / l00
        l11 = torch.sqrt(yy - l10.square())
        l21 = (yz - l20 * l10) / l11
        l22 = torch.sqrt(zz - l20.square() - l21.square())

        return torch.stack(
            (
                torch.log(l00),
                torch.log(l11),
                torch.log(l22),
                l10,
                l21,
                l20,
            ),
            dim=2,
        )


class WeightedFrechetMean3dLC(nn.Module):
    """
    Weighted Fréchet mean layer for SPD-valued 3D lattice data under the log-Cholesky metric (LCM).

    Under the LCM, the weighted Fréchet mean (wFM) becomes an ordinary Euclidean weighted average
    in log-Cholesky coordinates. Therefore, for this particular case, we can forgo iterative
    estimation of the wFM in exchange for an even more efficient Conv3d operation. The only caveat
    is that the Conv3d operation must:

    1. Ensure weights satisfy a convexity constraint.
    2. Ensure weights are shared across log-Cholesky coordinates.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Size of the cubic kernel.
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
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map of shape (B, C_in, F, D_in, H_in, W_in).

        Returns
        -------
        torch.Tensor
            Output feature map of shape (B, C_out, F, D_out, H_out, W_out).

        """
        B, _, F, D, H, W = x.shape

        x_folded = x.permute(0, 2, 1, 3, 4, 5).reshape(B * F, self.in_channels, D, H, W)

        weight = self.constrain_weight()

        y = nn.functional.conv3d(
            x_folded,
            weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
        )

        D_out, H_out, W_out = y.shape[-3:]
        return (
            y.view(B, F, self.out_channels, D_out, H_out, W_out)
            .permute(0, 2, 1, 3, 4, 5)
            .contiguous()
        )


class InvariantReadoutLC(nn.Module):
    """
    Invariant readout for log-Cholesky feature maps.

    Computes the Euclidean distance in log-Cholesky coordinates from each feature vector to the
    channel-wise mean. This converts the log-Cholesky coordinate field to a scalar distance field.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input log-Cholesky feature map of shape (B, C, F, D, H, W).

        Returns
        -------
        torch.Tensor
            Output distance field of shape (B, C, D, H, W).

        """
        return (x - x.mean(dim=1, keepdim=True)).norm(dim=2)
