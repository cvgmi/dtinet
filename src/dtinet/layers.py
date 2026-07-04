import torch
import torch.nn as nn


class SPD3ToLC(nn.Module):
    """
    Log-Cholesky coordinate map SPD(3) → 𝐑⁶.
    """

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
            Output log-Cholesky factor field of shape (B, C, 6, D, H, W).

        """
        if mask is not None:
            # set all voxels outside the mask to be the identity matrix
            mask = mask.unsqueeze(2)
            identity = x.new_tensor((1.0, 0.0, 1.0, 0.0, 0.0, 1.0))
            identity = identity.view(1, 1, 6, 1, 1, 1)
            x = torch.where(mask.bool(), x, identity)

        xx, xy, yy, xz, yz, zz = x.unbind(dim=2)

        # entries of the Cholesky factor are known in closed form. there is no need to call
        # torch.linalg.cholesky
        l00 = torch.sqrt(xx)
        l10 = xy / l00
        l20 = xz / l00
        l11 = torch.sqrt(yy - l10.square())
        l21 = (yz - l20 * l10) / l11
        l22 = torch.sqrt(zz - l20.square() - l21.square())

        out = torch.stack(
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

        if mask is not None:
            out = out * mask

        return out


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

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map of shape (B, C_in, F, D_in, H_in, W_in).
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

        # fold the log-Cholesky coordinate axis into the batch axis so the same Conv3d kernel is
        # applied independently to every log-Cholesky coordinate
        x = x.permute(0, 2, 1, 3, 4, 5).reshape(B * F, C, D, H, W)

        weight = self.constrain_weight()

        if mask is not None:
            # replicate the mask across log-Cholesky coordinates and fold it in the same way as x so
            # each coordinate sees the same valid support
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


class InvariantReadoutLC(nn.Module):
    """
    Invariant readout for log-Cholesky feature maps.

    Computes the Euclidean distance in log-Cholesky coordinates from each feature vector to the
    channel-wise mean. This converts the log-Cholesky coordinate field to a scalar distance field.
    """

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input log-Cholesky feature map of shape (B, C, F, D, H, W).
        mask : torch.Tensor or None, optional
            Binary spatial mask of shape (B, 1, D, H, W).

        Returns
        -------
        torch.Tensor
            Output distance field of shape (B, C, D, H, W).

        """
        out = (x - x.mean(dim=1, keepdim=True)).norm(dim=2)

        if mask is not None:
            out = out * mask

        return out
