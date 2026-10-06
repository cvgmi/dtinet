"""
Affine-invariant metric (AIM) layers for SPD(3)-valued 3D lattice data.

Under the affine-invariant metric there is no global coordinate chart, so the ManifoldNet
primitives operate directly on the Voigt-encoded SPD matrices:

- :class:`WeightedFrechetMean3dAIM` computes per-window recursive (inductive) Fréchet means,
  following the reference ManifoldNet implementation.
- :class:`FrechetBatchNorm3dAIM` centers each channel at its batch Fréchet mean, scales the
  tangent vectors by the geodesic standard deviation, parallel-transports them to a learned
  Fréchet mean, and maps back onto the manifold.
- :class:`InvariantReadoutAIM` converts the SPD field to a scalar distance field using the
  channel-wise Fréchet mean per voxel.
"""

import torch
import torch.nn as nn

from dtinet.geometry import IDENTITY_VOIGT6
from dtinet.geometry.aim import (
    aim_distance_sq,
    aim_exp,
    aim_gl_mean,
    aim_log,
    aim_parallel_transport,
    karcher_wfm,
    recursive_wfm,
)
from dtinet.geometry.spd import matrix_to_voigt6, voigt6_to_matrix
from dtinet.layers.batchnorm import BaseFrechetBatchNorm3d
from dtinet.layers.readout import BaseInvariantReadout
from dtinet.layers.wfm import BaseWeightedFrechetMean3d


class WeightedFrechetMean3dAIM(BaseWeightedFrechetMean3d):
    """
    Weighted Fréchet mean layer for SPD-valued 3D lattice data under the AIM.

    Each output voxel is the recursive (inductive) weighted Fréchet mean of the SPD matrices in
    the corresponding input window: the running mean starts at the first valid fiber and each
    subsequent valid fiber is folded in with an exact two-point geodesic mean.

    Weight semantics (following the reference ManifoldNet implementation): the weights are
    learnable *geodesic gates* — fiber ``k`` is folded in via ``running = fiber_k #_{w_k}
    running``, so ``w_k`` is the fraction retained of the running mean. This is NOT the
    mass-weighted iFME recursion of the paper (Eq. 4, fraction ``w_k / sum_{i<=k} w_i``
    weighting the new fiber); independent gates implicitly parameterize simplex masses, so
    the parameterizations are equivalent up to reparameterization. Weights are constrained
    to the open interval (0, 1) by clamping; unlike the coordinate-metric layer, they are
    not required to sum to one. Masked-out fibers are skipped WITHOUT renormalizing the
    surviving gates, so effective masses are mask-dependent (gates are learned jointly
    with the mask distribution of the data).

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
        Padding applied to the input. Padded voxels are filled with identity matrices and
        excluded from the Fréchet means.

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
        self.metric = "aim"
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(
            torch.rand(
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
                self.kernel_size,
            )
        )

    def constrain_weight(self) -> torch.Tensor:
        """
        Constrain weights to the open interval (0, 1) by clamping.
        """
        return self.weight.clamp(1e-3, 1 - 1e-3)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input Voigt-encoded SPD field of shape (B, C_in, 6, D_in, H_in, W_in).
        mask : torch.Tensor or None, optional
            Binary spatial mask of shape (B, 1, D_in, H_in, W_in).

        Returns
        -------
        torch.Tensor
            Output SPD field of shape (B, C_out, 6, D_out, H_out, W_out).
        tuple[torch.Tensor, torch.Tensor]
            Output SPD field and propagated binary mask of shape (B, 1, D_out, H_out, W_out).

        """
        B, C, F, D, H, W = x.shape
        k = self.kernel_size
        s = self.stride
        eye_voigt = torch.tensor(IDENTITY_VOIGT6, dtype=x.dtype, device=x.device)
        eye_voigt = eye_voigt.view(1, 1, 6, 1, 1, 1)

        provided_mask = mask is not None

        if mask is None and self.padding > 0:
            mask = torch.ones(B, 1, D, H, W, device=x.device, dtype=x.dtype)

        if self.padding > 0:
            pad = [self.padding] * 6
            x = nn.functional.pad(x, pad)
            # identity matrices at padded voxels keep every window on the manifold
            indicator = nn.functional.pad(x.new_ones(1, 1, 1, D, H, W), pad)
            x = x + (1 - indicator) * eye_voigt
            if mask is not None:
                mask = nn.functional.pad(mask, pad)

        # gather the C * k^3 fibers of each window into a set dimension, ordered
        # (channel, kd, kh, kw) to match the layout of the weight parameter
        u = x.unfold(3, k, s).unfold(4, k, s).unfold(5, k, s)
        D_out, H_out, W_out = u.shape[3:6]
        K = C * k**3
        u = u.permute(0, 3, 4, 5, 1, 6, 7, 8, 2).reshape(B, D_out, H_out, W_out, K, 6)
        fibers = voigt6_to_matrix(u)

        w = self.constrain_weight().reshape(self.out_channels, K)
        w = w.view(1, 1, 1, 1, self.out_channels, K).expand(B, D_out, H_out, W_out, -1, -1)

        if mask is not None:
            m = mask.to(torch.bool)
            m = m.unfold(2, k, s).unfold(3, k, s).unfold(4, k, s)
            m = m.permute(0, 2, 3, 4, 1, 5, 6, 7).expand(B, D_out, H_out, W_out, C, k, k, k)
            valid = m.reshape(B, D_out, H_out, W_out, K)

            # replace invalid fibers by the identity so all windows stay on the manifold
            eye3 = torch.eye(3, dtype=x.dtype, device=x.device)
            fibers = torch.where(valid.unsqueeze(-1).unsqueeze(-1), fibers, eye3)
            any_valid = valid.any(dim=-1)

            # invalid fibers get weight exactly 1 (running mean unchanged); the first valid
            # fiber gets weight 0 (running mean reset to it)
            valid_e = valid.unsqueeze(4).expand_as(w)
            w_cols = []
            initialized = torch.zeros_like(w[..., 0], dtype=torch.bool)
            for w_k, v_k in zip(w.unbind(-1), valid_e.unbind(-1)):
                w_cols.append(torch.where(v_k & ~initialized, w_k.new_zeros(()), w_k))
                initialized = initialized | v_k
            w = torch.where(valid_e, torch.stack(w_cols, dim=-1), w.new_ones(()))

        O = self.out_channels
        fibers = fibers.unsqueeze(4).expand(B, D_out, H_out, W_out, O, K, 3, 3)
        fm = recursive_wfm(fibers, w, dim=-3)
        y = matrix_to_voigt6(fm).permute(0, 4, 5, 1, 2, 3)

        if mask is None:
            return y

        out_mask = any_valid.unsqueeze(1)
        y = torch.where(out_mask.unsqueeze(2), y, eye_voigt)

        if not provided_mask:
            return y

        return y, out_mask


class FrechetBatchNorm3dAIM(BaseFrechetBatchNorm3d):
    """
    Fréchet batch normalization for SPD-valued 3D lattice data under the AIM.

    Each channel is centered at its batch Fréchet mean via the AIM logarithm, the tangent
    vectors are scaled by a learned positive factor over the geodesic standard deviation,
    parallel-transported to a learned target Fréchet mean ``expm(bias)``, and mapped back onto
    the manifold via the AIM exponential. The transport is the genuine geodesic parallel
    transport (an isometry); the two-hop construction of Brooks et al. (transport to the
    identity, then congruence to the target) is an isometry as well, but differs by a
    holonomy rotation for noncommuting means. Running statistics are tracked with a momentum
    update (geodesic EMA for the mean) for use in evaluation mode.

    Parameters
    ----------
    num_channels : int
        Number of input channels.
    num_coordinates : int, optional
        Number of Voigt coordinates per SPD matrix.
    eps : float, optional
        Value added to the variance for numerical stability.
    momentum : float, optional
        Momentum used to update running Fréchet statistics.

    """

    def __init__(
        self,
        num_channels: int,
        num_coordinates: int = 6,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.metric = "aim"
        self.num_coordinates = num_coordinates
        self.eps = eps
        self.momentum = momentum

        # learned target Fréchet mean (as a symmetric Voigt matrix log) and positive scale
        self.bias = nn.Parameter(torch.zeros(num_channels, num_coordinates))
        self.log_scale = nn.Parameter(torch.zeros(num_channels))
        self.register_buffer(
            "running_mean",
            torch.tensor(IDENTITY_VOIGT6).repeat(num_channels, 1),
        )
        self.register_buffer("running_var", torch.ones(num_channels))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input Voigt-encoded SPD field of shape (B, C, 6, D, H, W).
        mask : torch.Tensor or None, optional
            Binary spatial mask of shape (B, 1, D, H, W).

        Returns
        -------
        torch.Tensor
            Output SPD field of shape (B, C, 6, D, H, W). Voxels outside the mask are set to
            the identity matrix.

        """
        B, C, F, D, H, W = x.shape
        mats = voigt6_to_matrix(x.movedim(2, -1))
        eye3 = torch.eye(3, dtype=x.dtype, device=x.device)

        if mask is not None:
            valid = mask.to(torch.bool).squeeze(1)
            keep = valid.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            mats = torch.where(keep, mats, eye3)
            weights = valid.reshape(1, B * D * H * W).expand(C, -1).to(x.dtype)
        else:
            valid = None
            weights = x.new_ones(C, B * D * H * W)

        if self.training:
            # per-channel Fréchet statistics over the batch and spatial dimensions
            x_set = mats.permute(1, 0, 2, 3, 4, 5, 6).reshape(C, B * D * H * W, 3, 3)
            fm = karcher_wfm(x_set, weights, dim=-3)
            w_sum = weights.sum(dim=-1)
            has_stats = w_sum > 0
            # divide by the true weight sum wherever it is positive (see karcher_wfm)
            w_sum_safe = torch.where(has_stats, w_sum, torch.ones_like(w_sum)).unsqueeze(-1)
            w_norm = weights / w_sum_safe
            var = (w_norm * aim_distance_sq(x_set, fm.unsqueeze(-3))).sum(dim=-1)

            # channels without valid voxels fall back to the running statistics
            running = voigt6_to_matrix(self.running_mean.to(x.dtype))
            fm = torch.where(has_stats.view(C, 1, 1), fm, running)
            var = torch.where(has_stats, var, self.running_var.to(x.dtype))

            # geodesic EMA for the running mean; debiased EMA for the running variance
            n = w_sum
            debias = torch.where(n > 1, n / (n - 1).clamp_min(1), torch.ones_like(n))
            new_mean = matrix_to_voigt6(aim_gl_mean(running, fm.detach(), self.momentum))
            new_var = self.running_var * (1 - self.momentum) + var.detach() * debias * self.momentum
            keep_mean = torch.where(has_stats.unsqueeze(-1), new_mean, self.running_mean)
            self.running_mean.copy_(keep_mean)
            self.running_var.copy_(torch.where(has_stats, new_var, self.running_var))

            mean, var = fm, var
        else:
            mean = voigt6_to_matrix(self.running_mean.to(x.dtype))
            var = self.running_var.to(x.dtype)

        # normalize geodesic dispersion in the tangent space at the Fréchet mean, then
        # transport to the learned target Fréchet mean and map back onto the manifold
        # matrix_exp (Pade) instead of the eigh-based sym_expm: the eigh backward is
        # singular at the identity (bias = 0), where the target mean starts
        target = torch.linalg.matrix_exp(voigt6_to_matrix(self.bias))
        mean_b = mean.view(1, C, 1, 1, 1, 3, 3)
        target_b = target.view(1, C, 1, 1, 1, 3, 3)
        scale = (self.log_scale.exp() / torch.sqrt(var + self.eps)).view(1, C, 1, 1, 1, 1, 1)
        v = aim_log(mean_b, mats) * scale
        v = aim_parallel_transport(mean_b, target_b, v)
        out = matrix_to_voigt6(aim_exp(target_b, v)).movedim(-1, 2)

        if valid is not None:
            # masked-out voxels are set to the identity (not zeros) so downstream AIM layers
            # receive SPD inputs; the mask must be propagated alongside
            eye_voigt = torch.tensor(IDENTITY_VOIGT6, dtype=x.dtype, device=x.device)
            eye_voigt = eye_voigt.view(1, 1, 6, 1, 1, 1)
            out = torch.where(valid.unsqueeze(1).unsqueeze(2), out, eye_voigt)

        return out


class InvariantReadoutAIM(BaseInvariantReadout):
    """
    Invariant readout for SPD-valued feature maps under the AIM.

    Computes, per sample, the masked Fréchet mean over all fibers (channels and spatial
    positions), and outputs the squared affine-invariant geodesic distance from each fiber to
    it. This converts the SPD field to a scalar distance field, following the SPDLinear layer
    of the reference ManifoldNet implementation.
    """

    def __init__(self, **_kwargs):
        super().__init__()
        self.metric = "aim"

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input Voigt-encoded SPD field of shape (B, C, 6, D, H, W).
        mask : torch.Tensor or None, optional
            Binary spatial mask of shape (B, 1, D, H, W).

        Returns
        -------
        torch.Tensor
            Output squared-distance field of shape (B, C, D, H, W), zero outside the mask.

        """
        B, C, F, D, H, W = x.shape
        mats = voigt6_to_matrix(x.movedim(2, -1))

        # one Fréchet mean per sample over all channels and spatial positions
        if mask is not None:
            valid = mask.to(torch.bool).squeeze(1)
            eye3 = torch.eye(3, dtype=x.dtype, device=x.device)
            mats = torch.where(valid.unsqueeze(1).unsqueeze(-1).unsqueeze(-1), mats, eye3)
            weights = valid.unsqueeze(1).expand(B, C, D, H, W).reshape(B, C * D * H * W)
            weights = weights.to(x.dtype)
        else:
            weights = x.new_ones(B, C * D * H * W)

        x_set = mats.reshape(B, C * D * H * W, 3, 3)
        fm = karcher_wfm(x_set, weights, dim=-3)

        out = aim_distance_sq(mats, fm.view(B, 1, 1, 1, 1, 3, 3))

        if mask is not None:
            # also handles the empty-mask case: all distances are zeroed out
            out = out * mask.to(x.dtype)

        return out
