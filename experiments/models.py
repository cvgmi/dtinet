import torch
import torch.nn as nn

from dtinet.geometry import check_metric, is_coord_metric
from dtinet.layers import (
    BaseFrechetBatchNorm3d,
    BaseWeightedFrechetMean3d,
    FrechetBatchNorm3d,
    FrechetBatchNorm3dCoords,
    InvariantReadout,
    ManifoldResidualBlock3d,
    SPDToCoords,
    WeightedFrechetMean3d,
    WeightedFrechetMean3dCoords,
)
from dtinet.layers.bimap import BiMap3dLC
from dtinet.layers.residual import masked


class ManifoldNetClassifier(nn.Module):
    """
    ManifoldNet classifier for SPD(3)-valued 3D images.

    Stacks weighted Fréchet mean layers (the manifold analogue of convolutions), Fréchet batch
    normalization, and an invariant readout, followed by masked global average pooling and a
    linear classification head. All primitives are selected by the ``metric`` argument.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    num_layers : int
        Number of weighted Fréchet mean layers.
    num_channels : list[int]
        Output channels for each weighted Fréchet mean layer.
    metric : {"lcm", "lem", "aim"}, optional
        Metric on SPD(3) used by all layers. "lcm" (log-Cholesky) and "lem" (log-Euclidean) are
        coordinate metrics; "aim" is the affine-invariant metric.
    activation : str or None, optional
        Activation to apply after each weighted Fréchet mean layer. Only supported for
        coordinate metrics, where it acts on the Euclidean coordinates. If None, no activation
        is used.
    **kwargs
        Keyword arguments passed to each weighted Fréchet mean layer.

    """

    def __init__(
        self,
        num_classes: int,
        num_layers: int,
        num_channels: list[int],
        metric: str = "lcm",
        activation: str | None = None,
        **kwargs,
    ):
        super().__init__()

        check_metric(metric)
        if activation is not None and not is_coord_metric(metric):
            raise ValueError(
                f"Activations are only supported for coordinate metrics, got metric={metric!r}."
            )

        if len(num_channels) != num_layers:
            raise ValueError(f"Expected {num_layers} channel values, got {len(num_channels)}.")

        self.metric = metric

        layers = []
        c_in = 1
        for c_out in num_channels:
            layers.append(WeightedFrechetMean3d(c_in, c_out, metric=metric, **kwargs))
            layers.append(FrechetBatchNorm3d(c_out, metric=metric))
            if activation is not None:
                layers.append(_build_activation(activation))
            c_in = c_out

        # coordinate metrics operate on Euclidean coordinates; "aim" operates directly on the
        # Voigt-encoded SPD matrices
        self.to_coords = SPDToCoords(metric) if is_coord_metric(metric) else nn.Identity()
        self.encoder = nn.ModuleList(layers)
        self.readout = InvariantReadout(metric)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(c_in, num_classes)

    def forward(self, x, mask: torch.Tensor | None = None):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input Voigt-encoded SPD(3) field of shape (B, C_in, 6, D, H, W).
        mask : torch.Tensor or None, optional
            Binary spatial mask of shape (B, 1, D, H, W).

        Returns
        -------
        torch.Tensor
            Output class logits of shape (B, num_classes).

        """
        x = self.to_coords(x, mask) if is_coord_metric(self.metric) else self.to_coords(x)
        for layer in self.encoder:
            if isinstance(layer, BaseWeightedFrechetMean3d):
                if mask is not None:
                    x, mask = layer(x, mask)
                else:
                    x = layer(x)
            elif isinstance(layer, BaseFrechetBatchNorm3d):
                x = layer(x, mask)
            else:
                x = layer(x)
                if mask is not None:
                    x = x * mask.unsqueeze(2)

        x = self.readout(x, mask)

        if mask is None:
            x = self.pool(x).flatten(start_dim=1)
        else:
            x = (x * mask).sum(dim=(2, 3, 4)) / mask.sum(dim=(2, 3, 4)).clamp_min(1)

        return self.head(x)


def _masked_mean_std(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-channel masked mean and standard deviation of a (B, C, D, H, W) field -> (B, 2C)."""
    weights = mask.to(x.dtype)
    count = weights.sum(dim=(2, 3, 4)).clamp_min(1)
    mean = (x * weights).sum(dim=(2, 3, 4)) / count
    var = ((x - mean[..., None, None, None]).square() * weights).sum(dim=(2, 3, 4)) / count
    return torch.cat([mean, torch.sqrt(var + 1e-8)], dim=1)


class ManifoldResNetClassifier(nn.Module):
    """
    Residual ManifoldNet for SPD(3)-valued 3D images under a coordinate metric (LCM or LEM).

    Follows standard CNN design while keeping every operation a weighted Fréchet mean,
    Fréchet batch normalization, or a coordinate-wise activation:

    1. Stem: a stride-1 wFM layer at full resolution, so voxel-level information passes a
       per-channel normalization and nonlinearity before any downsampling. With ``bimap``,
       every normalization is followed by a voxelwise congruence that mixes components.
    2. Stages: each halves the resolution with a stride-2 wFM layer and then applies
       ``blocks_per_stage`` residual blocks (:class:`ManifoldResidualBlock3d`); channel
       counts typically double from stage to stage.
    3. Readout: per-voxel squared distance to the sample's Fréchet mean, pooled per channel
       into its masked mean and standard deviation, followed by dropout and a linear head.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    stem_channels : int
        Output channels of the full-resolution stem.
    stage_channels : list[int]
        Output channels of each downsampling stage.
    blocks_per_stage : int, optional
        Residual blocks after each downsampling layer.
    metric : {"lcm", "lem"}, optional
        Coordinate metric.
    activation : str or None, optional
        Coordinate-wise activation; None disables activations.
    kernel_size : int, optional
        Size of all cubic kernels.
    dropout : float, optional
        Dropout probability on the pooled features.
    init_std : float, optional
        Initialization scale of the pre-softmax wFM weights.
    bimap : bool, optional
        Insert a per-channel voxelwise congruence ``P -> W P W^T`` (:class:`BiMap3dLC`) after
        every batch normalization. wFM layers and coordinate-wise activations never combine
        tensor components; BiMap does, while keeping features SPD-valued. LCM only.

    """

    def __init__(
        self,
        num_classes: int,
        stem_channels: int,
        stage_channels: list[int],
        blocks_per_stage: int = 1,
        metric: str = "lcm",
        activation: str | None = "elu",
        kernel_size: int = 3,
        dropout: float = 0.0,
        init_std: float = 1.0,
        bimap: bool = False,
    ):
        super().__init__()
        if not is_coord_metric(metric):
            raise ValueError(f"{type(self).__name__} requires a coordinate metric, got {metric!r}.")
        if bimap and metric != "lcm":
            raise ValueError(f"bimap is implemented for metric='lcm' only, got {metric!r}")
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        padding = kernel_size // 2

        def mix(channels: int) -> nn.Module:
            return BiMap3dLC(channels) if bimap else nn.Identity()

        def act() -> nn.Module:
            return _build_activation(activation) if activation is not None else nn.Identity()

        self.metric = metric
        self.to_coords = SPDToCoords(metric)
        self.stem = WeightedFrechetMean3dCoords(
            1, stem_channels, kernel_size, metric=metric, padding=padding, init_std=init_std
        )
        self.stem_bn = FrechetBatchNorm3dCoords(stem_channels, metric=metric)
        self.stem_mix = mix(stem_channels)
        self.stem_act = act()
        self.downsample = nn.ModuleList()
        self.down_bn = nn.ModuleList()
        self.down_mix = nn.ModuleList()
        self.down_act = nn.ModuleList()
        self.blocks = nn.ModuleList()
        c_in = stem_channels
        for c_out in stage_channels:
            self.downsample.append(
                WeightedFrechetMean3dCoords(
                    c_in, c_out, kernel_size, metric=metric, stride=2, padding=padding,
                    init_std=init_std,
                )
            )
            self.down_bn.append(FrechetBatchNorm3dCoords(c_out, metric=metric))
            self.down_mix.append(mix(c_out))
            self.down_act.append(act())
            self.blocks.append(
                nn.ModuleList(
                    ManifoldResidualBlock3d(
                        c_out, metric=metric, kernel_size=kernel_size, activation=act(),
                        init_std=init_std, bimap=bimap,
                    )
                    for _ in range(blocks_per_stage)
                )
            )
            c_in = c_out
        self.readout = InvariantReadout(metric)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * c_in, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input Voigt-encoded SPD(3) field of shape (B, 1, 6, D, H, W).
        mask : torch.Tensor or None, optional
            Binary spatial mask of shape (B, 1, D, H, W); None treats every voxel as valid.

        Returns
        -------
        torch.Tensor
            Output class logits of shape (B, num_classes).

        """
        if mask is None:
            mask = x.new_ones(x.shape[0], 1, *x.shape[-3:])
        x = self.to_coords(x, mask)
        x, _ = self.stem(x, mask)
        x = masked(self.stem_act(self.stem_mix(self.stem_bn(masked(x, mask), mask))), mask)
        for down, bn, mix, act, blocks in zip(
            self.downsample, self.down_bn, self.down_mix, self.down_act, self.blocks, strict=True
        ):
            x, mask = down(x, mask)
            mask = mask.to(x.dtype)
            x = masked(act(mix(bn(x, mask))), mask)
            for block in blocks:
                x = block(x, mask)
        distances = self.readout(x, mask)
        return self.head(self.dropout(_masked_mean_std(distances, mask)))


class DTINetLC(ManifoldNetClassifier):
    """
    DTI classifier using log-Cholesky weighted Fréchet mean layers.

    Legacy alias for ``ManifoldNetClassifier(metric="lcm")``.
    """

    def __init__(
        self,
        num_classes: int,
        num_layers: int,
        num_channels: list[int],
        activation: str | None = None,
        **kwargs,
    ):
        super().__init__(
            num_classes,
            num_layers,
            num_channels,
            metric="lcm",
            activation=activation,
            **kwargs,
        )


def _build_activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }
    try:
        return activations[name.lower()]()
    except KeyError as exc:
        supported = ", ".join(sorted(activations))
        raise ValueError(f"Unsupported activation {name!r}. Supported: {supported}.") from exc
