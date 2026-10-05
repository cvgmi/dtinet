import torch
import torch.nn as nn

from dtinet.geometry import check_metric, is_coord_metric
from dtinet.layers import (
    BaseFrechetBatchNorm3d,
    BaseWeightedFrechetMean3d,
    FrechetBatchNorm3d,
    InvariantReadout,
    SPDToCoords,
    WeightedFrechetMean3d,
)


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
