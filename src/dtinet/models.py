import torch.nn as nn

from dtinet.layers import InvariantReadoutLC, SPD3ToLC, WeightedFrechetMean3dLC


class DTINetLC(nn.Module):
    """
    DTI classifier using log-Cholesky weighted Fréchet mean layers.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    num_layers : int
        Number of weighted Fréchet mean layers.
    num_channels : list[int]
        Output channels for each weighted Fréchet mean layer.
    activation : str or None, optional
        Activation to apply after each weighted Fréchet mean layer. If None, no activation is used.
    **kwargs
        Keyword arguments passed to each weighted Fréchet mean layer.

    """

    def __init__(
        self,
        num_classes: int,
        num_layers: int,
        num_channels: list[int],
        activation: str | None = None,
        **kwargs,
    ):
        super().__init__()

        if len(num_channels) != num_layers:
            raise ValueError(
                f"Expected {num_layers} channel values, got {len(num_channels)}."
            )

        layers = []
        c_in = 1
        for c_out in num_channels:
            layers.append(WeightedFrechetMean3dLC(c_in, c_out, **kwargs))
            if activation is not None:
                layers.append(_build_activation(activation))
            c_in = c_out

        self.to_lc = SPD3ToLC()
        self.encoder = nn.Sequential(*layers)
        self.readout = InvariantReadoutLC()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(c_in, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input Voigt-encoded SPD(3) field of shape (B, C_in, 6, D, H, W).

        Returns
        -------
        torch.Tensor
            Output class logits of shape (B, num_classes).

        """
        x = self.to_lc(x)
        x = self.encoder(x)
        x = self.readout(x)
        x = self.pool(x).flatten(start_dim=1)
        return self.head(x)


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
