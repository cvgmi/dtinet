"""
Metric registry for SPD(3)-valued feature maps.

Two families of metrics are supported:

- Coordinate metrics ("lcm", "lem"): admit a global coordinate chart onto Euclidean space
  (log-Cholesky and log-Euclidean coordinates, respectively). All ManifoldNet primitives
  (weighted Fréchet mean, Fréchet batch normalization, invariant readout) reduce to Euclidean
  operations in coordinates.
- "aim": the affine-invariant metric. No global coordinate chart exists; the primitives require
  genuine Riemannian machinery (log/exp maps, recursive or iterative Fréchet means).
"""

COORD_METRICS = ("lcm", "lem")
METRICS = ("lcm", "lem", "aim")

#: Number of scalar coordinates used to represent one SPD(3) matrix.
NUM_COORDS = 6

#: Voigt-6 encoding of the 3x3 identity matrix.
IDENTITY_VOIGT6 = (1.0, 0.0, 1.0, 0.0, 0.0, 1.0)


def check_metric(metric: str) -> str:
    """
    Validate a metric name, returning it unchanged.

    Raises
    ------
    ValueError
        If the metric is not supported.

    """
    if metric not in METRICS:
        supported = ", ".join(METRICS)
        raise ValueError(f"Unsupported metric {metric!r}. Supported: {supported}.")
    return metric


def is_coord_metric(metric: str) -> bool:
    """
    Whether the metric admits a global Euclidean coordinate chart.
    """
    return check_metric(metric) in COORD_METRICS
