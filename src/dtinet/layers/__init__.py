"""
ManifoldNet layers for SPD(3)-valued 3D lattice data.

All layer classes dispatch on a ``metric`` argument:

- "lcm": log-Cholesky metric (coordinate metric; Euclidean operations in log-Cholesky
  coordinates).
- "lem": log-Euclidean metric (coordinate metric; Euclidean operations in scaled log-matrix
  coordinates).
- "aim": affine-invariant metric (Riemannian log/exp maps and Fréchet means).
"""

from dtinet.layers.aim import (
    FrechetBatchNorm3dAIM,
    InvariantReadoutAIM,
    WeightedFrechetMean3dAIM,
)
from dtinet.layers.batchnorm import (
    BaseFrechetBatchNorm3d,
    FrechetBatchNorm3d,
    FrechetBatchNorm3dCoords,
    FrechetBatchNorm3dLC,
)
from dtinet.layers.bimap import BiMap3dLC
from dtinet.layers.coords import SPD3ToLC, SPDToCoords
from dtinet.layers.readout import (
    BaseInvariantReadout,
    InvariantReadout,
    InvariantReadoutCoords,
    InvariantReadoutLC,
)
from dtinet.layers.residual import ManifoldResidualBlock3d
from dtinet.layers.wfm import (
    BaseWeightedFrechetMean3d,
    WeightedFrechetMean3d,
    WeightedFrechetMean3dCoords,
    WeightedFrechetMean3dLC,
)

__all__ = [
    "BaseFrechetBatchNorm3d",
    "BaseInvariantReadout",
    "BaseWeightedFrechetMean3d",
    "BiMap3dLC",
    "FrechetBatchNorm3d",
    "FrechetBatchNorm3dAIM",
    "FrechetBatchNorm3dCoords",
    "FrechetBatchNorm3dLC",
    "InvariantReadout",
    "InvariantReadoutAIM",
    "InvariantReadoutCoords",
    "InvariantReadoutLC",
    "ManifoldResidualBlock3d",
    "SPD3ToLC",
    "SPDToCoords",
    "WeightedFrechetMean3d",
    "WeightedFrechetMean3dAIM",
    "WeightedFrechetMean3dCoords",
    "WeightedFrechetMean3dLC",
]
