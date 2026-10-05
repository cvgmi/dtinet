"""
dtinet: ManifoldNet primitives for SPD(3)-valued images.

Implements the ManifoldNet architecture (Chakraborty et al., "ManifoldNet: A Deep Neural
Network for Manifold-valued Data with Applications", TPAMI 2022) for images taking values in
the manifold of 3x3 symmetric positive definite matrices, with pluggable metrics:

- "lcm": log-Cholesky metric
- "lem": log-Euclidean metric
- "aim": affine-invariant metric
"""

from dtinet.models import DTINetLC, ManifoldNetClassifier

__all__ = ["DTINetLC", "ManifoldNetClassifier"]
__version__ = "0.1.0"
