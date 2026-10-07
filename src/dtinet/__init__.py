"""
dtinet: ManifoldNet primitives for SPD(3)-valued images.

Implements the building blocks of ManifoldNet (Chakraborty et al., "ManifoldNet: A Deep Neural
Network for Manifold-valued Data with Applications", TPAMI 2022) for images taking values in
the manifold of 3x3 symmetric positive definite matrices, with pluggable metrics:

- "lcm": log-Cholesky metric
- "lem": log-Euclidean metric
- "aim": affine-invariant metric

``dtinet.geometry`` holds the metric mathematics and ``dtinet.layers`` the network layers.
Assembled models and training code live in the repository's ``experiments`` package.
"""

__version__ = "0.1.0"
