"""
Affine-invariant metric (AIM) operations on SPD(3) matrices.

Under the affine-invariant metric, the geodesic from P to Q is
``P^{1/2} (P^{-1/2} Q P^{-1/2})^t P^{1/2}``, and the squared geodesic distance is
``||logm(P^{-1/2} Q P^{-1/2})||_F^2``. Unlike the coordinate metrics (LCM, LEM), there is no
global Euclidean chart, so Fréchet means are computed either recursively (inductive two-point
means, as in the reference ManifoldNet implementation) or iteratively (a fully batched
Karcher/Newton iteration).

All helpers are batched and operate on the trailing (3, 3) matrix dimensions, so they work for
arbitrary batch shapes (..., 3, 3).
"""

import torch

from dtinet.geometry.spd import (
    sym_expm,
    sym_inv_sqrtm,
    sym_logm,
    sym_powm,
    sym_sqrtm,
)


def aim_gl_mean(p: torch.Tensor, q: torch.Tensor, w: float | torch.Tensor) -> torch.Tensor:
    """
    Two-point weighted geodesic mean under the affine-invariant metric.

    Computes ``P^{1/2} (P^{-1/2} Q P^{-1/2})^w P^{1/2}``, the point at fraction ``w`` of the
    geodesic from ``p`` (``w`` = 0) to ``q`` (``w`` = 1).

    Parameters
    ----------
    p : torch.Tensor
        SPD matrices of shape (..., 3, 3).
    q : torch.Tensor
        SPD matrices of shape (..., 3, 3), broadcastable against ``p``.
    w : float or torch.Tensor
        Geodesic fraction weighting ``q``. If a tensor, it must be broadcastable against the
        batch shape of ``p`` and ``q``.

    Returns
    -------
    torch.Tensor
        SPD matrices of shape (..., 3, 3).

    """
    p_sqrt = sym_sqrtm(p)
    p_inv_sqrt = sym_inv_sqrtm(p)
    mid = p_inv_sqrt @ q @ p_inv_sqrt
    return p_sqrt @ sym_powm(mid, w) @ p_sqrt


def aim_log(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Riemannian logarithm of ``q`` at base point ``p`` under the affine-invariant metric.

    Computes ``P^{1/2} logm(P^{-1/2} Q P^{-1/2}) P^{1/2}``, a symmetric matrix in the tangent
    space at ``p``.

    Parameters
    ----------
    p : torch.Tensor
        SPD base points of shape (..., 3, 3).
    q : torch.Tensor
        SPD matrices of shape (..., 3, 3), broadcastable against ``p``.

    Returns
    -------
    torch.Tensor
        Symmetric tangent vectors at ``p`` of shape (..., 3, 3).

    """
    p_sqrt = sym_sqrtm(p)
    p_inv_sqrt = sym_inv_sqrtm(p)
    return p_sqrt @ sym_logm(p_inv_sqrt @ q @ p_inv_sqrt) @ p_sqrt


def aim_exp(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Riemannian exponential of tangent vector ``v`` at base point ``p``.

    Computes ``P^{1/2} expm(P^{-1/2} V P^{-1/2}) P^{1/2}`` for symmetric ``v``. Using the
    congruence ``P^{-1/2} V P^{-1/2} = P^{-1/2} (V P^{-1}) P^{1/2}``, this is evaluated as
    ``expm(V P^{-1}) P`` with a Pade matrix exponential, whose backward pass stays finite at
    degenerate base points (e.g. exactly at the identity) where the symmetric
    eigendecomposition is singular.

    Parameters
    ----------
    p : torch.Tensor
        SPD base points of shape (..., 3, 3).
    v : torch.Tensor
        Symmetric tangent vectors at ``p`` of shape (..., 3, 3), broadcastable against ``p``.

    Returns
    -------
    torch.Tensor
        SPD matrices of shape (..., 3, 3).

    """
    v = (v + v.transpose(-1, -2)) / 2
    out = torch.linalg.matrix_exp(v @ torch.linalg.inv(p)) @ p
    return (out + out.transpose(-1, -2)) / 2


def aim_distance_sq(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Squared affine-invariant geodesic distance, ``||logm(P^{-1/2} Q P^{-1/2})||_F^2``.

    Parameters
    ----------
    p : torch.Tensor
        SPD matrices of shape (..., 3, 3).
    q : torch.Tensor
        SPD matrices of shape (..., 3, 3), broadcastable against ``p``.

    Returns
    -------
    torch.Tensor
        Squared distances of shape (...).

    """
    p_inv_sqrt = sym_inv_sqrtm(p)
    log = sym_logm(p_inv_sqrt @ q @ p_inv_sqrt)
    return log.square().sum(dim=(-1, -2))


def recursive_wfm(x: torch.Tensor, weights: torch.Tensor, dim: int = -3) -> torch.Tensor:
    """
    Recursive (inductive) weighted Fréchet mean along a set dimension.

    Sequentially applies exact two-point geodesic means: the running mean is initialized at the
    first fiber, and each subsequent fiber is folded in via
    ``running = aim_gl_mean(fiber_k, running, w_k)``, so ``w_k`` weights the running mean. A
    weight of exactly 1 leaves the running mean unchanged (skips the fiber); a weight of 0
    replaces it with the fiber.

    Parameters
    ----------
    x : torch.Tensor
        SPD fibers of shape (..., K, 3, 3) with the set dimension at ``dim``.
    weights : torch.Tensor
        Per-fiber weights of shape (..., K), aligned with ``dim`` (the weight of the first
        fiber is unused). Tensor weights allow per-element (e.g. mask-dependent) weighting.
    dim : int, optional
        Dimension of ``x`` holding the set of fibers.

    Returns
    -------
    torch.Tensor
        SPD means of shape (..., 3, 3) with the set dimension removed.

    """
    abs_dim = dim % x.ndim
    x = x.movedim(abs_dim, -3)
    w = weights.to(x.dtype).movedim(abs_dim, -1)

    fibers = x.unbind(dim=-3)
    ws = w.unbind(dim=-1)
    running = fibers[0]
    for fiber, w_k in zip(fibers[1:], ws[1:]):
        running = aim_gl_mean(fiber, running, w_k)
    return running


def _weighted_aim_variance(
    x: torch.Tensor, mean: torch.Tensor, w_mat: torch.Tensor
) -> torch.Tensor:
    """
    Weighted Fréchet variance, ``sum_i w_i d^2(X_i, M)``, with broadcast set dimension -3.
    """
    m_inv_sqrt = sym_inv_sqrtm(mean).unsqueeze(-3)
    logs = sym_logm(m_inv_sqrt @ x @ m_inv_sqrt)
    return (w_mat * logs.square()).sum(dim=(-3, -2, -1))


def karcher_wfm(
    x: torch.Tensor,
    weights: torch.Tensor,
    dim: int = -3,
    max_iters: int = 20,
    tol: float = 1e-10,
) -> torch.Tensor:
    """
    Iterative weighted Fréchet mean under the affine-invariant metric.

    Fully batched Karcher/Newton iteration: initialized at the weighted arithmetic mean, each
    step computes the tangent update ``V = sum_i w_i Log_{M}(X_i)`` and then backtracks a step
    size ``t`` in {1, 1/2, 1/4, ...} (up to 10 halvings), accepting the first ``t`` that
    decreases the weighted Fréchet variance. Iteration stops when the AIM norm of the update
    falls below ``tol`` or when no step size decreases the variance; in the latter case (or
    when ``max_iters`` is exhausted) the best iterate found is returned. Weights are
    renormalized per problem; zero weights mark invalid elements. Problems whose weights sum
    to zero return the identity matrix.

    Parameters
    ----------
    x : torch.Tensor
        SPD matrices of shape (..., K, 3, 3) with the set dimension at ``dim``.
    weights : torch.Tensor
        Non-negative per-element weights of shape (..., K), aligned with ``dim``.
    dim : int, optional
        Dimension of ``x`` holding the set of matrices.
    max_iters : int, optional
        Maximum number of Karcher iterations.
    tol : float, optional
        Convergence tolerance on the AIM norm of the tangent update.

    Returns
    -------
    torch.Tensor
        SPD means of shape (..., 3, 3) with the set dimension removed.

    """
    abs_dim = dim % x.ndim
    x = x.movedim(abs_dim, -3)
    w = weights.to(x.dtype).movedim(abs_dim, -1)

    w_sum = w.sum(dim=-1, keepdim=True)
    has_mass = w_sum > 0
    # divide by the true sum wherever it is positive; an epsilon floor would
    # squash tiny-but-legitimate masses toward zero
    w = w / torch.where(has_mass, w_sum, torch.ones_like(w_sum))
    w_mat = w.unsqueeze(-1).unsqueeze(-1)

    # weighted arithmetic mean of SPD matrices is SPD for non-negative weights
    mean = (w_mat * x).sum(dim=-3)
    mean = (mean + mean.transpose(-1, -2)) / 2

    for _ in range(max_iters):
        m_sqrt = sym_sqrtm(mean)
        m_inv_sqrt = sym_inv_sqrtm(mean).unsqueeze(-3)
        # the base-point factors cancel in the weighted sum of tangent vectors, so the update
        # can be computed in whitened coordinates with a single logm per iteration
        logs = sym_logm(m_inv_sqrt @ x @ m_inv_sqrt)
        step = (w_mat * logs).sum(dim=-3)
        step = (step + step.transpose(-1, -2)) / 2
        if bool((step.square().sum(dim=(-1, -2)).detach() < tol * tol).all()):
            break

        # current weighted Fréchet variance comes for free in whitened coordinates
        var = (w_mat * logs.square()).sum(dim=(-3, -2, -1))

        # backtracking line search on the step size, per problem in the batch
        t = torch.ones_like(var)
        accepted = torch.zeros_like(var, dtype=torch.bool)
        best = mean
        for _ in range(10):
            step_t = t.unsqueeze(-1).unsqueeze(-1) * step
            candidate = m_sqrt @ sym_expm(step_t) @ m_sqrt
            var_t = _weighted_aim_variance(x, candidate, w_mat)
            newly = (var_t < var) & ~accepted
            best = torch.where(newly.unsqueeze(-1).unsqueeze(-1), candidate, best)
            accepted = accepted | newly
            t = torch.where(accepted, t, t * 0.5)
            if bool(accepted.all()):
                break

        # problems that never decreased the variance keep their previous (best) iterate
        mean = torch.where(accepted.unsqueeze(-1).unsqueeze(-1), best, mean)
        if not bool(accepted.any()):
            break

    eye = torch.eye(3, dtype=x.dtype, device=x.device).expand_as(mean)
    return torch.where(has_mass.unsqueeze(-1), mean, eye)


def aim_parallel_transport(p: torch.Tensor, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Parallel transport of tangent vector ``v`` at ``p`` to ``q`` along the AIM geodesic.

    Computes ``E V Eᵀ`` with ``E = P^{1/2} (P^{-1/2} Q P^{-1/2})^{1/2} P^{-1/2}``, which equals
    ``(Q P^{-1})^{1/2}``. This is the genuine parallel transport map of the affine-invariant
    metric along the geodesic from ``p`` to ``q``: it is an isometry between the tangent
    spaces and maps the geodesic velocity ``Log_P(Q)`` to ``-Log_Q(P)``. (Note that E is not
    symmetric in general; the symmetric square-root identification
    ``P^{-1/2} (P^{1/2} Q P^{1/2})^{1/2} P^{-1/2}`` is a different map.)

    Parameters
    ----------
    p : torch.Tensor
        SPD base points of shape (..., 3, 3).
    q : torch.Tensor
        SPD target points of shape (..., 3, 3), broadcastable against ``p``.
    v : torch.Tensor
        Symmetric tangent vectors at ``p`` of shape (..., 3, 3), broadcastable against ``p``.

    Returns
    -------
    torch.Tensor
        Symmetric tangent vectors at ``q`` of shape (..., 3, 3).

    """
    p_sqrt = sym_sqrtm(p)
    p_inv_sqrt = sym_inv_sqrtm(p)
    e = p_sqrt @ sym_sqrtm(p_inv_sqrt @ q @ p_inv_sqrt) @ p_inv_sqrt
    return e @ v @ e.transpose(-1, -2)
