"""
Tests for the affine-invariant metric (AIM) machinery: geometry ops and layers.
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
from dtinet.geometry.spd import (
    matrix_to_voigt6,
    sym_expm,
    sym_inv_sqrtm,
    sym_logm,
    sym_powm,
    sym_sqrtm,
    voigt6_to_matrix,
)
from dtinet.layers import (
    BaseFrechetBatchNorm3d,
    BaseInvariantReadout,
    BaseWeightedFrechetMean3d,
    FrechetBatchNorm3d,
    FrechetBatchNorm3dAIM,
    FrechetBatchNorm3dLC,
    InvariantReadout,
    InvariantReadoutAIM,
    InvariantReadoutLC,
    WeightedFrechetMean3d,
    WeightedFrechetMean3dAIM,
    WeightedFrechetMean3dLC,
)
from dtinet.models import ManifoldNetClassifier

ATOL = 1e-5


def rand_spd(*batch_shape, scale=0.3, seed=None):
    """
    Random SPD(3) matrices close to the identity, of shape (*batch_shape, 3, 3).
    """
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    a = torch.randn(*batch_shape, 3, 3, generator=g)
    eye = torch.eye(3).expand(*batch_shape, 3, 3)
    return eye + scale * (a @ a.transpose(-1, -2))


def rand_sym(*batch_shape, scale=1.0, seed=None):
    """
    Random symmetric matrices of shape (*batch_shape, 3, 3).
    """
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    a = torch.randn(*batch_shape, 3, 3, generator=g)
    return scale * (a + a.transpose(-1, -2)) / 2


def rand_spd_field(B, C, D, H, W, seed=None):
    """
    Random Voigt-encoded SPD field of shape (B, C, 6, D, H, W).
    """
    return matrix_to_voigt6(rand_spd(B, C, D, H, W, seed=seed)).movedim(-1, 2)


def aim_norm_sq(p, v):
    """
    Squared AIM norm of a tangent vector ``v`` at ``p``.
    """
    evals, evecs = torch.linalg.eigh((p + p.transpose(-1, -2)) / 2)
    p_inv_sqrt = (evecs * evals.clamp_min(1e-12).rsqrt().unsqueeze(-2)) @ evecs.transpose(-1, -2)
    return (p_inv_sqrt @ v @ p_inv_sqrt).square().sum(dim=(-1, -2))


# --------------------------------------------------------------------------- safe eigh


def test_sym_spectral_backward_at_degenerate_eigenvalues():
    cases = [
        torch.eye(3),
        torch.diag(torch.tensor([2.0, 2.0, 1.0])),
        torch.stack((torch.eye(3), torch.diag(torch.tensor([1.0, 3.0, 3.0])))),
    ]
    fns = [
        sym_logm,
        sym_expm,
        sym_sqrtm,
        sym_inv_sqrtm,
        lambda m: sym_powm(m, 0.3),
    ]
    for case in cases:
        for fn in fns:
            m = case.clone().requires_grad_(True)
            fn(m).sum().backward()
            assert torch.isfinite(m.grad).all(), (case, fn)


def test_safe_eigh_backward_matches_torch_on_distinct_eigenvalues():
    m = rand_spd(3, seed=31)
    upstream = rand_sym(3, seed=32)

    m1 = m.clone().requires_grad_(True)
    (sym_logm(m1) * upstream).sum().backward()

    def naive_logm(x):
        x = (x + x.transpose(-1, -2)) / 2
        evals, evecs = torch.linalg.eigh(x)
        return (evecs * evals.clamp_min(1e-12).log().unsqueeze(-2)) @ evecs.transpose(-1, -2)

    m2 = m.clone().requires_grad_(True)
    (naive_logm(m2) * upstream).sum().backward()

    assert torch.allclose(m1.grad, m2.grad, atol=1e-4)


# --------------------------------------------------------------------------- geometry ops


def test_gl_mean_endpoints_and_midpoint():
    p, q = rand_spd(2, seed=2)[0], rand_spd(2, seed=3)[1]
    assert torch.allclose(aim_gl_mean(p, q, 0.0), p, atol=ATOL)
    assert torch.allclose(aim_gl_mean(p, q, 1.0), q, atol=ATOL)
    mid = aim_gl_mean(p, q, 0.5)
    d1 = aim_distance_sq(p, mid)
    d2 = aim_distance_sq(mid, q)
    assert torch.allclose(d1, d2, atol=ATOL)


def test_gl_mean_determinant_identity():
    p, q = rand_spd(2, seed=4)[0], rand_spd(2, seed=5)[1]
    for w in (0.25, 0.5, 0.75):
        m = aim_gl_mean(p, q, w)
        det_expected = torch.linalg.det(p) ** (1 - w) * torch.linalg.det(q) ** w
        assert torch.allclose(torch.linalg.det(m), det_expected, rtol=1e-5)


def test_exp_log_are_mutual_inverses():
    p = rand_spd(4, seed=6)
    q = rand_spd(4, seed=7)
    assert torch.allclose(aim_exp(p, aim_log(p, q)), q, atol=ATOL)
    v = rand_sym(4, scale=0.2, seed=8)
    assert torch.allclose(aim_log(p, aim_exp(p, v)), v, atol=ATOL)


def test_distance_symmetry_and_degeneracy():
    p = rand_spd(4, seed=9)
    q = rand_spd(4, seed=10)
    assert torch.allclose(aim_distance_sq(p, q), aim_distance_sq(q, p), atol=ATOL)
    assert torch.allclose(aim_distance_sq(p, p), torch.zeros(4), atol=ATOL)


def test_recursive_vs_karcher_uniform_weights():
    x = rand_spd(4, scale=0.2, seed=11)
    # inductive weights reproducing the uniform mean: w_j = j / (j + 1)
    w = torch.tensor([0.0, 1 / 2, 2 / 3, 3 / 4])
    fm_recursive = recursive_wfm(x, w, dim=0)
    fm_karcher = karcher_wfm(x, torch.ones(4), dim=0)
    # the inductive mean is only an approximation of the Fréchet mean on curved manifolds
    assert torch.allclose(fm_recursive, fm_karcher, atol=1e-2)


def test_karcher_recovers_identity_for_symmetric_pair():
    p = rand_spd(3, seed=12)
    p_inv = torch.linalg.inv(p)
    x = torch.stack((p, p_inv), dim=-3)  # (3, 2, 3, 3)
    w = torch.ones(3, 2)
    fm = karcher_wfm(x, w, dim=-3)
    eye = torch.eye(3).expand(3, 3, 3)
    assert torch.allclose(fm, eye, atol=1e-4)


def test_karcher_zero_weights_returns_identity():
    x = rand_spd(2, 4, seed=13)
    w = torch.zeros(2, 4)
    fm = karcher_wfm(x, w, dim=-3)
    eye = torch.eye(3).expand(2, 3, 3)
    assert torch.allclose(fm, eye, atol=ATOL)


def test_karcher_partial_zero_weights():
    x = rand_spd(4, seed=14)
    w = torch.tensor([0.0, 1.0, 0.0, 1.0])
    fm = karcher_wfm(x, w, dim=0)
    fm_pair = karcher_wfm(x[[1, 3]], torch.ones(2), dim=0)
    assert torch.allclose(fm, fm_pair, atol=1e-4)


def test_karcher_converges_on_dispersed_spds():
    # widely dispersed SPDs: log-eigenvalues spread over [-5, 5] (condition numbers up to
    # ~e^10) with random anisotropic orientations. float64 is required: in float32 the
    # variance-decrease acceptance test stalls near ~3e-4 because the decrease falls below
    # machine precision relative to the variance magnitude.
    g = torch.Generator().manual_seed(33)
    mats = []
    for _ in range(6):
        q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g))
        log_evals = torch.empty(3).uniform_(-5, 5, generator=g)
        mats.append(q @ torch.diag(log_evals.exp()) @ q.T)
    x = torch.stack(mats).double()

    fm = karcher_wfm(x, torch.ones(6, dtype=torch.float64), dim=0, max_iters=100, tol=1e-12)
    assert torch.isfinite(fm).all()
    assert (torch.linalg.eigvalsh(fm) > 0).all()

    # tangent residual of the Fréchet mean condition
    residual = aim_log(fm, x).mean(dim=0).square().sum().sqrt()
    assert residual < 1e-4


def test_parallel_transport_preserves_norm():
    p = rand_spd(4, seed=15)
    q = rand_spd(4, seed=16)
    v = rand_sym(4, scale=0.2, seed=17)
    v_q = aim_parallel_transport(p, q, v)
    assert torch.allclose(aim_norm_sq(q, v_q), aim_norm_sq(p, v), rtol=1e-4)


def test_parallel_transport_reverses_geodesic_velocity():
    # geodesic parallel transport maps Log_P(Q) to -Log_Q(P)
    p = rand_spd(4, scale=0.5, seed=40)
    q = rand_spd(4, scale=0.5, seed=41)
    v_q = aim_parallel_transport(p, q, aim_log(p, q))
    assert torch.allclose(v_q, -aim_log(q, p), atol=1e-4)


# --------------------------------------------------------------------------- layers


def test_wfm_dispatcher_and_isinstance():
    layer = WeightedFrechetMean3d(2, 3, 3, metric="aim", padding=1)
    assert isinstance(layer, WeightedFrechetMean3dAIM)
    assert isinstance(layer, BaseWeightedFrechetMean3d)
    assert layer.metric == "aim"


def test_wfm_forward_shapes_no_mask():
    layer = WeightedFrechetMean3dAIM(2, 3, 3, padding=1)
    x = rand_spd_field(2, 2, 4, 4, 4, seed=18)
    y = layer(x)
    assert y.shape == (2, 3, 6, 4, 4, 4)
    assert torch.isfinite(y).all()
    # outputs are SPD matrices
    evals = torch.linalg.eigvalsh(voigt6_to_matrix(y.movedim(2, -1)))
    assert (evals > 0).all()


def test_wfm_forward_with_mask():
    layer = WeightedFrechetMean3dAIM(2, 3, 3, padding=1)
    x = rand_spd_field(2, 2, 4, 4, 4, seed=19)
    mask = torch.zeros(2, 1, 4, 4, 4)
    mask[:, :, 2:, 2:, 2:] = 1
    y, out_mask = layer(x, mask)
    assert y.shape == (2, 3, 6, 4, 4, 4)
    assert out_mask.shape == (2, 1, 4, 4, 4)
    assert out_mask.dtype == torch.bool
    assert torch.isfinite(y).all()
    # fully-masked corner windows produce the identity matrix and a False mask
    eye_voigt = torch.tensor(IDENTITY_VOIGT6).view(1, 1, 6, 1, 1, 1)
    invalid = ~out_mask.unsqueeze(2).expand_as(y)
    assert (~out_mask).any()  # corner windows have no valid fibers
    assert torch.allclose(y[invalid], eye_voigt.expand_as(y)[invalid], atol=ATOL)


def test_wfm_backward_finite():
    layer = WeightedFrechetMean3dAIM(2, 3, 3, padding=1)
    x = rand_spd_field(2, 2, 4, 4, 4, seed=20).requires_grad_(True)
    mask = torch.ones(2, 1, 4, 4, 4)
    mask[:, :, :2] = 0
    y, _ = layer(x, mask)
    y.sum().backward()
    assert torch.isfinite(layer.weight.grad).all()
    assert torch.isfinite(x.grad).all()


def test_batchnorm_dispatcher_and_isinstance():
    layer = FrechetBatchNorm3d(3, metric="aim")
    assert isinstance(layer, FrechetBatchNorm3dAIM)
    assert isinstance(layer, BaseFrechetBatchNorm3d)
    assert layer.metric == "aim"


def test_batchnorm_train_and_eval_modes():
    layer = FrechetBatchNorm3dAIM(2)
    x = rand_spd_field(2, 2, 4, 4, 4, seed=21)

    layer.train()
    y = layer(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    mean_before = layer.running_mean.clone()
    layer(x)
    assert not torch.allclose(layer.running_mean, mean_before)

    layer.eval()
    y_eval = layer(x)
    assert y_eval.shape == x.shape
    assert torch.isfinite(y_eval).all()
    # outputs are SPD matrices
    evals = torch.linalg.eigvalsh(voigt6_to_matrix(y_eval.movedim(2, -1)))
    assert (evals > 0).all()


def test_batchnorm_with_mask():
    layer = FrechetBatchNorm3dAIM(2)
    layer.train()
    x = rand_spd_field(2, 2, 4, 4, 4, seed=22)
    mask = torch.zeros(2, 1, 4, 4, 4)
    mask[:, :, 1:3, 1:3, 1:3] = 1
    y = layer(x, mask)
    assert torch.isfinite(y).all()
    # masked-out voxels are set to the identity Voigt, not zeros
    eye_voigt = torch.tensor(IDENTITY_VOIGT6).view(1, 1, 6, 1, 1, 1)
    invalid = ~mask.to(torch.bool).unsqueeze(2).expand_as(y)
    assert torch.allclose(y[invalid], eye_voigt.expand_as(y)[invalid], atol=ATOL)


def test_batchnorm_empty_mask_falls_back_to_running_stats():
    layer = FrechetBatchNorm3dAIM(2)
    layer.train()
    x = rand_spd_field(2, 2, 3, 3, 3, seed=23)
    mask = torch.zeros(2, 1, 3, 3, 3)
    mean_before, var_before = layer.running_mean.clone(), layer.running_var.clone()
    y = layer(x, mask)
    assert torch.isfinite(y).all()
    assert torch.allclose(layer.running_mean, mean_before)
    assert torch.allclose(layer.running_var, var_before)
    # an entirely empty mask must still produce finite gradients
    x.requires_grad_(True)
    layer(x, mask).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.bias.grad).all()
    assert torch.isfinite(layer.log_scale.grad).all()


def test_batchnorm_backward_finite():
    layer = FrechetBatchNorm3dAIM(2)
    layer.train()
    x = rand_spd_field(2, 2, 3, 3, 3, seed=24).requires_grad_(True)
    mask = torch.ones(2, 1, 3, 3, 3)
    mask[:, :, 0] = 0
    y = layer(x, mask)
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.bias.grad).all()
    assert torch.isfinite(layer.log_scale.grad).all()


def test_readout_dispatcher_and_isinstance():
    layer = InvariantReadout(metric="aim")
    assert isinstance(layer, InvariantReadoutAIM)
    assert isinstance(layer, BaseInvariantReadout)
    assert layer.metric == "aim"


def test_readout_forward_and_backward():
    layer = InvariantReadoutAIM()
    x = rand_spd_field(2, 3, 4, 4, 4, seed=25).requires_grad_(True)
    out = layer(x)
    assert out.shape == (2, 3, 4, 4, 4)
    assert torch.isfinite(out).all()
    assert (out >= 0).all()
    out.sum().backward()
    assert torch.isfinite(x.grad).all()

    mask = torch.ones(2, 1, 4, 4, 4)
    mask[:, :, :2] = 0
    x_masked = x.detach().clone().requires_grad_(True)
    out_masked = layer(x_masked, mask)
    assert torch.isfinite(out_masked).all()
    assert (out_masked[mask.expand_as(out_masked) == 0] == 0).all()
    out_masked.sum().backward()
    assert torch.isfinite(x_masked.grad).all()


def test_readout_empty_mask_is_zero():
    layer = InvariantReadoutAIM()
    x = rand_spd_field(2, 3, 3, 3, 3, seed=26)
    mask = torch.zeros(2, 1, 3, 3, 3)
    out = layer(x, mask)
    assert torch.isfinite(out).all()
    assert (out == 0).all()


def test_readout_single_channel_spatially_varying_nonzero():
    # a spatially varying single-channel field must produce non-zero readout features
    layer = InvariantReadoutAIM()
    x = rand_spd_field(1, 1, 3, 3, 3, seed=42)
    out = layer(x)
    assert out.shape == (1, 1, 3, 3, 3)
    assert (out > 0).any()
    # a constant field has zero dispersion around the global Fréchet mean
    constant = torch.full((1, 1, 6, 3, 3, 3), 0.0)
    constant[:, :, [0, 2, 5]] = 1.0
    assert torch.allclose(layer(constant), torch.zeros(1, 1, 3, 3, 3), atol=1e-5)


def test_readout_coords_single_channel_spatially_varying_nonzero():
    layer = InvariantReadout(metric="lcm")
    x = torch.randn(1, 1, 6, 3, 3, 3)
    out = layer(x)
    assert out.shape == (1, 1, 3, 3, 3)
    assert (out > 0).any()
    assert torch.allclose(layer(torch.ones(1, 1, 6, 3, 3, 3)), torch.zeros(1, 1, 3, 3, 3))


# --------------------------------------------------------------------------- legacy aliases


def test_legacy_aliases_positional_signatures():
    wfm = WeightedFrechetMean3dLC(1, 2, 3, 2, 1)
    assert (wfm.in_channels, wfm.out_channels, wfm.kernel_size) == (1, 2, 3)
    assert (wfm.stride, wfm.padding, wfm.metric) == (2, 1, "lcm")

    bn = FrechetBatchNorm3dLC(2, 6)
    assert (bn.num_channels, bn.num_coordinates, bn.metric) == (2, 6, "lcm")

    readout = InvariantReadoutLC()
    assert readout.metric == "lcm"

    # forward smoke checks through the legacy positional constructors
    # conv output size: floor((4 + 2*1 - 3)/2) + 1 = 2
    x = torch.randn(1, 1, 6, 4, 4, 4)
    y = wfm(x)
    assert y.shape == (1, 2, 6, 2, 2, 2)
    z = bn(torch.randn(1, 2, 6, 2, 2, 2))
    assert z.shape == (1, 2, 6, 2, 2, 2)
    assert readout(z).shape == (1, 2, 2, 2, 2)


# --------------------------------------------------------------------------- training smoke test


def _synthetic_spd_volume(label, D=4, seed=0):
    """
    One synthetic SPD volume of shape (1, 1, 6, D, D, D) for a separable two-class problem.
    """
    g = torch.Generator().manual_seed(seed)
    if label == 0:
        s = torch.randn(3, 3, generator=g)
        s = (s + s.T) / 2
        base = sym_expm(0.3 * s)
    else:
        q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g))
        base = q @ torch.diag(torch.tensor([torch.e, torch.e**-0.5, torch.e**-0.5])) @ q.T
    log_base = sym_logm(base)
    noise = rand_sym(D, D, D, scale=0.05, seed=seed + 1)
    voxels = sym_expm(log_base + noise)
    return matrix_to_voigt6(voxels).movedim(-1, 0).unsqueeze(0).unsqueeze(0)


def test_manifoldnet_aim_training_smoke():
    torch.manual_seed(0)
    model = ManifoldNetClassifier(
        num_classes=2, num_layers=1, num_channels=[2], metric="aim", kernel_size=3, padding=1
    )
    x = torch.cat([_synthetic_spd_volume(i % 2, seed=10 * i) for i in range(4)], dim=0)
    labels = torch.tensor([0, 1, 0, 1])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    model.train()
    for _ in range(3):
        optimizer.zero_grad()
        logits = model(x)
        assert torch.isfinite(logits).all()
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss)
        loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[1] < losses[0]
