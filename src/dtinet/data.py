"""
Synthetic SPD(3)-valued 3D image datasets.

The datasets here generate DTI-like fields in which every voxel is an independent sample from
a class-conditional distribution on SPD(3), so a classifier must aggregate spatial statistics
rather than rely on any single discriminative voxel.
"""

import torch
from torch.utils.data import Dataset

from dtinet.geometry.spd import matrix_to_voigt6, sym_expm, sym_sqrtm


def _rand_symmetric(n: int, generator: torch.Generator) -> torch.Tensor:
    """
    Sample symmetric matrices with i.i.d. standard normal entries.

    Parameters
    ----------
    n : int
        Number of matrices to sample.
    generator : torch.Generator
        Random number generator.

    Returns
    -------
    torch.Tensor
        Symmetric matrices of shape (n, 3, 3).

    """
    s = torch.zeros(n, 3, 3)
    idx = torch.triu_indices(3, 3)
    s[:, idx[0], idx[1]] = torch.randn(n, 6, generator=generator)
    return s + s.triu(diagonal=1).transpose(-1, -2)


def _rand_rotation(n: int, generator: torch.Generator) -> torch.Tensor:
    """
    Sample random rotation matrices (Haar measure on SO(3)) via QR of Gaussian matrices.

    Parameters
    ----------
    n : int
        Number of rotations to sample.
    generator : torch.Generator
        Random number generator.

    Returns
    -------
    torch.Tensor
        Rotation matrices of shape (n, 3, 3) with determinant +1.

    """
    q, r = torch.linalg.qr(torch.randn(n, 3, 3, generator=generator))
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)
    neg = torch.linalg.det(q) < 0
    q[neg, :, 0] = -q[neg, :, 0]
    return q


class SyntheticSPDFieldDataset(Dataset):
    """
    Synthetic two-class classification dataset of SPD(3)-valued 3D images.

    Each sample is a ``grid_size``³ field in which every voxel is an i.i.d. draw from the
    class-conditional distribution:

    - Class 0 (isotropic): ``P = s² · expm(noise · S)`` with ``S`` a random symmetric matrix
      (i.i.d. standard normal entries). No preferred orientation.
    - Class 1 (anisotropic): ``P = s² · R diag(e^a, e^{-a/2}, e^{-a/2}) Rᵀ`` with a random
      rotation ``R`` per voxel, followed by a left/right square-root perturbation
      ``P ← P^{1/2} expm(noise · S) P^{1/2}``. Strong preferred orientation per voxel.

    In both classes the overall scale ``s = exp(scale_jitter · z)``, ``z ~ N(0, 1)``, is
    resampled per voxel so scale alone is not discriminative.

    Samples are fully deterministic per index: the label is ``index % 2`` and a dedicated
    generator seeded from ``(seed, index)`` draws the field, so results are reproducible
    regardless of shuffling or dataloader workers.

    Parameters
    ----------
    num_samples : int, optional
        Number of samples in the dataset.
    grid_size : int or tuple of int, optional
        Spatial grid size; a scalar is broadcast to all three axes.
    noise : float, optional
        Standard deviation of the symmetric-matrix perturbation in log-space.
    anisotropy : float, optional
        Log-eigenvalue ``a`` of the anisotropic class; the eigenvalue ratio is ``e^{3a/2}``.
    scale_jitter : float, optional
        Standard deviation of the per-voxel log-scale jitter.
    seed : int, optional
        Base seed combined with the sample index.
    return_mask : bool, optional
        If True, also return an all-ones foreground mask.

    """

    def __init__(
        self,
        num_samples: int = 512,
        grid_size: int | tuple[int, int, int] = 8,
        noise: float = 0.3,
        anisotropy: float = 1.0,
        scale_jitter: float = 0.2,
        seed: int = 0,
        return_mask: bool = False,
    ):
        if isinstance(grid_size, int):
            grid_size = (grid_size, grid_size, grid_size)
        if any(g <= 0 for g in grid_size):
            raise ValueError(f"grid_size must be positive, got {grid_size}.")

        self.num_samples = num_samples
        self.grid_size = tuple(grid_size)
        self.noise = noise
        self.anisotropy = anisotropy
        self.scale_jitter = scale_jitter
        self.seed = seed
        self.return_mask = return_mask

    def __len__(self) -> int:
        return self.num_samples

    def _generator(self, index: int) -> torch.Generator:
        """Build the deterministic per-sample generator."""
        g = torch.Generator()
        g.manual_seed((self.seed + 1) * 1_000_003 + index)
        return g

    def _sample_class0(self, n: int, g: torch.Generator) -> torch.Tensor:
        """Sample ``n`` isotropic SPD matrices of shape (n, 3, 3)."""
        noise = sym_expm(self.noise * _rand_symmetric(n, g))
        s2 = torch.exp(self.scale_jitter * torch.randn(n, 1, 1, generator=g)).square()
        return s2 * noise

    def _sample_class1(self, n: int, g: torch.Generator) -> torch.Tensor:
        """Sample ``n`` anisotropic SPD matrices of shape (n, 3, 3)."""
        a = self.anisotropy
        d = torch.tensor([[a, -a / 2, -a / 2]]).exp().expand(n, 3)
        rot = _rand_rotation(n, g)
        p = (rot * d.unsqueeze(-2)) @ rot.transpose(-1, -2)
        p_sqrt = sym_sqrtm(p)
        p = p_sqrt @ sym_expm(self.noise * _rand_symmetric(n, g)) @ p_sqrt
        s2 = torch.exp(self.scale_jitter * torch.randn(n, 1, 1, generator=g)).square()
        return s2 * p

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        """
        Return one sample.

        Parameters
        ----------
        index : int
            Sample index; the label is ``index % 2``.

        Returns
        -------
        tuple of torch.Tensor
            ``(x, y)`` or, if ``return_mask``, ``(x, mask, y)`` where ``x`` is the
            Voigt-encoded field of shape (1, 6, D, H, W) in float32, ``mask`` is an
            all-ones mask of shape (1, D, H, W), and ``y`` is an int64 scalar label.

        """
        if not 0 <= index < self.num_samples:
            raise IndexError(f"Index {index} out of range for {self.num_samples} samples.")

        g = self._generator(index)
        label = index % 2
        d, h, w = self.grid_size
        n = d * h * w

        p = self._sample_class1(n, g) if label == 1 else self._sample_class0(n, g)
        x = matrix_to_voigt6(p).transpose(0, 1).reshape(1, 6, d, h, w).float()
        y = torch.tensor(label, dtype=torch.long)

        if self.return_mask:
            mask = torch.ones(1, d, h, w)
            return x, mask, y
        return x, y
