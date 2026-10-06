"""
Diagnose CUSOLVER_STATUS_INTERNAL_ERROR in batched torch.linalg.eigh on CUDA.

Sweeps batch counts of random SPD(3) matrices through torch.linalg.eigh under the
default (cuSOLVER) backend, then retries the failing sizes with the MAGMA backend
(if available), with chunked cuSOLVER calls, and with a CPU fallback. Run on a GPU
node, e.g. via slurm/diagnose_eigh.sbatch.
"""

import sys

import torch

SIZES = [1024, 8192, 32768, 65535, 65536, 65537, 131072, 262144, 524288]
CHUNK = 32768


def rand_spd(batch: int, device: torch.device) -> torch.Tensor:
    f = torch.randn(batch, 3, 3, device=device) * 0.15
    return f @ f.transpose(-1, -2) + torch.eye(3, device=device) * 0.5


def try_eigh(m: torch.Tensor, label: str) -> bool:
    try:
        evals, evecs = torch.linalg.eigh(m)
        torch.cuda.synchronize()
        assert torch.isfinite(evals).all() and torch.isfinite(evecs).all()
        print(f"  {label}: OK")
        return True
    except RuntimeError as exc:
        print(f"  {label}: FAILED ({str(exc).splitlines()[0]})")
        return False


def chunked_eigh(m: torch.Tensor, chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
    flat = m.reshape(-1, 3, 3)
    evals_out, evecs_out = [], []
    for slab in flat.split(chunk):
        e, v = torch.linalg.eigh(slab)
        evals_out.append(e)
        evecs_out.append(v)
    evals = torch.cat(evals_out).reshape(*m.shape[:-1])
    evecs = torch.cat(evecs_out).reshape(*m.shape)
    return evals, evecs


def main() -> int:
    print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("CUDA not available; nothing to diagnose.")
        return 1
    device = torch.device("cuda")
    print("device:", torch.cuda.get_device_name(0))
    print("linalg backend:", torch.backends.cuda.preferred_linalg_library())

    failures: list[int] = []
    print("\n== default backend sweep ==")
    for n in SIZES:
        m = rand_spd(n, device)
        if not try_eigh(m, f"batch={n}"):
            failures.append(n)

    if not failures:
        print("\nNo failures reproduced at any batch size.")
        return 0

    print(f"\nfailed batch sizes: {failures}")

    print("\n== MAGMA backend retry ==")
    try:
        torch.backends.cuda.preferred_linalg_library("magma")
        for n in failures:
            try_eigh(rand_spd(n, device), f"magma batch={n}")
    except RuntimeError as exc:
        print(f"  magma unavailable: {exc}")
    finally:
        torch.backends.cuda.preferred_linalg_library("cusolver")

    print(f"\n== chunked cuSOLVER (chunk={CHUNK}) ==")
    for n in failures:
        m = rand_spd(n, device)
        try:
            evals, _ = chunked_eigh(m, CHUNK)
            torch.cuda.synchronize()
            print(f"  chunked batch={n}: OK (finite={bool(torch.isfinite(evals).all())})")
        except RuntimeError as exc:
            print(f"  chunked batch={n}: FAILED ({str(exc).splitlines()[0]})")

    print("\n== near-identity inputs (AIM early-training regime), largest size ==")
    n = max(failures)
    m = torch.eye(3, device=device).expand(n, 3, 3).contiguous()
    try_eigh(m, f"exact identity batch={n}")
    m = m + 1e-6 * torch.randn(n, 3, 3, device=device)
    m = (m + m.transpose(-1, -2)) / 2
    try_eigh(m, f"near identity batch={n}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
