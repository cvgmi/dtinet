"""
Train a ManifoldNet classifier on a synthetic two-class SPD(3) dataset.

Each sample is a small 3D field in which every voxel is an i.i.d. draw from a
class-conditional distribution on SPD(3): isotropic tensors (class 0) versus anisotropic
tensors with a random per-voxel orientation (class 1). The network must aggregate spatial
statistics to classify the field.

The script doubles as a smoke test: it exits with code 0 only if the final validation
accuracy reaches 0.9.
"""

import argparse
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dtinet.data import SyntheticSPDFieldDataset
from dtinet.models import ManifoldNetClassifier

TARGET_ACCURACY = 0.9


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", choices=["lcm", "lem", "aim"], default="lcm")
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--channels", type=str, default="4,8")
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--padding", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Max gradient norm (0 disables clipping). Recommended for metric=aim.",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Do not pass a foreground mask to the model.",
    )
    return parser.parse_args(argv)


def unpack(
    batch: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """
    Unpack a batch of (x, y) or (x, mask, y) tuples.

    Returns
    -------
    tuple of torch.Tensor
        ``(x, mask, y)`` with ``mask=None`` when the dataset does not return masks.

    """
    if len(batch) == 3:
        return batch
    x, y = batch
    return x, None, y


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    use_mask: bool,
    device: torch.device,
) -> tuple[float, float]:
    """Compute mean loss and accuracy over a dataloader."""
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            x, mask, y = unpack(batch)
            x, y = x.to(device), y.to(device)
            mask = mask.to(device) if mask is not None else None
            logits = model(x, mask if use_mask else None)
            total_loss += criterion(logits, y).item() * len(y)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total += len(y)
    return total_loss / total, total_correct / total


def main(argv: list[str] | None = None) -> int:
    """Train and evaluate; return 0 on success (validation accuracy >= target)."""
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    channels = [int(c) for c in args.channels.split(",")]
    activation = "relu" if args.metric in ("lcm", "lem") else None
    use_mask = not args.no_mask

    dataset = SyntheticSPDFieldDataset(
        num_samples=args.num_samples,
        grid_size=args.grid_size,
        seed=args.seed,
        return_mask=use_mask,
    )
    num_train = int(0.8 * len(dataset))
    split_gen = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [num_train, len(dataset) - num_train], split_gen)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=0)

    model = ManifoldNetClassifier(
        num_classes=2,
        num_layers=len(channels),
        num_channels=channels,
        metric=args.metric,
        activation=activation,
        kernel_size=args.kernel_size,
        padding=args.padding,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch in train_loader:
            x, mask, y = unpack(batch)
            x, y = x.to(device), y.to(device)
            mask = mask.to(device) if mask is not None else None
            optimizer.zero_grad()
            logits = model(x, mask if use_mask else None)
            loss = criterion(logits, y)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += loss.item() * len(y)
            train_correct += (logits.argmax(dim=1) == y).sum().item()
            train_total += len(y)
        val_loss, val_acc = evaluate(model, val_loader, criterion, use_mask, device)
        print(
            f"epoch {epoch:02d}/{args.epochs} "
            f"train loss {train_loss / train_total:.4f} acc {train_correct / train_total:.4f} "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

    if val_acc < TARGET_ACCURACY:
        print(
            f"FAIL: final val accuracy {val_acc:.4f} < {TARGET_ACCURACY}. "
            "The example doubles as a smoke test; this run did not pass.",
            file=sys.stderr,
        )
        return 1
    print(f"PASS: final val accuracy {val_acc:.4f} >= {TARGET_ACCURACY}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
