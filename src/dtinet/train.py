#!/usr/bin/env python3
"""Train a ManifoldNet classifier (LCM/LEM/AIM) on a DTI cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from dtinet.models import ManifoldNetClassifier

REQUIRED_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Override the config data root (default: ./data).",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate config, ledger, and one sample per split without training.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    required = {"experiment", "data", "model", "optimizer", "training"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")
    return config


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def crop_slices(mask: np.ndarray, target: tuple[int, int, int]) -> tuple[slice, ...]:
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        raise ValueError("Mask is empty")
    center = np.rint(coordinates.mean(axis=0)).astype(int)
    slices = []
    for size, wanted, midpoint in zip(mask.shape, target, center, strict=True):
        if size < wanted:
            raise ValueError(f"Cannot crop axis of length {size} to {wanted}")
        start = max(0, min(size - wanted, midpoint - wanted // 2))
        slices.append(slice(start, start + wanted))
    return tuple(slices)


class DTILedgerDataset(Dataset):
    def __init__(
        self,
        ledger: dict,
        split: str,
        data_root: Path,
        crop_xyz: tuple[int, int, int],
        minimum_mask_retention: float,
    ):
        self.records = [row for row in ledger["records"].values() if row["split"] == split]
        self.records.sort(key=lambda row: row["image_id"])
        self.split = split
        self.data_root = data_root.resolve()
        self.crop_xyz = crop_xyz
        self.minimum_mask_retention = minimum_mask_retention
        if not self.records:
            raise ValueError(f"Ledger split {split!r} is empty")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        tensor_image = nib.load(resolve_data_path(self.data_root, record["tensor_path"]))
        mask_image = nib.load(resolve_data_path(self.data_root, record["mask_path"]))
        if tensor_image.shape[:3] != mask_image.shape or not np.allclose(
            tensor_image.affine, mask_image.affine
        ):
            raise ValueError(f"Tensor/mask geometry mismatch for {record['image_id']}")
        tensor = np.asanyarray(tensor_image.dataobj)
        if tensor.ndim == 5 and tensor.shape[3] == 1:
            tensor = tensor[:, :, :, 0, :]
        if tensor.ndim != 4 or tensor.shape[-1] != 6:
            raise ValueError(
                f"Expected (X,Y,Z,1,6) or (X,Y,Z,6), got {tensor.shape} for {record['image_id']}"
            )
        mask = np.asanyarray(mask_image.dataobj) > 0
        slices = crop_slices(mask, self.crop_xyz)
        cropped_mask = mask[slices]
        retention = float(cropped_mask.sum() / mask.sum())
        if retention < self.minimum_mask_retention:
            raise ValueError(
                f"Mask retention {retention:.6f} below {self.minimum_mask_retention} "
                f"for {record['image_id']}"
            )
        tensor = np.asarray(tensor[slices + (slice(None),)], dtype=np.float32)
        if not np.isfinite(tensor).all():
            raise ValueError(f"Non-finite tensor values for {record['image_id']}")
        # (X,Y,Z,6) -> (C_in=1, coordinates=6, D=Z, H=X, W=Y)
        tensor = np.ascontiguousarray(tensor.transpose(3, 2, 0, 1)[None])
        spatial_mask = np.ascontiguousarray(cropped_mask.transpose(2, 0, 1)[None])
        return {
            "tensor": torch.from_numpy(tensor),
            "mask": torch.from_numpy(spatial_mask.astype(np.float32)),
            "label": torch.tensor(record["label"], dtype=torch.long),
            "image_id": record["image_id"],
            "subject_id": record["subject_id"],
            "group": record["group"],
        }


def resolve_data_path(data_root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError(f"Ledger paths must be relative to the data root, got: {path}")
    resolved = (data_root / path).resolve()
    try:
        resolved.relative_to(data_root.resolve())
    except ValueError as error:
        raise ValueError(f"Ledger path escapes the data root: {path}") from error
    return resolved


def validate_ledger(ledger: dict) -> None:
    required = {
        "schema_version",
        "task",
        "label_map",
        "ledger_sha256",
        "split_summary",
        "split_subjects",
        "records",
    }
    missing = sorted(required - ledger.keys())
    if missing:
        raise ValueError(f"Ledger is missing keys: {missing}")
    if ledger["label_map"] != {"Control": 0, "PD": 1}:
        raise ValueError(f"Unexpected label mapping: {ledger['label_map']}")
    claimed_hash = ledger["ledger_sha256"]
    unhashed = {key: value for key, value in ledger.items() if key != "ledger_sha256"}
    observed_hash = canonical_hash(unhashed)
    if observed_hash != claimed_hash:
        raise ValueError(
            "Ledger content does not match ledger_sha256: "
            f"expected {claimed_hash}, observed {observed_hash}"
        )
    records = ledger["records"]
    subjects = defaultdict(set)
    labels = defaultdict(set)
    for image_id, record in records.items():
        if image_id != record["image_id"]:
            raise ValueError(f"Ledger key does not match Image ID: {image_id}")
        if record["split"] not in REQUIRED_SPLITS:
            raise ValueError(f"Invalid split for {image_id}: {record['split']}")
        subjects[record["subject_id"]].add(record["split"])
        labels[record["subject_id"]].add(record["label"])
    crossing = [subject for subject, splits in subjects.items() if len(splits) != 1]
    inconsistent = [subject for subject, values in labels.items() if len(values) != 1]
    if crossing:
        raise ValueError(f"Subjects cross splits: {crossing[:10]}")
    if inconsistent:
        raise ValueError(f"Subjects have inconsistent labels: {inconsistent[:10]}")


def build_loader(
    dataset: Dataset,
    config: dict,
    shuffle: bool,
    generator: torch.Generator,
) -> DataLoader:
    workers = int(config["data"]["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=bool(config["data"]["pin_memory"]),
        persistent_workers=workers > 0,
        worker_init_fn=worker_seed,
        generator=generator if shuffle else None,
    )


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = (probabilities >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    both_classes = len(np.unique(labels)) == 2
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)) if both_classes else 0.5,
        "pr_auc": float(average_precision_score(labels, probabilities))
        if both_classes
        else float(labels.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "accuracy": float(np.mean(labels == predictions)),
        "n": int(len(labels)),
        "class_counts": {
            "Control": int(np.sum(labels == 0)),
            "PD": int(np.sum(labels == 1)),
        },
    }


def aggregate_subjects(scan_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in scan_rows:
        grouped[row["subject_id"]].append(row)
    subjects = []
    for subject_id, rows in sorted(grouped.items()):
        labels = {row["label"] for row in rows}
        if len(labels) != 1:
            raise ValueError(f"Inconsistent labels for subject {subject_id}")
        probability = float(np.mean([row["probability_pd"] for row in rows]))
        label = labels.pop()
        subjects.append(
            {
                "subject_id": subject_id,
                "label": label,
                "group": "PD" if label else "Control",
                "probability_pd": probability,
                "prediction": int(probability >= 0.5),
                "n_scans": len(rows),
            }
        )
    return subjects


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
    amp: bool,
) -> tuple[dict, list[dict], list[dict]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    scan_rows = []
    for batch in loader:
        tensor = batch["tensor"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            logits = model(tensor, mask)
            loss = criterion(logits, labels)
        probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        label_values = labels.cpu().numpy()
        total_loss += float(loss.item()) * len(label_values)
        total_samples += len(label_values)
        for index, probability in enumerate(probabilities):
            scan_rows.append(
                {
                    "image_id": batch["image_id"][index],
                    "subject_id": batch["subject_id"][index],
                    "label": int(label_values[index]),
                    "group": batch["group"][index],
                    "probability_pd": float(probability),
                    "prediction": int(probability >= 0.5),
                }
            )
    scan_labels = np.array([row["label"] for row in scan_rows])
    scan_probabilities = np.array([row["probability_pd"] for row in scan_rows])
    subject_rows = aggregate_subjects(scan_rows)
    subject_labels = np.array([row["label"] for row in subject_rows])
    subject_probabilities = np.array([row["probability_pd"] for row in subject_rows])
    metrics = {
        "loss": total_loss / total_samples,
        "scan": classification_metrics(scan_labels, scan_probabilities),
        "subject": classification_metrics(subject_labels, subject_probabilities),
    }
    return metrics, scan_rows, subject_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def checkpoint_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_metric: float,
    bad_epochs: int,
    ledger_sha256: str,
    config_sha256: str,
    train_generator: torch.Generator,
) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "bad_epochs": bad_epochs,
        "ledger_sha256": ledger_sha256,
        "config_sha256": config_sha256,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "train_generator_state": train_generator.get_state(),
    }


def restore_checkpoint(
    checkpoint: dict,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    ledger_sha256: str,
    config_sha256: str,
    train_generator: torch.Generator,
) -> tuple[int, float, int]:
    if checkpoint["ledger_sha256"] != ledger_sha256:
        raise ValueError("Resume checkpoint ledger does not match the configured ledger")
    if checkpoint["config_sha256"] != config_sha256:
        raise ValueError("Resume checkpoint configuration does not match")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    random.setstate(checkpoint["python_random_state"])
    np.random.set_state(checkpoint["numpy_random_state"])
    torch.set_rng_state(checkpoint["torch_random_state"])
    if torch.cuda.is_available() and checkpoint["cuda_random_state"] is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])
    train_generator.set_state(checkpoint["train_generator_state"])
    return checkpoint["epoch"] + 1, checkpoint["best_metric"], checkpoint["bad_epochs"]


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    config_sha256 = canonical_hash(config)
    seed = int(config["experiment"]["seed"])
    deterministic = bool(config["experiment"]["deterministic"])
    seed_everything(seed, deterministic)

    configured_data_root = Path(config["data"].get("root", "data")).expanduser()
    data_root = (args.data_root or configured_data_root).resolve()
    ledger_path = resolve_data_path(data_root, config["data"]["ledger"])
    ledger = json.loads(ledger_path.read_text())
    validate_ledger(ledger)
    crop_xyz = tuple(int(value) for value in config["data"]["crop_xyz"])
    minimum_retention = float(config["data"]["minimum_mask_retention"])
    datasets = {
        split: DTILedgerDataset(ledger, split, data_root, crop_xyz, minimum_retention)
        for split in REQUIRED_SPLITS
    }
    print(f"Data root: {data_root}", flush=True)
    print(
        "Dataset: " + ", ".join(f"{split}={len(dataset)}" for split, dataset in datasets.items()),
        flush=True,
    )
    if args.validate_only:
        for split, dataset in datasets.items():
            sample = dataset[0]
            print(
                f"{split}: tensor={tuple(sample['tensor'].shape)} "
                f"mask={tuple(sample['mask'].shape)} label={sample['label'].item()} "
                f"image={sample['image_id']}",
                flush=True,
            )
        print("Configuration, ledger, and sample validation passed.", flush=True)
        return 0

    if config["experiment"]["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["experiment"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    amp = bool(config["training"]["amp"]) and device.type == "cuda"
    print(f"Device: {device}; AMP: {amp}", flush=True)

    output_dir = Path(config["experiment"]["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "last.pt").exists() and args.resume is None:
        raise FileExistsError(
            f"{output_dir}/last.pt exists; pass --resume or choose another output directory"
        )
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output_dir / "ledger_reference.json").write_text(
        json.dumps({"path": str(ledger_path), "sha256": ledger["ledger_sha256"]}, indent=2) + "\n"
    )

    train_generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": build_loader(datasets["train"], config, True, train_generator),
        "train_eval": build_loader(datasets["train"], config, False, train_generator),
        "val": build_loader(datasets["val"], config, False, train_generator),
        "test": build_loader(datasets["test"], config, False, train_generator),
    }
    model = ManifoldNetClassifier(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    epochs = int(config["training"]["epochs"])
    warmup = int(config["training"]["warmup_epochs"])

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(epochs - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    train_labels = np.array([row["label"] for row in datasets["train"].records])
    counts = np.bincount(train_labels, minlength=2)
    class_weights = len(train_labels) / (2 * counts)
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    print(
        f"Training class weights: Control={class_weights[0]:.6f}, PD={class_weights[1]:.6f}",
        flush=True,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    start_epoch, best_metric, bad_epochs = 0, -math.inf, 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        start_epoch, best_metric, bad_epochs = restore_checkpoint(
            checkpoint,
            model,
            optimizer,
            scheduler,
            scaler,
            ledger["ledger_sha256"],
            config_sha256,
            train_generator,
        )
        print(f"Resumed at epoch {start_epoch + 1}", flush=True)

    history_path = output_dir / "history.jsonl"
    if start_epoch == 0:
        history_path.write_text("")
    patience = int(config["training"]["early_stopping_patience"])
    eval_train_every = int(config["training"]["eval_train_every"])
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for batch in loaders["train"]:
            tensor = batch["tensor"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                loss = criterion(model(tensor, mask), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * len(labels)
            total_samples += len(labels)
        train_loss = total_loss / total_samples
        val_metrics, _, _ = evaluate(model, loaders["val"], device, criterion, amp)
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "val": val_metrics,
        }
        if epoch == 0 or (epoch + 1) % eval_train_every == 0:
            train_metrics, _, _ = evaluate(model, loaders["train_eval"], device, criterion, amp)
            row["train"] = train_metrics
        selection = val_metrics["scan"]["balanced_accuracy"]
        improved = selection > best_metric
        if improved:
            best_metric = selection
            bad_epochs = 0
        else:
            bad_epochs += 1
        state = checkpoint_state(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_metric,
            bad_epochs,
            ledger["ledger_sha256"],
            config_sha256,
            train_generator,
        )
        atomic_torch_save(state, output_dir / "last.pt")
        if improved:
            atomic_torch_save(state, output_dir / "best.pt")
        with history_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(
            f"epoch={epoch + 1:03d} train_loss={train_loss:.5f} "
            f"val_loss={val_metrics['loss']:.5f} "
            f"val_auc={val_metrics['scan']['roc_auc']:.4f} "
            f"val_bal_acc={selection:.4f} best={best_metric:.4f} "
            f"bad_epochs={bad_epochs} lr={optimizer.param_groups[0]['lr']:.3e}",
            flush=True,
        )
        if bad_epochs >= patience:
            print(f"Early stopping after epoch {epoch + 1}", flush=True)
            break

    best = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_metrics, scan_rows, subject_rows = evaluate(model, loaders["test"], device, criterion, amp)
    final = {
        "best_epoch": int(best["epoch"] + 1),
        "selection_metric": "validation scan balanced_accuracy",
        "best_validation_balanced_accuracy": float(best["best_metric"]),
        "test": test_metrics,
        "ledger_sha256": ledger["ledger_sha256"],
        "config_sha256": config_sha256,
    }
    (output_dir / "final_metrics.json").write_text(json.dumps(final, indent=2) + "\n")
    write_csv(output_dir / "test_scan_predictions.csv", scan_rows)
    write_csv(output_dir / "test_subject_predictions.csv", subject_rows)
    print(json.dumps(final, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
