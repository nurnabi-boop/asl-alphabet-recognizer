"""Train the landmark MLP classifier and checkpoint the best model."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import make_loaders
from src.model import LandmarkMLP, MLPConfig, save_checkpoint


def evaluate(model: LandmarkMLP, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += criterion(logits, y).item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def train(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_loader, val_loader, test_loader, encoder = make_loaders(
        csv_path=args.csv,
        batch_size=args.batch_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    print(f"classes ({encoder.num_classes}): {encoder.classes}")
    print(f"train/val/test sizes: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")

    cfg = MLPConfig(
        in_features=63,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        num_classes=encoder.num_classes,
        dropout=args.dropout,
    )
    model = LandmarkMLP(cfg).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    epochs_no_improve = 0
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    encoder.save(args.checkpoint.with_suffix(".labels.json"))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()
            running_loss += loss.item() * y.size(0)
            running_correct += (logits.argmax(dim=1) == y).sum().item()
            running_total += y.size(0)
        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        val_loss, val_acc = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        print(
            f"epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  ({elapsed:.1f}s)"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            save_checkpoint(model, encoder.classes, args.checkpoint)
            print(f"  -> new best val_acc={val_acc:.4f}, saved {args.checkpoint}")
        else:
            epochs_no_improve += 1
            if args.patience and epochs_no_improve >= args.patience:
                print(f"early stop after {args.patience} epochs without improvement")
                break

    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"final test: loss={test_loss:.4f} acc={test_acc:.4f}  best_val_acc={best_val_acc:.4f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the landmark MLP.")
    parser.add_argument("--csv", type=Path, default=Path("data/processed/landmarks.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/asl_mlp.pt"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hidden1", type=int, default=256)
    parser.add_argument("--hidden2", type=int, default=128)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=15, help="Early stop patience (0 to disable).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
