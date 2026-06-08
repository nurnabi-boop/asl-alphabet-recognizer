"""PyTorch dataset for the landmarks CSV produced by extract_landmarks.py."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.extract_landmarks import FEATURE_COLS


@dataclass
class LabelEncoder:
    classes: list[str]

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def encode(self, labels: list[str]) -> np.ndarray:
        index = {c: i for i, c in enumerate(self.classes)}
        return np.array([index[l] for l in labels], dtype=np.int64)

    def decode(self, idx: int) -> str:
        return self.classes[idx]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"classes": self.classes}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LabelEncoder":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(classes=list(data["classes"]))


class LandmarksDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def load_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError(f"{csv_path} missing 'label' column")
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing feature columns: {missing[:5]}...")
    classes = sorted(df["label"].unique().tolist())
    encoder = LabelEncoder(classes=classes)
    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = encoder.encode(df["label"].tolist())
    return X, y, encoder


def make_loaders(
    csv_path: Path,
    batch_size: int = 64,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, LabelEncoder]:
    X, y, encoder = load_csv(csv_path)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    val_relative = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_relative, stratify=y_trainval, random_state=seed
    )

    train_ds = LandmarksDataset(X_train, y_train)
    val_ds = LandmarksDataset(X_val, y_val)
    test_ds = LandmarksDataset(X_test, y_test)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)
    return train_loader, val_loader, test_loader, encoder
