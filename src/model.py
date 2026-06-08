"""Small MLP classifier over 63-dim normalized landmark vectors."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass
class MLPConfig:
    in_features: int = 63
    hidden1: int = 256
    hidden2: int = 128
    num_classes: int = 26
    dropout: float = 0.3


class LandmarkMLP(nn.Module):
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.in_features, cfg.hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden1, cfg.hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden2, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def save_checkpoint(model: LandmarkMLP, classes: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(model.cfg),
            "state_dict": model.state_dict(),
            "classes": classes,
        },
        path,
    )


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> tuple[LandmarkMLP, list[str]]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = MLPConfig(**ckpt["config"])
    model = LandmarkMLP(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, list(ckpt["classes"])
