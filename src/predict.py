"""Inference utilities for the ASL landmark MLP.

Provides:
 - ASLPredictor: holds the MediaPipe landmarker + the MLP, exposes predict_image() / predict_landmarks().
 - draw_overlay(): annotate a BGR frame with the prediction + 21 landmark dots.
 - CLI: `python -m src.predict path/to/image.jpg` for a single-image smoke test.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.extract_landmarks import (
    DEFAULT_MODEL,
    NUM_LANDMARKS,
    _to_mp_image,
    normalize_landmarks,
    open_landmarker,
)
from src.model import load_checkpoint

DEFAULT_CKPT = Path("models/asl_mlp.pt")


@dataclass
class Prediction:
    label: str
    confidence: float
    top3: list[tuple[str, float]]
    landmarks_xy: np.ndarray | None  # (21, 2) pixel coords, or None if no hand


class ASLPredictor:
    def __init__(
        self,
        ckpt_path: Path = DEFAULT_CKPT,
        landmarker_model: Path = DEFAULT_MODEL,
        running_mode: str = "image",
        device: str | torch.device = "cpu",
    ):
        self.device = torch.device(device)
        self.model, self.classes = load_checkpoint(ckpt_path, map_location=self.device)
        self.model.to(self.device).eval()
        self._stack = ExitStack()
        self.landmarker = self._stack.enter_context(
            open_landmarker(landmarker_model, running_mode=running_mode)
        )
        self.running_mode = running_mode

    def close(self) -> None:
        self._stack.close()

    def __enter__(self) -> "ASLPredictor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _detect(self, image_bgr: np.ndarray, timestamp_ms: int | None):
        mp_img = _to_mp_image(image_bgr)
        if self.running_mode == "image":
            return self.landmarker.detect(mp_img)
        if self.running_mode == "video":
            assert timestamp_ms is not None, "video mode requires timestamp_ms"
            return self.landmarker.detect_for_video(mp_img, timestamp_ms)
        raise RuntimeError(f"unsupported mode: {self.running_mode}")

    def predict_image(self, image_bgr: np.ndarray, timestamp_ms: int | None = None) -> Prediction | None:
        result = self._detect(image_bgr, timestamp_ms)
        if not result.hand_landmarks:
            return None
        lm = result.hand_landmarks[0]
        pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        vec = normalize_landmarks(pts)

        h, w = image_bgr.shape[:2]
        landmarks_xy = np.column_stack([pts[:, 0] * w, pts[:, 1] * h]).astype(np.int32)
        return self.predict_landmarks(vec, landmarks_xy=landmarks_xy)

    def predict_landmarks(self, vec: np.ndarray, landmarks_xy: np.ndarray | None = None) -> Prediction:
        with torch.no_grad():
            x = torch.from_numpy(vec.reshape(1, -1).astype(np.float32)).to(self.device)
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        order = np.argsort(probs)[::-1]
        top3 = [(self.classes[int(i)], float(probs[int(i)])) for i in order[:3]]
        best_idx = int(order[0])
        return Prediction(
            label=self.classes[best_idx],
            confidence=float(probs[best_idx]),
            top3=top3,
            landmarks_xy=landmarks_xy,
        )


def draw_overlay(frame_bgr: np.ndarray, pred: Prediction | None) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    if pred is None:
        cv2.putText(out, "no hand detected", (16, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2, cv2.LINE_AA)
        return out

    if pred.landmarks_xy is not None:
        for x, y in pred.landmarks_xy:
            cv2.circle(out, (int(x), int(y)), 3, (0, 255, 0), -1)

    txt = f"{pred.label}  {pred.confidence*100:.1f}%"
    cv2.rectangle(out, (8, 8), (8 + 14 * len(txt), 48), (0, 0, 0), -1)
    cv2.putText(out, txt, (16, 38), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 2, cv2.LINE_AA)

    y = h - 16 - 24 * (len(pred.top3) - 1)
    for label, p in pred.top3:
        cv2.putText(out, f"{label}: {p*100:5.1f}%", (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y += 24
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ASL prediction on a single image.")
    parser.add_argument("image", type=Path, help="Path to an image to classify.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--landmarker-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--save-overlay", type=Path, default=None,
                        help="Optional path to save the annotated image.")
    args = parser.parse_args()

    img = cv2.imread(str(args.image))
    if img is None:
        print(f"could not read image: {args.image}")
        return 1

    with ASLPredictor(ckpt_path=args.checkpoint, landmarker_model=args.landmarker_model) as predictor:
        pred = predictor.predict_image(img)

    if pred is None:
        print(f"{args.image}: no hand detected")
        return 2

    print(f"{args.image} -> {pred.label}  conf={pred.confidence:.4f}")
    print("  top3:")
    for label, p in pred.top3:
        print(f"    {label}: {p:.4f}")

    if args.save_overlay is not None:
        annotated = draw_overlay(img, pred)
        args.save_overlay.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_overlay), annotated)
        print(f"  overlay saved to {args.save_overlay}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
