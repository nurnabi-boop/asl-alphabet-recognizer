"""Extract MediaPipe Hand landmarks from images and save as a CSV dataset.

Folder layout expected (Kaggle "ASL Alphabet" style):
    data/raw/
        A/   <png/jpg images>
        B/
        ...
        Z/
        del/  space/  nothing/   (optional)

Output: data/processed/landmarks.csv with columns
    label, x0, y0, z0, x1, y1, z1, ..., x20, y20, z20

Uses the new MediaPipe Tasks API (mp.tasks.vision.HandLandmarker), which
requires a downloaded `hand_landmarker.task` model file. By default we look at
`models/hand_landmarker.task`.
"""
from __future__ import annotations

import argparse
import csv
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
NUM_LANDMARKS = 21
FEATURE_COLS = [f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")]
DEFAULT_MODEL = Path("models/hand_landmarker.task")


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Translate to wrist-relative coords and scale by max distance.

    Returns a (63,) vector invariant to position and overall hand size.
    """
    wrist = landmarks[0]
    centered = landmarks - wrist
    scale = np.linalg.norm(centered, axis=1).max()
    if scale > 0:
        centered = centered / scale
    return centered.flatten().astype(np.float32)


@contextmanager
def open_landmarker(model_path: Path, running_mode: str = "image"):
    """Yield a HandLandmarker configured for the given mode ('image' or 'video')."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Hand landmarker model not found at {model_path}.\n"
            "Download it from "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        )
    mode_map = {
        "image": mp_vision.RunningMode.IMAGE,
        "video": mp_vision.RunningMode.VIDEO,
        "live_stream": mp_vision.RunningMode.LIVE_STREAM,
    }
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mode_map[running_mode],
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)
    try:
        yield landmarker
    finally:
        landmarker.close()


def _to_mp_image(image_bgr: np.ndarray) -> mp.Image:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)


def extract_from_image(image_bgr: np.ndarray, landmarker) -> np.ndarray | None:
    """Run MediaPipe on a single BGR image, return normalized 63-dim vector or None."""
    result = landmarker.detect(_to_mp_image(image_bgr))
    if not result.hand_landmarks:
        return None
    lm = result.hand_landmarks[0]
    pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
    return normalize_landmarks(pts)


def iter_image_paths(root: Path) -> Iterable[tuple[str, Path]]:
    for label_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        label = label_dir.name
        for img_path in sorted(label_dir.iterdir()):
            if img_path.suffix.lower() in IMG_EXTS:
                yield label, img_path


def build_dataset(
    raw_dir: Path,
    out_csv: Path,
    model_path: Path,
    limit_per_class: int | None = None,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    skipped = 0
    written = 0
    per_class_count: dict[str, int] = {}

    with open_landmarker(model_path, "image") as landmarker, out_csv.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["label", *FEATURE_COLS])

        for label, img_path in tqdm(list(iter_image_paths(raw_dir)), desc="extracting"):
            if limit_per_class is not None and per_class_count.get(label, 0) >= limit_per_class:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue
            vec = extract_from_image(img, landmarker)
            if vec is None:
                skipped += 1
                continue
            writer.writerow([label, *vec.tolist()])
            written += 1
            per_class_count[label] = per_class_count.get(label, 0) + 1

    print(f"wrote {written} rows -> {out_csv} (skipped {skipped})")


def smoke_test(test_dir: Path, model_path: Path) -> int:
    images = [p for p in test_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    if not images:
        print(f"no images found under {test_dir}", file=sys.stderr)
        return 1
    print(f"running smoke test on {len(images)} images...")
    ok = 0
    with open_landmarker(model_path, "image") as landmarker:
        for p in images:
            img = cv2.imread(str(p))
            if img is None:
                print(f"  [READ FAIL] {p}")
                continue
            vec = extract_from_image(img, landmarker)
            if vec is None:
                print(f"  [NO HAND ] {p}  shape={img.shape}")
            else:
                print(
                    f"  [OK      ] {p}  vec[:3]={vec[:3].round(3).tolist()}  "
                    f"norm={np.linalg.norm(vec):.3f}"
                )
                ok += 1
    print(f"smoke test: {ok}/{len(images)} hands detected")
    return 0 if ok > 0 else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract MediaPipe hand landmarks.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-csv", type=Path, default=Path("data/processed/landmarks.csv"))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--smoke-test", type=Path, help="Run a smoke test on a folder of images and exit.")
    args = parser.parse_args()

    if args.smoke_test is not None:
        return smoke_test(args.smoke_test, args.model)

    if not args.raw_dir.exists():
        print(f"raw dir not found: {args.raw_dir}", file=sys.stderr)
        return 1
    build_dataset(args.raw_dir, args.out_csv, args.model, args.limit_per_class)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
