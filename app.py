"""Gradio app: Live Webcam + Upload Image tabs for the ASL recognizer.

Run with:  python app.py

Notes on the live mode:
- We instantiate one ASLPredictor per running mode so MediaPipe gets the right
  RunningMode. The image-tab predictor uses IMAGE mode; the webcam-tab predictor
  uses VIDEO mode (which MediaPipe optimizes for sequential frames).
- Gradio's `streaming=True` for an Image input gives us per-frame callbacks; the
  matplotlib bar chart is regenerated on each frame.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.predict import ASLPredictor, draw_overlay, Prediction

CKPT = Path("models/asl_mlp.pt")
LANDMARKER_MODEL = Path("models/hand_landmarker.task")

if not CKPT.exists():
    raise SystemExit(
        f"checkpoint not found at {CKPT}. Train one with `python -m src.train` first."
    )

_image_predictor = ASLPredictor(CKPT, LANDMARKER_MODEL, running_mode="image")
_video_predictor = ASLPredictor(CKPT, LANDMARKER_MODEL, running_mode="video")
_video_t0 = time.time()


def _bar_chart(pred: Prediction | None):
    fig, ax = plt.subplots(figsize=(4, 2.4), dpi=110)
    if pred is None:
        ax.text(0.5, 0.5, "no hand", ha="center", va="center", fontsize=14, color="#999")
        ax.set_axis_off()
    else:
        labels = [t[0] for t in pred.top3]
        values = [t[1] * 100 for t in pred.top3]
        bars = ax.barh(labels[::-1], values[::-1], color="#3b82f6")
        ax.set_xlim(0, 100)
        ax.set_xlabel("confidence (%)")
        ax.set_title("top-3")
        for b, v in zip(bars, values[::-1]):
            ax.text(min(v + 1.5, 95), b.get_y() + b.get_height() / 2,
                    f"{v:.1f}%", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def predict_upload(rgb_image: np.ndarray | None):
    if rgb_image is None:
        return None, _bar_chart(None), "Upload an image."
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    pred = _image_predictor.predict_image(bgr)
    overlay_bgr = draw_overlay(bgr, pred)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    if pred is None:
        text = "No hand detected."
    else:
        text = f"Prediction: **{pred.label}**  ({pred.confidence*100:.1f}%)"
    return overlay_rgb, _bar_chart(pred), text


def predict_webcam(rgb_frame: np.ndarray | None):
    if rgb_frame is None:
        return None, _bar_chart(None), "Webcam off."
    bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    ts_ms = int((time.time() - _video_t0) * 1000)
    pred = _video_predictor.predict_image(bgr, timestamp_ms=ts_ms)
    overlay_bgr = draw_overlay(bgr, pred)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    text = "No hand detected." if pred is None else (
        f"Prediction: **{pred.label}**  ({pred.confidence*100:.1f}%)"
    )
    return overlay_rgb, _bar_chart(pred), text


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="ASL Alphabet Recognizer") as demo:
        gr.Markdown("# ASL Alphabet Recognizer\nMediaPipe Hands + small MLP, trained on 21 normalized landmarks.")

        with gr.Tabs():
            with gr.Tab("Live Webcam"):
                with gr.Row():
                    with gr.Column(scale=2):
                        cam_in = gr.Image(sources=["webcam"], streaming=True,
                                          label="webcam", type="numpy", height=380)
                    with gr.Column(scale=2):
                        cam_overlay = gr.Image(label="overlay", type="numpy", height=380)
                        cam_chart = gr.Plot(label="top-3")
                        cam_text = gr.Markdown("Show your hand to the camera.")
                cam_in.stream(
                    predict_webcam,
                    inputs=cam_in,
                    outputs=[cam_overlay, cam_chart, cam_text],
                    stream_every=0.15,
                    show_progress="hidden",
                )

            with gr.Tab("Upload Image"):
                with gr.Row():
                    with gr.Column():
                        up_in = gr.Image(sources=["upload"], type="numpy", label="image", height=380)
                        up_btn = gr.Button("Predict", variant="primary")
                    with gr.Column():
                        up_overlay = gr.Image(label="overlay", type="numpy", height=380)
                        up_chart = gr.Plot(label="top-3")
                        up_text = gr.Markdown("")
                up_btn.click(predict_upload, inputs=up_in,
                             outputs=[up_overlay, up_chart, up_text])
                up_in.change(predict_upload, inputs=up_in,
                             outputs=[up_overlay, up_chart, up_text])

    return demo


if __name__ == "__main__":
    build_ui().launch()
