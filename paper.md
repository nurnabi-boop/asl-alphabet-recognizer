# Real-Time ASL Alphabet Recognition via MediaPipe Hand Landmarks and a Compact Multilayer Perceptron

**Anonymous Author**

*Working draft — system description and methodology paper*

---

## Abstract

We present a lightweight, real-time American Sign Language (ASL) alphabet recognition system that decouples hand localization from gesture classification. Rather than learning visual features end-to-end from raw pixels, we delegate hand detection and keypoint regression to MediaPipe Hands and train only a small multilayer perceptron (MLP) on the resulting 63-dimensional landmark vectors. A simple wrist-relative, scale-normalized representation makes the classifier invariant to hand position and apparent size in the frame. The system streams predictions from a webcam at interactive rates on CPU-only hardware, exposes a single-image inference CLI, and ships a Gradio demo UI with live webcam and image-upload tabs. We describe the data pipeline, normalization scheme, training procedure, and an end-to-end smoke test on a four-class synthetic dataset that confirms correctness of the pipeline. Reproducible code is provided.

**Keywords:** sign language recognition, hand pose estimation, MediaPipe, MLP, real-time inference.

---

## 1. Introduction

Automatic recognition of fingerspelled sign-language letters from a webcam is a well-studied problem with a useful property: most of the visual ambiguity has already been solved by general-purpose hand pose estimators. Models such as MediaPipe Hands [1] regress 21 anatomical keypoints per hand from a single RGB image with low latency on commodity hardware. Given those keypoints, the remaining task — mapping a hand pose to one of 26 letters — is a low-dimensional classification problem that does not require a deep convolutional network.

This paper describes a system built on that observation. The contributions are:

1. A practical pipeline that uses the modern MediaPipe Tasks API (`mp.tasks.vision.HandLandmarker`) for both batch landmark extraction and live video inference.
2. A wrist-relative, scale-normalized 63-dimensional feature that is robust to in-frame translation and apparent hand size.
3. A two-hidden-layer MLP (~50K parameters) that runs in microseconds on CPU and is trivially deployable.
4. A reference implementation with a CLI, a Gradio UI, and an end-to-end smoke test on a synthetic four-class dataset that validates the pipeline.

---

## 2. Related Work

End-to-end CNN approaches to ASL alphabet recognition are common in the Kaggle ecosystem [2], typically training MobileNet- or EfficientNet-class models directly on the labeled ASL Alphabet dataset. These models conflate two problems: localizing the hand in the frame, and identifying the gesture once localized.

The alternative — explicit two-stage pipelines that separate detection/keypoint regression from classification — has a long history, from data-glove-based approaches through the modern landmark-based work enabled by MediaPipe [1] and OpenPose [3]. Landmark-based classifiers have been shown to generalize better across backgrounds and lighting than pixel-input CNNs of comparable size, because the upstream detector absorbs that variability.

Our work is a small, faithful instance of this pattern, focused on real-time deployment.

---

## 3. Methodology

### 3.1 System Overview

```
camera frame  →  MediaPipe HandLandmarker  →  21 (x,y,z) keypoints
                                                       ↓
                                               normalize_landmarks()
                                                       ↓
                                               63-dim feature vector
                                                       ↓
                                                  MLP (256→128→K)
                                                       ↓
                                                 letter + softmax
```

The full pipeline is implemented in five Python modules: `extract_landmarks.py`, `dataset.py`, `model.py`, `train.py`, `predict.py`, plus an `app.py` for the Gradio UI.

### 3.2 Landmark Extraction

For each input image we run the MediaPipe HandLandmarker (float16 model variant) in single-hand mode (`num_hands=1`) with `min_hand_detection_confidence=0.5`. The output is a list of 21 normalized landmarks, each carrying `(x, y, z)` where `x, y ∈ [0, 1]` are image-relative and `z` is depth in the same scale as `x`.

The IMAGE running mode is used for offline dataset construction and for the upload-image UI tab; the VIDEO running mode (which enables MediaPipe's internal temporal smoothing) is used for the live webcam tab.

### 3.3 Feature Normalization

Let `L ∈ R^{21×3}` be the raw landmark matrix. We translate by the wrist landmark `L_0` and scale by the maximum distance from the wrist:

```
L'_i = (L_i − L_0) / max_j ‖L_j − L_0‖_2
```

The flattened 63-dimensional vector `vec(L')` is the classifier input. Two desirable invariances follow directly:

- **Translation invariance.** Any whole-hand shift in the frame is absorbed by the wrist subtraction.
- **Scale invariance.** Any apparent-size change (distance from camera, image resolution) is absorbed by the per-sample max-distance normalization.

Rotation invariance is *not* enforced; ASL handshapes do encode orientation (e.g., 'P' vs 'K'), so we want the classifier to see it.

### 3.4 Classifier

The classifier is a two-hidden-layer MLP:

| Layer  | Out features | Activation | Dropout |
|--------|--------------|------------|---------|
| Linear | 256          | ReLU       | 0.3     |
| Linear | 128          | ReLU       | 0.3     |
| Linear | K            | —          | —       |

where K is the number of classes (26 for the standard ASL alphabet, or 29 if the Kaggle dataset's `space`, `del`, and `nothing` classes are kept). Parameter count is approximately 51K when K=26.

### 3.5 Training Procedure

We use cross-entropy loss and the Adam optimizer (lr = 1e-3, weight decay = 1e-4, batch size = 64). Data is split into train/val/test (70/15/15) with stratification by label. The validation set is used for early stopping (patience = 15) and best-checkpoint selection. The full training script with all flags is in `src/train.py`.

### 3.6 Real-Time Inference

The Gradio app holds two persistent `ASLPredictor` instances, one per MediaPipe RunningMode (IMAGE for uploads, VIDEO for webcam). For the webcam tab, frames arrive at 5–10 Hz via Gradio's `streaming=True`; for each frame we (a) compute landmarks, (b) normalize, (c) forward through the MLP, (d) overlay the predicted letter, the 21 keypoint dots, and a top-3 bar chart on the output image. End-to-end per-frame latency is dominated by the MediaPipe inference call.

---

## 4. Experiments

### 4.1 Datasets

The intended training corpus is the Kaggle **ASL Alphabet** dataset [2], which contains approximately 87,000 256×256 RGB images across 29 classes (A–Z plus `space`, `del`, `nothing`). The data pipeline in `src/extract_landmarks.py` consumes any directory with one subfolder per class.

For pipeline validation we also constructed a **synthetic four-class dataset** by extracting landmarks from four MediaPipe-provided reference hand images (`pointing_up`, `thumbs_down`, `victory`, `woman_hands`), labeling them `A`/`B`/`C`/`D`, and generating 50 noisy copies of each (Gaussian jitter σ = 0.02 in the normalized landmark space). This yields 200 rows, intentionally easy to separate, used solely to verify correctness of the dataset/model/train/predict modules end-to-end.

### 4.2 Implementation

All experiments run on Python 3.13 with MediaPipe 0.10.35, PyTorch 2.11, OpenCV 4.13, and Gradio 6.14, on CPU. The `hand_landmarker.task` float16 model (~7.8 MB) is downloaded once from the official MediaPipe model bucket.

### 4.3 Pipeline Validation (Synthetic Data)

On the four-class synthetic dataset, the model converges to 100% test accuracy in fewer than 10 epochs (Table 1). This is the expected result — the synthetic classes are trivially separable — and its purpose is to confirm that data loading, label encoding, the training loop, and checkpoint serialization all behave correctly.

**Table 1.** Training on the four-class synthetic dataset (200 rows, 70/15/15 split).

| Epoch | Train loss | Train acc | Val loss | Val acc |
|------:|-----------:|----------:|---------:|--------:|
| 1     | 1.253      | 0.734     | 1.029    | 1.000   |
| 5     | 0.065      | 1.000     | 0.023    | 1.000   |
| 10    | 0.003      | 1.000     | 0.001    | 1.000   |
| 15    | 0.001      | 1.000     | 0.000    | 1.000   |

Final test accuracy: 1.000. Best val accuracy: 1.000.

We then ran single-image inference (`src.predict`) on each of the four reference images plus a fifth distractor image with no detectable hand. The four hand images each recovered their assigned label; the no-hand image produced the expected "no hand detected" output. An example overlay is rendered correctly with 21 keypoint dots, a label/confidence badge, and a top-3 readout.

### 4.4 Results on Real ASL Data

*(Pending.)* Training on the full Kaggle ASL Alphabet dataset is left as the next step; the data pipeline is already in place. We expect landmark-based MLP results in the high 90s on the standard 70/15/15 split based on comparable published baselines [4], with the caveats discussed below.

---

## 5. Discussion

**Why this works at all.** MediaPipe Hands is trained on a large and diverse human-hand corpus, so by the time the MLP sees an input the hand has already been localized and reduced to a canonical 21-point skeleton. The classifier therefore does not need to learn appearance robustness — only geometric discrimination between 26 hand shapes.

**Why MLP, not CNN.** With landmarks as input, spatial convolution buys nothing: the input is already a sparse, ordered 63-dim vector. A small MLP is the right inductive bias.

**Why wrist-relative scaling.** Absolute MediaPipe coordinates depend on image aspect ratio and on how large the hand appears in the frame. Without normalization, the classifier would have to learn that "the same hand shape closer to the camera" is the same class, wasting capacity on irrelevant variation.

---

## 6. Limitations

1. **Single-hand only.** Letters like 'J' and 'Z' involve motion, and ASL more broadly involves two hands. This system handles only the static, single-hand subset of the alphabet.
2. **Upstream failure modes.** If MediaPipe fails to detect a hand (poor lighting, extreme occlusion, motion blur), the classifier never runs. Our pipeline returns a `None` prediction in that case, which the UI renders as "no hand detected."
3. **Orientation sensitivity.** Because we deliberately do *not* rotation-normalize, the classifier may be sensitive to wrist roll. This is appropriate for letter recognition but may need a more elaborate canonicalization for other gestural vocabularies.
4. **Z-coordinate noise.** MediaPipe's `z` is a relative depth estimate, not a true 3D measurement, and is noisier than `x, y`. The classifier still benefits from it but we have not ablated its contribution.

---

## 7. Conclusion

We described a small, deployable ASL alphabet recognizer built on the principle that a strong upstream keypoint detector reduces gesture classification to a low-dimensional MLP problem. The full pipeline — data extraction, training, single-image inference, and a live webcam UI — is implemented and validated end-to-end. The remaining empirical work is to train on a standard ASL corpus and report numbers, which is straightforward given the framework presented here.

---

## References

[1] F. Zhang, V. Bazarevsky, A. Vakunov, A. Tkachenka, G. Sung, C.-L. Chang, and M. Grundmann. *MediaPipe Hands: On-device Real-time Hand Tracking.* arXiv:2006.10214, 2020.

[2] Akash. *ASL Alphabet.* Kaggle dataset. https://www.kaggle.com/datasets/grassknoted/asl-alphabet

[3] Z. Cao, G. Hidalgo, T. Simon, S.-E. Wei, and Y. Sheikh. *OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields.* IEEE TPAMI, 2019.

[4] S. Bantupalli and Y. Xie. *American Sign Language Recognition Using Deep Learning and Computer Vision.* IEEE BigData, 2018.

---

## Appendix A. Reproducibility

```bash
# 1. Install
pip install -r requirements.txt

# 2. Download upstream MediaPipe model
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

# 3. Extract landmarks (data/raw/<CLASS>/*.jpg → CSV)
python -m src.extract_landmarks --raw-dir data/raw --out-csv data/processed/landmarks.csv

# 4. Train
python -m src.train --csv data/processed/landmarks.csv --epochs 80

# 5. Inference (single image)
python -m src.predict path/to/hand.jpg --save-overlay overlay.jpg

# 6. Demo UI
python app.py
```

## Appendix B. Hyperparameters

| Hyperparameter         | Value     |
|------------------------|-----------|
| Hidden layer 1         | 256       |
| Hidden layer 2         | 128       |
| Dropout                | 0.3       |
| Optimizer              | Adam      |
| Learning rate          | 1e-3      |
| Weight decay           | 1e-4      |
| Batch size             | 64        |
| Train/Val/Test split   | 70/15/15  |
| Early-stopping patience| 15 epochs |
| Max epochs             | 80        |
| Random seed            | 42        |
