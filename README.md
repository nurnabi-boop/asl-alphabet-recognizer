# ASL Alphabet Recognizer

Real-time American Sign Language alphabet recognizer.

**Pipeline:** webcam frame → MediaPipe Hands (21 landmarks) → small PyTorch MLP → predicted letter overlaid on the video feed.

The classifier is small and fast because MediaPipe already does the heavy lifting (hand detection + landmark localization). The MLP only has to map a 63-dim landmark vector to one of 26 letters.

## Project layout

```
asl-recognizer/
├── data/
│   ├── raw/                 # input dataset (one folder per class)
│   └── processed/
│       └── landmarks.csv    # produced by extract_landmarks.py
├── models/
│   ├── hand_landmarker.task # MediaPipe model (downloaded once)
│   ├── asl_mlp.pt           # trained classifier checkpoint
│   └── asl_mlp.labels.json  # class names
├── src/
│   ├── extract_landmarks.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
├── app.py                   # Gradio UI
├── requirements.txt
└── README.md
```

## Setup

Python 3.11–3.13 (MediaPipe does not yet ship 3.14 wheels). Tested on Python 3.13.

```bash
pip install -r requirements.txt
```

Download the MediaPipe hand-landmark model:

```bash
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

## Data

Use the [Kaggle ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) or record your own samples. Lay it out as one subfolder per class:

```
data/raw/
├── A/  *.jpg
├── B/  *.jpg
...
└── Z/  *.jpg
```

## Run the pipeline

```bash
# 1. Extract landmarks (once per dataset)
python -m src.extract_landmarks --raw-dir data/raw --out-csv data/processed/landmarks.csv

# 2. Train the MLP
python -m src.train --csv data/processed/landmarks.csv --epochs 80

# 3. Single-image inference
python -m src.predict path/to/hand.jpg --save-overlay overlay.jpg

# 4. Launch the Gradio UI
python app.py
```

`extract_landmarks.py` also has a smoke-test mode:

```bash
python -m src.extract_landmarks --smoke-test tests/smoke
```

## Design notes

- **Landmark normalization** (`normalize_landmarks` in [extract_landmarks.py](src/extract_landmarks.py)): every landmark is shifted relative to the wrist (landmark 0) and scaled by the max distance from the wrist. This makes the feature vector invariant to where the hand is in frame and to its absolute size — only the relative finger geometry survives, which is what carries the letter identity.
- **Model** ([model.py](src/model.py)): two hidden layers (256 → 128) with ReLU + dropout. ~50K parameters. Inference is microseconds on CPU.
- **MediaPipe Tasks API**: this project uses `mp.tasks.vision.HandLandmarker` (the new API). The legacy `mp.solutions.hands` module is no longer shipped on Python 3.13. The IMAGE-mode landmarker is used for static images; the VIDEO-mode landmarker is used in the live webcam tab so MediaPipe can use temporal smoothing.
- **Single-hand assumption**: `num_hands=1`. If you sign with two hands, only the first detected hand is classified.
