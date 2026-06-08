// Build paper.docx from the markdown paper content.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType, ShadingType,
  PageOrientation, PageBreak,
} = require("docx");

// ---- helpers ----
const FONT = "Calibri";
const MONO = "Consolas";

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120 },
    alignment: opts.alignment ?? AlignmentType.LEFT,
    children: [new TextRun({ text, font: FONT, size: opts.size ?? 22, bold: !!opts.bold, italics: !!opts.italics })],
  });
}

function pRuns(runs, opts = {}) {
  return new Paragraph({
    spacing: { after: 120 },
    alignment: opts.alignment ?? AlignmentType.LEFT,
    children: runs,
  });
}

function r(text, opts = {}) {
  return new TextRun({
    text,
    font: opts.mono ? MONO : FONT,
    size: opts.size ?? 22,
    bold: !!opts.bold,
    italics: !!opts.italics,
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 280, after: 140 },
    children: [new TextRun({ text, font: FONT, size: 28, bold: true })],
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 220, after: 120 },
    children: [new TextRun({ text, font: FONT, size: 24, bold: true })],
  });
}

function title(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 240 },
    children: [new TextRun({ text, font: FONT, size: 36, bold: true })],
  });
}

function subtitleCenter(text, italics=false) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 120 },
    children: [new TextRun({ text, font: FONT, size: 24, italics })],
  });
}

function hr() {
  return new Paragraph({
    spacing: { after: 200, before: 200 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "808080", space: 1 } },
    children: [new TextRun({ text: "", font: FONT, size: 2 })],
  });
}

function code(lines) {
  // Each line becomes a paragraph with monospace font and a light-gray box.
  return lines.map((line, idx) => new Paragraph({
    spacing: { after: 0 },
    shading: { fill: "F2F2F2", type: ShadingType.CLEAR },
    children: [new TextRun({ text: line.length === 0 ? " " : line, font: MONO, size: 18 })],
  }));
}

function bullet(text) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 80 },
    children: [new TextRun({ text, font: FONT, size: 22 })],
  });
}

function ordered(text) {
  return new Paragraph({
    numbering: { reference: "numbers", level: 0 },
    spacing: { after: 80 },
    children: [new TextRun({ text, font: FONT, size: 22 })],
  });
}

// ---- table helpers ----
const BORDER = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const BORDERS = { top: BORDER, bottom: BORDER, left: BORDER, right: BORDER };
const CELL_MARGINS = { top: 80, bottom: 80, left: 120, right: 120 };
const CONTENT_WIDTH = 9360; // US Letter, 1" margins

function tableCellText(text, { bold = false, header = false, align = AlignmentType.LEFT, width } = {}) {
  return new TableCell({
    borders: BORDERS,
    width: { size: width, type: WidthType.DXA },
    margins: CELL_MARGINS,
    shading: header ? { fill: "D9E2F3", type: ShadingType.CLEAR } : undefined,
    children: [new Paragraph({
      alignment: align,
      children: [new TextRun({ text, font: FONT, size: 20, bold: bold || header })],
    })],
  });
}

function makeTable(columnWidths, rowsData, headerCount = 1) {
  const totalWidth = columnWidths.reduce((a, b) => a + b, 0);
  const rows = rowsData.map((row, rowIdx) => new TableRow({
    children: row.map((cellText, colIdx) => tableCellText(String(cellText), {
      width: columnWidths[colIdx],
      header: rowIdx < headerCount,
      align: (rowIdx >= headerCount && colIdx > 0) ? AlignmentType.RIGHT : AlignmentType.LEFT,
    })),
  }));
  return new Table({
    width: { size: totalWidth, type: WidthType.DXA },
    columnWidths,
    rows,
  });
}

// ---- document content ----
const children = [];

// Title block
children.push(title("Real-Time ASL Alphabet Recognition via MediaPipe Hand Landmarks and a Compact Multilayer Perceptron"));
children.push(subtitleCenter("Anonymous Author"));
children.push(subtitleCenter("Working draft — system description and methodology paper", true));
children.push(hr());

// Abstract
children.push(h1("Abstract"));
children.push(p(
  "We present a lightweight, real-time American Sign Language (ASL) alphabet recognition system that decouples hand localization from gesture classification. Rather than learning visual features end-to-end from raw pixels, we delegate hand detection and keypoint regression to MediaPipe Hands and train only a small multilayer perceptron (MLP) on the resulting 63-dimensional landmark vectors. A simple wrist-relative, scale-normalized representation makes the classifier invariant to hand position and apparent size in the frame. The system streams predictions from a webcam at interactive rates on CPU-only hardware, exposes a single-image inference CLI, and ships a Gradio demo UI with live webcam and image-upload tabs. We describe the data pipeline, normalization scheme, training procedure, and an end-to-end smoke test on a four-class synthetic dataset that confirms correctness of the pipeline. Reproducible code is provided."
));
children.push(pRuns([
  r("Keywords: ", { bold: true }),
  r("sign language recognition, hand pose estimation, MediaPipe, MLP, real-time inference."),
]));
children.push(hr());

// 1 Introduction
children.push(h1("1. Introduction"));
children.push(p(
  "Automatic recognition of fingerspelled sign-language letters from a webcam is a well-studied problem with a useful property: most of the visual ambiguity has already been solved by general-purpose hand pose estimators. Models such as MediaPipe Hands [1] regress 21 anatomical keypoints per hand from a single RGB image with low latency on commodity hardware. Given those keypoints, the remaining task — mapping a hand pose to one of 26 letters — is a low-dimensional classification problem that does not require a deep convolutional network."
));
children.push(p("This paper describes a system built on that observation. The contributions are:"));
children.push(ordered("A practical pipeline that uses the modern MediaPipe Tasks API (mp.tasks.vision.HandLandmarker) for both batch landmark extraction and live video inference."));
children.push(ordered("A wrist-relative, scale-normalized 63-dimensional feature that is robust to in-frame translation and apparent hand size."));
children.push(ordered("A two-hidden-layer MLP (~50K parameters) that runs in microseconds on CPU and is trivially deployable."));
children.push(ordered("A reference implementation with a CLI, a Gradio UI, and an end-to-end smoke test on a synthetic four-class dataset that validates the pipeline."));

// 2 Related Work
children.push(h1("2. Related Work"));
children.push(p(
  "End-to-end CNN approaches to ASL alphabet recognition are common in the Kaggle ecosystem [2], typically training MobileNet- or EfficientNet-class models directly on the labeled ASL Alphabet dataset. These models conflate two problems: localizing the hand in the frame, and identifying the gesture once localized."
));
children.push(p(
  "The alternative — explicit two-stage pipelines that separate detection/keypoint regression from classification — has a long history, from data-glove-based approaches through the modern landmark-based work enabled by MediaPipe [1] and OpenPose [3]. Landmark-based classifiers have been shown to generalize better across backgrounds and lighting than pixel-input CNNs of comparable size, because the upstream detector absorbs that variability."
));
children.push(p("Our work is a small, faithful instance of this pattern, focused on real-time deployment."));

// 3 Methodology
children.push(h1("3. Methodology"));

children.push(h2("3.1 System Overview"));
children.push(...code([
  "camera frame  ->  MediaPipe HandLandmarker  ->  21 (x,y,z) keypoints",
  "                                                       |",
  "                                                       v",
  "                                                normalize_landmarks()",
  "                                                       |",
  "                                                       v",
  "                                                63-dim feature vector",
  "                                                       |",
  "                                                       v",
  "                                                   MLP (256->128->K)",
  "                                                       |",
  "                                                       v",
  "                                                 letter + softmax",
]));
children.push(p("The full pipeline is implemented in five Python modules: extract_landmarks.py, dataset.py, model.py, train.py, predict.py, plus an app.py for the Gradio UI."));

children.push(h2("3.2 Landmark Extraction"));
children.push(p(
  "For each input image we run the MediaPipe HandLandmarker (float16 model variant) in single-hand mode (num_hands=1) with min_hand_detection_confidence=0.5. The output is a list of 21 normalized landmarks, each carrying (x, y, z) where x, y are in [0, 1] image-relative coordinates and z is depth in the same scale as x."
));
children.push(p(
  "The IMAGE running mode is used for offline dataset construction and for the upload-image UI tab; the VIDEO running mode (which enables MediaPipe's internal temporal smoothing) is used for the live webcam tab."
));

children.push(h2("3.3 Feature Normalization"));
children.push(p(
  "Let L be the 21x3 raw landmark matrix. We translate by the wrist landmark L_0 and scale by the maximum Euclidean distance from the wrist:"
));
children.push(...code([
  "L'_i = (L_i - L_0) / max_j || L_j - L_0 ||_2",
]));
children.push(p("The flattened 63-dimensional vector vec(L') is the classifier input. Two desirable invariances follow directly:"));
children.push(bullet("Translation invariance. Any whole-hand shift in the frame is absorbed by the wrist subtraction."));
children.push(bullet("Scale invariance. Any apparent-size change (distance from camera, image resolution) is absorbed by the per-sample max-distance normalization."));
children.push(p("Rotation invariance is NOT enforced; ASL handshapes do encode orientation (e.g., 'P' vs 'K'), so we want the classifier to see it."));

children.push(h2("3.4 Classifier"));
children.push(p("The classifier is a two-hidden-layer MLP:"));
children.push(makeTable(
  [2200, 2200, 2480, 2480],
  [
    ["Layer", "Out features", "Activation", "Dropout"],
    ["Linear", "256", "ReLU", "0.3"],
    ["Linear", "128", "ReLU", "0.3"],
    ["Linear", "K", "—", "—"],
  ],
));
children.push(p(
  "where K is the number of classes (26 for the standard ASL alphabet, or 29 if the Kaggle dataset's space, del, and nothing classes are kept). Parameter count is approximately 51K when K=26."
));

children.push(h2("3.5 Training Procedure"));
children.push(p(
  "We use cross-entropy loss and the Adam optimizer (lr = 1e-3, weight decay = 1e-4, batch size = 64). Data is split into train/val/test (70/15/15) with stratification by label. The validation set is used for early stopping (patience = 15) and best-checkpoint selection. The full training script with all flags is in src/train.py."
));

children.push(h2("3.6 Real-Time Inference"));
children.push(p(
  "The Gradio app holds two persistent ASLPredictor instances, one per MediaPipe RunningMode (IMAGE for uploads, VIDEO for webcam). For the webcam tab, frames arrive at 5-10 Hz via Gradio's streaming=True; for each frame we (a) compute landmarks, (b) normalize, (c) forward through the MLP, (d) overlay the predicted letter, the 21 keypoint dots, and a top-3 bar chart on the output image. End-to-end per-frame latency is dominated by the MediaPipe inference call."
));

// 4 Experiments
children.push(h1("4. Experiments"));

children.push(h2("4.1 Datasets"));
children.push(p(
  "The intended training corpus is the Kaggle ASL Alphabet dataset [2], which contains approximately 87,000 256x256 RGB images across 29 classes (A-Z plus space, del, nothing). The data pipeline in src/extract_landmarks.py consumes any directory with one subfolder per class."
));
children.push(p(
  "For pipeline validation we also constructed a synthetic four-class dataset by extracting landmarks from four MediaPipe-provided reference hand images (pointing_up, thumbs_down, victory, woman_hands), labeling them A/B/C/D, and generating 50 noisy copies of each (Gaussian jitter sigma = 0.02 in the normalized landmark space). This yields 200 rows, intentionally easy to separate, used solely to verify correctness of the dataset/model/train/predict modules end-to-end."
));

children.push(h2("4.2 Implementation"));
children.push(p(
  "All experiments run on Python 3.13 with MediaPipe 0.10.35, PyTorch 2.11, OpenCV 4.13, and Gradio 6.14, on CPU. The hand_landmarker.task float16 model (~7.8 MB) is downloaded once from the official MediaPipe model bucket."
));

children.push(h2("4.3 Pipeline Validation (Synthetic Data)"));
children.push(p(
  "On the four-class synthetic dataset, the model converges to 100% test accuracy in fewer than 10 epochs (Table 1). This is the expected result — the synthetic classes are trivially separable — and its purpose is to confirm that data loading, label encoding, the training loop, and checkpoint serialization all behave correctly."
));
children.push(pRuns([
  r("Table 1. ", { bold: true }),
  r("Training on the four-class synthetic dataset (200 rows, 70/15/15 split)."),
]));
children.push(makeTable(
  [1500, 1900, 1900, 2000, 2060],
  [
    ["Epoch", "Train loss", "Train acc", "Val loss", "Val acc"],
    ["1", "1.253", "0.734", "1.029", "1.000"],
    ["5", "0.065", "1.000", "0.023", "1.000"],
    ["10", "0.003", "1.000", "0.001", "1.000"],
    ["15", "0.001", "1.000", "0.000", "1.000"],
  ],
));
children.push(p("Final test accuracy: 1.000. Best val accuracy: 1.000."));
children.push(p(
  "We then ran single-image inference (src.predict) on each of the four reference images plus a fifth distractor image with no detectable hand. The four hand images each recovered their assigned label; the no-hand image produced the expected \"no hand detected\" output. An example overlay is rendered correctly with 21 keypoint dots, a label/confidence badge, and a top-3 readout."
));

children.push(h2("4.4 Results on Real ASL Data"));
children.push(pRuns([
  r("(Pending.) ", { italics: true }),
  r("Training on the full Kaggle ASL Alphabet dataset is left as the next step; the data pipeline is already in place. We expect landmark-based MLP results in the high 90s on the standard 70/15/15 split based on comparable published baselines [4], with the caveats discussed below."),
]));

// 5 Discussion
children.push(h1("5. Discussion"));
children.push(pRuns([
  r("Why this works at all. ", { bold: true }),
  r("MediaPipe Hands is trained on a large and diverse human-hand corpus, so by the time the MLP sees an input the hand has already been localized and reduced to a canonical 21-point skeleton. The classifier therefore does not need to learn appearance robustness — only geometric discrimination between 26 hand shapes."),
]));
children.push(pRuns([
  r("Why MLP, not CNN. ", { bold: true }),
  r("With landmarks as input, spatial convolution buys nothing: the input is already a sparse, ordered 63-dim vector. A small MLP is the right inductive bias."),
]));
children.push(pRuns([
  r("Why wrist-relative scaling. ", { bold: true }),
  r("Absolute MediaPipe coordinates depend on image aspect ratio and on how large the hand appears in the frame. Without normalization, the classifier would have to learn that \"the same hand shape closer to the camera\" is the same class, wasting capacity on irrelevant variation."),
]));

// 6 Limitations
children.push(h1("6. Limitations"));
children.push(ordered("Single-hand only. Letters like 'J' and 'Z' involve motion, and ASL more broadly involves two hands. This system handles only the static, single-hand subset of the alphabet."));
children.push(ordered("Upstream failure modes. If MediaPipe fails to detect a hand (poor lighting, extreme occlusion, motion blur), the classifier never runs. Our pipeline returns a None prediction in that case, which the UI renders as \"no hand detected.\""));
children.push(ordered("Orientation sensitivity. Because we deliberately do NOT rotation-normalize, the classifier may be sensitive to wrist roll. This is appropriate for letter recognition but may need a more elaborate canonicalization for other gestural vocabularies."));
children.push(ordered("Z-coordinate noise. MediaPipe's z is a relative depth estimate, not a true 3D measurement, and is noisier than x, y. The classifier still benefits from it but we have not ablated its contribution."));

// 7 Conclusion
children.push(h1("7. Conclusion"));
children.push(p(
  "We described a small, deployable ASL alphabet recognizer built on the principle that a strong upstream keypoint detector reduces gesture classification to a low-dimensional MLP problem. The full pipeline — data extraction, training, single-image inference, and a live webcam UI — is implemented and validated end-to-end. The remaining empirical work is to train on a standard ASL corpus and report numbers, which is straightforward given the framework presented here."
));

// References
children.push(h1("References"));
children.push(p("[1] F. Zhang, V. Bazarevsky, A. Vakunov, A. Tkachenka, G. Sung, C.-L. Chang, and M. Grundmann. MediaPipe Hands: On-device Real-time Hand Tracking. arXiv:2006.10214, 2020."));
children.push(p("[2] Akash. ASL Alphabet. Kaggle dataset. https://www.kaggle.com/datasets/grassknoted/asl-alphabet"));
children.push(p("[3] Z. Cao, G. Hidalgo, T. Simon, S.-E. Wei, and Y. Sheikh. OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields. IEEE TPAMI, 2019."));
children.push(p("[4] S. Bantupalli and Y. Xie. American Sign Language Recognition Using Deep Learning and Computer Vision. IEEE BigData, 2018."));

// Page break before appendices
children.push(new Paragraph({ children: [new PageBreak()] }));

// Appendix A
children.push(h1("Appendix A. Reproducibility"));
children.push(...code([
  "# 1. Install",
  "pip install -r requirements.txt",
  "",
  "# 2. Download upstream MediaPipe model",
  "curl -L -o models/hand_landmarker.task \\",
  "  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
  "",
  "# 3. Extract landmarks (data/raw/<CLASS>/*.jpg -> CSV)",
  "python -m src.extract_landmarks --raw-dir data/raw --out-csv data/processed/landmarks.csv",
  "",
  "# 4. Train",
  "python -m src.train --csv data/processed/landmarks.csv --epochs 80",
  "",
  "# 5. Inference (single image)",
  "python -m src.predict path/to/hand.jpg --save-overlay overlay.jpg",
  "",
  "# 6. Demo UI",
  "python app.py",
]));

// Appendix B
children.push(h1("Appendix B. Hyperparameters"));
children.push(makeTable(
  [5360, 4000],
  [
    ["Hyperparameter", "Value"],
    ["Hidden layer 1", "256"],
    ["Hidden layer 2", "128"],
    ["Dropout", "0.3"],
    ["Optimizer", "Adam"],
    ["Learning rate", "1e-3"],
    ["Weight decay", "1e-4"],
    ["Batch size", "64"],
    ["Train/Val/Test split", "70/15/15"],
    ["Early-stopping patience", "15 epochs"],
    ["Max epochs", "80"],
    ["Random seed", "42"],
  ],
));

// ---- assemble document ----
const doc = new Document({
  creator: "Claude",
  title: "Real-Time ASL Alphabet Recognition via MediaPipe Hand Landmarks and a Compact Multilayer Perceptron",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: FONT, color: "1F3864" },
        paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: FONT, color: "2E5395" },
        paragraph: { spacing: { before: 220, after: 120 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [
        { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
      ]},
      { reference: "numbers", levels: [
        { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
      ]},
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840, orientation: PageOrientation.PORTRAIT },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    children,
  }],
});

const outPath = path.join(__dirname, "paper.docx");
Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(outPath, buffer);
  console.log("wrote " + outPath + "  " + buffer.length + " bytes");
});
