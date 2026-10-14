<p align="center">
  <h1 align="center">SignLingo</h1>
  <p align="center">
    <strong>Multi-Language Sign Language Recognition &amp; Sentence Formation</strong>
  </p>
  <p align="center">
    <a href="#"><img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white" alt="Python"></a>
    <a href="#"><img src="https://img.shields.io/badge/TensorFlow-2.12+-orange?logo=tensorflow&logoColor=white" alt="TensorFlow"></a>
    <a href="#"><img src="https://img.shields.io/badge/MediaPipe-0.10+-green?logo=google&logoColor=white" alt="MediaPipe"></a>
    <a href="#"><img src="https://img.shields.io/badge/License-MIT-purple" alt="License"></a>
    <a href="#"><img src="https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey" alt="Platform"></a>
  </p>
</p>

---

A deep learning system for recognizing sign language gestures from video and forming grammatically coherent English sentences. Supports **American Sign Language (ASL)** and **Indian Sign Language (ISL)**, with an extensible architecture for any sign language.

## ✨ Key Features

- **Multi-Model Architecture** — Implements and benchmarks 5 deep learning architectures: Vision Transformer (ViT), BiLSTM+Attention, ResNet-50, VGG-16, and VGG-19
- **88.2% Top-1 Accuracy** — ViT achieves best performance across all models on 50-class WLASL dataset
- **Real-Time Recognition** — Live webcam inference with MediaPipe hand landmark extraction at 30+ FPS
- **Sentence Formation** — Rule-based NLP module converts recognized sign glosses into grammatically coherent English sentences
- **Multi-Language Support** — Configurable architecture supporting ASL and ISL datasets via a single config switch
- **Data Augmentation** — 9 augmentation strategies (noise, scaling, temporal warp, dropout, mirror, shift, 2D rotation, speed variation, per-hand scale) for improved generalization

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     SignLingo Pipeline                        │
├──────────────┬───────────────┬───────────────┬───────────────┤
│  Video Input │  MediaPipe    │  Deep Learning│  Sentence     │
│              │  Landmark     │  Model        │  Formation    │
│  📹 Webcam   │  Extraction   │               │               │
│  📁 Video    │               │  ViT ★        │  Gloss→POS    │
│  files       │  21 keypoints │  BiLSTM+Attn  │  SVO Reorder  │
│              │  × 3 coords   │  ResNet-50    │  Inflection   │
│              │  × 2 hands +  │  VGG-16/19    │  Punctuation  │
│              │  inter-hand   │               │               │
│              │  = 132 feat.  │               │               │
├──────────────┴───────────────┴───────────────┴───────────────┤
│ Input: Video/Webcam → Landmarks (T×132) → Prediction → Text │
└──────────────────────────────────────────────────────────────┘
```

### Vision Transformer (ViT) — Primary Model

```
Input (30, 132) → Linear Projection → + Positional Embedding → + [CLS] Token
  → 4 × TransformerBlock(MultiHeadAttention + FFN)
  → [CLS] output → LayerNorm → MLP Head → Softmax(50)
```

The ViT treats each frame's landmark vector as a "token" (analogous to image patches in standard ViT), enabling the model to learn temporal relationships between frames through self-attention.

## 📊 Benchmark Results

All models trained on the same 80/20 split of the WLASL dataset (50 sign classes).

| Model | Top-1 Accuracy | Top-3 Accuracy | Parameters | Epochs |
|-------|:--------------:|:--------------:|:----------:|:------:|
| **ViT** ★ | **88.24%** | **96.08%** | 548K | 67 |
| BiLSTM+Attention | 84.31% | 94.12% | 1.2M | 72 |
| ResNet-50 | 82.35% | 92.16% | 3.5M | 58 |
| VGG-16 | 78.43% | 90.20% | 2.9M | 61 |
| VGG-19 | 76.47% | 88.24% | 3.1M | 55 |

<p align="center">
  <img src="results/model_comparison.png" alt="Model Comparison" width="85%">
</p>

### Training Curves (ViT)

<p align="center">
  <img src="results/vit_training_history.png" alt="Training History" width="80%">
</p>

### Confusion Matrix (ViT — 10 classes shown)

<p align="center">
  <img src="results/vit_confusion_matrix.png" alt="Confusion Matrix" width="55%">
</p>

## 📁 Project Structure

```
signlingo/
├── signlingo/                  # Core Python package
│   ├── config.py               # Centralized configuration (ASL/ISL presets)
│   ├── data/
│   │   ├── landmark_extractor.py   # MediaPipe hand landmark extraction
│   │   ├── dataset.py              # Data loading, splitting, normalization
│   │   └── preprocessing.py        # 9 augmentation strategies
│   ├── models/
│   │   ├── vit.py                  # Vision Transformer (primary)
│   │   ├── lstm_attention.py       # Bidirectional LSTM + Attention
│   │   ├── resnet.py               # ResNet-50 (1D temporal)
│   │   ├── vgg.py                  # VGG-16 / VGG-19 (1D temporal)
│   │   └── factory.py              # Model registry & builder
│   ├── sentence/
│   │   └── formation.py            # Gloss → English sentence pipeline
│   └── utils/
│       ├── visualization.py        # Training plots, confusion matrices
│       └── metrics.py              # Evaluation metrics & reporting
├── scripts/
│   ├── train.py                # Unified training (any arch, any language)
│   ├── evaluate.py             # Model evaluation
│   ├── benchmark.py            # Multi-model comparison benchmark
│   └── demo_webcam.py          # Real-time webcam demo with sentences
├── results/                    # Generated plots and benchmark data
├── models/                     # Trained model weights (git-ignored)
├── data/                       # Dataset directories (git-ignored)
├── requirements.txt
├── setup.py
└── README.md
```

## Running this copy on Windows

The app reads a still photo from the camera or an uploaded image. It uses the 21 hand landmarks, not the Sign Language MNIST CNN, because a normal photo does not look like those 28×28 images.

```powershell
.\run_app.bat
```

The browser app uses port **8510**. If that port is already taken, close the previous SignLingo window and run `run_app.bat` again.

The 88% table below is from the original word-level writeup. This folder does not include the WLASL videos or those weight files, so that number was not measured here.

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- macOS (Apple Silicon optimized) / Linux / Windows
- Webcam (for live demo)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/signlingo.git
cd signlingo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Install the package (editable mode)
pip install -e .
```

### Dataset Setup

**ASL (WLASL Dataset)**

1. Download the WLASL dataset from [WLASL Project Page](https://dxli94.github.io/WLASL/)
2. Extract to `wlasl-processed/`
3. Organize into class folders:
```bash
python organize_dataset.py
```

**ISL (Indian Sign Language)**

1. Download an ISL dataset (e.g., [INCLUDE](https://zenodo.org/record/4010759) or [ISL-CSLTR](https://zenodo.org/record/6aborton))
2. Organize videos into `data/isl_organized/<class_name>/*.mp4`
3. Train with `--language ISL` flag

## 💻 Usage

### Training

```bash
# Train ViT (default, recommended)
python scripts/train.py --arch vit

# Train other architectures
python scripts/train.py --arch lstm_attention
python scripts/train.py --arch resnet50
python scripts/train.py --arch vgg16

# Train on ISL dataset
python scripts/train.py --arch vit --language ISL --data-dir data/isl_organized

# Custom hyperparameters
python scripts/train.py --arch vit --epochs 100 --lr 0.001 --batch-size 32
```

### Multi-Model Benchmark

```bash
# Run full benchmark (all 5 architectures)
python scripts/benchmark.py

# Quick comparison (subset)
python scripts/benchmark.py --archs vit lstm_attention resnet50
```

### Evaluation

```bash
python scripts/evaluate.py --model models/best_vit.keras
```

### Live Webcam Demo

```bash
python scripts/demo_webcam.py --model models/best_vit.keras
```

**Controls:**
| Key | Action |
|-----|--------|
| `S` | Save screenshot |
| `F` | Form sentence from accumulated signs |
| `R` | Reset buffer |
| `Q` | Quit |

### Sentence Formation (Python API)

```python
from signlingo.sentence.formation import SentenceFormer

former = SentenceFormer()

# ASL gloss order → English sentence
former.form_sentence(["I", "GO", "SCHOOL"])
# → "I am going to the school."

former.form_sentence(["BOOK", "I", "LIKE"])
# → "I am liking the book."

former.form_sentence(["WHAT", "YOU", "DRINK"])
# → "What are drinking you?"
```

## 🧠 Technical Details

### MediaPipe Landmark Extraction

Each video frame is processed by MediaPipe Hands to extract landmarks for both hands separately, assigned to left/right slots via handedness labels. Each hand's 21 keypoints are normalized relative to the wrist and scaled by max inter-landmark distance (translation and scale invariant). A 6-dimensional inter-hand feature vector captures the relative position between the two hands, giving **132 features per frame** total. Frames are uniformly sampled and padded/truncated to a fixed sequence length of 30.

### Data Augmentation Pipeline

| Strategy | Description |
|----------|-------------|
| Gaussian Noise | N(0, 0.01) noise added to coordinates |
| Random Scaling | Uniform scale factor [0.95, 1.05] |
| Temporal Shift | Circular shift ±2 frames |
| Temporal Warp | Smooth random stretch/compress of time axis |
| Frame Dropout | Random zeroing of 5% of frames |
| X-Mirror | Flip x-coordinates and swap left↔right hand slots |
| 2D Rotation | Rotate (x, y) plane ±15° to simulate wrist roll / camera tilt |
| Speed Variation | Resample sequence at 0.8×–1.2× speed to simulate signer pace |
| Per-Hand Scale | Independently scale each hand to simulate perspective depth |

### Sentence Formation Module

The sentence formation pipeline handles the grammar translation from sign language (which uses Topic-Comment or SOV order) to English SVO:

1. **De-duplication** — Remove consecutive repeated glosses
2. **POS Tagging** — Classify glosses (pronoun, verb, noun, adjective, etc.)
3. **SVO Reordering** — Rearrange from sign language grammar to English word order
4. **Inflection** — Add auxiliaries, articles, verb conjugation
5. **Polish** — Capitalize, punctuate, fix spacing

## 🔧 Configuration

All settings are centralized in `signlingo/config.py`:

```python
from signlingo.config import get_config

# ASL with ViT
cfg = get_config(language="ASL", model_arch="vit", epochs=100)

# ISL with BiLSTM
cfg = get_config(language="ISL", model_arch="lstm_attention", data_dir="data/isl_organized")

# Available architectures
# "vit", "lstm_attention", "resnet50", "vgg16", "vgg19"
```

## 🛠️ Tech Stack

- **Deep Learning**: TensorFlow / Keras, TensorFlow Metal (Apple Silicon GPU)
- **Computer Vision**: OpenCV, MediaPipe Hands
- **Data Science**: NumPy, scikit-learn, pandas
- **Visualization**: Matplotlib, Seaborn
- **Architecture**: Vision Transformer, BiLSTM, ResNet-50, VGG-16/19

## 🙏 Acknowledgements

- [WLASL Dataset](https://dxli94.github.io/WLASL/) — Word-Level American Sign Language Video Dataset
- [MediaPipe](https://mediapipe.dev/) — Google's framework for building multimodal ML pipelines
- [Dosovitskiy et al.](https://arxiv.org/abs/2010.11929) — "An Image is Worth 16×16 Words" (ViT paper)
