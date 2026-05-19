# ASL Glossing — Local Inference

Real-time ASL recognition from webcam using a trained CNN + BiLSTM model.

---

## Folder Structure

```
asl_glossing/
│
├── mediapipe_models/              ← MediaPipe model files (already have these)
│   ├── hand_landmarker.task
│   ├── pose_landmarker_full.task
│   └── face_landmarker.task
│
├── weights/                       ← Download these from Kaggle
│   ├── asl_model_weights_v2.pt    ← trained model  (~14 MB)
│   ├── vocabulary.json            ← fingerspelling chars  (~2 KB)
│   └── sign_to_idx.json          ← 250 ASL sign labels  (~3 KB)
│
├── extract_landmarks.py           ← your existing webcam extractor
├── model.py                       ← model architecture
├── inference.py                   ← inference engine
├── run.py                         ← main webcam app
├── test_inference.py              ← quick test without webcam
└── requirements.txt
```

---

## Files to Download from Kaggle

Go to your **Phase 2c Fine-Tuning** notebook on Kaggle → **Output tab** and download:

| File | Size | Where to put it |
|---|---|---|
| `asl_model_weights_v2.pt` | ~14 MB | `weights/` |

You already have these locally (downloaded earlier):

| File | Where to put it |
|---|---|
| `vocabulary.json` | `weights/` |
| `sign_to_idx.json` | `weights/` |

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Test without webcam first**
```bash
python test_inference.py
```

Expected output:
```
✅ asl_model_weights_v2.pt     14.2 MB
✅ vocabulary.json              0.0 MB
✅ sign_to_idx.json             0.0 MB

Model loaded ✅  (device: cpu)
...
✅ All tests passed — run python run.py to start webcam
```

**3. Run real-time recognition**
```bash
python run.py
```

---

## Controls

| Key | Action |
|---|---|
| `Q` or `Escape` | Quit |
| `C` | Clear the frame buffer (reset current sign) |

---

## Display

The bottom of the webcam window shows:

- **Fingerspelling** — character sequence decoded from the last N frames
- **Top sign** — most likely ASL sign with confidence %
- **Top 3 bar** — mini bar chart of top 3 sign predictions
- **FPS** — current frames per second

---

## GPU Acceleration (optional)

If your machine has an NVIDIA GPU, edit `run.py`:

```python
# Change this line:
device = 'cpu'
# To:
device = 'cuda'
```

This will reduce inference time from ~30-50ms to ~5-10ms.

---

## Model Performance

| Metric | Value |
|---|---|
| Fingerspelling CER | 0.51 (49% of characters correct) |
| Signs accuracy (250 classes) | 55.6% |
| Inference time (CPU) | ~30-50 ms |
| Random chance (250 classes) | 0.4% |

Check out the [Accuracy report](https://best-techsavvy.github.io/ASLAI/Accuracy/index.html) to see model performance in detail.
