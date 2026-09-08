# BoltMetrics
https://boltmetrics.onrender.com/

Identify bolts from a photo w/ a US quarter for scale. BoltMetrics detects both objects, measures the bolt, and matches it to a standard size in metric or imperial

Built with YOLO11 + OpenCV. Trained on a custom dataset via [Roboflow](https://universe.roboflow.com/boltmetrics/detect-and-classify-object-detection-qxref).

> BoltMetrics estimates size from a photograph; it is not a calibrated inspection instrument. For best results, place one bolt and one quarter flat on the same surface and photograph them directly overhead.


## Setup

A trained model (`model/best.pt`) is included in the repo, no training needed

## Run

```bash
python app.py
```

Open **http://127.0.0.1:5000**, upload a photo (or use the camera on mobile), and Analyze

You can also run detection from the terminal via ->

```bash
python bolt_analyzer.py --image test_images/bqm10.jpg
```

## Environment Variables

Create a `.env` file in the project root for production. For local development this is optional.

`FLASK_SECRET_KEY` signs browser session cookies so they cannot be tampered with. Generate one:

```bash
python -c "import os; print(os.urandom(32).hex())"
```

- `FLASK_SECRET_KEY` — stable random secret used to sign Flask session cookies
- `FLASK_DEBUG` — set to `1` during development for auto-reload when you edit code. **Never set to `1` in production**
- `ANALYZER_BACKEND` — `local`, `roboflow`, or `auto` (default). Auto uses Roboflow when all three Roboflow settings exist, otherwise local weights
- `ROBOFLOW_API_KEY` — required for the Roboflow backend
- `ROBOFLOW_WORKSPACE` — Roboflow workspace slug
- `ROBOFLOW_WORKFLOW_ID` — deployed Workflow ID
- `MODEL_PATH` — optional local `.pt` path override

## How It Works

1. **Detection** — YOLO11 locates the bolt and quarter in the image
2. **Scale** — The quarter's known diameter (24.26 mm) sets a direction-aware pixel-to-mm calibration; its ellipse also estimates modest camera tilt
3. **Measurement** — OpenCV builds a silhouette-width profile, measuring shank diameter and length from under the head to the tip
4. **Classification** — Shank diameter is matched to metric or imperial reference sizes. Wrench size and common coarse pitch/TPI are table lookups, not directly measured

## Project Layout

| File | Corresponding Purpose |
|---|---|
| `app.py` | Web server — handles uploads, calls the analyzer, renders results |
| `bolt_analyzer.py` | Core logic — detection, measurement, classification |
| `scripts/train.py` | Retrains the model on updated datasets |
| `model/best.pt` | Trained YOLO11 weights (included, ready to use) |
| `dataset/` | Training data from Roboflow (YOLO11 format) |
| `test_images/` | Sample photos for testing |

## Retraining

If you add more images to the dataset and want to retrain:

```bash
python scripts/train.py
```

This takes ~3 hours on CPU. After training, copy the new weights into the model folder:

```bash
cp runs/detect/train*/weights/best.pt model/best.pt
```

The app always loads `model/best.pt` first, falling back to the latest `runs/` weights if no model folder exists.

## Dataset
Annotated via [Roboflow](https://universe.roboflow.com/boltmetrics/detect-and-classify-object-detection-qxref) with a 70/20/10 train/valid/test split.

- **Classes:** Bolt, Quarter
- **License:** CC BY 4.0
