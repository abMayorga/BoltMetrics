# BoltMetrics
https://boltmetrics.onrender.com/

Identify bolts from a photo w/ a US quarter for scale. BoltMetrics detects both objects, measures the bolt, and matches it to a standard size in metric or imperial

Built with YOLO11 + OpenCV. Trained on a custom dataset via [Roboflow](https://universe.roboflow.com/boltmetrics/detect-and-classify-object-detection-qxref).


## SetupSteps

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

Create a `.env` file in the project root for production. For local dev, this is optional — defaults work fine.

signs browser session cookies so they can't be tampered with. Generate one

```bash
python -c "import os; print(os.urandom(32).hex())"
```

- `FLASK_DEBUG` — set to `1` during development for auto-reload when you edit code. **Never set to `1` in production**

## How It Works

1. **Detection** — YOLO11 locates the bolt and quarter in the image
2. **Scale** — The quarter's known diameter (24.26 mm) sets a pixel-to-mm ratio
3. **Measurement** — Multiple segmentation measure the bolt's width and length
4. **Classification** — Width is matched to the nearest standard hex head size (metric or imperial)

## Project Layout

| File | Corresponding Purpose 
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
